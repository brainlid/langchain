defmodule LangChain.ScriptedResponsesAdapter do
  @moduledoc """
  A `Req` adapter that answers each request with the next of a scripted list of
  OpenAI Responses API bodies, so a chain can run a whole multi-call loop
  offline against the real HTTP decode path.

  A request carrying an `:into` collector gets the body as the SSE event stream
  the API sends for that response, split into small chunks so events straddle
  chunk boundaries the way they do on the wire. Any other request gets the body
  as decoded JSON.

  A chat model reaches this through its `:req_config`:

      ChatOpenAIResponses.new!(%{
        model: "gpt-5.4",
        api_key: "test",
        req_config: %{adapter: LangChain.ScriptedResponsesAdapter, retry: false}
      })

  The script and the request bodies sent live in the process dictionary of the
  process that makes the requests, which is the test process for a chain run
  synchronously. `next_body/1` is public so a stub standing in for another
  client library can draw from the same script.
  """

  @chunk_size 97

  @doc """
  Replace the script with `bodies`, served in order.
  """
  def script(bodies), do: Process.put(__MODULE__, %{bodies: bodies, sent: []})

  @doc """
  Record `sent` as the next request and return the next scripted body.
  """
  def next_body(sent) do
    case Process.get(__MODULE__) do
      %{bodies: [body | rest], sent: previous} ->
        Process.put(__MODULE__, %{bodies: rest, sent: [sent | previous]})
        body

      _exhausted ->
        raise "ScriptedResponsesAdapter: request made after the script ran out"
    end
  end

  @doc """
  What each request sent, oldest first.
  """
  def sent, do: Enum.reverse(Process.get(__MODULE__).sent)

  @doc """
  The scripted bodies no request has consumed.
  """
  def remaining, do: Process.get(__MODULE__).bodies

  @doc false
  def run(request) do
    body = request.body |> IO.iodata_to_binary() |> Jason.decode!() |> next_body()

    case request.into do
      into when is_function(into, 2) ->
        body
        |> sse()
        |> chunks()
        |> Enum.reduce_while({request, Req.Response.new(status: 200)}, fn chunk, acc ->
          into.({:data, chunk}, acc)
        end)

      _no_collector ->
        {request, Req.Response.new(status: 200, body: body)}
    end
  end

  @doc """
  The SSE body the API streams for `response`.
  """
  def sse(response) do
    Enum.map_join(sse_events(response), fn event ->
      "event: #{event["type"]}\ndata: #{Jason.encode!(event)}\n\n"
    end)
  end

  @doc """
  The decoded events the API streams for `response`, in order.

  Each output item opens with `response.output_item.added` and closes with
  `response.output_item.done`. A message item's `added` event already carries
  its `phase`, and its text arrives as several `response.output_text.delta`
  events. The stream ends with `response.completed` carrying the whole
  response.

  A response whose `status` is `"incomplete"` was cut off, for example by
  `max_output_tokens`. Its stream ends with `response.incomplete` instead, and
  an item whose own `status` is `"incomplete"` stops after its last delta,
  with no closing events.
  """
  def sse_events(response) do
    in_progress = %{response | "status" => "in_progress", "output" => []}

    items =
      response["output"]
      |> Enum.with_index()
      |> Enum.flat_map(fn {item, index} -> item_events(item, index) end)

    [
      %{"type" => "response.created", "response" => in_progress},
      %{"type" => "response.in_progress", "response" => in_progress}
    ]
    |> Kernel.++(items)
    |> Kernel.++([%{"type" => terminal_event(response), "response" => response}])
    |> Enum.with_index()
    |> Enum.map(fn {event, seq} -> Map.put(event, "sequence_number", seq) end)
  end

  defp item_events(%{"type" => "reasoning"} = item, index) do
    [
      %{
        "type" => "response.output_item.added",
        "output_index" => index,
        "item" => Map.put(item, "encrypted_content", nil)
      },
      %{"type" => "response.output_item.done", "output_index" => index, "item" => item}
    ]
  end

  defp item_events(%{"type" => "message", "content" => [%{"text" => text} = part]} = item, index) do
    ids = %{"item_id" => item["id"], "output_index" => index, "content_index" => 0}

    deltas =
      for piece <- text_pieces(text) do
        Map.merge(ids, %{
          "type" => "response.output_text.delta",
          "delta" => piece,
          "logprobs" => []
        })
      end

    [
      %{
        "type" => "response.output_item.added",
        "output_index" => index,
        "item" => %{item | "status" => "in_progress", "content" => []}
      },
      Map.merge(ids, %{"type" => "response.content_part.added", "part" => %{part | "text" => ""}})
    ] ++
      deltas ++
      [
        Map.merge(ids, %{"type" => "response.output_text.done", "text" => text, "logprobs" => []}),
        Map.merge(ids, %{"type" => "response.content_part.done", "part" => part}),
        %{"type" => "response.output_item.done", "output_index" => index, "item" => item}
      ]
  end

  defp item_events(%{"type" => "function_call", "status" => "incomplete"} = item, index) do
    item
    |> Map.put("status", "completed")
    |> item_events(index)
    |> Enum.take(2)
  end

  defp item_events(%{"type" => "function_call"} = item, index) do
    ids = %{"item_id" => item["id"], "output_index" => index}

    [
      %{
        "type" => "response.output_item.added",
        "output_index" => index,
        "item" => %{item | "status" => "in_progress", "arguments" => ""}
      },
      Map.merge(ids, %{
        "type" => "response.function_call_arguments.delta",
        "delta" => item["arguments"]
      }),
      Map.merge(ids, %{
        "type" => "response.function_call_arguments.done",
        "arguments" => item["arguments"]
      }),
      %{"type" => "response.output_item.done", "output_index" => index, "item" => item}
    ]
  end

  defp terminal_event(%{"status" => "incomplete"}), do: "response.incomplete"
  defp terminal_event(_response), do: "response.completed"

  # A few words per delta, as the API streams them.
  defp text_pieces(text) do
    text |> String.split(~r/(?<= )/) |> Enum.chunk_every(3) |> Enum.map(&Enum.join/1)
  end

  defp chunks(binary) when byte_size(binary) <= @chunk_size, do: [binary]

  defp chunks(binary) do
    rest = binary_part(binary, @chunk_size, byte_size(binary) - @chunk_size)
    [binary_part(binary, 0, @chunk_size) | chunks(rest)]
  end
end
