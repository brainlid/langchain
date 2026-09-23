defmodule LangChain.ChatModels.ChatOpenAIResponsesPhaseLiveTest do
  @moduledoc """
  Live capture of an OpenAI Responses turn in which the model narrates before it
  acts.

  The Responses API labels each assistant message item with a `phase`. A
  `"commentary"` item is the model speaking about work it intends to do;
  `"final_answer"` is the closeout. `LLMChain` ends a turn on an assistant
  message that carries no tool calls, so a commentary item with nothing attached
  reads to the chain as a finished answer and the run stops.

  These tests exist to record what that looks like on the wire. They answer:

  - whether a non-streaming `"output"` array carries `"phase"` as a sibling of
    `"role"` and `"content"` on a message item
  - whether the streamed `response.output_item.done` event carries `"phase"`
    inside `"item"`, or whether the label only shows up on `response.completed`
  - what `response.status` is on a response whose only message is commentary
  - what the API does with a follow-up request whose `input` ends on an
    assistant message, which is the request shape a chain has to send to let a
    narrated turn continue

  The prompt is the reproduction case: a system block that tells the model to
  rephrase the goal before touching a tool and to narrate each step, plus an
  open-ended investigation request. The instruction to speak before calling any
  tool is the load-bearing part, because it asks for an utterance that is by
  construction neither the answer nor a tool call.

  Every exchange is written to `test/fixtures/openai_phase/captures/`, which is
  scratch. The fixtures the offline tests read sit one level up and are curated
  by hand from runs of this test, so a fresh run cannot overwrite them. The
  model is non-deterministic, so a run that produces no commentary item is a
  miss rather than a failure; the summary printed to stdout says which
  happened.

  `verbose_api: true` prints the exchange as it happens, but `IO.inspect`
  truncates collections at 50 entries and strings at 4096 bytes, so the files
  rather than the console output are the record.

  Run with:

      mix test test/chat_models/chat_open_ai_responses_phase_live_test.exs --include live_open_ai

  Override the model with `OPENAI_PHASE_TEST_MODEL`.
  """
  use LangChain.BaseCase

  alias LangChain.ChatModels.ChatOpenAIResponses
  alias LangChain.Chains.LLMChain
  alias LangChain.Config
  alias LangChain.Function
  alias LangChain.Message
  alias LangChain.RawCaptureAdapter

  @moduletag live_call: true, live_open_ai: true
  @moduletag timeout: 600_000

  @capture_dir Path.expand("../fixtures/openai_phase/captures", __DIR__)

  @default_model "gpt-5.4"

  @system_prompt """
  <tool_preambles>
  - Always begin by rephrasing the user's goal in a friendly, clear, and concise manner, before calling any tools.
  - Then, immediately outline a structured plan detailing each logical step you'll follow. - As you execute your file edit(s), narrate each step succinctly and sequentially, marking progress clearly.
  - Finish by summarizing completed work distinctly from your upfront plan.
  </tool_preambles>
  """

  @user_prompt "why did this fail check a bunch of things pls"

  # The preamble block on its own produces commentary, but the model tends to
  # attach the tool calls to the same response, which keeps the chain running.
  # Telling it to hand over after the plan isolates the commentary item.
  @hand_off_instruction """

  Send your restated goal and your plan as your first reply, on its own. Do not
  call any tools in that same reply. Begin calling tools only after that reply
  has been sent.
  """

  setup do
    File.mkdir_p!(@capture_dir)
    RawCaptureAdapter.reset()
    :ok
  end

  describe "non-streaming" do
    test "captures a narrated turn and the follow-up request that continues it" do
      model = chat_model(%{stream: false})
      tool = inspect_tool()

      {:ok, chain} =
        %{llm: model, verbose: false}
        |> LLMChain.new!()
        |> LLMChain.add_messages([
          Message.new_system!(@system_prompt),
          Message.new_user!(@user_prompt)
        ])
        |> LLMChain.add_tools([tool])
        |> LLMChain.run(mode: :while_needs_response)

      write_exchanges("non_streaming")

      IO.puts("\n=== non-streaming run ===")
      Enum.each(RawCaptureAdapter.exchanges(), &summarize_exchange/1)
      summarize_chain(chain)

      assert [_ | _] = RawCaptureAdapter.exchanges()

      # The chain has stopped. Whatever it stopped on is what the next request
      # has to carry back, so send exactly that and record how the API answers a
      # request whose input ends on an assistant message.
      RawCaptureAdapter.reset()

      IO.puts("\n=== replay: follow-up request built from the stopped chain ===")
      result = ChatOpenAIResponses.call(model, chain.messages, [tool])

      write_exchanges("non_streaming_followup")
      Enum.each(RawCaptureAdapter.exchanges(), &summarize_exchange/1)

      case result do
        {:ok, %Message{} = message} ->
          IO.puts(
            "follow-up accepted. role=#{message.role} tool_calls=#{length(message.tool_calls || [])}"
          )

        {:ok, other} ->
          IO.puts("follow-up returned: #{inspect(other)}")

        {:error, error} ->
          IO.puts("follow-up rejected: #{inspect(error)}")
      end

      assert [_ | _] = RawCaptureAdapter.exchanges()
    end
  end

  describe "streaming" do
    test "captures the raw SSE event sequence for the same prompt" do
      model = chat_model(%{stream: true})
      tool = inspect_tool()

      messages = [
        Message.new_system!(@system_prompt),
        Message.new_user!(@user_prompt)
      ]

      # Issue the request without a collector so Req buffers the whole
      # `text/event-stream` body. The adapter cannot record a streamed body,
      # because the library's `:into` collector consumes it as it arrives.
      body = ChatOpenAIResponses.for_api(model, messages, [tool])

      {:ok, response} =
        Req.post(
          url: model.endpoint,
          json: body,
          auth: {:bearer, api_key()},
          receive_timeout: model.receive_timeout,
          compressed: false,
          retry: false
        )

      raw = if is_binary(response.body), do: response.body, else: inspect(response.body)

      write_capture("streaming_request.json", encode(body))
      write_capture("streaming_events.txt", raw)

      IO.puts("\n=== streaming run (status #{response.status}) ===")
      summarize_sse(raw)

      assert response.status == 200
      assert raw =~ "response.completed"
    end
  end

  describe "commentary-only" do
    test "hunts for a response whose only output item is a commentary message" do
      tool = inspect_tool()

      # A stall needs the model to end a response on commentary, with no tool
      # call to keep the loop going and no answer in it. The reported symptom
      # was mid-trajectory rather than on the opening turn, so the hunt varies
      # the model, the reasoning effort, and whether the prompt asks the model
      # to hand off after its plan.
      attempts =
        for model_name <- [default_model(), "gpt-5.6-terra"],
            {effort_label, overrides} <- [
              {"default", %{}},
              {"high", %{reasoning: %{effort: :high}}}
            ],
            {prompt_label, system} <- [
              {"handoff", @system_prompt <> @hand_off_instruction},
              {"preamble", @system_prompt}
            ] do
          {"#{model_name}_#{effort_label}_#{prompt_label}",
           Map.put(overrides, :model, model_name), system}
        end

      outcome =
        Enum.reduce_while(attempts, :not_found, fn {label, overrides, system}, _acc ->
          RawCaptureAdapter.reset()
          model = chat_model(Map.merge(%{stream: false}, overrides))

          messages = [Message.new_system!(system), Message.new_user!(@user_prompt)]
          result = ChatOpenAIResponses.call(model, messages, [tool])

          IO.puts("\n=== attempt: #{label} ===")
          Enum.each(RawCaptureAdapter.exchanges(), &summarize_exchange/1)
          write_exchanges("hunt_#{label}")

          if commentary_only?(RawCaptureAdapter.exchanges()) do
            IO.puts("captured a commentary-only response on attempt #{label}")
            demonstrate_early_stop(result, tool)
            {:halt, :found}
          else
            {:cont, :not_found}
          end
        end)

      if outcome == :not_found do
        IO.puts("\nno commentary-only response in #{length(attempts)} opening turns")
      end

      # The model chooses whether to hand off, so a miss is a miss rather than a
      # failure. The run either way records what came back.
      assert outcome in [:found, :not_found]
    end
  end

  # A response whose whole output is commentary is the shape that ends a turn
  # early: no tool call to keep the loop going and no answer in it. Reasoning
  # items sit alongside and do not change that.
  defp commentary_only?(exchanges) do
    Enum.any?(exchanges, fn %{response_body: raw} ->
      case Jason.decode(raw || "") do
        {:ok, %{"output" => output}} ->
          speech = Enum.reject(output, &(&1["type"] == "reasoning"))

          speech != [] and
            Enum.all?(speech, &(&1["type"] == "message" and &1["phase"] == "commentary"))

        _ ->
          false
      end
    end)
  end

  # What the chain makes of it: an assistant message with no tool calls closes
  # the turn, so the run ends on the narration.
  defp demonstrate_early_stop({:ok, %Message{} = message}, tool) do
    chain =
      %{llm: chat_model(%{stream: false}), verbose: false}
      |> LLMChain.new!()
      |> LLMChain.add_tools([tool])
      |> LLMChain.add_message(message)

    IO.puts(
      "chain sees: role=#{message.role} tool_calls=#{length(message.tool_calls || [])} " <>
        "content_parts=#{length(message.content || [])} needs_response=#{chain.needs_response}"
    )
  end

  defp demonstrate_early_stop(other, _tool),
    do: IO.puts("no message to inspect: #{inspect(other)}")

  # -- model and tool ---------------------------------------------------------

  defp chat_model(overrides) do
    %{
      model: default_model(),
      receive_timeout: 300_000,
      # `IO.inspect/2` truncates, so the captured files are the record. Set
      # OPENAI_PHASE_VERBOSE=1 to also watch the exchange go past.
      verbose_api: System.get_env("OPENAI_PHASE_VERBOSE") in ["1", "true"],
      # The tee adapter records the bytes; `compressed: false` keeps the
      # recorded response body readable rather than gzipped.
      req_config: %{adapter: RawCaptureAdapter, compressed: false}
    }
    |> Map.merge(overrides)
    |> ChatOpenAIResponses.new!()
  end

  defp default_model, do: System.get_env("OPENAI_PHASE_TEST_MODEL", @default_model)

  defp api_key, do: Config.resolve(:openai_key, "")

  # A tool with a canned answer. The point is to give the model something to
  # announce before calling, not to model anything real.
  defp inspect_tool do
    Function.new!(%{
      name: "inspect_resource",
      description:
        "Inspect one named resource belonging to the failed deployment and return its current details.",
      parameters_schema: %{
        "type" => "object",
        "properties" => %{
          "resource" => %{
            "type" => "string",
            "description" =>
              "The resource to inspect, such as \"deployment\", \"pod_logs\", \"events\", or \"image\"."
          }
        },
        "required" => ["resource"]
      },
      function: fn %{"resource" => resource}, _context ->
        {:ok,
         "#{resource}: status=Failed reason=ImagePullBackOff " <>
           "image=registry.internal/app:v4.2.1 detail=\"manifest unknown\" restarts=6"}
      end
    })
  end

  # -- capture ----------------------------------------------------------------

  defp write_exchanges(label) do
    RawCaptureAdapter.exchanges()
    |> Enum.with_index(1)
    |> Enum.each(fn {exchange, index} ->
      write_capture("#{label}_#{index}_request.json", pretty(exchange.request_body))
      write_capture("#{label}_#{index}_response.json", pretty(exchange.response_body))
    end)
  end

  defp write_capture(name, contents) do
    path = Path.join(@capture_dir, name)
    File.write!(path, contents)
    IO.puts("wrote #{Path.relative_to_cwd(path)}")
  end

  defp pretty(nil), do: ""
  defp pretty(raw) when is_list(raw), do: raw |> IO.iodata_to_binary() |> pretty()

  defp pretty(raw) when is_binary(raw) do
    case Jason.decode(raw) do
      {:ok, decoded} -> encode(decoded)
      {:error, _} -> raw
    end
  end

  defp pretty(other), do: encode(other)

  defp encode(term), do: Jason.encode!(term, pretty: true)

  # -- summaries --------------------------------------------------------------

  defp summarize_exchange(%{request_body: raw} = exchange) do
    summarize_request(raw)
    summarize_response(exchange)
  end

  # The `input` array is the question 2.3 turns on: what an assistant turn looks
  # like on the way back out, and whether it carries `phase`.
  defp summarize_request(raw) do
    case Jason.decode(raw || "") do
      {:ok, %{"input" => input}} ->
        IO.puts("  request input: #{length(input)} items")

        Enum.each(input, fn item ->
          IO.puts(
            "    - type=#{inspect(item["type"])} role=#{inspect(item["role"])} " <>
              "phase=#{inspect(item["phase"])} name=#{inspect(item["name"])}"
          )
        end)

      _ ->
        IO.puts("  request: (unparsed)")
    end
  end

  defp summarize_response(%{status: status, response_body: raw}) do
    IO.puts("HTTP #{status}")

    case Jason.decode(raw || "") do
      {:ok, %{"output" => output} = decoded} ->
        IO.puts("  response.status: #{inspect(decoded["status"])}")
        IO.puts("  output items: #{length(output)}")
        Enum.each(output, &summarize_output_item/1)

      {:ok, decoded} ->
        IO.puts("  #{inspect(decoded, limit: :infinity, printable_limit: 400)}")

      {:error, _} ->
        IO.puts("  (body was not JSON) #{String.slice(raw || "", 0, 400)}")
    end
  end

  defp summarize_output_item(%{"type" => "message"} = item) do
    text =
      item
      |> Map.get("content", [])
      |> Enum.map(&(Map.get(&1, "text") || Map.get(&1, "refusal") || ""))
      |> Enum.join(" ")

    IO.puts("  - message phase=#{inspect(Map.get(item, "phase"))} text=#{preview(text)}")
  end

  defp summarize_output_item(%{"type" => "function_call"} = item) do
    IO.puts("  - function_call name=#{item["name"]} args=#{preview(item["arguments"])}")
  end

  defp summarize_output_item(%{"type" => type} = item) do
    IO.puts("  - #{type} keys=#{inspect(Map.keys(item))}")
  end

  defp summarize_chain(%LLMChain{} = chain) do
    IO.puts("\nchain stopped with #{length(chain.messages)} messages")

    Enum.each(chain.messages, fn message ->
      IO.puts(
        "  #{message.role} tool_calls=#{length(message.tool_calls || [])} " <>
          "content=#{preview(content_text(message))}"
      )
    end)
  end

  defp content_text(%Message{content: parts}) when is_list(parts) do
    parts
    |> Enum.map(&(Map.get(&1, :content) || ""))
    |> Enum.filter(&is_binary/1)
    |> Enum.join(" ")
  end

  defp content_text(%Message{content: content}) when is_binary(content), do: content
  defp content_text(%Message{}), do: ""

  # Report which event types carried a `phase`, which is the streaming question
  # the decoder depends on.
  defp summarize_sse(raw) do
    events =
      raw
      |> String.split("\n\n", trim: true)
      |> Enum.flat_map(fn chunk ->
        chunk
        |> String.split("\n")
        |> Enum.find_value(fn
          "data: " <> json -> json
          _ -> nil
        end)
        |> case do
          nil -> []
          json -> List.wrap(json |> Jason.decode() |> elem(1))
        end
      end)
      |> Enum.filter(&is_map/1)

    counts = events |> Enum.frequencies_by(&Map.get(&1, "type"))

    IO.puts("  #{length(events)} events")
    Enum.each(counts, fn {type, count} -> IO.puts("  #{count}x #{type}") end)

    carrying_phase =
      events
      |> Enum.filter(&(&1 |> Jason.encode!() |> String.contains?("\"phase\"")))
      |> Enum.map(&Map.get(&1, "type"))
      |> Enum.uniq()

    IO.puts("  events mentioning \"phase\": #{inspect(carrying_phase)}")

    events
    |> Enum.filter(&(Map.get(&1, "type") == "response.output_item.done"))
    |> Enum.each(fn event ->
      item = Map.get(event, "item", %{})

      IO.puts(
        "  output_item.done type=#{inspect(item["type"])} phase=#{inspect(item["phase"])} " <>
          "keys=#{inspect(Map.keys(item))}"
      )
    end)
  end

  defp preview(nil), do: "nil"

  defp preview(text) when is_binary(text) do
    text |> String.slice(0, 160) |> inspect()
  end

  defp preview(other), do: inspect(other)
end
