defmodule LangChain.RawCaptureAdapter do
  @moduledoc """
  A `Req` adapter that records the raw bytes of each HTTP exchange and then
  delegates to the normal Finch adapter.

  An adapter runs after every request step and before every response step, so
  `request.body` is the encoded JSON that goes on the wire and `response.body`
  is the untouched payload the server sent back, before decompression or JSON
  decoding. That is what makes this usable for recording fixtures: the bytes are
  the provider's, not a round trip through our own structs.

  A chat model reaches this through its `:req_config`, because `Req.merge/2`
  splits `:adapter` out of the option list before validating registered options:

      ChatOpenAIResponses.new!(%{
        model: "gpt-5.4",
        req_config: %{adapter: LangChain.RawCaptureAdapter, compressed: false}
      })

  `compressed: false` keeps the recorded body as readable JSON rather than a
  gzip blob.

  `request.body` is normalized from iodata to a binary, which is lossless and
  saves every caller from doing it.

  Exchanges accumulate in the process dictionary of whichever process made the
  request. A non-streaming `Req.post/1` runs in its caller, so a test that
  builds the model, runs the chain, and reads `exchanges/0` sees all of them.

  A streamed request hands its body to the `:into` collector instead, so the
  recorded `response_body` for one of those is empty. Capture SSE by issuing the
  request without a collector and keeping the buffered body.
  """

  @doc """
  Perform the request, record it, and return Req's `{request, response}` pair.
  """
  def run(request) do
    case Req.Finch.run(request) do
      {request, %Req.Response{} = response} ->
        record(%{
          url: URI.to_string(request.url),
          request_body: to_binary(request.body),
          status: response.status,
          response_headers: response.headers,
          response_body: response.body
        })

        {request, response}

      {request, exception} ->
        record(%{
          url: URI.to_string(request.url),
          request_body: to_binary(request.body),
          status: nil,
          response_headers: %{},
          response_body: nil,
          exception: exception
        })

        {request, exception}
    end
  end

  @doc """
  The exchanges recorded in this process, oldest first.
  """
  def exchanges, do: Enum.reverse(Process.get(__MODULE__, []))

  @doc """
  Discard the exchanges recorded in this process.
  """
  def reset, do: Process.delete(__MODULE__)

  defp to_binary(body) when is_binary(body), do: body
  defp to_binary(body) when is_list(body), do: IO.iodata_to_binary(body)
  defp to_binary(body), do: body

  defp record(exchange) do
    Process.put(__MODULE__, [exchange | Process.get(__MODULE__, [])])
  end
end
