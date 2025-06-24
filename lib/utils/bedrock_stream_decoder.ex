defmodule LangChain.Utils.BedrockStreamDecoder do
  alias LangChain.Utils.AwsEventstreamDecoder
  require Logger

  def decode_stream({chunk, buffer}, chunks \\ []) do
    combined_data = buffer <> chunk

    case decode_chunk(combined_data) do
      {:ok, chunk, remaining} ->
        chunks = [chunk | chunks]
        finish_or_decode_remaining(chunks, remaining)

      {:incomplete_message, _} ->
        finish(chunks, combined_data)

      {:exception_response, response, remaining} ->
        chunks = [response | chunks]
        finish_or_decode_remaining(chunks, remaining)

      {:error, error} ->
        Logger.error("Failed to decode Bedrock chunk: #{inspect(error)}")
        finish(chunks, combined_data)
    end
  end

  defp finish_or_decode_remaining(chunks, remaining) when byte_size(remaining) > 0 do
    decode_stream({"", remaining}, chunks)
  end

  defp finish_or_decode_remaining(chunks, remaining) do
    finish(chunks, remaining)
  end

  defp finish(chunks, remaining) do
    {Enum.reverse(chunks), remaining}
  end

  defp decode_chunk(chunk) do
    with {:ok, decoded_message, remaining} <- AwsEventstreamDecoder.decode(chunk),
         {:ok, response_json} <- decode_json(decoded_message),
         {:ok, bytes} <- get_bytes(response_json, remaining),
         {:ok, json} <- decode_base64(bytes),
         {:ok, payload} <- decode_json(json) do
      {:ok, payload, remaining}
    end
  end

  defp decode_json(data) do
    case Jason.decode(data) do
      {:ok, json} ->
        {:ok, json}

      {:error, error} ->
        {:error, "Unable to decode JSON: #{inspect(error)}"}
    end
  end

  defp decode_base64(bytes) do
    case Base.decode64(bytes) do
      {:ok, bytes} ->
        {:ok, bytes}

      :error ->
        {:error, "Unable to decode base64 \"bytes\" from Bedrock response"}
    end
  end

  defp get_bytes(%{"bytes" => bytes}, _remaining) do
    {:ok, bytes}
  end

  # bytes is likely missing from the response in exception cases
  # https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_InvokeModelWithResponseStream.html
  defp get_bytes(response, remaining) do
    Logger.debug("Bedrock response is an exception: #{inspect(response)}")
    exception_message = Map.keys(response) |> Enum.join(", ")
    # Make it easier to match on this pattern in process_data fns
    response = Map.put(response, :bedrock_exception, exception_message)
    {:exception_response, response, remaining}
  end
end
