defmodule LangChain.Utils.BedrockStreamDecoder do
  alias LangChain.Utils.AwsEventstreamDecoder
  require Logger

  def decode_stream({chunk, buffer}, chunks \\ []) do
    combined_data = buffer <> chunk

    case decode_chunk(combined_data) do
      {:ok, chunk, remaining} ->
        chunks = [chunk | chunks]

        if byte_size(remaining) > 0 do
          decode_stream({"", remaining}, chunks)
        else
          {Enum.reverse(chunks), ""}
        end

      {:incomplete_message, _} ->
        {chunks, combined_data}

      {:error, error} ->
        Logger.error("Failed to decode Bedrock chunk: #{inspect(error)}")
        {chunks, combined_data}
    end
  end

  defp decode_chunk(chunk) do
    with {:ok, decoded_message, remaining} <- AwsEventstreamDecoder.decode(chunk),
         {:ok, %{"bytes" => bytes}} <- decode_json(decoded_message),
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
end
