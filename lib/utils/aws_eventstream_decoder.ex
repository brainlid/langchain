defmodule LangChain.Utils.AwsEventstreamDecoder do
  @moduledoc """
  Decodes AWS messages in the application/vnd.amazon.eventstream content-type.
  Ignores the headers because on Bedrock it's the same content type, event type & message type headers in every message.
  """

  def decode(<<
        message_length::32,
        headers_length::32,
        prelude_checksum::32,
        headers::binary-size(headers_length),
        body::binary-size(message_length - headers_length - 16),
        message_checksum::32,
        rest::bitstring
      >>) do
    message_without_checksum =
      <<message_length::32, headers_length::32, prelude_checksum::32,
        headers::binary-size(headers_length),
        body::binary-size(message_length - headers_length - 16)>>

    with :ok <-
           verify_checksum(<<message_length::32, headers_length::32>>, prelude_checksum, :prelude),
         :ok <- verify_checksum(message_without_checksum, message_checksum, :message) do
      {:ok, body, rest}
    end
  end

  def decode(<<message_length::32, _message::bitstring>> = data) do
    {:incomplete_message, "Expected message length #{message_length} but got #{byte_size(data)}"}
  end

  def decode(_) do
    {:error, "Unable to decode message"}
  end

  defp verify_checksum(data, checksum, part) do
    if :erlang.crc32(data) == checksum do
      :ok
    else
      {:error, "Checksum mismatch for #{part}"}
    end
  end
end
