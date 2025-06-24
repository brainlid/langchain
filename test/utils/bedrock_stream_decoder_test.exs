defmodule LangChain.Utils.BedrockStreamDecoderTest do
  use ExUnit.Case
  use Mimic
  alias LangChain.Utils.AwsEventstreamDecoder
  alias LangChain.Utils.BedrockStreamDecoder

  @message <<0, 0, 1, 17, 0, 0, 0, 75, 18, 240, 42, 230, 11, 58, 101, 118, 101, 110, 116, 45, 116,
             121, 112, 101, 7, 0, 5, 99, 104, 117, 110, 107, 13, 58, 99, 111, 110, 116, 101, 110,
             116, 45, 116, 121, 112, 101, 7, 0, 16, 97, 112, 112, 108, 105, 99, 97, 116, 105, 111,
             110, 47, 106, 115, 111, 110, 13, 58, 109, 101, 115, 115, 97, 103, 101, 45, 116, 121,
             112, 101, 7, 0, 5, 101, 118, 101, 110, 116, 123, 34, 98, 121, 116, 101, 115, 34, 58,
             34, 101, 121, 74, 48, 101, 88, 66, 108, 73, 106, 111, 105, 89, 50, 57, 117, 100, 71,
             86, 117, 100, 70, 57, 105, 98, 71, 57, 106, 97, 49, 57, 107, 90, 87, 120, 48, 89, 83,
             73, 115, 73, 109, 108, 117, 90, 71, 86, 52, 73, 106, 111, 119, 76, 67, 74, 107, 90,
             87, 120, 48, 89, 83, 73, 54, 101, 121, 74, 48, 101, 88, 66, 108, 73, 106, 111, 105,
             100, 71, 86, 52, 100, 70, 57, 107, 90, 87, 120, 48, 89, 83, 73, 115, 73, 110, 82,
             108, 101, 72, 81, 105, 79, 105, 74, 68, 98, 50, 120, 118, 99, 109, 90, 49, 98, 67,
             66, 85, 97, 72, 74, 108, 89, 87, 82, 122, 73, 110, 49, 57, 34, 44, 34, 112, 34, 58,
             34, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113,
             114, 115, 116, 117, 118, 119, 120, 121, 122, 65, 66, 67, 68, 69, 70, 71, 72, 73, 34,
             125, 181, 231, 17, 159>>
  @message_decoded %{
    "delta" => %{"text" => "Colorful Threads", "type" => "text_delta"},
    "index" => 0,
    "type" => "content_block_delta"
  }
  @message_twice @message <> @message

  describe "integration" do
    test "decodes a single message" do
      assert decode_stream({@message, ""}) == {[@message_decoded], ""}
    end

    test "decodes multiple messages" do
      assert decode_stream({@message_twice, ""}) == {[@message_decoded, @message_decoded], ""}
    end

    test "returns buffer of incomplete messages" do
      <<incomplete::bitstring-size(bit_size(@message_twice) - 32), rest::bitstring>> =
        @message_twice

      <<expected_buffer::bitstring-size(bit_size(@message) - 32), _rest::bitstring>> = @message

      {chunks, buffer} = decode_stream({incomplete, ""})
      assert chunks == [@message_decoded]
      assert buffer == expected_buffer
      {chunks, buffer} = decode_stream({rest, buffer})
      assert chunks == [@message_decoded]
      assert buffer == ""
    end
  end

  describe "unit" do
    defp wrap_message(message) do
      %{"bytes" => message |> Jason.encode!() |> Base.encode64()} |> Jason.encode!()
    end

    defp text_delta_message(text) do
      wrap_message(%{
        "delta" => %{"text" => text, "type" => "text_delta"},
        "index" => 0,
        "type" => "content_block_delta"
      })
    end

    defp success_message, do: wrap_message(%{"type" => "content_block_start"})

    @exception_message %{
                         "internalServerError" => %{}
                       }
                       |> Jason.encode!()

    test "it passes through successfully decoded messages" do
      stub(AwsEventstreamDecoder, :decode, fn _ -> {:ok, success_message(), ""} end)
      assert decode_stream({"", ""}) == {[%{"type" => "content_block_start"}], ""}
    end

    test "it passes through an exception_message when bytes aren't present" do
      stub(AwsEventstreamDecoder, :decode, fn _ -> {:ok, @exception_message, ""} end)
      message = %{"internalServerError" => %{}, bedrock_exception: "internalServerError"}
      assert decode_stream({"", ""}) == {[message], ""}
    end

    test "it decodes multiple messages and returns chunks in the correct order" do
      AwsEventstreamDecoder
      |> expect(:decode, fn _ -> {:ok, text_delta_message("Color"), "remaining"} end)
      |> expect(:decode, fn _ -> {:ok, text_delta_message("ful Thr"), "remaining"} end)
      |> expect(:decode, fn _ -> {:ok, text_delta_message("eads"), ""} end)

      {chunks, buffer} = decode_stream({"", ""})
      assert chunks |> Enum.map(& &1["delta"]["text"]) == ["Color", "ful Thr", "eads"]

      assert chunks |> Enum.map(& &1["type"]) == [
               "content_block_delta",
               "content_block_delta",
               "content_block_delta"
             ]

      assert buffer == ""
    end

    test "it decodes multiple messages and returns chunks in the correct order when an error is returned" do
      AwsEventstreamDecoder
      |> expect(:decode, fn _ -> {:ok, text_delta_message("Color"), "remaining"} end)
      |> expect(:decode, fn _ -> {:ok, text_delta_message("ful Thr"), "remaining"} end)
      |> expect(:decode, fn _ -> {:ok, text_delta_message("eads"), "last remaining"} end)
      |> expect(:decode, fn _ -> {:error, "error"} end)

      {chunks, buffer} = decode_stream({"", ""})
      assert chunks |> Enum.map(& &1["delta"]["text"]) == ["Color", "ful Thr", "eads"]

      assert chunks |> Enum.map(& &1["type"]) == [
               "content_block_delta",
               "content_block_delta",
               "content_block_delta"
             ]

      assert buffer == "last remaining"
    end

    test "it decodes multiple messages and returns chunks in the correct order when an incomplete message is returned" do
      AwsEventstreamDecoder
      |> expect(:decode, fn _ -> {:ok, text_delta_message("Color"), "remaining"} end)
      |> expect(:decode, fn _ -> {:ok, text_delta_message("ful Thr"), "remaining"} end)
      |> expect(:decode, fn _ -> {:ok, text_delta_message("eads"), "last remaining"} end)
      |> expect(:decode, fn _ -> {:incomplete_message, "incomplete"} end)

      {chunks, buffer} = decode_stream({"", ""})
      assert chunks |> Enum.map(& &1["delta"]["text"]) == ["Color", "ful Thr", "eads"]

      assert chunks |> Enum.map(& &1["type"]) == [
               "content_block_delta",
               "content_block_delta",
               "content_block_delta"
             ]

      assert buffer == "last remaining"
    end
  end

  defdelegate decode_stream(data), to: BedrockStreamDecoder
end
