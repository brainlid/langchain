defmodule LangChain.MessageDeltaTest do
  use ExUnit.Case
  doctest LangChain.MessageDelta
  import LangChain.Fixtures
  alias LangChain.Message
  alias LangChain.MessageDelta
  alias LangChain.LangChainError

  describe "new/1" do
    test "works with minimal attrs" do
      assert {:ok, %MessageDelta{} = msg} = MessageDelta.new(%{})
      assert msg.role == :unknown
      assert msg.content == nil
      assert msg.function_name == nil
      assert msg.arguments == nil
      assert msg.status == :incomplete
      assert msg.index == nil
    end

    test "accepts normal content attributes" do
      assert {:ok, %MessageDelta{} = msg} =
               MessageDelta.new(%{
                 "content" => "Hi!",
                 "role" => "assistant",
                 "index" => 1,
                 "status" => "complete"
               })

      assert msg.role == :assistant
      assert msg.content == "Hi!"
      assert msg.function_name == nil
      assert msg.arguments == nil
      assert msg.status == :complete
      assert msg.index == 1
    end

    test "accepts normal function attributes" do
      assert {:ok, %MessageDelta{} = msg} =
               MessageDelta.new(%{
                 "content" => nil,
                 "role" => "assistant",
                 "function_name" => "hello_world",
                 "arguments" => Jason.encode!(%{greeting: "Howdy"}),
                 "index" => 1,
                 "status" => "complete"
               })

      assert msg.role == :assistant
      assert msg.content == nil
      assert msg.function_name == "hello_world"
      assert msg.arguments == "{\"greeting\":\"Howdy\"}"
      assert msg.status == :complete
      assert msg.index == 1
    end

    test "accepts arguments as an empty string" do
      {:ok, msg} = MessageDelta.new(%{"role" => "assistant", "arguments" => " "})
      assert msg.role == :assistant
      assert msg.arguments == " "

      {:ok, msg} = MessageDelta.new(%{role: "assistant", arguments: " "})
      assert msg.role == :assistant
      assert msg.arguments == " "

      {:ok, msg} = MessageDelta.new(%{role: "assistant", content: "\r\n"})
      assert msg.role == :assistant
      assert msg.content == "\r\n"

      {:ok, msg} = MessageDelta.new(%{role: "assistant", content: "\n"})
      assert msg.role == :assistant
      assert msg.content == "\n"
    end

    test "returns error when invalid" do
      assert {:error, changeset} = MessageDelta.new(%{role: "invalid", index: "abc"})
      assert {"is invalid", _} = changeset.errors[:role]
      assert {"is invalid", _} = changeset.errors[:index]
    end
  end

  describe "new!/1" do
    test "returns struct when valid" do
      assert %MessageDelta{role: :assistant, content: "Hi!", status: :incomplete} =
               MessageDelta.new!(%{
                 "content" => "Hi!",
                 "role" => "assistant"
               })
    end

    test "raises exception when invalid" do
      assert_raise LangChainError, "role: is invalid; index: is invalid", fn ->
        MessageDelta.new!(%{role: "invalid", index: "abc"})
      end
    end
  end

  describe "merge_delta/2" do
    test "correctly merges assistant content message" do
      [first | rest] = delta_content_sample()

      merged =
        Enum.reduce(rest, first, fn new_delta, acc ->
          MessageDelta.merge_delta(acc, new_delta)
        end)

      expected = %LangChain.MessageDelta{
        content: "Hello! How can I assist you today?",
        index: 0,
        function_name: nil,
        role: :assistant,
        arguments: nil,
        status: :complete
      }

      assert merged == expected
    end

    test "correctly merges multiple tool calls in a delta" do
      [first | rest] = deltas_for_multiple_tool_calls()

      merged =
        Enum.reduce(rest, first, fn new_delta, acc ->
          MessageDelta.merge_delta(acc, new_delta)
        end)

      expected = %LangChain.MessageDelta{
        content: nil,
        index: 0,
        function_name: "GOES AWAY",
        role: :assistant,
        arguments: "{}",
        status: :complete
      }

      assert merged == expected
    end

    test "correctly merges function_call message with no arguments" do
      [first | rest] = delta_function_no_args()

      merged =
        Enum.reduce(rest, first, fn new_delta, acc ->
          MessageDelta.merge_delta(acc, new_delta)
        end)

      expected = %LangChain.MessageDelta{
        content: nil,
        index: 0,
        function_name: "hello_world",
        role: :assistant,
        arguments: "{}",
        status: :complete
      }

      assert merged == expected
    end

    test "correctly merges function_call message with streamed arguments" do
      [first | rest] = delta_function_streamed_args()

      merged =
        Enum.reduce(rest, first, fn new_delta, acc ->
          MessageDelta.merge_delta(acc, new_delta)
        end)

      expected = %LangChain.MessageDelta{
        content: nil,
        index: 0,
        function_name: "calculator",
        role: :assistant,
        arguments: "{\n  \"expression\": \"100 + 300 - 200\"\n}",
        status: :complete
      }

      assert merged == expected
    end

    test "correctly merges assistant content with function_call" do
      [first | rest] = delta_content_with_function_call() |> List.flatten()

      merged =
        Enum.reduce(rest, first, fn new_delta, acc ->
          MessageDelta.merge_delta(acc, new_delta)
        end)

      expected = %LangChain.MessageDelta{
        content:
          "Sure, I can help with that. First, let's check which regions are currently available for deployment on Fly.io. Please wait a moment while I fetch this information for you.",
        index: 0,
        function_name: "regions_list",
        role: :assistant,
        arguments: "{}",
        status: :complete
      }

      assert merged == expected
    end
  end

  describe "to_message/1" do
    test "transform a merged and complete MessageDelta to a Message" do
      # :assistant content type
      delta = %LangChain.MessageDelta{
        content: "Hello! How can I assist you?",
        role: :assistant,
        status: :complete
      }

      {:ok, %Message{} = msg} = MessageDelta.to_message(delta)
      assert msg.role == :assistant
      assert msg.content == "Hello! How can I assist you?"

      # :assistant type
      delta = %LangChain.MessageDelta{
        role: :assistant,
        function_name: "calculator",
        arguments: "{\n  \"expression\": \"100 + 300 - 200\"\n}",
        status: :complete
      }

      {:ok, %Message{} = msg} = MessageDelta.to_message(delta)
      assert msg.role == :assistant
      assert msg.function_name == "calculator"
      # parses the arguments
      assert msg.arguments == %{"expression" => "100 + 300 - 200"}
      assert msg.content == nil
    end

    test "does not transform an incomplete MessageDelta to a Message" do
      delta = %LangChain.MessageDelta{
        content: "Hello! How can I assist ",
        role: :assistant,
        status: :incomplete
      }

      assert {:error, "Cannot convert incomplete message"} = MessageDelta.to_message(delta)
    end

    test "transforms a delta stopped for length" do
      delta = %LangChain.MessageDelta{
        content: "Hello! How can I assist ",
        role: :assistant,
        status: :length
      }

      assert {:ok, message} = MessageDelta.to_message(delta)
      assert message.role == :assistant
      assert message.content == "Hello! How can I assist "
      assert message.status == :length
    end

    test "for a function_call, return an error when delta is invalid" do
      # a partially merged delta is invalid. It may have the "complete" flag but
      # if previous message deltas are missing and were not merged, the
      # to_message function will fail.
      delta = %LangChain.MessageDelta{
        role: :assistant,
        function_name: "calculator",
        arguments: "{\n  \"expression\": \"100 + 300 - 200\"",
        status: :complete
      }

      {:error, reason} = MessageDelta.to_message(delta)
      assert reason == "arguments: invalid JSON function arguments"
    end
  end
end
