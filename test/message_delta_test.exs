defmodule Langchain.MessageDeltaTest do
  use ExUnit.Case
  doctest Langchain.MessageDelta
  alias Langchain.Message
  alias Langchain.MessageDelta
  alias Langchain.LangchainError

  describe "new/1" do
    test "works with minimal attrs" do
      assert {:ok, %MessageDelta{} = msg} = MessageDelta.new(%{})
      assert msg.role == :unknown
      assert msg.content == nil
      assert msg.function_name == nil
      assert msg.arguments == nil
      assert msg.complete == false
      assert msg.index == nil
    end

    test "accepts normal content attributes" do
      assert {:ok, %MessageDelta{} = msg} =
               MessageDelta.new(%{
                 "content" => "Hi!",
                 "role" => "assistant",
                 "index" => 1,
                 "complete" => true
               })

      assert msg.role == :assistant
      assert msg.content == "Hi!"
      assert msg.function_name == nil
      assert msg.arguments == nil
      assert msg.complete == true
      assert msg.index == 1
    end

    test "accepts normal function attributes" do
      assert {:ok, %MessageDelta{} = msg} =
               MessageDelta.new(%{
                 "content" => nil,
                 "role" => "function_call",
                 "function_name" => "hello_world",
                 "arguments" => Jason.encode!(%{greeting: "Howdy"}),
                 "index" => 1,
                 "complete" => true
               })

      assert msg.role == :function_call
      assert msg.content == nil
      assert msg.function_name == "hello_world"
      assert msg.arguments == "{\"greeting\":\"Howdy\"}"
      assert msg.complete == true
      assert msg.index == 1
    end

    test "accepts arguments as an empty string" do
      {:ok, msg} = MessageDelta.new(%{"role" => "function_call", "arguments" => " "})
      assert msg.role == :function_call
      assert msg.arguments == " "

      {:ok, msg} = MessageDelta.new(%{role: "function_call", arguments: " "})
      assert msg.role == :function_call
      assert msg.arguments == " "
    end

    test "returns error when invalid" do
      assert {:error, changeset} = MessageDelta.new(%{role: "invalid", index: "abc"})
      assert {"is invalid", _} = changeset.errors[:role]
      assert {"is invalid", _} = changeset.errors[:index]
    end
  end

  describe "new!/1" do
    test "returns struct when valid" do
      assert %MessageDelta{role: :assistant, content: "Hi!", complete: false} =
               MessageDelta.new!(%{
                 "content" => "Hi!",
                 "role" => "assistant"
               })
    end

    test "raises exception when invalid" do
      assert_raise LangchainError, "role: is invalid; index: is invalid", fn ->
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

      expected = %Langchain.MessageDelta{
        content: "Hello! How can I assist you today?",
        index: 0,
        function_name: nil,
        role: :assistant,
        arguments: nil,
        complete: true
      }

      assert merged == expected
    end

    test "correctly merges function_call message with no arguments" do
      [first | rest] = delta_function_no_args()

      merged =
        Enum.reduce(rest, first, fn new_delta, acc ->
          MessageDelta.merge_delta(acc, new_delta)
        end)

      expected = %Langchain.MessageDelta{
        content: nil,
        index: 0,
        function_name: "hello_world",
        role: :function_call,
        arguments: "{}",
        complete: true
      }

      assert merged == expected
    end

    test "correctly merges function_call message with streamed arguments" do
      [first | rest] = delta_function_streamed_args()

      merged =
        Enum.reduce(rest, first, fn new_delta, acc ->
          MessageDelta.merge_delta(acc, new_delta)
        end)

      expected = %Langchain.MessageDelta{
        content: nil,
        index: 0,
        function_name: "calculator",
        role: :function_call,
        arguments: "{\n  \"expression\": \"100 + 300 - 200\"\n}",
        complete: true
      }

      assert merged == expected
    end
  end

  describe "to_message/1" do
    test "transform a merged and complete MessageDelta to a Message" do
      # :assistant content type
      delta = %Langchain.MessageDelta{
        content: "Hello! How can I assist you?",
        role: :assistant,
        complete: true
      }

      {:ok, %Message{} = msg} = MessageDelta.to_message(delta)
      assert msg.role == :assistant
      assert msg.content == "Hello! How can I assist you?"

      # :function_call type
      delta = %Langchain.MessageDelta{
        role: :function_call,
        function_name: "calculator",
        arguments: "{\n  \"expression\": \"100 + 300 - 200\"\n}",
        complete: true
      }

      {:ok, %Message{} = msg} = MessageDelta.to_message(delta)
      assert msg.role == :function_call
      assert msg.function_name == "calculator"
      # parses the arguments
      assert msg.arguments == %{"expression" => "100 + 300 - 200"}
      assert msg.content == nil
    end

    test "does not transform an incomplete MessageDelta to a Message" do
      delta = %Langchain.MessageDelta{
        content: "Hello! How can I assist ",
        role: :assistant,
        complete: false
      }

      assert {:error, "Cannot convert incomplete message"} = MessageDelta.to_message(delta)
    end

    test "for a function_call, return an error when delta is invalid" do
      # a partially merged delta is invalid. It may have the "complete" flag but
      # if previous message deltas are missing and were not merged, the
      # to_message function will fail.
      delta = %Langchain.MessageDelta{
        role: :function_call,
        function_name: "calculator",
        arguments: "{\n  \"expression\": \"100 + 300 - 200\"",
        complete: true
      }

      {:error, reason} = MessageDelta.to_message(delta)
      assert reason == "arguments: invalid JSON function arguments"
    end
  end

  defp delta_content_sample do
    # built from actual responses parsed to MessageDeltas
    # results = Enum.flat_map(delta_content, &ChatOpenAI.do_process_response(&1))
    # IO.inspect results
    [
      %Langchain.MessageDelta{
        content: nil,
        index: 0,
        function_name: nil,
        role: :assistant,
        arguments: nil,
        complete: false
      },
      %Langchain.MessageDelta{
        content: "Hello",
        index: 0,
        function_name: nil,
        role: :unknown,
        arguments: nil,
        complete: false
      },
      %Langchain.MessageDelta{
        content: "!",
        index: 0,
        function_name: nil,
        role: :unknown,
        arguments: nil,
        complete: false
      },
      %Langchain.MessageDelta{
        content: " How",
        index: 0,
        function_name: nil,
        role: :unknown,
        arguments: nil,
        complete: false
      },
      %Langchain.MessageDelta{
        content: " can",
        index: 0,
        function_name: nil,
        role: :unknown,
        arguments: nil,
        complete: false
      },
      %Langchain.MessageDelta{
        content: " I",
        index: 0,
        function_name: nil,
        role: :unknown,
        arguments: nil,
        complete: false
      },
      %Langchain.MessageDelta{
        content: " assist",
        index: 0,
        function_name: nil,
        role: :unknown,
        arguments: nil,
        complete: false
      },
      %Langchain.MessageDelta{
        content: " you",
        index: 0,
        function_name: nil,
        role: :unknown,
        arguments: nil,
        complete: false
      },
      %Langchain.MessageDelta{
        content: " today",
        index: 0,
        function_name: nil,
        role: :unknown,
        arguments: nil,
        complete: false
      },
      %Langchain.MessageDelta{
        content: "?",
        index: 0,
        function_name: nil,
        role: :unknown,
        arguments: nil,
        complete: false
      },
      %Langchain.MessageDelta{
        content: nil,
        index: 0,
        function_name: nil,
        role: :unknown,
        arguments: nil,
        complete: true
      }
    ]
  end

  defp delta_function_no_args() do
    [
      %Langchain.MessageDelta{
        content: nil,
        index: 0,
        function_name: "hello_world",
        role: :function_call,
        arguments: nil,
        complete: false
      },
      %Langchain.MessageDelta{
        content: nil,
        index: 0,
        function_name: nil,
        role: :function_call,
        arguments: "{}",
        complete: false
      },
      %Langchain.MessageDelta{
        content: nil,
        index: 0,
        function_name: nil,
        role: :unknown,
        arguments: nil,
        complete: true
      }
    ]
  end

  defp delta_function_streamed_args() do
    [
      %Langchain.MessageDelta{
        content: nil,
        index: 0,
        function_name: "calculator",
        role: :function_call,
        arguments: "",
        complete: false
      },
      %Langchain.MessageDelta{
        content: nil,
        index: 0,
        function_name: nil,
        role: :function_call,
        arguments: "{\n",
        complete: false
      },
      %Langchain.MessageDelta{
        content: nil,
        index: 0,
        function_name: nil,
        role: :function_call,
        arguments: " ",
        complete: false
      },
      %Langchain.MessageDelta{
        content: nil,
        index: 0,
        function_name: nil,
        role: :function_call,
        arguments: " \"",
        complete: false
      },
      %Langchain.MessageDelta{
        content: nil,
        index: 0,
        function_name: nil,
        role: :function_call,
        arguments: "expression",
        complete: false
      },
      %Langchain.MessageDelta{
        content: nil,
        index: 0,
        function_name: nil,
        role: :function_call,
        arguments: "\":",
        complete: false
      },
      %Langchain.MessageDelta{
        content: nil,
        index: 0,
        function_name: nil,
        role: :function_call,
        arguments: " \"",
        complete: false
      },
      %Langchain.MessageDelta{
        content: nil,
        index: 0,
        function_name: nil,
        role: :function_call,
        arguments: "100",
        complete: false
      },
      %Langchain.MessageDelta{
        content: nil,
        index: 0,
        function_name: nil,
        role: :function_call,
        arguments: " +",
        complete: false
      },
      %Langchain.MessageDelta{
        content: nil,
        index: 0,
        function_name: nil,
        role: :function_call,
        arguments: " ",
        complete: false
      },
      %Langchain.MessageDelta{
        content: nil,
        index: 0,
        function_name: nil,
        role: :function_call,
        arguments: "300",
        complete: false
      },
      %Langchain.MessageDelta{
        content: nil,
        index: 0,
        function_name: nil,
        role: :function_call,
        arguments: " -",
        complete: false
      },
      %Langchain.MessageDelta{
        content: nil,
        index: 0,
        function_name: nil,
        role: :function_call,
        arguments: " ",
        complete: false
      },
      %Langchain.MessageDelta{
        content: nil,
        index: 0,
        function_name: nil,
        role: :function_call,
        arguments: "200",
        complete: false
      },
      %Langchain.MessageDelta{
        content: nil,
        index: 0,
        function_name: nil,
        role: :function_call,
        arguments: "\"\n",
        complete: false
      },
      %Langchain.MessageDelta{
        content: nil,
        index: 0,
        function_name: nil,
        role: :function_call,
        arguments: "}",
        complete: false
      },
      %Langchain.MessageDelta{
        content: nil,
        index: 0,
        function_name: nil,
        role: :unknown,
        arguments: nil,
        complete: true
      }
    ]
  end
end
