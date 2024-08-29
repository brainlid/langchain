defmodule LangChain.Message.ToolCallTest do
  use ExUnit.Case
  doctest LangChain.Message.ToolCall
  alias LangChain.Message.ToolCall
  alias LangChain.LangChainError

  describe "new/1" do
    test "works with minimal attrs" do
      assert {:ok, %ToolCall{} = msg} = ToolCall.new(%{})
      assert msg.status == :incomplete
      assert msg.type == :function
      assert msg.call_id == nil
      assert msg.name == nil
      assert msg.arguments == nil
      assert msg.index == nil
    end

    test "accepts normal content attributes" do
      assert {:ok, %ToolCall{} = msg} =
               ToolCall.new(%{
                 "status" => :incomplete,
                 "type" => "function",
                 "index" => 1,
                 "call_id" => "call_asdf",
                 "name" => "hello_world",
                 "arguments" => "{\"key\": 1}"
               })

      assert msg.status == :incomplete
      assert msg.type == :function
      assert msg.name == "hello_world"
      assert msg.arguments == "{\"key\": 1}"
      assert msg.call_id == "call_asdf"
      assert msg.index == 1
    end

    test "parses arguments when complete" do
      assert {:ok, %ToolCall{} = msg} =
               ToolCall.new(%{
                 "status" => :complete,
                 "type" => "function",
                 "index" => 0,
                 "call_id" => "call_asdf",
                 "name" => "hello_world",
                 "arguments" => "{\"key\": 1}"
               })

      assert msg.status == :complete
      assert msg.type == :function
      assert msg.name == "hello_world"
      assert msg.arguments == %{"key" => 1}
      assert msg.call_id == "call_asdf"
      assert msg.index == 0
    end
  end

  describe "complete/1" do
    test "returns already complete as unchanged" do
      call =
        ToolCall.new!(%{
          "status" => :complete,
          "type" => "function",
          "index" => 0,
          "call_id" => "call_asdf",
          "name" => "hello_world",
          "arguments" => "{\"key\": 1}"
        })

      assert {:ok, call} == ToolCall.complete(call)
    end

    test "completes an incomplete and converts the arguments" do
      call =
        ToolCall.new!(%{
          "status" => :incomplete,
          "type" => "function",
          "index" => 0,
          "call_id" => "call_asdf",
          "name" => "hello_world",
          "arguments" => "{\"key\": 1}"
        })

      # incomplete to start with
      assert call.status == :incomplete
      # arguments are still a string
      assert call.arguments == "{\"key\": 1}"

      {:ok, updated} = ToolCall.complete(call)
      assert updated.status == :complete
      assert updated.arguments == %{"key" => 1}
    end

    test "adds errors when argument json is invalid" do
      call =
        ToolCall.new!(%{
          "status" => :incomplete,
          "type" => "function",
          "index" => 0,
          "call_id" => "call_asdf",
          "name" => "hello_world",
          "arguments" => "invalid"
        })

      {:error, changeset} = ToolCall.complete(call)
      assert {"invalid json", _} = changeset.errors[:arguments]
    end
  end

  describe "merge/2" do
    test "takes new part when nothing existing found" do
      received = %ToolCall{
        status: :incomplete,
        type: :function,
        call_id: nil,
        name: "get_weather",
        arguments: nil,
        index: 0
      }

      result = ToolCall.merge(nil, received)
      assert result == received
    end

    test "does not duplicate incomplete call" do
      received = %ToolCall{
        status: :incomplete,
        type: :function,
        call_id: nil,
        name: "get_weather",
        arguments: nil,
        index: 0
      }

      result = ToolCall.merge(received, received)
      assert result == received
    end

    test "updates tool name" do
      call_1 = %ToolCall{
        status: :incomplete,
        type: :function,
        call_id: nil,
        name: "get_weat",
        arguments: nil,
        index: 0
      }

      call_2 = %ToolCall{
        status: :incomplete,
        type: :function,
        call_id: nil,
        name: "her",
        arguments: nil,
        index: 0
      }

      result = ToolCall.merge(call_1, call_2)
      assert result.name == "get_weather"
      assert result.type == :function
      assert result.status == :incomplete
    end

    test "updates status to complete and tool id" do
      call_1 = %ToolCall{
        status: :incomplete,
        type: :function,
        call_id: nil,
        name: "get_weather",
        arguments: nil,
        index: 0
      }

      call_2 = %ToolCall{
        status: :complete,
        type: :function,
        call_id: "call_123",
        name: nil,
        arguments: nil,
        index: 0
      }

      result = ToolCall.merge(call_1, call_2)
      assert result.status == :complete
      assert result.call_id == "call_123"
    end

    test "appends arguments" do
      call_1 = %ToolCall{
        status: :incomplete,
        type: :function,
        call_id: "call_123",
        name: "get_weather",
        arguments: nil,
        index: 1
      }

      call_2 = %ToolCall{
        status: :incomplete,
        type: :function,
        call_id: nil,
        name: nil,
        arguments: "{\"ci",
        index: 1
      }

      call_3 = %ToolCall{
        status: :incomplete,
        type: :function,
        call_id: nil,
        name: nil,
        arguments: "ty\": \"Portland\", \"state\": \"OR\"}",
        index: 1
      }

      result =
        call_1
        |> ToolCall.merge(call_2)
        |> ToolCall.merge(call_3)

      assert result.status == :incomplete
      assert result.arguments == "{\"city\": \"Portland\", \"state\": \"OR\"}"
    end

    test "does not unset call_id, function name or arguments" do
      call_1 = %ToolCall{
        status: :complete,
        type: :function,
        call_id: "call_123",
        name: "get_weather",
        arguments: "{\"city\": \"Portland\", \"state\": \"OR\"}",
        index: 1
      }

      call_2 = %ToolCall{
        status: :incomplete,
        type: :function,
        call_id: nil,
        name: nil,
        arguments: nil,
        index: 1
      }

      result = ToolCall.merge(call_1, call_2)
      assert result.status == :complete
      assert result.arguments == "{\"city\": \"Portland\", \"state\": \"OR\"}"
    end

    test "raises exception if merging different indexes together" do
      # the index indicates which, of several, tool calls it refers to. It is
      # invalid to merge tool call parts from different indexes.
      call_1 = %ToolCall{
        status: :incomplete,
        type: :function,
        name: "get_weat",
        index: 0
      }

      call_2 = %ToolCall{
        status: :incomplete,
        type: :function,
        name: "her",
        index: 2
      }

      assert_raise LangChainError, "Can only merge tool calls with the same index", fn ->
        ToolCall.merge(call_1, call_2)
      end
    end
  end
end
