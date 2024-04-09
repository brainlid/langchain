defmodule LangChain.Message.ToolCallTest do
  use ExUnit.Case
  doctest LangChain.Message.ToolCall
  alias LangChain.Message.ToolCall
  import LangChain.Fixtures
  alias LangChain.LangChainError

  describe "new/1" do
    test "works with minimal attrs" do
      assert {:ok, %ToolCall{} = msg} = ToolCall.new(%{})
      assert msg.status == :incomplete
      assert msg.type == :function
      assert msg.tool_id == nil
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
                 "tool_id" => "tool_asdf",
                 "name" => "hello_world",
                 "arguments" => "{\"key\": 1}"
               })

      assert msg.status == :incomplete
      assert msg.type == :function
      assert msg.name == "hello_world"
      assert msg.arguments == "{\"key\": 1}"
      assert msg.tool_id == "tool_asdf"
      assert msg.index == 1
    end

    test "parses arguments when complete" do
      assert {:ok, %ToolCall{} = msg} =
               ToolCall.new(%{
                 "status" => :complete,
                 "type" => "function",
                 "index" => 0,
                 "tool_id" => "tool_asdf",
                 "name" => "hello_world",
                 "arguments" => "{\"key\": 1}"
               })

      assert msg.status == :complete
      assert msg.type == :function
      assert msg.name == "hello_world"
      assert msg.arguments == %{"key" => 1}
      assert msg.tool_id == "tool_asdf"
      assert msg.index == 0
    end
  end

  describe "merge/2" do
    test "takes new part when nothing existing found"

    test "updates tool name"
    test "updates status to complete"
    test "appends arguments"
    test "updates tool_id"
  end
end
