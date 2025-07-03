defmodule LangChain.Message.ToolResultTest do
  use ExUnit.Case
  doctest LangChain.Message.ToolResult
  alias LangChain.Message.ToolResult
  alias LangChain.LangChainError
  alias LangChain.Message.ContentPart

  describe "new/1" do
    test "works with minimal attrs" do
      assert {:ok, %ToolResult{} = msg} =
               ToolResult.new(%{tool_call_id: "call_123", content: "OK"})

      assert msg.type == :function
      assert msg.tool_call_id == "call_123"
      assert msg.content == [ContentPart.text!("OK")]
      assert msg.name == nil
      assert msg.display_text == nil
      assert msg.is_error == false
    end

    test "accepts valid input" do
      {:ok, %ToolResult{} = msg} =
        ToolResult.new(%{
          type: :function,
          tool_call_id: "call_123asdf",
          name: "hello_world",
          content: "Hello World!",
          display_text: "Ran hello_world function"
        })

      assert msg.type == :function
      assert msg.name == "hello_world"
      assert msg.content == [ContentPart.text!("Hello World!")]
      assert msg.tool_call_id == "call_123asdf"
      assert msg.display_text == "Ran hello_world function"
    end

    test "accepts multiple content parts" do
      content_list = [
        ContentPart.text!("Hello World!"),
        ContentPart.text!("Hello World! 2", cache_control: true)
      ]

      {:ok, %ToolResult{} = msg} =
        ToolResult.new(%{
          type: :function,
          tool_call_id: "call_123asdf",
          name: "hello_world",
          content: content_list,
          display_text: "Ran hello_world function"
        })

      assert msg.type == :function
      assert msg.name == "hello_world"
      assert msg.content == content_list
      assert msg.tool_call_id == "call_123asdf"
      assert msg.display_text == "Ran hello_world function"
    end

    test "flags when an error" do
      {:ok, %ToolResult{} = msg} =
        ToolResult.new(%{
          type: :function,
          tool_call_id: "call_123asdf",
          name: "hello_world",
          content: "The world was destroyed before we could say hello",
          display_text: "Failed to run hello_world",
          is_error: true
        })

      assert msg.type == :function
      assert msg.name == "hello_world"

      assert msg.content == [
               ContentPart.text!("The world was destroyed before we could say hello")
             ]

      assert msg.tool_call_id == "call_123asdf"
      assert msg.display_text == "Failed to run hello_world"
      assert msg.is_error == true
    end

    test "returns errors when invalid" do
      {:error, changeset} =
        ToolResult.new(%{tool_call_id: nil, content: nil})

      assert {"can't be blank", _} = changeset.errors[:tool_call_id]
      assert {"can't be blank", _} = changeset.errors[:content]
    end
  end

  describe "new!/1" do
    test "accepts valid input" do
      %ToolResult{} = result = ToolResult.new!(%{tool_call_id: "call_123", content: "SUCCESS"})
      assert result.tool_call_id == "call_123"
      assert result.content == [ContentPart.text!("SUCCESS")]
    end

    test "raises error when invalid" do
      assert_raise LangChainError, "tool_call_id: can't be blank; content: can't be blank", fn ->
        ToolResult.new!(%{tool_call_id: nil, content: nil})
      end
    end
  end
end
