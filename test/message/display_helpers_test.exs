defmodule LangChain.Message.DisplayHelpersTest do
  use LangChain.BaseCase, async: true

  alias LangChain.Message
  alias LangChain.Message.ContentPart
  alias LangChain.Message.ToolCall
  alias LangChain.Message.ToolResult
  alias LangChain.Message.DisplayHelpers

  describe "extract_display_items/1" do
    test "extracts simple text from assistant message" do
      message = Message.new_assistant!("Hello world")

      items = DisplayHelpers.extract_display_items(message)

      assert [
        %{
          type: :text,
          message_type: :assistant,
          content: %{"text" => "Hello world"}
        }
      ] = items
    end

    test "extracts thinking and text content parts" do
      message = Message.new_assistant!([
        ContentPart.thinking!("Let me think..."),
        ContentPart.text!("Here's my answer")
      ])

      items = DisplayHelpers.extract_display_items(message)

      assert [
        %{type: :thinking, content: %{"text" => "Let me think..."}},
        %{type: :text, content: %{"text" => "Here's my answer"}}
      ] = items
    end

    test "extracts multiple tool calls" do
      tool_call_1 = ToolCall.new!(%{call_id: "1", name: "search", arguments: %{"q" => "elixir"}})
      tool_call_2 = ToolCall.new!(%{call_id: "2", name: "weather", arguments: %{"city" => "NYC"}})

      message = Message.new_assistant!(%{tool_calls: [tool_call_1, tool_call_2]})

      items = DisplayHelpers.extract_display_items(message)

      assert [
        %{
          type: :tool_call,
          message_type: :assistant,
          content: %{
            "call_id" => "1",
            "name" => "search",
            "arguments" => %{"q" => "elixir"}
          }
        },
        %{
          type: :tool_call,
          message_type: :assistant,
          content: %{
            "call_id" => "2",
            "name" => "weather",
            "arguments" => %{"city" => "NYC"}
          }
        }
      ] = items
    end

    test "extracts multiple tool results" do
      result_1 = ToolResult.new!(%{tool_call_id: "1", name: "search", content: "Found...", is_error: false})
      result_2 = ToolResult.new!(%{tool_call_id: "2", name: "weather", content: "Sunny", is_error: false})

      message = Message.new_tool_result!(%{tool_results: [result_1, result_2]})

      items = DisplayHelpers.extract_display_items(message)

      assert [
        %{
          type: :tool_result,
          message_type: :tool,
          content: %{
            "tool_call_id" => "1",
            "name" => "search",
            "content" => "Found...",
            "is_error" => false
          }
        },
        %{
          type: :tool_result,
          message_type: :tool,
          content: %{
            "tool_call_id" => "2",
            "name" => "weather",
            "content" => "Sunny",
            "is_error" => false
          }
        }
      ] = items
    end

    test "extracts text content plus tool calls (mixed)" do
      message = Message.new_assistant!(%{
        content: [ContentPart.text!("Let me search for that")],
        tool_calls: [
          ToolCall.new!(%{call_id: "1", name: "search", arguments: %{"q" => "elixir"}})
        ]
      })

      items = DisplayHelpers.extract_display_items(message)

      assert [
        %{type: :text, content: %{"text" => "Let me search for that"}},
        %{type: :tool_call, content: %{"call_id" => "1", "name" => "search", "arguments" => %{"q" => "elixir"}}}
      ] = items
    end

    test "returns empty list for message with no displayable content" do
      message = Message.new_assistant!(%{content: nil, tool_calls: []})

      items = DisplayHelpers.extract_display_items(message)

      assert [] = items
    end

    test "filters out empty text content" do
      message = Message.new_assistant!(%{content: ""})

      items = DisplayHelpers.extract_display_items(message)

      assert [] = items
    end
  end
end
