defmodule LangChain.Agents.Middleware.PatchToolCallsTest do
  use LangChain.BaseCase, async: true

  alias LangChain.Agents.Middleware.PatchToolCalls
  alias LangChain.Agents.State
  alias LangChain.Message
  alias LangChain.Message.ToolCall
  alias LangChain.Message.ToolResult

  describe "before_model/2" do
    test "returns state unchanged when no messages" do
      state = State.new!(%{messages: []})
      assert {:ok, ^state} = PatchToolCalls.before_model(state, nil)
    end

    test "returns state unchanged when messages is nil" do
      state = State.new!(%{})
      assert {:ok, ^state} = PatchToolCalls.before_model(state, nil)
    end

    test "returns state unchanged when no tool calls exist" do
      messages = [
        Message.new_system!("You are helpful"),
        Message.new_user!("Hello"),
        Message.new_assistant!("Hi there!")
      ]

      state = State.new!(%{messages: messages})
      assert {:ok, ^state} = PatchToolCalls.before_model(state, nil)
    end

    test "returns state unchanged when all tool calls have responses" do
      tool_call =
        ToolCall.new!(%{
          call_id: "123",
          name: "search",
          arguments: %{"q" => "test"},
          status: :complete
        })

      tool_result =
        ToolResult.new!(%{
          tool_call_id: "123",
          name: "search",
          content: "Search results here"
        })

      messages = [
        Message.new_system!("You are helpful"),
        Message.new_user!("Search for test"),
        Message.new_assistant!(%{tool_calls: [tool_call]}),
        Message.new_tool_result!(%{tool_results: [tool_result]}),
        Message.new_user!("Thanks!")
      ]

      state = State.new!(%{messages: messages})
      assert {:ok, ^state} = PatchToolCalls.before_model(state, nil)
    end

    test "patches single dangling tool call" do
      tool_call =
        ToolCall.new!(%{
          call_id: "123",
          name: "get_events_for_days",
          arguments: %{"days" => 3},
          status: :complete
        })

      messages = [
        Message.new_system!("You are a helpful assistant."),
        Message.new_user!("Hello, how are you?"),
        Message.new_assistant!(%{
          content: "I'm doing well, thank you!",
          tool_calls: [tool_call]
        }),
        Message.new_user!("What is the weather in Tokyo?")
      ]

      state = State.new!(%{messages: messages})
      assert {:ok, %State{messages: patched_messages}} = PatchToolCalls.before_model(state, nil)

      # Should have patched message inserted after assistant (5 messages total)
      assert [
               %Message{role: :system},
               %Message{role: :user},
               %Message{role: :assistant},
               %Message{role: :tool, tool_results: [result]},
               %Message{role: :user}
             ] = patched_messages

      assert %ToolResult{tool_call_id: "123", name: "get_events_for_days"} = result

      # Content is converted to ContentParts
      assert [%Message.ContentPart{content: content}] = result.content
      assert content =~ "Tool call get_events_for_days with id 123 was cancelled"
    end

    test "patches multiple dangling tool calls in single message" do
      tool_call1 =
        ToolCall.new!(%{
          call_id: "123",
          name: "search",
          arguments: %{"q" => "test"},
          status: :complete
        })

      tool_call2 =
        ToolCall.new!(%{
          call_id: "456",
          name: "calculator",
          arguments: %{"expr" => "2+2"},
          status: :complete
        })

      messages = [
        Message.new_system!("You are helpful"),
        Message.new_assistant!(%{tool_calls: [tool_call1, tool_call2]}),
        Message.new_user!("Never mind")
      ]

      state = State.new!(%{messages: messages})
      assert {:ok, %State{messages: patched_messages}} = PatchToolCalls.before_model(state, nil)

      # Should have both patches inserted after assistant message (5 messages total)
      assert [
               %Message{role: :system},
               %Message{role: :assistant},
               %Message{role: :tool, tool_results: [result1]},
               %Message{role: :tool, tool_results: [result2]},
               %Message{role: :user}
             ] = patched_messages

      assert result1.tool_call_id == "123"
      assert result1.name == "search"

      assert result2.tool_call_id == "456"
      assert result2.name == "calculator"
    end

    test "patches multiple dangling tool calls across different messages" do
      tool_call1 =
        ToolCall.new!(%{
          call_id: "123",
          name: "get_events_for_days",
          arguments: %{},
          status: :complete
        })

      tool_call2 =
        ToolCall.new!(%{
          call_id: "456",
          name: "get_events_for_days",
          arguments: %{},
          status: :complete
        })

      messages = [
        Message.new_system!("You are a helpful assistant."),
        Message.new_user!("Hello, how are you?"),
        Message.new_assistant!(%{
          content: "I'm doing well, thank you!",
          tool_calls: [tool_call1]
        }),
        Message.new_user!("What is the weather in Tokyo?"),
        Message.new_assistant!(%{
          content: "Let me check that.",
          tool_calls: [tool_call2]
        }),
        Message.new_user!("Actually, never mind")
      ]

      state = State.new!(%{messages: messages})
      assert {:ok, %State{messages: patched_messages}} = PatchToolCalls.before_model(state, nil)

      # Should have patches inserted after each assistant message (8 messages total)
      assert [
               %Message{role: :system},
               %Message{role: :user},
               %Message{role: :assistant},
               %Message{role: :tool, tool_results: [result1]},
               %Message{role: :user},
               %Message{role: :assistant},
               %Message{role: :tool, tool_results: [result2]},
               %Message{role: :user}
             ] = patched_messages

      assert result1.tool_call_id == "123"
      assert result2.tool_call_id == "456"
    end

    test "patches only dangling tool calls when some have responses" do
      tool_call1 =
        ToolCall.new!(%{
          call_id: "123",
          name: "search",
          arguments: %{},
          status: :complete
        })

      tool_call2 =
        ToolCall.new!(%{
          call_id: "456",
          name: "calculator",
          arguments: %{},
          status: :complete
        })

      tool_result1 =
        ToolResult.new!(%{
          tool_call_id: "123",
          name: "search",
          content: "Results"
        })

      messages = [
        Message.new_system!("You are helpful"),
        Message.new_assistant!(%{tool_calls: [tool_call1, tool_call2]}),
        Message.new_tool_result!(%{tool_results: [tool_result1]}),
        Message.new_user!("What about the calculator?")
      ]

      state = State.new!(%{messages: messages})
      assert {:ok, %State{messages: patched_messages}} = PatchToolCalls.before_model(state, nil)

      # Should have patch for tool_call2 inserted after assistant message (5 messages total)
      assert [
               %Message{role: :system},
               %Message{role: :assistant},
               %Message{role: :tool, tool_results: [patch_result]},
               %Message{role: :tool, tool_results: [^tool_result1]},
               %Message{role: :user}
             ] = patched_messages

      assert patch_result.tool_call_id == "456"
      assert patch_result.name == "calculator"
    end

    test "handles tool results with multiple results in single message" do
      tool_call1 =
        ToolCall.new!(%{
          call_id: "123",
          name: "search",
          arguments: %{},
          status: :complete
        })

      tool_call2 =
        ToolCall.new!(%{
          call_id: "456",
          name: "calculator",
          arguments: %{},
          status: :complete
        })

      tool_result1 =
        ToolResult.new!(%{
          tool_call_id: "123",
          name: "search",
          content: "Results"
        })

      tool_result2 =
        ToolResult.new!(%{
          tool_call_id: "456",
          name: "calculator",
          content: "42"
        })

      messages = [
        Message.new_system!("You are helpful"),
        Message.new_assistant!(%{tool_calls: [tool_call1, tool_call2]}),
        Message.new_tool_result!(%{tool_results: [tool_result1, tool_result2]}),
        Message.new_user!("Thanks")
      ]

      state = State.new!(%{messages: messages})
      assert {:ok, ^state} = PatchToolCalls.before_model(state, nil)
    end

    test "only searches forward for tool results" do
      tool_call =
        ToolCall.new!(%{
          call_id: "123",
          name: "search",
          arguments: %{},
          status: :complete
        })

      # Tool result appears BEFORE the tool call (shouldn't happen but testing the logic)
      tool_result =
        ToolResult.new!(%{
          tool_call_id: "123",
          name: "search",
          content: "Results"
        })

      messages = [
        Message.new_system!("You are helpful"),
        Message.new_tool_result!(%{tool_results: [tool_result]}),
        Message.new_assistant!(%{tool_calls: [tool_call]}),
        Message.new_user!("Thanks")
      ]

      state = State.new!(%{messages: messages})
      assert {:ok, %State{messages: patched_messages}} = PatchToolCalls.before_model(state, nil)

      # Should patch because forward search doesn't find the result
      assert [
               %Message{role: :system},
               %Message{role: :tool},
               %Message{role: :assistant},
               %Message{role: :tool},
               %Message{role: :user}
             ] = patched_messages
    end

    test "creates proper cancellation message format" do
      tool_call =
        ToolCall.new!(%{
          call_id: "call_abc123",
          name: "my_awesome_tool",
          arguments: %{"param" => "value"},
          status: :complete
        })

      messages = [
        Message.new_assistant!(%{tool_calls: [tool_call]}),
        Message.new_user!("Cancel that")
      ]

      state = State.new!(%{messages: messages})
      assert {:ok, %State{messages: patched_messages}} = PatchToolCalls.before_model(state, nil)

      assert [
               %Message{role: :assistant},
               %Message{role: :tool, tool_results: [result]},
               %Message{role: :user}
             ] = patched_messages

      assert result.tool_call_id == "call_abc123"
      assert result.name == "my_awesome_tool"
      assert result.type == :function

      # Content is converted to ContentParts
      assert [%Message.ContentPart{content: content}] = result.content

      assert content ==
               "Tool call my_awesome_tool with id call_abc123 was cancelled - " <>
                 "another message came in before it could be completed."
    end
  end

  describe "patch_dangling_tool_calls/1" do
    test "returns empty list for empty input" do
      assert [] = PatchToolCalls.patch_dangling_tool_calls([])
    end

    test "returns original list when no tool calls" do
      messages = [
        Message.new_user!("Hello"),
        Message.new_assistant!("Hi")
      ]

      assert ^messages = PatchToolCalls.patch_dangling_tool_calls(messages)
    end

    test "returns original list when all tool calls complete" do
      tool_call =
        ToolCall.new!(%{
          call_id: "123",
          name: "search",
          arguments: %{},
          status: :complete
        })

      tool_result =
        ToolResult.new!(%{
          tool_call_id: "123",
          name: "search",
          content: "Results"
        })

      messages = [
        Message.new_assistant!(%{tool_calls: [tool_call]}),
        Message.new_tool_result!(%{tool_results: [tool_result]})
      ]

      assert ^messages = PatchToolCalls.patch_dangling_tool_calls(messages)
    end

    test "patches dangling tool call" do
      tool_call =
        ToolCall.new!(%{
          call_id: "123",
          name: "search",
          arguments: %{},
          status: :complete
        })

      messages = [
        Message.new_assistant!(%{tool_calls: [tool_call]}),
        Message.new_user!("Never mind")
      ]

      patched = PatchToolCalls.patch_dangling_tool_calls(messages)

      # Verify patch is inserted in correct position
      assert [
               %Message{role: :assistant},
               %Message{role: :tool},
               %Message{role: :user}
             ] = patched
    end

    test "handles non-list input gracefully" do
      assert [] = PatchToolCalls.patch_dangling_tool_calls(nil)
      assert [] = PatchToolCalls.patch_dangling_tool_calls("not a list")
      assert [] = PatchToolCalls.patch_dangling_tool_calls(%{})
    end
  end

  describe "edge cases" do
    test "handles assistant message with empty tool_calls list" do
      messages = [
        Message.new_assistant!(%{content: "Hello", tool_calls: []})
      ]

      state = State.new!(%{messages: messages})
      assert {:ok, ^state} = PatchToolCalls.before_model(state, nil)
    end

    test "handles assistant message with nil tool_calls" do
      message = %Message{
        role: :assistant,
        content: [Message.ContentPart.text!("Hello")],
        status: :complete,
        tool_calls: nil
      }

      messages = [message]
      state = State.new!(%{messages: messages})
      assert {:ok, ^state} = PatchToolCalls.before_model(state, nil)
    end

    test "handles tool message with empty tool_results" do
      tool_call =
        ToolCall.new!(%{
          call_id: "123",
          name: "search",
          arguments: %{},
          status: :complete
        })

      messages = [
        Message.new_assistant!(%{tool_calls: [tool_call]}),
        Message.new_tool_result!(%{tool_results: []}),
        Message.new_user!("Hi")
      ]

      state = State.new!(%{messages: messages})
      assert {:ok, %State{messages: patched_messages}} = PatchToolCalls.before_model(state, nil)

      # Should still patch because tool_results is empty
      assert [
               %Message{role: :assistant},
               %Message{role: :tool},
               %Message{role: :tool},
               %Message{role: :user}
             ] = patched_messages
    end

    test "handles tool message with nil tool_results" do
      tool_call =
        ToolCall.new!(%{
          call_id: "123",
          name: "search",
          arguments: %{},
          status: :complete
        })

      message_with_nil_results = %Message{
        role: :tool,
        tool_results: nil,
        status: :complete
      }

      messages = [
        Message.new_assistant!(%{tool_calls: [tool_call]}),
        message_with_nil_results,
        Message.new_user!("Hi")
      ]

      state = State.new!(%{messages: messages})
      assert {:ok, %State{messages: patched_messages}} = PatchToolCalls.before_model(state, nil)

      # Should still patch because tool_results is nil
      assert [
               %Message{role: :assistant},
               %Message{role: :tool},
               %Message{role: :tool},
               %Message{role: :user}
             ] = patched_messages
    end

    test "preserves message order and content" do
      tool_call =
        ToolCall.new!(%{
          call_id: "123",
          name: "search",
          arguments: %{"q" => "test"},
          status: :complete
        })

      user_msg1 = Message.new_user!("First message")
      assistant_msg = Message.new_assistant!(%{content: "Let me search", tool_calls: [tool_call]})
      user_msg2 = Message.new_user!("Second message")

      messages = [user_msg1, assistant_msg, user_msg2]

      state = State.new!(%{messages: messages})
      assert {:ok, %State{messages: patched_messages}} = PatchToolCalls.before_model(state, nil)

      # Verify original messages are preserved with patch inserted
      assert [^user_msg1, ^assistant_msg, %Message{role: :tool}, ^user_msg2] = patched_messages
    end

    test "handles complex conversation with multiple interruptions" do
      # Simulate a complex scenario with multiple tool calls, some completed, some not
      tc1 = ToolCall.new!(%{call_id: "1", name: "tool1", arguments: %{}, status: :complete})
      tc2 = ToolCall.new!(%{call_id: "2", name: "tool2", arguments: %{}, status: :complete})
      tc3 = ToolCall.new!(%{call_id: "3", name: "tool3", arguments: %{}, status: :complete})
      tc4 = ToolCall.new!(%{call_id: "4", name: "tool4", arguments: %{}, status: :complete})

      tr1 = ToolResult.new!(%{tool_call_id: "1", name: "tool1", content: "Result 1"})
      tr3 = ToolResult.new!(%{tool_call_id: "3", name: "tool3", content: "Result 3"})

      messages = [
        Message.new_system!("You are helpful"),
        Message.new_user!("Do task 1 and 2"),
        Message.new_assistant!(%{tool_calls: [tc1, tc2]}),
        Message.new_tool_result!(%{tool_results: [tr1]}),
        # tc2 is dangling
        Message.new_user!("Also do task 3 and 4"),
        Message.new_assistant!(%{tool_calls: [tc3, tc4]}),
        Message.new_tool_result!(%{tool_results: [tr3]}),
        # tc4 is dangling
        Message.new_user!("Thanks")
      ]

      state = State.new!(%{messages: messages})
      assert {:ok, %State{messages: patched_messages}} = PatchToolCalls.before_model(state, nil)

      # Find synthetic patches (those with "cancelled" in content)
      synthetic_patches =
        Enum.filter(patched_messages, fn msg ->
          msg.role == :tool and
            Enum.any?(msg.tool_results || [], fn r ->
              case r.content do
                [%Message.ContentPart{content: content}] -> content =~ "was cancelled"
                _ -> false
              end
            end)
        end)

      # Should have 2 synthetic patches
      assert [patch1, patch2] = synthetic_patches

      # Verify the patches are for tc2 and tc4
      [result1] = patch1.tool_results
      [result2] = patch2.tool_results
      patch_ids = Enum.sort([result1.tool_call_id, result2.tool_call_id])

      assert patch_ids == ["2", "4"]
    end
  end
end
