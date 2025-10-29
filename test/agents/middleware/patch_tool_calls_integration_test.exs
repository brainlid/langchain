defmodule LangChain.Agents.Middleware.PatchToolCallsIntegrationTest do
  use ExUnit.Case, async: true

  alias LangChain.Agents.Agent
  alias LangChain.Agents.State
  alias LangChain.Message
  alias LangChain.Message.ToolCall
  alias LangChain.Message.ToolResult
  alias LangChain.ChatModels.ChatAnthropic

  describe "integration with Agent" do
    setup do
      # Create a simple model for testing (won't actually call it)
      model = ChatAnthropic.new!(%{model: "claude-3-5-sonnet-20241022", stream: false})

      {:ok, agent} =
        Agent.new(
          model: model,
          system_prompt: "You are a helpful assistant."
        )

      {:ok, agent: agent}
    end

    test "middleware is included in default middleware stack", %{agent: agent} do
      # Verify PatchToolCalls is in the middleware list
      middleware_modules =
        agent.middleware
        |> Enum.map(fn {module, _config} -> module end)

      assert LangChain.Agents.Middleware.PatchToolCalls in middleware_modules
    end

    test "patches dangling tool calls during agent execution", %{agent: agent} do
      # Create a state with dangling tool call
      tool_call =
        ToolCall.new!(%{
          call_id: "123",
          name: "search",
          arguments: %{"q" => "elixir"},
          status: :complete
        })

      initial_messages = [
        Message.new_user!("Search for elixir"),
        Message.new_assistant!(%{
          content: "I'll search for that.",
          tool_calls: [tool_call]
        }),
        Message.new_user!("Actually, never mind that")
      ]

      state = State.new!(%{messages: initial_messages})

      # When we execute the agent, the before_model hooks will run
      # We can't actually call the LLM, but we can verify the state
      # transformation by directly calling the middleware hooks
      middleware_list = agent.middleware

      # Apply before_model hooks manually
      {:ok, processed_state} =
        Enum.reduce_while(middleware_list, {:ok, state}, fn mw, {:ok, current_state} ->
          case LangChain.Agents.Middleware.apply_before_model(current_state, mw) do
            {:ok, updated_state} -> {:cont, {:ok, updated_state}}
            {:error, reason} -> {:halt, {:error, reason}}
          end
        end)

      # The dangling tool call should now have a synthetic response
      # Pattern match to verify structure (may have system messages from other middleware)
      tool_messages = Enum.filter(processed_state.messages, &(&1.role == :tool))

      assert [%Message{tool_results: [%ToolResult{tool_call_id: "123", name: "search"}]}] =
               tool_messages
    end

    test "does not modify messages when all tool calls are complete", %{agent: agent} do
      tool_call =
        ToolCall.new!(%{
          call_id: "123",
          name: "search",
          arguments: %{"q" => "elixir"},
          status: :complete
        })

      tool_result =
        ToolResult.new!(%{
          tool_call_id: "123",
          name: "search",
          content: "Found results about Elixir"
        })

      initial_messages = [
        Message.new_user!("Search for elixir"),
        Message.new_assistant!(%{tool_calls: [tool_call]}),
        Message.new_tool_result!(%{tool_results: [tool_result]}),
        Message.new_user!("Thanks!")
      ]

      state = State.new!(%{messages: initial_messages})
      middleware_list = agent.middleware

      # Apply before_model hooks
      {:ok, processed_state} =
        Enum.reduce_while(middleware_list, {:ok, state}, fn mw, {:ok, current_state} ->
          case LangChain.Agents.Middleware.apply_before_model(current_state, mw) do
            {:ok, updated_state} -> {:cont, {:ok, updated_state}}
            {:error, reason} -> {:halt, {:error, reason}}
          end
        end)

      # Messages should be unchanged (length remains 4)
      assert processed_state.messages == state.messages
    end

    test "works with custom middleware stack", _context do
      model = ChatAnthropic.new!(%{model: "claude-3-5-sonnet-20241022", stream: false})

      {:ok, agent} =
        Agent.new(
          model: model,
          replace_default_middleware: true,
          middleware: [
            LangChain.Agents.Middleware.PatchToolCalls
          ]
        )

      tool_call =
        ToolCall.new!(%{
          call_id: "456",
          name: "calculator",
          arguments: %{"expr" => "2+2"},
          status: :complete
        })

      messages = [
        Message.new_assistant!(%{tool_calls: [tool_call]}),
        Message.new_user!("Stop")
      ]

      state = State.new!(%{messages: messages})

      # Apply before_model hooks
      {:ok, processed_state} =
        Enum.reduce_while(agent.middleware, {:ok, state}, fn mw, {:ok, current_state} ->
          case LangChain.Agents.Middleware.apply_before_model(current_state, mw) do
            {:ok, updated_state} -> {:cont, {:ok, updated_state}}
            {:error, reason} -> {:halt, {:error, reason}}
          end
        end)

      # Should have patched the dangling tool call
      assert [
               %Message{role: :assistant},
               %Message{role: :tool},
               %Message{role: :user}
             ] = processed_state.messages
    end

    test "handles multiple middleware in sequence", %{agent: agent} do
      # This test verifies that PatchToolCalls works correctly
      # when other middleware has already modified the state

      tool_call =
        ToolCall.new!(%{
          call_id: "789",
          name: "file_write",
          arguments: %{"path" => "test.txt", "content" => "hello"},
          status: :complete
        })

      initial_messages = [
        Message.new_user!("Write a file"),
        Message.new_assistant!(%{tool_calls: [tool_call]}),
        Message.new_user!("Cancel that")
      ]

      state = State.new!(%{messages: initial_messages})

      # Apply all middleware in order (TodoList, Filesystem, PatchToolCalls)
      {:ok, processed_state} =
        Enum.reduce_while(agent.middleware, {:ok, state}, fn mw, {:ok, current_state} ->
          case LangChain.Agents.Middleware.apply_before_model(current_state, mw) do
            {:ok, updated_state} -> {:cont, {:ok, updated_state}}
            {:error, reason} -> {:halt, {:error, reason}}
          end
        end)

      # The dangling tool call should be patched
      # (Other middleware might add system messages but won't affect this test)
      # Find our synthetic patch
      synthetic_patch =
        Enum.find(processed_state.messages, fn msg ->
          msg.role == :tool and
            Enum.any?(msg.tool_results || [], fn r -> r.tool_call_id == "789" end)
        end)

      assert %Message{role: :tool, tool_results: [result]} = synthetic_patch
      assert result.tool_call_id == "789"
      assert result.name == "file_write"
    end

    test "preserves message metadata during patching", %{agent: agent} do
      tool_call =
        ToolCall.new!(%{
          call_id: "meta_test",
          name: "test_tool",
          arguments: %{},
          status: :complete
        })

      # Create messages with metadata
      user_msg = Message.new_user!("Test")
      user_msg = %{user_msg | metadata: %{source: "test", timestamp: 12345}}

      assistant_msg = Message.new_assistant!(%{tool_calls: [tool_call]})
      assistant_msg = %{assistant_msg | metadata: %{model: "test-model"}}

      messages = [user_msg, assistant_msg, Message.new_user!("Stop")]
      state = State.new!(%{messages: messages})

      # Apply middleware
      {:ok, processed_state} =
        Enum.reduce_while(agent.middleware, {:ok, state}, fn mw, {:ok, current_state} ->
          case LangChain.Agents.Middleware.apply_before_model(current_state, mw) do
            {:ok, updated_state} -> {:cont, {:ok, updated_state}}
            {:error, reason} -> {:halt, {:error, reason}}
          end
        end)

      # Original messages should preserve metadata
      # The patched messages list contains at least the originals
      patched_user =
        Enum.find(processed_state.messages, fn m -> m.metadata[:source] == "test" end)

      assert patched_user != nil
      assert patched_user.metadata[:timestamp] == 12345

      patched_assistant =
        Enum.find(processed_state.messages, fn m -> m.metadata[:model] == "test-model" end)

      assert patched_assistant != nil
    end
  end
end
