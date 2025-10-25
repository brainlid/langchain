defmodule LangChain.DeepAgents.Middleware.PatchToolCalls do
  @moduledoc """
  Middleware that identifies and resolves "dangling tool calls" in the message history.

  A "dangling tool call" occurs when an AI message contains tool calls that lack
  corresponding tool result messages in the conversation history. This creates an
  incomplete request-response cycle that can confuse the agent and cause LLM API
  errors.

  ## Problem Scenarios

  Dangling tool calls commonly occur due to:

  1. **User Interruption**: User sends a new message before tool execution completes
  2. **Agent Resets**: Agent state is restored with incomplete tool calls
  3. **Error Handling**: Tool execution fails without generating a tool result
  4. **State Corruption**: Incomplete state updates or message history corruption

  ## Solution

  This middleware runs in the `before_model` phase and:

  1. Scans message history for assistant messages with tool calls
  2. For each tool call, searches forward for a corresponding tool result message
  3. Creates synthetic tool result messages for any dangling tool calls
  4. Returns updated state with patched message history

  ## Position in Middleware Stack

  This middleware should run relatively late in the before_model phase, after
  middleware that might generate or modify messages but before any middleware
  that expects complete tool call sequences (like HumanInTheLoop).

  ## Usage

      # Add to agent with default middleware
      {:ok, agent} = Agent.new(
        model: model,
        middleware: [PatchToolCalls]
      )

      # Or with custom middleware stack
      {:ok, agent} = Agent.new(
        model: model,
        replace_default_middleware: true,
        middleware: [
          TodoList,
          Filesystem,
          PatchToolCalls,  # Position before HITL
          HumanInTheLoop,
          MyMiddleware
        ]
      )

  ## Example

      # Before patching:
      messages = [
        %Message{role: :system, content: "You are helpful"},
        %Message{role: :assistant, tool_calls: [
          %ToolCall{call_id: "123", name: "search", arguments: %{q: "test"}}
        ]},
        %Message{role: :user, content: "Never mind"}  # User interrupted!
      ]

      # After patching:
      messages = [
        %Message{role: :system, content: "You are helpful"},
        %Message{role: :assistant, tool_calls: [
          %ToolCall{call_id: "123", name: "search", arguments: %{q: "test"}}
        ]},
        %Message{role: :tool, tool_results: [
          %ToolResult{
            tool_call_id: "123",
            name: "search",
            content: "Tool call search with id 123 was cancelled - another message came in before it could be completed."
          }
        ]},
        %Message{role: :user, content: "Never mind"}
      ]
  """

  @behaviour LangChain.DeepAgents.Middleware

  alias LangChain.DeepAgents.State
  alias LangChain.Message
  alias LangChain.Message.ToolCall
  alias LangChain.Message.ToolResult

  @impl true
  def before_model(%State{messages: messages} = state, _config)
      when is_list(messages) and messages != [] do
    case patch_dangling_tool_calls(messages) do
      ^messages ->
        # No changes needed
        {:ok, state}

      patched_messages ->
        # Messages were patched, return updated state
        {:ok, %{state | messages: patched_messages}}
    end
  end

  def before_model(%State{} = state, _config) do
    # No messages or messages is nil
    {:ok, state}
  end

  @doc """
  Scan messages for dangling tool calls and create synthetic tool results.

  A tool call is "dangling" if there is no corresponding tool result message
  with a matching tool_call_id in any subsequent message.

  Returns the patched message list. If no patches are needed, returns the
  original list unchanged.
  """
  def patch_dangling_tool_calls(messages) when is_list(messages) and messages == [] do
    messages
  end

  def patch_dangling_tool_calls(messages) when is_list(messages) do
    {patched, _index} =
      Enum.reduce(messages, {[], 0}, fn msg, {patched_messages, index} ->
        # Add the current message
        patched_with_msg = patched_messages ++ [msg]

        # Check if this is an assistant message with tool calls
        case msg do
          %Message{role: :assistant, tool_calls: tool_calls}
          when is_list(tool_calls) and tool_calls != [] ->
            # For each tool call, check if there's a corresponding result
            patches =
              Enum.flat_map(tool_calls, fn tool_call ->
                if has_tool_result?(messages, index, tool_call.call_id) do
                  # Tool result exists, no patch needed
                  []
                else
                  # Create synthetic tool result for dangling call
                  [create_cancellation_message(tool_call)]
                end
              end)

            {patched_with_msg ++ patches, index + 1}

          _other ->
            {patched_with_msg, index + 1}
        end
      end)

    patched
  end

  def patch_dangling_tool_calls(_messages) do
    # Not a list, return empty list
    []
  end

  # Check if a tool result exists for the given tool call ID
  # Only searches in messages AFTER the current index (forward search)
  defp has_tool_result?(messages, current_index, call_id) do
    messages
    |> Enum.drop(current_index + 1)
    |> Enum.any?(fn
      %Message{role: :tool, tool_results: tool_results} when is_list(tool_results) ->
        Enum.any?(tool_results, fn
          %ToolResult{tool_call_id: ^call_id} -> true
          _other -> false
        end)

      _other ->
        false
    end)
  end

  # Create a synthetic tool result message indicating the tool was cancelled
  defp create_cancellation_message(%ToolCall{call_id: call_id, name: name}) do
    content =
      "Tool call #{name} with id #{call_id} was cancelled - " <>
        "another message came in before it could be completed."

    tool_result =
      ToolResult.new!(%{
        tool_call_id: call_id,
        name: name,
        content: content,
        type: :function
      })

    Message.new_tool_result!(%{tool_results: [tool_result]})
  end
end
