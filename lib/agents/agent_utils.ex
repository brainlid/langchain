defmodule LangChain.Agents.AgentUtils do
  @moduledoc """
  Shared utilities for Agent and SubAgent HITL (Human-in-the-Loop) support.

  Provides common functions for:
  - Checking for HITL interrupts
  - Building full decisions lists
  - Extracting tool calls from chains

  These utilities provide consistent HITL behavior across Agent and SubAgent
  implementations.
  """

  alias LangChain.Chains.LLMChain
  alias LangChain.Message

  @doc """
  Check if a chain has pending tool calls that require human approval.

  ## Parameters
  - chain: LLMChain with potentially pending tool calls
  - interrupt_on: Map of tool_name => true/false/config

  ## Returns
  - `{:interrupt, interrupt_data}` - Some tools need approval
  - `:continue` - No tools need approval or no tool calls present

  ## Example
      case AgentUtils.check_for_hitl_interrupt(chain, %{"write_file" => true}) do
        {:interrupt, interrupt_data} ->
          # Pause and request human decisions
          {:interrupt, state, interrupt_data}

        :continue ->
          # Execute tools automatically
          execute_tools(chain)
      end
  """
  def check_for_hitl_interrupt(%LLMChain{} = chain, interrupt_on) when is_map(interrupt_on) do
    case chain.last_message do
      %Message{role: :assistant, tool_calls: tool_calls}
      when is_list(tool_calls) and length(tool_calls) > 0 ->
        # Filter tool calls that need approval
        hitl_tool_calls =
          Enum.filter(tool_calls, fn tc ->
            requires_approval?(tc.name, interrupt_on)
          end)

        if hitl_tool_calls == [] do
          :continue
        else
          # Build interrupt data
          action_requests =
            Enum.map(hitl_tool_calls, fn tc ->
              %{
                tool_call_id: tc.call_id,
                tool_name: tc.name,
                arguments: tc.arguments
              }
            end)

          hitl_tool_call_ids = Enum.map(hitl_tool_calls, & &1.call_id)

          interrupt_data = %{
            action_requests: action_requests,
            hitl_tool_call_ids: hitl_tool_call_ids
          }

          {:interrupt, interrupt_data}
        end

      _ ->
        # No tool calls in last message
        :continue
    end
  end

  @doc """
  Build a full decisions list for ALL tool calls in a message.

  Mixes human decisions (for HITL tools) with auto-approvals (for non-HITL tools).
  This is needed because LLMChain.execute_tool_calls_with_decisions expects a decision
  for EVERY tool call, not just the ones that needed approval.

  ## Parameters
  - all_tool_calls: All tool calls from assistant message (HITL + non-HITL)
  - hitl_tool_call_ids: List of tool_call_ids that needed human approval
  - human_decisions: List of decisions from human (same order as action_requests)
  - action_requests: List of action_requests (to map decisions to tool_call_ids)

  ## Returns
  - List of decisions matching all_tool_calls order

  ## Example
      all_tool_calls = [tc1, tc2, tc3]  # 3 total tool calls
      hitl_tool_call_ids = [tc1.call_id]  # Only tc1 needed approval
      action_requests = [%{tool_call_id: tc1.call_id, ...}]
      human_decisions = [%{type: :approve}]  # Human approved tc1

      full_decisions = build_full_decisions(all_tool_calls, hitl_tool_call_ids, human_decisions, action_requests)
      # => [%{type: :approve}, %{type: :approve}, %{type: :approve}]
      # First is human decision, others are auto-approved
  """
  def build_full_decisions(all_tool_calls, hitl_tool_call_ids, human_decisions, action_requests) do
    # Build decisions map indexed by tool_call_id
    decisions_by_id =
      action_requests
      |> Enum.zip(human_decisions)
      |> Map.new(fn {action_req, decision} ->
        {action_req.tool_call_id, decision}
      end)

    # Build full decisions list matching ALL tool calls
    Enum.map(all_tool_calls, fn tc ->
      if tc.call_id in hitl_tool_call_ids do
        # Use human decision for HITL tool
        Map.fetch!(decisions_by_id, tc.call_id)
      else
        # Auto-approve non-HITL tool
        %{type: :approve}
      end
    end)
  end

  @doc """
  Extract tool calls from the last assistant message in a chain.

  ## Parameters
  - chain: LLMChain

  ## Returns
  - List of tool calls or empty list

  ## Example
      tool_calls = AgentUtils.get_tool_calls_from_last_message(chain)
      # => [%ToolCall{call_id: "1", name: "write_file", ...}, ...]
  """
  def get_tool_calls_from_last_message(%LLMChain{} = chain) do
    case chain.last_message do
      %Message{role: :assistant, tool_calls: tool_calls} when is_list(tool_calls) ->
        tool_calls

      _ ->
        []
    end
  end

  # Private helpers

  defp requires_approval?(tool_name, interrupt_on) do
    case Map.get(interrupt_on, tool_name) do
      # Not in config = no approval needed
      nil -> false
      # Explicitly false = no approval
      false -> false
      # Explicitly true = requires approval
      true -> true
      # Config map = requires approval
      %{} -> true
    end
  end
end
