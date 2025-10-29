defmodule LangChain.Agents.Middleware.HumanInTheLoop do
  @moduledoc """
  Middleware that enables human oversight and intervention in agent workflows.

  HumanInTheLoop (HITL) allows pausing agent execution before executing sensitive
  or critical tool operations, providing humans with the ability to approve, edit,
  or reject tool calls.

  ## Configuration

  The middleware is configured with an `interrupt_on` map that specifies which tools
  should trigger human approval:

      # Simple boolean configuration
      interrupt_on = %{
        "write_file" => true,           # Enable with default decisions
        "delete_file" => true,
        "read_file" => false            # No interruption
      }

      # Advanced configuration with custom decisions
      interrupt_on = %{
        "write_file" => %{
          allowed_decisions: [:approve, :edit, :reject]
        },
        "delete_file" => %{
          allowed_decisions: [:approve, :reject]  # No edit option
        }
      }

  ### Decision Types

  - `:approve` - Execute the tool with original arguments
  - `:edit` - Execute the tool with modified arguments
  - `:reject` - Skip tool execution entirely

  ## Usage

      # Create agent with HITL middleware
      {:ok, agent} = Agent.new(
        model: model,
        interrupt_on: %{
          "write_file" => true,
          "delete_file" => true
        }
      )

      # Execute - will return interrupt if tool needs approval
      state = State.new!(%{messages: [...]})
      result = Agent.execute(agent, state)

      case result do
        {:ok, state} ->
          # Normal completion
          handle_response(state)

        {:interrupt, state, interrupt_data} ->
          # Human approval needed
          decisions = get_human_decisions(interrupt_data)
          {:ok, final_state} = Agent.resume(agent, state, decisions)
      end

  ## Interrupt Structure

  When a tool requires approval, execution returns an interrupt tuple with detailed
  information about the tools requiring approval.

  ### Structure

      {:interrupt, state, %{
        action_requests: [action_request, ...],
        review_configs: %{tool_name => config, ...}
      }}

  ### Action Requests

  Each action request contains complete information about the tool call:

      %{
        tool_call_id: "call_123",      # Unique identifier for this tool call
        tool_name: "write_file",       # Name of the tool being called
        arguments: %{...}              # Arguments that would be passed to the tool
      }

  ### Review Configs

  A map of tool names to their approval configuration. More efficient than the
  Python library's array-based approach, providing O(1) lookup:

      %{
        "write_file" => %{
          allowed_decisions: [:approve, :edit, :reject]
        },
        "delete_file" => %{
          allowed_decisions: [:approve, :reject]  # Edit not allowed
        }
      }

  ### Complete Example

      {:interrupt, state, %{
        action_requests: [
          %{
            tool_call_id: "call_123",
            tool_name: "write_file",
            arguments: %{"path" => "file.txt", "content" => "data"}
          },
          %{
            tool_call_id: "call_456",
            tool_name: "delete_file",
            arguments: %{"path" => "old.txt"}
          }
        ],
        review_configs: %{
          "write_file" => %{allowed_decisions: [:approve, :edit, :reject]},
          "delete_file" => %{allowed_decisions: [:approve, :reject]}
        }
      }}

  ## Resume Structure

  Resume execution by providing decisions that correspond to each action request in order.

  ### Decision Types

  - `:approve` - Execute the tool with its original arguments
  - `:edit` - Execute the tool with modified arguments (requires `:arguments` field)
  - `:reject` - Skip tool execution, provide rejection message to agent

  ### Decision Format

  Each decision is a map with a required `:type` field:

      # Approve decision
      %{type: :approve}

      # Edit decision (must include :arguments with the modified parameters)
      %{type: :edit, arguments: %{"path" => "modified.txt", "content" => "new data"}}

      # Reject decision
      %{type: :reject}

  ### Complete Resume Example

  The decisions list must match the order and count of action_requests:

      # Given interrupt with 3 action requests
      {:interrupt, state, %{action_requests: [req1, req2, req3], ...}}

      # Provide corresponding decisions
      decisions = [
        %{type: :approve},                                    # Approve req1
        %{type: :edit, arguments: %{"path" => "other.txt"}}, # Edit req2's arguments
        %{type: :reject}                                      # Reject req3
      ]

      # Resume execution
      {:ok, final_state} = Agent.resume(agent, state, decisions)

  ## Position in Middleware Stack

  HITL middleware should run late in the after_model phase, after PatchToolCalls
  but before any execution middleware. This ensures tool calls are complete and
  valid before requesting human approval.

  Default stack position:
  1. TodoListMiddleware
  2. FilesystemMiddleware
  3. PatchToolCallsMiddleware
  4. **HumanInTheLoopMiddleware** ← Position in stack
  5. Custom user middleware
  """

  @behaviour LangChain.Agents.Middleware

  alias LangChain.Agents.State
  alias LangChain.Message
  alias LangChain.Message.ToolCall
  alias LangChain.Message.ToolResult

  @type interrupt_config :: %{
          allowed_decisions: [atom()]
        }

  @type interrupt_on_config :: %{
          optional(String.t()) => boolean() | interrupt_config()
        }

  @type action_request :: %{
          tool_call_id: String.t(),
          tool_name: String.t(),
          arguments: map()
        }

  @type interrupt_data :: %{
          action_requests: [action_request()],
          review_configs: %{String.t() => interrupt_config()}
        }

  @type decision :: %{
          required(:type) => :approve | :edit | :reject,
          optional(:arguments) => map()
        }

  @default_decisions [:approve, :edit, :reject]

  @impl true
  def init(opts) do
    interrupt_on = Keyword.get(opts, :interrupt_on, %{})
    config = %{interrupt_on: normalize_interrupt_config(interrupt_on)}
    {:ok, config}
  end

  @impl true
  def after_model(%State{} = state, config) do
    # Check if the last message is an assistant message with tool calls
    case get_last_assistant_message_with_tools(state.messages) do
      nil ->
        # No tool calls to intercept
        {:ok, state}

      assistant_message ->
        # Check if any tool calls require human approval
        tool_calls = assistant_message.tool_calls || []
        interrupt_requests = collect_interrupt_requests(tool_calls, config.interrupt_on)

        if interrupt_requests == [] do
          {:ok, state}
        else
          # Generate interrupt
          interrupt_data = build_interrupt_data(interrupt_requests, config.interrupt_on)
          {:interrupt, state, interrupt_data}
        end
    end
  end

  @doc """
  Process human decisions and update state with tool results or modifications.

  This is called by Agent.resume/3 to apply human decisions to interrupted tool calls.

  ## Parameters

  - `state` - The state at the point of interruption
  - `decisions` - List of decision maps
  - `config` - Middleware configuration

  ## Returns

  - `{:ok, updated_state}` - State with decisions applied
  - `{:error, reason}` - Invalid decisions or other error
  """
  def process_decisions(%State{} = state, decisions, config) when is_list(decisions) do
    # Get the assistant message with tool calls
    case get_last_assistant_message_with_tools(state.messages) do
      nil ->
        {:error, "No tool calls found to process decisions for"}

      assistant_message ->
        tool_calls = assistant_message.tool_calls || []

        # Validate decision count matches tool call count
        if length(decisions) != length(tool_calls) do
          {:error,
           "Decision count (#{length(decisions)}) does not match tool call count (#{length(tool_calls)})"}
        else
          # Process each decision
          apply_decisions(state, tool_calls, decisions, config)
        end
    end
  end

  # Private functions

  defp normalize_interrupt_config(interrupt_on) when is_map(interrupt_on) do
    Map.new(interrupt_on, fn
      {tool_name, true} when is_binary(tool_name) ->
        {tool_name, %{allowed_decisions: @default_decisions}}

      {tool_name, false} when is_binary(tool_name) ->
        {tool_name, %{allowed_decisions: []}}

      {tool_name, %{allowed_decisions: decisions} = config}
      when is_binary(tool_name) and is_list(decisions) ->
        {tool_name, config}

      {tool_name, config} when is_binary(tool_name) and is_map(config) ->
        # If no allowed_decisions specified, use defaults
        {tool_name, Map.put_new(config, :allowed_decisions, @default_decisions)}
    end)
  end

  defp normalize_interrupt_config(_), do: %{}

  defp get_last_assistant_message_with_tools(messages) do
    messages
    |> Enum.reverse()
    |> Enum.find(fn msg ->
      msg.role == :assistant && msg.tool_calls != nil && msg.tool_calls != []
    end)
  end

  defp collect_interrupt_requests(tool_calls, interrupt_on) do
    Enum.filter(tool_calls, fn %ToolCall{name: tool_name} ->
      case Map.get(interrupt_on, tool_name) do
        %{allowed_decisions: decisions} when is_list(decisions) and decisions != [] ->
          true

        _ ->
          false
      end
    end)
  end

  defp build_interrupt_data(tool_calls, interrupt_on) do
    action_requests =
      Enum.map(tool_calls, fn %ToolCall{} = tc ->
        %{
          tool_call_id: tc.call_id,
          tool_name: tc.name,
          arguments: tc.arguments || %{}
        }
      end)

    review_configs =
      tool_calls
      |> Enum.map(fn %ToolCall{name: tool_name} ->
        {tool_name, Map.get(interrupt_on, tool_name, %{allowed_decisions: @default_decisions})}
      end)
      |> Enum.uniq_by(fn {tool_name, _} -> tool_name end)
      |> Map.new()

    %{
      action_requests: action_requests,
      review_configs: review_configs
    }
  end

  defp apply_decisions(state, tool_calls, decisions, config) do
    # Pair tool calls with decisions
    paired =
      Enum.zip(tool_calls, decisions)
      |> Enum.with_index()

    # Validate each decision
    with :ok <- validate_decisions(paired, config.interrupt_on) do
      # Apply decisions and build tool results
      results = build_tool_results(paired)

      # Add tool result message to state
      tool_message = Message.new_tool_result!(%{tool_results: results})
      updated_state = State.add_message(state, tool_message)

      {:ok, updated_state}
    end
  end

  defp validate_decisions(paired_decisions, interrupt_on) do
    Enum.reduce_while(paired_decisions, :ok, fn {{%ToolCall{name: tool_name}, decision}, index},
                                                _ ->
      tool_config = Map.get(interrupt_on, tool_name, %{allowed_decisions: @default_decisions})
      allowed = tool_config.allowed_decisions || @default_decisions

      cond do
        !Map.has_key?(decision, :type) ->
          {:halt,
           {:error,
            "Decision at index #{index} missing required 'type' field: #{inspect(decision)}"}}

        decision.type not in allowed ->
          {:halt,
           {:error,
            "Decision type '#{decision.type}' not allowed for tool '#{tool_name}'. Allowed: #{inspect(allowed)}"}}

        decision.type == :edit && !Map.has_key?(decision, :arguments) ->
          {:halt,
           {:error,
            "Decision at index #{index} with type 'edit' must include 'arguments' field: #{inspect(decision)}"}}

        true ->
          {:cont, :ok}
      end
    end)
  end

  defp build_tool_results(paired_decisions) do
    Enum.map(paired_decisions, fn {{%ToolCall{} = tc, decision}, _index} ->
      case decision.type do
        :approve ->
          # Create a synthetic "approved" result
          # In a real implementation, tools would be executed here
          ToolResult.new!(%{
            tool_call_id: tc.call_id,
            name: tc.name,
            content:
              "Tool call '#{tc.name}' (#{tc.call_id}) was approved for execution with arguments: #{inspect(tc.arguments)}"
          })

        :edit ->
          # Use edited arguments
          edited_args = Map.get(decision, :arguments, %{})

          ToolResult.new!(%{
            tool_call_id: tc.call_id,
            name: tc.name,
            content:
              "Tool call '#{tc.name}' (#{tc.call_id}) was approved for execution with edited arguments: #{inspect(edited_args)}"
          })

        :reject ->
          # Create rejection result
          ToolResult.new!(%{
            tool_call_id: tc.call_id,
            name: tc.name,
            content: "Tool call '#{tc.name}' (#{tc.call_id}) was rejected by human reviewer."
          })
      end
    end)
  end
end
