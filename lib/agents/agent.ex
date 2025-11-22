defmodule LangChain.Agents.Agent do
  @moduledoc """
  Main entry point for creating Agents.

  A DeepAgent is an AI agent with composable middleware that provides
  capabilities like TODO management, filesystem operations, and task delegation.

  ## Basic Usage

      # Create agent with default middleware
      {:ok, agent} = Agent.new(%{
        agent_id: "my-agent-1",
        model: ChatAnthropic.new!(%{model: "claude-3-5-sonnet-20241022"}),
        system_prompt: "You are a helpful assistant."
      })

      # Execute with messages
      state = State.new!(%{messages: [%{role: "user", content: "Hello!"}]})
      {:ok, result_state} = Agent.execute(agent, state)

  ## Middleware Composition

      # Append custom middleware to defaults
      {:ok, agent} = Agent.new(%{
        middleware: [MyCustomMiddleware]
      })

      # Customize default middleware
      {:ok, agent} = Agent.new(%{
        filesystem_opts: [long_term_memory: true]
      })

      # Provide complete middleware stack
      {:ok, agent} = Agent.new(%{
        replace_default_middleware: true,
        middleware: [{MyMiddleware, []}]
      })
  """

  use Ecto.Schema
  import Ecto.Changeset
  require Logger

  alias __MODULE__
  alias LangChain.Agents.Middleware
  alias LangChain.Agents.State
  alias LangChain.LangChainError
  alias LangChain.Message
  alias LangChain.Chains.LLMChain

  @primary_key false
  embedded_schema do
    field :agent_id, :string
    field :model, :any, virtual: true
    field :system_prompt, :string
    field :tools, {:array, :any}, default: [], virtual: true
    field :middleware, {:array, :any}, default: [], virtual: true
    field :name, :string
  end

  @type t :: %Agent{}

  @create_fields [:agent_id, :model, :system_prompt, :tools, :middleware, :name]
  @required_fields [:agent_id, :model]

  @doc """
  Create a new DeepAgent.

  ## Attributes

  - `:agent_id` - Unique identifier for the agent (optional, auto-generated if not provided)
  - `:model` - LangChain ChatModel struct (required)
  - `:system_prompt` - Base system instructions (default: "")
  - `:tools` - Additional tools beyond middleware (default: [])
  - `:middleware` - List of middleware modules/configs (default: [])
  - `:name` - Agent name for identification (default: nil)

  ## Options

  - `:replace_default_middleware` - If true, use only provided middleware (default: false)
  - `:todo_opts` - Options for TodoList middleware
  - `:filesystem_opts` - Options for Filesystem middleware
  - `:summarization_opts` - Options for Summarization middleware (e.g., `[max_tokens_before_summary: 150_000, messages_to_keep: 8]`)
  - `:subagent_opts` - Options for SubAgent middleware
  - `:interrupt_on` - Map of tool names to interrupt configuration (default: nil)

  ### Human-in-the-loop configuration

  The `:interrupt_on` option enables human oversight for specific tools:

      # Simple boolean configuration
      interrupt_on: %{
        "write_file" => true,    # Require approval
        "delete_file" => true,
        "read_file" => false     # No approval needed
      }

      # Advanced configuration
      interrupt_on: %{
        "write_file" => %{
          allowed_decisions: [:approve, :edit, :reject]
        }
      }

  ## Examples

      # Basic agent
      {:ok, agent} = Agent.new(%{
        agent_id: "basic-agent",
        model: ChatAnthropic.new!(%{model: "claude-3-5-sonnet-20241022"}),
        system_prompt: "You are helpful."
      })

      # With custom tools
      {:ok, agent} = Agent.new(%{
        agent_id: "tool-agent",
        model: model,
        tools: [write_file_tool, search_tool]
      })

      # With human-in-the-loop for file operations
      {:ok, agent} = Agent.new(
        %{
          agent_id: "hitl-agent",
          model: model,
          tools: [write_file_tool, delete_file_tool]
        },
        interrupt_on: %{
          "write_file" => true,  # Require approval for writes
          "delete_file" => %{allowed_decisions: [:approve, :reject]}  # No edit for deletes
        }
      )

      # Execute and handle interrupts
      case Agent.execute(agent, state) do
        {:ok, final_state} ->
          IO.puts("Agent completed successfully")

        {:interrupt, interrupted_state, interrupt_data} ->
          # Present interrupt_data.action_requests to user
          # Get their decisions
          decisions = UI.get_decisions(interrupt_data)
          {:ok, final_state} = Agent.resume(agent, interrupted_state, decisions)

        {:error, reason} ->
          Logger.error("Agent failed: \#{inspect(reason)}")
      end

      # With custom middleware configuration
      {:ok, agent} = Agent.new(
        %{
          agent_id: "custom-middleware-agent",
          model: model
        },
        filesystem_opts: [long_term_memory: true]
      )
  """
  def new(attrs \\ %{}, opts \\ []) do
    %Agent{}
    |> cast(attrs, @create_fields)
    |> put_agent_id_if_missing()
    |> validate_required(@required_fields)
    |> put_default_system_prompt()
    |> build_and_initialize_middleware(opts)
    |> assemble_full_system_prompt()
    |> collect_all_tools()
    |> apply_action(:insert)
  end

  @doc """
  Create a new DeepAgent, raising on error.
  """
  def new!(attrs \\ %{}, opts \\ []) do
    case new(attrs, opts) do
      {:ok, agent} -> agent
      {:error, changeset} -> raise LangChainError, changeset
    end
  end

  @doc false
  def changeset(agent \\ %Agent{}, attrs) do
    agent
    |> cast(attrs, @create_fields)
    |> validate_required(@required_fields)
    |> put_default_system_prompt()
  end

  defp put_default_system_prompt(changeset) do
    # Ensure system_prompt is never nil - default to empty string
    case get_field(changeset, :system_prompt) do
      nil -> put_change(changeset, :system_prompt, "")
      _ -> changeset
    end
  end

  defp put_agent_id_if_missing(changeset) do
    case get_field(changeset, :agent_id) do
      nil -> put_change(changeset, :agent_id, generate_agent_id())
      _ -> changeset
    end
  end

  defp build_and_initialize_middleware(changeset, opts) do
    # Build middleware list
    replace_defaults = Keyword.get(opts, :replace_default_middleware, false)
    user_middleware = get_field(changeset, :middleware) || []

    middleware_list =
      case replace_defaults do
        false ->
          # Append user middleware to defaults
          model = get_field(changeset, :model)
          agent_id = get_field(changeset, :agent_id)
          build_default_middleware(model, agent_id, opts) ++ user_middleware

        true ->
          # Use only user-provided middleware
          user_middleware
      end

    # Initialize middleware
    case initialize_middleware_list(middleware_list) do
      {:ok, initialized} ->
        put_change(changeset, :middleware, initialized)

      {:error, reason} ->
        add_error(changeset, :middleware, "initialization failed: #{inspect(reason)}")
    end
  end

  @doc """
  Build the default middleware stack.

  This is a utility function that can be used to build the default middleware
  stack with custom options. Useful when you want to customize middleware
  configuration or when building subagents.

  ## Parameters

  - `model` - The LangChain ChatModel struct
  - `agent_id` - The agent's unique identifier
  - `opts` - Keyword list of middleware options

  ## Options

  - `:todo_opts` - Options for TodoList middleware
  - `:filesystem_opts` - Options for Filesystem middleware
  - `:summarization_opts` - Options for Summarization middleware
  - `:subagent_opts` - Options for SubAgent middleware
  - `:interrupt_on` - Map of tool names to interrupt configuration

  ## Examples

      middleware = Agent.build_default_middleware(
        model,
        "agent-123",
        filesystem_opts: [long_term_memory: true],
        interrupt_on: %{"write_file" => true}
      )
  """
  def build_default_middleware(model, agent_id, opts \\ []) do
    # Build middleware stack Note: SubAgent middleware accesses parent
    # middleware/tools via runtime context, not via init configuration. See
    # execute_loop/3 where custom_context is set.
    base_middleware = [
      # TodoList middleware for task management
      {LangChain.Agents.Middleware.TodoList, Keyword.get(opts, :todo_opts, [])},
      # Filesystem middleware for mock file operations
      {LangChain.Agents.Middleware.FileSystem,
       Keyword.merge([agent_id: agent_id], Keyword.get(opts, :filesystem_opts, []))},
      # SubAgent middleware for delegating to specialized sub-agents
      {LangChain.Agents.Middleware.SubAgent,
       Keyword.merge(
         [agent_id: agent_id, model: model],
         Keyword.get(opts, :subagent_opts, [])
       )},
      # Summarization middleware for managing conversation length
      {LangChain.Agents.Middleware.Summarization,
       Keyword.merge([model: model], Keyword.get(opts, :summarization_opts, []))},
      # PatchToolCalls middleware to fix dangling tool calls
      {LangChain.Agents.Middleware.PatchToolCalls, []}
    ]

    # Conditionally add HumanInTheLoop middleware if interrupt_on is configured
    case Keyword.get(opts, :interrupt_on) do
      nil ->
        base_middleware

      interrupt_on when is_map(interrupt_on) ->
        base_middleware ++
          [{LangChain.Agents.Middleware.HumanInTheLoop, [interrupt_on: interrupt_on]}]
    end
  end

  defp initialize_middleware_list(middleware_list) do
    try do
      initialized =
        middleware_list
        |> Enum.map(&Middleware.init_middleware/1)

      {:ok, initialized}
    rescue
      e -> {:error, Exception.message(e)}
    end
  end

  defp assemble_full_system_prompt(changeset) do
    base_prompt = get_field(changeset, :system_prompt) || ""
    initialized_middleware = get_field(changeset, :middleware) || []

    middleware_prompts =
      initialized_middleware
      |> Enum.map(&Middleware.get_system_prompt/1)
      |> Enum.reject(&(&1 == ""))

    full_prompt =
      [base_prompt | middleware_prompts]
      |> Enum.reject(&(&1 == ""))
      |> Enum.join("\n\n")

    put_change(changeset, :system_prompt, full_prompt)
  end

  defp collect_all_tools(changeset) do
    base_tools = get_field(changeset, :tools) || []
    initialized_middleware = get_field(changeset, :middleware) || []

    middleware_tools =
      initialized_middleware
      |> Enum.flat_map(&Middleware.get_tools/1)

    all_tools = base_tools ++ middleware_tools

    put_change(changeset, :tools, all_tools)
  end

  @doc """
  Execute the agent with the given state.

  Applies middleware hooks in order:
  1. before_model hooks (in order)
  2. LLM execution
  3. after_model hooks (in reverse order)

  ## Returns

  - `{:ok, state}` - Normal completion
  - `{:interrupt, state, interrupt_data}` - Execution paused for human approval
  - `{:error, reason}` - Execution failed

  ## Examples

      state = State.new!(%{messages: [%{role: "user", content: "Hello"}]})

      case Agent.execute(agent, state) do
        {:ok, final_state} ->
          # Normal completion
          handle_response(final_state)

        {:interrupt, interrupted_state, interrupt_data} ->
          # Human approval needed
          decisions = get_human_decisions(interrupt_data)
          {:ok, final_state} = Agent.resume(agent, interrupted_state, decisions)
          handle_response(final_state)

        {:error, err} ->
          # Handle error
          Logger.error("Agent execution failed: \#{inspect(err)}")
      end
  """
  def execute(%Agent{} = agent, %State{} = state, opts \\ []) do
    callbacks = Keyword.get(opts, :callbacks)

    with {:ok, prepared_state} <- apply_before_model_hooks(state, agent.middleware) do
      case execute_model(agent, prepared_state, callbacks) do
        {:ok, response_state} ->
          # Normal completion - run after_model hooks
          apply_after_model_hooks(response_state, agent.middleware)

        {:interrupt, interrupted_state, interrupt_data} ->
          # Interrupt from execute_model - return immediately without after_model hooks
          {:interrupt, interrupted_state, interrupt_data}

        {:error, reason} ->
          {:error, reason}
      end
    end
  end

  @doc """
  Resume agent execution after a human-in-the-loop interrupt.

  Takes decisions from the human reviewer and continues agent execution.

  ## Parameters

  - `agent` - The agent instance
  - `state` - The state at the point of interruption
  - `decisions` - List of decision maps from human reviewer

  ## Decision Format

      decisions = [
        %{type: :approve},                                    # Approve with original arguments
        %{type: :edit, arguments: %{"path" => "other.txt"}}, # Edit arguments
        %{type: :reject}                                      # Reject execution
      ]

  ## Examples

      # Get interrupt from execution
      {:interrupt, state, interrupt_data} = Agent.execute(agent, initial_state)

      # Examine the interrupt data
      # interrupt_data.action_requests - List of tools needing approval
      # interrupt_data.review_configs - Map of tool_name => %{allowed_decisions: [...]}

      # Example: Display to user and get decisions
      decisions =
        Enum.map(interrupt_data.action_requests, fn request ->
          # Show request.tool_name, request.arguments to user
          case get_user_choice(request) do
            :approve -> %{type: :approve}
            :reject -> %{type: :reject}
            {:edit, new_args} -> %{type: :edit, arguments: new_args}
          end
        end)

      # Resume execution with decisions
      {:ok, final_state} = Agent.resume(agent, state, decisions)

      # Or handle edit decision example
      decisions = [
        %{type: :approve},  # Approve first tool
        %{type: :edit, arguments: %{"path" => "/tmp/safe.txt"}},  # Edit second tool's path
        %{type: :reject}  # Reject third tool
      ]

      {:ok, final_state} = Agent.resume(agent, state, decisions)
  """
  def resume(%Agent{} = agent, %State{} = state, decisions, opts \\ [])
      when is_list(decisions) do
    # Find the HumanInTheLoop middleware in the stack
    hitl_middleware =
      Enum.find(agent.middleware, fn {module, _config} ->
        module == LangChain.Agents.Middleware.HumanInTheLoop
      end)

    case hitl_middleware do
      nil ->
        {:error, "Agent does not have HumanInTheLoop middleware configured"}

      {module, config} ->
        # Validate decisions first
        case module.process_decisions(state, decisions, config) do
          {:ok, ^state} ->
            # Validation passed, now execute the approved/edited tools
            execute_approved_tools_and_update_state(
              agent,
              state,
              decisions,
              opts
            )

          {:error, reason} ->
            {:error, reason}
        end
    end
  end

  defp execute_approved_tools_and_update_state(
         %Agent{} = agent,
         %State{} = state,
         decisions,
         opts
       ) do
    callbacks = Keyword.get(opts, :callbacks)

    # Rebuild chain to access tools and execute tool calls with decisions
    with {:ok, langchain_messages} <- validate_messages(state.messages),
         {:ok, chain} <- build_chain(agent, langchain_messages, state, callbacks) do
      # Get the assistant message with tool calls
      assistant_msg =
        Enum.reverse(state.messages)
        |> Enum.find(fn msg ->
          msg.role == :assistant && msg.tool_calls != nil && msg.tool_calls != []
        end)

      case assistant_msg do
        nil ->
          {:error, "No tool calls found in state"}

        %{tool_calls: all_tool_calls} ->
          # Get interrupt_data from state to determine which tools need HITL
          interrupt_data = state.interrupt_data || %{}
          hitl_tool_call_ids = Map.get(interrupt_data, :hitl_tool_call_ids, [])
          action_requests = Map.get(interrupt_data, :action_requests, [])

          # Build decisions map indexed by tool_call_id
          decisions_by_id =
            action_requests
            |> Enum.zip(decisions)
            |> Map.new(fn {action_req, decision} ->
              {action_req.tool_call_id, decision}
            end)

          # Build full decisions array matching ALL tool calls
          # Auto-approve non-HITL tools, use human decisions for HITL tools
          full_decisions =
            Enum.map(all_tool_calls, fn tc ->
              if tc.call_id in hitl_tool_call_ids do
                # Use human decision for HITL tool
                Map.fetch!(decisions_by_id, tc.call_id)
              else
                # Auto-approve non-HITL tool
                %{type: :approve}
              end
            end)

          # Use LLMChain's API to execute tool calls with full decisions
          # This handles tool execution, callbacks, and adding the tool result message
          updated_chain =
            LLMChain.execute_tool_calls_with_decisions(chain, all_tool_calls, full_decisions)

          # Extract the NEW tool result message that was added to the chain
          # (the last message in the chain's exchanged_messages)
          tool_result_message = List.last(updated_chain.exchanged_messages)

          # Add the tool result message to our state
          state_with_results = State.add_message(state, tool_result_message)

          # Continue agent execution with the tool results
          # This allows the LLM to respond to the tool results
          execute(agent, state_with_results, opts)
      end
    end
  end

  # Private functions

  defp generate_agent_id do
    # Generate a unique ID using Elixir's Uniq library or a simple UUID
    ("agent_" <> :crypto.strong_rand_bytes(16)) |> Base.url_encode64(padding: false)
  end

  defp apply_before_model_hooks(state, middleware) do
    Enum.reduce_while(middleware, {:ok, state}, fn mw, {:ok, current_state} ->
      case Middleware.apply_before_model(current_state, mw) do
        {:ok, updated_state} -> {:cont, {:ok, updated_state}}
        {:error, reason} -> {:halt, {:error, reason}}
      end
    end)
  end

  defp apply_after_model_hooks(state, middleware) do
    # Apply in reverse order
    middleware
    |> Enum.reverse()
    |> Enum.reduce_while({:ok, state}, fn mw, {:ok, current_state} ->
      case Middleware.apply_after_model(current_state, mw) do
        {:ok, updated_state} ->
          {:cont, {:ok, updated_state}}

        {:interrupt, interrupted_state, interrupt_data} ->
          # Middleware requested an interrupt, halt and return interrupt
          {:halt, {:interrupt, interrupted_state, interrupt_data}}

        {:error, reason} ->
          {:halt, {:error, reason}}
      end
    end)
  end

  defp execute_model(%Agent{} = agent, %State{} = state, callbacks) do
    with {:ok, langchain_messages} <- validate_messages(state.messages),
         {:ok, chain} <- build_chain(agent, langchain_messages, state, callbacks),
         result <- execute_chain(chain, agent.middleware) do
      case result do
        {:ok, executed_chain} ->
          extract_state_from_chain(executed_chain, state)

        {:interrupt, interrupted_chain, interrupt_data} ->
          # Tool calls need human approval - return interrupt with current state
          case extract_state_from_chain(interrupted_chain, state) do
            {:ok, interrupted_state} ->
              # Add interrupt_data to state so it's available during resume
              state_with_interrupt_data = %{interrupted_state | interrupt_data: interrupt_data}
              {:interrupt, state_with_interrupt_data, interrupt_data}

            {:error, reason} ->
              {:error, reason}
          end

        {:error, reason} ->
          {:error, reason}
      end
    end
  end

  defp validate_messages(messages) do
    # Messages are already LangChain.Message structs, just validate them
    if Enum.all?(messages, &is_struct(&1, Message)) do
      {:ok, messages}
    else
      {:error, "All messages must be LangChain.Message structs"}
    end
  end

  defp build_chain(agent, messages, state, callbacks) do
    # Add system message if we have a system prompt
    messages_with_system =
      case agent.system_prompt do
        prompt when is_binary(prompt) and prompt != "" ->
          [Message.new_system!(prompt) | messages]

        _ ->
          messages
      end

    chain =
      LLMChain.new!(%{
        llm: agent.model,
        custom_context: %{
          state: state,
          # Make parent agent's middleware and tools available to tools (e.g., SubAgent middleware)
          parent_middleware: agent.middleware,
          parent_tools: agent.tools
        }
        # verbose: true
      })
      |> LLMChain.add_tools(agent.tools)
      |> LLMChain.add_messages(messages_with_system)
      |> maybe_add_callbacks(callbacks)

    {:ok, chain}
  rescue
    error -> {:error, "Failed to build chain: #{inspect(error)}"}
  end

  # Helper to conditionally add callbacks to chain
  defp maybe_add_callbacks(chain, nil), do: chain

  defp maybe_add_callbacks(chain, callbacks) when is_map(callbacks) do
    LLMChain.add_callback(chain, callbacks)
  end

  defp execute_chain(chain, middleware) do
    # Check if we should use HITL execution mode
    if has_middleware?(middleware, LangChain.Agents.Middleware.HumanInTheLoop) do
      execute_chain_with_hitl(chain, middleware)
    else
      # Normal execution without interrupts
      case LLMChain.run(chain, mode: :while_needs_response) do
        {:ok, updated_chain} ->
          {:ok, updated_chain}

        {:error, _chain, %LangChainError{} = reason} ->
          {:error, reason.message}

        {:error, _chain, reason} ->
          {:error, "Agent execution failed: #{inspect(reason)}"}
      end
    end
  end

  defp execute_chain_with_hitl(chain, middleware) do
    # Custom execution loop that checks for interrupts BEFORE executing tools
    # 1. Call LLM to get a response
    # 2. Check if response contains tool calls that need approval
    # 3. If yes, return interrupt WITHOUT executing tools
    # 4. If no, execute tools and continue loop

    case LLMChain.run(chain) do
      {:ok, chain_after_llm} ->
        # Check if we have tool calls that need approval
        case check_for_interrupts(chain_after_llm, middleware) do
          {:interrupt, interrupt_data} ->
            # Stop here - don't execute tools yet
            {:interrupt, chain_after_llm, interrupt_data}

          :continue ->
            # No interrupt needed - execute tools and continue
            chain_after_tools = LLMChain.execute_tool_calls(chain_after_llm)

            # Check if we need to continue (needs_response)
            if chain_after_tools.needs_response do
              # Continue the loop
              execute_chain_with_hitl(chain_after_tools, middleware)
            else
              # Done
              {:ok, chain_after_tools}
            end
        end

      {:error, _chain, reason} ->
        {:error, reason}
    end
  end

  defp extract_state_from_chain(chain, original_state) do
    # Extract any state updates from tool results (used by middleware like TodoList)
    # Some tools return State objects as processed_content to update agent state
    state_updates =
      chain.exchanged_messages
      |> Enum.filter(&(&1.role == :tool))
      |> Enum.flat_map(fn message ->
        case message.tool_results do
          nil ->
            []

          tool_results when is_list(tool_results) ->
            Enum.filter(tool_results, fn result ->
              is_struct(result.processed_content, State)
            end)
            |> Enum.map(& &1.processed_content)
        end
      end)

    # Start with the original state and add all messages including tool results
    updated_state = State.add_messages(original_state, chain.exchanged_messages)

    # Merge in any state updates from tools (e.g., todo list updates)
    final_state =
      Enum.reduce(state_updates, updated_state, fn state_update, acc ->
        State.merge_states(acc, state_update)
      end)

    {:ok, final_state}
  rescue
    error -> {:error, "Failed to extract state: #{inspect(error)}"}
  end

  # HITL (Human-in-the-Loop) helper functions

  defp has_middleware?(middleware_list, target_module) do
    Enum.any?(middleware_list, fn {module, _config} ->
      module == target_module
    end)
  end

  defp check_for_interrupts(chain, middleware_list) do
    # Convert chain to state for middleware inspection
    # We only need the messages for HITL to check tool calls
    state = %State{
      messages: chain.exchanged_messages,
      metadata: %{}
    }

    # Find the HumanInTheLoop middleware
    hitl_middleware =
      Enum.find(middleware_list, fn {module, _config} ->
        module == LangChain.Agents.Middleware.HumanInTheLoop
      end)

    case hitl_middleware do
      nil ->
        :continue

      {module, config} ->
        # Check if there are any tool calls that need approval
        case module.check_for_interrupt(state, config) do
          {:interrupt, interrupt_data} ->
            {:interrupt, interrupt_data}

          :continue ->
            :continue
        end
    end
  end
end
