defmodule LangChain.DeepAgents.Agent do
  @moduledoc """
  Main entry point for creating DeepAgents.

  A DeepAgent is an AI agent with composable middleware that provides
  capabilities like TODO management, filesystem operations, and task delegation.

  ## Basic Usage

      # Create agent with default middleware
      {:ok, agent} = Agent.new(
        model: ChatAnthropic.new!(%{model: "claude-3-5-sonnet-20241022"}),
        system_prompt: "You are a helpful assistant."
      )

      # Execute with messages
      state = State.new!(%{messages: [%{role: "user", content: "Hello!"}]})
      {:ok, result_state} = Agent.execute(agent, state)

  ## Middleware Composition

      # Append custom middleware to defaults
      {:ok, agent} = Agent.new(
        middleware: [MyCustomMiddleware]
      )

      # Customize default middleware
      {:ok, agent} = Agent.new(
        filesystem_opts: [long_term_memory: true]
      )

      # Provide complete middleware stack
      {:ok, agent} = Agent.new(
        replace_default_middleware: true,
        middleware: [{MyMiddleware, []}]
      )
  """

  use Ecto.Schema
  require Logger

  alias __MODULE__
  alias LangChain.DeepAgents.{Middleware, State}
  alias LangChain.LangChainError
  alias LangChain.Message
  alias LangChain.Function
  alias LangChain.Chains.LLMChain

  @primary_key false
  embedded_schema do
    field :model, :any, virtual: true
    field :system_prompt, :string
    field :tools, {:array, :any}, default: [], virtual: true
    field :middleware, {:array, :any}, default: [], virtual: true
    field :name, :string
  end

  @type t :: %Agent{}

  @doc """
  Create a new DeepAgent.

  ## Options

  - `:model` - LangChain ChatModel struct (required)
  - `:system_prompt` - Base system instructions (default: "")
  - `:tools` - Additional tools beyond middleware (default: [])
  - `:middleware` - List of middleware modules/configs (default: [])
  - `:replace_default_middleware` - If true, use only provided middleware (default: false)
  - `:name` - Agent name for identification (default: nil)
  - `:interrupt_on` - Map of tool names to interrupt configuration (default: nil)

  ### Middleware-specific options

  These options customize the default middleware when `replace_default_middleware` is false:

  - `:todo_opts` - Options for TodoList middleware
  - `:filesystem_opts` - Options for Filesystem middleware
  - `:summarization_opts` - Options for Summarization middleware (e.g., `[max_tokens_before_summary: 150_000, messages_to_keep: 8]`)
  - `:subagent_opts` - Options for SubAgent middleware

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
      {:ok, agent} = Agent.new(
        model: ChatAnthropic.new!(%{model: "claude-3-5-sonnet-20241022"}),
        system_prompt: "You are helpful."
      )

      # With custom tools
      {:ok, agent} = Agent.new(
        model: model,
        tools: [write_file_tool, search_tool]
      )

      # With human-in-the-loop for file operations
      {:ok, agent} = Agent.new(
        model: model,
        tools: [write_file_tool, delete_file_tool],
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
        model: model,
        filesystem_opts: [long_term_memory: true]
      )
  """
  def new(opts \\ []) do
    with {:ok, model} <- validate_model(opts),
         {:ok, middleware_list} <- build_middleware_list(opts),
         {:ok, initialized_middleware} <- initialize_middleware(middleware_list),
         {:ok, system_prompt} <- assemble_system_prompt(opts, initialized_middleware),
         {:ok, all_tools} <- collect_tools(opts, initialized_middleware) do
      agent = %Agent{
        model: model,
        system_prompt: system_prompt,
        tools: all_tools,
        middleware: initialized_middleware,
        name: Keyword.get(opts, :name)
      }

      {:ok, agent}
    end
  end

  @doc """
  Create a new DeepAgent, raising on error.
  """
  def new!(opts \\ []) do
    case new(opts) do
      {:ok, agent} -> agent
      {:error, reason} -> raise LangChainError, reason
    end
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
  def execute(%Agent{} = agent, %State{} = state) do
    with {:ok, prepared_state} <- apply_before_model_hooks(state, agent.middleware),
         {:ok, response_state} <- execute_model(agent, prepared_state),
         result <- apply_after_model_hooks(response_state, agent.middleware) do
      result
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
  def resume(%Agent{} = agent, %State{} = state, decisions) when is_list(decisions) do
    # Find the HumanInTheLoop middleware in the stack
    hitl_middleware =
      Enum.find(agent.middleware, fn {module, _config} ->
        module == LangChain.DeepAgents.Middleware.HumanInTheLoop
      end)

    case hitl_middleware do
      nil ->
        {:error, "Agent does not have HumanInTheLoop middleware configured"}

      {module, config} ->
        # Process the decisions through the middleware
        case module.process_decisions(state, decisions, config) do
          {:ok, updated_state} ->
            # Continue execution - skip before_model and execute_model phases
            # Just run remaining after_model hooks if needed
            {:ok, updated_state}

          {:error, reason} ->
            {:error, reason}
        end
    end
  end

  # Private functions

  defp validate_model(opts) do
    case Keyword.get(opts, :model) do
      nil -> {:error, "Model is required"}
      model -> {:ok, model}
    end
  end

  defp build_middleware_list(opts) do
    replace_defaults = Keyword.get(opts, :replace_default_middleware, false)
    user_middleware = Keyword.get(opts, :middleware, [])

    middleware =
      case replace_defaults do
        false ->
          # Append user middleware to defaults
          default_middleware(opts) ++ user_middleware

        true ->
          # Use only user-provided middleware
          user_middleware
      end

    {:ok, middleware}
  end

  defp default_middleware(opts) do
    model = Keyword.get(opts, :model)

    # Build default middleware stack
    base_middleware = [
      # TodoList middleware for task management
      {LangChain.DeepAgents.Middleware.TodoList, Keyword.get(opts, :todo_opts, [])},
      # Filesystem middleware for mock file operations
      {LangChain.DeepAgents.Middleware.Filesystem, Keyword.get(opts, :filesystem_opts, [])},
      # Summarization middleware for managing conversation length
      {LangChain.DeepAgents.Middleware.Summarization,
       Keyword.merge([model: model], Keyword.get(opts, :summarization_opts, []))},
      # PatchToolCalls middleware to fix dangling tool calls
      {LangChain.DeepAgents.Middleware.PatchToolCalls, []}
    ]

    # Conditionally add HumanInTheLoop middleware if interrupt_on is configured
    middleware_with_hitl =
      case Keyword.get(opts, :interrupt_on) do
        nil ->
          base_middleware

        interrupt_on when is_map(interrupt_on) ->
          base_middleware ++
            [{LangChain.DeepAgents.Middleware.HumanInTheLoop, [interrupt_on: interrupt_on]}]
      end

    middleware_with_hitl
  end

  defp initialize_middleware(middleware_list) do
    try do
      initialized =
        middleware_list
        |> Enum.map(&Middleware.init_middleware/1)

      {:ok, initialized}
    rescue
      e -> {:error, Exception.message(e)}
    end
  end

  defp assemble_system_prompt(opts, initialized_middleware) do
    base_prompt = Keyword.get(opts, :system_prompt, "")

    middleware_prompts =
      initialized_middleware
      |> Enum.map(&Middleware.get_system_prompt/1)
      |> Enum.reject(&(&1 == ""))

    full_prompt =
      [base_prompt | middleware_prompts]
      |> Enum.reject(&(&1 == ""))
      |> Enum.join("\n\n")

    {:ok, full_prompt}
  end

  defp collect_tools(opts, initialized_middleware) do
    base_tools = Keyword.get(opts, :tools, [])

    middleware_tools =
      initialized_middleware
      |> Enum.flat_map(&Middleware.get_tools/1)

    all_tools = base_tools ++ middleware_tools

    {:ok, all_tools}
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

  defp execute_model(%Agent{} = agent, %State{} = state) do
    with {:ok, langchain_messages} <- validate_messages(state.messages),
         {:ok, chain} <- build_chain(agent, langchain_messages, state),
         {:ok, executed_chain} <- execute_chain(chain),
         {:ok, updated_state} <- extract_state_from_chain(executed_chain, state) do
      {:ok, updated_state}
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

  defp build_chain(agent, messages, state) do
    # Wrap tools to capture state updates and pass the current state
    wrapped_tools = wrap_tools_for_state_capture(agent.tools, state)

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
        tools: wrapped_tools,
        custom_context: %{state: state}
      })
      |> LLMChain.add_messages(messages_with_system)

    {:ok, chain}
  rescue
    error -> {:error, "Failed to build chain: #{inspect(error)}"}
  end

  defp wrap_tools_for_state_capture(tools, initial_state) do
    Enum.map(tools, fn %Function{} = tool ->
      # Wrap the original function to handle state updates
      original_fn = tool.function

      wrapped_fn = fn args, context ->
        # Get the most recent state from context, or use initial state
        current_state = get_in(context, [:state]) || initial_state

        # Call the original function with updated context that includes state
        context_with_state = Map.put(context || %{}, :state, current_state)

        case original_fn.(args, context_with_state) do
          # Tool returned state update
          {:ok, result, %State{} = updated_state} ->
            # Store the state in processed_content
            {:ok, result, updated_state}

          # Standard tool result
          {:ok, result} ->
            {:ok, result}

          # Tool error
          {:error, reason} ->
            {:error, reason}
        end
      end

      %Function{tool | function: wrapped_fn}
    end)
  end

  defp execute_chain(chain) do
    case LLMChain.run(chain, mode: :while_needs_response) do
      {:ok, updated_chain} ->
        {:ok, updated_chain}

      {:error, _chain, reason} ->
        {:error, "Chain execution failed: #{inspect(reason)}"}
    end
  end

  defp extract_state_from_chain(chain, original_state) do
    # Get only non-tool messages (assistant and potentially user messages from the exchange)
    # Tool messages are internal to the chain execution
    new_messages =
      chain.exchanged_messages
      |> Enum.reject(&(&1.role == :tool))

    # Extract any state updates from tool results
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

    # Start with the original state and add new messages
    updated_state = State.add_messages(original_state, new_messages)

    # Merge in any state updates from tools
    final_state =
      Enum.reduce(state_updates, updated_state, fn state_update, acc ->
        State.merge_states(acc, state_update)
      end)

    {:ok, final_state}
  rescue
    error -> {:error, "Failed to extract state: #{inspect(error)}"}
  end
end
