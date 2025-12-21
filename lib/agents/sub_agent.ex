defmodule LangChain.Agents.SubAgent do
  @moduledoc """
  A runnable, pausable, and resumable agent execution context.

  ## Core Philosophy

  **"The SubAgent struct HOLDS the LLMChain."**

  This is the key insight that makes pause/resume trivial:
  - **The chain persists** in the SubAgent struct between pause and resume
  - The chain remembers all messages, tool calls, and state
  - Pause = stop executing, save the SubAgent struct
  - Resume = continue the SAME chain with decisions
  - No reconstruction needed

  ## The SubAgent Struct

  The SubAgent holds:
  - **The LLMChain** - THE KEY FIELD - manages the entire conversation
  - **Status tracking** (idle, running, interrupted, completed, error)
  - **Interrupt data** when paused (which tools need approval)
  - **Error** when failed
  - **Metadata** (id, parent_agent_id, created_at)

  ## How It Works

  1. **Initialization**: Create LLMChain with initial messages, tools, and model
  2. **Execution**: Run LLMChain in a loop until completion or interrupt
  3. **Conversation**: All messages live in the chain - it's all there
  4. **Results**: Extract from the final message in the chain

  ## Key Design Principles

  1. **Chain Persistence = Simple Resume**
     - Chain is already paused at the right spot
     - Resume just continues the chain with decisions
     - The chain remembers everything

  2. **Direct Chain Management**
     - SubAgent.execute runs LLMChain.run directly
     - No delegation to Agent
     - Full control over the execution loop

  3. **HITL = Pause/Resume**
     - Interrupt → Chain pauses after LLM returns tool calls
     - Save interrupt data (which tools, what arguments)
     - Resume → Apply decisions, execute tools, continue chain
     - Can have multiple pause/resume cycles naturally

  ## SubAgent Execution Flow

  ### Creating a SubAgent

      # From configuration
      subagent = SubAgent.new_from_config(
        parent_agent_id: "main-agent",
        instructions: "Research renewable energy",
        agent_config: agent_from_registry,
        parent_state: parent_state
      )

  ### Executing

      case SubAgent.execute(subagent) do
        {:ok, completed_subagent} ->
          # Extract result
          result = SubAgent.extract_result(completed_subagent)

        {:interrupt, interrupted_subagent} ->
          # Needs human approval
          # interrupted_subagent.interrupt_data contains action requests

        {:error, error_subagent} ->
          # Execution failed
          # error_subagent.error contains the error
      end

  ### Resuming After Interrupt

      case SubAgent.resume(interrupted_subagent, decisions) do
        {:ok, completed_subagent} -> # Completed
        {:interrupt, interrupted_subagent} -> # Another interrupt
        {:error, error_subagent} -> # Failed
      end

  ## Multiple Interrupts

  The beauty of this design: multiple interrupts just repeat the pause/resume:

      # First execution
      {:interrupt, subagent1} = SubAgent.execute(subagent0)
      # chain paused at: [user, assistant_with_tool_call_1]

      # First resume
      {:interrupt, subagent2} = SubAgent.resume(subagent1, [decision1])
      # chain paused at: [user, assistant_1, tool_result_1, assistant_with_tool_call_2]

      # Second resume
      {:ok, subagent3} = SubAgent.resume(subagent2, [decision2])
      # chain completed
  """

  use Ecto.Schema
  import Ecto.Changeset
  require Logger

  alias __MODULE__
  alias LangChain.Agents.AgentUtils
  alias LangChain.Agents.State
  alias LangChain.Chains.LLMChain
  alias LangChain.Message
  alias LangChain.Utils

  @primary_key false
  embedded_schema do
    # Core execution context - THE KEY FIELD
    # LLMChain struct
    field :chain, :any, virtual: true

    # Status tracking
    field :status, Ecto.Enum,
      values: [:idle, :running, :interrupted, :completed, :error],
      default: :idle

    # HITL configuration - which tools require approval
    field :interrupt_on, :map, virtual: true

    # Interrupt data (when status = :interrupted)
    field :interrupt_data, :map, virtual: true

    # Error information (when status = :error)
    field :error, :any, virtual: true

    # Metadata
    field :id, :string
    field :parent_agent_id, :string
    field :created_at, :utc_datetime
  end

  @type t :: %__MODULE__{
          chain: LLMChain.t() | nil,
          status: :idle | :running | :interrupted | :completed | :error,
          interrupt_on: map() | nil,
          interrupt_data: map() | nil,
          error: term() | nil,
          id: String.t() | nil,
          parent_agent_id: String.t() | nil,
          created_at: DateTime.t() | nil
        }

  ## Construction Functions

  @doc """
  Create a SubAgent from configuration (dynamic subagent).

  The main agent configures a subagent through the task tool by providing:
  - Instructions (becomes user message)
  - Agent configuration (Agent struct)
  - Parent state (for inheritance)

  ## Options

  - `:parent_agent_id` - Parent's agent ID (required)
  - `:instructions` - Task description (required)
  - `:agent_config` - Agent struct with tools, model, middleware (required)
  - `:parent_state` - Parent agent's current state (required)

  ## Examples

      subagent = SubAgent.new_from_config(
        parent_agent_id: "main-agent",
        instructions: "Research renewable energy impacts",
        agent_config: agent_struct,
        parent_state: parent_state
      )
  """
  def new_from_config(opts) do
    parent_agent_id = Keyword.fetch!(opts, :parent_agent_id)
    instructions = Keyword.fetch!(opts, :instructions)
    agent_config = Keyword.fetch!(opts, :agent_config)
    _parent_state = Keyword.fetch!(opts, :parent_state)

    # Generate unique ID
    sub_agent_id = "#{parent_agent_id}-sub-#{:erlang.unique_integer([:positive])}"

    # Build the chain with system prompt + user message
    messages = build_initial_messages(agent_config.assembled_system_prompt, instructions)

    chain =
      LLMChain.new!(%{llm: agent_config.model})
      |> LLMChain.add_tools(agent_config.tools)
      |> LLMChain.add_messages(messages)

    # Extract interrupt_on configuration from agent_config middleware
    interrupt_on = extract_interrupt_on_from_middleware(agent_config.middleware)

    %SubAgent{
      id: sub_agent_id,
      parent_agent_id: parent_agent_id,
      chain: chain,
      interrupt_on: interrupt_on,
      status: :idle,
      created_at: DateTime.utc_now()
    }
  end

  @doc """
  Create a SubAgent from compiled agent (pre-built).

  A compiled subagent is pre-defined by the application with complete control
  over configuration.

  ## Options

  - `:parent_agent_id` - Parent's agent ID (required)
  - `:instructions` - Task description (required)
  - `:compiled_agent` - Pre-built Agent struct (required)
  - `:parent_state` - Parent agent's current state (required)
  - `:initial_messages` - Optional initial message sequence (default: [])

  ## Examples

      subagent = SubAgent.new_from_compiled(
        parent_agent_id: "main-agent",
        instructions: "Extract structured data",
        compiled_agent: data_extractor_agent,
        parent_state: parent_state,
        initial_messages: [prep_message]
      )
  """
  def new_from_compiled(opts) do
    parent_agent_id = Keyword.fetch!(opts, :parent_agent_id)
    instructions = Keyword.fetch!(opts, :instructions)
    compiled_agent = Keyword.fetch!(opts, :compiled_agent)
    _parent_state = Keyword.fetch!(opts, :parent_state)
    initial_messages = Keyword.get(opts, :initial_messages, [])

    # Generate unique ID
    sub_agent_id = "#{parent_agent_id}-sub-#{:erlang.unique_integer([:positive])}"

    # Build chain with optional initial messages + instructions
    system_messages = build_initial_messages(compiled_agent.assembled_system_prompt, nil)
    user_message = Message.new_user!(instructions)
    all_messages = system_messages ++ initial_messages ++ [user_message]

    chain =
      LLMChain.new!(%{llm: compiled_agent.model})
      |> LLMChain.add_tools(compiled_agent.tools)
      |> LLMChain.add_messages(all_messages)

    # Extract interrupt_on configuration from compiled_agent middleware
    interrupt_on = extract_interrupt_on_from_middleware(compiled_agent.middleware)

    %SubAgent{
      id: sub_agent_id,
      parent_agent_id: parent_agent_id,
      chain: chain,
      interrupt_on: interrupt_on,
      status: :idle,
      created_at: DateTime.utc_now()
    }
  end

  ## Execution Functions

  @doc """
  Execute the SubAgent.

  Runs the LLMChain until:
  - Natural completion (no more tool calls)
  - HITL interrupt (tool needs approval)
  - Error

  Returns updated SubAgent struct with new status.

  ## Examples

      case SubAgent.execute(subagent) do
        {:ok, completed_subagent} ->
          result = SubAgent.extract_result(completed_subagent)

        {:interrupt, interrupted_subagent} ->
          # interrupted_subagent.interrupt_data contains action_requests

        {:error, error_subagent} ->
          # error_subagent.error contains the error
      end
  """
  def execute(%SubAgent{status: :idle, chain: chain, interrupt_on: interrupt_on} = subagent) do
    Logger.debug("SubAgent #{subagent.id} executing")

    # Update status to running
    running_subagent = %{subagent | status: :running}

    # Execute with HITL support using execution loop
    case execute_chain_with_hitl(chain, interrupt_on) do
      {:ok, final_chain} ->
        # Chain completed successfully (needs_response = false)
        Logger.debug("SubAgent #{subagent.id} completed successfully")

        {:ok,
         %SubAgent{
           running_subagent
           | status: :completed,
             chain: final_chain
         }}

      {:interrupt, interrupted_chain, interrupt_data} ->
        # Chain hit HITL interrupt
        Logger.debug("SubAgent #{subagent.id} interrupted for HITL")

        {:interrupt,
         %SubAgent{
           running_subagent
           | status: :interrupted,
             chain: interrupted_chain,
             interrupt_data: interrupt_data
         }}

      {:error, reason} ->
        Logger.error("SubAgent #{subagent.id} execution error: #{inspect(reason)}")

        {:error,
         %SubAgent{
           running_subagent
           | status: :error,
             error: reason
         }}
    end
  end

  def execute(%SubAgent{status: status}) do
    {:error, {:invalid_status, status, :expected_idle}}
  end

  @doc """
  Resume the SubAgent after HITL interrupt.

  Takes human decisions and continues execution from where it left off.
  This is the magic - the agent state is already paused at the right spot.

  ## Parameters

  - `subagent` - SubAgent with status :interrupted
  - `decisions` - List of decision maps from human reviewer

  ## Returns

  - `{:ok, completed_subagent}` - Execution completed
  - `{:interrupt, interrupted_subagent}` - Another interrupt (multiple HITL tools)
  - `{:error, error_subagent}` - Resume failed

  ## Examples

      decisions = [
        %{type: :approve},
        %{type: :edit, arguments: %{"path" => "safe.txt"}}
      ]

      case SubAgent.resume(interrupted_subagent, decisions) do
        {:ok, completed_subagent} ->
          result = SubAgent.extract_result(completed_subagent)

        {:interrupt, interrupted_again} ->
          # Another interrupt - repeat the process

        {:error, error_subagent} ->
          # Handle error
      end
  """
  def resume(
        %SubAgent{
          status: :interrupted,
          chain: chain,
          interrupt_on: interrupt_on,
          interrupt_data: interrupt_data
        } = subagent,
        decisions
      ) do
    Logger.debug("SubAgent #{subagent.id} resuming with #{length(decisions)} decisions")

    # Update status to running
    running_subagent = %{
      subagent
      | status: :running,
        interrupt_data: nil
    }

    # Extract interrupt context
    action_requests = interrupt_data.action_requests
    hitl_tool_call_ids = interrupt_data.hitl_tool_call_ids

    # Get ALL tool calls from the last assistant message (HITL + non-HITL)
    all_tool_calls = AgentUtils.get_tool_calls_from_last_message(chain)

    # Build full decisions list (mix human decisions with auto-approvals)
    # This is needed because LLMChain.execute_tool_calls_with_decisions expects
    # a decision for EVERY tool call, not just the HITL ones
    full_decisions =
      AgentUtils.build_full_decisions(
        all_tool_calls,
        hitl_tool_call_ids,
        decisions,
        action_requests
      )

    # Use LLMChain to execute tool calls with decisions
    # This handles approve/edit/reject logic and creates tool result messages
    chain_with_results =
      LLMChain.execute_tool_calls_with_decisions(
        chain,
        all_tool_calls,
        full_decisions
      )

    # Continue execution loop after applying decisions
    case execute_chain_with_hitl(chain_with_results, interrupt_on) do
      {:ok, final_chain} ->
        # SubAgent completed after resume
        Logger.debug("SubAgent #{subagent.id} completed after resume")

        {:ok,
         %SubAgent{
           running_subagent
           | status: :completed,
             chain: final_chain
         }}

      {:interrupt, interrupted_chain, new_interrupt_data} ->
        # Another interrupt (multiple HITL tools in sequence)
        Logger.debug("SubAgent #{subagent.id} interrupted again")

        {:interrupt,
         %SubAgent{
           running_subagent
           | status: :interrupted,
             chain: interrupted_chain,
             interrupt_data: new_interrupt_data
         }}

      {:error, reason} ->
        Logger.error("SubAgent #{subagent.id} resume failed: #{inspect(reason)}")

        {:error,
         %SubAgent{
           running_subagent
           | status: :error,
             error: reason
         }}
    end
  end

  def resume(%SubAgent{status: status}, _decisions) do
    {:error, {:invalid_status, status, :expected_interrupted}}
  end

  ## State Query Functions

  @doc """
  Check if SubAgent can be executed.

  Only SubAgents with status :idle can be executed.
  """
  def can_execute?(%SubAgent{status: :idle}), do: true
  def can_execute?(%SubAgent{}), do: false

  @doc """
  Check if SubAgent can be resumed.

  Only SubAgents with status :interrupted can be resumed.
  """
  def can_resume?(%SubAgent{status: :interrupted}), do: true
  def can_resume?(%SubAgent{}), do: false

  @doc """
  Check if SubAgent is in a terminal state.

  Terminal states are :completed and :error.
  """
  def is_terminal?(%SubAgent{status: :completed}), do: true
  def is_terminal?(%SubAgent{status: :error}), do: true
  def is_terminal?(%SubAgent{}), do: false

  ## Result Extraction

  @doc """
  Extract result from completed SubAgent.

  For completed SubAgents, extracts the final message content as a string.
  This is the default extraction - middleware or custom logic can provide
  different extraction.

  Returns `{:ok, string}` on success or `{:error, reason}` on failure.

  ## Examples

      {:ok, completed_subagent} = SubAgent.execute(subagent)
      {:ok, result} = SubAgent.extract_result(completed_subagent)
      # => {:ok, "Research complete: Solar energy has shown..."}
  """
  def extract_result(%SubAgent{status: :completed, chain: chain}) do
    case Utils.ChainResult.to_string(chain) do
      {:ok, result} -> {:ok, result}
      {:error, _chain, reason} -> {:error, reason}
    end
  end

  def extract_result(%SubAgent{status: status}) do
    {:error, {:invalid_status, status, :expected_completed}}
  end

  ## Private Helper Functions

  defp build_initial_messages(system_prompt, instructions)
       when is_binary(system_prompt) and system_prompt != "" do
    case instructions do
      nil -> [Message.new_system!(system_prompt)]
      _ -> [Message.new_system!(system_prompt), Message.new_user!(instructions)]
    end
  end

  defp build_initial_messages(_system_prompt, instructions) when is_binary(instructions) do
    [Message.new_user!(instructions)]
  end

  defp build_initial_messages(_system_prompt, nil), do: []

  # Extract interrupt_on configuration from middleware list
  defp extract_interrupt_on_from_middleware(middleware) when is_list(middleware) do
    Enum.find_value(middleware, %{}, fn
      %LangChain.Agents.MiddlewareEntry{
        module: LangChain.Agents.Middleware.HumanInTheLoop,
        config: config
      } ->
        cond do
          is_map(config) -> Map.get(config, :interrupt_on)
          is_list(config) -> Keyword.get(config, :interrupt_on)
          true -> nil
        end

      _ ->
        nil
    end)
  end

  defp extract_interrupt_on_from_middleware(_), do: %{}

  # Execute chain with HITL interrupt detection
  # This mirrors Agent.execute_chain_with_hitl from agent.ex
  defp execute_chain_with_hitl(chain, interrupt_on) do
    # Call LLM to get response
    case LLMChain.run(chain) do
      {:ok, chain_after_llm} ->
        # Check if response has tool calls that need approval
        case AgentUtils.check_for_hitl_interrupt(chain_after_llm, interrupt_on || %{}) do
          {:interrupt, interrupt_data} ->
            # Stop here - don't execute tools yet, wait for human approval
            {:interrupt, chain_after_llm, interrupt_data}

          :continue ->
            # No interrupt needed - execute tools automatically
            chain_after_tools = LLMChain.execute_tool_calls(chain_after_llm)

            # Check if chain needs more work (needs_response flag)
            # If needs_response is nil or true, continue; if false, we're done
            if Map.get(chain_after_tools, :needs_response, false) do
              # Continue the loop - LLM needs to respond to tool results
              execute_chain_with_hitl(chain_after_tools, interrupt_on)
            else
              # Chain is done
              {:ok, chain_after_tools}
            end
        end

      {:error, _chain, reason} ->
        {:error, reason}
    end
  end

  ## Nested Modules (Config and Compiled)

  defmodule Config do
    @moduledoc """
    Configuration for dynamically-created SubAgents.

    Defines all parameters needed to instantiate a SubAgent at runtime.
    """

    use Ecto.Schema
    import Ecto.Changeset
    alias __MODULE__

    @primary_key false
    embedded_schema do
      field :name, :string
      field :description, :string
      field :system_prompt, :string
      field :tools, {:array, :any}, default: [], virtual: true
      field :model, :any, virtual: true
      field :middleware, {:array, :any}, default: [], virtual: true
      field :interrupt_on, :map
    end

    @type t :: %Config{
            name: String.t(),
            description: String.t(),
            system_prompt: String.t(),
            tools: [LangChain.Function.t()],
            model: term() | nil,
            middleware: list(),
            interrupt_on: map() | nil
          }

    def new(attrs) do
      %Config{}
      |> cast(attrs, [
        :name,
        :description,
        :system_prompt,
        :tools,
        :model,
        :middleware,
        :interrupt_on
      ])
      |> validate_required([:name, :description, :system_prompt, :tools])
      |> validate_length(:name, min: 1, max: 100)
      |> validate_length(:description, min: 1, max: 500)
      |> validate_length(:system_prompt, min: 1, max: 10_000)
      |> validate_tools()
      |> apply_action(:insert)
    end

    def new!(attrs) do
      case new(attrs) do
        {:ok, config} -> config
        {:error, changeset} -> raise LangChain.LangChainError, changeset
      end
    end

    defp validate_tools(changeset) do
      case get_field(changeset, :tools) do
        tools when is_list(tools) and length(tools) > 0 ->
          if Enum.all?(tools, &is_struct(&1, LangChain.Function)) do
            changeset
          else
            add_error(changeset, :tools, "must be a list of LangChain.Function structs")
          end

        [] ->
          add_error(changeset, :tools, "must contain at least one tool")

        _ ->
          add_error(changeset, :tools, "must be a list")
      end
    end
  end

  defmodule Compiled do
    @moduledoc """
    Pre-compiled SubAgent with an existing Agent instance.
    """

    use Ecto.Schema
    import Ecto.Changeset
    alias __MODULE__

    @primary_key false
    embedded_schema do
      field :name, :string
      field :description, :string
      field :agent, :any, virtual: true
      field :extract_result, :any, virtual: true
      field :initial_messages, {:array, :any}, default: [], virtual: true
    end

    @type t :: %Compiled{
            name: String.t(),
            description: String.t(),
            agent: LangChain.Agents.Agent.t(),
            extract_result: (State.t() -> any()) | nil,
            initial_messages: [LangChain.Message.t()]
          }

    def new(attrs) do
      %Compiled{}
      |> cast(attrs, [:name, :description, :agent, :extract_result, :initial_messages])
      |> validate_required([:name, :description, :agent])
      |> validate_length(:name, min: 1, max: 100)
      |> validate_length(:description, min: 1, max: 500)
      |> validate_agent()
      |> validate_extract_result()
      |> validate_initial_messages()
      |> apply_action(:insert)
    end

    def new!(attrs) do
      case new(attrs) do
        {:ok, compiled} -> compiled
        {:error, changeset} -> raise LangChain.LangChainError, changeset
      end
    end

    defp validate_agent(changeset) do
      case get_field(changeset, :agent) do
        %LangChain.Agents.Agent{} ->
          changeset

        _ ->
          add_error(changeset, :agent, "must be a LangChain.Agents.Agent struct")
      end
    end

    defp validate_extract_result(changeset) do
      case get_field(changeset, :extract_result) do
        nil ->
          changeset

        fun when is_function(fun, 1) ->
          changeset

        _ ->
          add_error(
            changeset,
            :extract_result,
            "must be a function that takes one argument (State)"
          )
      end
    end

    defp validate_initial_messages(changeset) do
      case get_field(changeset, :initial_messages) do
        nil ->
          # Treat nil as empty list
          put_change(changeset, :initial_messages, [])

        [] ->
          changeset

        messages when is_list(messages) ->
          # Validate all items are Message structs
          if Enum.all?(messages, &is_struct(&1, LangChain.Message)) do
            changeset
          else
            add_error(changeset, :initial_messages, "must be a list of Message structs")
          end

        _ ->
          add_error(changeset, :initial_messages, "must be a list of Message structs")
      end
    end
  end

  ## AgentMap Building Functions

  @doc """
  Build an agent map of subagents from configurations.
  """
  def build_agent_map(configs, default_model, default_middleware \\ []) do
    try do
      registry =
        Enum.reduce(configs, %{}, fn config, acc ->
          agent = configure_new_subagent(config, default_model, default_middleware)
          Map.put(acc, config.name, agent)
        end)

      {:ok, registry}
    rescue
      e -> {:error, "Failed to build subagent registry: #{Exception.message(e)}"}
    end
  end

  @doc """
  Build a registry of subagents, raising on error.
  """
  def build_agent_map!(configs, default_model, default_middleware \\ []) do
    case build_agent_map(configs, default_model, default_middleware) do
      {:ok, registry} -> registry
      {:error, reason} -> raise LangChain.LangChainError, reason
    end
  end

  @doc """
  Build descriptions map for subagents.
  """
  def build_descriptions(configs) do
    Enum.reduce(configs, %{}, fn config, acc ->
      Map.put(acc, config.name, config.description)
    end)
  end

  @doc """
  Get the middleware stack for a subagent.

  SubAgents get the default middleware stack but with SubAgent middleware
  removed to prevent nesting (subagents can't spawn sub-subagents).
  """
  def subagent_middleware_stack(default_middleware, additional_middleware \\ []) do
    # Remove SubAgent middleware to prevent nesting
    filtered =
      Enum.reject(default_middleware, fn
        {LangChain.Agents.Middleware.SubAgent, _} -> true
        LangChain.Agents.Middleware.SubAgent -> true
        _ -> false
      end)

    filtered ++ additional_middleware
  end

  ## Private Functions

  defp configure_new_subagent(%Config{} = config, default_model, default_middleware) do
    # Use config's model or fall back to default
    model = config.model || default_model

    # Build middleware stack (filters out SubAgent middleware)
    middleware = subagent_middleware_stack(default_middleware, config.middleware)

    # Create the agent with explicit middleware (replace defaults to avoid duplication)
    LangChain.Agents.Agent.new!(
      %{
        model: model,
        base_system_prompt: config.system_prompt,
        tools: config.tools,
        middleware: middleware
      },
      replace_default_middleware: true,
      interrupt_on: config.interrupt_on
    )
  end

  defp configure_new_subagent(%Compiled{} = compiled, _default_model, _default_middleware) do
    # Return the entire Compiled struct to preserve initial_messages and other metadata
    compiled
  end
end
