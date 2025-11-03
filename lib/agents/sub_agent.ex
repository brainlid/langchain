defmodule LangChain.Agents.SubAgent do
  @moduledoc """
  SubAgent configuration structures for defining specialized task delegation agents.

  SubAgents enable hierarchical task delegation where a main agent can spawn
  specialized ephemeral agents to handle specific subtasks. Each subagent runs
  in complete isolation with its own conversation context while sharing the
  parent's filesystem and state.

  ## SubAgent Types

  ### Configuration-Based SubAgents

  Built dynamically from configuration. The system creates an Agent instance
  with the specified tools, model, and middleware:

      config = SubAgent.Config.new!(%{
        name: "research-agent",
        description: "Conducts thorough research on complex topics",
        system_prompt: "You are an expert researcher...",
        tools: [internet_search_tool]
      })

  ### Compiled SubAgents

  Pre-built Agent instances for maximum control:

      compiled = SubAgent.Compiled.new!(%{
        name: "custom-agent",
        description: "Specialized custom agent",
        agent: pre_built_agent
      })

  ## Key Characteristics

  - **Ephemeral**: SubAgents are created per-task and don't persist
  - **Isolated**: Fresh conversation context with no parent history
  - **Shared File System**: Access to parent's shared file system
  - **Multi-Step Capable**: Can perform multiple tool calls and operations
  - **Natural Completion**: Finish when LLM produces message without tool calls
  - **No Nesting**: SubAgents cannot spawn sub-subagents
  - **Blocking Execution**: Uses `Agent.execute/2` (blocking, not streaming)
  - **Hidden Internal Work**: Parent only sees final message, not intermediate steps

  ## Execution Model

  SubAgents use **blocking invocation** via `Agent.execute/2`:
  - SubAgent runs until natural completion (no more tool calls)
  - Can make 20+ tool calls and have 50+ message conversation
  - Parent only receives final message as result

  **Token Efficiency**: This provides massive token savings. A SubAgent might:
  - Execute 20 different tool calls
  - Have 50 messages in conversation history
  - Use multiple LLM reasoning steps

  But the parent agent only sees:
  - Single tool result message (the final report)
  - File system changes
  - State updates

  This can save 10,000+ tokens per SubAgent invocation.

  ## Data Flow

  **Input to SubAgent:**
  - Task description (becomes single user message)
  - Parent's filesystem state
  - Parent's custom state (excluding messages and todos)

  **Output from SubAgent:**
  - Final message content (the "report")
  - Modified filesystem state
  - Updated custom state fields
  - ❌ NOT returned: Subagent's conversation history or todos

  ## Usage in Middleware

  SubAgents are registered in the SubAgent middleware and accessed via
  the `task` tool:

      # Main agent uses task tool
      task(
        description: "Research renewable energy impacts",
        subagent_type: "research-agent"
      )

  See `LangChain.Agents.Middleware.SubAgent` for implementation details.
  """

  alias LangChain.Agents.State
  alias LangChain.Message

  # State keys that are NOT transferred to/from SubAgents
  # These maintain conversation and task isolation
  @excluded_state_keys [:messages, :todos]

  @doc """
  Get the list of state keys excluded from SubAgent transfer.

  These keys maintain isolation:
  - `:messages` - Conversation history is not shared
  - `:todos` - Task tracking remains separate

  ## Examples

      SubAgent.excluded_state_keys()
      # => [:messages, :todos]
  """
  def excluded_state_keys, do: @excluded_state_keys

  @doc """
  Prepare state for SubAgent execution.

  Creates an isolated state with:
  - Fresh conversation (single user message with task description)
  - Inherited filesystem and custom state
  - Empty TODO list
  - No parent conversation history

  This implements the state transfer pattern from Python deepagents where
  SubAgents receive `{k: v for k, v in runtime.state.items() if k not in _EXCLUDED_STATE_KEYS}`.

  ## Parameters

  - `task_description` - The task instruction for the SubAgent
  - `parent_state` - The parent agent's current state

  ## Returns

  A new State struct for SubAgent execution

  ## Examples

      parent_state = State.new!(%{
        messages: [Message.new_user!("Original task")],
        files: %{"data.txt" => "content"},
        todos: [todo1, todo2]
      })

      subagent_state = SubAgent.prepare_subagent_state(
        "Research renewable energy",
        parent_state
      )

      # Fresh conversation
      assert length(subagent_state.messages) == 1
      assert subagent_state.messages |> hd() |> Map.get(:content) =~ "renewable energy"

      # Inherited filesystem
      assert subagent_state.files == parent_state.files

      # Empty todos (not inherited)
      assert subagent_state.todos == []
  """
  def prepare_subagent_state(task_description, %State{} = parent_state)
      when is_binary(task_description) do
    # Start with parent state
    parent_state
    # Replace with fresh conversation
    |> Map.put(:messages, [Message.new_user!(task_description)])
    # Empty TODO list (SubAgent starts fresh)
    |> Map.put(:todos, [])
  end

  @doc """
  Extract result from SubAgent execution for return to parent.

  Extracts only:
  - Final message content (the "deliverable")
  - Modified filesystem state
  - Updated custom state fields

  Excludes:
  - SubAgent's conversation history (massive token savings)
  - SubAgent's TODO list (internal tracking)

  This implements the pattern from Python deepagents:
  `{k: v for k, v in result.items() if k not in _EXCLUDED_STATE_KEYS}`

  ## Parameters

  - `subagent_result_state` - The final state from SubAgent.execute()
  - `tool_call_id` - Optional tool call ID for message tracking

  ## Returns

  A map with:
  - `:final_message` - The last message from SubAgent
  - `:state_updates` - Map of state changes (excluding messages/todos)
  - `:tool_call_id` - The tool call ID if provided

  ## Examples

      result_state = %State{
        messages: [msg1, msg2, msg3, final_msg],  # 4 messages
        files: %{"output.txt" => "result"},
        todos: [todo1, todo2],
        metadata: %{computed: true}
      }

      result = SubAgent.extract_subagent_result(result_state, "call_123")

      # Only final message returned
      assert result.final_message == final_msg

      # State updates exclude messages and todos
      assert result.state_updates[:files] == %{"output.txt" => "result"}
      assert result.state_updates[:metadata] == %{computed: true}
      refute Map.has_key?(result.state_updates, :messages)
      refute Map.has_key?(result.state_updates, :todos)
  """
  def extract_subagent_result(%State{} = subagent_result_state, tool_call_id \\ nil) do
    # Get the final message
    final_message = List.last(subagent_result_state.messages)

    # Extract all state EXCEPT excluded keys
    state_updates =
      subagent_result_state
      |> Map.from_struct()
      |> Enum.reject(fn {key, _value} -> key in @excluded_state_keys end)
      |> Map.new()

    %{
      final_message: final_message,
      state_updates: state_updates,
      tool_call_id: tool_call_id
    }
  end

  @doc """
  Merge SubAgent result back into parent state.

  Applies SubAgent's state changes to parent while maintaining proper isolation:
  - Metadata is deep merged
  - Custom state fields are merged
  - Parent's conversation and TODOs are preserved

  **Note**: Files are managed by FileSystemServer and are not part of State.
  File changes made by SubAgents are automatically persisted to the shared FileSystemServer
  and don't need to be merged here.

  ## Parameters

  - `parent_state` - The parent agent's current state
  - `subagent_result` - Result from `extract_subagent_result/2`

  ## Returns

  Updated parent State with SubAgent changes merged

  ## Examples

      parent_state = State.new!(%{
        metadata: %{step: 1}
      })

      subagent_result = %{
        final_message: Message.new_assistant!("Done"),
        state_updates: %{
          metadata: %{step: 2, computed: true}
        }
      }

      merged = SubAgent.merge_subagent_result(parent_state, subagent_result)

      # Metadata deep merged
      assert merged.metadata == %{step: 2, computed: true}
  """
  def merge_subagent_result(%State{} = parent_state, subagent_result) do
    state_updates = subagent_result.state_updates

    # Start with parent state
    parent_state
    # Deep merge metadata
    |> Map.put(
      :metadata,
      deep_merge_maps(parent_state.metadata, state_updates[:metadata] || %{})
    )
    # Merge other custom state fields
    |> merge_custom_state_fields(state_updates)
  end

  defp merge_custom_state_fields(state, state_updates) do
    # Merge any custom fields that aren't standard State fields
    standard_fields = [:messages, :todos, :metadata, :middleware_state]

    state_updates
    |> Enum.reject(fn {key, _} -> key in standard_fields end)
    |> Enum.reduce(state, fn {key, value}, acc ->
      Map.put(acc, key, value)
    end)
  end

  defp deep_merge_maps(left, right) when is_nil(left), do: right
  defp deep_merge_maps(left, right) when is_nil(right), do: left

  defp deep_merge_maps(left, right) when is_map(left) and is_map(right) do
    Map.merge(left, right, fn _key, left_val, right_val ->
      if is_map(left_val) and is_map(right_val) do
        deep_merge_maps(left_val, right_val)
      else
        right_val
      end
    end)
  end

  defp deep_merge_maps(left, _right) when is_map(left), do: left
  defp deep_merge_maps(_left, right) when is_map(right), do: right
  defp deep_merge_maps(_left, _right), do: %{}

  defmodule Config do
    @moduledoc """
    Configuration for dynamically-created SubAgents.

    Defines all parameters needed to instantiate a SubAgent at runtime.
    The SubAgent middleware uses this configuration to create an Agent
    instance when the task tool is invoked.

    ## Fields

    - `:name` - Unique identifier for the subagent (required)
    - `:description` - Description shown to main agent for task delegation (required)
    - `:system_prompt` - Base system instructions for the subagent (required)
    - `:tools` - List of Function tools available to subagent (required)
    - `:model` - Optional ChatModel struct (uses parent's model if nil)
    - `:middleware` - Additional middleware beyond defaults (default: [])
    - `:interrupt_on` - Tool interruption configuration (default: nil)

    ## Examples

        # Minimal configuration
        {:ok, config} = Config.new(%{
          name: "research-agent",
          description: "Conducts research on topics",
          system_prompt: "You are a researcher.",
          tools: [search_tool]
        })

        # With custom model
        {:ok, config} = Config.new(%{
          name: "code-agent",
          description: "Writes code",
          system_prompt: "You are a coding assistant.",
          tools: [run_code_tool],
          model: ChatAnthropic.new!(%{model: "claude-3-opus-20240229"})
        })

        # With additional middleware
        {:ok, config} = Config.new(%{
          name: "careful-agent",
          description: "Careful analysis",
          system_prompt: "Analyze carefully.",
          tools: [analysis_tool],
          middleware: [CustomValidationMiddleware]
        })
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

    @doc """
    Create a new SubAgent configuration.

    ## Required Fields
    - `:name` - Unique identifier
    - `:description` - Agent description
    - `:system_prompt` - System instructions
    - `:tools` - List of tools

    ## Optional Fields
    - `:model` - Custom model (uses parent's if nil)
    - `:middleware` - Additional middleware
    - `:interrupt_on` - Interrupt configuration

    ## Examples

        {:ok, config} = Config.new(%{
          name: "researcher",
          description: "Research agent",
          system_prompt: "You research topics.",
          tools: [search_tool]
        })
    """
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

    @doc """
    Create a new SubAgent configuration, raising on error.
    """
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

    Use this when you want complete control over the subagent's configuration
    or when you've already built an Agent instance. The agent is used as-is
    without additional middleware or configuration.

    ## Fields

    - `:name` - Unique identifier for the subagent (required)
    - `:description` - Description shown to main agent (required)
    - `:agent` - Pre-built Agent instance (required)

    ## Examples

        # Create a custom agent
        {:ok, custom_agent} = Agent.new(
          model: model,
          system_prompt: "Custom instructions",
          tools: [tool1, tool2],
          middleware: [CustomMiddleware]
        )

        # Wrap in Compiled config
        {:ok, compiled} = Compiled.new(%{
          name: "custom-agent",
          description: "My custom agent",
          agent: custom_agent
        })
    """

    use Ecto.Schema
    import Ecto.Changeset
    alias __MODULE__

    @primary_key false
    embedded_schema do
      field :name, :string
      field :description, :string
      field :agent, :any, virtual: true
    end

    @type t :: %Compiled{
            name: String.t(),
            description: String.t(),
            agent: LangChain.Agents.Agent.t()
          }

    @doc """
    Create a new compiled SubAgent configuration.

    ## Required Fields
    - `:name` - Unique identifier
    - `:description` - Agent description
    - `:agent` - Pre-built Agent instance

    ## Examples

        {:ok, compiled} = Compiled.new(%{
          name: "specialized",
          description: "Specialized agent",
          agent: pre_built_agent
        })
    """
    def new(attrs) do
      %Compiled{}
      |> cast(attrs, [:name, :description, :agent])
      |> validate_required([:name, :description, :agent])
      |> validate_length(:name, min: 1, max: 100)
      |> validate_length(:description, min: 1, max: 500)
      |> validate_agent()
      |> apply_action(:insert)
    end

    @doc """
    Create a new compiled SubAgent configuration, raising on error.
    """
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
  end

  @doc """
  Build a registry of subagents from configurations.

  Takes a list of SubAgent configs (Config or Compiled) and builds a map
  of name => agent instance. Config subagents are instantiated using the
  provided defaults, while Compiled subagents use their pre-built agent.

  ## Parameters

  - `configs` - List of SubAgent.Config or SubAgent.Compiled structs
  - `default_model` - Default model for Config subagents without a model
  - `default_middleware` - Base middleware stack for all Config subagents

  ## Returns

  `{:ok, registry}` where registry is a map of `name => agent_instance`

  ## Examples

      configs = [
        SubAgent.Config.new!(%{
          name: "researcher",
          description: "Research agent",
          system_prompt: "You research.",
          tools: [search]
        }),
        SubAgent.Compiled.new!(%{
          name: "custom",
          description: "Custom agent",
          agent: pre_built
        })
      ]

      {:ok, registry} = SubAgent.build_registry(configs, model, middleware)
      # => %{"researcher" => agent1, "custom" => agent2}
  """
  def build_registry(configs, default_model, default_middleware \\ []) do
    try do
      registry =
        Enum.reduce(configs, %{}, fn config, acc ->
          agent = instantiate_subagent(config, default_model, default_middleware)
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
  def build_registry!(configs, default_model, default_middleware \\ []) do
    case build_registry(configs, default_model, default_middleware) do
      {:ok, registry} -> registry
      {:error, reason} -> raise LangChain.LangChainError, reason
    end
  end

  @doc """
  Build descriptions map for subagents.

  Creates a map of name => description for use in the task tool's
  dynamic description generation.

  ## Examples

      configs = [config1, config2]
      descriptions = SubAgent.build_descriptions(configs)
      # => %{"researcher" => "Research agent", "coder" => "Coding agent"}
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

  Additional middleware from the Config is appended after filtering.

  ## Examples

      middleware = SubAgent.subagent_middleware_stack(
        default_middleware,
        config.middleware
      )
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

  defp instantiate_subagent(%Config{} = config, default_model, default_middleware) do
    # Use config's model or fall back to default
    model = config.model || default_model

    # Build middleware stack
    middleware = subagent_middleware_stack(default_middleware, config.middleware)

    # Create the agent
    LangChain.Agents.Agent.new!(
      model: model,
      system_prompt: config.system_prompt,
      tools: config.tools,
      middleware: middleware,
      interrupt_on: config.interrupt_on
    )
  end

  defp instantiate_subagent(%Compiled{} = compiled, _default_model, _default_middleware) do
    # Use pre-built agent as-is
    compiled.agent
  end
end
