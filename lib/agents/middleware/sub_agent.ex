defmodule LangChain.Agents.Middleware.SubAgent do
  @moduledoc """
  Middleware for delegating tasks to specialized SubAgents.

  Provides a `task` tool that allows the main agent to delegate complex,
  multi-step work to specialized SubAgents. SubAgents run in isolated
  contexts with their own conversation history, providing token efficiency
  and clean separation of concerns.

  ## Features

  - **Dynamic SubAgents**: Create SubAgents from configuration at runtime
  - **Pre-compiled SubAgents**: Use pre-built Agent instances
  - **HITL Propagation**: SubAgent interrupts automatically propagate to parent
  - **Token Efficiency**: Parent only sees final result, not SubAgent's internal work
  - **Process Isolation**: SubAgents run as supervised processes

  ## Configuration Options

  The middleware accepts these options:

    * `:subagents` - List of `SubAgent.Config` or `SubAgent.Compiled` configurations
      for pre-defined subagents. Defaults to `[]`.

    * `:model` - The chat model for dynamic subagents. Required.

    * `:middleware` - Additional middleware to add to subagents. Defaults to `[]`.

    * `:block_middleware` - List of middleware modules to exclude from general-purpose
      subagent inheritance. Defaults to `[]`. See "Middleware Filtering" below.

  ## Configuration Example

      middleware = [
        {SubAgent, [
          model: model,
          subagents: [
            SubAgent.Config.new!(%{
              name: "researcher",
              description: "Research topics using internet search",
              system_prompt: "You are an expert researcher...",
              tools: [internet_search_tool]
            }),
            SubAgent.Compiled.new!(%{
              name: "coder",
              description: "Write code for specific tasks",
              agent: pre_built_coder_agent
            })
          ],
          block_middleware: [ConversationTitle, Summarization]
        ]}
      ]

  ## Middleware Filtering

  When a general-purpose subagent is created, it inherits the parent agent's middleware
  stack with certain exclusions:

  1. **SubAgent middleware is ALWAYS excluded** - This prevents recursive subagent
     nesting which could lead to resource exhaustion. You cannot override this.

  2. **Blocked middleware is excluded** - Any modules listed in `:block_middleware`
     are filtered out before passing to the subagent.

  ### Example: Blocking Unnecessary Middleware

  Some middleware is inappropriate for short-lived subagents:

      {SubAgent, [
        model: model,
        subagents: [],

        # These middleware modules won't be inherited by general-purpose subagents
        block_middleware: [
          LangChain.Agents.Middleware.ConversationTitle,  # Subagents don't need titles
          LangChain.Agents.Middleware.Summarization       # Short tasks don't need summarization
        ]
      ]}

  ### Pre-configured Subagents

  The `:block_middleware` option only affects **general-purpose** subagents created
  dynamically via the `task` tool. Pre-configured subagents (defined in `:subagents`)
  use their own explicitly defined middleware and are NOT affected by this option.

      {SubAgent, [
        subagents: [
          # This subagent defines its own middleware - block_middleware doesn't apply
          SubAgent.Config.new!(%{
            name: "researcher",
            middleware: [ConversationTitle]  # Explicitly included
          })
        ],
        block_middleware: [ConversationTitle]  # Only affects general-purpose
      ]}

  ## Usage Example

      # Main agent decides to delegate work
      "I need to research renewable energy. I'll use the researcher SubAgent."
      → Calls: task("Research renewable energy impacts", "researcher")

      # SubAgent executes independently
      # If SubAgent hits HITL interrupt (e.g., internet_search needs approval):
      #   1. SubAgent pauses
      #   2. Interrupt propagates to parent
      #   3. User sees: "SubAgent 'researcher' needs approval for 'internet_search'"
      #   4. User approves
      #   5. Parent resumes, which resumes SubAgent
      #   6. SubAgent completes and returns result

  ## Architecture

      Main Agent
        │
        ├─ task("research task", "researcher")
        │   │
        │   └─ SubAgent (as SubAgentServer process)
        │       ├─ Fresh conversation
        │       ├─ Specialized tools
        │       ├─ LLM executes
        │       └─ Returns final message only
        │
        └─ Receives result, continues

  ## HITL Interrupt Flow

      1. SubAgent hits HITL interrupt
      2. SubAgentServer.execute() returns {:interrupt, interrupt_data}
      3. Task tool receives interrupt
      4. Task tool returns {:interrupt, enhanced_data} to parent
      5. Parent agent propagates to AgentServer
      6. User approves
      7. Parent agent resumes
      8. Task tool calls SubAgentServer.resume(decisions)
      9. SubAgent continues and completes
  """

  @behaviour LangChain.Agents.Middleware

  require Logger

  alias LangChain.Agents.SubAgent
  alias LangChain.Agents.SubAgentServer
  alias LangChain.Agents.SubAgentsDynamicSupervisor
  alias LangChain.Function

  ## Middleware Callbacks

  @impl true
  def init(opts) do
    # Extract configuration
    subagents = Keyword.get(opts, :subagents, [])
    agent_id = Keyword.fetch!(opts, :agent_id)
    model = Keyword.fetch!(opts, :model)
    middleware = Keyword.get(opts, :middleware, [])
    block_middleware = Keyword.get(opts, :block_middleware, [])

    # Validate block_middleware entries (warn about potential issues)
    validate_block_middleware(block_middleware, middleware)

    # Build agent lookup map from subagent configs
    # Returns {:ok, %{"researcher" => agent_struct, "coder" => agent_struct}}
    # This is just a MAP for looking up which Agent to use, NOT a process Registry
    case SubAgent.build_agent_map(subagents, model, middleware) do
      {:ok, agent_map} ->
        # Build descriptions map for tool schema
        descriptions = SubAgent.build_descriptions(subagents)

        # Add "general-purpose" entry for dynamic subagent creation
        # This special marker enables runtime tool inheritance
        agent_map_with_general = Map.put(agent_map, "general-purpose", :dynamic)

        descriptions_with_general =
          Map.put(
            descriptions,
            "general-purpose",
            "General-purpose subagent for complex, multi-step tasks. " <>
              "Inherits all tools and middleware from parent agent. " <>
              "Use when you need to delegate independent work that can run in isolation."
          )

        config = %{
          agent_map: agent_map_with_general,
          descriptions: descriptions_with_general,
          agent_id: agent_id,
          model: model,
          block_middleware: block_middleware
        }

        {:ok, config}

      {:error, reason} ->
        {:error, "Failed to build subagent lookup map: #{reason}"}
    end
  end

  @impl true
  def system_prompt(_config) do
    """
    ## SubAgent Delegation

    You have access to a `task` tool for delegating work to specialized SubAgents.

    **Use SubAgents when:**
    - Task is complex and multi-step
    - Task can be fully delegated in isolation
    - You only care about the final result
    - Heavy context/token usage would benefit from isolation

    **Do NOT use SubAgents when:**
    - Task is trivial (single tool call)
    - You need to see intermediate reasoning
    - Task requires iterative back-and-forth

    SubAgents have their own conversation context and will work independently
    to complete the task. You will receive only their final result.
    """
  end

  @impl true
  def tools(config) do
    [build_task_tool(config)]
  end

  ## Private Functions - Tool Building

  defp build_task_tool(config) do
    # Get list of available subagent names from the lookup map
    subagent_names = config.agent_map |> Map.keys()

    # Build description with available subagents
    description = build_task_description(config.descriptions)

    Function.new!(%{
      name: "task",
      description: description,
      parameters_schema: %{
        type: "object",
        required: ["instructions", "subagent_type"],
        properties: %{
          "instructions" => %{
            type: "string",
            description:
              "Detailed instructions for what the SubAgent should accomplish. Be specific about the task, expected output, and any context needed."
          },
          "subagent_type" => %{
            type: "string",
            enum: subagent_names,
            description: "Which specialized SubAgent to use for this task"
          },
          "system_prompt" => %{
            type: "string",
            description:
              "Optional custom system prompt to define how the SubAgent should behave. " <>
                "Only applicable for 'general-purpose' type. Defines role, capabilities, and constraints. " <>
                "If omitted, a default general-purpose prompt will be used."
          }
        }
      },
      function: fn args, context ->
        execute_task(args, context, config)
      end,
      # Allow multiple SubAgents to run in parallel
      async: true
    })
  end

  defp build_task_description(descriptions) do
    base = "Delegate a task to a specialized SubAgent.\n\nAvailable SubAgents:\n"

    subagent_list =
      descriptions
      |> Enum.map(fn {name, desc} -> "- #{name}: #{desc}" end)
      |> Enum.join("\n")

    base <> subagent_list
  end

  ## Private Functions - Task Execution

  defp execute_task(args, context, config) do
    instructions = Map.fetch!(args, "instructions")
    subagent_type = Map.fetch!(args, "subagent_type")

    # Check if we're resuming an existing SubAgent
    case get_resume_context(context) do
      {:resume, sub_agent_id} ->
        # Resume existing SubAgent with decisions
        resume_subagent(sub_agent_id, context)

      :new ->
        # Start new SubAgent (pass full args for system_prompt support)
        start_subagent(instructions, subagent_type, args, context, config)
    end
  end

  defp get_resume_context(context) do
    # Check if this is a resume operation
    # The context will contain resume info from Agent.resume
    case Map.get(context, :resume_info) do
      %{sub_agent_id: sub_agent_id} ->
        {:resume, sub_agent_id}

      _ ->
        :new
    end
  end

  @doc """
  Starts and executes a new SubAgent to delegate work.

  This function allows custom tools and middleware to spawn SubAgents for
  delegating complex, multi-step tasks, similar to how the built-in `task` tool
  works. The SubAgent runs as an isolated, supervised process with its own
  conversation context.

  ## Parameters

  - `instructions` - Detailed instructions for what the SubAgent should
    accomplish. Be specific about the task, expected output, and any context
    needed.

  - `subagent_type` - The name/type of SubAgent to use. Must match a configured
    SubAgent name (from middleware init) or "general-purpose" for dynamic
    SubAgents.

  - `args` - Full arguments map containing:
    - `"instructions"` (required) - Same as instructions parameter
    - `"subagent_type"` (required) - Same as subagent_type parameter
    - `"system_prompt"` (optional) - Custom system prompt for general-purpose
      SubAgents

  - `context` - Tool execution context map containing:
    - `:agent_id` - Parent agent ID
    - `:state` - Parent agent state
    - `:parent_middleware` - Parent middleware list (for general-purpose
      SubAgents)
    - `:resume_info` - Resume information if continuing interrupted SubAgent

  - `config` - Middleware configuration map containing:
    - `:agent_map` - Map of subagent_type -> Agent struct
    - `:descriptions` - Map of subagent_type -> description string
    - `:agent_id` - Parent agent ID
    - `:model` - Model configuration

  ## Returns

  - `{:ok, result}` - SubAgent completed successfully, returns final message
    content
  - `{:interrupt, interrupt_data}` - SubAgent hit HITL interrupt, needs approval
  - `{:error, reason}` - Failed to start or execute SubAgent

  ## Example

  Using from a custom tool function:

      def my_research_tool_function(args, context) do
        # Build config from middleware state
        subagent_config = %{
          agent_map: context.subagent_map,
          descriptions: context.subagent_descriptions,
          agent_id: context.agent_id,
          model: context.model
        }

        # Prepare arguments
        task_args = %{
          "instructions" => "Research quantum computing developments",
          "subagent_type" => "researcher"
        }

        # Start SubAgent
        case SubAgent.start_subagent(
          "Research quantum computing developments",
          "researcher",
          task_args,
          context,
          subagent_config
        ) do
          {:ok, result} ->
            {:ok, "Research complete: " <> result}

          {:interrupt, interrupt_data} ->
            # Propagate interrupt to parent
            {:interrupt, interrupt_data}

          {:error, reason} ->
            {:error, "Failed to research: " <> reason}
        end
      end

  ## Notes

  - SubAgents run in isolated process contexts with their own conversation
    history
  - Parent only sees final result, not intermediate reasoning (token efficient)
  - HITL interrupts from SubAgents automatically propagate to parent
  - For "general-purpose" type, tools and middleware are inherited from parent
  - SubAgents are supervised and cleaned up automatically
  """
  @spec start_subagent(String.t(), String.t(), map(), map(), map()) ::
          {:ok, String.t()} | {:interrupt, map()} | {:error, String.t()}
  def start_subagent(instructions, subagent_type, args, context, config) do
    Logger.debug("Starting SubAgent: #{subagent_type}")

    # Get agent from lookup map
    case Map.fetch(config.agent_map, subagent_type) do
      {:ok, :dynamic} ->
        # Handle "general-purpose" dynamic subagent with tool inheritance
        start_dynamic_subagent(instructions, args, context, config)

      {:ok, agent_config} ->
        # Create SubAgent struct from pre-configured agent
        # Check if it's a Compiled struct (with initial_messages) or just an Agent
        subagent =
          case agent_config do
            %SubAgent.Compiled{} = compiled ->
              # Use new_from_compiled to include initial_messages
              SubAgent.new_from_compiled(
                parent_agent_id: config.agent_id,
                instructions: instructions,
                compiled_agent: compiled.agent,
                initial_messages: compiled.initial_messages || []
              )

            agent ->
              # Regular Agent struct from Config
              SubAgent.new_from_config(
                parent_agent_id: config.agent_id,
                instructions: instructions,
                agent_config: agent
              )
          end

        # Get supervisor for this parent agent
        # Uses existing LangChain.Agents.Registry for process lookup
        supervisor_name = SubAgentsDynamicSupervisor.get_name(config.agent_id)

        # Spawn SubAgentServer under supervision
        # SubAgentServer will register itself in LangChain.Agents.Registry
        child_spec = %{
          id: subagent.id,
          start: {SubAgentServer, :start_link, [[subagent: subagent]]},
          # Don't restart on crash
          restart: :temporary
        }

        try do
          case DynamicSupervisor.start_child(supervisor_name, child_spec) do
            {:ok, _pid} ->
              # Execute SubAgent synchronously (blocks until complete or interrupt)
              execute_subagent(subagent.id, subagent_type)

            {:error, reason} ->
              {:error, "Failed to start SubAgent: #{inspect(reason)}"}
          end
        catch
          :exit, reason ->
            {:error, "Failed to start SubAgent: #{inspect(reason)}"}
        end

      :error ->
        {:error, "Unknown SubAgent type: #{subagent_type}"}
    end
  end

  ## Starting Dynamic SubAgent (general-purpose with tool inheritance)

  defp start_dynamic_subagent(instructions, args, context, config) do
    Logger.debug("Starting dynamic general-purpose SubAgent")

    # Extract parent capabilities from context (set by Agent.build_chain)
    parent_middleware = Map.get(context, :parent_middleware, [])

    # Get optional custom system prompt, or use default
    system_prompt = Map.get(args, "system_prompt", default_general_purpose_prompt())

    # Validate system prompt
    case validate_system_prompt(system_prompt) do
      :ok ->
        # Filter middleware using block list from config
        filtered_middleware =
          SubAgent.subagent_middleware_stack(
            parent_middleware,
            [],
            block_middleware: Map.get(config, :block_middleware, [])
          )

        # Convert MiddlewareEntry structs back to raw middleware specs
        # parent_middleware contains initialized MiddlewareEntry structs, but Agent.new!
        # expects raw middleware specs (module or {module, opts} tuples)
        raw_middleware_specs = LangChain.Agents.MiddlewareEntry.to_raw_specs(filtered_middleware)

        # Build Agent struct with inherited middleware capabilities
        # Do NOT pass parent_tools - let filtered_middleware provide tools naturally
        # This ensures SubAgent "task" tool is not inherited after filtering out SubAgent middleware
        agent_config =
          LangChain.Agents.Agent.new!(
            %{
              model: config.model,
              base_system_prompt: system_prompt,
              middleware: raw_middleware_specs
            },
            replace_default_middleware: true,
            interrupt_on: nil
          )

        # Create SubAgent struct
        subagent =
          SubAgent.new_from_config(
            parent_agent_id: config.agent_id,
            instructions: instructions,
            agent_config: agent_config
          )

        # Get supervisor and start SubAgent (same as pre-configured)
        supervisor_name = SubAgentsDynamicSupervisor.get_name(config.agent_id)

        child_spec = %{
          id: subagent.id,
          start: {SubAgentServer, :start_link, [[subagent: subagent]]},
          restart: :temporary
        }

        try do
          case DynamicSupervisor.start_child(supervisor_name, child_spec) do
            {:ok, _pid} ->
              execute_subagent(subagent.id, "general-purpose")

            {:error, reason} ->
              {:error, "Failed to start dynamic SubAgent: #{inspect(reason)}"}
          end
        catch
          :exit, reason ->
            {:error, "Failed to start dynamic SubAgent: #{inspect(reason)}"}
        end

      {:error, reason} ->
        {:error, "Invalid system_prompt: #{reason}"}
    end
  end

  defp default_general_purpose_prompt() do
    """
    You are a general-purpose assistant SubAgent. You have access to tools. Focus on completing the specific task you've been given.
    Return a clear, concise result suitable for the parent agent to use.
    """
  end

  defp validate_system_prompt(system_prompt) when is_binary(system_prompt) do
    cond do
      String.length(system_prompt) == 0 ->
        {:error, "system_prompt cannot be empty"}

      String.length(system_prompt) > 10_000 ->
        {:error, "system_prompt too long (max 10,000 characters)"}

      contains_potential_injection?(system_prompt) ->
        {:error, "system_prompt contains potentially unsafe content"}

      true ->
        :ok
    end
  end

  defp validate_system_prompt(_), do: {:error, "system_prompt must be a string"}

  # Validate block_middleware entries and log warnings for potential issues
  # Note: We only validate that entries are atoms and loaded modules.
  # We don't check if they're in the parent middleware stack because that
  # information isn't available at init time - the actual parent middleware
  # is passed via context when creating subagents at runtime.
  defp validate_block_middleware(block_list, _middleware) when is_list(block_list) do
    for module <- block_list do
      cond do
        not is_atom(module) ->
          Logger.warning(
            "[SubAgent] block_middleware entry #{inspect(module)} is not a module atom"
          )

        not Code.ensure_loaded?(module) ->
          Logger.warning(
            "[SubAgent] block_middleware module #{inspect(module)} is not loaded"
          )

        true ->
          :ok
      end
    end

    :ok
  end

  defp validate_block_middleware(block_list, _parent_middleware) do
    Logger.warning("[SubAgent] block_middleware must be a list, got: #{inspect(block_list)}")
    :ok
  end

  # Basic safety check for prompt injection patterns
  defp contains_potential_injection?(text) do
    # Check for common prompt injection patterns
    dangerous_patterns = [
      ~r/ignore\s+(all\s+)?previous\s+instructions/i,
      ~r/disregard\s+(all\s+)?previous\s+instructions/i,
      ~r/forget\s+(all\s+)?previous\s+instructions/i,
      ~r/new\s+instructions:/i,
      ~r/system\s*:\s*you\s+are\s+now/i
    ]

    Enum.any?(dangerous_patterns, fn pattern ->
      Regex.match?(pattern, text)
    end)
  end

  defp execute_subagent(sub_agent_id, subagent_type) do
    Logger.debug("Executing SubAgent: #{sub_agent_id}")

    case SubAgentServer.execute(sub_agent_id) do
      {:ok, final_result} ->
        # SubAgent completed successfully
        Logger.debug("SubAgent #{sub_agent_id} completed")
        {:ok, final_result}

      {:interrupt, interrupt_data} ->
        # SubAgent needs HITL approval
        # Propagate interrupt to parent with enhanced metadata
        Logger.info("SubAgent '#{subagent_type}' interrupted for HITL")

        {:interrupt,
         %{
           type: :subagent_hitl,
           sub_agent_id: sub_agent_id,
           subagent_type: subagent_type,
           interrupt_data: interrupt_data
         }}

      {:error, reason} ->
        Logger.error("SubAgent #{sub_agent_id} failed: #{inspect(reason)}")
        {:error, "SubAgent execution failed: #{inspect(reason)}"}
    end
  end

  ## Resuming Existing SubAgent

  defp resume_subagent(sub_agent_id, context) do
    Logger.debug("Resuming SubAgent: #{sub_agent_id}")

    # Extract decisions from resume context
    decisions = Map.get(context.resume_info, :decisions, [])
    subagent_type = Map.get(context.resume_info, :subagent_type, "unknown")

    case SubAgentServer.resume(sub_agent_id, decisions) do
      {:ok, final_result} ->
        # SubAgent completed after approval
        Logger.debug("SubAgent #{sub_agent_id} completed after resume")
        {:ok, final_result}

      {:interrupt, interrupt_data} ->
        # Another interrupt (e.g., SubAgent needs approval for another tool)
        Logger.info("SubAgent '#{subagent_type}' interrupted again")

        {:interrupt,
         %{
           type: :subagent_hitl,
           sub_agent_id: sub_agent_id,
           subagent_type: subagent_type,
           interrupt_data: interrupt_data
         }}

      {:error, reason} ->
        Logger.error("SubAgent #{sub_agent_id} resume failed: #{inspect(reason)}")
        {:error, "SubAgent resume failed: #{inspect(reason)}"}
    end
  end
end
