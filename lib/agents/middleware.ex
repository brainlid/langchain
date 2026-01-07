defmodule LangChain.Agents.Middleware do
  @moduledoc """
  Behavior for DeepAgent middleware components.

  Middleware provides a composable pattern for adding capabilities to agents. Each
  middleware component can contribute:

  - System prompt additions
  - Tools (Functions)
  - State schema modifications
  - Pre/post processing hooks

  ## Middleware Lifecycle

  1. **Initialization** - `init/1` is called when middleware is configured
  2. **Tool Collection** - `tools/1` provides tools to add to the agent
  3. **Prompt Assembly** - `system_prompt/1` contributes to the system prompt
  4. **Before Model** - `before_model/2` preprocesses state before LLM call
  5. **After Model** - `after_model/2` postprocesses state after LLM response

  ## Example

      defmodule MyMiddleware do
        @behaviour LangChain.Agents.Middleware

        @impl true
        def init(opts) do
          config = %{enabled: Keyword.get(opts, :enabled, true)}
          {:ok, config}
        end

        @impl true
        def system_prompt(_config) do
          "You have access to custom capabilities."
        end

        @impl true
        def tools(_config) do
          [my_custom_tool()]
        end

        @impl true
        def before_model(state, _config) do
          # Preprocess state
          {:ok, state}
        end
      end

  ## Middleware Configuration

  Middleware can be specified as:

  - Module name: `MyMiddleware`
  - Tuple with options: `{MyMiddleware, [enabled: true]}`
  """
  alias LangChain.Agents.State
  alias LangChain.Agents.MiddlewareEntry

  @type config :: keyword()
  @type middleware_config :: any()
  @type middleware_result ::
          {:ok, State.t()} | {:interrupt, State.t(), any()} | {:error, term()}

  @doc """
  Initialize middleware with configuration options.

  Called once when the middleware is added to an agent. Returns configuration
  that will be passed to other callbacks.

  ## Convention

  - Input: `opts` as keyword list
  - Output: `config` as map for efficient runtime access

  Defaults to converting opts to a map if not implemented.

  ## Example

      def init(opts) do
        config = %{
          enabled: Keyword.get(opts, :enabled, true),
          max_retries: Keyword.get(opts, :max_retries, 3)
        }
        {:ok, config}
      end
  """
  @callback init(config) :: {:ok, middleware_config} | {:error, term()}

  @doc """
  Provide system prompt text for this middleware.

  Can return a single string or list of strings that will be joined.

  Defaults to empty string if not implemented.
  """
  @callback system_prompt(middleware_config) :: String.t() | [String.t()]

  @doc """
  Provide tools (Functions) that this middleware adds to the agent.

  Defaults to empty list if not implemented.
  """
  @callback tools(middleware_config) :: [LangChain.Function.t()]

  @doc """
  Process state before it's sent to the LLM.

  Receives the current agent state and can modify messages, add context, or
  perform validation before the LLM is invoked.

  Defaults to `{:ok, state}` if not implemented.

  ## Parameters

  - `state` - The current `LangChain.Agents.State` struct
  - `config` - The middleware configuration from `init/1`

  ## Returns

  - `{:ok, updated_state}` - Success with potentially modified state
  - `{:error, reason}` - Failure, halts execution
  """
  @callback before_model(State.t(), middleware_config) :: middleware_result()

  @doc """
  Process state after receiving LLM response.

  Receives the state after the LLM has responded and can modify the response,
  extract information, or update state.

  Defaults to `{:ok, state}` if not implemented.

  ## Parameters

  - `state` - The current `LangChain.Agents.State` struct (with LLM response)
  - `config` - The middleware configuration from `init/1`

  ## Returns

  - `{:ok, updated_state}` - Success with potentially modified state
  - `{:interrupt, state, interrupt_data}` - Pause execution for human intervention
  - `{:error, reason}` - Failure, halts execution
  """
  @callback after_model(State.t(), middleware_config) :: middleware_result()

  @doc """
  Handle asynchronous messages sent to this middleware.

  Middleware can spawn async tasks that send messages back to the AgentServer using
  `send(server_pid, {:middleware_message, middleware_id, message})`. When such messages
  are received, they are routed to this callback.

  This enables patterns like:
  - Spawning background processing tasks
  - Receiving results from async operations
  - Updating state based on external events

  Defaults to `{:ok, state}` if not implemented.

  ## Parameters

  - `message` - The message payload sent from the async task
  - `state` - The current `LangChain.Agents.State` struct
  - `config` - The middleware configuration from `init/1`

  ## Returns

  - `{:ok, updated_state}` - Success with potentially modified state
  - `{:error, reason}` - Failure (logged but does not halt agent execution)

  ## Example

      def handle_message({:title_generated, title}, state, _config) do
        updated_state = State.put_metadata(state, "conversation_title", title)
        {:ok, updated_state}
      end

      def handle_message({:title_generation_failed, reason}, state, _config) do
        Logger.warning("Title generation failed: \#{inspect(reason)}")
        {:ok, state}
      end
  """
  @callback handle_message(message :: term(), State.t(), middleware_config) ::
              {:ok, State.t()}
              | {:error, term()}

  @doc """
  Provide the state schema module for this middleware.

  If the middleware needs to add fields to the agent state, it should
  return a module that defines those fields.

  Defaults to `nil` if not implemented.
  """
  @callback state_schema() :: module() | nil

  @doc """
  Called when the AgentServer starts or restarts.

  This allows middleware to perform initialization actions that require
  the AgentServer to be running, such as broadcasting initial state to
  subscribers (e.g., TODOs for UI display).

  Receives the current state and middleware config.
  Returns `{:ok, state}` (state is not typically modified here but could be).

  Defaults to `{:ok, state}` if not implemented.

  ## Parameters

  - `state` - The current `LangChain.Agents.State` struct
  - `config` - The middleware configuration from `init/1`

  ## Returns

  - `{:ok, state}` - Success (state typically unchanged)
  - `{:error, reason}` - Failure (logged but does not halt agent)

  ## Example

      def on_server_start(state, _config) do
        # Broadcast initial todos when AgentServer starts
        broadcast_todos(state.agent_id, state.todos)
        {:ok, state}
      end
  """
  @callback on_server_start(State.t(), middleware_config) :: {:ok, State.t()} | {:error, term()}

  @optional_callbacks [
    init: 1,
    system_prompt: 1,
    tools: 1,
    before_model: 2,
    after_model: 2,
    handle_message: 3,
    state_schema: 0,
    on_server_start: 2
  ]

  @doc """
  Normalize middleware specification to {module, config} tuple.

  Accepts:
  - Module atom: `MyMiddleware` -> `{MyMiddleware, []}`
  - Tuple with keyword list: `{MyMiddleware, [key: value]}` -> `{MyMiddleware, [key: value]}`
  """
  def normalize(middleware) when is_atom(middleware) do
    {middleware, []}
  end

  def normalize({module, opts}) when is_atom(module) and is_list(opts) do
    {module, opts}
  end

  def normalize({module, opts}) when is_atom(module) and is_map(opts) do
    # Convert map to keyword list for consistency
    {module, Map.to_list(opts)}
  end

  def normalize(middleware) do
    raise ArgumentError,
          "Invalid middleware specification: #{inspect(middleware)}. " <>
            "Expected module or {module, opts} tuple with keyword list options."
  end

  @doc """
  Initialize a middleware module with its configuration.
  Returns a MiddlewareEntry struct.

  ## Configuration Convention

  - Input `opts` should be a keyword list
  - Returned `config` should be a map for efficient runtime access
  """
  def init_middleware(middleware) do
    {module, opts} = normalize(middleware)

    config =
      try do
        case module.init(opts) do
          {:ok, config} when is_map(config) ->
            config

          {:ok, config} when is_list(config) ->
            # Convert keyword list to map for consistency
            Map.new(config)

          {:error, reason} ->
            raise "Failed to initialize #{module}: #{inspect(reason)}"
        end
      rescue
        UndefinedFunctionError ->
          # If no init/1, convert opts to map
          Map.new(opts)
      end

    # Determine middleware ID (use custom :id from config, or default to module name)
    middleware_id = Map.get(config, :id, module)

    %MiddlewareEntry{
      id: middleware_id,
      module: module,
      config: config
    }
  end

  @doc """
  Get system prompt from middleware.
  """
  def get_system_prompt(%MiddlewareEntry{module: module, config: config}) do
    try do
      case module.system_prompt(config) do
        prompts when is_list(prompts) -> Enum.join(prompts, "\n\n")
        prompt when is_binary(prompt) -> prompt
      end
    rescue
      UndefinedFunctionError -> ""
    end
  end

  @doc """
  Get tools from middleware.
  """
  def get_tools(%MiddlewareEntry{module: module, config: config}) do
    # Don't use function_exported? as it can return false positives
    # in test environments due to code reloading issues
    try do
      module.tools(config)
    rescue
      UndefinedFunctionError -> []
    end
  end

  @doc """
  Apply before_model hook from middleware.

  ## Parameters

  - `state` - The current agent state
  - `entry` - MiddlewareEntry struct with module and config

  ## Returns

  - `{:ok, updated_state}` - Success with potentially modified state
  - `{:error, reason}` - Error from middleware
  """
  @spec apply_before_model(State.t(), LangChain.Agents.MiddlewareEntry.t()) ::
          middleware_result()
  def apply_before_model(state, %MiddlewareEntry{module: module, config: config}) do
    try do
      module.before_model(state, config)
    rescue
      UndefinedFunctionError -> {:ok, state}
    end
  end

  @doc """
  Apply after_model hook from middleware.

  ## Parameters

  - `state` - The current agent state (with LLM response)
  - `entry` - MiddlewareEntry struct with module and config

  ## Returns

  - `{:ok, updated_state}` - Success with potentially modified state
  - `{:error, reason}` - Error from middleware
  """
  @spec apply_after_model(State.t(), LangChain.Agents.MiddlewareEntry.t()) ::
          middleware_result()
  def apply_after_model(state, %MiddlewareEntry{module: module, config: config}) do
    try do
      module.after_model(state, config)
    rescue
      UndefinedFunctionError -> {:ok, state}
    end
  end

  @doc """
  Apply handle_message callback from middleware.

  ## Parameters

  - `message` - The message payload to handle
  - `state` - The current agent state
  - `entry` - MiddlewareEntry struct with module and config

  ## Returns

  - `{:ok, updated_state}` - Success with potentially modified state
  - `{:error, reason}` - Error from middleware
  """
  @spec apply_handle_message(term(), State.t(), LangChain.Agents.MiddlewareEntry.t()) ::
          {:ok, State.t()} | {:error, term()}
  def apply_handle_message(message, state, %MiddlewareEntry{module: module, config: config}) do
    try do
      module.handle_message(message, state, config)
    rescue
      UndefinedFunctionError -> {:ok, state}
    end
  end

  @doc """
  Apply on_server_start callback from middleware.

  Called when the AgentServer starts to allow middleware to perform
  initialization actions like broadcasting initial state.

  ## Parameters

  - `state` - The current agent state
  - `entry` - MiddlewareEntry struct with module and config

  ## Returns

  - `{:ok, state}` - Success (state typically unchanged)
  - `{:error, reason}` - Error from middleware
  """
  @spec apply_on_server_start(State.t(), LangChain.Agents.MiddlewareEntry.t()) ::
          {:ok, State.t()} | {:error, term()}
  def apply_on_server_start(state, %MiddlewareEntry{module: module, config: config}) do
    try do
      module.on_server_start(state, config)
    rescue
      UndefinedFunctionError -> {:ok, state}
    end
  end
end
