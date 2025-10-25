defmodule LangChain.DeepAgents.Middleware do
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
        @behaviour LangChain.DeepAgents.Middleware

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
  alias LangChain.DeepAgents.State

  @type config :: keyword()
  @type middleware_config :: any()
  @type middleware_result :: {:ok, State.t()} | {:error, term()}

  @doc """
  Initialize middleware with configuration options.

  Called once when the middleware is added to an agent. Returns configuration
  that will be passed to other callbacks.

  Defaults to returning `{:ok, opts}` if not implemented.
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

  - `state` - The current `LangChain.DeepAgents.State` struct
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

  - `state` - The current `LangChain.DeepAgents.State` struct (with LLM response)
  - `config` - The middleware configuration from `init/1`

  ## Returns

  - `{:ok, updated_state}` - Success with potentially modified state
  - `{:error, reason}` - Failure, halts execution
  """
  @callback after_model(State.t(), middleware_config) :: middleware_result()

  @doc """
  Provide the state schema module for this middleware.

  If the middleware needs to add fields to the agent state, it should
  return a module that defines those fields.

  Defaults to `nil` if not implemented.
  """
  @callback state_schema() :: module() | nil

  @optional_callbacks [
    init: 1,
    system_prompt: 1,
    tools: 1,
    before_model: 2,
    after_model: 2,
    state_schema: 0
  ]

  @doc """
  Normalize middleware specification to {module, config} tuple.
  """
  def normalize(middleware) when is_atom(middleware) do
    {middleware, []}
  end

  def normalize({module, opts}) when is_atom(module) and is_list(opts) do
    {module, opts}
  end

  def normalize(middleware) do
    raise ArgumentError,
          "Invalid middleware specification: #{inspect(middleware)}. " <>
            "Expected module or {module, opts} tuple."
  end

  @doc """
  Initialize a middleware module with its configuration.
  """
  def init_middleware(middleware) do
    {module, opts} = normalize(middleware)

    config =
      try do
        case module.init(opts) do
          {:ok, config} -> config
          {:error, reason} -> raise "Failed to initialize #{module}: #{inspect(reason)}"
        end
      rescue
        UndefinedFunctionError -> opts
      end

    {module, config}
  end

  @doc """
  Get system prompt from middleware.
  """
  def get_system_prompt({module, config}) do
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
  def get_tools({module, config}) do
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
  - `initialized_middleware` - Tuple of `{module, config}` from `init_middleware/1`

  ## Returns

  - `{:ok, updated_state}` - Success with potentially modified state
  - `{:error, reason}` - Error from middleware
  """
  @spec apply_before_model(State.t(), {module(), middleware_config()}) ::
          middleware_result()
  def apply_before_model(state, {module, config}) do
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
  - `initialized_middleware` - Tuple of `{module, config}` from `init_middleware/1`

  ## Returns

  - `{:ok, updated_state}` - Success with potentially modified state
  - `{:error, reason}` - Error from middleware
  """
  @spec apply_after_model(State.t(), {module(), middleware_config()}) ::
          middleware_result()
  def apply_after_model(state, {module, config}) do
    try do
      module.after_model(state, config)
    rescue
      UndefinedFunctionError -> {:ok, state}
    end
  end
end
