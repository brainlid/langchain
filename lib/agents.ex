defmodule LangChain.Agents do
  @moduledoc """
  Agents provides hierarchical agent capabilities with composable middleware.

  Agents extends LangChain with powerful features:

  - **Middleware System**: Composable components for agent capabilities
  - **TODO Management**: Task planning and progress tracking
  - **Mock Filesystem**: File operations for agent workflows
  - **Task Delegation**: Hierarchical sub-agents for complex tasks
  - **Context Management**: Automatic summarization and optimization

  ## Quick Start

      alias LangChain.Agents
      alias LangChain.ChatModels.ChatAnthropic

      # Create an agent
      {:ok, agent} = Agents.new(
        model: ChatAnthropic.new!(%{model: "claude-3-5-sonnet-20241022"}),
        system_prompt: "You are a helpful assistant."
      )

      # Execute with messages
      {:ok, result} = Agents.execute(agent, [
        %{role: "user", content: "Hello!"}
      ])

  ## Middleware Composition

  Agents uses a middleware pattern for extensibility:

      # Use default middleware (TODO, Filesystem, SubAgent, etc.)
      {:ok, agent} = Agents.new(
        model: model,
        middleware: [MyCustomMiddleware]
      )

      # Customize default middleware
      {:ok, agent} = Agents.new(
        model: model,
        filesystem_opts: [long_term_memory: true]
      )

      # Provide complete middleware stack
      {:ok, agent} = Agents.new(
        model: model,
        replace_default_middleware: true,
        middleware: [MyMiddleware1, MyMiddleware2]
      )

  ## Creating Custom Middleware

      defmodule MyMiddleware do
        @behaviour LangChain.Agents.Middleware

        @impl true
        def init(opts) do
          {:ok, %{enabled: Keyword.get(opts, :enabled, true)}}
        end

        @impl true
        def system_prompt(_config) do
          "Custom instructions for the agent."
        end

        @impl true
        def tools(_config) do
          [my_custom_tool()]
        end

        @impl true
        def before_model(state, _config) do
          # Preprocess state before LLM
          {:ok, state}
        end

        @impl true
        def after_model(state, _config) do
          # Postprocess after LLM response
          {:ok, state}
        end
      end

  ## State Management

  Agent state flows through middleware and execution:

      state = State.new!(%{
        messages: [%{role: "user", content: "Hello"}],
        files: %{"/notes.txt" => "content"},
        metadata: %{session_id: "123"}
      })

      {:ok, result_state} = Agents.execute(agent, state)

  See `LangChain.Agents.State` for state management functions.
  """

  alias LangChain.Agents.Agent, as: Agent
  alias LangChain.Agents.State, as: State

  @doc """
  Create a new DeepAgent with default middleware stack.

  This is a convenience function that delegates to `Agent.new/1` with
  sensible defaults for common use cases.

  ## Options

  All options from `Agent.new/1` are supported. Common options:

  - `:model` - LangChain ChatModel struct (required)
  - `:system_prompt` - Base system instructions
  - `:tools` - Additional tools beyond middleware
  - `:middleware` - Extra middleware (appended to defaults)
  - `:replace_default_middleware` - Use only provided middleware (default: false)
  - `:name` - Agent name for identification

  ## Examples

      # Basic agent
      {:ok, agent} = Agents.new(
        model: ChatAnthropic.new!(%{model: "claude-3-5-sonnet-20241022"}),
        system_prompt: "You are helpful."
      )

      # With custom tools
      {:ok, agent} = Agents.new(
        model: model,
        tools: [calculator_tool, search_tool]
      )

      # With custom middleware
      {:ok, agent} = Agents.new(
        model: model,
        middleware: [LoggingMiddleware, MetricsMiddleware]
      )
  """
  defdelegate new(opts \\ []), to: Agent

  @doc """
  Create a new DeepAgent, raising on error.

  See `new/1` for options.
  """
  defdelegate new!(opts \\ []), to: Agent

  @doc """
  Execute an agent with the given state.

  ## Examples

      # Execute with State struct
      state = State.new!(%{messages: [%{role: "user", content: "Hello"}]})
      {:ok, result_state} = Agents.execute(agent, state)

      # Execute with message list (convenience)
      {:ok, result_state} = Agents.execute(agent, [
        %{role: "user", content: "Hello"}
      ])
  """
  def execute(agent, %State{} = state) do
    Agent.execute(agent, state)
  end

  def execute(agent, messages) when is_list(messages) do
    state = State.new!(%{messages: messages})
    Agent.execute(agent, state)
  end

  @doc """
  Execute an agent asynchronously.

  Returns a Task that can be awaited.

  ## Examples

      task = Agents.execute_async(agent, messages)
      {:ok, result_state} = Task.await(task)
  """
  def execute_async(agent, state_or_messages) do
    Task.async(fn -> execute(agent, state_or_messages) end)
  end
end
