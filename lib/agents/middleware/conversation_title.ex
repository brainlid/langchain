defmodule LangChain.Agents.Middleware.ConversationTitle do
  @moduledoc """
  Middleware that automatically generates conversation titles based on the initial user message.

  Spawns an async task after the first user message to generate a concise title
  using an LLM. The title is stored in state metadata and broadcast to subscribers.

  ## Usage

      {:ok, agent} = Agent.new(
        llm: main_model,
        middleware: [
          {ConversationTitle, [
            chat_model: ChatAnthropic.new!(%{model: "claude-3-5-haiku-latest"}),
            fallbacks: [backup_model]
          ]}
        ]
      )

  ## Configuration Options

  - `:chat_model` (required) - The LLM model to use for title generation
  - `:fallbacks` - List of fallback models if primary fails (default: [])
  - `:prompt_template` - Custom prompt template (default: uses TextToTitleChain defaults)
  - `:examples` - List of example titles to guide LLM (default: provided examples)
  - `:id` - Custom middleware ID for multiple instances (default: module name)

  ## Events

  Broadcasts the following events to PubSub:

  - `{:conversation_title_generated, title, agent_id}` - For backward compatibility with LiveView
  - `{:agent_state_update, %{conversation_title: title}}` - Generic agent state update notification

  ## Metadata

  Stores the generated title in state metadata:

  - `"conversation_title"` - The generated title string (presence indicates title has been generated)
  """

  @behaviour LangChain.Agents.Middleware

  require Logger
  alias LangChain.Agents.State
  alias LangChain.Chains.TextToTitleChain
  alias LangChain.Message
  alias LangChain.Agents.AgentServer
  alias LangChain.Agents.State

  @default_examples [
    "Debug payment processing error",
    "Plan quarterly team objectives",
    "Analyze website traffic patterns"
  ]

  @doc """
  Initialize the ConversationTitle middleware with configuration options.

  ## Required Options

  - `:chat_model` - The ChatModel to use for title generation

  ## Optional Options

  - `:fallbacks` - List of fallback ChatModels if primary fails (default: `[]`)
  - `:prompt_template` - Custom prompt template for title generation (default: uses TextToTitleChain defaults)
  - `:examples` - List of example titles to guide the LLM (default: provided examples)
  - `:id` - Custom middleware ID for multiple instances (default: module name)

  ## Returns

  - `{:ok, config}` - Configuration map to be passed to other callbacks
  - `{:error, reason}` - Initialization failed

  ## Example

      {:ok, config} = ConversationTitle.init([
        chat_model: ChatAnthropic.new!(%{model: "claude-3-5-haiku-latest"}),
        fallbacks: [backup_model],
        examples: ["Debug API error", "Plan sprint goals"]
      ])
  """
  @impl true
  def init(opts) do
    # Extract configuration options
    chat_model = Keyword.get(opts, :chat_model)
    fallbacks = Keyword.get(opts, :fallbacks, [])
    prompt_template = Keyword.get(opts, :prompt_template)
    examples = Keyword.get(opts, :examples, @default_examples)
    middleware_id = Keyword.get(opts, :id)

    # Validate required options
    unless chat_model do
      {:error, "ConversationTitle middleware requires :chat_model option"}
    else
      config = %{
        chat_model: chat_model,
        fallbacks: fallbacks,
        prompt_template: prompt_template,
        examples: examples
      }

      # Add custom ID if provided
      config =
        if middleware_id do
          Map.put(config, :id, middleware_id)
        else
          config
        end

      {:ok, config}
    end
  end

  @doc """
  Called after the LLM model completes its response.

  Checks if a conversation title has already been generated. If not, spawns
  an async task to generate one based on the user's message content.

  The async task will send a message back to the AgentServer when complete,
  which will be handled by `handle_message/3`.

  ## Parameters

  - `state` - The current agent state after model completion
  - `config` - The middleware configuration from `init/1`

  ## Returns

  - `{:ok, state}` - Always returns the unchanged state (title generation happens asynchronously)
  """
  @impl true
  def after_model(state, config) do
    # Simple logic: if we don't have a title yet, generate one
    if State.get_metadata(state, "conversation_title") == nil do
      spawn_title_generation_task(state, config)
    end

    {:ok, state}
  end

  @doc """
  Handle messages from async title generation tasks.

  This callback receives messages sent by the async task spawned in `after_model/2`.
  It handles both success and failure cases.

  ## Success Message: `{:title_generated, title}`

  When a title is successfully generated:
  - Stores the title in state metadata under the key `"conversation_title"`
  - Requests a broadcast to notify subscribers (LiveViews, external clients)

  ## Failure Message: `{:title_generation_failed, reason}`

  When title generation fails:
  - Logs a warning with the failure reason
  - Does not update state or broadcast

  ## Parameters

  - `message` - The message tuple from the async task
  - `state` - The current agent state
  - `config` - The middleware configuration

  ## Returns

  - `{:ok, updated_state}` - For successful title generation
  - `{:ok, state}` - For failures (no state changes)
  """
  @impl true
  def handle_message({:title_generated, title}, %State{} = state, _config) do
    Logger.debug("Received title: #{title}")
    # Update the Agent state with the generated title
    updated_state = State.put_metadata(state, "conversation_title", title)

    {:ok, updated_state}
  end

  def handle_message({:title_generation_failed, reason}, state, _config) do
    Logger.warning("Title generation failed: #{inspect(reason)}")
    # Don't update state, just log the error
    {:ok, state}
  end

  ## Private Functions

  # Spawn async task to generate title
  defp spawn_title_generation_task(state, config) do
    # Capture AgentServer PID (self() in middleware callback context)
    server_pid = self()

    # TODO: THE MIDDLEWARE ID shouldn't reading with default shouldn't be done here like this. This is business logic to me. It should be the ID. The ID should be set correctly earlier. Or it should use a function so the default behavior is more consistent.

    # Get the middleware ID for message routing
    middleware_id = Map.get(config, :id, __MODULE__)

    # Extract text from the last user message (the user's actual first message)
    user_text = extract_last_user_message_text(state)

    # Get agent_id from state for event broadcasting
    # This should always be present - if nil, it indicates improper state deserialization
    agent_id = state.agent_id

    unless agent_id do
      Logger.error("""
      ConversationTitle middleware: state.agent_id is nil!
      This indicates the state was not properly initialized or deserialized.
      When deserializing state, ensure you call: State.from_serialized(agent_id, data)
      """)

      raise ArgumentError,
            "state.agent_id is required for ConversationTitle middleware. " <>
              "Ensure State.from_serialized(agent_id, data) is called when deserializing."
    end

    # Emit telemetry for task spawn
    :telemetry.execute(
      [:middleware, :task, :spawned],
      %{count: 1},
      %{middleware: middleware_id, task_type: :title_generation}
    )

    Task.start(fn ->
      try do
        # Publish debug event for title generation start
        AgentServer.publish_debug_event_from(
          agent_id,
          {:middleware_action, __MODULE__, {:title_generation_started, String.slice(user_text, 0, 100)}}
        )

        # Generate title using TextToTitleChain
        title = generate_title(user_text, config)

        # Publish debug event for successful completion
        AgentServer.publish_debug_event_from(
          agent_id,
          {:middleware_action, __MODULE__, {:title_generation_completed, title}}
        )

        # Emit telemetry for successful completion
        :telemetry.execute(
          [:middleware, :task, :completed],
          %{count: 1},
          %{middleware: middleware_id, task_type: :title_generation}
        )

        # Send success message back to AgentServer
        AgentServer.send_middleware_message(agent_id, middleware_id, {:title_generated, title})


        # Publish the event
        AgentServer.publish_event_from(agent_id, {:conversation_title_generated, title, agent_id})
      rescue
        error ->
          stacktrace = __STACKTRACE__

          # Publish debug event for failure
          AgentServer.publish_debug_event_from(
            agent_id,
            {:middleware_action, __MODULE__, {:title_generation_failed, inspect(error)}}
          )

          Logger.error(
            "Title generation task failed: #{inspect(error)}\n#{Exception.format_stacktrace(stacktrace)}"
          )

          # Emit telemetry for failure
          :telemetry.execute(
            [:middleware, :task, :failed],
            %{count: 1},
            %{middleware: middleware_id, task_type: :title_generation, error: inspect(error)}
          )

          # Send failure message back to AgentServer
          send(
            server_pid,
            {:middleware_message, middleware_id, {:title_generation_failed, error}}
          )
      catch
        kind, reason ->
          stacktrace = __STACKTRACE__

          # Publish debug event for crash
          AgentServer.publish_debug_event_from(
            agent_id,
            {:middleware_action, __MODULE__, {:title_generation_failed, "#{kind}: #{inspect(reason)}"}}
          )

          Logger.error(
            "Title generation task crashed (#{kind}): #{inspect(reason)}\n#{Exception.format_stacktrace(stacktrace)}"
          )

          :telemetry.execute(
            [:middleware, :task, :failed],
            %{count: 1},
            %{
              middleware: middleware_id,
              task_type: :title_generation,
              error: "#{kind}: #{inspect(reason)}"
            }
          )

          send(
            server_pid,
            {:middleware_message, middleware_id, {:title_generation_failed, {kind, reason}}}
          )
      end
    end)
  end

  # Generate title using TextToTitleChain
  defp generate_title(user_text, config) do
    chain_config = %{
      llm: config.chat_model,
      input_text: user_text,
      examples: config.examples
    }

    # Add custom prompt template if provided
    chain_config =
      if config.prompt_template do
        Map.put(chain_config, :override_system_prompt, config.prompt_template)
      else
        chain_config
      end

    # Build chain and evaluate
    chain_config
    |> TextToTitleChain.new!()
    |> TextToTitleChain.evaluate(with_fallbacks: config.fallbacks)
  end

  # Extract text from the last user message
  # This handles pre-loaded conversations where earlier user messages might be examples
  defp extract_last_user_message_text(state) do
    state.messages
    |> Enum.reverse()
    |> Enum.find(fn msg -> msg.role == :user end)
    |> case do
      nil ->
        ""

      message ->
        Message.ContentPart.parts_to_string(message.content)
    end
  end
end
