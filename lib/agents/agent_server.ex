defmodule LangChain.Agents.AgentServer do
  @moduledoc """
  GenServer that wraps a DeepAgent and its State, managing execution lifecycle
  and broadcasting events via PubSub.

  The AgentServer provides:
  - Asynchronous agent execution
  - State management and tracking
  - Event broadcasting for UI updates
  - Human-in-the-loop interrupt handling

  ## Understanding agent_id

  The `agent_id` is a **runtime identifier** used for process management and
  inter-process communication. It serves several critical purposes:

  ### Process Registration
  The `agent_id` is used to construct a Registry key via `get_name(agent_id)`,
  which returns a `:via` tuple for GenServer registration:
  - Format: `{:via, Registry, {LangChain.Agents.Registry, {:agent_server,
    agent_id}}}`
  - Ensures only one AgentServer process exists per agent_id
  - Enables process lookup without maintaining PIDs

  ### PubSub Topics
  The `agent_id` forms the basis for PubSub topic construction:
  - Topic format: `"agent_server:\#{agent_id}"`
  - External clients subscribe using: `AgentServer.subscribe(agent_id)`
  - Events broadcast include: status changes, LLM deltas, todos updates

  ### Middleware Context
  The `agent_id` is passed to middleware during initialization, enabling:
  - Coordination with agent-specific services (FileSystemServer,
    SubAgentsDynamicSupervisor)
  - Parent-child relationship establishment in SubAgent hierarchies
  - Per-agent resource isolation (virtual filesystems, etc.)

  ### Supervision Tree Coordination
  The `agent_id` flows through the entire supervision tree via AgentSupervisor,
  ensuring all child processes (FileSystemServer, AgentServer,
  SubAgentsDynamicSupervisor) are coordinated under the same agent context.

  ### What agent_id IS NOT

  **Not Part of Conversation State**: The `agent_id` is NOT included in
  serialized state (via `export_state/1`). It's a runtime identifier, not
  conversation data. This separation provides important benefits:

  - **Flexibility**: Restore the same conversation state under a different
    `agent_id`
  - **State Cloning**: Clone conversations for testing or forking scenarios
  - **Clean Architecture**: Clear separation between runtime identity and data

  When restoring state via `start_link_from_state/2`, you must provide the
  `agent_id` as a parameter. This enables use cases like:

      # Restore with same agent_id
      AgentServer.start_link_from_state(saved_state, agent_id: "conversation-123")

      # Clone conversation with different agent_id
      AgentServer.start_link_from_state(saved_state, agent_id: "conversation-123-fork")

  The `agent_id` can be any value that makes sense for your application:
  - Database conversation IDs: `"conv_a1b2c3d4"`
  - User-scoped identifiers: `"user-\#{user_id}-session-\#{session_id}"`
  - Randomly generated GUIDs: `UUID.uuid4()`
  - Application-defined values: `"demo-agent-001"`

  ## Events

  The server broadcasts events on the topic `"agent_server:\#{agent_id}"`:

  ### Todo Events
  - `{:todos_updated, todos}` - Complete snapshot of current TODO list

  ### Status Events
  - `{:status_changed, :idle, nil}` - Server ready for work
  - `{:status_changed, :running, nil}` - Agent executing
  - `{:status_changed, :interrupted, interrupt_data}` - Awaiting human decision
  - `{:status_changed, :completed, final_state}` - Execution completed
    successfully
  - `{:status_changed, :cancelled, nil}` - Execution was cancelled by user
  - `{:status_changed, :error, reason}` - Execution failed

  ### LLM Streaming Events
  - `{:llm_deltas, [%MessageDelta{}]}` - Streaming tokens/deltas received (list
    of deltas)
  - `{:llm_message, %Message{}}` - Complete message received and processed
  - `{:llm_token_usage, %TokenUsage{}}` - Token usage information
  - `{:tool_response, %Message{}}` - Tool execution result (optional)

  ### Message Persistence Events
  - `{:display_message_saved, display_message}` - Broadcast after message is
    persisted via save_new_message_fn callback (only when callback is configured
    and succeeds). The `{:llm_message, ...}` event is also broadcast alongside
    this event for backward compatibility

  **Note**: File events are NOT broadcast by AgentServer. Files are managed by
  `FileSystemServer` which provides its own event handling mechanism.

  ## Debug Events

  When debug PubSub is configured, additional debug events are broadcast on the
  topic `"agent_server:debug:\#{agent_id}"`. These events provide deeper insight
  into agent execution for debugging and monitoring purposes:

  ### Middleware Debug Events
  - `{:agent_state_update, state}` - Middleware state update with
    full state snapshot

  ## Usage

      # Start a server
      {:ok, agent} = Agent.new(
        agent_id: "my-agent-1",
        model: model,
        base_system_prompt: "You are a helpful assistant."
      )

      initial_state = State.new!(%{
        messages: [Message.new_user!("Write a hello world program")]
      })

      {:ok, _pid} = AgentServer.start_link(
        agent: agent,
        initial_state: initial_state,
        name: AgentServer.get_name("my-agent-1")
      )

      # Subscribe to events
      AgentServer.subscribe("my-agent-1")

      # Execute the agent
      :ok = AgentServer.execute("my-agent-1")

      # Cancel execution if needed
      :ok = AgentServer.cancel("my-agent-1")

      # Listen for events
      receive do
        {:todos_updated, todos} -> IO.inspect(todos, label: "Current TODOs")
        {:status_changed, :completed, final_state} -> IO.puts("Done!")
      end

  ## Human-in-the-Loop Example

      # Configure agent with interrupts
      {:ok, agent} = Agent.new(
        agent_id: "my-agent-1",
        model: model,
        interrupt_on: %{"write_file" => true}
      )

      {:ok, _pid} = AgentServer.start_link(
        agent: agent,
        initial_state: state,
        name: AgentServer.get_name("my-agent-1")
      )

      AgentServer.subscribe("my-agent-1")

      # Execute
      AgentServer.execute("my-agent-1")

      # Wait for interrupt
      receive do
        {:status_changed, :interrupted, interrupt_data} ->
          # Display interrupt_data.action_requests to user
          decisions = get_user_decisions(interrupt_data)
          AgentServer.resume("my-agent-1", decisions)
      end

      # Wait for completion
      receive do
        {:status_changed, :completed, final_state} -> :ok
      end
  """

  use GenServer
  require Logger

  alias LangChain.Agents.Agent
  alias LangChain.Agents.State
  alias LangChain.Agents.AgentSupervisor
  alias LangChain.Agents.Middleware
  alias LangChain.Agents.MiddlewareEntry
  alias LangChain.Persistence.StateSerializer

  @registry LangChain.Agents.Registry

  @typedoc "Status of the agent server"
  @type status :: :idle | :running | :interrupted | :cancelled | :error

  defmodule ServerState do
    @moduledoc false
    defstruct [
      :agent,
      :state,
      :status,
      :pubsub,
      :topic,
      :debug_pubsub,
      :debug_topic,
      :interrupt_data,
      :error,
      :inactivity_timeout,
      :inactivity_timer_ref,
      :last_activity_at,
      :shutdown_delay,
      :task,
      :middleware_registry,
      :presence_config,
      :conversation_id,
      :save_new_message_fn
    ]

    @type t :: %__MODULE__{
            agent: Agent.t(),
            state: State.t(),
            status: :idle | :running | :interrupted | :completed | :cancelled | :error,
            pubsub: {module(), atom()} | nil,
            topic: String.t(),
            debug_pubsub: {module(), atom()} | nil,
            debug_topic: String.t() | nil,
            interrupt_data: map() | nil,
            error: term() | nil,
            inactivity_timeout: pos_integer() | nil | :infinity,
            inactivity_timer_ref: reference() | nil,
            last_activity_at: DateTime.t() | nil,
            shutdown_delay: pos_integer() | nil,
            task: Task.t() | nil,
            middleware_registry: %{(atom() | String.t()) => MiddlewareEntry.t()},
            presence_config:
              %{enabled: boolean(), presence_module: module(), topic: String.t()} | nil,
            conversation_id: String.t() | nil,
            save_new_message_fn: (String.t(), LangChain.Message.t() -> {:ok, list()} | {:error, term()}) | nil
          }
  end

  ## Client API

  @doc """
  Start an AgentServer.

  ## Options

  - `:agent` - The Agent struct (required)
  - `:initial_state` - Initial State (default: empty state)
  - `:pubsub` - PubSub configuration as `{module(), atom()}` tuple or `nil` to disable (default: nil)
    Example: `{Phoenix.PubSub, :my_app_pubsub}`
  - `:debug_pubsub` - Optional separate PubSub for debug events as `{module(), atom()}` or `nil` (default: nil)
    Example: `{Phoenix.PubSub, :my_debug_pubsub}`
  - `:name` - Server name registration (optional, defaults to `get_name(agent.agent_id)`)
  - `:inactivity_timeout` - Timeout in milliseconds for automatic shutdown due to inactivity (default: 300_000 - 5 minutes)
    Set to `nil` or `:infinity` to disable automatic shutdown
  - `:shutdown_delay` - Delay in milliseconds to allow the supervisor to gracefully stop all children (default: 5000)
  - `:conversation_id` - Optional conversation identifier for message persistence (default: nil)
  - `:save_new_message_fn` - Optional callback function for persisting messages (default: nil)
    Function signature: `(conversation_id :: String.t(), message :: LangChain.Message.t()) -> {:ok, [saved_messages]} | {:error, reason}`

  ## Examples

      # Start with automatic name (recommended)
      {:ok, pid} = AgentServer.start_link(
        agent: agent,
        initial_state: state
      )

      # With PubSub enabled
      {:ok, pid} = AgentServer.start_link(
        agent: agent,
        initial_state: state,
        pubsub: {Phoenix.PubSub, :my_app_pubsub}
      )

      # Start with explicit name (advanced use cases)
      {:ok, pid} = AgentServer.start_link(
        agent: agent,
        initial_state: state,
        name: :my_custom_name
      )

      # With custom inactivity timeout
      {:ok, pid} = AgentServer.start_link(
        agent: agent,
        inactivity_timeout: 600_000  # 10 minutes
      )

      # Disable automatic shutdown
      {:ok, pid} = AgentServer.start_link(
        agent: agent,
        inactivity_timeout: nil
      )

      # Enable debug pubsub
      {:ok, pid} = AgentServer.start_link(
        agent: agent,
        pubsub: {Phoenix.PubSub, :my_app_pubsub},
        debug_pubsub: {Phoenix.PubSub, :my_debug_pubsub}
      )
  """
  def start_link(opts) do
    # Determine default name from agent or restore_agent_id
    default_name =
      cond do
        # When restoring from state, use restore_agent_id
        Keyword.has_key?(opts, :restore_agent_id) ->
          get_name(Keyword.get(opts, :restore_agent_id))

        # When starting fresh, use agent's agent_id
        agent = Keyword.get(opts, :agent) ->
          get_name(agent.agent_id)

        # Fallback
        true ->
          __MODULE__
      end

    {name, opts} = Keyword.pop(opts, :name, default_name)

    GenServer.start_link(__MODULE__, opts, name: name)
  end

  @doc """
  Get the name of the AgentServer process for a specific agent.

  ## Examples

      name = AgentServer.get_name("my-agent-1")
      GenServer.call(name, :get_status)
  """
  @spec get_name(String.t()) :: {:via, Registry, {Registry, String.t()}}
  def get_name(agent_id) when is_binary(agent_id) do
    {:via, Registry, {@registry, {:agent_server, agent_id}}}
  end

  @doc """
  Get the pid of the AgentServer process for a specific agent.

  ## Examples

      pid = AgentServer.get_pit("my-agent-1")
      send(pid, message)
  """
  @spec get_pid(String.t()) :: pid() | {atom(), node()} | nil
  def get_pid(agent_id) when is_binary(agent_id) do
    agent_id
    |> get_name()
    |> GenServer.whereis()
  end

  @doc """
  Send the AgentServer a message intended to be processed by a middleware
  handler. This is a mechanism for a middleware to send itself a message, as any
  message processed by a middleware must be designed to be handled.
  """
  @spec send_middleware_message(String.t(), term(), term()) :: :ok
  def send_middleware_message(agent_id, middleware_id, message) do
    agent_id
    |> get_pid()
    |> case do
      pid when is_pid(pid) ->
        send(pid, {:middleware_message, middleware_id, message})
        :ok

      _other ->
        :ok
    end
  end

  @doc """
  Subscribe to events from this AgentServer.

  The calling process will receive messages for all events broadcast by this server.

  Returns `:ok` on success or `{:error, reason}` if PubSub is not configured.

  ## Examples

      AgentServer.subscribe("my-agent-1")
  """
  @spec subscribe(String.t()) :: :ok | {:error, term()}
  def subscribe(agent_id) do
    case GenServer.call(get_name(agent_id), :get_pubsub_info) do
      {pubsub, pubsub_name, topic} ->
        # subscribe the client process executing this request to the pubsub
        # topic the server is using for broadcasting events
        pubsub.subscribe(pubsub_name, topic)

      nil ->
        {:error, :no_pubsub}
    end
  end

  @doc """
  Unsubscribe from events from this AgentServer.

  The calling process will stop receiving messages broadcast by this server.

  Returns `:ok` on success or `{:error, reason}` if PubSub is not configured.

  ## Examples

      AgentServer.unsubscribe("my-agent-1")
  """
  @spec unsubscribe(String.t()) :: :ok | {:error, term()}
  def unsubscribe(agent_id) do
    case GenServer.call(get_name(agent_id), :get_pubsub_info) do
      {pubsub, pubsub_name, topic} ->
        # unsubscribe the client process from the pubsub topic
        pubsub.unsubscribe(pubsub_name, topic)

      nil ->
        {:error, :no_pubsub}
    end
  end

  @doc """
  Subscribe to debug events from this AgentServer.

  The calling process will receive messages for all debug events broadcast by this server.
  Debug events provide additional debugging insight into what the agent is doing, such as
  middleware state updates.

  Returns `:ok` on success or `{:error, reason}` if debug PubSub is not configured.

  ## Examples

      AgentServer.subscribe_debug("my-agent-1")
  """
  @spec subscribe_debug(String.t()) :: :ok | {:error, term()}
  def subscribe_debug(agent_id) do
    case GenServer.call(get_name(agent_id), :get_debug_pubsub_info) do
      {debug_pubsub, debug_pubsub_name, debug_topic} ->
        # subscribe the client process executing this request to the debug pubsub
        # topic the server is using for broadcasting debug events
        debug_pubsub.subscribe(debug_pubsub_name, debug_topic)

      nil ->
        {:error, :no_debug_pubsub}
    end
  end

  @doc """
  Unsubscribe from debug events from this AgentServer.

  The calling process will stop receiving debug messages broadcast by this server.

  Returns `:ok` on success or `{:error, reason}` if debug PubSub is not configured.

  ## Examples

      AgentServer.unsubscribe_debug("my-agent-1")
  """
  @spec unsubscribe_debug(String.t()) :: :ok | {:error, term()}
  def unsubscribe_debug(agent_id) do
    case GenServer.call(get_name(agent_id), :get_debug_pubsub_info) do
      {debug_pubsub, debug_pubsub_name, debug_topic} ->
        # unsubscribe the client process from the debug pubsub topic
        debug_pubsub.unsubscribe(debug_pubsub_name, debug_topic)

      nil ->
        {:error, :no_debug_pubsub}
    end
  end

  @doc """
  Request the AgentServer to publish an specific PubSub message or event.

  Designed to make it easier for a middleware desiring to publish messages to
  the Agent's PubSub.

  A PubSub message is only broadcast if the AgentServer is configured with
  PubSub.
  """
  @spec publish_event_from(String.t(), term()) :: :ok
  def publish_event_from(agent_id, event) do
    GenServer.cast(get_name(agent_id), {:publish_event, event})
  end

  @doc """
  Lists all currently running agent processes.

  Returns a list of agent_ids for all running AgentServer processes registered
  in the LangChain.Agents.Registry.

  ## Examples

      AgentServer.list_running_agents()
      # => ["conversation-1", "conversation-2", "user-123"]
  """
  @spec list_running_agents() :: [String.t()]
  def list_running_agents do
    # Query the Registry for agent_server entries only (not supervisors)
    # Registry stores entries as {key, pid, value}
    # We only want {:agent_server, agent_id} entries
    Registry.select(LangChain.Agents.Registry, [
      {{{:agent_server, :"$1"}, :_, :_}, [], [:"$1"]}
    ])
  end

  @doc """
  Gets all running agents matching a glob pattern.

  Supports wildcard patterns using `*` which matches any sequence of characters.

  ## Examples

      # Get all conversation agents
      AgentServer.list_agents_matching("conversation-*")
      # => ["conversation-1", "conversation-2", "conversation-123"]

      # Get all user agents
      AgentServer.list_agents_matching("user-*")
      # => ["user-42", "user-99"]

      # Get specific prefix
      AgentServer.list_agents_matching("demo-*")
      # => ["demo-agent-001"]
  """
  @spec list_agents_matching(String.t()) :: [String.t()]
  def list_agents_matching(pattern) do
    regex = pattern_to_regex(pattern)

    list_running_agents()
    |> Enum.filter(&Regex.match?(regex, &1))
  end

  @doc """
  Gets count of currently running agents.

  Returns the total number of AgentServer processes registered in the
  LangChain.Agents.Registry.

  ## Examples

      AgentServer.agent_count()
      # => 5
  """
  @spec agent_count() :: non_neg_integer()
  def agent_count do
    Registry.count(LangChain.Agents.Registry)
  end

  @doc """
  Gets detailed information about a running agent.

  Returns a map with agent status and state information, or `nil` if the agent
  is not running.

  ## Return Value

  If the agent is running, returns a map containing:
  - `:agent_id` - The agent identifier
  - `:pid` - The process ID
  - `:status` - Current execution status (`:idle`, `:running`, `:interrupted`, etc.)
  - `:state` - Exported state snapshot
  - `:message_count` - Number of messages in the state
  - `:has_interrupt` - Boolean indicating if there's pending interrupt data

  ## Examples

      AgentServer.agent_info("conversation-1")
      # => %{
      #   agent_id: "conversation-1",
      #   pid: #PID<0.1234.0>,
      #   status: :idle,
      #   state: %State{...},
      #   message_count: 5,
      #   has_interrupt: false
      # }

      AgentServer.agent_info("nonexistent")
      # => nil
  """
  @spec agent_info(String.t()) :: map() | nil
  def agent_info(agent_id) do
    case get_pid(agent_id) do
      nil ->
        nil

      pid ->
        state = export_state(agent_id)
        status = get_status(agent_id)

        %{
          agent_id: agent_id,
          pid: pid,
          status: status,
          state: state,
          message_count: length(state.messages),
          has_interrupt: state.interrupt_data != nil
        }
    end
  end

  # Convert glob pattern to regex
  # "conversation-*" -> ~r/^conversation-.*$/
  defp pattern_to_regex(pattern) do
    escaped = Regex.escape(pattern)
    regex_str = String.replace(escaped, "\\*", ".*")
    Regex.compile!("^#{regex_str}$")
  end

  @doc """
  Execute the agent.

  Starts agent execution asynchronously. The server will broadcast events as the
  agent runs. Returns `:ok` immediately.

  Returns `{:error, reason}` if the server is not idle (already running, interrupted, etc.).

  ## Examples

      :ok = AgentServer.execute("my-agent-1")
  """
  @spec execute(String.t()) :: :ok | {:error, term()}
  def execute(agent_id) do
    GenServer.call(get_name(agent_id), :execute, :infinity)
  end

  @doc """
  Cancel a running LLM task.

  Stops the currently executing agent task and transitions the server to completed status.
  Returns `{:error, reason}` if the server is not running (no task to cancel).

  ## Examples

      :ok = AgentServer.cancel("my-agent-1")
  """
  @spec cancel(String.t()) :: :ok | {:error, term()}
  def cancel(agent_id) do
    GenServer.call(get_name(agent_id), :cancel)
  end

  @doc """
  Resume agent execution after a human-in-the-loop interrupt.

  ## Parameters

  - `agent_id` - The agent identifier
  - `decisions` - List of decision maps from human reviewer (see `LangChain.Agents.Agent.resume/3`)

  ## Examples

      decisions = [
        %{type: :approve},
        %{type: :edit, arguments: %{"path" => "safe.txt"}},
        %{type: :reject}
      ]

      :ok = AgentServer.resume("my-agent-1", decisions)
  """
  @spec resume(String.t(), list(map())) :: :ok | {:error, term()}
  def resume(agent_id, decisions) when is_list(decisions) do
    GenServer.call(get_name(agent_id), {:resume, decisions}, :infinity)
  end

  # The `get_state/1` function is available to aid in testing and not intended as a general public API.
  @doc false
  @spec get_state(String.t()) :: State.t()
  def get_state(agent_id) do
    GenServer.call(get_name(agent_id), :get_state)
  end

  @doc """
  Get the current status of the server.

  Returns one of: `:idle`, `:running`, `:interrupted`, `:completed`, `:error`

  ## Examples

      status = AgentServer.get_status("my-agent-1")
  """
  @spec get_status(String.t()) :: status()
  def get_status(agent_id) do
    GenServer.call(get_name(agent_id), :get_status)
  end

  @doc """
  Get server info including status, state, and any error or interrupt data.

  Returns a map with:
  - `:status` - Current status
  - `:state` - Current State
  - `:interrupt_data` - Interrupt data if status is `:interrupted`
  - `:error` - Error reason if status is `:error`

  ## Examples

      info = AgentServer.get_info("my-agent-1")
  """
  @spec get_info(String.t()) :: map()
  def get_info(agent_id) do
    GenServer.call(get_name(agent_id), :get_info)
  end

  @doc """
  Add a message to the agent's state and transition to idle if completed.

  This is useful for conversational interfaces where you want to add a new user
  message after the agent has completed a previous execution.

  Returns `:ok` on success.

  ## Examples

      # After agent completes
      :ok = AgentServer.add_message("my-agent-1", Message.new_user!("What's next?"))
      :ok = AgentServer.execute("my-agent-1")
  """
  @spec add_message(String.t(), LangChain.Message.t()) :: :ok
  def add_message(agent_id, %LangChain.Message{} = message) do
    case GenServer.call(get_name(agent_id), {:add_message, message}) do
      :ok ->
        execute(agent_id)

      {:error, reason} ->
        {:error, reason}
    end
  end

  @doc """
  Reset the agent's state and filesystem to start fresh.

  This clears:
  - All messages
  - All TODOs
  - Middleware state
  - Memory-only files (completely removed)
  - In-memory modifications to persisted files (discarded)

  This preserves:
  - Metadata (configuration)
  - Persisted files (reverted to pristine state from storage)

  Status transitions:
  - `:completed`, `:error`, or `:cancelled` → `:idle` (ready for new execution)
  - Other statuses remain unchanged

  Returns `:ok` on success.

  ## Examples

      # After agent completes or encounters error
      :ok = AgentServer.reset("my-agent-1")
      # Now you can execute again with clean state
      :ok = AgentServer.execute("my-agent-1")
  """
  @spec reset(String.t()) :: :ok
  def reset(agent_id) do
    GenServer.call(get_name(agent_id), :reset)
  end

  @doc """
  Get the current inactivity status of an agent.

  Returns a map with:
  - `:inactivity_timeout` - Configured timeout in milliseconds (or nil/:infinity)
  - `:last_activity_at` - DateTime of last activity
  - `:timer_active` - Boolean indicating if timer is currently running
  - `:time_since_activity` - Milliseconds since last activity (or nil if no activity yet)

  ## Examples

      status = AgentServer.get_inactivity_status("my-agent-1")
      # => %{
      #   inactivity_timeout: 300_000,
      #   last_activity_at: ~U[2025-11-06 10:15:30.123Z],
      #   timer_active: true,
      #   time_since_activity: 45_000
      # }
  """
  @spec get_inactivity_status(String.t()) :: map()
  def get_inactivity_status(agent_id) do
    GenServer.call(get_name(agent_id), :get_inactivity_status)
  end

  @doc """
  Set the complete TODO list for the agent.

  This replaces the entire TODO list and broadcasts appropriate TODO events
  (created, updated, deleted) for UI synchronization.

  Useful for:
  - Thread restoration (restoring persisted TODOs)
  - Testing scenarios (setting sample TODOs)
  - Bulk TODO updates

  ## Parameters

  - `agent_id` - The agent identifier
  - `todos` - List of Todo structs

  ## Examples

      todos = [
        Todo.new!(%{content: "Task 1", status: :completed}),
        Todo.new!(%{content: "Task 2", status: :in_progress})
      ]
      :ok = AgentServer.set_todos("my-agent-1", todos)
  """
  @spec set_todos(String.t(), list(LangChain.Agents.Todo.t())) :: :ok
  def set_todos(agent_id, todos) when is_list(todos) do
    GenServer.call(get_name(agent_id), {:set_todos, todos})
  end

  @doc """
  Set the complete message list for the agent.

  This replaces the entire message list and broadcasts individual `{:llm_message, message}`
  events for each message, allowing UI synchronization.

  Useful for:
  - Thread restoration (restoring persisted messages)
  - Testing scenarios (setting sample messages)
  - Bulk message updates

  ## Parameters

  - `agent_id` - The agent identifier
  - `messages` - List of Message structs

  ## Examples

      messages = [
        Message.new_system!("You are helpful"),
        Message.new_user!("Hello")
      ]
      :ok = AgentServer.set_messages("my-agent-1", messages)
  """
  @spec set_messages(String.t(), list(LangChain.Message.t())) :: :ok
  def set_messages(agent_id, messages) when is_list(messages) do
    GenServer.call(get_name(agent_id), {:set_messages, messages})
  end

  @doc """
  Check if an agent is running.
  """
  def running?(agent_id) do
    case get_pid(agent_id) do
      nil -> false
      _pid -> true
    end
  end

  @doc """
  Stop the AgentServer.

  ## Examples

      :ok = AgentServer.stop("my-agent-1")
  """
  @spec stop(String.t()) :: :ok
  def stop(agent_id) do
    GenServer.stop(get_name(agent_id))
  end

  @doc """
  Export the current conversation state to a serializable format.

  This can be persisted to a database and later used to restore the conversation
  state. The exported state uses string keys (not atoms) for compatibility
  with JSON/JSONB storage.

  Returns a map with string keys containing:
  - `"version"` - Serialization format version
  - `"state"` - The conversation state (messages, todos, metadata)
  - `"serialized_at"` - ISO8601 timestamp

  **What is NOT included**:
  - Agent configuration (middleware, tools, model) - must come from application code
  - `agent_id` - runtime identifier provided when restoring

  This design allows you to restore the same conversation state under a different
  agent_id, enabling use cases like state cloning and conversation forking.

  ## Examples

      state = AgentServer.export_state("my-agent-1")
      # Save to database
      MyApp.Conversations.save_agent_state(conversation_id, state)
  """
  @spec export_state(String.t()) :: map()
  def export_state(agent_id) do
    GenServer.call(get_name(agent_id), :export_state)
  end

  @doc """
  Restore agent state from a previously exported state.

  This updates an already-running agent to restore its state from a
  previously serialized format. The state should be a map with string
  keys (as returned by `export_state/1`).

  Returns `:ok` on success or `{:error, reason}` on failure.

  ## Examples

      # Load from database
      {:ok, persisted_state} = MyApp.Conversations.load_agent_state(conversation_id)

      # Restore into existing agent
      :ok = AgentServer.restore_state("my-agent-1", persisted_state)
  """
  @spec restore_state(String.t(), map()) :: :ok | {:error, term()}
  def restore_state(agent_id, persisted_state) when is_map(persisted_state) do
    GenServer.call(get_name(agent_id), {:restore_state, persisted_state})
  end

  @doc """
  Updates both the agent configuration and state.

  This is the recommended way to restore a conversation:
  1. Create agent from current code using your agent factory
  2. Load state from database
  3. Call this function to update the running AgentServer

  ## Parameters

  - `agent_id` - The agent server's identifier
  - `agent` - The new agent configuration (from current code)
  - `state` - The restored state (from database)

  ## Examples

      # Restore conversation
      {:ok, agent} = MyApp.Agents.create_demo_agent(agent_id: "demo-123")
      {:ok, state_data} = MyApp.Conversations.load_state(conversation_id)
      {:ok, state} = LangChain.Agents.State.from_serialized(state_data["state"])

      :ok = AgentServer.update_agent_and_state("demo-123", agent, state)

  ## Returns

  - `:ok` - Agent and state updated successfully
  - `{:error, reason}` - If agent server is not running or update fails
  """
  def update_agent_and_state(agent_id, agent, state) do
    GenServer.call(get_name(agent_id), {:update_agent_and_state, agent, state})
  end

  @doc """
  Start a new AgentServer with restored state.

  This is the preferred way to resume a conversation from persisted state.
  The persisted_state should be a map with string keys (as returned by
  `export_state/1`).

  The `agent_id` will be used as the runtime identifier for this agent,
  enabling process registration and PubSub topic setup. You can restore
  the same conversation state under a different agent_id, which is useful
  for state cloning or conversation forking.

  ## Agent Configuration from Code

  **REQUIRED**: You MUST provide the agent from your application code using the
  `:agent` option. The persisted state contains ONLY conversation state (messages,
  todos, metadata). Agent configuration (middleware, tools, model) comes from your
  application code.

  This design ensures:
  - Library upgrades automatically benefit all conversations
  - Code changes automatically apply to all conversations
  - Per-user capabilities can be controlled via application logic

  ## Options

  - `:agent_id` - The runtime identifier for this agent (required)
  - `:agent` - Agent struct from code (REQUIRED)
  - `:pubsub` - PubSub configuration as `{module(), atom()}` tuple or `nil` (default: nil)
  - `:debug_pubsub` - Optional separate PubSub for debug events as `{module(), atom()}` or `nil` (default: nil)
  - `:name` - Server name registration (optional, defaults to `get_name(agent_id)`)
  - `:inactivity_timeout` - Timeout in milliseconds (default: 300_000)
  - `:shutdown_delay` - Delay in milliseconds (default: 5000)

  ## Examples

      # Standard restoration pattern
      {:ok, persisted_state} = MyApp.Conversations.load_agent_state(conversation_id)

      # Agent from code (ALWAYS required)
      {:ok, agent} = MyApp.Agents.create_agent(
        agent_id: "my-agent-1",
        model: model,
        middleware: [TodoList, FileSystem, SubAgent]
      )

      # Start with restored state
      {:ok, pid} = AgentServer.start_link_from_state(
        persisted_state,
        agent: agent,
        agent_id: "my-agent-1",
        pubsub: {Phoenix.PubSub, :my_app_pubsub}
      )

      # Clone conversation with a different agent_id
      {:ok, pid} = AgentServer.start_link_from_state(
        persisted_state,
        agent: agent,
        agent_id: "my-agent-clone",
        pubsub: {Phoenix.PubSub, :my_app_pubsub}
      )
  """
  @spec start_link_from_state(map(), keyword()) :: GenServer.on_start()
  def start_link_from_state(persisted_state, opts \\ []) when is_map(persisted_state) do
    # agent_id is required in opts
    agent_id = Keyword.fetch!(opts, :agent_id)

    # Add a marker to opts to indicate this is a restore operation
    opts =
      opts
      |> Keyword.put(:restore_from, persisted_state)
      |> Keyword.put(:restore_agent_id, agent_id)

    start_link(opts)
  end

  ## Server Callbacks

  @impl true
  def init(opts) do
    # Trap exits to ensure terminate/2 is called for graceful shutdown
    Process.flag(:trap_exit, true)

    # Check if we're restoring from persisted state
    case Keyword.get(opts, :restore_from) do
      nil ->
        # Normal initialization
        init_fresh(opts)

      persisted_state ->
        # Restore from persisted state
        init_from_persisted(persisted_state, opts)
    end
  end

  defp init_fresh(opts) do
    agent = Keyword.fetch!(opts, :agent)
    initial_state = Keyword.get(opts, :initial_state) || State.new!()

    build_server_state(agent, initial_state, opts)
  end

  defp init_from_persisted(persisted_state, opts) do
    # Get the agent_id from opts (required)
    agent_id = Keyword.fetch!(opts, :restore_agent_id)
    agent = Keyword.fetch!(opts, :agent)

    # deserialize only conversation state
    # agent_id is not serialized, so we provide it when deserializing
    case StateSerializer.deserialize_state(agent_id, persisted_state["state"]) do
      {:ok, state} ->
        # Update agent_id to match the runtime identifier
        agent = %{agent | agent_id: agent_id}
        build_server_state(agent, state, opts)

      {:error, reason} ->
        {:stop, {:restore_failed, reason}}
    end
  end

  # Build ServerState from agent, state, and opts
  # This is the shared logic used by both init_fresh/1 and init_from_persisted/2
  defp build_server_state(agent, state, opts) do
    # Ensure agent_id is set in the state
    state = %{state | agent_id: agent.agent_id}

    # Get pubsub configuration as {module(), atom()} tuple or nil
    # Expected format: pubsub: {Phoenix.PubSub, :my_app_pubsub}
    pubsub = Keyword.get(opts, :pubsub)

    # Get debug_pubsub configuration as {module(), atom()} tuple or nil
    # Expected format: debug_pubsub: {Phoenix.PubSub, :my_debug_pubsub}
    debug_pubsub = Keyword.get(opts, :debug_pubsub)

    # allow a nil value to disable the timeout
    inactivity_timeout = Keyword.get(opts, :inactivity_timeout, 300_000)
    shutdown_delay = Keyword.get(opts, :shutdown_delay, 5_000)

    # Extract presence configuration
    presence_opts = Keyword.get(opts, :presence_tracking)

    presence_config =
      if presence_opts do
        %{
          enabled: Keyword.get(presence_opts, :enabled, true),
          presence_module: Keyword.fetch!(presence_opts, :presence_module),
          topic: Keyword.fetch!(presence_opts, :topic)
        }
      else
        nil
      end

    topic = "agent_server:#{agent.agent_id}"
    debug_topic = if debug_pubsub, do: "agent_server:debug:#{agent.agent_id}", else: nil

    # Build list of MiddlewareEntry structs
    middleware_entries = build_middleware_entries(agent.middleware)

    # Build registry map for O(1) message routing
    middleware_registry = Map.new(middleware_entries, fn entry -> {entry.id, entry} end)

    # Update agent with middleware entries
    updated_agent = %{agent | middleware: middleware_entries}

    # Extract conversation_id and save_new_message_fn from opts
    conversation_id = Keyword.get(opts, :conversation_id)
    save_new_message_fn = Keyword.get(opts, :save_new_message_fn)

    server_state = %ServerState{
      agent: updated_agent,
      state: state,
      status: :idle,
      pubsub: pubsub,
      topic: topic,
      debug_pubsub: debug_pubsub,
      debug_topic: debug_topic,
      interrupt_data: nil,
      error: nil,
      inactivity_timeout: inactivity_timeout,
      inactivity_timer_ref: nil,
      last_activity_at: DateTime.utc_now(),
      shutdown_delay: shutdown_delay,
      middleware_registry: middleware_registry,
      presence_config: presence_config,
      conversation_id: conversation_id,
      save_new_message_fn: save_new_message_fn
    }

    # Start the inactivity timer
    server_state = reset_inactivity_timer(server_state)

    {:ok, server_state}
  end

  @impl true
  def handle_call(:get_pubsub_info, _from, server_state) do
    result =
      case server_state.pubsub do
        {pubsub, pubsub_name} ->
          {pubsub, pubsub_name, server_state.topic}

        nil ->
          nil
      end

    {:reply, result, server_state}
  end

  @impl true
  def handle_call(:get_debug_pubsub_info, _from, server_state) do
    result =
      case server_state.debug_pubsub do
        {debug_pubsub, debug_pubsub_name} ->
          {debug_pubsub, debug_pubsub_name, server_state.debug_topic}

        nil ->
          nil
      end

    {:reply, result, server_state}
  end

  @impl true
  def handle_call(:execute, _from, %ServerState{status: :idle} = server_state) do
    # Build callback handlers that will forward events via PubSub
    callbacks = build_llm_callbacks(server_state)

    # Transition to running
    new_state = %{server_state | status: :running}
    broadcast_event(new_state, {:status_changed, :running, nil})

    # Reset inactivity timer on execution start
    new_state = reset_inactivity_timer(new_state)

    # Start async execution
    task =
      Task.async(fn ->
        execute_agent(new_state, callbacks)
      end)

    # Store task reference if needed, or just let it run
    # For now, we'll handle the result in handle_info
    {:reply, :ok, Map.put(new_state, :task, task)}
  end

  @impl true
  def handle_call(:execute, _from, server_state) do
    {:reply, {:error, "Cannot execute, server is in state: #{server_state.status}"}, server_state}
  end

  @impl true
  def handle_call(:cancel, _from, %ServerState{status: :running, task: task} = server_state)
      when not is_nil(task) do
    Logger.info("Cancelling agent execution for agent: #{server_state.agent.agent_id}")

    # Shutdown the running task
    Task.shutdown(task, :brutal_kill)

    # Transition to cancelled status (not completed)
    # Note: We don't include the state in the broadcast because it may be in an
    # inconsistent state after brutal task termination
    new_state = %{server_state | status: :cancelled}

    # Reset inactivity timer after cancellation
    new_state = reset_inactivity_timer(new_state)

    # Broadcast cancellation event
    broadcast_event(new_state, {:status_changed, :cancelled, nil})

    {:reply, :ok, Map.put(new_state, :task, nil)}
  end

  @impl true
  def handle_call(:cancel, _from, server_state) do
    {:reply, {:error, "Cannot cancel, server is not running (status: #{server_state.status})"},
     server_state}
  end

  @impl true
  def handle_call({:resume, decisions}, _from, %ServerState{status: :interrupted} = server_state) do
    # Transition back to running
    new_state = %{
      server_state
      | status: :running,
        interrupt_data: nil
    }

    broadcast_event(new_state, {:status_changed, :running, nil})

    # Reset inactivity timer on resume
    new_state = reset_inactivity_timer(new_state)

    # Resume execution async (callbacks are built in resume_agent)
    task =
      Task.async(fn ->
        resume_agent(new_state, decisions)
      end)

    {:reply, :ok, Map.put(new_state, :task, task)}
  end

  @impl true
  def handle_call({:resume, _decisions}, _from, server_state) do
    {:reply, {:error, "Cannot resume, server is not interrupted"}, server_state}
  end

  @impl true
  def handle_call(:get_state, _from, server_state) do
    {:reply, server_state.state, server_state}
  end

  @impl true
  def handle_call(:get_status, _from, server_state) do
    {:reply, server_state.status, server_state}
  end

  @impl true
  def handle_call(:get_info, _from, server_state) do
    info = %{
      status: server_state.status,
      state: server_state.state,
      interrupt_data: server_state.interrupt_data,
      error: server_state.error
    }

    {:reply, info, server_state}
  end

  @impl true
  def handle_call({:add_message, message}, _from, server_state) do
    # Add message to the state
    new_state = State.add_message(server_state.state, message)

    # Transition to idle if we were completed/error/cancelled to allow new execution
    new_status =
      case server_state.status do
        :completed -> :idle
        :error -> :idle
        :cancelled -> :idle
        status -> status
      end

    updated_server_state = %{
      server_state
      | state: new_state,
        status: new_status,
        error: nil
    }

    # Reset inactivity timer on user message
    updated_server_state = reset_inactivity_timer(updated_server_state)

    # Save and broadcast messages immediately
    # Note: During LLM execution, assistant messages are also saved via on_message_processed callback
    # But if manually adding assistant messages, we should also save them here
    maybe_save_and_broadcast_message(updated_server_state, message)

    # Broadcast debug event for state update
    broadcast_debug_event(updated_server_state, {:agent_state_update, new_state})

    {:reply, :ok, updated_server_state}
  end

  @impl true
  def handle_call(:reset, _from, server_state) do
    # Reset the filesystem first (clears memory files, unloads persisted files)
    agent_id = server_state.agent.agent_id
    :ok = LangChain.Agents.FileSystemServer.reset(agent_id)

    # Reset the agent state (clears messages, todos)
    reset_state = State.reset(server_state.state)

    # Transition to idle if we were completed/error/cancelled
    new_status =
      case server_state.status do
        :completed -> :idle
        :error -> :idle
        :cancelled -> :idle
        status -> status
      end

    updated_server_state = %{
      server_state
      | state: reset_state,
        status: new_status,
        error: nil,
        interrupt_data: nil
    }

    # Broadcast status change if status changed
    if new_status != server_state.status do
      broadcast_event(server_state, {:status_changed, new_status, nil})
    end

    broadcast_state_changes(server_state, reset_state)

    # Reset activity timer
    updated_server_state = reset_inactivity_timer(updated_server_state)

    {:reply, :ok, updated_server_state}
  end

  @impl true
  def handle_call(:get_inactivity_status, _from, server_state) do
    status = %{
      inactivity_timeout: server_state.inactivity_timeout,
      last_activity_at: server_state.last_activity_at,
      timer_active: !is_nil(server_state.inactivity_timer_ref),
      time_since_activity: time_since(server_state.last_activity_at)
    }

    {:reply, status, server_state}
  end

  @impl true
  def handle_call({:set_todos, todos}, _from, server_state) do
    # Update state with new todos
    new_state = State.set_todos(server_state.state, todos)

    # Broadcast complete snapshot of current TODOs
    broadcast_todos(server_state, new_state)

    # Update server state
    updated_server_state = %{server_state | state: new_state}

    # Reset inactivity timer on todo update
    updated_server_state = reset_inactivity_timer(updated_server_state)

    # Broadcast debug event for state update
    broadcast_debug_event(updated_server_state, {:agent_state_update, new_state})

    {:reply, :ok, updated_server_state}
  end

  @impl true
  def handle_call({:set_messages, messages}, _from, server_state) do
    # Update state with new messages
    new_state = State.set_messages(server_state.state, messages)

    # Broadcast complete snapshot of current messages
    broadcast_messages(server_state, new_state)

    # Update server state
    updated_server_state = %{server_state | state: new_state}

    # Reset inactivity timer on message update
    updated_server_state = reset_inactivity_timer(updated_server_state)

    # Broadcast debug event for state update
    broadcast_debug_event(updated_server_state, {:agent_state_update, new_state})

    {:reply, :ok, updated_server_state}
  end

  @impl true
  def handle_call(:export_state, _from, server_state) do
    # Serialize the current state using StateSerializer
    serialized =
      StateSerializer.serialize_server_state(
        server_state.agent,
        server_state.state
      )

    {:reply, serialized, server_state}
  end

  @impl true
  def handle_call({:restore_state, persisted_state}, _from, server_state) do
    # Deserialize only conversation state (not agent config)
    # Get agent_id from the running agent
    agent_id = server_state.agent.agent_id

    case StateSerializer.deserialize_state(agent_id, persisted_state["state"]) do
      {:ok, state} ->
        # Update only the state, keep existing agent from code
        # This function is for updating state in a running agent server
        updated_server_state = %{
          server_state
          | state: state,
            status: :idle,
            error: nil,
            interrupt_data: nil
        }

        # Broadcast state changes to subscribers
        broadcast_state_changes(server_state, state)

        # Reset inactivity timer after restore
        updated_server_state = reset_inactivity_timer(updated_server_state)

        {:reply, :ok, updated_server_state}

      {:error, reason} ->
        {:reply, {:error, reason}, server_state}
    end
  end

  @impl true
  def handle_call({:update_agent_and_state, new_agent, new_state}, _from, server_state) do
    Logger.info("Updating agent configuration and state for #{new_agent.agent_id}")

    # Validate that state has agent_id set (critical for middleware functionality)
    unless new_state.agent_id do
      error_msg =
        "State.agent_id is nil. When deserializing state, you must provide agent_id: State.from_serialized(agent_id, data)"

      Logger.error(error_msg)
      {:reply, {:error, error_msg}, server_state}
    else
      # Update both agent and state atomically
      updated_state = %{server_state | agent: new_agent, state: new_state}

      # Broadcast state change event
      broadcast_event(updated_state, {:state_restored, new_state})

      {:reply, :ok, updated_state}
    end
  end

  @impl true
  def handle_cast({:publish_event, event}, server_state) do
    broadcast_event(server_state, event)
    {:noreply, server_state}
  end

  @impl true
  def handle_info({ref, result}, server_state) when is_reference(ref) do
    # Task completed
    Process.demonitor(ref, [:flush])

    # If we're already cancelled, ignore the task result (race condition)
    if server_state.status == :cancelled do
      {:noreply, Map.delete(server_state, :task)}
    else
      handle_execution_result(result, server_state)
    end
  end

  @impl true
  def handle_info({:DOWN, _ref, :process, _pid, :normal}, server_state) do
    # Task process exited normally, already handled in handle_info above
    {:noreply, server_state}
  end

  @impl true
  def handle_info({:DOWN, _ref, :process, _pid, reason}, server_state) do
    # Task crashed or was killed
    # If status is already :cancelled, this is expected (brutal_kill side effect)
    if server_state.status == :cancelled do
      {:noreply, Map.delete(server_state, :task)}
    else
      # Unexpected crash
      Logger.error("Agent execution task crashed: #{inspect(reason)}")

      new_state = %{server_state | status: :error, error: reason}
      broadcast_event(new_state, {:status_changed, :error, reason})

      {:noreply, Map.delete(new_state, :task)}
    end
  end

  @impl true
  def handle_info({:EXIT, _pid, :normal}, server_state) do
    # Process exited normally (we trap exits), this is expected when shutting
    # down for inactivity
    {:noreply, server_state}
  end

  @impl true
  def handle_info({:llm_deltas, _deltas}, server_state) do
    # Deltas are broadcast via on_llm_new_delta callback in build_llm_callbacks
    # No need to process them here - the client (chat_live.ex) will handle merging
    {:noreply, server_state}
  end

  @impl true
  def handle_info(:inactivity_timeout, server_state) do
    agent_id = server_state.agent.agent_id
    Logger.info("Agent #{agent_id} shutting down due to inactivity")

    # Broadcast shutdown event
    broadcast_event(
      server_state,
      {:agent_shutdown,
       %{
         agent_id: agent_id,
         reason: :inactivity,
         last_activity_at: server_state.last_activity_at,
         shutdown_at: DateTime.utc_now()
       }}
    )

    # Stop the parent AgentSupervisor, which will stop all children
    case AgentSupervisor.stop(agent_id, server_state.shutdown_delay) do
      :ok ->
        :ok

      {:error, :not_found} ->
        Logger.warning("AgentSupervisor for agent #{agent_id} was not found, stopping self")
    end

    # Let the supervisor tree shutdown take care of it
    {:noreply, server_state}
  end

  @impl true
  def handle_info(:shutdown_no_viewers, server_state) do
    agent_id = server_state.agent.agent_id
    Logger.info("Agent #{agent_id} shutting down - idle with no viewers")

    # Broadcast shutdown event
    broadcast_event(
      server_state,
      {:agent_shutdown,
       %{
         agent_id: agent_id,
         reason: :no_viewers,
         last_activity_at: server_state.last_activity_at,
         shutdown_at: DateTime.utc_now()
       }}
    )

    # Stop the parent AgentSupervisor, which will stop all children
    case AgentSupervisor.stop(agent_id, server_state.shutdown_delay) do
      :ok ->
        :ok

      {:error, :not_found} ->
        Logger.warning("AgentSupervisor for agent #{agent_id} was not found, stopping self")
    end

    # Let the supervisor tree shutdown take care of it
    {:noreply, server_state}
  end

  @impl true
  def handle_info({:middleware_message, middleware_id, message}, server_state) do
    # Emit telemetry event
    :telemetry.execute(
      [:langchain, :middleware, :message, :received],
      %{count: 1},
      %{middleware_id: middleware_id, agent_id: server_state.agent.agent_id}
    )

    # Look up middleware from registry
    case Map.get(server_state.middleware_registry, middleware_id) do
      nil ->
        Logger.warning("Received message for unknown middleware: #{inspect(middleware_id)}")
        {:noreply, server_state}

      entry ->
        # Call handle_message on the middleware
        case Middleware.apply_handle_message(message, server_state.state, entry) do
          {:ok, updated_state} ->
            # Update server state
            new_server_state = %{server_state | state: updated_state}
            # if debug pubsub is enabled, it will be notified.
            broadcast_debug_event(
              new_server_state,
              {:agent_state_update, middleware_id, updated_state}
            )

            {:noreply, new_server_state}

          {:error, reason} ->
            Logger.error(
              "Error handling middleware message for #{inspect(middleware_id)}: #{inspect(reason)}"
            )

            {:noreply, server_state}
        end
    end
  end

  @impl true
  def handle_info(msg, server_state) do
    # Catch-all for unexpected messages (log at debug level to avoid noise)
    Logger.debug("AgentServer received unexpected message: #{inspect(msg)}")
    {:noreply, server_state}
  end

  @impl true
  def terminate(_reason, server_state) do
    # Cancel timer if present
    cancel_inactivity_timer(server_state)

    # FileSystemServer traps exits and will flush_all in its own terminate/2

    # SubAgentsDynamicSupervisor will be stopped by AgentSupervisor
    # due to rest_for_one strategy

    :ok
  end

  ## Private Functions

  @doc false
  # Build callback handlers that forward LLM events via PubSub
  defp build_llm_callbacks(%ServerState{} = server_state) do
    %{
      # Callback for streaming deltas (tokens as they arrive)
      on_llm_new_delta: fn _chain, deltas ->
        # deltas is a list of MessageDelta structs
        # Some LLM services return blocks of deltas at once, so we broadcast
        # the entire list in a single PubSub message for efficiency
        broadcast_event(server_state, {:llm_deltas, deltas})
      end,

      # Callback for complete message (either through delta or non-streamed messages)
      on_message_processed: fn _chain, message ->
        # Save and broadcast message (if callback configured)
        maybe_save_and_broadcast_message(server_state, message)
      end,

      # Callback for token usage information
      on_llm_token_usage: fn _chain, usage ->
        broadcast_event(server_state, {:llm_token_usage, usage})
      end
    }
  end

  defp execute_agent(%ServerState{} = server_state, callbacks) do
    # Execute agent with callbacks
    case Agent.execute(server_state.agent, server_state.state, callbacks: callbacks) do
      {:ok, new_state} ->
        # Broadcast state changes
        broadcast_state_changes(server_state, new_state)
        {:ok, new_state}

      {:interrupt, %State{} = interrupted_state, interrupt_data} ->
        # Broadcast state changes up to interrupt point
        broadcast_state_changes(server_state, interrupted_state)
        {:interrupt, interrupted_state, interrupt_data}

      {:error, reason} ->
        {:error, reason}
    end
  end

  defp resume_agent(server_state, decisions) do
    # Build callbacks for resume execution as well
    callbacks = build_llm_callbacks(server_state)

    case Agent.resume(server_state.agent, server_state.state, decisions, callbacks: callbacks) do
      {:ok, new_state} ->
        # Broadcast state changes
        broadcast_state_changes(server_state, new_state)
        {:ok, new_state}

      {:interrupt, interrupted_state, interrupt_data} ->
        # Another interrupt occurred during resume
        broadcast_state_changes(server_state, interrupted_state)
        {:interrupt, interrupted_state, interrupt_data}

      {:error, reason} ->
        {:error, reason}
    end
  end

  defp handle_execution_result({:ok, new_state}, server_state) do
    updated_state = %{
      server_state
      | status: :idle,
        state: new_state,
        error: nil
    }

    broadcast_event(updated_state, {:status_changed, :idle, nil})

    # Check if we should shutdown based on presence
    maybe_shutdown_if_no_viewers(updated_state)

    # Reset activity timer after completion
    updated_state = reset_inactivity_timer(updated_state)

    # Broadcast debug event for state update
    broadcast_debug_event(updated_state, {:agent_state_update, new_state})

    {:noreply, Map.delete(updated_state, :task)}
  end

  defp handle_execution_result({:interrupt, interrupted_state, interrupt_data}, server_state) do
    updated_state = %{
      server_state
      | status: :interrupted,
        state: interrupted_state,
        interrupt_data: interrupt_data
    }

    broadcast_event(updated_state, {:status_changed, :interrupted, interrupt_data})

    # Reset activity timer after interrupt
    updated_state = reset_inactivity_timer(updated_state)

    # Broadcast debug event for state update
    broadcast_debug_event(updated_state, {:agent_state_update, interrupted_state})

    {:noreply, Map.delete(updated_state, :task)}
  end

  defp handle_execution_result({:error, reason}, server_state) do
    updated_state = %{
      server_state
      | status: :error,
        error: reason
    }

    broadcast_event(updated_state, {:status_changed, :error, reason})

    # Reset activity timer after error
    updated_state = reset_inactivity_timer(updated_state)

    {:noreply, Map.delete(updated_state, :task)}
  end

  defp broadcast_state_changes(%ServerState{} = old_server_state, %State{} = new_state) do
    # Broadcast complete TODO snapshot if todos changed
    if old_server_state.state.todos != new_state.todos do
      broadcast_todos(old_server_state, new_state)
    end
  end

  defp maybe_shutdown_if_no_viewers(server_state) do
    case server_state.presence_config do
      %{enabled: true, presence_module: presence_mod, topic: topic} ->
        # Check who's viewing this agent's conversation
        viewers = presence_mod.list(topic)

        if map_size(viewers) == 0 do
          Logger.info(
            "Agent #{server_state.agent.agent_id} idle with no viewers, " <>
              "scheduling shutdown to free resources"
          )

          # Schedule shutdown after brief delay (let final events propagate)
          Process.send_after(self(), :shutdown_no_viewers, 1000)
        else
          Logger.debug(
            "Agent #{server_state.agent.agent_id} idle but has #{map_size(viewers)} " <>
              "viewers, keeping alive"
          )
        end

      _ ->
        # Presence tracking disabled, use standard inactivity timeout
        :ok
    end
  end

  defp broadcast_todos(%ServerState{} = server_state, %State{} = new_state) do
    # Broadcast complete snapshot of current TODOs
    broadcast_event(server_state, {:todos_updated, new_state.todos})
  end

  defp broadcast_messages(%ServerState{} = server_state, %State{} = new_state) do
    # Broadcast individual messages using the existing {:llm_message, message} event
    Enum.each(new_state.messages, fn message ->
      broadcast_event(server_state, {:llm_message, message})
    end)
  end

  # Save message via callback and broadcast display messages
  defp maybe_save_and_broadcast_message(server_state, message) do
    Logger.debug("maybe_save_and_broadcast_message called - callback: #{inspect(not is_nil(server_state.save_new_message_fn))}, conversation_id: #{inspect(server_state.conversation_id)}, message role: #{message.role}")

    if server_state.save_new_message_fn && server_state.conversation_id do
      Logger.debug("Calling save callback for conversation #{server_state.conversation_id}")

      try do
        case server_state.save_new_message_fn.(server_state.conversation_id, message) do
          {:ok, display_messages} when is_list(display_messages) ->
            Logger.debug("Successfully saved #{length(display_messages)} display messages, broadcasting...")
            # Broadcast each saved DisplayMessage
            Enum.each(display_messages, fn display_msg ->
              broadcast_event(server_state, {:display_message_saved, display_msg})
            end)

            # Also broadcast the original message event
            # This maintains backward compatibility and allows UI to handle message completion
            broadcast_event(server_state, {:llm_message, message})
            :ok

          {:error, reason} ->
            Logger.error("Failed to save message: #{inspect(reason)}")
            # When callback fails, don't broadcast any message events
            # The UI should handle the absence of events appropriately
            :ok

          other ->
            Logger.error("Invalid callback return format: #{inspect(other)}. Expected {:ok, list()} or {:error, term()}")
            # When callback returns invalid format, don't broadcast any message events
            :ok
        end
      rescue
        exception ->
          Logger.error("Callback raised exception: #{inspect(exception)}")
          # When callback raises exception, don't broadcast any message events
          :ok
      end
    else
      Logger.debug("No save callback configured, using original :llm_message event")
      # No save function configured, use original event
      broadcast_event(server_state, {:llm_message, message})
    end
  end

  defp broadcast_event(%ServerState{} = server_state, event) do
    case server_state.pubsub do
      {pubsub, pubsub_name} ->
        # Use "broadcast_from" to avoid sending to self
        pubsub.broadcast_from(pubsub_name, self(), server_state.topic, event)

      nil ->
        # No PubSub configured
        :ok
    end
  end

  # Does not broadcast if debug_pubsub is `nil`
  defp broadcast_debug_event(%ServerState{} = server_state, event) do
    case server_state.debug_pubsub do
      {debug_pubsub, debug_pubsub_name} ->
        # Use "broadcast_from" to avoid sending to self
        debug_pubsub.broadcast_from(debug_pubsub_name, self(), server_state.debug_topic, event)

      nil ->
        # No debug PubSub configured
        :ok
    end
  end

  ## Inactivity Timer Management

  # Reset the inactivity timer
  defp reset_inactivity_timer(state) do
    # Cancel existing timer if present
    state = cancel_inactivity_timer(state)

    # Don't schedule if timeout is nil or :infinity
    case state.inactivity_timeout do
      nil ->
        state

      :infinity ->
        state

      timeout when is_integer(timeout) and timeout > 0 ->
        timer_ref = Process.send_after(self(), :inactivity_timeout, timeout)

        %{state | inactivity_timer_ref: timer_ref, last_activity_at: DateTime.utc_now()}

      _ ->
        state
    end
  end

  # Cancel the current timer
  defp cancel_inactivity_timer(%ServerState{inactivity_timer_ref: nil} = state), do: state

  defp cancel_inactivity_timer(%ServerState{inactivity_timer_ref: ref} = state) do
    Process.cancel_timer(ref)

    # Flush the message if it was already sent
    receive do
      :inactivity_timeout -> :ok
    after
      0 -> :ok
    end

    %{state | inactivity_timer_ref: nil}
  end

  # Calculate time since last activity in milliseconds
  defp time_since(nil), do: nil

  defp time_since(datetime) do
    DateTime.diff(DateTime.utc_now(), datetime, :millisecond)
  end

  # Returns the middleware entries list as-is.
  #
  # The agent's middleware list already contains MiddlewareEntry structs
  # that were initialized by Middleware.init_middleware/1 during Agent.new/2.
  #
  # The list preserves order (needed for before_model/after_model hooks)
  # A registry map can be built from this list for O(1) message routing
  defp build_middleware_entries(middleware_list) when is_list(middleware_list) do
    # Middleware entries are already properly initialized by Middleware.init_middleware/1
    # Just return them as-is
    middleware_list
  end

  defp build_middleware_entries(_), do: []
end
