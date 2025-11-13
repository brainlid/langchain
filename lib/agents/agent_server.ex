defmodule LangChain.Agents.AgentServer do
  @moduledoc """
  GenServer that wraps a DeepAgent and its State, managing execution lifecycle
  and broadcasting events via PubSub.

  The AgentServer provides:
  - Asynchronous agent execution
  - State management and tracking
  - Event broadcasting for UI updates
  - Human-in-the-loop interrupt handling

  ## Events

  The server broadcasts events on the topic `"agent_server:\#{server_id}"`:

  ### Todo Events
  - `{:todos_updated, todos}` - Complete snapshot of current TODO list

  ### Status Events
  - `{:status_changed, :idle, nil}` - Server ready for work
  - `{:status_changed, :running, nil}` - Agent executing
  - `{:status_changed, :interrupted, interrupt_data}` - Awaiting human decision
  - `{:status_changed, :completed, final_state}` - Execution completed successfully
  - `{:status_changed, :error, reason}` - Execution failed

  ### LLM Streaming Events
  - `{:llm_deltas, [%MessageDelta{}]}` - Streaming tokens/deltas received (list of deltas)
  - `{:llm_message, %Message{}}` - Complete message received and processed from LLM
  - `{:llm_token_usage, %TokenUsage{}}` - Token usage information
  - `{:tool_response, %Message{}}` - Tool execution result (optional)

  **Note**: File events are NOT broadcast by AgentServer. Files are managed by
  `FileSystemServer` which provides its own event handling mechanism.

  ## Usage

      # Start a server
      {:ok, agent} = Agent.new(
        agent_id: "my-agent-1",
        model: model,
        system_prompt: "You are a helpful assistant."
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

      # Get current state
      state = AgentServer.get_state("my-agent-1")

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

  @registry LangChain.Agents.Registry

  @typedoc "Status of the agent server"
  @type status :: :idle | :running | :interrupted | :completed | :error

  defmodule ServerState do
    @moduledoc false
    defstruct [
      :agent,
      :state,
      :status,
      :pubsub,
      :topic,
      :interrupt_data,
      :error,
      :inactivity_timeout,
      :inactivity_timer_ref,
      :last_activity_at,
      :shutdown_delay,
      :task
    ]

    @type t :: %__MODULE__{
            agent: Agent.t(),
            state: State.t(),
            status: :idle | :running | :interrupted | :completed | :error,
            pubsub: module() | nil,
            topic: String.t(),
            interrupt_data: map() | nil,
            error: term() | nil,
            inactivity_timeout: pos_integer() | nil | :infinity,
            inactivity_timer_ref: reference() | nil,
            last_activity_at: DateTime.t() | nil,
            shutdown_delay: pos_integer() | nil,
            task: Task.t() | nil
          }
  end

  ## Client API

  @doc """
  Start an AgentServer.

  ## Options

  - `:agent` - The Agent struct (required)
  - `:initial_state` - Initial State (default: empty state)
  - `:pubsub` - PubSub module to use (default: Phoenix.PubSub if available, nil otherwise)
  - `:pubsub_name` - Name of the PubSub instance (default: :langchain_pubsub)
  - `:name` - Server name registration
  - `:id` - Unique identifier for the server (default: auto-generated)
  - `:inactivity_timeout` - Timeout in milliseconds for automatic shutdown due to inactivity (default: 300_000 - 5 minutes)
    Set to `nil` or `:infinity` to disable automatic shutdown
  - `:shutdown_delay` - Delay in milliseconds to allow the supervisor to gracefully stop all children (default: 5000)

  ## Examples

      {:ok, pid} = AgentServer.start_link(
        agent: agent,
        initial_state: state,
        name: :my_agent
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
  """
  def start_link(opts) do
    {name, opts} = Keyword.pop(opts, :name, __MODULE__)

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
  - `decisions` - List of decision maps from human reviewer (see `Agent.resume/3`)

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

  @doc """
  Get the current state of the agent.

  Returns the current State struct.

  ## Examples

      state = AgentServer.get_state("my-agent-1")
  """
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
  - `:completed` or `:error` → `:idle` (ready for new execution)
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
  Stop the AgentServer.

  ## Examples

      :ok = AgentServer.stop("my-agent-1")
  """
  @spec stop(String.t()) :: :ok
  def stop(agent_id) do
    GenServer.stop(get_name(agent_id))
  end

  ## Server Callbacks

  @impl true
  def init(opts) do
    # Trap exits to ensure terminate/2 is called for graceful shutdown
    Process.flag(:trap_exit, true)

    agent = Keyword.fetch!(opts, :agent)
    initial_state = Keyword.get(opts, :initial_state) || State.new!()
    # allow pubsub to be explicitly set to nil to disable it
    pubsub = Keyword.get(opts, :pubsub, default_pubsub())
    pubsub_name = Keyword.get(opts, :pubsub_name) || :langchain_pubsub
    id = Keyword.get(opts, :id) || generate_id()
    # allow a nil value to disable the timeout
    inactivity_timeout = Keyword.get(opts, :inactivity_timeout, 300_000)
    shutdown_delay = Keyword.get(opts, :shutdown_delay, 5_000)

    topic = "agent_server:#{id}"

    server_state = %ServerState{
      agent: agent,
      state: initial_state,
      status: :idle,
      pubsub: if(pubsub, do: {pubsub, pubsub_name}, else: nil),
      topic: topic,
      interrupt_data: nil,
      error: nil,
      inactivity_timeout: inactivity_timeout,
      inactivity_timer_ref: nil,
      last_activity_at: DateTime.utc_now(),
      shutdown_delay: shutdown_delay
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
    # Shutdown the running task
    Task.shutdown(task, :brutal_kill)

    # Transition to completed status
    new_state = %{server_state | status: :completed}
    broadcast_event(new_state, {:status_changed, :completed, server_state.state})

    # Reset inactivity timer after cancellation
    new_state = reset_inactivity_timer(new_state)

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

    # Transition to idle if we were completed/error to allow new execution
    new_status =
      case server_state.status do
        :completed -> :idle
        :error -> :idle
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

    {:reply, :ok, updated_server_state}
  end

  @impl true
  def handle_call(:reset, _from, server_state) do
    # Reset the filesystem first (clears memory files, unloads persisted files)
    agent_id = server_state.agent.agent_id
    :ok = LangChain.Agents.FileSystemServer.reset(agent_id)

    # Reset the agent state (clears messages, todos)
    reset_state = State.reset(server_state.state)

    # Transition to idle if we were completed/error
    new_status =
      case server_state.status do
        :completed -> :idle
        :error -> :idle
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

    {:reply, :ok, updated_server_state}
  end

  @impl true
  def handle_info({ref, result}, server_state) when is_reference(ref) do
    # Task completed
    Process.demonitor(ref, [:flush])

    handle_execution_result(result, server_state)
  end

  @impl true
  def handle_info({:DOWN, _ref, :process, _pid, :normal}, server_state) do
    # Task process exited normally, already handled in handle_info above
    {:noreply, server_state}
  end

  @impl true
  def handle_info({:DOWN, _ref, :process, _pid, reason}, server_state) do
    # Task crashed
    Logger.error("Agent execution task crashed: #{inspect(reason)}")

    new_state = %{server_state | status: :error, error: reason}
    broadcast_event(new_state, {:status_changed, :error, reason})

    {:noreply, Map.delete(new_state, :task)}
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
        Logger.debug("AgentSupervisor for agent #{agent_id} acknowledged shutdown request")

      {:error, :not_found} ->
        Logger.warning("AgentSupervisor for agent #{agent_id} was not found, stopping self")
    end

    # Let the supervisor tree shutdown take care of it
    {:noreply, server_state}
  end

  @impl true
  def terminate(reason, server_state) do
    Logger.debug("AgentServer terminating: #{inspect(reason)}")

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
        broadcast_event(server_state, {:llm_message, message})
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

      {:error, reason} ->
        {:error, reason}
    end
  end

  defp handle_execution_result({:ok, new_state}, server_state) do
    updated_state = %{
      server_state
      | status: :completed,
        state: new_state,
        error: nil
    }

    broadcast_event(updated_state, {:status_changed, :completed, new_state})

    # Reset activity timer after completion
    updated_state = reset_inactivity_timer(updated_state)

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

  defp default_pubsub do
    # Try to use Phoenix.PubSub if available
    Code.ensure_loaded?(Phoenix.PubSub) && Phoenix.PubSub
  end

  defp generate_id do
    :crypto.strong_rand_bytes(16)
    |> Base.url_encode64(padding: false)
    |> String.slice(0, 22)
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
end
