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

  ### File Events
  - `{:file_added, path, content}` - New file created
  - `{:file_updated, path, content}` - Existing file modified
  - `{:file_deleted, path}` - File removed

  ### Todo Events
  - `{:todo_created, todo}` - New todo item created
  - `{:todo_updated, todo}` - Todo status or content changed
  - `{:todo_deleted, todo_id}` - Todo item removed

  ### Status Events
  - `{:status_changed, :idle, nil}` - Server ready for work
  - `{:status_changed, :running, nil}` - Agent executing
  - `{:status_changed, :interrupted, interrupt_data}` - Awaiting human decision
  - `{:status_changed, :completed, final_state}` - Execution completed successfully
  - `{:status_changed, :error, reason}` - Execution failed

  ## Usage

      # Start a server
      {:ok, agent} = Agent.new(
        model: model,
        system_prompt: "You are a helpful assistant."
      )

      initial_state = State.new!(%{
        messages: [Message.new_user!("Write a hello world program")]
      })

      {:ok, pid} = AgentServer.start_link(
        agent: agent,
        initial_state: initial_state,
        name: {:global, :my_agent}
      )

      # Subscribe to events
      AgentServer.subscribe(pid)

      # Execute the agent
      :ok = AgentServer.execute(pid)

      # Listen for events
      receive do
        {:todo_created, todo} -> IO.inspect(todo, label: "New Todo")
        {:file_added, path, _content} -> IO.puts("File created: \#{path}")
        {:status_changed, :completed, final_state} -> IO.puts("Done!")
      end

      # Get current state
      state = AgentServer.get_state(pid)

  ## Human-in-the-Loop Example

      # Configure agent with interrupts
      {:ok, agent} = Agent.new(
        model: model,
        interrupt_on: %{"write_file" => true}
      )

      {:ok, pid} = AgentServer.start_link(agent: agent, initial_state: state)
      AgentServer.subscribe(pid)

      # Execute
      AgentServer.execute(pid)

      # Wait for interrupt
      receive do
        {:status_changed, :interrupted, interrupt_data} ->
          # Display interrupt_data.action_requests to user
          decisions = get_user_decisions(interrupt_data)
          AgentServer.resume(pid, decisions)
      end

      # Wait for completion
      receive do
        {:status_changed, :completed, final_state} -> :ok
      end
  """

  use GenServer
  require Logger

  alias LangChain.Agents.{Agent, State}

  @typedoc "Server reference for calls"
  @type server :: pid() | atom() | {:global, term()} | {:via, module(), term()}

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
      :error
    ]

    @type t :: %__MODULE__{
            agent: Agent.t(),
            state: State.t(),
            status: :idle | :running | :interrupted | :completed | :error,
            pubsub: module() | nil,
            topic: String.t(),
            interrupt_data: map() | nil,
            error: term() | nil
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

  ## Examples

      {:ok, pid} = AgentServer.start_link(
        agent: agent,
        initial_state: state,
        name: :my_agent
      )
  """
  def start_link(opts) do
    {name, opts} = Keyword.pop(opts, :name)

    if name do
      GenServer.start_link(__MODULE__, opts, name: name)
    else
      GenServer.start_link(__MODULE__, opts)
    end
  end

  @doc """
  Subscribe to events from this AgentServer.

  The calling process will receive messages for all events broadcast by this server.

  Returns `:ok` on success or `{:error, reason}` if PubSub is not configured.
  """
  @spec subscribe(server()) :: :ok | {:error, term()}
  def subscribe(server) do
    case GenServer.call(server, :get_pubsub_info) do
      {pubsub, pubsub_name, topic} ->
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

      :ok = AgentServer.execute(pid)
  """
  @spec execute(server()) :: :ok | {:error, term()}
  def execute(server) do
    GenServer.call(server, :execute, :infinity)
  end

  @doc """
  Resume agent execution after a human-in-the-loop interrupt.

  ## Parameters

  - `server` - The server process
  - `decisions` - List of decision maps from human reviewer (see `Agent.resume/3`)

  ## Examples

      decisions = [
        %{type: :approve},
        %{type: :edit, arguments: %{"path" => "safe.txt"}},
        %{type: :reject}
      ]

      :ok = AgentServer.resume(pid, decisions)
  """
  @spec resume(server(), list(map())) :: :ok | {:error, term()}
  def resume(server, decisions) when is_list(decisions) do
    GenServer.call(server, {:resume, decisions}, :infinity)
  end

  @doc """
  Get the current state of the agent.

  Returns the current State struct.
  """
  @spec get_state(server()) :: State.t()
  def get_state(server) do
    GenServer.call(server, :get_state)
  end

  @doc """
  Get the current status of the server.

  Returns one of: `:idle`, `:running`, `:interrupted`, `:completed`, `:error`
  """
  @spec get_status(server()) :: status()
  def get_status(server) do
    GenServer.call(server, :get_status)
  end

  @doc """
  Get server info including status, state, and any error or interrupt data.

  Returns a map with:
  - `:status` - Current status
  - `:state` - Current State
  - `:interrupt_data` - Interrupt data if status is `:interrupted`
  - `:error` - Error reason if status is `:error`
  """
  @spec get_info(server()) :: map()
  def get_info(server) do
    GenServer.call(server, :get_info)
  end

  @doc """
  Stop the AgentServer.

  ## Examples

      :ok = AgentServer.stop(pid)
  """
  @spec stop(server()) :: :ok
  def stop(server) do
    GenServer.stop(server)
  end

  ## Server Callbacks

  @impl true
  def init(opts) do
    agent = Keyword.fetch!(opts, :agent)
    initial_state = Keyword.get(opts, :initial_state, State.new!())
    pubsub = Keyword.get(opts, :pubsub, default_pubsub())
    pubsub_name = Keyword.get(opts, :pubsub_name, :langchain_pubsub)
    id = Keyword.get(opts, :id, generate_id())

    topic = "agent_server:#{id}"

    server_state = %ServerState{
      agent: agent,
      state: initial_state,
      status: :idle,
      pubsub: if(pubsub, do: {pubsub, pubsub_name}, else: nil),
      topic: topic,
      interrupt_data: nil,
      error: nil
    }

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
    # Transition to running
    new_state = %{server_state | status: :running}
    broadcast_event(new_state, {:status_changed, :running, nil})

    # Start async execution
    task =
      Task.async(fn ->
        execute_agent(new_state)
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
  def handle_call({:resume, decisions}, _from, %ServerState{status: :interrupted} = server_state) do
    # Transition back to running
    new_state = %{server_state | status: :running, interrupt_data: nil}
    broadcast_event(new_state, {:status_changed, :running, nil})

    # Resume execution async
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

  ## Private Functions

  defp execute_agent(server_state) do
    # Track the state before execution for comparison
    old_state = server_state.state

    case Agent.execute(server_state.agent, server_state.state) do
      {:ok, new_state} ->
        # Broadcast state changes
        broadcast_state_changes(server_state, old_state, new_state)
        {:ok, new_state}

      {:interrupt, interrupted_state, interrupt_data} ->
        # Broadcast state changes up to interrupt point
        broadcast_state_changes(server_state, old_state, interrupted_state)
        {:interrupt, interrupted_state, interrupt_data}

      {:error, reason} ->
        {:error, reason}
    end
  end

  defp resume_agent(server_state, decisions) do
    old_state = server_state.state

    case Agent.resume(server_state.agent, server_state.state, decisions) do
      {:ok, new_state} ->
        # Broadcast state changes
        broadcast_state_changes(server_state, old_state, new_state)
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

    {:noreply, Map.delete(updated_state, :task)}
  end

  defp handle_execution_result({:error, reason}, server_state) do
    updated_state = %{
      server_state
      | status: :error,
        error: reason
    }

    broadcast_event(updated_state, {:status_changed, :error, reason})

    {:noreply, Map.delete(updated_state, :task)}
  end

  defp broadcast_state_changes(server_state, old_state, new_state) do
    # Broadcast file changes
    broadcast_file_changes(server_state, old_state.files, new_state.files)

    # Broadcast todo changes
    broadcast_todo_changes(server_state, old_state.todos, new_state.todos)
  end

  defp broadcast_file_changes(server_state, old_files, new_files) do
    # Find new or updated files
    Enum.each(new_files, fn {path, file_data} ->
      old_file_data = Map.get(old_files, path)
      content = State.extract_file_content(file_data)

      cond do
        # New file
        old_file_data == nil ->
          broadcast_event(server_state, {:file_added, path, content})

        # File updated (content changed)
        State.extract_file_content(old_file_data) != content ->
          broadcast_event(server_state, {:file_updated, path, content})

        # No change
        true ->
          :ok
      end
    end)

    # Find deleted files
    old_paths = MapSet.new(Map.keys(old_files))
    new_paths = MapSet.new(Map.keys(new_files))
    deleted_paths = MapSet.difference(old_paths, new_paths)

    Enum.each(deleted_paths, fn path ->
      broadcast_event(server_state, {:file_deleted, path})
    end)
  end

  defp broadcast_todo_changes(server_state, old_todos, new_todos) do
    # Convert to maps keyed by ID for easier comparison
    old_todos_map = Map.new(old_todos, fn todo -> {todo.id, todo} end)
    new_todos_map = Map.new(new_todos, fn todo -> {todo.id, todo} end)

    # Find new or updated todos
    Enum.each(new_todos, fn todo ->
      old_todo = Map.get(old_todos_map, todo.id)

      cond do
        # New todo
        old_todo == nil ->
          broadcast_event(server_state, {:todo_created, todo})

        # Todo updated (status or content changed)
        old_todo != todo ->
          broadcast_event(server_state, {:todo_updated, todo})

        # No change
        true ->
          :ok
      end
    end)

    # Find deleted todos
    old_ids = MapSet.new(Map.keys(old_todos_map))
    new_ids = MapSet.new(Map.keys(new_todos_map))
    deleted_ids = MapSet.difference(old_ids, new_ids)

    Enum.each(deleted_ids, fn todo_id ->
      broadcast_event(server_state, {:todo_deleted, todo_id})
    end)
  end

  defp broadcast_event(server_state, event) do
    case server_state.pubsub do
      {pubsub, pubsub_name} ->
        # Use broadcast_from to avoid sending to self
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
end
