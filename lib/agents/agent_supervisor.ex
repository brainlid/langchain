defmodule LangChain.Agents.AgentSupervisor do
  @moduledoc """
  Custom supervisor for managing an Agent and its supporting infrastructure.

  AgentSupervisor coordinates the lifecycle of:
  - AgentServer - Agent execution and state management
  - SubAgentsDynamicSupervisor - Dynamic supervisor for spawning sub-agents

  ## Supervision Strategy

  Uses `:rest_for_one` strategy to ensure crash resilience:
  - If AgentServer crashes, SubAgentsDynamicSupervisor restarts
  - SubAgentsDynamicSupervisor crashes only affect itself

  ## Filesystem Integration

  Filesystems are now managed independently via FileSystemSupervisor.
  Agents reference filesystems through the `filesystem_scope` field in the Agent struct.
  See `LangChain.Agents.FileSystem` for filesystem lifecycle management.

  ## Configuration

  Accepts a keyword list with:
  - `:agent` - The Agent struct (required)
  - `:initial_state` - Initial State for AgentServer (optional)
  - `:pubsub` - PubSub configuration as `{module(), atom()}` tuple or `nil` (optional, default: nil)
  - `:shutdown_delay` - Delay in milliseconds to allow the supervisor to gracefully stop all children (optional, default: 5000)
  - `:conversation_id` - Optional conversation identifier for message persistence (optional, default: nil)
  - `:save_new_message_fn` - Optional callback function for persisting messages (optional, default: nil)

  ## Examples

      # Minimal configuration
      {:ok, agent} = Agent.new(
        agent_id: "my-agent",
        model: model,
        base_system_prompt: "You are a helpful assistant."
      )

      {:ok, sup_pid} = AgentSupervisor.start_link(agent: agent)

      # With initial state and PubSub
      initial_state = State.new!(%{
        messages: [Message.new_user!("Hello!")]
      })

      {:ok, sup_pid} = AgentSupervisor.start_link(
        agent: agent,
        initial_state: initial_state,
        pubsub: {Phoenix.PubSub, :my_app_pubsub}
      )

      # With external filesystem
      # Start filesystem separately
      {:ok, config} = FileSystemConfig.new(%{
        scope_key: {:user, 123},
        base_directory: "Memories",
        persistence_module: LangChain.Agents.FileSystem.Persistence.Disk,
        storage_opts: [path: "/data/users/123"]
      })
      {:ok, _fs_pid} = FileSystem.ensure_filesystem({:user, 123}, [config])

      # Create agent with filesystem reference
      {:ok, agent} = Agent.new(%{
        agent_id: "my-agent",
        model: model,
        filesystem_scope: {:user, 123}
      })

      {:ok, sup_pid} = AgentSupervisor.start_link(agent: agent)
  """

  use Supervisor
  require Logger

  alias LangChain.Agents.Agent
  alias LangChain.Agents.AgentServer
  alias LangChain.Agents.SubAgentsDynamicSupervisor
  alias LangChain.Agents.State

  @registry LangChain.Agents.Registry

  @doc """
  Get the name of the AgentSupervisor process for a specific agent.

  ## Examples

      name = AgentSupervisor.get_name("my-agent-1")
      AgentSupervisor.stop(name)
  """
  @spec get_name(String.t()) :: {:via, Registry, {Registry, String.t()}}
  def get_name(agent_id) when is_binary(agent_id) do
    {:via, Registry, {@registry, {:agent_supervisor, agent_id}}}
  end

  @doc """
  Stop the AgentSupervisor.

  ## Examples

      AgentSupervisor.stop("my-agent-1")
      AgentSupervisor.stop("my-agent-1", 10_000) # 10 second timeout

  """
  @spec stop(String.t(), timeout()) :: :ok | {:error, :not_found}
  def stop(agent_id, timeout \\ 5_000) do
    case Registry.lookup(@registry, {:agent_supervisor, agent_id}) do
      [{pid, _}] when is_pid(pid) ->
        Supervisor.stop(pid, :normal, timeout)

      [] ->
        {:error, :not_found}
    end
  end

  @doc """
  Start the AgentSupervisor.

  ## Options

  - `:agent` - The Agent struct (required)
  - `:initial_state` - Initial State for AgentServer (optional)
  - `:pubsub` - PubSub configuration as `{module(), atom()}` tuple or `nil` (optional, default: nil)
  - `:debug_pubsub` - Optional debug PubSub configuration as `{module(), atom()}` or `nil` (optional, default: nil)
  - `:inactivity_timeout` - Timeout in milliseconds for automatic shutdown (optional, default: 300_000 - 5 minutes)
    Set to `nil` or `:infinity` to disable automatic shutdown
  - `:name` - Supervisor name registration (optional)
  - `:shutdown_delay` - Delay in milliseconds to allow the supervisor to gracefully stop all children (optional, default: 5000)
  - `:conversation_id` - Optional conversation identifier for message persistence (optional, default: nil)
  - `:save_new_message_fn` - Optional callback function for persisting messages (optional, default: nil)

  ## Examples

      {:ok, sup_pid} = AgentSupervisor.start_link(
        agent: agent,
        pubsub: {Phoenix.PubSub, :my_app_pubsub}
      )

      {:ok, sup_pid} = AgentSupervisor.start_link(
        agent: agent,
        name: AgentSupervisor.get_name("agent-123")
      )

      # With custom inactivity timeout
      {:ok, sup_pid} = AgentSupervisor.start_link(
        agent: agent,
        inactivity_timeout: 600_000  # 10 minutes
      )

      # Disable automatic shutdown
      {:ok, sup_pid} = AgentSupervisor.start_link(
        agent: agent,
        inactivity_timeout: nil
      )

      # With debug pubsub
      {:ok, sup_pid} = AgentSupervisor.start_link(
        agent: agent,
        pubsub: {Phoenix.PubSub, :my_app_pubsub},
        debug_pubsub: {Phoenix.PubSub, :my_debug_pubsub}
      )
  """
  @spec start_link(keyword()) :: Supervisor.on_start()
  def start_link(config) do
    {name, config} = Keyword.pop(config, :name, __MODULE__)

    Supervisor.start_link(__MODULE__, config, name: name)
  end

  @doc """
  Start the AgentSupervisor and wait for the AgentServer to be ready.

  This is a synchronous version of `start_link/1` that waits for the AgentServer
  child process to be fully registered before returning. This prevents race
  conditions where callers try to interact with the AgentServer (e.g., subscribe,
  add_message) before it's ready.

  ## Why This Is Needed

  When `start_link/1` returns successfully, the supervisor is running but its
  children are still completing their initialization asynchronously. If you
  immediately try to call `AgentServer.get_pid/1` or `AgentServer.subscribe/1`,
  the AgentServer might not be registered yet, causing failures.

  This function solves that by:
  1. Starting the supervisor normally
  2. Polling for the AgentServer to be registered with exponential backoff
  3. Returning only when the AgentServer is confirmed ready

  ## Use Cases

  Use this function when you need immediate access to the AgentServer:
  - Web request handlers that need to subscribe or send messages right away
  - CLI tools that need synchronous agent startup
  - Test setups that require the agent to be fully ready

  Use regular `start_link/1` when:
  - Startup timing isn't critical
  - You can handle eventual consistency
  - You're starting many agents in parallel

  ## Options

  Same as `start_link/1`, plus:
  - `:startup_timeout` - Maximum time to wait for AgentServer readiness (default: 5000ms)

  ## Examples

      # Start and wait for agent to be ready
      {:ok, sup_pid} = AgentSupervisor.start_link_sync(
        agent: agent,
        pubsub: {Phoenix.PubSub, :my_app_pubsub}
      )

      # Agent is guaranteed to be ready - safe to use immediately
      :ok = AgentServer.subscribe(agent.agent_id)

      # Custom startup timeout
      {:ok, sup_pid} = AgentSupervisor.start_link_sync(
        agent: agent,
        startup_timeout: 10_000
      )

  ## Returns

  - `{:ok, supervisor_pid}` - Supervisor started and AgentServer is ready
  - `{:error, {:agent_startup_timeout, agent_id}}` - AgentServer failed to become ready
  - `{:error, reason}` - Supervisor failed to start
  """
  @spec start_link_sync(keyword()) :: {:ok, pid()} | {:error, term()}
  def start_link_sync(config) do
    # Extract startup_timeout before passing to start_link
    {startup_timeout, config} = Keyword.pop(config, :startup_timeout, 5_000)

    # Get agent_id for waiting
    agent = Keyword.fetch!(config, :agent)
    agent_id = agent.agent_id

    # Start supervisor normally
    case start_link(config) do
      {:ok, sup_pid} ->
        # Wait for AgentServer to be ready
        case wait_for_agent_ready(agent_id, startup_timeout) do
          :ok ->
            {:ok, sup_pid}

          {:error, :timeout} ->
            {:error, {:agent_startup_timeout, agent_id}}
        end

      {:error, {:already_started, sup_pid}} ->
        # Already running - verify AgentServer is ready
        case wait_for_agent_ready(agent_id, startup_timeout) do
          :ok ->
            {:ok, sup_pid}

          {:error, :timeout} ->
            {:error, {:agent_startup_timeout, agent_id}}
        end

      error ->
        error
    end
  end

  @impl true
  def init(config) do
    # Extract and validate agent first (before accessing agent_id)
    agent = Keyword.fetch!(config, :agent)

    # Validate agent
    unless is_struct(agent, Agent) do
      raise ArgumentError, "`:agent` must be a LangChain.Agents.Agent struct"
    end

    # Extract agent_id from the agent
    agent_id = agent.agent_id

    unless is_binary(agent_id) and agent_id != "" do
      raise ArgumentError, "Agent must have a valid agent_id"
    end

    # Extract remaining configuration
    initial_state = Keyword.get(config, :initial_state, State.new!(%{agent_id: agent.agent_id}))
    pubsub = Keyword.get(config, :pubsub)
    debug_pubsub = Keyword.get(config, :debug_pubsub)
    inactivity_timeout = Keyword.get(config, :inactivity_timeout, 300_000)
    shutdown_delay = Keyword.get(config, :shutdown_delay, 5000)
    presence_tracking = Keyword.get(config, :presence_tracking)
    conversation_id = Keyword.get(config, :conversation_id)
    save_new_message_fn = Keyword.get(config, :save_new_message_fn)
    presence_module = Keyword.get(config, :presence_module)

    # Build AgentServer options
    agent_server_opts = [
      agent: agent,
      initial_state: initial_state,
      inactivity_timeout: inactivity_timeout,
      shutdown_delay: shutdown_delay,
      id: agent_id,
      name: AgentServer.get_name(agent_id)
    ]

    # Add pubsub if provided
    agent_server_opts =
      if pubsub, do: Keyword.put(agent_server_opts, :pubsub, pubsub), else: agent_server_opts

    # Add debug_pubsub if provided
    agent_server_opts =
      if debug_pubsub,
        do: Keyword.put(agent_server_opts, :debug_pubsub, debug_pubsub),
        else: agent_server_opts

    # Add presence_tracking if provided
    agent_server_opts =
      if presence_tracking,
        do: Keyword.put(agent_server_opts, :presence_tracking, presence_tracking),
        else: agent_server_opts

    # Add conversation_id if provided
    agent_server_opts =
      if conversation_id,
        do: Keyword.put(agent_server_opts, :conversation_id, conversation_id),
        else: agent_server_opts

    # Add save_new_message_fn if provided
    agent_server_opts =
      if save_new_message_fn,
        do: Keyword.put(agent_server_opts, :save_new_message_fn, save_new_message_fn),
        else: agent_server_opts

    # Add presence_module if provided
    agent_server_opts =
      if presence_module,
        do: Keyword.put(agent_server_opts, :presence_module, presence_module),
        else: agent_server_opts

    # Build child specifications
    # Note: FileSystemServer is now managed independently via FileSystemSupervisor
    # Agents reference filesystems through the filesystem_scope field
    children = [
      # 1. AgentServer - manages agent execution
      {AgentServer, agent_server_opts},

      # 2. SubAgentsDynamicSupervisor - for spawning sub-agents
      {SubAgentsDynamicSupervisor, agent_id: agent_id}
    ]

    # Use :rest_for_one strategy
    # - AgentServer crash: SubAgentsDynamicSupervisor restarts
    # - SubAgentsDynamicSupervisor crash: only itself restarts
    Supervisor.init(children, strategy: :rest_for_one)
  end

  ## Private Helpers

  # Wait for the AgentServer to be registered and ready
  # Retries with exponential backoff up to the timeout
  defp wait_for_agent_ready(agent_id, timeout_ms) do
    deadline = System.monotonic_time(:millisecond) + timeout_ms
    do_wait_for_agent_ready(agent_id, deadline, 10, 0)
  end

  defp do_wait_for_agent_ready(agent_id, deadline, retry_delay_ms, attempt) do
    case AgentServer.get_pid(agent_id) do
      nil ->
        now = System.monotonic_time(:millisecond)
        time_left = deadline - now

        if now >= deadline do
          Logger.error(
            "AgentServer #{agent_id} failed to register within timeout (#{attempt} attempts)"
          )

          {:error, :timeout}
        else
          # Log first few attempts and then periodically
          if attempt < 3 or rem(attempt, 10) == 0 do
            Logger.debug(
              "AgentServer #{agent_id} not yet registered, retrying... (attempt #{attempt}, #{time_left}ms left)"
            )
          end

          # Sleep briefly and retry with exponential backoff (max 100ms)
          Process.sleep(retry_delay_ms)
          next_delay = min(retry_delay_ms * 2, 100)
          do_wait_for_agent_ready(agent_id, deadline, next_delay, attempt + 1)
        end

      pid when is_pid(pid) ->
        :ok
    end
  end
end
