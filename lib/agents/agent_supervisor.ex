defmodule LangChain.Agents.AgentSupervisor do
  @moduledoc """
  Custom supervisor for managing an Agent and its supporting infrastructure.

  AgentSupervisor coordinates the lifecycle of:
  - FileSystemServer - Persistent virtual filesystem with ETS storage
  - AgentServer - Agent execution and state management
  - SubAgentsDynamicSupervisor - Dynamic supervisor for spawning sub-agents

  ## Supervision Strategy

  Uses `:rest_for_one` strategy to ensure crash resilience:
  - If FileSystemServer crashes, all children restart
  - If AgentServer crashes, only AgentServer and SubAgentsDynamicSupervisor restart
  - FileSystemServer (with ETS state) survives AgentServer crashes

  ## Configuration

  Accepts a keyword list with:
  - `:agent` - The Agent struct (required)
  - `:persistence_configs` - List of FileSystemConfig structs (optional, default: [])
  - `:initial_state` - Initial State for AgentServer (optional)
  - `:pubsub` - PubSub module for AgentServer (optional)
  - `:pubsub_name` - PubSub instance name (optional)
  - `:shutdown_delay` - Delay in milliseconds to allow the supervisor to gracefully stop all children (optional, default: 5000)

  ## Examples

      # Minimal configuration
      {:ok, agent} = Agent.new(
        agent_id: "my-agent",
        model: model,
        system_prompt: "You are a helpful assistant."
      )

      {:ok, sup_pid} = AgentSupervisor.start_link(agent: agent)

      # With persistence
      {:ok, config} = FileSystemConfig.new(%{
        base_directory: "Memories",
        persistence_module: LangChain.Agents.FileSystem.Persistence.Disk,
        storage_opts: [path: "/data/agents"]
      })

      {:ok, sup_pid} = AgentSupervisor.start_link(
        agent: agent,
        persistence_configs: [config]
      )

      # With initial state and PubSub
      initial_state = State.new!(%{
        messages: [Message.new_user!("Hello!")]
      })

      {:ok, sup_pid} = AgentSupervisor.start_link(
        agent: agent,
        initial_state: initial_state,
        pubsub: Phoenix.PubSub,
        pubsub_name: :my_app_pubsub
      )
  """

  use Supervisor

  alias LangChain.Agents.Agent
  alias LangChain.Agents.AgentServer
  alias LangChain.Agents.FileSystemServer
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
  - `:persistence_configs` - List of FileSystemConfig structs (optional, default: [])
  - `:initial_state` - Initial State for AgentServer (optional)
  - `:pubsub` - PubSub module for AgentServer (optional)
  - `:pubsub_name` - PubSub instance name (optional)
  - `:inactivity_timeout` - Timeout in milliseconds for automatic shutdown (optional, default: 300_000 - 5 minutes)
    Set to `nil` or `:infinity` to disable automatic shutdown
  - `:name` - Supervisor name registration (optional)
  - `:shutdown_delay` - Delay in milliseconds to allow the supervisor to gracefully stop all children (optional, default: 5000)

  ## Examples

      {:ok, sup_pid} = AgentSupervisor.start_link(
        agent: agent,
        persistence_configs: [config]
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
  """
  @spec start_link(keyword()) :: Supervisor.on_start()
  def start_link(config) do
    {name, config} = Keyword.pop(config, :name, __MODULE__)

    Supervisor.start_link(__MODULE__, config, name: name)
  end

  @impl true
  def init(config) do
    # Extract configuration
    agent = Keyword.fetch!(config, :agent)
    persistence_configs = Keyword.get(config, :persistence_configs, [])
    initial_state = Keyword.get(config, :initial_state, State.new!())
    pubsub = Keyword.get(config, :pubsub)
    pubsub_name = Keyword.get(config, :pubsub_name)
    inactivity_timeout = Keyword.get(config, :inactivity_timeout, 300_000)
    shutdown_delay = Keyword.get(config, :shutdown_delay, 5000)

    # Validate agent
    unless is_struct(agent, Agent) do
      raise ArgumentError, "`:agent` must be a LangChain.Agents.Agent struct"
    end

    # Extract agent_id from the agent
    agent_id = agent.agent_id

    unless is_binary(agent_id) and agent_id != "" do
      raise ArgumentError, "Agent must have a valid agent_id"
    end

    # Build child specifications
    children = [
      # 1. FileSystemServer - starts first, survives AgentServer crashes
      {FileSystemServer,
       [
         agent_id: agent_id,
         persistence_configs: persistence_configs
       ]},

      # 2. AgentServer - manages agent execution
      {AgentServer,
       [
         agent: agent,
         initial_state: initial_state,
         pubsub: pubsub,
         pubsub_name: pubsub_name,
         inactivity_timeout: inactivity_timeout,
         shutdown_delay: shutdown_delay,
         id: agent_id,
         name: AgentServer.get_name(agent_id)
       ]},

      # 3. SubAgentsDynamicSupervisor - for spawning sub-agents
      {SubAgentsDynamicSupervisor, agent_id: agent_id}
    ]

    # Use :rest_for_one strategy
    # - FileSystemServer crash: all restart
    # - AgentServer crash: AgentServer and SubAgentsDynamicSupervisor restart
    # - SubAgentsDynamicSupervisor crash: only itself restarts
    Supervisor.init(children, strategy: :rest_for_one)
  end
end
