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
  - `:registry` - Registry module name for process registration (optional, defaults to LangChain.Agents.Registry)
  - `:persistence_configs` - List of FileSystemConfig structs (optional, default: [])
  - `:initial_state` - Initial State for AgentServer (optional)
  - `:pubsub` - PubSub module for AgentServer (optional)
  - `:pubsub_name` - PubSub instance name (optional)

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

      # With custom registry
      {:ok, sup_pid} = AgentSupervisor.start_link(
        agent: agent,
        registry: MyApp.AgentRegistry
      )
  """

  use Supervisor

  alias LangChain.Agents.Agent
  alias LangChain.Agents.AgentRegistry
  alias LangChain.Agents.AgentServer
  alias LangChain.Agents.FileSystemServer
  alias LangChain.Agents.SubAgentsDynamicSupervisor
  alias LangChain.Agents.State

  @doc """
  Start the AgentSupervisor.

  ## Options

  - `:agent` - The Agent struct (required)
  - `:registry` - Registry module name for process registration (optional, defaults to LangChain.Agents.Registry)
  - `:persistence_configs` - List of FileSystemConfig structs (optional, default: [])
  - `:initial_state` - Initial State for AgentServer (optional)
  - `:pubsub` - PubSub module for AgentServer (optional)
  - `:pubsub_name` - PubSub instance name (optional)

  ## Examples

      # Using default registry
      {:ok, sup_pid} = AgentSupervisor.start_link(
        agent: agent,
        persistence_configs: [config]
      )

      # Using custom registry
      {:ok, sup_pid} = AgentSupervisor.start_link(
        agent: agent,
        registry: MyApp.AgentRegistry
      )
  """
  @spec start_link(keyword()) :: Supervisor.on_start()
  def start_link(config) do
    registry = Keyword.get(config, :registry, AgentRegistry.default_registry())
    agent = Keyword.fetch!(config, :agent)
    agent_id = agent.agent_id

    # Build via tuple for this supervisor
    name = AgentRegistry.via_tuple(registry, :agent_supervisor, agent_id)

    # Add registry to config so it's available in init/1
    config = Keyword.put(config, :registry, registry)

    Supervisor.start_link(__MODULE__, config, name: name)
  end

  @doc """
  Get the AgentSupervisor PID for an agent.

  ## Examples

      pid = AgentSupervisor.whereis(MyApp.Registry, "agent-123")
  """
  @spec whereis(atom(), String.t()) :: pid() | nil
  def whereis(registry, agent_id) do
    AgentRegistry.whereis(registry, :agent_supervisor, agent_id)
  end

  @impl true
  def init(config) do
    # Extract configuration
    agent = Keyword.fetch!(config, :agent)
    registry = Keyword.fetch!(config, :registry)
    persistence_configs = Keyword.get(config, :persistence_configs, [])
    initial_state = Keyword.get(config, :initial_state, State.new!())
    pubsub = Keyword.get(config, :pubsub)
    pubsub_name = Keyword.get(config, :pubsub_name)

    # Validate agent
    unless is_struct(agent, Agent) do
      raise ArgumentError, "`:agent` must be a LangChain.Agents.Agent struct"
    end

    # Extract agent_id from the agent
    agent_id = agent.agent_id

    unless is_binary(agent_id) and agent_id != "" do
      raise ArgumentError, "Agent must have a valid agent_id"
    end

    # Build child specifications with registry
    children = [
      # 1. FileSystemServer - starts first, survives AgentServer crashes
      {FileSystemServer,
       [
         agent_id: agent_id,
         registry: registry,
         persistence_configs: persistence_configs
       ]},

      # 2. AgentServer - manages agent execution
      {AgentServer,
       [
         agent: agent,
         registry: registry,
         initial_state: initial_state,
         pubsub: pubsub,
         pubsub_name: pubsub_name
       ]
       |> Enum.reject(fn {_k, v} -> is_nil(v) end)},

      # 3. SubAgentsDynamicSupervisor - for spawning sub-agents
      {SubAgentsDynamicSupervisor,
       agent_id: agent_id,
       registry: registry}
    ]

    # Use :rest_for_one strategy
    # - FileSystemServer crash: all restart
    # - AgentServer crash: AgentServer and SubAgentsDynamicSupervisor restart
    # - SubAgentsDynamicSupervisor crash: only itself restarts
    Supervisor.init(children, strategy: :rest_for_one)
  end
end
