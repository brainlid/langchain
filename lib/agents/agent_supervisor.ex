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

  @doc """
  Start the AgentSupervisor.

  ## Options

  - `:agent` - The Agent struct (required)
  - `:persistence_configs` - List of FileSystemConfig structs (optional, default: [])
  - `:initial_state` - Initial State for AgentServer (optional)
  - `:pubsub` - PubSub module for AgentServer (optional)
  - `:pubsub_name` - PubSub instance name (optional)
  - `:name` - Supervisor name registration (optional)

  ## Examples

      {:ok, sup_pid} = AgentSupervisor.start_link(
        agent: agent,
        persistence_configs: [config]
      )

      {:ok, sup_pid} = AgentSupervisor.start_link(
        agent: agent,
        name: {:via, Registry, {MyRegistry, :agent_sup}}
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
         pubsub_name: pubsub_name
       ]
       |> Enum.reject(fn {_k, v} -> is_nil(v) end)},

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
