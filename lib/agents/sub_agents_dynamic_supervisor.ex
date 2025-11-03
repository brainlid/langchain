defmodule LangChain.Agents.SubAgentsDynamicSupervisor do
  @moduledoc """
  DynamicSupervisor for managing ephemeral sub-agent processes.

  SubAgentsDynamicSupervisor provides isolated execution environments for
  sub-agents spawned during task delegation. Each sub-agent runs independently
  with its own conversation context while sharing the parent's filesystem.

  ## Purpose

  - **Dynamic spawning**: Creates sub-agent processes on-demand
  - **Isolation**: Each sub-agent runs in its own process
  - **Clean lifecycle**: Sub-agents are automatically cleaned up after completion
  - **Fault tolerance**: Sub-agent crashes don't affect parent agent

  ## Usage

  This supervisor is automatically started by AgentSupervisor and typically
  not used directly. The SubAgent middleware interacts with it to spawn
  sub-agent processes.

  ## Examples

      # Started automatically by AgentSupervisor
      {:ok, sup_pid} = AgentSupervisor.start_link(agent: agent)

      # SubAgent middleware will use this supervisor internally
      # to spawn ephemeral sub-agent processes
  """

  use DynamicSupervisor

  alias LangChain.Agents.AgentRegistry

  @doc """
  Start the SubAgentsDynamicSupervisor.

  ## Options

  - `:agent_id` - The parent agent's ID (required)
  - `:registry` - Registry module name for process registration (optional, defaults to LangChain.Agents.Registry)

  ## Examples

      {:ok, pid} = SubAgentsDynamicSupervisor.start_link(agent_id: "agent-123")

      {:ok, pid} = SubAgentsDynamicSupervisor.start_link(
        agent_id: "agent-123",
        registry: MyApp.AgentRegistry
      )
  """
  @spec start_link(keyword()) :: Supervisor.on_start()
  def start_link(opts) do
    agent_id = Keyword.fetch!(opts, :agent_id)
    registry = Keyword.get(opts, :registry, AgentRegistry.default_registry())

    {_name, _opts} = Keyword.pop(opts, :name)
    name = via_tuple(registry, agent_id)

    DynamicSupervisor.start_link(__MODULE__, opts, name: name)
  end

  @doc """
  Get the SubAgentsDynamicSupervisor PID for an agent.

  Returns the PID if found, nil otherwise.

  ## Examples

      pid = SubAgentsDynamicSupervisor.whereis(MyApp.Registry, "agent-123")
  """
  @spec whereis(atom(), String.t()) :: pid() | nil
  def whereis(registry, agent_id) do
    AgentRegistry.whereis(registry, :sub_agents_supervisor, agent_id)
  end

  @impl true
  def init(_opts) do
    DynamicSupervisor.init(strategy: :one_for_one)
  end

  # Private helpers

  defp via_tuple(registry, agent_id) do
    AgentRegistry.via_tuple(registry, :sub_agents_supervisor, agent_id)
  end
end
