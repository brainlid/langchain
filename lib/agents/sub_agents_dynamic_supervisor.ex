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

  @registry LangChain.Agents.Registry

  @doc """
  Start the SubAgentsDynamicSupervisor.

  ## Options

  - `:agent_id` - The parent agent's ID (required)
  - `:name` - Supervisor name registration (optional)

  ## Examples

      {:ok, pid} = SubAgentsDynamicSupervisor.start_link(agent_id: "agent-123")

      {:ok, pid} = SubAgentsDynamicSupervisor.start_link(
        agent_id: "agent-123",
        name: SubAgentsDynamicSupervisor.get_name("agent-123")
      )
  """
  @spec start_link(keyword()) :: Supervisor.on_start()
  def start_link(opts) do
    agent_id = Keyword.fetch!(opts, :agent_id)
    {name, _opts} = Keyword.pop(opts, :name, via_tuple(agent_id))

    DynamicSupervisor.start_link(__MODULE__, opts, name: name)
  end

  @doc """
  Get the SubAgentsDynamicSupervisor PID for an agent.

  Returns the PID if found, nil otherwise.

  ## Examples

      pid = SubAgentsDynamicSupervisor.whereis("agent-123")
  """
  @spec whereis(String.t()) :: pid() | nil
  def whereis(agent_id) do
    case Registry.lookup(@registry, {:sub_agents_supervisor, agent_id}) do
      [{pid, _}] -> pid
      [] -> nil
    end
  end

  @doc """
  Get the name of the SubAgentsDynamicSupervisor process for a specific agent.

  ## Examples

      name = SubAgentsDynamicSupervisor.get_name("agent-123")
      # => {:via, Registry, {LangChain.Agents.Registry, {:sub_agents_supervisor, "agent-123"}}}
  """
  def get_name(agent_id) do
    {:via, Registry, {@registry, {:sub_agents_supervisor, agent_id}}}
  end

  @impl true
  def init(_opts) do
    DynamicSupervisor.init(strategy: :one_for_one)
  end
end
