defmodule LangChain.Agents.AgentRegistry do
  @moduledoc """
  Registry utilities for agent process naming and lookup.

  Provides helper functions for building via tuples and looking up
  agent-related processes in a configurable Registry.

  ## Usage

  Applications can define their own Registry and pass it to AgentSupervisor:

      # In your application.ex
      children = [
        {Registry, keys: :unique, name: MyApp.AgentRegistry},
        {AgentSupervisor, agent: agent, registry: MyApp.AgentRegistry}
      ]

  ## Default Registry

  If no registry is specified, the library uses `LangChain.Agents.Registry`
  as the default. Applications are responsible for starting this registry
  if they don't provide their own.

  ## Process Naming Pattern

  All agent processes are registered using tuple keys:
  - `{:agent_supervisor, agent_id}` - AgentSupervisor
  - `{:agent_server, agent_id}` - AgentServer
  - `{:file_system_server, agent_id}` - FileSystemServer
  - `{:sub_agents_supervisor, agent_id}` - SubAgentsDynamicSupervisor

  ## Examples

      # Build a via tuple
      via = AgentRegistry.via_tuple(MyApp.Registry, :agent_server, "agent-123")
      # => {:via, Registry, {MyApp.Registry, {:agent_server, "agent-123"}}}

      # Look up a process
      pid = AgentRegistry.whereis(MyApp.Registry, :agent_server, "agent-123")
      # => #PID<0.123.0> or nil
  """

  @default_registry LangChain.Agents.Registry

  @doc """
  Get the default registry name.

  This is used when no registry is explicitly configured.

  ## Examples

      iex> AgentRegistry.default_registry()
      LangChain.Agents.Registry
  """
  @spec default_registry() :: atom()
  def default_registry, do: @default_registry

  @doc """
  Build a via tuple for Registry-based process naming.

  ## Parameters

  - `registry_name` - The Registry module name (atom)
  - `component_type` - The type of component (atom), e.g., `:agent_server`
  - `agent_id` - The agent identifier (string)

  ## Examples

      iex> AgentRegistry.via_tuple(MyApp.Registry, :file_system_server, "agent-123")
      {:via, Registry, {MyApp.Registry, {:file_system_server, "agent-123"}}}
  """
  @spec via_tuple(atom(), atom(), String.t()) :: {:via, Registry, {atom(), {atom(), String.t()}}}
  def via_tuple(registry_name, component_type, agent_id)
      when is_atom(registry_name) and is_atom(component_type) and is_binary(agent_id) do
    {:via, Registry, {registry_name, {component_type, agent_id}}}
  end

  @doc """
  Look up a process in the registry.

  Returns the PID if the process is registered, `nil` otherwise.

  ## Parameters

  - `registry_name` - The Registry module name (atom)
  - `component_type` - The type of component (atom), e.g., `:agent_server`
  - `agent_id` - The agent identifier (string)

  ## Examples

      iex> AgentRegistry.whereis(MyApp.Registry, :file_system_server, "agent-123")
      #PID<0.123.0>

      iex> AgentRegistry.whereis(MyApp.Registry, :file_system_server, "nonexistent")
      nil
  """
  @spec whereis(atom(), atom(), String.t()) :: pid() | nil
  def whereis(registry_name, component_type, agent_id)
      when is_atom(registry_name) and is_atom(component_type) and is_binary(agent_id) do
    case Registry.lookup(registry_name, {component_type, agent_id}) do
      [{pid, _}] -> pid
      [] -> nil
    end
  end
end
