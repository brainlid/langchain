defmodule LangChain.Agents.SubAgentServer do
  @moduledoc """
  GenServer wrapper for SubAgent providing blocking API.

  ## Simple Design

  SubAgentServer is a **simple wrapper** that:
  1. **Holds a SubAgent struct** in its state
  2. Provides a **blocking API** for execute/resume
  3. Uses **Registry** for named access (via sub-agent ID)
  4. **NO PubSub** (simpler than AgentServer)
  5. **NO auto-shutdown** - lifecycle managed by parent agent

  The GenServer state is SIMPLE - just the SubAgent struct! Everything is
  delegated to SubAgent functions.

  ## New Architecture

  The SubAgent struct HOLDS the execution state including the LLMChain.
  This makes the server's job trivial:
  - Store the SubAgent struct
  - Delegate execute/resume to SubAgent module
  - Return results

  ## Lifecycle

  1. **Spawned** - Created by task tool under SubAgentsDynamicSupervisor
  2. **Execute** - Tool calls `execute/1`, blocking until completion or interrupt
  3. **Interrupt** (optional) - Returns `{:interrupt, interrupt_data}` if HITL needed
  4. **Resume** (optional) - Tool calls `resume/2` with decisions, blocks again
  5. **Complete** - Returns `{:ok, final_result}` when done
  6. **Shutdown** - Cleaned up when parent agent terminates

  ## Usage

      # Create a SubAgent struct
      subagent = SubAgent.new_from_config(
        parent_agent_id: "main-agent",
        instructions: "Research renewable energy",
        agent_config: agent,
        parent_state: parent_state
      )

      # Start the server
      {:ok, _pid} = SubAgentServer.start_link(subagent: subagent)

      # Execute synchronously (blocks until completion or interrupt)
      case SubAgentServer.execute(subagent.id) do
        {:ok, final_result} ->
          {:ok, final_result}

        {:interrupt, interrupt_data} ->
          # SubAgent needs HITL approval
          # Propagate interrupt to parent
          {:interrupt, %{
            type: :subagent_hitl,
            sub_agent_id: subagent.id,
            interrupt_data: interrupt_data
          }}

        {:error, reason} ->
          {:error, reason}
      end

      # Resume after user provides decisions
      case SubAgentServer.resume(subagent.id, decisions) do
        {:ok, final_result} -> {:ok, final_result}
        {:interrupt, interrupt_data} -> # Another interrupt
        {:error, reason} -> {:error, reason}
      end

  ## Supervision

  SubAgentServers are supervised by `SubAgentsDynamicSupervisor` with
  `:temporary` restart strategy. If a SubAgent crashes:
  - The supervisor logs the crash
  - The blocking call receives an exit signal
  - The parent agent can handle the error
  - No automatic restart (SubAgents are ephemeral)
  """

  use GenServer
  require Logger

  alias LangChain.Agents.SubAgent

  @registry LangChain.Agents.Registry

  defmodule ServerState do
    @moduledoc false
    defstruct [:subagent]

    @type t :: %__MODULE__{
            subagent: SubAgent.t()
          }
  end

  ## Client API

  @doc """
  Start a SubAgentServer.

  ## Options

  - `:subagent` - The SubAgent struct (required)

  ## Examples

      subagent = SubAgent.new_from_config(
        parent_agent_id: "main-agent",
        instructions: "Research renewable energy",
        agent_config: agent,
        parent_state: parent_state
      )

      {:ok, pid} = SubAgentServer.start_link(subagent: subagent)
  """
  def start_link(opts) do
    subagent = Keyword.fetch!(opts, :subagent)
    name = get_name(subagent.id)
    GenServer.start_link(__MODULE__, opts, name: name)
  end

  @doc """
  Get the registry name for a SubAgent.

  ## Examples

      name = SubAgentServer.get_name("main-agent-sub-1")
  """
  @spec get_name(String.t()) :: {:via, Registry, {Registry, tuple()}}
  def get_name(sub_agent_id) when is_binary(sub_agent_id) do
    {:via, Registry, {@registry, {:sub_agent, sub_agent_id}}}
  end

  @doc """
  Find the PID of a SubAgent by ID.

  Returns `nil` if the SubAgent doesn't exist.

  ## Examples

      pid = SubAgentServer.whereis("main-agent-sub-1")
  """
  @spec whereis(String.t()) :: pid() | nil
  def whereis(sub_agent_id) when is_binary(sub_agent_id) do
    case Registry.lookup(@registry, {:sub_agent, sub_agent_id}) do
      [{pid, _}] -> pid
      [] -> nil
    end
  end

  @doc """
  Execute the SubAgent synchronously.

  This function blocks until the SubAgent either:
  - Completes successfully: `{:ok, final_result}`
  - Encounters an HITL interrupt: `{:interrupt, interrupt_data}`
  - Fails with an error: `{:error, reason}`

  **Important**: This uses `:infinity` timeout because SubAgents may perform
  multiple LLM calls and tool executions. The blocking is intentional - the
  caller waits for the SubAgent to finish.

  ## Examples

      case SubAgentServer.execute("main-agent-sub-1") do
        {:ok, final_result} -> handle_completion(final_result)
        {:interrupt, interrupt_data} -> propagate_interrupt(interrupt_data)
        {:error, reason} -> handle_error(reason)
      end
  """
  @spec execute(String.t()) :: {:ok, String.t()} | {:interrupt, map()} | {:error, term()}
  def execute(sub_agent_id) when is_binary(sub_agent_id) do
    GenServer.call(get_name(sub_agent_id), :execute, :infinity)
  end

  @doc """
  Resume the SubAgent after an HITL interrupt.

  This function blocks until the SubAgent either:
  - Completes successfully: `{:ok, final_result}`
  - Encounters another HITL interrupt: `{:interrupt, interrupt_data}`
  - Fails with an error: `{:error, reason}`

  ## Parameters

  - `sub_agent_id` - The SubAgent identifier
  - `decisions` - List of decision maps from human reviewer (see `LangChain.Agent.resume/3`)

  ## Examples

      decisions = [
        %{type: :approve},
        %{type: :edit, arguments: %{"path" => "safe.txt"}}
      ]

      case SubAgentServer.resume("main-agent-sub-1", decisions) do
        {:ok, final_result} -> handle_completion(final_result)
        {:interrupt, interrupt_data} -> propagate_interrupt(interrupt_data)
        {:error, reason} -> handle_error(reason)
      end
  """
  @spec resume(String.t(), list(map())) ::
          {:ok, String.t()} | {:interrupt, map()} | {:error, term()}
  def resume(sub_agent_id, decisions) when is_binary(sub_agent_id) and is_list(decisions) do
    GenServer.call(get_name(sub_agent_id), {:resume, decisions}, :infinity)
  end

  @doc """
  Get the current status of the SubAgent.

  Returns one of: `:idle`, `:running`, `:interrupted`, `:completed`, `:error`

  ## Examples

      status = SubAgentServer.get_status("main-agent-sub-1")
  """
  @spec get_status(String.t()) :: atom()
  def get_status(sub_agent_id) when is_binary(sub_agent_id) do
    GenServer.call(get_name(sub_agent_id), :get_status)
  end

  @doc """
  Get the current SubAgent struct.

  **Note**: This is primarily for debugging. In normal operation, the SubAgent
  should stay encapsulated in the process.

  ## Examples

      subagent = SubAgentServer.get_subagent("main-agent-sub-1")
  """
  @spec get_subagent(String.t()) :: SubAgent.t()
  def get_subagent(sub_agent_id) when is_binary(sub_agent_id) do
    GenServer.call(get_name(sub_agent_id), :get_subagent)
  end

  ## Server Callbacks

  @impl true
  def init(opts) do
    subagent = Keyword.fetch!(opts, :subagent)

    server_state = %ServerState{
      subagent: subagent
    }

    Logger.debug(
      "SubAgentServer started for #{subagent.id} (parent: #{subagent.parent_agent_id})"
    )

    {:ok, server_state}
  end

  @impl true
  def handle_call(:execute, _from, %ServerState{subagent: subagent} = server_state) do
    Logger.debug("SubAgentServer executing #{subagent.id}")

    # Delegate to SubAgent.execute
    case SubAgent.execute(subagent) do
      {:ok, completed_subagent} ->
        Logger.debug("SubAgentServer #{subagent.id} completed successfully")

        # Extract result - returns {:ok, result} or {:error, reason}
        case SubAgent.extract_result(completed_subagent) do
          {:ok, result} ->
            new_state = %{server_state | subagent: completed_subagent}
            {:reply, {:ok, result}, new_state}

          {:error, reason} ->
            Logger.error(
              "SubAgentServer #{subagent.id} result extraction error: #{inspect(reason)}"
            )

            new_state = %{server_state | subagent: completed_subagent}
            {:reply, {:error, reason}, new_state}
        end

      {:interrupt, interrupted_subagent} ->
        # SubAgent hit HITL interrupt
        Logger.debug("SubAgentServer #{subagent.id} interrupted")
        new_state = %{server_state | subagent: interrupted_subagent}
        {:reply, {:interrupt, interrupted_subagent.interrupt_data}, new_state}

      {:error, error_subagent} ->
        Logger.error(
          "SubAgentServer #{subagent.id} execution error: #{inspect(error_subagent.error)}"
        )

        new_state = %{server_state | subagent: error_subagent}

        {:reply, {:error, error_subagent.error}, new_state}
    end
  end

  @impl true
  def handle_call({:resume, decisions}, _from, %ServerState{subagent: subagent} = server_state) do
    Logger.debug("SubAgentServer resuming #{subagent.id} with decisions")

    # Delegate to SubAgent.resume
    case SubAgent.resume(subagent, decisions) do
      {:ok, completed_subagent} ->
        Logger.debug("SubAgentServer #{subagent.id} completed after resume")

        # Extract result - returns {:ok, result} or {:error, reason}
        case SubAgent.extract_result(completed_subagent) do
          {:ok, result} ->
            new_state = %{server_state | subagent: completed_subagent}
            {:reply, {:ok, result}, new_state}

          {:error, reason} ->
            Logger.error(
              "SubAgentServer #{subagent.id} result extraction error: #{inspect(reason)}"
            )

            new_state = %{server_state | subagent: completed_subagent}
            {:reply, {:error, reason}, new_state}
        end

      {:interrupt, interrupted_subagent} ->
        # Another interrupt (multiple HITL tools)
        Logger.debug("SubAgentServer #{subagent.id} interrupted again")
        new_state = %{server_state | subagent: interrupted_subagent}
        {:reply, {:interrupt, interrupted_subagent.interrupt_data}, new_state}

      {:error, %SubAgent{} = error_subagent} ->
        # Error is a SubAgent struct with error field
        Logger.error(
          "SubAgentServer #{subagent.id} resume error: #{inspect(error_subagent.error)}"
        )

        new_state = %{server_state | subagent: error_subagent}

        {:reply, {:error, error_subagent.error}, new_state}

      {:error, reason} ->
        # Error is a plain reason (e.g., invalid status)
        Logger.error("SubAgentServer #{subagent.id} resume error: #{inspect(reason)}")

        {:reply, {:error, reason}, server_state}
    end
  end

  @impl true
  def handle_call(:get_status, _from, %ServerState{subagent: subagent} = server_state) do
    {:reply, subagent.status, server_state}
  end

  @impl true
  def handle_call(:get_subagent, _from, %ServerState{subagent: subagent} = server_state) do
    {:reply, subagent, server_state}
  end
end
