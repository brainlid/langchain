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

  alias LangChain.Agents.AgentServer
  alias LangChain.Agents.SubAgent
  alias LangChain.TokenUsage

  @registry LangChain.Agents.Registry

  defmodule ServerState do
    @moduledoc false
    defstruct [:subagent, :started_at]

    @type t :: %__MODULE__{
            subagent: SubAgent.t(),
            # Monotonic time for duration calculation (milliseconds)
            started_at: integer() | nil
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
    started_at = System.monotonic_time(:millisecond)

    server_state = %ServerState{
      subagent: subagent,
      started_at: started_at
    }

    Logger.debug(
      "SubAgentServer started for #{subagent.id} (parent: #{subagent.parent_agent_id})"
    )

    # Broadcast subagent_started via parent AgentServer
    broadcast_subagent_event(server_state, {:subagent_started, build_started_metadata(subagent)})

    {:ok, server_state}
  end

  @impl true
  def handle_call(:execute, _from, %ServerState{subagent: subagent} = server_state) do
    # Broadcast status change to running
    broadcast_subagent_event(server_state, {:subagent_status_changed, :running})

    # Build callbacks for real-time message broadcasting
    callbacks = build_llm_callbacks(server_state)

    # Delegate to SubAgent.execute with callbacks
    case SubAgent.execute(subagent, callbacks: callbacks) do
      {:ok, completed_subagent} ->
        # Extract result - returns {:ok, result} or {:error, reason}
        case SubAgent.extract_result(completed_subagent) do
          {:ok, result} ->
            new_state = %{server_state | subagent: completed_subagent}

            # Broadcast completion event
            broadcast_subagent_event(
              new_state,
              {:subagent_completed, build_completed_metadata(new_state, result)}
            )

            {:reply, {:ok, result}, new_state}

          {:error, reason} ->
            Logger.error(
              "SubAgentServer: #{subagent.id} result extraction error: #{inspect(reason)}"
            )

            new_state = %{server_state | subagent: completed_subagent}

            # Broadcast error event
            broadcast_subagent_event(new_state, {:subagent_error, reason})

            {:reply, {:error, reason}, new_state}
        end

      {:interrupt, interrupted_subagent} ->
        # SubAgent hit HITL interrupt
        new_state = %{server_state | subagent: interrupted_subagent}

        # Broadcast interrupt status
        broadcast_subagent_event(new_state, {:subagent_status_changed, :interrupted})

        {:reply, {:interrupt, interrupted_subagent.interrupt_data}, new_state}

      {:error, error_subagent} ->
        Logger.error(
          "SubAgentServer: #{subagent.id} execution error: #{inspect(error_subagent.error)}"
        )

        new_state = %{server_state | subagent: error_subagent}

        # Broadcast error event
        broadcast_subagent_event(new_state, {:subagent_error, error_subagent.error})

        {:reply, {:error, error_subagent.error}, new_state}
    end
  end

  @impl true
  def handle_call({:resume, decisions}, _from, %ServerState{subagent: subagent} = server_state) do
    Logger.debug("SubAgentServer resuming #{subagent.id} with decisions")

    # Broadcast status change to running (resuming)
    broadcast_subagent_event(server_state, {:subagent_status_changed, :running})

    # Build callbacks for real-time message broadcasting
    callbacks = build_llm_callbacks(server_state)

    # Delegate to SubAgent.resume with callbacks
    case SubAgent.resume(subagent, decisions, callbacks: callbacks) do
      {:ok, completed_subagent} ->
        Logger.debug("SubAgentServer #{subagent.id} completed after resume")

        # Extract result - returns {:ok, result} or {:error, reason}
        case SubAgent.extract_result(completed_subagent) do
          {:ok, result} ->
            new_state = %{server_state | subagent: completed_subagent}

            # Broadcast completion event
            broadcast_subagent_event(
              new_state,
              {:subagent_completed, build_completed_metadata(new_state, result)}
            )

            {:reply, {:ok, result}, new_state}

          {:error, reason} ->
            Logger.error(
              "SubAgentServer #{subagent.id} result extraction error: #{inspect(reason)}"
            )

            new_state = %{server_state | subagent: completed_subagent}

            # Broadcast error event
            broadcast_subagent_event(new_state, {:subagent_error, reason})

            {:reply, {:error, reason}, new_state}
        end

      {:interrupt, interrupted_subagent} ->
        # Another interrupt (multiple HITL tools)
        Logger.debug("SubAgentServer #{subagent.id} interrupted again")
        new_state = %{server_state | subagent: interrupted_subagent}

        # Broadcast interrupt status
        broadcast_subagent_event(new_state, {:subagent_status_changed, :interrupted})

        {:reply, {:interrupt, interrupted_subagent.interrupt_data}, new_state}

      {:error, %SubAgent{} = error_subagent} ->
        # Error is a SubAgent struct with error field
        Logger.error(
          "SubAgentServer #{subagent.id} resume error: #{inspect(error_subagent.error)}"
        )

        new_state = %{server_state | subagent: error_subagent}

        # Broadcast error event
        broadcast_subagent_event(new_state, {:subagent_error, error_subagent.error})

        {:reply, {:error, error_subagent.error}, new_state}

      {:error, reason} ->
        # Error is a plain reason (e.g., invalid status)
        Logger.error("SubAgentServer #{subagent.id} resume error: #{inspect(reason)}")

        # Broadcast error event
        broadcast_subagent_event(server_state, {:subagent_error, reason})

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

  ## Private Helper Functions

  # Build callbacks for LLMChain that broadcast message events to the parent's debug PubSub.
  # This enables real-time visibility into sub-agent execution.
  defp build_llm_callbacks(%ServerState{} = server_state) do
    %{
      on_message_processed: fn _chain, message ->
        broadcast_subagent_event(server_state, {:subagent_llm_message, message})
      end
    }
  end

  # Broadcast an event via the parent AgentServer's debug topic.
  # Events are wrapped as {:subagent, subagent_id, event} and the AgentServer
  # will wrap them as {:agent, {:debug, {:subagent, ...}}} for consistent routing.
  #
  # This is a no-op if the parent AgentServer doesn't have debug_pubsub configured.
  defp broadcast_subagent_event(%ServerState{subagent: subagent}, event) do
    parent_id = subagent.parent_agent_id
    subagent_id = subagent.id

    # Wrap event with subagent identification
    wrapped_event = {:subagent, subagent_id, event}

    # Use parent's debug broadcast - this is a no-op if parent has no debug_pubsub
    AgentServer.publish_debug_event_from(parent_id, wrapped_event)
  end

  # Build metadata for subagent_started event
  defp build_started_metadata(subagent) do
    %{
      id: subagent.id,
      parent_id: subagent.parent_agent_id,
      name: get_subagent_name(subagent),
      instructions: extract_instructions(subagent),
      tools: extract_tools(subagent),
      middleware: extract_middleware(subagent),
      model: get_model_name(subagent)
    }
  end

  # Build metadata for subagent_completed event
  # Note: messages are no longer included here as they are now broadcast in real-time
  # via {:subagent_llm_message, message} events during execution
  defp build_completed_metadata(%ServerState{subagent: subagent, started_at: started_at}, result) do
    duration_ms =
      if started_at do
        System.monotonic_time(:millisecond) - started_at
      else
        nil
      end

    # Get messages for token usage extraction only
    messages =
      if subagent.chain do
        subagent.chain.messages
      else
        []
      end

    # Extract token usage from the last assistant message
    token_usage = extract_token_usage(messages)

    %{
      id: subagent.id,
      result: result,
      duration_ms: duration_ms,
      token_usage: token_usage
    }
  end

  # Extract token usage from the last assistant message in the chain
  defp extract_token_usage(messages) when is_list(messages) do
    # Find the last assistant message which typically contains the final token usage
    messages
    |> Enum.reverse()
    |> Enum.find_value(fn message ->
      if message.role == :assistant do
        TokenUsage.get(message)
      else
        nil
      end
    end)
  end

  defp extract_token_usage(_), do: nil

  # Extract a friendly name from the subagent
  defp get_subagent_name(subagent) do
    cond do
      # Check if chain has a name in custom_context
      subagent.chain && is_map(subagent.chain.custom_context) &&
          Map.get(subagent.chain.custom_context, :config_name) ->
        subagent.chain.custom_context.config_name

      # Default to general-purpose
      true ->
        "general-purpose"
    end
  end

  # Extract instructions from the subagent's initial user message
  defp extract_instructions(subagent) do
    if subagent.chain && is_list(subagent.chain.messages) do
      case Enum.find(subagent.chain.messages, &(&1.role == :user)) do
        nil -> nil
        %{content: content} when is_binary(content) -> content
        %{content: parts} when is_list(parts) -> extract_text_from_parts(parts)
        _ -> nil
      end
    else
      nil
    end
  end

  defp extract_text_from_parts(parts) do
    parts
    |> Enum.filter(&(Map.get(&1, :type) == :text))
    |> Enum.map_join("\n", &Map.get(&1, :content, ""))
    |> case do
      "" -> nil
      text -> text
    end
  end

  # Extract tools from the subagent's chain as maps for UI display
  # Returns list of maps with :name, :description, :parameters, :async fields
  defp extract_tools(subagent) do
    if subagent.chain && is_list(subagent.chain.tools) do
      Enum.map(subagent.chain.tools, fn tool ->
        # Handle both Function structs and other tool types
        case tool do
          %{name: name, description: desc} = t ->
            %{
              name: name,
              description: desc,
              parameters: extract_parameters(t),
              async: Map.get(t, :async, false)
            }

          %{function: %{name: name, description: desc} = f} ->
            %{
              name: name,
              description: desc,
              parameters: extract_parameters(f),
              async: Map.get(f, :async, false)
            }

          %{name: name} when is_binary(name) ->
            %{name: name, description: nil, parameters: [], async: false}

          _ ->
            %{name: inspect(tool), description: nil, parameters: [], async: false}
        end
      end)
    else
      []
    end
  end

  # Extract parameters from a tool/function for UI display
  defp extract_parameters(%{parameters: params}) when is_list(params) do
    Enum.map(params, fn param ->
      %{
        name: Map.get(param, :name, "unknown"),
        description: Map.get(param, :description, ""),
        required: Map.get(param, :required, false)
      }
    end)
  end

  defp extract_parameters(_), do: []

  # Extract middleware from the subagent's chain custom_context
  # Returns list of middleware entries (MiddlewareEntry structs) for UI display
  defp extract_middleware(subagent) do
    if subagent.chain && is_map(subagent.chain.custom_context) do
      Map.get(subagent.chain.custom_context, :parent_middleware, [])
    else
      []
    end
  end

  # Get the model name from the subagent's chain
  defp get_model_name(subagent) do
    if subagent.chain && subagent.chain.llm do
      subagent.chain.llm.__struct__
      |> Module.split()
      |> List.last()
    else
      "Unknown"
    end
  end
end
