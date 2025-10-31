defmodule LangChain.Agents.FileSystemServer do
  @moduledoc """
  GenServer managing virtual filesystem with debounce-based auto-persistence.

  Owns ETS table for file storage and manages per-file debounce timers.
  This server is designed to outlive AgentServer crashes when supervised with
  `:rest_for_one` strategy, providing crash resilience.

  ## Supervision

  FileSystemServer should be the first child in an AgentSupervisor with
  `:rest_for_one` strategy. This ensures that if AgentServer crashes,
  FileSystemServer survives and preserves all filesystem state.

  ## Configuration

  - `:agent_id` - Agent identifier (required)
  - `:persistence_module` - Persistence backend module (optional, default: nil)
  - `:debounce_ms` - Milliseconds of inactivity before auto-persist (default: 5000)
  - `:storage_opts` - Configuration for persistence module (optional)

  ## Examples

      # Memory-only filesystem
      {:ok, pid} = start_link(agent_id: "agent-123")

      # Disk persistence with 5-second debounce
      {:ok, pid} = start_link(
        agent_id: "agent-123",
        persistence_module: LangChain.Agents.FileSystem.Persistence.Disk,
        debounce_ms: 5000,
        storage_opts: [path: "/data/agents"]
      )
  """

  use GenServer
  require Logger

  alias LangChain.Agents.FileSystem.{FileSystemState, FileSystemConfig}

  @type fs_ref :: pid()
  @type table_ref :: :ets.tid()

  # ============================================================================
  # Client API
  # ============================================================================

  @doc """
  Start FileSystemServer for an agent.

  ## Options

  - `:agent_id` - Agent identifier (required)
  - `:persistence_module` - Persistence backend module (optional, default: nil)
  - `:debounce_ms` - Milliseconds of inactivity before auto-persist (default: 5000)
  - `:storage_opts` - Configuration for persistence module (optional)
    - `:memories_directory` - Which virtual directory triggers persistence (default: "Memories")
    - Additional options depend on the persistence module being used

  ## Examples

      # Memory-only filesystem
      {:ok, pid} = start_link(agent_id: "agent-123")

      # Disk persistence with 5-second debounce
      {:ok, pid} = start_link(
        agent_id: "agent-123",
        persistence_module: LangChain.Agents.FileSystem.Persistence.Disk,
        debounce_ms: 5000,
        storage_opts: [path: "/data/agents"]
      )
  """
  @spec start_link(keyword()) :: GenServer.on_start()
  def start_link(opts) do
    agent_id = Keyword.fetch!(opts, :agent_id)
    GenServer.start_link(__MODULE__, opts, name: via_tuple(agent_id))
  end

  @doc """
  Get ETS table reference for direct reads.

  SubAgents should use this to get the table for fast read operations.
  """
  @spec get_table(fs_ref()) :: table_ref()
  def get_table(fs_ref) do
    GenServer.call(fs_ref, :get_table)
  end

  @doc """
  Get the FileSystemServer PID for an agent.
  """
  @spec whereis(String.t()) :: pid() | nil
  def whereis(agent_id) do
    case Registry.lookup(LangChain.Agents.Registry, {:file_system_server, agent_id}) do
      [{pid, _}] -> pid
      [] -> nil
    end
  end

  @doc """
  Write content to a file path.

  Triggers debounce timer for auto-persistence if file is in persistence directory.

  ## Options

  - `:metadata` - Custom metadata map
  - `:mime_type` - MIME type string

  ## Examples

      iex> write_file(fs_pid, "/tmp/notes.txt", "Hello")
      :ok

      iex> write_file(fs_pid, "/Memories/chat_log.txt", data)
      :ok  # Auto-persists after 5s (default) of no more writes
  """
  @spec write_file(fs_ref(), String.t(), String.t(), keyword()) :: :ok | {:error, term()}
  def write_file(fs_ref, path, content, opts \\ []) do
    GenServer.call(fs_ref, {:write_file, path, content, opts})
  end

  @doc """
  Delete file from filesystem.

  If file was persisted, it's also removed from storage immediately (no debounce).
  """
  @spec delete_file(fs_ref(), String.t()) :: :ok | {:error, term()}
  def delete_file(fs_ref, path) do
    GenServer.call(fs_ref, {:delete_file, path})
  end

  @doc """
  Register a new persistence configuration.

  Allows dynamically adding persistence backends for different base directories.

  ## Parameters

  - `fs_ref` - FileSystemServer PID
  - `config` - FileSystemConfig struct

  ## Returns

  - `:ok` on success
  - `{:error, reason}` if base_directory already registered

  ## Examples

      iex> config = FileSystemConfig.new!(%{
      ...>   base_directory: "user_files",
      ...>   persistence_module: MyApp.Persistence.Disk,
      ...>   storage_opts: [path: "/data/users"]
      ...> })
      iex> FileSystemServer.register_persistence(fs_pid, config)
      :ok
  """
  @spec register_persistence(fs_ref(), FileSystemConfig.t()) :: :ok | {:error, term()}
  def register_persistence(fs_ref, %FileSystemConfig{} = config) do
    GenServer.call(fs_ref, {:register_persistence, config})
  end

  @doc """
  Get all registered persistence configurations.

  Returns a map of base_directory => FileSystemConfig.

  ## Examples

      iex> FileSystemServer.get_persistence_configs(fs_pid)
      %{"user_files" => %FileSystemConfig{}, "S3" => %FileSystemConfig{}}
  """
  @spec get_persistence_configs(fs_ref()) :: %{String.t() => FileSystemConfig.t()}
  def get_persistence_configs(fs_ref) do
    GenServer.call(fs_ref, :get_persistence_configs)
  end

  @doc """
  Flush all pending debounce timers and persist immediately.

  Useful for graceful shutdown or checkpoints.
  """
  @spec flush_all(fs_ref()) :: :ok
  def flush_all(fs_ref) do
    GenServer.call(fs_ref, :flush_all)
  end

  @doc """
  Get filesystem statistics.

  Returns map with various statistics about the filesystem state.
  """
  @spec stats(fs_ref()) :: {:ok, map()}
  def stats(fs_ref) do
    GenServer.call(fs_ref, :stats)
  end

  # ============================================================================
  # GenServer Callbacks
  # ============================================================================

  @impl true
  def init(opts) do
    case FileSystemState.new(opts) do
      {:ok, state} ->
        agent_id = state.agent_id
        Logger.debug("FileSystemServer started for agent #{agent_id}")
        {:ok, state}

      {:error, reason} ->
        {:stop, reason}
    end
  end

  @impl true
  def handle_call(:get_table, _from, state) do
    {:reply, state.fs_table, state}
  end

  @impl true
  def handle_call({:register_persistence, config}, _from, state) do
    case FileSystemState.register_persistence(state, config) do
      {:ok, new_state} ->
        {:reply, :ok, new_state}

      {:error, reason} ->
        {:reply, {:error, reason}, state}
    end
  end

  @impl true
  def handle_call(:get_persistence_configs, _from, state) do
    {:reply, state.persistence_configs, state}
  end

  @impl true
  def handle_call({:write_file, path, content, opts}, _from, state) do
    case FileSystemState.write_file(state, path, content, opts) do
      {:ok, new_state} ->
        {:reply, :ok, new_state}

      {:error, reason, state} ->
        {:reply, {:error, reason}, state}
    end
  end

  @impl true
  def handle_call({:delete_file, path}, _from, state) do
    case FileSystemState.delete_file(state, path) do
      {:ok, new_state} ->
        {:reply, :ok, new_state}

      {:error, reason, state} ->
        {:reply, {:error, reason}, state}
    end
  end

  @impl true
  def handle_call(:flush_all, _from, state) do
    new_state = FileSystemState.flush_all(state)
    {:reply, :ok, new_state}
  end

  @impl true
  def handle_call(:stats, _from, state) do
    stats = FileSystemState.stats(state)
    {:reply, {:ok, stats}, state}
  end

  @impl true
  def handle_info({:persist_file, path}, state) do
    new_state = FileSystemState.persist_file(state, path)
    {:noreply, new_state}
  end

  @impl true
  def terminate(_reason, state) do
    # Flush all pending writes before terminating
    FileSystemState.flush_all(state)
    :ok
  end

  defp via_tuple(agent_id) do
    {:via, Registry, {LangChain.Agents.Registry, {:file_system_server, agent_id}}}
  end
end
