defmodule LangChain.Agents.FileSystemServer do
  @moduledoc """
  GenServer managing virtual filesystem with debounce-based auto-persistence.

  Owns a named ETS table for file storage and manages per-file debounce timers.
  The ETS table is named based on agent_id, allowing direct access without needing
  to pass around table references or PIDs.

  This server is designed to outlive AgentServer crashes when supervised with
  `:rest_for_one` strategy, providing crash resilience.

  ## Supervision

  FileSystemServer should be the first child in an AgentSupervisor with
  `:rest_for_one` strategy. This ensures that if AgentServer crashes,
  FileSystemServer survives and preserves all filesystem state.

  ## Configuration

  - `:agent_id` - Agent identifier (required)
  - `:persistence_configs` - List of FileSystemConfig structs (optional, default: [])

  ## Examples

      # Memory-only filesystem
      {:ok, pid} = start_link(agent_id: "agent-123")

      # With disk persistence
      {:ok, config} = FileSystemConfig.new(%{
        base_directory: "Memories",
        persistence_module: LangChain.Agents.FileSystem.Persistence.Disk,
        debounce_ms: 5000,
        storage_opts: [path: "/data/agents"]
      })
      {:ok, pid} = start_link(
        agent_id: "agent-123",
        persistence_configs: [config]
      )
  """

  use GenServer
  require Logger

  alias LangChain.Agents.FileSystem.FileSystemState
  alias LangChain.Agents.FileSystem.FileSystemConfig
  alias LangChain.Agents.FileSystem.FileEntry

  # ============================================================================
  # Client API
  # ============================================================================

  @doc """
  Start FileSystemServer for an agent.

  ## Options

  - `:agent_id` - Agent identifier (required)
  - `:persistence_configs` - List of FileSystemConfig structs (optional, default: [])

  ## Examples

      # Memory-only filesystem
      {:ok, pid} = start_link(agent_id: "agent-123")

      # With disk persistence
      {:ok, config} = FileSystemConfig.new(%{
        base_directory: "Memories",
        persistence_module: LangChain.Agents.FileSystem.Persistence.Disk,
        debounce_ms: 5000,
        storage_opts: [path: "/data/agents"]
      })
      {:ok, pid} = start_link(
        agent_id: "agent-123",
        persistence_configs: [config]
      )
  """
  @spec start_link(keyword()) :: GenServer.on_start()
  def start_link(opts) do
    agent_id = Keyword.fetch!(opts, :agent_id)
    GenServer.start_link(__MODULE__, opts, name: via_tuple(agent_id))
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

      iex> write_file("agent-123", "/tmp/notes.txt", "Hello")
      :ok

      iex> write_file("agent-123", "/Memories/chat_log.txt", data)
      :ok  # Auto-persists after 5s (default) of no more writes
  """
  @spec write_file(String.t(), String.t(), String.t(), keyword()) :: :ok | {:error, term()}
  def write_file(agent_id, path, content, opts \\ []) do
    GenServer.call(via_tuple(agent_id), {:write_file, path, content, opts})
  end

  @doc """
  Read file content from filesystem with lazy loading.

  This function optimizes for concurrent reads by:
  1. Reading directly from named ETS table if file is already loaded (no GenServer bottleneck)
  2. Making a GenServer call only if file needs to be loaded from persistence
  3. After loading, reading from ETS again (avoiding large data transfer across processes)

  ## Returns

  - `{:ok, content}` - File content as string
  - `{:error, :enoent}` - File doesn't exist
  - `{:error, reason}` - Other errors (permission, load failure, etc.)

  ## Examples

      iex> read_file("agent-123", "/Memories/notes.txt")
      {:ok, "My notes..."}

      iex> read_file("agent-123", "/nonexistent.txt")
      {:error, :enoent}
  """
  @spec read_file(String.t(), String.t()) :: {:ok, String.t()} | {:error, term()}
  def read_file(agent_id, path) do
    # Try to read directly from named ETS table using FileSystemState
    case FileSystemState.read_file(agent_id, path) do
      {:ok, %FileEntry{loaded: true, content: content}} ->
        # File is loaded, return content immediately
        {:ok, content}

      {:ok, %FileEntry{loaded: false}} ->
        # File exists but not loaded - ask GenServer to load it
        case GenServer.call(via_tuple(agent_id), {:load_file, path}) do
          :ok ->
            # File loaded, read from ETS again
            case FileSystemState.read_file(agent_id, path) do
              {:ok, %FileEntry{content: content}} ->
                {:ok, content}

              {:error, :enoent} ->
                {:error, :enoent}
            end

          {:error, reason} ->
            {:error, reason}
        end

      {:error, :enoent} ->
        # File doesn't exist
        {:error, :enoent}
    end
  end

  @doc """
  Delete file from filesystem.

  If file was persisted, it's also removed from storage immediately (no debounce).
  """
  @spec delete_file(String.t(), String.t()) :: :ok | {:error, term()}
  def delete_file(agent_id, path) do
    GenServer.call(via_tuple(agent_id), {:delete_file, path})
  end

  @doc """
  Register a new persistence configuration.

  Allows dynamically adding persistence backends for different base directories.

  ## Parameters

  - `agent_id` - Agent identifier
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
      iex> FileSystemServer.register_persistence("agent-123", config)
      :ok
  """
  @spec register_persistence(String.t(), FileSystemConfig.t()) :: :ok | {:error, term()}
  def register_persistence(agent_id, %FileSystemConfig{} = config) do
    GenServer.call(via_tuple(agent_id), {:register_persistence, config})
  end

  @doc """
  Register file entries in the filesystem.

  Useful for pre-populating the filesystem with file metadata.
  Accepts either a single FileEntry or a list of FileEntry structs.

  ## Parameters

  - `agent_id` - Agent identifier
  - `file_entry_or_entries` - FileEntry struct or list of FileEntry structs

  ## Returns

  - `:ok` on success

  ## Examples

      iex> {:ok, entry} = FileEntry.new_memory_file("/scratch/temp.txt", "data")
      iex> FileSystemServer.register_files("agent-123", entry)
      :ok

      iex> {:ok, entry1} = FileEntry.new_memory_file("/scratch/file1.txt", "data1")
      iex> {:ok, entry2} = FileEntry.new_memory_file("/scratch/file2.txt", "data2")
      iex> FileSystemServer.register_files("agent-123", [entry1, entry2])
      :ok
  """
  @spec register_files(String.t(), FileEntry.t() | [FileEntry.t()]) :: :ok
  def register_files(agent_id, %FileEntry{} = file_entry) do
    register_files(agent_id, [file_entry])
  end

  def register_files(agent_id, file_entries) when is_list(file_entries) do
    GenServer.call(via_tuple(agent_id), {:register_files, file_entries})
  end

  @doc """
  Get all registered persistence configurations.

  Returns a map of base_directory => FileSystemConfig.

  ## Examples

      iex> FileSystemServer.get_persistence_configs("agent-123")
      %{"user_files" => %FileSystemConfig{}, "S3" => %FileSystemConfig{}}
  """
  @spec get_persistence_configs(String.t()) :: %{String.t() => FileSystemConfig.t()}
  def get_persistence_configs(agent_id) do
    GenServer.call(via_tuple(agent_id), :get_persistence_configs)
  end

  @doc """
  Flush all pending debounce timers and persist immediately.

  Useful for graceful shutdown or checkpoints.
  """
  @spec flush_all(String.t()) :: :ok
  def flush_all(agent_id) do
    GenServer.call(via_tuple(agent_id), :flush_all)
  end

  @doc """
  Get filesystem statistics.

  Returns map with various statistics about the filesystem state.
  """
  @spec stats(String.t()) :: {:ok, map()}
  def stats(agent_id) do
    GenServer.call(via_tuple(agent_id), :stats)
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
  def handle_call({:register_files, file_entries}, _from, state) do
    {:ok, new_state} = FileSystemState.register_files(state, file_entries)
    {:reply, :ok, new_state}
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
  def handle_call({:load_file, path}, _from, state) do
    case FileSystemState.load_file(state, path) do
      {:ok, new_state} ->
        # Return :ok WITHOUT the file content (client will read from ETS)
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
