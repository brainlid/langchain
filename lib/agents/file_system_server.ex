defmodule LangChain.Agents.FileSystemServer do
  @moduledoc """
  GenServer managing virtual filesystem with debounce-based auto-persistence.

  Manages file storage in GenServer state and handles per-file debounce timers.

  This server is designed to outlive AgentServer crashes when supervised with
  `:rest_for_one` strategy, providing crash resilience.

  ## Supervision

  FileSystemServer should be the first child in an AgentSupervisor with
  `:rest_for_one` strategy. This ensures that if AgentServer crashes,
  FileSystemServer survives and preserves all filesystem state.

  ## Graceful Shutdown

  FileSystemServer traps exits to ensure graceful shutdown. When the process
  terminates (via supervisor shutdown or any other reason), it automatically
  flushes all pending debounced writes to persistence before terminating.

  ## Configuration

  - `:scope_key` - Scope identifier (required) - Can be any unique term
    - Tuple format: `{:user, 123}`, `{:agent, "uuid"}`, `{:project, id}`
    - UUID string: `"550e8400-e29b-41d4-a716-446655440000"`
    - Database ID: `"12345"`
  - `:configs` - List of FileSystemConfig structs (optional, default: [])
  - `:pubsub` - PubSub configuration as `{module(), atom()}` tuple or `nil` (optional, default: nil)
    Example: `{Phoenix.PubSub, :my_app_pubsub}`
    When configured, broadcasts `{:files_updated, file_list}` after write/delete operations.

  ## Examples

      # Memory-only filesystem with tuple scope
      {:ok, pid} = start_link(scope_key: {:user, 123})

      # Memory-only filesystem with UUID
      {:ok, pid} = start_link(scope_key: "550e8400-e29b-41d4-a716-446655440000")

      # Memory-only filesystem with database ID
      {:ok, pid} = start_link(scope_key: 789)

      # With disk persistence (tuple scope)
      {:ok, config} = FileSystemConfig.new(%{
        base_directory: "Memories",
        persistence_module: LangChain.Agents.FileSystem.Persistence.Disk,
        debounce_ms: 5000,
        storage_opts: [path: "/data/users/123"]
      })
      {:ok, pid} = start_link(
        scope_key: {:user, 123},
        configs: [config]
      )
  """

  use GenServer
  require Logger

  alias LangChain.Agents.FileSystem.FileSystemState
  alias LangChain.Agents.FileSystem.FileSystemConfig
  alias LangChain.Agents.FileSystem.FileEntry

  # ======================================================================
  # Client API
  # ======================================================================

  @doc """
  Start FileSystemServer for a scope.

  ## Options

  - `:scope_key` - Scope identifier (required) - Can be any term that uniquely identifies the scope
    - Tuple: `{:user, 123}`, `{:agent, uuid}`, `{:project, id}`
    - UUID: `"550e8400-e29b-41d4-a716-446655440000"`
    - Database ID: `12345` or `"12345"`
  - `:configs` - List of FileSystemConfig structs (optional, default: [])

  ## Examples

      # Memory-only filesystem with tuple scope
      {:ok, pid} = start_link(scope_key: {:user, 123})

      # Memory-only filesystem with UUID
      {:ok, pid} = start_link(scope_key: "550e8400-e29b-41d4-a716-446655440000")

      # Memory-only filesystem with database ID
      {:ok, pid} = start_link(scope_key: 789)

      # With disk persistence
      {:ok, config} = FileSystemConfig.new(%{
        base_directory: "Memories",
        persistence_module: LangChain.Agents.FileSystem.Persistence.Disk,
        debounce_ms: 5000,
        storage_opts: [path: "/data/users/123"]
      })
      {:ok, pid} = start_link(
        scope_key: {:user, 123},
        configs: [config]
      )
  """
  @spec start_link(keyword()) :: GenServer.on_start()
  def start_link(opts) do
    scope_key = Keyword.fetch!(opts, :scope_key)
    GenServer.start_link(__MODULE__, opts, name: get_name(scope_key))
  end

  @doc """
  Child spec for starting under a supervisor.
  """
  def child_spec(opts) do
    scope_key = Keyword.fetch!(opts, :scope_key)

    %{
      id: {:filesystem_server, scope_key},
      start: {__MODULE__, :start_link, [opts]},
      restart: :transient
    }
  end

  @doc """
  Get the FileSystemServer PID by scope key.

  The scope_key can be any term that uniquely identifies the filesystem scope.
  Common patterns include tuples like `{:user, 123}`, UUIDs like
  `"550e8400-e29b-41d4-a716-446655440000"`, or database IDs like `12345`.
  """
  @spec whereis(term()) :: pid() | nil
  def whereis(scope_key) do
    case Registry.lookup(LangChain.Agents.Registry, {:filesystem_server, scope_key}) do
      [{pid, _}] -> pid
      [] -> nil
    end
  end

  @doc """
  Get the scope key for a FileSystemServer PID.

  Returns the scope_key that was used to start the server.
  """
  @spec get_scope(pid()) :: {:ok, term()} | {:error, term()}
  def get_scope(pid) when is_pid(pid) do
    GenServer.call(pid, :get_scope)
  end

  @doc """
  Get the via tuple name for a scope key.

  The scope_key can be any term that uniquely identifies the scope.
  Common patterns include tuples like `{:user, 123}` or strings like `"agent-abc"`.
  """
  def get_name(scope_key) do
    {:via, Registry, {LangChain.Agents.Registry, {:filesystem_server, scope_key}}}
  end

  @doc """
  Write content to a file path.

  Triggers debounce timer for auto-persistence if file is in persistence directory.

  ## Options

  - `:metadata` - Custom metadata map
  - `:mime_type` - MIME type string

  ## Examples

      # With tuple scope
      iex> write_file({:user, 123}, "/tmp/notes.txt", "Hello")
      :ok

      # With UUID scope
      iex> write_file("550e8400-e29b-41d4-a716-446655440000", "/Memories/chat_log.txt", data)
      :ok  # Auto-persists after 5s (default) of no more writes
  """
  @spec write_file(term(), String.t(), String.t(), keyword()) ::
          :ok | {:error, term()}
  def write_file(scope_key, path, content, opts \\ []) do
    GenServer.call(get_name(scope_key), {:write_file, path, content, opts})
  end

  @doc """
  Read file content from filesystem with lazy loading.

  ## Returns

  - `{:ok, content}` - File content as string
  - `{:error, :enoent}` - File doesn't exist
  - `{:error, reason}` - Other errors (permission, load failure, etc.)

  ## Examples

      # With tuple scope
      iex> read_file({:user, 123}, "/Memories/notes.txt")
      {:ok, "My notes..."}

      # With database ID scope
      iex> read_file(789, "/nonexistent.txt")
      {:error, :enoent}
  """
  @spec read_file(term(), String.t()) :: {:ok, String.t()} | {:error, term()}
  def read_file(scope_key, path) do
    GenServer.call(get_name(scope_key), {:read_file, path})
  end

  @doc """
  Delete file from filesystem.

  If file was persisted, it's also removed from storage immediately (no debounce).
  """
  @spec delete_file(term(), String.t()) :: :ok | {:error, term()}
  def delete_file(scope_key, path) do
    GenServer.call(get_name(scope_key), {:delete_file, path})
  end

  @doc """
  Register a new persistence configuration.

  Allows dynamically adding persistence backends for different base directories.

  ## Parameters

  - `scope_key` - Scope identifier tuple
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
      iex> FileSystemServer.register_persistence({:user, 123}, config)
      :ok
  """
  @spec register_persistence(term(), FileSystemConfig.t()) :: :ok | {:error, term()}
  def register_persistence(scope_key, %FileSystemConfig{} = config) do
    GenServer.call(get_name(scope_key), {:register_persistence, config})
  end

  @doc """
  Register file entries in the filesystem.

  Useful for pre-populating the filesystem with file metadata.
  Accepts either a single FileEntry or a list of FileEntry structs.

  ## Parameters

  - `scope_key` - Scope identifier tuple
  - `file_entry_or_entries` - FileEntry struct or list of FileEntry structs

  ## Returns

  - `:ok` on success

  ## Examples

      iex> {:ok, entry} = FileEntry.new_memory_file("/scratch/temp.txt", "data")
      iex> FileSystemServer.register_files({:user, 123}, entry)
      :ok

      iex> {:ok, entry1} = FileEntry.new_memory_file("/scratch/file1.txt", "data1")
      iex> {:ok, entry2} = FileEntry.new_memory_file("/scratch/file2.txt", "data2")
      iex> FileSystemServer.register_files({:user, 123}, [entry1, entry2])
      :ok
  """
  @spec register_files(term(), FileEntry.t() | [FileEntry.t()]) :: :ok
  def register_files(scope_key, %FileEntry{} = file_entry) do
    register_files(scope_key, [file_entry])
  end

  def register_files(scope_key, file_entries) when is_list(file_entries) do
    GenServer.call(get_name(scope_key), {:register_files, file_entries})
  end

  @doc """
  Get all registered persistence configurations.

  Returns a map of base_directory => FileSystemConfig.

  ## Examples

      iex> FileSystemServer.get_persistence_configs({:user, 123})
      %{"user_files" => %FileSystemConfig{}, "S3" => %FileSystemConfig{}}
  """
  @spec get_persistence_configs(term()) :: %{String.t() => FileSystemConfig.t()}
  def get_persistence_configs(scope_key) do
    GenServer.call(get_name(scope_key), :get_persistence_configs)
  end

  @doc """
  Flush all pending debounce timers and persist immediately.

  Useful for graceful shutdown or checkpoints.
  """
  @spec flush_all(term()) :: :ok
  def flush_all(scope_key) do
    GenServer.call(get_name(scope_key), :flush_all)
  end

  @doc """
  Reset the filesystem to pristine persisted state.

  This operation:
  - Removes all memory-only files (not persisted)
  - Unloads all persisted files (discards in-memory modifications)
  - Cancels all pending debounce timers (discards unsaved changes)

  **Result**: Next read will reload persisted files from storage in their original state.

  This is useful when resetting to start fresh without carrying over
  transient in-memory file modifications.

  ## Examples

      iex> FileSystemServer.reset({:user, 123})
      :ok
  """
  @spec reset(term()) :: :ok
  def reset(scope_key) do
    GenServer.call(get_name(scope_key), :reset)
  end

  @doc """
  List all file paths in the filesystem.

  Returns paths for both memory and persisted files, regardless of load status.

  ## Examples

      iex> list_files({:user, 123})
      ["/file1.txt", "/Memories/file2.txt"]
  """
  @spec list_files(nil | term()) :: [String.t()]
  def list_files(scope_key)
  def list_files(nil), do: []

  def list_files(scope_key) do
    GenServer.call(get_name(scope_key), :list_files)
  end

  @doc """
  Check if a file exists in the filesystem.

  ## Examples

      iex> file_exists?({:user, 123}, "/notes.txt")
      true

      iex> file_exists?({:user, 123}, "/nonexistent.txt")
      false
  """
  @spec file_exists?(term(), String.t()) :: boolean()
  def file_exists?(scope_key, path) do
    GenServer.call(get_name(scope_key), {:file_exists?, path})
  end

  @doc """
  Get filesystem statistics.

  Returns map with various statistics about the filesystem state.
  """
  @spec stats(term()) :: {:ok, map()}
  def stats(scope_key) do
    GenServer.call(get_name(scope_key), :stats)
  end

  @doc """
  Subscribe to file change events for a filesystem scope.

  Events broadcast (wrapped in `{:file_system, event}` tuple):
  - `{:file_system, {:file_updated, path}}` - File was created or updated at path
  - `{:file_system, {:file_deleted, path}}` - File was deleted at path

  ## Examples

      # Subscribe to user's filesystem
      :ok = FileSystemServer.subscribe({:user, 123})

      # Receive events
      receive do
        {:file_system, {:file_updated, path}} -> IO.puts("File updated: \#{path}")
        {:file_system, {:file_deleted, path}} -> IO.puts("File deleted: \#{path}")
      end
  """
  @spec subscribe(term()) :: :ok | {:error, :no_pubsub | :process_not_found}
  def subscribe(scope_key) do
    try do
      case GenServer.call(get_name(scope_key), :get_pubsub_info) do
        nil ->
          {:error, :no_pubsub}

        {pubsub, pubsub_name, topic} ->
          pubsub.subscribe(pubsub_name, topic)
      end
    catch
      :exit, _ ->
        {:error, :process_not_found}
    end
  end

  @doc """
  Unsubscribe from file change events for a filesystem scope.
  """
  @spec unsubscribe(term()) :: :ok | {:error, :no_pubsub | :process_not_found}
  def unsubscribe(scope_key) do
    try do
      case GenServer.call(get_name(scope_key), :get_pubsub_info) do
        nil ->
          {:error, :no_pubsub}

        {pubsub, pubsub_name, topic} ->
          pubsub.unsubscribe(pubsub_name, topic)
      end
    catch
      :exit, _ ->
        {:error, :process_not_found}
    end
  end

  # ============================================================================
  # GenServer Callbacks
  # ============================================================================

  @impl true
  def init(opts) do
    # Trap exits to ensure terminate/2 is called for graceful shutdown
    Process.flag(:trap_exit, true)

    case FileSystemState.new(opts) do
      {:ok, state} ->
        scope_key = state.scope_key
        Logger.debug("FileSystemServer started for scope #{inspect(scope_key)}")
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
  def handle_call(:get_scope, _from, state) do
    {:reply, {:ok, state.scope_key}, state}
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
        broadcast_file_change(new_state, {:file_updated, path})
        {:reply, :ok, new_state}

      {:error, reason, state} ->
        {:reply, {:error, reason}, state}
    end
  end

  @impl true
  def handle_call({:read_file, path}, _from, state) do
    case FileSystemState.read_file(state, path) do
      {:ok, %FileEntry{loaded: true, content: content}} ->
        # File is loaded, return content
        {:reply, {:ok, content}, state}

      {:ok, %FileEntry{loaded: false}} ->
        # File exists but not loaded - load it first
        case FileSystemState.load_file(state, path) do
          {:ok, new_state} ->
            # Now get the content
            case FileSystemState.read_file(new_state, path) do
              {:ok, %FileEntry{content: content}} ->
                {:reply, {:ok, content}, new_state}

              {:error, :enoent} ->
                {:reply, {:error, :enoent}, new_state}
            end

          {:error, reason, state} ->
            {:reply, {:error, reason}, state}
        end

      {:error, :enoent} ->
        {:reply, {:error, :enoent}, state}
    end
  end

  @impl true
  def handle_call({:load_file, path}, _from, state) do
    case FileSystemState.load_file(state, path) do
      {:ok, new_state} ->
        # Return :ok WITHOUT the file content
        {:reply, :ok, new_state}

      {:error, reason, state} ->
        {:reply, {:error, reason}, state}
    end
  end

  @impl true
  def handle_call(:list_files, _from, state) do
    files = FileSystemState.list_files(state)
    {:reply, files, state}
  end

  @impl true
  def handle_call({:file_exists?, path}, _from, state) do
    exists = FileSystemState.file_exists?(state, path)
    {:reply, exists, state}
  end

  @impl true
  def handle_call(:stats, _from, state) do
    stats = FileSystemState.stats(state)
    {:reply, {:ok, stats}, state}
  end

  @impl true
  def handle_call(:get_pubsub_info, _from, state) do
    case state.pubsub do
      nil ->
        {:reply, nil, state}

      {pubsub, pubsub_name} ->
        {:reply, {pubsub, pubsub_name, state.topic}, state}
    end
  end

  @impl true
  def handle_call({:delete_file, path}, _from, state) do
    case FileSystemState.delete_file(state, path) do
      {:ok, new_state} ->
        broadcast_file_change(new_state, {:file_deleted, path})
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
  def handle_call(:reset, _from, state) do
    new_state = FileSystemState.reset(state)
    {:reply, :ok, new_state}
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

  # ============================================================================
  # Private Helpers
  # ============================================================================

  # Broadcast file changes to subscribers
  # change_info is a tuple like {:file_updated, path} or {:file_deleted, path}
  # Events are wrapped as {:file_system, change_info} for easier pattern matching
  defp broadcast_file_change(state, change_info) do
    case state.pubsub do
      {pubsub, pubsub_name} ->
        pubsub.broadcast_from(pubsub_name, self(), state.topic, {:file_system, change_info})

      nil ->
        :ok
    end
  end
end
