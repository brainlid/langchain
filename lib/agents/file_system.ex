defmodule LangChain.Agents.FileSystem do
  @moduledoc """
  Public API for filesystem lifecycle management.

  Provides convenience functions for starting, stopping, and accessing
  filesystem instances independent of agent lifecycles.

  ## Filesystem Scopes

  Filesystems are identified by scope keys (tuples) that determine their lifecycle:
  - `{:user, user_id}` - User-scoped filesystem
  - `{:project, project_id}` - Project-scoped filesystem
  - `{:organization, org_id}` - Organization-scoped filesystem
  - `{:agent, agent_id}` - Agent-scoped filesystem (backward compatible)

  ## Usage

      # Start a filesystem (idempotent)
      {:ok, config} = FileSystemConfig.new(%{
        scope_key: {:user, 123},
        base_directory: "Documents",
        persistence_module: Disk,
        storage_opts: [path: "/data/users/123"]
      })

      {:ok, pid} = FileSystem.ensure_filesystem({:user, 123}, [config])

      # Check if running
      true = FileSystem.filesystem_running?({:user, 123})

      # Get PID
      {:ok, pid} = FileSystem.get_filesystem_pid({:user, 123})

      # Get scope from PID
      {:user, 123} = FileSystem.get_scope(pid)

      # Stop filesystem
      :ok = FileSystem.stop_filesystem({:user, 123})
  """

  alias LangChain.Agents.FileSystem.FileSystemSupervisor
  alias LangChain.Agents.FileSystemServer

  @doc """
  Ensure a filesystem is running for the given scope (idempotent).

  If the filesystem is already running, returns the existing PID.
  If not running, starts a new filesystem with the given configs.

  ## Parameters

  - `scope_key` - Tuple identifying the filesystem scope (e.g., `{:user, 123}`)
  - `configs` - List of FileSystemConfig structs
  - `opts` - Additional options:
    - `:supervisor` - Supervisor reference (PID or registered name). Defaults to `FileSystemSupervisor`.
    - `:pubsub` - PubSub configuration as `{module(), atom()}` tuple (optional)

  ## Returns

  - `{:ok, pid}` - Filesystem PID (existing or newly started)
  - `{:error, reason}` - Error starting filesystem

  ## Examples

      {:ok, config} = FileSystemConfig.new(%{
        scope_key: {:user, 123},
        base_directory: "Documents",
        persistence_module: Disk,
        storage_opts: [path: "/data/users/123"]
      })

      # First call starts the filesystem
      {:ok, pid} = ensure_filesystem({:user, 123}, [config])

      # Subsequent calls return the same PID
      {:ok, ^pid} = ensure_filesystem({:user, 123}, [config])

      # With PubSub
      {:ok, pid} = ensure_filesystem({:user, 123}, [config], pubsub: {Phoenix.PubSub, :my_pubsub})
  """
  @spec ensure_filesystem(tuple(), list(), keyword()) :: {:ok, pid()} | {:error, term()}

  def ensure_filesystem(scope_key, configs, opts \\ [])

  def ensure_filesystem(scope_key, _configs, _opts) when not is_tuple(scope_key) do
    {:error, :invalid_arguments}
  end

  def ensure_filesystem(_scope_key, configs, _opts) when not is_list(configs) do
    {:error, :invalid_arguments}
  end

  def ensure_filesystem(scope_key, configs, opts)
      when is_tuple(scope_key) and is_list(configs) do
    case FileSystemSupervisor.get_filesystem(scope_key) do
      {:ok, pid} ->
        # Already running
        {:ok, pid}

      {:error, :not_found} ->
        # Not running, start it
        case FileSystemSupervisor.start_filesystem(scope_key, configs, opts) do
          {:ok, pid} -> {:ok, pid}
          {:error, {:already_started, pid}} -> {:ok, pid}
          {:error, reason} -> {:error, reason}
        end
    end
  end

  @doc """
  Start a new filesystem for the given scope.

  Returns an error if a filesystem is already running for this scope.
  For idempotent behavior, use `ensure_filesystem/3` instead.

  ## Parameters

  - `scope_key` - Tuple identifying the filesystem scope
  - `configs` - List of FileSystemConfig structs
  - `opts` - Additional options:
    - `:supervisor` - Supervisor reference (PID or registered name). Defaults to `FileSystemSupervisor`.
    - `:pubsub` - PubSub configuration as `{module(), atom()}` tuple (optional)

  ## Returns

  - `{:ok, pid}` - Successfully started filesystem
  - `{:error, {:already_started, pid}}` - Filesystem already running
  - `{:error, reason}` - Other error

  ## Examples

      {:ok, pid} = start_filesystem({:user, 123}, [config])
  """
  @spec start_filesystem(tuple(), list(), keyword()) :: {:ok, pid()} | {:error, term()}

  def start_filesystem(scope_key, configs, opts \\ []) do
    FileSystemSupervisor.start_filesystem(scope_key, configs, opts)
  end

  @doc """
  Stop a running filesystem.

  The filesystem will be gracefully terminated, allowing it to flush any pending writes.

  ## Parameters

  - `scope_key` - Tuple identifying the filesystem scope
  - `opts` - Additional options:
    - `:supervisor` - Supervisor reference (PID or registered name). Defaults to `FileSystemSupervisor`.

  ## Returns

  - `:ok` - Successfully stopped
  - `{:error, :not_found}` - Filesystem not running

  ## Examples

      :ok = stop_filesystem({:user, 123})
  """
  @spec stop_filesystem(tuple(), keyword()) :: :ok | {:error, :not_found}

  def stop_filesystem(scope_key, opts \\ []) do
    FileSystemSupervisor.stop_filesystem(scope_key, opts)
  end

  @doc """
  Check if a filesystem is running for the given scope.

  ## Parameters

  - `scope_key` - Tuple identifying the filesystem scope

  ## Returns

  Boolean indicating whether the filesystem is running.

  ## Examples

      true = filesystem_running?({:user, 123})
      false = filesystem_running?({:user, 999})
  """
  @spec filesystem_running?(tuple()) :: boolean()
  def filesystem_running?(scope_key) when is_tuple(scope_key) do
    case FileSystemSupervisor.get_filesystem(scope_key) do
      {:ok, _pid} -> true
      {:error, :not_found} -> false
    end
  end

  @doc """
  Get the PID of a running filesystem by scope key.

  ## Parameters

  - `scope_key` - Tuple identifying the filesystem scope

  ## Returns

  - `{:ok, pid}` - Filesystem found
  - `{:error, :not_found}` - Filesystem not running

  ## Examples

      {:ok, pid} = get_filesystem_pid({:user, 123})
  """
  @spec get_filesystem_pid(tuple()) :: {:ok, pid()} | {:error, :not_found}
  def get_filesystem_pid(scope_key) do
    FileSystemSupervisor.get_filesystem(scope_key)
  end

  @doc """
  Get the scope key for a filesystem PID.

  ## Parameters

  - `pid` - Filesystem process PID

  ## Returns

  The scope key tuple, or `{:error, reason}` if the scope cannot be determined.

  ## Examples

      {:ok, {:user, 123}} = get_scope(pid)
  """
  @spec get_scope(pid()) :: {:ok, tuple()} | {:error, term()}
  def get_scope(pid) when is_pid(pid) do
    FileSystemServer.get_scope(pid)
  end

  @doc """
  List all running filesystems.

  ## Returns

  List of `{scope_key, pid}` tuples.

  ## Examples

      filesystems = list_filesystems()
      # => [{:user, 123, #PID<0.123.0>}, {:project, 456, #PID<0.124.0>}]
  """
  @spec list_filesystems() :: [{tuple(), pid()}]
  def list_filesystems do
    FileSystemSupervisor.list_filesystems()
  end
end
