defmodule LangChain.Agents.FileSystem.FileSystemSupervisor do
  @moduledoc """
  Dynamic supervisor for managing FileSystemServer instances.

  Manages filesystem lifecycles independent of agent/conversation lifecycles.
  Filesystems are identified by scope keys like `{:user, user_id}` or `{:project, project_id}`.

  ## Scope Keys

  Scope keys are tuples that identify the context of a filesystem:
  - `{:user, user_id}` - User-scoped filesystem
  - `{:project, project_id}` - Project-scoped filesystem
  - `{:organization, org_id}` - Organization-scoped filesystem
  - `{:agent, agent_id}` - Agent-scoped filesystem (backward compatible)

  ## Usage

      # Start the supervisor
      {:ok, pid} = FileSystemSupervisor.start_link(name: MyApp.FileSystemSupervisor)

      # Start a filesystem for a user
      {:ok, fs_pid} = FileSystemSupervisor.start_filesystem(
        {:user, 123},
        [config1, config2]
      )

      # Get a filesystem PID
      {:ok, fs_pid} = FileSystemSupervisor.get_filesystem({:user, 123})

      # Stop a filesystem
      :ok = FileSystemSupervisor.stop_filesystem({:user, 123})

      # List all running filesystems
      filesystems = FileSystemSupervisor.list_filesystems()
  """

  use DynamicSupervisor
  require Logger

  alias LangChain.Agents.FileSystemServer

  # ============================================================================
  # Public API
  # ============================================================================

  @doc """
  Start the FileSystemSupervisor.

  ## Options

  - `:name` - Registered name for the supervisor (optional)

  ## Examples

      {:ok, pid} = start_link(name: MyApp.FileSystemSupervisor)
  """
  @spec start_link(keyword()) :: Supervisor.on_start()
  def start_link(opts \\ []) do
    name = Keyword.get(opts, :name, __MODULE__)
    DynamicSupervisor.start_link(__MODULE__, opts, name: name)
  end

  @doc """
  Wait for the FileSystemSupervisor to be ready.

  This function polls for the supervisor to be available, useful in scenarios
  where the supervisor might be starting up asynchronously (e.g., in async tests).

  ## Parameters

  - `supervisor` - Supervisor reference (PID or registered name). Defaults to `__MODULE__`.
  - `timeout_ms` - Maximum time to wait in milliseconds (default: 5000)

  ## Returns

  - `:ok` - Supervisor is ready
  - `{:error, :supervisor_not_ready}` - Timeout waiting for supervisor

  ## Examples

      :ok = wait_for_supervisor_ready()
      :ok = wait_for_supervisor_ready(__MODULE__, 10_000)
  """
  @spec wait_for_supervisor_ready() :: :ok | {:error, :supervisor_not_ready}
  @spec wait_for_supervisor_ready(atom() | pid(), non_neg_integer()) ::
          :ok | {:error, :supervisor_not_ready}

  def wait_for_supervisor_ready(supervisor \\ __MODULE__, timeout_ms \\ 5_000) do
    deadline = System.monotonic_time(:millisecond) + timeout_ms
    do_wait_for_supervisor_ready(supervisor, deadline, 10)
  end

  @doc """
  Start a new filesystem for the given scope.

  ## Parameters

  - `supervisor` - Supervisor reference (PID or registered name). Defaults to `__MODULE__`.
  - `scope_key` - Tuple identifying the filesystem scope (e.g., `{:user, 123}`)
  - `configs` - List of FileSystemConfig structs for this filesystem

  ## Returns

  - `{:ok, pid}` - Successfully started filesystem
  - `{:error, {:already_started, pid}}` - Filesystem already running for this scope
  - `{:error, reason}` - Other error

  ## Examples

      {:ok, config} = FileSystemConfig.new(%{
        scope_key: {:user, 123},
        base_directory: "Documents",
        persistence_module: Disk,
        storage_opts: [path: "/data/users/123"]
      })

      # With default supervisor
      {:ok, pid} = start_filesystem({:user, 123}, [config])

      # With explicit supervisor
      {:ok, pid} = start_filesystem(my_supervisor, {:user, 123}, [config])
  """
  @spec start_filesystem(tuple(), list()) :: {:ok, pid()} | {:error, term()}
  def start_filesystem(scope_key, configs) when is_tuple(scope_key) and is_list(configs) do
    start_filesystem(__MODULE__, scope_key, configs)
  end

  @spec start_filesystem(atom() | pid(), tuple(), list()) :: {:ok, pid()} | {:error, term()}
  def start_filesystem(supervisor, scope_key, _configs) when not is_tuple(scope_key) do
    _ = supervisor
    {:error, :invalid_scope_key}
  end

  def start_filesystem(supervisor, _scope_key, configs) when not is_list(configs) do
    _ = supervisor
    {:error, :invalid_configs}
  end

  def start_filesystem(supervisor, scope_key, configs)
      when is_tuple(scope_key) and is_list(configs) do
    # Wait for supervisor to be ready (handles async startup scenarios)
    case wait_for_supervisor_ready(supervisor, 5_000) do
      :ok ->
        # Check if filesystem already running for this scope
        case get_filesystem(scope_key) do
          {:ok, pid} ->
            {:error, {:already_started, pid}}

          {:error, :not_found} ->
            # Start new filesystem
            child_spec = %{
              id: {:filesystem_server, scope_key},
              start: {FileSystemServer, :start_link, [[scope_key: scope_key, configs: configs]]},
              restart: :transient
            }

            case DynamicSupervisor.start_child(supervisor, child_spec) do
              {:ok, pid} ->
                Logger.debug(
                  "Started filesystem for scope #{inspect(scope_key)}, pid: #{inspect(pid)}"
                )

                {:ok, pid}

              {:error, reason} = error ->
                Logger.error(
                  "Failed to start filesystem for scope #{inspect(scope_key)}: #{inspect(reason)}"
                )

                error
            end
        end

      {:error, :supervisor_not_ready} = error ->
        Logger.warning(
          "FileSystemSupervisor not available when attempting to start filesystem for scope #{inspect(scope_key)}"
        )

        error
    end
  end

  @doc """
  Stop an existing filesystem.

  The filesystem will be gracefully terminated, allowing it to flush any pending writes.

  ## Parameters

  - `supervisor` - Supervisor reference (PID or registered name). Defaults to `__MODULE__`.
  - `scope_key` - Tuple identifying the filesystem scope

  ## Returns

  - `:ok` - Successfully stopped
  - `{:error, :not_found}` - Filesystem not running for this scope

  ## Examples

      # With default supervisor
      :ok = stop_filesystem({:user, 123})

      # With explicit supervisor
      :ok = stop_filesystem(my_supervisor, {:user, 123})
  """
  @spec stop_filesystem(tuple()) :: :ok | {:error, :not_found}
  @spec stop_filesystem(atom() | pid(), tuple()) :: :ok | {:error, :not_found}

  def stop_filesystem(scope_key) when is_tuple(scope_key) do
    stop_filesystem(__MODULE__, scope_key)
  end

  def stop_filesystem(supervisor, scope_key) when is_tuple(scope_key) do
    case get_filesystem(scope_key) do
      {:ok, pid} ->
        case DynamicSupervisor.terminate_child(supervisor, pid) do
          :ok ->
            Logger.debug("Stopped filesystem for scope #{inspect(scope_key)}")
            :ok

          {:error, :not_found} ->
            # Process already stopped
            {:error, :not_found}
        end

      {:error, :not_found} ->
        {:error, :not_found}
    end
  end

  @doc """
  Get the PID of a running filesystem by scope key.

  ## Parameters

  - `scope_key` - Tuple identifying the filesystem scope

  ## Returns

  - `{:ok, pid}` - Filesystem found
  - `{:error, :not_found}` - Filesystem not running for this scope

  ## Examples

      {:ok, pid} = get_filesystem({:user, 123})
  """
  @spec get_filesystem(tuple()) :: {:ok, pid()} | {:error, :not_found}
  def get_filesystem(scope_key) when is_tuple(scope_key) do
    case Registry.lookup(LangChain.Agents.Registry, {:filesystem_server, scope_key}) do
      [{pid, _}] -> {:ok, pid}
      [] -> {:error, :not_found}
    end
  end

  @doc """
  List all running filesystem scopes and their PIDs.

  ## Returns

  List of `{scope_key, pid}` tuples.

  ## Examples

      filesystems = list_filesystems()
      # => [{:user, 123, #PID<0.123.0>}, {:project, 456, #PID<0.124.0>}]
  """
  @spec list_filesystems() :: [{tuple(), pid()}]
  def list_filesystems do
    Registry.select(LangChain.Agents.Registry, [
      {
        {{:filesystem_server, :"$1"}, :"$2", :_},
        [],
        [{{:"$1", :"$2"}}]
      }
    ])
  end

  # ============================================================================
  # DynamicSupervisor Callbacks
  # ============================================================================

  @impl true
  def init(_opts) do
    DynamicSupervisor.init(strategy: :one_for_one)
  end

  # ============================================================================
  # Private Helpers
  # ============================================================================

  # Wait for the supervisor to be registered and ready
  # Retries with exponential backoff up to the timeout
  # Uses fast-fail strategy: if supervisor not found on first check, only retry briefly
  defp do_wait_for_supervisor_ready(supervisor, deadline, retry_delay_ms) do
    # Try to check if the supervisor is alive
    ready =
      try do
        case supervisor do
          pid when is_pid(pid) ->
            Process.alive?(pid)

          name when is_atom(name) ->
            case Process.whereis(name) do
              nil -> false
              pid -> Process.alive?(pid)
            end
        end
      catch
        _, _ -> false
      end

    if ready do
      :ok
    else
      now = System.monotonic_time(:millisecond)

      # Fast-fail strategy: if this is the first check (retry_delay_ms == 10) and supervisor
      # is not found, only wait briefly (up to 100ms total) to handle startup race conditions.
      # This prevents long waits in scenarios where the supervisor will never be available
      # (e.g., async tests).
      time_remaining = deadline - now

      should_give_up =
        if retry_delay_ms == 10 do
          # First check failed - reduce deadline to 100ms for fast-fail
          time_remaining > 100
        else
          # Already retrying - use original deadline
          false
        end

      adjusted_deadline = if should_give_up, do: now + 100, else: deadline

      if now >= adjusted_deadline do
        {:error, :supervisor_not_ready}
      else
        # Sleep briefly and retry with exponential backoff (max 100ms)
        Process.sleep(retry_delay_ms)
        next_delay = min(retry_delay_ms * 2, 100)
        do_wait_for_supervisor_ready(supervisor, adjusted_deadline, next_delay)
      end
    end
  end
end
