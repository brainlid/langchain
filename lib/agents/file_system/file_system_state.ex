defmodule LangChain.Agents.FileSystem.FileSystemState do
  @moduledoc """
  State management for the FileSystem.

  This module handles all state transitions and ETS operations for the virtual
  filesystem. It's designed to be testable independently of the GenServer.
  """

  require Logger
  alias __MODULE__
  alias LangChain.Agents.FileSystem.{FileEntry, FileSystemConfig}

  defstruct [
    :agent_id,
    :fs_table,
    :persistence_configs,
    :debounce_timers
  ]

  @type t :: %FileSystemState{
          agent_id: String.t(),
          fs_table: :ets.tid(),
          persistence_configs: %{String.t() => FileSystemConfig.t()},
          debounce_timers: %{String.t() => reference()}
        }

  @doc """
  Creates a new FileSystemState and initializes the ETS table.

  ## Options

  - `:agent_id` - Agent identifier (required)
  - `:persistence_configs` - List of FileSystemConfig structs (optional, default: [])
  """
  @spec new(keyword()) :: {:ok, t()} | {:error, term()}
  def new(opts) do
    with {:ok, agent_id} <- fetch_agent_id(opts) do
      # Create ETS table
      table_name = :"agent_filesystem_#{agent_id}"

      fs_table =
        :ets.new(table_name, [
          :set,
          :public,
          {:read_concurrency, true},
          {:write_concurrency, false},
          {:heir, :none}
        ])

      # Build persistence configs map
      persistence_configs = build_persistence_configs(opts)

      state = %FileSystemState{
        agent_id: agent_id,
        fs_table: fs_table,
        persistence_configs: persistence_configs,
        debounce_timers: %{}
      }

      Logger.debug("FileSystemState initialized for agent #{agent_id}")
      {:ok, state}
    end
  end

  @doc """
  Registers a new persistence configuration.

  ## Parameters

  - `state` - Current FileSystemState
  - `config` - FileSystemConfig to register

  ## Returns

  - `{:ok, new_state}` on success
  - `{:error, reason}` if base_directory already registered

  ## Examples

      iex> config = FileSystemConfig.new!(%{
      ...>   base_directory: "user_files",
      ...>   persistence_module: MyApp.Persistence.Disk
      ...> })
      iex> {:ok, new_state} = FileSystemState.register_persistence(state, config)
  """
  @spec register_persistence(t(), FileSystemConfig.t()) :: {:ok, t()} | {:error, term()}
  def register_persistence(%FileSystemState{} = state, %FileSystemConfig{} = config) do
    base_dir = config.base_directory

    if Map.has_key?(state.persistence_configs, base_dir) do
      {:error, "Base directory '#{base_dir}' already has a registered persistence config"}
    else
      new_configs = Map.put(state.persistence_configs, base_dir, config)
      {:ok, %{state | persistence_configs: new_configs}}
    end
  end

  @doc """
  Writes a file to the filesystem.

  Returns `{:ok, new_state}` or `{:error, reason, state}`.
  """
  @spec write_file(t(), String.t(), String.t(), keyword()) ::
          {:ok, t()} | {:error, term(), t()}
  def write_file(%FileSystemState{} = state, path, content, opts \\ []) do
    # Build metadata
    mime_type = Keyword.get(opts, :mime_type, "text/plain")
    custom = Keyword.get(opts, :custom, %{})
    metadata_opts = [mime_type: mime_type, custom: custom]

    # Find matching persistence config
    config = find_config_for_path(state, path)

    # Check if readonly
    if config && config.readonly do
      {:error, "Cannot write to read-only directory: #{config.base_directory}", state}
    else
      # Create file entry
      entry_result =
        if config do
          FileEntry.new_persisted_file(path, content, metadata_opts)
        else
          FileEntry.new_memory_file(path, content, metadata_opts)
        end

      case entry_result do
        {:ok, entry} ->
          # Write to ETS
          :ets.insert(state.fs_table, {path, entry})

          # Schedule debounce timer if persisted
          new_state =
            if config do
              schedule_persist(state, path, config)
            else
              state
            end

          {:ok, new_state}

        {:error, reason} ->
          {:error, reason, state}
      end
    end
  end

  @doc """
  Deletes a file from the filesystem.

  Returns `{:ok, new_state}` or `{:error, reason, state}`.
  """
  @spec delete_file(t(), String.t()) :: {:ok, t()} | {:error, term(), t()}
  def delete_file(%FileSystemState{} = state, path) do
    # Find matching config
    config = find_config_for_path(state, path)

    # Check if readonly
    if config && config.readonly do
      {:error, "Cannot delete from read-only directory: #{config.base_directory}", state}
    else
      case :ets.lookup(state.fs_table, path) do
        [{^path, %FileEntry{persistence: :persisted} = entry}] ->
          # Cancel any pending timer
          new_state = cancel_timer(state, path)

          # Delete from storage immediately if we have a config
          if config do
            opts = FileSystemConfig.build_storage_opts(config, state.agent_id)

            case config.persistence_module.delete_from_storage(entry, opts) do
              :ok ->
                :ets.delete(state.fs_table, path)
                {:ok, new_state}

              {:error, reason} ->
                Logger.error("Failed to delete #{path} from storage: #{inspect(reason)}")
                {:error, reason, state}
            end
          else
            :ets.delete(state.fs_table, path)
            {:ok, new_state}
          end

        [{^path, _entry}] ->
          # Memory-only file, just delete from ETS
          :ets.delete(state.fs_table, path)
          {:ok, state}

        [] ->
          # File doesn't exist, that's OK
          {:ok, state}
      end
    end
  end

  @doc """
  Persists a file to storage (called when debounce timer fires).

  Returns updated state.
  """
  @spec persist_file(t(), String.t()) :: t()
  def persist_file(%FileSystemState{} = state, path) do
    # Remove timer from map (it's fired)
    state = %{state | debounce_timers: Map.delete(state.debounce_timers, path)}

    # Find matching config
    config = find_config_for_path(state, path)

    # Persist the file
    case :ets.lookup(state.fs_table, path) do
      [{^path, %FileEntry{dirty: true, persistence: :persisted} = entry}] ->
        if config do
          opts = FileSystemConfig.build_storage_opts(config, state.agent_id)

          case config.persistence_module.write_to_storage(entry, opts) do
            :ok ->
              # Mark file as clean
              updated_entry = FileEntry.mark_clean(entry)
              :ets.insert(state.fs_table, {path, updated_entry})
              Logger.debug("Persisted file after debounce: #{path}")

            {:error, reason} ->
              Logger.error("Failed to persist #{path}: #{inspect(reason)}")
          end
        end

        state

      _ ->
        # File no longer dirty or doesn't exist - no-op
        state
    end
  end

  @doc """
  Flushes all pending debounce timers by persisting files synchronously.

  Returns updated state with cleared timers.
  """
  @spec flush_all(t()) :: t()
  def flush_all(%FileSystemState{} = state) do
    # Get all paths with pending timers
    paths = Map.keys(state.debounce_timers)

    # Cancel all timers
    state = %{state | debounce_timers: %{}}

    # Persist each file synchronously
    Enum.reduce(paths, state, fn path, acc_state ->
      persist_file(acc_state, path)
    end)
  end

  @doc """
  Computes filesystem statistics.
  """
  @spec stats(t()) :: map()
  def stats(%FileSystemState{} = state) do
    all_entries = :ets.tab2list(state.fs_table)

    total_files = length(all_entries)

    memory_files =
      Enum.count(all_entries, fn {_path, entry} -> entry.persistence == :memory end)

    persisted_files =
      Enum.count(all_entries, fn {_path, entry} -> entry.persistence == :persisted end)

    loaded_files = Enum.count(all_entries, fn {_path, entry} -> entry.loaded end)
    not_loaded_files = Enum.count(all_entries, fn {_path, entry} -> not entry.loaded end)
    dirty_files = Enum.count(all_entries, fn {_path, entry} -> entry.dirty end)

    total_size =
      all_entries
      |> Enum.filter(fn {_path, entry} -> entry.loaded and not is_nil(entry.content) end)
      |> Enum.reduce(0, fn {_path, entry}, acc ->
        acc + byte_size(entry.content)
      end)

    %{
      total_files: total_files,
      memory_files: memory_files,
      persisted_files: persisted_files,
      loaded_files: loaded_files,
      not_loaded_files: not_loaded_files,
      dirty_files: dirty_files,
      pending_persist: map_size(state.debounce_timers),
      total_size: total_size
    }
  end

  # Private helpers

  defp fetch_agent_id(opts) do
    case Keyword.fetch(opts, :agent_id) do
      {:ok, agent_id} -> {:ok, agent_id}
      :error -> {:error, :agent_id_required}
    end
  end

  # Build persistence configs map from list of FileSystemConfig structs
  defp build_persistence_configs(opts) do
    opts
    |> Keyword.get(:persistence_configs, [])
    |> Enum.map(fn config -> {config.base_directory, config} end)
    |> Map.new()
  end

  # Find the persistence config that matches a given path
  defp find_config_for_path(state, path) do
    Enum.find_value(state.persistence_configs, fn {_base_dir, config} ->
      if FileSystemConfig.matches_path?(config, path), do: config
    end)
  end

  defp schedule_persist(%FileSystemState{} = state, path, config) do
    # Cancel existing timer for this path (if any)
    state = cancel_timer(state, path)

    # Start new debounce timer by sending a message
    timer_ref = Process.send_after(self(), {:persist_file, path}, config.debounce_ms)

    # Store timer reference
    %{state | debounce_timers: Map.put(state.debounce_timers, path, timer_ref)}
  end

  defp cancel_timer(%FileSystemState{} = state, path) do
    case Map.get(state.debounce_timers, path) do
      nil ->
        state

      timer_ref ->
        Process.cancel_timer(timer_ref)
        %{state | debounce_timers: Map.delete(state.debounce_timers, path)}
    end
  end
end
