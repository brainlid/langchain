defmodule LangChain.Agents.FileSystem.FileSystemStateTest do
  use ExUnit.Case, async: true

  alias LangChain.Agents.FileSystem.{FileSystemState, FileSystemConfig}

  # Helper to add persistence config to state
  defp add_persistence_config(state, module, base_dir \\ "Memories", opts \\ []) do
    debounce_ms = Keyword.get(opts, :debounce_ms, 5000)
    storage_opts = Keyword.get(opts, :storage_opts, [])

    {:ok, config} =
      FileSystemConfig.new(%{
        base_directory: base_dir,
        persistence_module: module,
        debounce_ms: debounce_ms,
        storage_opts: storage_opts
      })

    {:ok, new_state} = FileSystemState.register_persistence(state, config)
    new_state
  end

  describe "new/1" do
    test "creates state with ETS table" do
      agent_id = "test_agent_#{System.unique_integer([:positive])}"
      assert {:ok, state} = FileSystemState.new(agent_id: agent_id)

      assert state.agent_id == agent_id
      assert is_reference(state.fs_table)
      assert state.persistence_configs == %{}
      assert state.debounce_timers == %{}

      # Verify ETS table exists and is accessible
      assert :ets.info(state.fs_table) != :undefined
    end
  end

  describe "write_file/4" do
    setup do
      agent_id = "test_agent_#{System.unique_integer([:positive])}"
      {:ok, state} = FileSystemState.new(agent_id: agent_id)
      %{state: state}
    end

    test "writes a memory file", %{state: state} do
      path = "/scratch/test.txt"
      content = "test content"

      assert {:ok, new_state} = FileSystemState.write_file(state, path, content, [])

      # Verify file is in ETS
      assert [{^path, entry}] = :ets.lookup(new_state.fs_table, path)
      assert entry.path == path
      assert entry.content == content
      assert entry.persistence == :memory
      assert entry.loaded == true
      assert entry.dirty == false
    end

    test "writes a persisted file and schedules debounce timer", %{state: state} do
      defmodule TestPersistence1 do
        @behaviour LangChain.Agents.FileSystem.Persistence

        def write_to_storage(_entry, _opts), do: :ok
        def load_from_storage(_entry, _opts), do: {:error, :enoent}
        def delete_from_storage(_entry, _opts), do: :ok
        def list_persisted_files(_agent_id, _opts), do: {:ok, []}
      end

      state = add_persistence_config(state, TestPersistence1)

      path = "/Memories/file.txt"
      content = "persisted content"

      assert {:ok, new_state} = FileSystemState.write_file(state, path, content, [])

      # Verify file is in ETS
      assert [{^path, entry}] = :ets.lookup(new_state.fs_table, path)
      assert entry.persistence == :persisted
      assert entry.dirty == true
      assert entry.loaded == true

      # Verify debounce timer was created
      assert Map.has_key?(new_state.debounce_timers, path)
      assert is_reference(new_state.debounce_timers[path])
    end

    test "rejects writes to readonly directories", %{state: state} do
      defmodule TestPersistence2 do
        @behaviour LangChain.Agents.FileSystem.Persistence

        def write_to_storage(_entry, _opts), do: :ok
        def load_from_storage(_entry, _opts), do: {:error, :enoent}
        def delete_from_storage(_entry, _opts), do: :ok
        def list_persisted_files(_agent_id, _opts), do: {:ok, []}
      end

      {:ok, config} =
        FileSystemConfig.new(%{
          base_directory: "readonly",
          persistence_module: TestPersistence2,
          readonly: true
        })

      {:ok, state} = FileSystemState.register_persistence(state, config)

      path = "/readonly/file.txt"
      content = "data"

      assert {:error, reason, _state} = FileSystemState.write_file(state, path, content, [])
      assert reason =~ "read-only"
    end
  end

  describe "delete_file/2" do
    setup do
      agent_id = "test_agent_#{System.unique_integer([:positive])}"
      {:ok, state} = FileSystemState.new(agent_id: agent_id)
      %{state: state}
    end

    test "deletes a memory file", %{state: state} do
      path = "/scratch/file.txt"

      {:ok, state} = FileSystemState.write_file(state, path, "data", [])
      assert :ets.lookup(state.fs_table, path) != []

      assert {:ok, new_state} = FileSystemState.delete_file(state, path)
      assert :ets.lookup(new_state.fs_table, path) == []
    end

    test "deletes a persisted file and cancels timer", %{state: state} do
      defmodule TestPersistence3 do
        @behaviour LangChain.Agents.FileSystem.Persistence

        def write_to_storage(_entry, _opts), do: :ok
        def load_from_storage(_entry, _opts), do: {:error, :enoent}
        def delete_from_storage(_entry, _opts), do: :ok
        def list_persisted_files(_agent_id, _opts), do: {:ok, []}
      end

      state = add_persistence_config(state, TestPersistence3)

      path = "/Memories/file.txt"
      {:ok, state} = FileSystemState.write_file(state, path, "data", [])

      # Verify timer exists
      assert Map.has_key?(state.debounce_timers, path)

      assert {:ok, new_state} = FileSystemState.delete_file(state, path)

      # Verify file is gone
      assert :ets.lookup(new_state.fs_table, path) == []

      # Verify timer was cancelled
      refute Map.has_key?(new_state.debounce_timers, path)
    end

    test "rejects delete from readonly directories", %{state: state} do
      defmodule TestPersistence4 do
        @behaviour LangChain.Agents.FileSystem.Persistence

        def write_to_storage(_entry, _opts), do: :ok
        def load_from_storage(_entry, _opts), do: {:error, :enoent}
        def delete_from_storage(_entry, _opts), do: :ok
        def list_persisted_files(_agent_id, _opts), do: {:ok, []}
      end

      {:ok, config} =
        FileSystemConfig.new(%{
          base_directory: "readonly",
          persistence_module: TestPersistence4,
          readonly: true
        })

      {:ok, state} = FileSystemState.register_persistence(state, config)

      path = "/readonly/file.txt"

      assert {:error, reason, _state} = FileSystemState.delete_file(state, path)
      assert reason =~ "read-only"
    end
  end

  describe "register_persistence/2" do
    setup do
      agent_id = "test_agent_#{System.unique_integer([:positive])}"
      {:ok, state} = FileSystemState.new(agent_id: agent_id)
      %{state: state}
    end

    test "registers a new persistence config", %{state: state} do
      defmodule TestPersistence5 do
        @behaviour LangChain.Agents.FileSystem.Persistence

        def write_to_storage(_entry, _opts), do: :ok
        def load_from_storage(_entry, _opts), do: {:error, :enoent}
        def delete_from_storage(_entry, _opts), do: :ok
        def list_persisted_files(_agent_id, _opts), do: {:ok, []}
      end

      {:ok, config} =
        FileSystemConfig.new(%{
          base_directory: "user_files",
          persistence_module: TestPersistence5
        })

      assert {:ok, new_state} = FileSystemState.register_persistence(state, config)
      assert map_size(new_state.persistence_configs) == 1
      assert Map.has_key?(new_state.persistence_configs, "user_files")
    end

    test "prevents registering same base_directory twice", %{state: state} do
      defmodule TestPersistence6 do
        @behaviour LangChain.Agents.FileSystem.Persistence

        def write_to_storage(_entry, _opts), do: :ok
        def load_from_storage(_entry, _opts), do: {:error, :enoent}
        def delete_from_storage(_entry, _opts), do: :ok
        def list_persisted_files(_agent_id, _opts), do: {:ok, []}
      end

      {:ok, config1} =
        FileSystemConfig.new(%{
          base_directory: "user_files",
          persistence_module: TestPersistence6
        })

      {:ok, state} = FileSystemState.register_persistence(state, config1)

      {:ok, config2} =
        FileSystemConfig.new(%{
          base_directory: "user_files",
          persistence_module: TestPersistence6,
          debounce_ms: 10000
        })

      assert {:error, reason} = FileSystemState.register_persistence(state, config2)
      assert reason =~ "already has a registered persistence config"
    end
  end

  describe "stats/1" do
    setup do
      agent_id = "test_agent_#{System.unique_integer([:positive])}"
      {:ok, state} = FileSystemState.new(agent_id: agent_id)
      %{state: state}
    end

    test "returns stats for empty filesystem", %{state: state} do
      stats = FileSystemState.stats(state)

      assert stats.total_files == 0
      assert stats.memory_files == 0
      assert stats.persisted_files == 0
      assert stats.dirty_files == 0
      assert stats.pending_persist == 0
    end

    test "counts different file types correctly", %{state: state} do
      defmodule TestPersistence7 do
        @behaviour LangChain.Agents.FileSystem.Persistence

        def write_to_storage(_entry, _opts), do: :ok
        def load_from_storage(_entry, _opts), do: {:error, :enoent}
        def delete_from_storage(_entry, _opts), do: :ok
        def list_persisted_files(_agent_id, _opts), do: {:ok, []}
      end

      state = add_persistence_config(state, TestPersistence7)

      # Write memory files
      {:ok, state} = FileSystemState.write_file(state, "/scratch/file1.txt", "data1", [])
      {:ok, state} = FileSystemState.write_file(state, "/scratch/file2.txt", "data2", [])

      # Write persisted files
      {:ok, state} = FileSystemState.write_file(state, "/Memories/file3.txt", "data3", [])
      {:ok, state} = FileSystemState.write_file(state, "/Memories/file4.txt", "data4", [])

      stats = FileSystemState.stats(state)

      assert stats.total_files == 4
      assert stats.memory_files == 2
      assert stats.persisted_files == 2
      assert stats.dirty_files == 2
      assert stats.pending_persist == 2
    end
  end
end
