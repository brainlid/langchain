defmodule LangChain.Agents.FileSystem.FileSystemStateTest do
  use ExUnit.Case, async: true

  alias LangChain.Agents.FileSystem.FileSystemState
  alias LangChain.Agents.FileSystem.FileSystemConfig

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
    test "creates state with empty files map" do
      agent_id = "test_agent_#{System.unique_integer([:positive])}"
      assert {:ok, state} = FileSystemState.new(scope_key: {:agent, agent_id})

      assert state.scope_key == {:agent, agent_id}
      assert state.files == %{}
      assert state.persistence_configs == %{}
      assert state.debounce_timers == %{}
    end
  end

  describe "write_file/4" do
    setup do
      agent_id = "test_agent_#{System.unique_integer([:positive])}"
      {:ok, state} = FileSystemState.new(scope_key: {:agent, agent_id})
      %{state: state}
    end

    test "writes a memory file", %{state: state} do
      path = "/scratch/test.txt"
      content = "test content"

      assert {:ok, new_state} = FileSystemState.write_file(state, path, content, [])

      # Verify file is in state
      assert Map.has_key?(new_state.files, path)
      entry = Map.get(new_state.files, path)
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

      # Verify file is in state
      assert Map.has_key?(new_state.files, path)
      entry = Map.get(new_state.files, path)
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
      {:ok, state} = FileSystemState.new(scope_key: {:agent, agent_id})
      %{state: state}
    end

    test "deletes a memory file", %{state: state} do
      path = "/scratch/file.txt"

      {:ok, state} = FileSystemState.write_file(state, path, "data", [])
      assert Map.has_key?(state.files, path)

      assert {:ok, new_state} = FileSystemState.delete_file(state, path)
      refute Map.has_key?(new_state.files, path)
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
      refute Map.has_key?(new_state.files, path)

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
      {:ok, state} = FileSystemState.new(scope_key: {:agent, agent_id})
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
      {:ok, state} = FileSystemState.new(scope_key: {:agent, agent_id})
      %{state: state}
    end

    test "returns stats for empty filesystem", %{state: state} do
      stats = FileSystemState.stats(state)

      assert stats.total_files == 0
      assert stats.memory_files == 0
      assert stats.persisted_files == 0
      assert stats.dirty_files == 0
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
    end
  end

  describe "reset/1" do
    setup do
      agent_id = "test_agent_#{System.unique_integer([:positive])}"
      {:ok, state} = FileSystemState.new(scope_key: {:agent, agent_id})
      %{state: state}
    end

    test "removes memory-only files", %{state: state} do
      # Write memory files
      {:ok, state} = FileSystemState.write_file(state, "/scratch/temp1.txt", "temp data 1", [])
      {:ok, state} = FileSystemState.write_file(state, "/scratch/temp2.txt", "temp data 2", [])

      # Verify files exist
      assert Map.has_key?(state.files, "/scratch/temp1.txt")
      assert Map.has_key?(state.files, "/scratch/temp2.txt")

      # Reset
      reset_state = FileSystemState.reset(state)

      # Memory files should be gone
      refute Map.has_key?(reset_state.files, "/scratch/temp1.txt")
      refute Map.has_key?(reset_state.files, "/scratch/temp2.txt")
      assert reset_state.files == %{}
    end

    test "cancels all debounce timers", %{state: state} do
      defmodule TestPersistence8 do
        @behaviour LangChain.Agents.FileSystem.Persistence

        def write_to_storage(_entry, _opts), do: :ok
        def load_from_storage(_entry, _opts), do: {:error, :enoent}
        def delete_from_storage(_entry, _opts), do: :ok
        def list_persisted_files(_agent_id, _opts), do: {:ok, []}
      end

      state = add_persistence_config(state, TestPersistence8)

      # Write a persisted file, which will create a debounce timer
      {:ok, state} = FileSystemState.write_file(state, "/Memories/file.txt", "data", [])

      # Verify timer exists
      assert Map.has_key?(state.debounce_timers, "/Memories/file.txt")
      timer_ref = state.debounce_timers["/Memories/file.txt"]
      assert is_reference(timer_ref)

      # Reset
      reset_state = FileSystemState.reset(state)

      # Timers should be cleared
      assert reset_state.debounce_timers == %{}
    end

    test "re-indexes persisted files from storage", %{state: state} do
      defmodule TestPersistence9 do
        @behaviour LangChain.Agents.FileSystem.Persistence

        def write_to_storage(_entry, _opts), do: :ok
        def load_from_storage(_entry, _opts), do: {:error, :enoent}
        def delete_from_storage(_entry, _opts), do: :ok

        # Simulate files existing in storage
        def list_persisted_files(_agent_id, _opts) do
          {:ok, ["/Memories/existing1.txt", "/Memories/existing2.txt"]}
        end
      end

      state = add_persistence_config(state, TestPersistence9)

      # Initially, the files should be indexed from storage
      assert Map.has_key?(state.files, "/Memories/existing1.txt")
      assert Map.has_key?(state.files, "/Memories/existing2.txt")

      # Add some memory files
      {:ok, state} = FileSystemState.write_file(state, "/scratch/temp.txt", "temp", [])

      # Reset
      reset_state = FileSystemState.reset(state)

      # Memory files should be gone
      refute Map.has_key?(reset_state.files, "/scratch/temp.txt")

      # Persisted files should still be indexed (but unloaded)
      assert Map.has_key?(reset_state.files, "/Memories/existing1.txt")
      assert Map.has_key?(reset_state.files, "/Memories/existing2.txt")

      # Files should be marked as unloaded
      entry1 = reset_state.files["/Memories/existing1.txt"]
      assert entry1.loaded == false
      assert entry1.dirty == false
    end

    test "works with empty filesystem", %{state: state} do
      # Reset empty state
      reset_state = FileSystemState.reset(state)

      assert reset_state.files == %{}
      assert reset_state.debounce_timers == %{}
      assert reset_state.scope_key == state.scope_key
      assert reset_state.persistence_configs == state.persistence_configs
    end

    test "clears dirty flags and unloads modified persisted files", %{state: state} do
      defmodule TestPersistence10 do
        @behaviour LangChain.Agents.FileSystem.Persistence

        def write_to_storage(_entry, _opts), do: :ok

        def load_from_storage(_entry, _opts) do
          # Return original content from storage
          {:ok, "original content from storage"}
        end

        def delete_from_storage(_entry, _opts), do: :ok

        def list_persisted_files(_agent_id, _opts) do
          {:ok, ["/Memories/file.txt"]}
        end
      end

      state = add_persistence_config(state, TestPersistence10)

      # The file is indexed from storage
      assert Map.has_key?(state.files, "/Memories/file.txt")

      # Modify the file (this would load and mark it dirty)
      {:ok, state} = FileSystemState.write_file(state, "/Memories/file.txt", "modified", [])

      # Verify it's dirty and loaded
      entry = state.files["/Memories/file.txt"]
      assert entry.dirty == true
      assert entry.loaded == true
      assert entry.content == "modified"

      # Reset
      reset_state = FileSystemState.reset(state)

      # File should still be indexed (persisted), but unloaded and not dirty
      assert Map.has_key?(reset_state.files, "/Memories/file.txt")
      reset_entry = reset_state.files["/Memories/file.txt"]
      assert reset_entry.loaded == false
      assert reset_entry.dirty == false
      # Content would be nil since it's unloaded
    end
  end
end
