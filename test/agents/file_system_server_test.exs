defmodule LangChain.Agents.FileSystemServerTest do
  use ExUnit.Case, async: true

  alias LangChain.Agents.FileSystemServer
  alias LangChain.Agents.FileSystem.{FileEntry, FileSystemConfig, FileSystemState}

  # Mock persistence modules for testing
  defmodule MockPersistence do
    @behaviour LangChain.Agents.FileSystem.Persistence

    def write_to_storage(entry, _opts), do: {:ok, %{entry | dirty: false}}
    def load_from_storage(_entry, _opts), do: {:error, :enoent}
    def delete_from_storage(_entry, _opts), do: :ok
    def list_persisted_files(_agent_id, _opts), do: {:ok, []}
  end

  defmodule TestModule do
    @behaviour LangChain.Agents.FileSystem.Persistence

    def write_to_storage(entry, _opts), do: {:ok, %{entry | dirty: false}}
    def load_from_storage(_entry, _opts), do: {:error, :enoent}
    def delete_from_storage(_entry, _opts), do: :ok
    def list_persisted_files(_agent_id, _opts), do: {:ok, []}
  end

  setup do
    # Start a test registry for this test
    agent_id = "test_agent_#{System.unique_integer([:positive])}"

    {:ok, _registry} =
      start_supervised({Registry, keys: :unique, name: LangChain.Agents.Registry})

    on_exit(fn ->
      # Cleanup any running FileSystemServer
      # Wrap in try-catch because Registry might be gone already during cleanup
      try do
        case FileSystemServer.whereis(agent_id) do
          nil -> :ok
          pid -> GenServer.stop(pid, :normal)
        end
      rescue
        ArgumentError -> :ok
      catch
        :exit, _ -> :ok
      end
    end)

    %{agent_id: agent_id}
  end

  # Helper to create persistence config for tests
  defp make_config(module, base_dir, opts \\ []) do
    debounce_ms = Keyword.get(opts, :debounce_ms, 100)
    storage_opts = Keyword.get(opts, :storage_opts, [])

    {:ok, config} =
      FileSystemConfig.new(%{
        base_directory: base_dir,
        persistence_module: module,
        debounce_ms: debounce_ms,
        storage_opts: storage_opts
      })

    config
  end

  describe "start_link/1" do
    test "starts with minimal config", %{agent_id: agent_id} do
      assert {:ok, pid} = FileSystemServer.start_link(agent_id: agent_id)
      assert Process.alive?(pid)
    end

    test "starts with persistence configuration", %{agent_id: agent_id} do
      config =
        make_config(MockPersistence, "Memories",
          debounce_ms: 1000,
          storage_opts: [path: "/tmp/test"]
        )

      opts = [
        agent_id: agent_id,
        persistence_configs: [config]
      ]

      assert {:ok, pid} = FileSystemServer.start_link(opts)
      assert Process.alive?(pid)

      # Verify configuration
      configs = FileSystemServer.get_persistence_configs(agent_id)
      assert map_size(configs) == 1
      assert %{"Memories" => loaded_config} = configs
      assert loaded_config.persistence_module == MockPersistence
    end

    test "requires agent_id", %{agent_id: _agent_id} do
      assert_raise KeyError, fn ->
        FileSystemServer.start_link([])
      end
    end

    test "can be found via whereis", %{agent_id: agent_id} do
      assert FileSystemServer.whereis(agent_id) == nil

      {:ok, pid} = FileSystemServer.start_link(agent_id: agent_id)

      assert FileSystemServer.whereis(agent_id) == pid
    end
  end

  describe "named ETS table" do
    test "creates named ETS table accessible by agent_id", %{agent_id: agent_id} do
      {:ok, _pid} = FileSystemServer.start_link(agent_id: agent_id)

      table_name = FileSystemState.get_table_name(agent_id)
      assert is_atom(table_name)

      # Verify table is accessible by name
      assert :ets.info(table_name) != :undefined
    end
  end

  describe "write_file/4" do
    test "writes a memory file", %{agent_id: agent_id} do
      {:ok, _pid} = FileSystemServer.start_link(agent_id: agent_id)
      table = FileSystemState.get_table_name(agent_id)

      path = "/scratch/test.txt"
      content = "test content"

      assert :ok = FileSystemServer.write_file(agent_id, path, content)

      # Verify file exists in ETS
      assert [{^path, entry}] = :ets.lookup(table, path)
      assert entry.path == path
      assert entry.content == content
      assert entry.persistence == :memory
      assert entry.loaded == true
      assert entry.dirty == false
    end

    test "writes to unconfigured directory as memory-only", %{
      agent_id: agent_id
    } do
      {:ok, _pid} = FileSystemServer.start_link(agent_id: agent_id)
      table = FileSystemState.get_table_name(agent_id)

      path = "/Memories/important.txt"
      content = "important data"

      # No persistence config for /Memories/, so should be memory-only
      assert :ok = FileSystemServer.write_file(agent_id, path, content)

      # File should be memory-only
      assert [{^path, entry}] = :ets.lookup(table, path)
      assert entry.persistence == :memory
    end

    test "writes persisted file and schedules debounce timer", %{agent_id: agent_id} do
      defmodule TestPersistence do
        @behaviour LangChain.Agents.FileSystem.Persistence

        def write_to_storage(entry, _opts), do: {:ok, %{entry | dirty: false}}
        def load_from_storage(_entry, _opts), do: {:error, :enoent}
        def delete_from_storage(_entry, _opts), do: :ok
        def list_persisted_files(_agent_id, _opts), do: {:ok, []}
      end

      config = make_config(TestPersistence, "Memories")

      {:ok, _pid} =
        FileSystemServer.start_link(
          agent_id: agent_id,
          persistence_configs: [config]
        )

      table = FileSystemState.get_table_name(agent_id)

      path = "/Memories/file.txt"
      content = "persisted content"

      assert :ok = FileSystemServer.write_file(agent_id, path, content)

      # File should be marked as persisted and dirty
      assert [{^path, entry}] = :ets.lookup(table, path)
      assert entry.persistence == :persisted
      assert entry.dirty == true
      assert entry.loaded == true

      # Wait for debounce timer to fire
      Process.sleep(150)

      # File should now be clean
      assert [{^path, clean_entry}] = :ets.lookup(table, path)
      assert clean_entry.dirty == false
    end

    test "updates existing file and resets debounce timer", %{agent_id: agent_id} do
      test_pid = self()

      defmodule TestPersistence2 do
        @behaviour LangChain.Agents.FileSystem.Persistence

        def write_to_storage(entry, opts) do
          # Get test_pid from opts
          case Keyword.get(opts, :test_pid) do
            nil -> :ok
            test_pid -> send(test_pid, {:persisted, System.monotonic_time()})
          end

          {:ok, %{entry | dirty: false}}
        end

        def load_from_storage(_entry, _opts), do: {:error, :enoent}
        def delete_from_storage(_entry, _opts), do: :ok
        def list_persisted_files(_agent_id, _opts), do: {:ok, []}
      end

      config = make_config(TestPersistence2, "Memories", storage_opts: [test_pid: test_pid])

      {:ok, _pid} =
        FileSystemServer.start_link(
          agent_id: agent_id,
          persistence_configs: [config]
        )

      path = "/Memories/file.txt"

      # Write multiple times rapidly
      FileSystemServer.write_file(agent_id, path, "v1")
      Process.sleep(50)
      FileSystemServer.write_file(agent_id, path, "v2")
      Process.sleep(50)
      FileSystemServer.write_file(agent_id, path, "v3")

      # Should only persist once after all writes complete
      Process.sleep(150)

      # Should receive only one persist call
      assert_received {:persisted, _time}
      refute_received {:persisted, _time}
    end

    test "writes with custom metadata", %{agent_id: agent_id} do
      {:ok, _pid} = FileSystemServer.start_link(agent_id: agent_id)
      table = FileSystemState.get_table_name(agent_id)

      path = "/data.json"
      content = ~s({"key": "value"})

      opts = [
        mime_type: "application/json",
        custom: %{"author" => "test"}
      ]

      assert :ok = FileSystemServer.write_file(agent_id, path, content, opts)

      assert [{^path, entry}] = :ets.lookup(table, path)
      assert entry.metadata.mime_type == "application/json"
      assert entry.metadata.custom == %{"author" => "test"}
    end
  end

  describe "delete_file/2" do
    test "deletes memory file", %{agent_id: agent_id} do
      {:ok, _pid} = FileSystemServer.start_link(agent_id: agent_id)
      table = FileSystemState.get_table_name(agent_id)

      path = "/scratch/file.txt"
      FileSystemServer.write_file(agent_id, path, "data")
      assert :ets.lookup(table, path) != []

      assert :ok = FileSystemServer.delete_file(agent_id, path)
      assert :ets.lookup(table, path) == []
    end

    test "deletes persisted file and cancels timer", %{agent_id: agent_id} do
      test_pid = self()

      defmodule TestPersistence3 do
        @behaviour LangChain.Agents.FileSystem.Persistence

        def write_to_storage(entry, _opts), do: {:ok, %{entry | dirty: false}}

        def load_from_storage(_entry, _opts), do: {:error, :enoent}

        def delete_from_storage(_entry, opts) do
          # Get test_pid from opts
          case Keyword.get(opts, :test_pid) do
            nil -> :ok
            test_pid -> send(test_pid, :deleted_from_storage)
          end

          :ok
        end

        def list_persisted_files(_agent_id, _opts), do: {:ok, []}
      end

      config =
        make_config(TestPersistence3, "Memories",
          debounce_ms: 5000,
          storage_opts: [test_pid: test_pid]
        )

      {:ok, _pid} =
        FileSystemServer.start_link(
          agent_id: agent_id,
          persistence_configs: [config]
        )

      table = FileSystemState.get_table_name(agent_id)

      path = "/Memories/file.txt"
      FileSystemServer.write_file(agent_id, path, "data")

      # Verify file exists and is dirty (pending persist)
      stats = FileSystemServer.stats(agent_id)
      assert stats.dirty_files == 1

      # Delete the file
      assert :ok = FileSystemServer.delete_file(agent_id, path)

      # Verify storage deletion was called
      assert_received :deleted_from_storage

      # Verify file is gone from ETS
      assert :ets.lookup(table, path) == []

      # Verify no dirty files remain
      stats = FileSystemServer.stats(agent_id)
      assert stats.dirty_files == 0
    end

    test "deletes non-existent file returns ok", %{agent_id: agent_id} do
      {:ok, _pid} = FileSystemServer.start_link(agent_id: agent_id)

      assert :ok = FileSystemServer.delete_file(agent_id, "/nonexistent.txt")
    end
  end

  describe "flush_all/1" do
    test "persists all dirty files immediately", %{agent_id: agent_id} do
      test_pid = self()

      defmodule TestPersistence4 do
        @behaviour LangChain.Agents.FileSystem.Persistence

        def write_to_storage(entry, opts) do
          # Get test_pid from opts
          case Keyword.get(opts, :test_pid) do
            nil -> :ok
            test_pid -> send(test_pid, {:flushed, entry.path})
          end

          {:ok, %{entry | dirty: false}}
        end

        def load_from_storage(_entry, _opts), do: {:error, :enoent}
        def delete_from_storage(_entry, _opts), do: :ok
        def list_persisted_files(_agent_id, _opts), do: {:ok, []}
      end

      config =
        make_config(TestPersistence4, "Memories",
          debounce_ms: 10000,
          storage_opts: [test_pid: test_pid]
        )

      {:ok, _pid} =
        FileSystemServer.start_link(
          agent_id: agent_id,
          persistence_configs: [config]
        )

      # Write multiple files
      FileSystemServer.write_file(agent_id, "/Memories/file1.txt", "data1")
      FileSystemServer.write_file(agent_id, "/Memories/file2.txt", "data2")
      FileSystemServer.write_file(agent_id, "/Memories/file3.txt", "data3")

      # Verify files are dirty (timers pending)
      stats_before = FileSystemServer.stats(agent_id)
      assert stats_before.dirty_files == 3

      # Flush all
      assert :ok = FileSystemServer.flush_all(agent_id)

      # Should receive persist calls for all files
      assert_received {:flushed, "/Memories/file1.txt"}
      assert_received {:flushed, "/Memories/file2.txt"}
      assert_received {:flushed, "/Memories/file3.txt"}

      # Give it time to process
      Process.sleep(50)

      # Verify all files are now clean and no timers pending
      stats_after = FileSystemServer.stats(agent_id)
      assert stats_after.dirty_files == 0
    end
  end

  describe "stats/1" do
    test "returns filesystem statistics", %{agent_id: agent_id} do
      {:ok, _pid} = FileSystemServer.start_link(agent_id: agent_id)

      # Empty filesystem
      stats = FileSystemServer.stats(agent_id)
      assert stats.total_files == 0
      assert stats.memory_files == 0
      assert stats.persisted_files == 0
      assert stats.loaded_files == 0
      assert stats.not_loaded_files == 0
      assert stats.dirty_files == 0
      assert stats.total_size == 0
    end

    test "counts different file types correctly", %{agent_id: agent_id} do
      defmodule TestPersistence5 do
        @behaviour LangChain.Agents.FileSystem.Persistence

        def write_to_storage(entry, _opts), do: {:ok, %{entry | dirty: false}}
        def load_from_storage(_entry, _opts), do: {:error, :enoent}
        def delete_from_storage(_entry, _opts), do: :ok
        def list_persisted_files(_agent_id, _opts), do: {:ok, []}
      end

      config = make_config(TestPersistence5, "Memories", debounce_ms: 5000)

      {:ok, _pid} =
        FileSystemServer.start_link(
          agent_id: agent_id,
          persistence_configs: [config]
        )

      # Write memory files
      FileSystemServer.write_file(agent_id, "/scratch/file1.txt", "data1")
      FileSystemServer.write_file(agent_id, "/scratch/file2.txt", "data2")

      # Write persisted files
      FileSystemServer.write_file(agent_id, "/Memories/file3.txt", "data3")
      FileSystemServer.write_file(agent_id, "/Memories/file4.txt", "data4")

      stats = FileSystemServer.stats(agent_id)

      assert stats.total_files == 4
      assert stats.memory_files == 2
      assert stats.persisted_files == 2
      assert stats.loaded_files == 4
      assert stats.not_loaded_files == 0
      assert stats.dirty_files == 2
      assert stats.total_size == byte_size("data1data2data3data4")
    end
  end

  describe "get_persistence_configs/1" do
    test "returns persistence configurations", %{agent_id: agent_id} do
      config = make_config(TestModule, "data", storage_opts: [path: "/test"])

      {:ok, _pid} =
        FileSystemServer.start_link(
          agent_id: agent_id,
          persistence_configs: [config]
        )

      configs = FileSystemServer.get_persistence_configs(agent_id)

      assert map_size(configs) == 1
      assert %{"data" => loaded_config} = configs
      assert loaded_config.persistence_module == TestModule
      assert loaded_config.storage_opts[:path] == "/test"
      assert loaded_config.base_directory == "data"
    end

    test "returns empty map when no persistence configured", %{agent_id: agent_id} do
      {:ok, _pid} = FileSystemServer.start_link(agent_id: agent_id)

      configs = FileSystemServer.get_persistence_configs(agent_id)

      assert configs == %{}
    end
  end

  describe "terminate/2" do
    test "flushes pending writes on termination", %{agent_id: agent_id} do
      defmodule TestPersistence6 do
        @behaviour LangChain.Agents.FileSystem.Persistence

        def write_to_storage(entry, _opts) do
          send(:test_process, {:flushed_on_terminate, entry.path})
          {:ok, %{entry | dirty: false}}
        end

        def load_from_storage(_entry, _opts), do: {:error, :enoent}
        def delete_from_storage(_entry, _opts), do: :ok
        def list_persisted_files(_agent_id, _opts), do: {:ok, []}
      end

      Process.register(self(), :test_process)

      config = make_config(TestPersistence6, "Memories", debounce_ms: 10000)

      {:ok, pid} =
        FileSystemServer.start_link(
          agent_id: agent_id,
          persistence_configs: [config]
        )

      # Write files with long debounce
      FileSystemServer.write_file(agent_id, "/Memories/file1.txt", "data1")
      FileSystemServer.write_file(agent_id, "/Memories/file2.txt", "data2")

      # Stop the server (should flush pending writes)
      GenServer.stop(pid, :normal)

      # Should have received flush calls
      assert_received {:flushed_on_terminate, "/Memories/file1.txt"}
      assert_received {:flushed_on_terminate, "/Memories/file2.txt"}
    end
  end

  describe "configurable memories directory" do
    test "uses custom memories_directory from storage_opts", %{agent_id: agent_id} do
      defmodule TestPersistence7 do
        @behaviour LangChain.Agents.FileSystem.Persistence

        def write_to_storage(entry, _opts), do: {:ok, %{entry | dirty: false}}
        def load_from_storage(_entry, _opts), do: {:error, :enoent}
        def delete_from_storage(_entry, _opts), do: :ok
        def list_persisted_files(_agent_id, _opts), do: {:ok, []}
      end

      config = make_config(TestPersistence7, "persistent")

      {:ok, _pid} =
        FileSystemServer.start_link(
          agent_id: agent_id,
          persistence_configs: [config]
        )

      table = FileSystemState.get_table_name(agent_id)

      # Files under /persistent/ should be persisted
      FileSystemServer.write_file(agent_id, "/persistent/file.txt", "data")
      assert [{_, entry}] = :ets.lookup(table, "/persistent/file.txt")
      assert entry.persistence == :persisted

      # Files under /Memories/ should be memory-only with this config
      FileSystemServer.write_file(agent_id, "/Memories/file.txt", "data")
      assert [{_, entry2}] = :ets.lookup(table, "/Memories/file.txt")
      assert entry2.persistence == :memory
    end
  end

  describe "register_files/2" do
    test "registers a single file entry", %{agent_id: agent_id} do
      {:ok, _pid} = FileSystemServer.start_link(agent_id: agent_id)

      # Create a file entry
      {:ok, entry} = FileEntry.new_memory_file("/test/data.txt", "test content")

      # Register it
      assert :ok = FileSystemServer.register_files(agent_id, entry)

      # Should be able to read it immediately
      assert {:ok, content} = FileSystemServer.read_file(agent_id, "/test/data.txt")
      assert content == "test content"
    end

    test "registers multiple file entries", %{agent_id: agent_id} do
      {:ok, _pid} = FileSystemServer.start_link(agent_id: agent_id)

      # Create multiple file entries
      {:ok, entry1} = FileEntry.new_memory_file("/test/file1.txt", "content1")
      {:ok, entry2} = FileEntry.new_memory_file("/test/file2.txt", "content2")
      {:ok, entry3} = FileEntry.new_memory_file("/test/file3.txt", "content3")

      # Register them all at once
      assert :ok = FileSystemServer.register_files(agent_id, [entry1, entry2, entry3])

      # All should be readable
      assert {:ok, "content1"} = FileSystemServer.read_file(agent_id, "/test/file1.txt")
      assert {:ok, "content2"} = FileSystemServer.read_file(agent_id, "/test/file2.txt")
      assert {:ok, "content3"} = FileSystemServer.read_file(agent_id, "/test/file3.txt")
    end

    test "registers indexed files for lazy loading", %{agent_id: agent_id} do
      {:ok, _pid} = FileSystemServer.start_link(agent_id: agent_id)

      # Create indexed file entries (not loaded)
      {:ok, entry1} = FileEntry.new_indexed_file("/data/lazy1.txt")
      {:ok, entry2} = FileEntry.new_indexed_file("/data/lazy2.txt")

      # Register them
      assert :ok = FileSystemServer.register_files(agent_id, [entry1, entry2])

      table = FileSystemState.get_table_name(agent_id)

      # Files should exist but not be loaded
      assert [{"/data/lazy1.txt", e1}] = :ets.lookup(table, "/data/lazy1.txt")
      assert e1.loaded == false
      assert e1.content == nil

      assert [{"/data/lazy2.txt", e2}] = :ets.lookup(table, "/data/lazy2.txt")
      assert e2.loaded == false
      assert e2.content == nil
    end
  end

  describe "concurrent operations" do
    test "handles multiple writes to different files", %{agent_id: agent_id} do
      {:ok, _pid} = FileSystemServer.start_link(agent_id: agent_id)
      table = FileSystemState.get_table_name(agent_id)

      # Simulate concurrent writes
      tasks =
        for i <- 1..10 do
          Task.async(fn ->
            FileSystemServer.write_file(agent_id, "/file#{i}.txt", "data#{i}")
          end)
        end

      results = Task.await_many(tasks)
      assert Enum.all?(results, &(&1 == :ok))

      # Verify all files exist
      for i <- 1..10 do
        assert [{_, entry}] = :ets.lookup(table, "/file#{i}.txt")
        assert entry.content == "data#{i}"
      end
    end
  end

  describe "read_file/2" do
    test "reads memory file directly from ETS", %{agent_id: agent_id} do
      {:ok, _pid} = FileSystemServer.start_link(agent_id: agent_id)

      # Write a memory file
      :ok = FileSystemServer.write_file(agent_id, "/scratch/notes.txt", "My notes")

      # Read should work immediately
      assert {:ok, "My notes"} = FileSystemServer.read_file(agent_id, "/scratch/notes.txt")
    end

    test "returns error for nonexistent file", %{agent_id: agent_id} do
      {:ok, _pid} = FileSystemServer.start_link(agent_id: agent_id)

      assert {:error, :enoent} = FileSystemServer.read_file(agent_id, "/nonexistent.txt")
    end

    test "lazy loads persisted file on first read", %{agent_id: agent_id} do
      # Create a fake persistence module that tracks when files are loaded
      test_pid = self()

      # Create a shared ETS table for fake storage
      storage_table = :ets.new(:test_storage_8, [:set, :public])
      :ets.insert(storage_table, {"/Memories/data.txt", "persisted content"})

      defmodule TestPersistence8 do
        @behaviour LangChain.Agents.FileSystem.Persistence

        def write_to_storage(entry, opts) do
          # Store in ETS to simulate persistence
          storage_table = Keyword.get(opts, :storage_table)
          :ets.insert(storage_table, {entry.path, entry.content})
          {:ok, %{entry | dirty: false}}
        end

        def load_from_storage(%{path: path} = entry, opts) do
          # Notify test when load happens
          test_pid = Keyword.get(opts, :test_pid)
          storage_table = Keyword.get(opts, :storage_table)

          if test_pid, do: send(test_pid, {:loaded, path})

          # Load from ETS
          case :ets.lookup(storage_table, path) do
            [{^path, content}] ->
              {:ok, %{entry | content: content, loaded: true, dirty: false}}
            [] -> {:error, :enoent}
          end
        end

        def delete_from_storage(_entry, _opts), do: :ok

        def list_persisted_files(_agent_id, opts) do
          # Return files from ETS
          storage_table = Keyword.get(opts, :storage_table)
          paths = :ets.tab2list(storage_table) |> Enum.map(fn {path, _} -> path end)
          {:ok, paths}
        end
      end

      config = make_config(TestPersistence8, "Memories", storage_opts: [test_pid: test_pid, storage_table: storage_table])

      {:ok, _pid} =
        FileSystemServer.start_link(
          agent_id: agent_id,
          persistence_configs: [config]
        )

      table = FileSystemState.get_table_name(agent_id)

      # File should be indexed but NOT loaded
      assert [{"/Memories/data.txt", entry}] = :ets.lookup(table, "/Memories/data.txt")
      assert entry.persistence == :persisted
      assert entry.loaded == false
      assert entry.content == nil

      # We should NOT have received a load message yet
      refute_received {:loaded, _}

      # Now read the file - this should trigger lazy loading
      assert {:ok, content} = FileSystemServer.read_file(agent_id, "/Memories/data.txt")
      assert content == "persisted content"

      # Should have received load message
      assert_receive {:loaded, "/Memories/data.txt"}, 100

      # File should now be loaded
      assert [{"/Memories/data.txt", loaded_entry}] = :ets.lookup(table, "/Memories/data.txt")
      assert loaded_entry.loaded == true
      assert loaded_entry.content == "persisted content"

      # Cleanup
      :ets.delete(storage_table)
    end

    test "supports concurrent reads from ETS without GenServer bottleneck", %{agent_id: agent_id} do
      {:ok, _pid} = FileSystemServer.start_link(agent_id: agent_id)

      # Write files
      for i <- 1..20 do
        :ok = FileSystemServer.write_file(agent_id, "/file#{i}.txt", "content#{i}")
      end

      # Simulate concurrent reads
      tasks =
        for i <- 1..20 do
          Task.async(fn ->
            FileSystemServer.read_file(agent_id, "/file#{i}.txt")
          end)
        end

      # All reads should succeed
      results = Task.await_many(tasks)

      for {result, i} <- Enum.zip(results, 1..20) do
        expected = "content#{i}"
        assert {:ok, ^expected} = result
      end
    end

    test "supports concurrent reads after files are loaded", %{agent_id: agent_id} do
      {:ok, _pid} = FileSystemServer.start_link(agent_id: agent_id)

      # Write multiple files
      for i <- 1..5 do
        :ok = FileSystemServer.write_file(agent_id, "/file#{i}.txt", "content#{i}")
      end

      # Concurrent reads should all succeed
      tasks =
        for i <- 1..5 do
          Task.async(fn ->
            FileSystemServer.read_file(agent_id, "/file#{i}.txt")
          end)
        end

      results = Task.await_many(tasks, 5000)

      # All should succeed with correct content
      for {result, i} <- Enum.zip(results, 1..5) do
        expected = "content#{i}"
        assert {:ok, ^expected} = result
      end
    end
  end
end
