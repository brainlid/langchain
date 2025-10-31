defmodule LangChain.Agents.FileSystemServerTest do
  use ExUnit.Case, async: true

  alias LangChain.Agents.FileSystemServer
  alias LangChain.Agents.FileSystem.FileSystemConfig

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
      configs = FileSystemServer.get_persistence_configs(pid)
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

  describe "get_table/1" do
    test "returns ETS table reference", %{agent_id: agent_id} do
      {:ok, pid} = FileSystemServer.start_link(agent_id: agent_id)

      table = FileSystemServer.get_table(pid)
      assert is_reference(table)

      # Verify table is accessible
      assert :ets.info(table) != :undefined
    end
  end

  describe "write_file/4" do
    test "writes a memory file", %{agent_id: agent_id} do
      {:ok, pid} = FileSystemServer.start_link(agent_id: agent_id)
      table = FileSystemServer.get_table(pid)

      path = "/scratch/test.txt"
      content = "test content"

      assert :ok = FileSystemServer.write_file(pid, path, content)

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
      {:ok, pid} = FileSystemServer.start_link(agent_id: agent_id)
      table = FileSystemServer.get_table(pid)

      path = "/Memories/important.txt"
      content = "important data"

      # No persistence config for /Memories/, so should be memory-only
      assert :ok = FileSystemServer.write_file(pid, path, content)

      # File should be memory-only
      assert [{^path, entry}] = :ets.lookup(table, path)
      assert entry.persistence == :memory
    end

    test "writes persisted file and schedules debounce timer", %{agent_id: agent_id} do
      defmodule TestPersistence do
        @behaviour LangChain.Agents.FileSystem.Persistence

        def write_to_storage(_entry, _opts), do: :ok
        def load_from_storage(_entry, _opts), do: {:error, :enoent}
        def delete_from_storage(_entry, _opts), do: :ok
        def list_persisted_files(_agent_id, _opts), do: {:ok, []}
      end

      config = make_config(TestPersistence, "Memories")

      {:ok, pid} =
        FileSystemServer.start_link(
          agent_id: agent_id,
          persistence_configs: [config]
        )

      table = FileSystemServer.get_table(pid)

      path = "/Memories/file.txt"
      content = "persisted content"

      assert :ok = FileSystemServer.write_file(pid, path, content)

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

        def write_to_storage(_entry, opts) do
          # Get test_pid from opts
          case Keyword.get(opts, :test_pid) do
            nil -> :ok
            test_pid -> send(test_pid, {:persisted, System.monotonic_time()})
          end

          :ok
        end

        def load_from_storage(_entry, _opts), do: {:error, :enoent}
        def delete_from_storage(_entry, _opts), do: :ok
        def list_persisted_files(_agent_id, _opts), do: {:ok, []}
      end

      config = make_config(TestPersistence2, "Memories", storage_opts: [test_pid: test_pid])

      {:ok, pid} =
        FileSystemServer.start_link(
          agent_id: agent_id,
          persistence_configs: [config]
        )

      path = "/Memories/file.txt"

      # Write multiple times rapidly
      FileSystemServer.write_file(pid, path, "v1")
      Process.sleep(50)
      FileSystemServer.write_file(pid, path, "v2")
      Process.sleep(50)
      FileSystemServer.write_file(pid, path, "v3")

      # Should only persist once after all writes complete
      Process.sleep(150)

      # Should receive only one persist call
      assert_received {:persisted, _time}
      refute_received {:persisted, _time}
    end

    test "writes with custom metadata", %{agent_id: agent_id} do
      {:ok, pid} = FileSystemServer.start_link(agent_id: agent_id)
      table = FileSystemServer.get_table(pid)

      path = "/data.json"
      content = ~s({"key": "value"})

      opts = [
        mime_type: "application/json",
        custom: %{"author" => "test"}
      ]

      assert :ok = FileSystemServer.write_file(pid, path, content, opts)

      assert [{^path, entry}] = :ets.lookup(table, path)
      assert entry.metadata.mime_type == "application/json"
      assert entry.metadata.custom == %{"author" => "test"}
    end
  end

  describe "delete_file/2" do
    test "deletes memory file", %{agent_id: agent_id} do
      {:ok, pid} = FileSystemServer.start_link(agent_id: agent_id)
      table = FileSystemServer.get_table(pid)

      path = "/scratch/file.txt"
      FileSystemServer.write_file(pid, path, "data")
      assert :ets.lookup(table, path) != []

      assert :ok = FileSystemServer.delete_file(pid, path)
      assert :ets.lookup(table, path) == []
    end

    test "deletes persisted file and cancels timer", %{agent_id: agent_id} do
      test_pid = self()

      defmodule TestPersistence3 do
        @behaviour LangChain.Agents.FileSystem.Persistence

        def write_to_storage(_entry, _opts), do: :ok

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

      {:ok, pid} =
        FileSystemServer.start_link(
          agent_id: agent_id,
          persistence_configs: [config]
        )

      table = FileSystemServer.get_table(pid)

      path = "/Memories/file.txt"
      FileSystemServer.write_file(pid, path, "data")

      # Verify file exists and has pending timer
      {:ok, stats} = FileSystemServer.stats(pid)
      assert stats.pending_persist == 1

      # Delete the file
      assert :ok = FileSystemServer.delete_file(pid, path)

      # Verify storage deletion was called
      assert_received :deleted_from_storage

      # Verify file is gone from ETS
      assert :ets.lookup(table, path) == []

      # Verify timer was cancelled
      {:ok, stats} = FileSystemServer.stats(pid)
      assert stats.pending_persist == 0
    end

    test "deletes non-existent file returns ok", %{agent_id: agent_id} do
      {:ok, pid} = FileSystemServer.start_link(agent_id: agent_id)

      assert :ok = FileSystemServer.delete_file(pid, "/nonexistent.txt")
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

          :ok
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

      {:ok, pid} =
        FileSystemServer.start_link(
          agent_id: agent_id,
          persistence_configs: [config]
        )

      # Write multiple files
      FileSystemServer.write_file(pid, "/Memories/file1.txt", "data1")
      FileSystemServer.write_file(pid, "/Memories/file2.txt", "data2")
      FileSystemServer.write_file(pid, "/Memories/file3.txt", "data3")

      # Verify timers are pending
      {:ok, stats_before} = FileSystemServer.stats(pid)
      assert stats_before.pending_persist == 3
      assert stats_before.dirty_files == 3

      # Flush all
      assert :ok = FileSystemServer.flush_all(pid)

      # Should receive persist calls for all files
      assert_received {:flushed, "/Memories/file1.txt"}
      assert_received {:flushed, "/Memories/file2.txt"}
      assert_received {:flushed, "/Memories/file3.txt"}

      # Give it time to process
      Process.sleep(50)

      # Verify all files are now clean and no timers pending
      {:ok, stats_after} = FileSystemServer.stats(pid)
      assert stats_after.pending_persist == 0
      assert stats_after.dirty_files == 0
    end
  end

  describe "stats/1" do
    test "returns filesystem statistics", %{agent_id: agent_id} do
      {:ok, pid} = FileSystemServer.start_link(agent_id: agent_id)

      # Empty filesystem
      {:ok, stats} = FileSystemServer.stats(pid)
      assert stats.total_files == 0
      assert stats.memory_files == 0
      assert stats.persisted_files == 0
      assert stats.loaded_files == 0
      assert stats.not_loaded_files == 0
      assert stats.dirty_files == 0
      assert stats.pending_persist == 0
      assert stats.total_size == 0
    end

    test "counts different file types correctly", %{agent_id: agent_id} do
      defmodule TestPersistence5 do
        @behaviour LangChain.Agents.FileSystem.Persistence

        def write_to_storage(_entry, _opts), do: :ok
        def load_from_storage(_entry, _opts), do: {:error, :enoent}
        def delete_from_storage(_entry, _opts), do: :ok
        def list_persisted_files(_agent_id, _opts), do: {:ok, []}
      end

      config = make_config(TestPersistence5, "Memories", debounce_ms: 5000)

      {:ok, pid} =
        FileSystemServer.start_link(
          agent_id: agent_id,
          persistence_configs: [config]
        )

      # Write memory files
      FileSystemServer.write_file(pid, "/scratch/file1.txt", "data1")
      FileSystemServer.write_file(pid, "/scratch/file2.txt", "data2")

      # Write persisted files
      FileSystemServer.write_file(pid, "/Memories/file3.txt", "data3")
      FileSystemServer.write_file(pid, "/Memories/file4.txt", "data4")

      {:ok, stats} = FileSystemServer.stats(pid)

      assert stats.total_files == 4
      assert stats.memory_files == 2
      assert stats.persisted_files == 2
      assert stats.loaded_files == 4
      assert stats.not_loaded_files == 0
      assert stats.dirty_files == 2
      assert stats.pending_persist == 2
      assert stats.total_size == byte_size("data1data2data3data4")
    end
  end

  describe "get_persistence_configs/1" do
    test "returns persistence configurations", %{agent_id: agent_id} do
      config = make_config(TestModule, "data", storage_opts: [path: "/test"])

      {:ok, pid} =
        FileSystemServer.start_link(
          agent_id: agent_id,
          persistence_configs: [config]
        )

      configs = FileSystemServer.get_persistence_configs(pid)

      assert map_size(configs) == 1
      assert %{"data" => loaded_config} = configs
      assert loaded_config.persistence_module == TestModule
      assert loaded_config.storage_opts[:path] == "/test"
      assert loaded_config.base_directory == "data"
    end

    test "returns empty map when no persistence configured", %{agent_id: agent_id} do
      {:ok, pid} = FileSystemServer.start_link(agent_id: agent_id)

      configs = FileSystemServer.get_persistence_configs(pid)

      assert configs == %{}
    end
  end

  describe "terminate/2" do
    test "flushes pending writes on termination", %{agent_id: agent_id} do
      defmodule TestPersistence6 do
        @behaviour LangChain.Agents.FileSystem.Persistence

        def write_to_storage(entry, _opts) do
          send(:test_process, {:flushed_on_terminate, entry.path})
          :ok
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
      FileSystemServer.write_file(pid, "/Memories/file1.txt", "data1")
      FileSystemServer.write_file(pid, "/Memories/file2.txt", "data2")

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

        def write_to_storage(_entry, _opts), do: :ok
        def load_from_storage(_entry, _opts), do: {:error, :enoent}
        def delete_from_storage(_entry, _opts), do: :ok
        def list_persisted_files(_agent_id, _opts), do: {:ok, []}
      end

      config = make_config(TestPersistence7, "persistent")

      {:ok, pid} =
        FileSystemServer.start_link(
          agent_id: agent_id,
          persistence_configs: [config]
        )

      table = FileSystemServer.get_table(pid)

      # Files under /persistent/ should be persisted
      FileSystemServer.write_file(pid, "/persistent/file.txt", "data")
      assert [{_, entry}] = :ets.lookup(table, "/persistent/file.txt")
      assert entry.persistence == :persisted

      # Files under /Memories/ should be memory-only with this config
      FileSystemServer.write_file(pid, "/Memories/file.txt", "data")
      assert [{_, entry2}] = :ets.lookup(table, "/Memories/file.txt")
      assert entry2.persistence == :memory
    end
  end

  describe "concurrent operations" do
    test "handles multiple writes to different files", %{agent_id: agent_id} do
      {:ok, pid} = FileSystemServer.start_link(agent_id: agent_id)
      table = FileSystemServer.get_table(pid)

      # Simulate concurrent writes
      tasks =
        for i <- 1..10 do
          Task.async(fn ->
            FileSystemServer.write_file(pid, "/file#{i}.txt", "data#{i}")
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
end
