defmodule LangChain.Agents.FileSystem.PersistenceIntegrationTest do
  use ExUnit.Case, async: false

  alias LangChain.Agents.FileSystemServer
  alias LangChain.Agents.FileSystem.Persistence.Disk
  alias LangChain.Agents.FileSystem.FileSystemConfig

  # Helper to get file entry from GenServer state
  defp get_entry(agent_id, path) do
    pid = FileSystemServer.whereis(agent_id)
    state = :sys.get_state(pid)
    Map.get(state.files, path)
  end

  @moduletag :tmp_dir

  setup %{tmp_dir: tmp_dir} do
    agent_id = "test_agent_#{System.unique_integer([:positive])}"

    # Start registry for this test if not already started
    # Using start_supervised ensures proper cleanup
    case start_supervised({Registry, keys: :unique, name: LangChain.Agents.Registry}) do
      {:ok, _pid} -> :ok
      {:error, {:already_started, _pid}} -> :ok
    end

    on_exit(fn ->
      # Cleanup any running FileSystemServer
      # Wrap in try-catch because Registry might be gone already during cleanup
      try do
        via_tuple = {:via, Registry, {LangChain.Agents.Registry, {:file_system_server, agent_id}}}

        case GenServer.whereis(via_tuple) do
          nil -> :ok
          pid when is_pid(pid) -> GenServer.stop(pid, :normal)
        end
      rescue
        ArgumentError -> :ok
      catch
        :exit, _ -> :ok
      end
    end)

    %{agent_id: agent_id, tmp_dir: tmp_dir}
  end

  # Helper to create persistence config for tests
  defp make_config(module, base_dir, tmp_dir, opts \\ []) do
    debounce_ms = Keyword.get(opts, :debounce_ms, 100)
    storage_opts = [path: tmp_dir] ++ Keyword.get(opts, :storage_opts, [])

    {:ok, config} =
      FileSystemConfig.new(%{
        base_directory: base_dir,
        persistence_module: module,
        debounce_ms: debounce_ms,
        storage_opts: storage_opts
      })

    config
  end

  describe "full persistence workflow" do
    test "writes persist to disk after debounce", %{agent_id: agent_id, tmp_dir: tmp_dir} do
      config = make_config(Disk, "Memories", tmp_dir)

      {:ok, _pid} =
        FileSystemServer.start_link(
          agent_id: agent_id,
          persistence_configs: [config]
        )


      # Write a file to persisted directory
      path = "/Memories/test.txt"
      content = "test content"
      assert :ok = FileSystemServer.write_file(agent_id, path, content)

      # File should be in ETS immediately
      entry = get_entry(agent_id, path)
      assert entry.content == content
      assert entry.persistence == :persisted
      assert entry.dirty == true

      # Wait for debounce
      Process.sleep(150)

      # File should now be clean
      clean_entry = get_entry(agent_id, path)
      assert clean_entry.dirty == false

      # Verify file exists on disk
      disk_path = Path.join(tmp_dir, "test.txt")
      assert File.exists?(disk_path)
      assert File.read!(disk_path) == content
    end

    test "rapid writes batch into single persist", %{agent_id: agent_id, tmp_dir: tmp_dir} do
      # Track persistence calls
      test_pid = self()

      defmodule TrackingDisk do
        @behaviour LangChain.Agents.FileSystem.Persistence

        def write_to_storage(entry, opts) do
          send(:test_process, {:persisted, entry.path, System.monotonic_time()})
          LangChain.Agents.FileSystem.Persistence.Disk.write_to_storage(entry, opts)
        end

        def load_from_storage(entry, opts) do
          LangChain.Agents.FileSystem.Persistence.Disk.load_from_storage(entry, opts)
        end

        def delete_from_storage(entry, opts) do
          LangChain.Agents.FileSystem.Persistence.Disk.delete_from_storage(entry, opts)
        end

        def list_persisted_files(agent_id, opts) do
          LangChain.Agents.FileSystem.Persistence.Disk.list_persisted_files(agent_id, opts)
        end
      end

      Process.register(test_pid, :test_process)

      config = make_config(TrackingDisk, "Memories", tmp_dir)

      {:ok, _pid} =
        FileSystemServer.start_link(
          agent_id: agent_id,
          persistence_configs: [config]
        )

      path = "/Memories/rapid.txt"

      # Write 10 times rapidly
      for i <- 1..10 do
        FileSystemServer.write_file(agent_id, path, "version #{i}")
        Process.sleep(10)
      end

      # Wait for debounce
      Process.sleep(150)

      # Should only have persisted once
      assert_received {:persisted, ^path, _time}
      refute_received {:persisted, ^path, _time}
    end

    test "memory files are not persisted", %{agent_id: agent_id, tmp_dir: tmp_dir} do
      config = make_config(Disk, "Memories", tmp_dir)

      {:ok, _pid} =
        FileSystemServer.start_link(
          agent_id: agent_id,
          persistence_configs: [config]
        )


      # Write to non-persisted directory
      path = "/scratch/temp.txt"
      content = "temporary content"
      assert :ok = FileSystemServer.write_file(agent_id, path, content)

      # File should be in ETS
      entry = get_entry(agent_id, path)
      assert entry.persistence == :memory
      assert entry.dirty == false

      # Wait longer than debounce
      Process.sleep(150)

      # File should NOT exist on disk
      disk_path = Path.join(tmp_dir, "temp.txt")
      refute File.exists?(disk_path)
    end

    test "flush_all persists immediately", %{agent_id: agent_id, tmp_dir: tmp_dir} do
      config = make_config(Disk, "Memories", tmp_dir, debounce_ms: 10000)

      {:ok, _pid} =
        FileSystemServer.start_link(
          agent_id: agent_id,
          persistence_configs: [config]
        )

      # Write files
      FileSystemServer.write_file(agent_id, "/Memories/file1.txt", "content1")
      FileSystemServer.write_file(agent_id, "/Memories/file2.txt", "content2")
      FileSystemServer.write_file(agent_id, "/Memories/file3.txt", "content3")

      # Verify they're dirty
      {:ok, stats_before} = FileSystemServer.stats(agent_id)
      assert stats_before.dirty_files == 3

      # Flush all
      assert :ok = FileSystemServer.flush_all(agent_id)

      # Give time to process
      Process.sleep(100)

      # All should be clean now
      {:ok, stats_after} = FileSystemServer.stats(agent_id)
      assert stats_after.dirty_files == 0

      # All files should exist on disk
      for i <- 1..3 do
        disk_path = Path.join(tmp_dir, "file#{i}.txt")
        assert File.exists?(disk_path)
        assert File.read!(disk_path) == "content#{i}"
      end
    end

    test "termination flushes pending writes", %{agent_id: agent_id, tmp_dir: tmp_dir} do
      config = make_config(Disk, "Memories", tmp_dir, debounce_ms: 10000)

      {:ok, pid} =
        FileSystemServer.start_link(
          agent_id: agent_id,
          persistence_configs: [config]
        )

      # Write files with long debounce
      FileSystemServer.write_file(agent_id, "/Memories/file1.txt", "data1")
      FileSystemServer.write_file(agent_id, "/Memories/file2.txt", "data2")

      # Stop the server immediately (should flush on terminate)
      GenServer.stop(pid, :normal)

      # Files should be on disk
      disk_path1 = Path.join(tmp_dir, "file1.txt")
      disk_path2 = Path.join(tmp_dir, "file2.txt")

      assert File.exists?(disk_path1)
      assert File.exists?(disk_path2)
      assert File.read!(disk_path1) == "data1"
      assert File.read!(disk_path2) == "data2"
    end

    test "delete removes file from disk immediately", %{agent_id: agent_id, tmp_dir: tmp_dir} do
      config = make_config(Disk, "Memories", tmp_dir)

      {:ok, _pid} =
        FileSystemServer.start_link(
          agent_id: agent_id,
          persistence_configs: [config]
        )


      # Write and persist file
      path = "/Memories/delete_me.txt"
      FileSystemServer.write_file(agent_id, path, "content")

      # Wait for persist
      Process.sleep(150)

      disk_path = Path.join(tmp_dir, "delete_me.txt")
      assert File.exists?(disk_path)

      # Delete file
      assert :ok = FileSystemServer.delete_file(agent_id, path)

      # Should be gone from ETS
      assert !FileSystemServer.file_exists?(agent_id, path)

      # Should be gone from disk (no debounce on delete)
      refute File.exists?(disk_path)
    end
  end

  describe "lazy loading" do
    test "indexes persisted files on startup without loading content", %{
      agent_id: agent_id,
      tmp_dir: tmp_dir
    } do
      # First: create server, write files, persist, then stop
      config = make_config(Disk, "Memories", tmp_dir)

      {:ok, pid1} =
        FileSystemServer.start_link(
          agent_id: agent_id,
          persistence_configs: [config]
        )

      FileSystemServer.write_file(agent_id, "/Memories/file1.txt", "content1")
      FileSystemServer.write_file(agent_id, "/Memories/file2.txt", "content2")

      Process.sleep(150)
      GenServer.stop(pid1, :normal)

      # Second: start new server (should index files)
      # Note: Lazy loading indexing would need to be implemented in FileSystemState.new/1
      # For now, this tests the disk backend's list_persisted_files capability
      {:ok, files} =
        Disk.list_persisted_files(agent_id, path: tmp_dir, base_directory: "Memories")

      assert length(files) == 2
      assert "/Memories/file1.txt" in files
      assert "/Memories/file2.txt" in files
    end
  end

  describe "custom memories directory" do
    test "uses custom memories_directory for persistence", %{
      agent_id: agent_id,
      tmp_dir: tmp_dir
    } do
      config = make_config(Disk, "persistent", tmp_dir)

      {:ok, _pid} =
        FileSystemServer.start_link(
          agent_id: agent_id,
          persistence_configs: [config]
        )


      # Files under /persistent/ should be persisted
      FileSystemServer.write_file(agent_id, "/persistent/data.txt", "persisted")

      entry = get_entry(agent_id, "/persistent/data.txt")
      assert entry.persistence == :persisted

      Process.sleep(150)

      disk_path = Path.join(tmp_dir, "data.txt")
      assert File.exists?(disk_path)
      assert File.read!(disk_path) == "persisted"

      # Files under /Memories/ should NOT be persisted with this config
      FileSystemServer.write_file(agent_id, "/Memories/temp.txt", "not persisted")

      entry2 = get_entry(agent_id, "/Memories/temp.txt")
      assert entry2.persistence == :memory

      Process.sleep(150)

      disk_path2 = Path.join(tmp_dir, "temp.txt")
      refute File.exists?(disk_path2)
    end
  end

  describe "nested directory persistence" do
    test "handles deeply nested file paths", %{agent_id: agent_id, tmp_dir: tmp_dir} do
      config = make_config(Disk, "Memories", tmp_dir)

      {:ok, _pid} =
        FileSystemServer.start_link(
          agent_id: agent_id,
          persistence_configs: [config]
        )

      path = "/Memories/year/2024/month/10/day/30/log.txt"
      content = "deep nested content"

      FileSystemServer.write_file(agent_id, path, content)

      Process.sleep(150)

      disk_path = Path.join([tmp_dir, "year", "2024", "month", "10", "day", "30", "log.txt"])
      assert File.exists?(disk_path)
      assert File.read!(disk_path) == content
    end

    test "lists files from nested directories", %{agent_id: agent_id, tmp_dir: tmp_dir} do
      config = make_config(Disk, "Memories", tmp_dir)

      {:ok, _pid} =
        FileSystemServer.start_link(
          agent_id: agent_id,
          persistence_configs: [config]
        )

      files = [
        "/Memories/root.txt",
        "/Memories/dir1/file1.txt",
        "/Memories/dir1/subdir/file2.txt",
        "/Memories/dir2/file3.txt"
      ]

      for path <- files do
        FileSystemServer.write_file(agent_id, path, "content")
      end

      Process.sleep(150)

      {:ok, listed_files} =
        Disk.list_persisted_files(agent_id, path: tmp_dir, base_directory: "Memories")

      assert length(listed_files) == 4

      for path <- files do
        assert path in listed_files
      end
    end
  end

  describe "concurrent SubAgent simulation" do
    test "multiple writers (simulating SubAgents) don't lose data", %{
      agent_id: agent_id,
      tmp_dir: tmp_dir
    } do
      config = make_config(Disk, "Memories", tmp_dir)

      {:ok, _pid} =
        FileSystemServer.start_link(
          agent_id: agent_id,
          persistence_configs: [config]
        )

      # Simulate 10 SubAgents writing concurrently
      tasks =
        for i <- 1..10 do
          Task.async(fn ->
            FileSystemServer.write_file(agent_id, "/Memories/subagent_#{i}.txt", "result #{i}")
          end)
        end

      results = Task.await_many(tasks)
      assert Enum.all?(results, &(&1 == :ok))

      # Wait for all debounces
      Process.sleep(200)

      # All files should be on disk
      {:ok, files} =
        Disk.list_persisted_files(agent_id, path: tmp_dir, base_directory: "Memories")

      assert length(files) == 10

      for i <- 1..10 do
        disk_path = Path.join(tmp_dir, "subagent_#{i}.txt")
        assert File.exists?(disk_path)
        assert File.read!(disk_path) == "result #{i}"
      end
    end
  end

  describe "stats integration" do
    test "stats reflect persistence state correctly", %{agent_id: agent_id, tmp_dir: tmp_dir} do
      config = make_config(Disk, "Memories", tmp_dir)

      {:ok, _pid} =
        FileSystemServer.start_link(
          agent_id: agent_id,
          persistence_configs: [config]
        )

      # Initial state
      {:ok, stats} = FileSystemServer.stats(agent_id)
      assert stats.total_files == 0
      assert stats.dirty_files == 0

      # Write files
      FileSystemServer.write_file(agent_id, "/scratch/temp.txt", "temp")
      FileSystemServer.write_file(agent_id, "/Memories/persist1.txt", "data1")
      FileSystemServer.write_file(agent_id, "/Memories/persist2.txt", "data2")

      # Check stats before persist
      {:ok, stats_before} = FileSystemServer.stats(agent_id)
      assert stats_before.total_files == 3
      assert stats_before.memory_files == 1
      assert stats_before.persisted_files == 2
      assert stats_before.dirty_files == 2

      # Wait for persist
      Process.sleep(150)

      # Check stats after persist
      {:ok, stats_after} = FileSystemServer.stats(agent_id)
      assert stats_after.total_files == 3
      assert stats_after.dirty_files == 0
    end
  end

  describe "error handling" do
    test "continues operation after persist failure", %{agent_id: agent_id, tmp_dir: tmp_dir} do
      defmodule FailingPersistence do
        @behaviour LangChain.Agents.FileSystem.Persistence

        def write_to_storage(_entry, _opts) do
          {:error, :disk_full}
        end

        def load_from_storage(_entry, _opts), do: {:error, :enoent}
        def delete_from_storage(_entry, _opts), do: :ok
        def list_persisted_files(_agent_id, _opts), do: {:ok, []}
      end

      config = make_config(FailingPersistence, "Memories", tmp_dir)

      {:ok, pid} =
        FileSystemServer.start_link(
          agent_id: agent_id,
          persistence_configs: [config]
        )


      # Write should succeed (ETS write)
      assert :ok = FileSystemServer.write_file(agent_id, "/Memories/file.txt", "content")

      # File should be in ETS
      entry = get_entry(agent_id, "/Memories/file.txt")
      assert entry.content == "content"

      # Wait for persist attempt (should fail but not crash)
      Process.sleep(150)

      # Server should still be alive
      assert Process.alive?(pid)

      # File should still be dirty (persist failed)
      entry_after = get_entry(agent_id, "/Memories/file.txt")
      assert entry_after.dirty == true
    end
  end
end
