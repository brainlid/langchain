defmodule LangChain.Agents.FileSystem.MultiPersistenceTest do
  use ExUnit.Case, async: false

  alias LangChain.Agents.FileSystemServer
  alias LangChain.Agents.FileSystem.{FileSystemConfig, Persistence}

  @moduletag :tmp_dir

  setup %{tmp_dir: tmp_dir} do
    agent_id = "test_agent_#{System.unique_integer([:positive])}"

    # Start registry for this test
    case start_supervised({Registry, keys: :unique, name: LangChain.Agents.Registry}) do
      {:ok, _pid} -> :ok
      {:error, {:already_started, _pid}} -> :ok
    end

    on_exit(fn ->
      # Cleanup any running FileSystemServer
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

    %{agent_id: agent_id, tmp_dir: tmp_dir}
  end

  describe "multiple persistence backends" do
    test "can register multiple persistence configs for different directories", %{
      agent_id: agent_id,
      tmp_dir: tmp_dir
    } do
      # Start with memory-only filesystem
      {:ok, pid} = FileSystemServer.start_link(agent_id: agent_id)
      table = FileSystemServer.get_table(pid)

      # Verify no persistence configs initially
      assert FileSystemServer.get_persistence_configs(pid) == %{}

      # Register first persistence backend for "user_files"
      {:ok, user_config} =
        FileSystemConfig.new(%{
          base_directory: "user_files",
          persistence_module: Persistence.Disk,
          debounce_ms: 100,
          storage_opts: [path: Path.join(tmp_dir, "users")]
        })

      assert :ok = FileSystemServer.register_persistence(pid, user_config)

      # Register second persistence backend for "account_files" (read-only)
      {:ok, account_config} =
        FileSystemConfig.new(%{
          base_directory: "account_files",
          persistence_module: Persistence.Disk,
          readonly: true,
          debounce_ms: 200,
          storage_opts: [path: Path.join(tmp_dir, "accounts")]
        })

      assert :ok = FileSystemServer.register_persistence(pid, account_config)

      # Verify both configs are registered
      configs = FileSystemServer.get_persistence_configs(pid)
      assert map_size(configs) == 2
      assert Map.has_key?(configs, "user_files")
      assert Map.has_key?(configs, "account_files")

      # Write to user_files (should persist)
      assert :ok = FileSystemServer.write_file(pid, "/user_files/data.txt", "user data")

      assert [{_, entry}] = :ets.lookup(table, "/user_files/data.txt")
      assert entry.persistence == :persisted
      assert entry.dirty == true

      # Wait for debounce
      Process.sleep(150)

      assert [{_, clean_entry}] = :ets.lookup(table, "/user_files/data.txt")
      assert clean_entry.dirty == false

      # Verify file exists on disk
      user_path = Path.join([tmp_dir, "users", agent_id, "user_files", "data.txt"])
      assert File.exists?(user_path)
      assert File.read!(user_path) == "user data"

      # Try to write to account_files (read-only, should fail)
      assert {:error, reason} =
               FileSystemServer.write_file(pid, "/account_files/data.txt", "account data")

      assert reason =~ "read-only"

      # Write to memory-only location
      assert :ok = FileSystemServer.write_file(pid, "/scratch/temp.txt", "temp data")

      assert [{_, temp_entry}] = :ets.lookup(table, "/scratch/temp.txt")
      assert temp_entry.persistence == :memory
    end

    test "prevents registering same base_directory twice", %{agent_id: agent_id} do
      {:ok, pid} = FileSystemServer.start_link(agent_id: agent_id)

      {:ok, config1} =
        FileSystemConfig.new(%{
          base_directory: "user_files",
          persistence_module: Persistence.Disk
        })

      assert :ok = FileSystemServer.register_persistence(pid, config1)

      # Try to register same base_directory again
      {:ok, config2} =
        FileSystemConfig.new(%{
          base_directory: "user_files",
          persistence_module: Persistence.Disk,
          debounce_ms: 10000
        })

      assert {:error, reason} = FileSystemServer.register_persistence(pid, config2)
      assert reason =~ "already has a registered persistence config"
    end

    test "different configs can have different debounce times", %{
      agent_id: agent_id,
      tmp_dir: tmp_dir
    } do
      test_pid = self()

      # Create a tracking persistence module
      defmodule TrackingPersistence do
        @behaviour Persistence

        def write_to_storage(entry, opts) do
          case Keyword.get(opts, :test_pid) do
            nil -> :ok
            test_pid -> send(test_pid, {:persisted, entry.path, System.monotonic_time()})
          end

          # Actually persist using Disk
          Persistence.Disk.write_to_storage(entry, opts)
        end

        def load_from_storage(entry, opts), do: Persistence.Disk.load_from_storage(entry, opts)

        def delete_from_storage(entry, opts),
          do: Persistence.Disk.delete_from_storage(entry, opts)

        def list_persisted_files(agent_id, opts),
          do: Persistence.Disk.list_persisted_files(agent_id, opts)
      end

      {:ok, pid} = FileSystemServer.start_link(agent_id: agent_id)

      # Register fast config (100ms debounce)
      {:ok, fast_config} =
        FileSystemConfig.new(%{
          base_directory: "fast",
          persistence_module: TrackingPersistence,
          debounce_ms: 100,
          storage_opts: [path: tmp_dir, test_pid: test_pid]
        })

      assert :ok = FileSystemServer.register_persistence(pid, fast_config)

      # Register slow config (500ms debounce)
      {:ok, slow_config} =
        FileSystemConfig.new(%{
          base_directory: "slow",
          persistence_module: TrackingPersistence,
          debounce_ms: 500,
          storage_opts: [path: tmp_dir, test_pid: test_pid]
        })

      assert :ok = FileSystemServer.register_persistence(pid, slow_config)

      # Write to both directories at the same time
      start_time = System.monotonic_time()
      FileSystemServer.write_file(pid, "/fast/file.txt", "fast data")
      FileSystemServer.write_file(pid, "/slow/file.txt", "slow data")

      # Wait for fast persistence
      assert_receive {:persisted, "/fast/file.txt", fast_time}, 200
      fast_elapsed = System.convert_time_unit(fast_time - start_time, :native, :millisecond)

      # Fast should persist around 100ms
      assert fast_elapsed >= 90 and fast_elapsed < 200

      # Wait for slow persistence
      assert_receive {:persisted, "/slow/file.txt", slow_time}, 600
      slow_elapsed = System.convert_time_unit(slow_time - start_time, :native, :millisecond)

      # Slow should persist around 500ms
      assert slow_elapsed >= 450 and slow_elapsed < 700
    end

    test "readonly config prevents deletes", %{agent_id: agent_id, tmp_dir: tmp_dir} do
      {:ok, pid} = FileSystemServer.start_link(agent_id: agent_id)

      {:ok, config} =
        FileSystemConfig.new(%{
          base_directory: "readonly",
          persistence_module: Persistence.Disk,
          readonly: true,
          storage_opts: [path: tmp_dir]
        })

      assert :ok = FileSystemServer.register_persistence(pid, config)

      # Try to delete from readonly directory (should fail)
      assert {:error, reason} = FileSystemServer.delete_file(pid, "/readonly/file.txt")
      assert reason =~ "read-only"
    end
  end
end
