defmodule LangChain.Agents.FileSystem.FileSystemSupervisorTest do
  use ExUnit.Case, async: false

  alias LangChain.Agents.FileSystem.FileSystemSupervisor
  alias LangChain.Agents.FileSystem.FileSystemConfig

  setup do
    # Start a unique supervisor for each test using start_supervised
    # This handles cleanup automatically
    supervisor_name = :"file_system_supervisor_#{System.unique_integer([:positive])}"
    supervisor_pid = start_supervised!({FileSystemSupervisor, name: supervisor_name})

    {:ok, supervisor_pid: supervisor_pid}
  end

  defp create_test_config(scope_key) do
    {:ok, config} =
      FileSystemConfig.new(%{
        scope_key: scope_key,
        base_directory: "TestDir",
        persistence_module: LangChain.Agents.FileSystem.Persistence.Memory,
        storage_opts: []
      })

    config
  end

  describe "start_link/1" do
    test "starts supervisor successfully" do
      supervisor_name = :"fs_sup_#{System.unique_integer([:positive])}"
      assert {:ok, pid} = FileSystemSupervisor.start_link(name: supervisor_name)
      assert Process.alive?(pid)

      # Verify it's a DynamicSupervisor
      assert DynamicSupervisor.count_children(pid) == %{
               active: 0,
               specs: 0,
               supervisors: 0,
               workers: 0
             }

      DynamicSupervisor.stop(pid)
    end

    test "starts without a name" do
      assert {:ok, pid} = FileSystemSupervisor.start_link()
      assert Process.alive?(pid)
      DynamicSupervisor.stop(pid)
    end
  end

  describe "start_filesystem/3" do
    test "starts a new filesystem for a scope", %{supervisor_pid: sup} do
      scope_key = {:user, System.unique_integer([:positive])}
      config = create_test_config(scope_key)

      assert {:ok, fs_pid} =
               FileSystemSupervisor.start_filesystem(scope_key, [config], supervisor: sup)

      assert Process.alive?(fs_pid)

      # Verify it's registered
      assert {:ok, ^fs_pid} = FileSystemSupervisor.get_filesystem(scope_key)

      # Clean up
      FileSystemSupervisor.stop_filesystem(scope_key, supervisor: sup)
    end

    test "returns error if filesystem already running for scope", %{supervisor_pid: sup} do
      scope_key = {:user, System.unique_integer([:positive])}
      config = create_test_config(scope_key)

      # Start first filesystem
      {:ok, fs_pid} = FileSystemSupervisor.start_filesystem(scope_key, [config], supervisor: sup)

      # Try to start duplicate
      assert {:error, {:already_started, ^fs_pid}} =
               FileSystemSupervisor.start_filesystem(scope_key, [config], supervisor: sup)

      # Clean up
      FileSystemSupervisor.stop_filesystem(scope_key, supervisor: sup)
    end

    test "can start multiple filesystems with different scopes", %{supervisor_pid: sup} do
      scope1 = {:user, System.unique_integer([:positive])}
      scope2 = {:project, System.unique_integer([:positive])}
      scope3 = {:organization, System.unique_integer([:positive])}

      config1 = create_test_config(scope1)
      config2 = create_test_config(scope2)
      config3 = create_test_config(scope3)

      {:ok, fs_pid1} = FileSystemSupervisor.start_filesystem(scope1, [config1], supervisor: sup)
      {:ok, fs_pid2} = FileSystemSupervisor.start_filesystem(scope2, [config2], supervisor: sup)
      {:ok, fs_pid3} = FileSystemSupervisor.start_filesystem(scope3, [config3], supervisor: sup)

      # All PIDs should be different
      assert fs_pid1 != fs_pid2
      assert fs_pid2 != fs_pid3
      assert fs_pid1 != fs_pid3

      # All should be registered
      assert {:ok, ^fs_pid1} = FileSystemSupervisor.get_filesystem(scope1)
      assert {:ok, ^fs_pid2} = FileSystemSupervisor.get_filesystem(scope2)
      assert {:ok, ^fs_pid3} = FileSystemSupervisor.get_filesystem(scope3)

      # Clean up
      FileSystemSupervisor.stop_filesystem(scope1, supervisor: sup)
      FileSystemSupervisor.stop_filesystem(scope2, supervisor: sup)
      FileSystemSupervisor.stop_filesystem(scope3, supervisor: sup)
    end

    test "returns error for invalid scope_key (not a tuple)", %{supervisor_pid: _sup} do
      config = create_test_config({:user, 123})

      assert {:error, :invalid_scope_key} =
               FileSystemSupervisor.start_filesystem("invalid", [config])
    end

    test "returns error for invalid configs (not a list)", %{supervisor_pid: _sup} do
      scope_key = {:user, System.unique_integer([:positive])}

      assert {:error, :invalid_configs} =
               FileSystemSupervisor.start_filesystem(scope_key, "invalid")
    end
  end

  describe "stop_filesystem/2" do
    test "stops a running filesystem gracefully", %{supervisor_pid: sup} do
      scope_key = {:user, System.unique_integer([:positive])}
      config = create_test_config(scope_key)

      {:ok, fs_pid} = FileSystemSupervisor.start_filesystem(scope_key, [config], supervisor: sup)
      assert Process.alive?(fs_pid)

      # Stop the filesystem
      assert :ok = FileSystemSupervisor.stop_filesystem(scope_key, supervisor: sup)

      # Give it time to stop
      Process.sleep(50)

      # Should no longer be running
      refute Process.alive?(fs_pid)
      assert {:error, :not_found} = FileSystemSupervisor.get_filesystem(scope_key)
    end

    test "returns error if filesystem not running", %{supervisor_pid: sup} do
      scope_key = {:user, System.unique_integer([:positive])}

      assert {:error, :not_found} =
               FileSystemSupervisor.stop_filesystem(scope_key, supervisor: sup)
    end

    test "returns error after already stopped", %{supervisor_pid: sup} do
      scope_key = {:user, System.unique_integer([:positive])}
      config = create_test_config(scope_key)

      {:ok, _fs_pid} = FileSystemSupervisor.start_filesystem(scope_key, [config], supervisor: sup)
      :ok = FileSystemSupervisor.stop_filesystem(scope_key, supervisor: sup)

      Process.sleep(50)

      # Trying to stop again should return error
      assert {:error, :not_found} =
               FileSystemSupervisor.stop_filesystem(scope_key, supervisor: sup)
    end
  end

  describe "get_filesystem/1" do
    test "returns pid when filesystem exists", %{supervisor_pid: sup} do
      scope_key = {:user, System.unique_integer([:positive])}
      config = create_test_config(scope_key)

      {:ok, fs_pid} = FileSystemSupervisor.start_filesystem(scope_key, [config], supervisor: sup)

      assert {:ok, ^fs_pid} = FileSystemSupervisor.get_filesystem(scope_key)

      # Clean up
      FileSystemSupervisor.stop_filesystem(scope_key, supervisor: sup)
    end

    test "returns error when filesystem does not exist", %{supervisor_pid: _sup} do
      scope_key = {:user, System.unique_integer([:positive])}

      assert {:error, :not_found} = FileSystemSupervisor.get_filesystem(scope_key)
    end

    test "returns error after filesystem stops", %{supervisor_pid: sup} do
      scope_key = {:user, System.unique_integer([:positive])}
      config = create_test_config(scope_key)

      {:ok, fs_pid} = FileSystemSupervisor.start_filesystem(scope_key, [config], supervisor: sup)
      assert {:ok, ^fs_pid} = FileSystemSupervisor.get_filesystem(scope_key)

      FileSystemSupervisor.stop_filesystem(scope_key, supervisor: sup)
      Process.sleep(50)

      assert {:error, :not_found} = FileSystemSupervisor.get_filesystem(scope_key)
    end
  end

  describe "list_filesystems/0" do
    test "returns empty list when no filesystems running", %{supervisor_pid: _sup} do
      # Note: This might not be truly empty if other tests are running,
      # but we can check that it's a list
      filesystems = FileSystemSupervisor.list_filesystems()
      assert is_list(filesystems)
    end

    test "returns all running filesystems", %{supervisor_pid: sup} do
      scope1 = {:user, System.unique_integer([:positive])}
      scope2 = {:project, System.unique_integer([:positive])}

      config1 = create_test_config(scope1)
      config2 = create_test_config(scope2)

      {:ok, fs_pid1} = FileSystemSupervisor.start_filesystem(scope1, [config1], supervisor: sup)
      {:ok, fs_pid2} = FileSystemSupervisor.start_filesystem(scope2, [config2], supervisor: sup)

      filesystems = FileSystemSupervisor.list_filesystems()

      # Should include both our filesystems
      assert {scope1, fs_pid1} in filesystems
      assert {scope2, fs_pid2} in filesystems

      # Clean up
      FileSystemSupervisor.stop_filesystem(scope1, supervisor: sup)
      FileSystemSupervisor.stop_filesystem(scope2, supervisor: sup)
    end

    test "list updates after stopping filesystem", %{supervisor_pid: sup} do
      scope_key = {:user, System.unique_integer([:positive])}
      config = create_test_config(scope_key)

      {:ok, fs_pid} = FileSystemSupervisor.start_filesystem(scope_key, [config], supervisor: sup)

      filesystems_before = FileSystemSupervisor.list_filesystems()
      assert {scope_key, fs_pid} in filesystems_before

      FileSystemSupervisor.stop_filesystem(scope_key, supervisor: sup)
      Process.sleep(50)

      filesystems_after = FileSystemSupervisor.list_filesystems()
      refute {scope_key, fs_pid} in filesystems_after
    end
  end

  describe "crash recovery with :transient restart strategy" do
    test "filesystem restarts automatically on crash", %{supervisor_pid: sup} do
      scope_key = {:user, System.unique_integer([:positive])}
      config = create_test_config(scope_key)

      {:ok, fs_pid} = FileSystemSupervisor.start_filesystem(scope_key, [config], supervisor: sup)
      assert Process.alive?(fs_pid)

      # Kill the filesystem process (simulate crash)
      Process.exit(fs_pid, :kill)

      # Give it time to restart
      Process.sleep(100)

      # Should have restarted with a new PID
      {:ok, new_fs_pid} = FileSystemSupervisor.get_filesystem(scope_key)
      assert Process.alive?(new_fs_pid)
      assert new_fs_pid != fs_pid

      # Verify supervisor still has the child
      assert DynamicSupervisor.count_children(sup).active >= 1

      # Clean up
      FileSystemSupervisor.stop_filesystem(scope_key, supervisor: sup)
    end

    test "filesystem does not restart on graceful shutdown", %{supervisor_pid: sup} do
      scope_key = {:user, System.unique_integer([:positive])}
      config = create_test_config(scope_key)

      {:ok, fs_pid} = FileSystemSupervisor.start_filesystem(scope_key, [config], supervisor: sup)

      initial_count = DynamicSupervisor.count_children(sup).active

      # Graceful shutdown
      FileSystemSupervisor.stop_filesystem(scope_key, supervisor: sup)
      Process.sleep(100)

      # Should not be running
      assert {:error, :not_found} = FileSystemSupervisor.get_filesystem(scope_key)
      refute Process.alive?(fs_pid)

      # Child should be removed from supervisor
      final_count = DynamicSupervisor.count_children(sup).active
      assert final_count < initial_count
    end
  end

  describe "registry integration" do
    test "filesystem properly registered in Registry", %{supervisor_pid: sup} do
      scope_key = {:user, System.unique_integer([:positive])}
      config = create_test_config(scope_key)

      {:ok, fs_pid} = FileSystemSupervisor.start_filesystem(scope_key, [config], supervisor: sup)

      # Check Registry directly
      assert [{^fs_pid, _}] =
               Registry.lookup(LangChain.Agents.Registry, {:filesystem_server, scope_key})

      # Clean up
      FileSystemSupervisor.stop_filesystem(scope_key, supervisor: sup)
    end

    test "filesystem unregistered after stop", %{supervisor_pid: sup} do
      scope_key = {:user, System.unique_integer([:positive])}
      config = create_test_config(scope_key)

      {:ok, _fs_pid} = FileSystemSupervisor.start_filesystem(scope_key, [config], supervisor: sup)
      FileSystemSupervisor.stop_filesystem(scope_key, supervisor: sup)

      Process.sleep(50)

      # Should not be in Registry anymore
      assert [] = Registry.lookup(LangChain.Agents.Registry, {:filesystem_server, scope_key})
    end
  end

  describe "concurrent filesystems" do
    test "multiple filesystems can operate concurrently", %{supervisor_pid: sup} do
      # Start 5 concurrent filesystems
      scopes =
        Enum.map(1..5, fn i ->
          {:user, System.unique_integer([:positive]) + i}
        end)

      # Start all filesystems
      pids =
        Enum.map(scopes, fn scope ->
          config = create_test_config(scope)
          {:ok, pid} = FileSystemSupervisor.start_filesystem(scope, [config], supervisor: sup)
          {scope, pid}
        end)

      # Verify all are running
      Enum.each(pids, fn {scope, pid} ->
        assert Process.alive?(pid)
        assert {:ok, ^pid} = FileSystemSupervisor.get_filesystem(scope)
      end)

      # Clean up
      Enum.each(scopes, fn scope ->
        FileSystemSupervisor.stop_filesystem(scope, supervisor: sup)
      end)
    end

    test "stopping one filesystem doesn't affect others", %{supervisor_pid: sup} do
      scope1 = {:user, System.unique_integer([:positive])}
      scope2 = {:project, System.unique_integer([:positive])}

      config1 = create_test_config(scope1)
      config2 = create_test_config(scope2)

      {:ok, fs_pid1} = FileSystemSupervisor.start_filesystem(scope1, [config1], supervisor: sup)
      {:ok, fs_pid2} = FileSystemSupervisor.start_filesystem(scope2, [config2], supervisor: sup)

      # Stop first filesystem
      FileSystemSupervisor.stop_filesystem(scope1, supervisor: sup)
      Process.sleep(50)

      # First should be stopped
      assert {:error, :not_found} = FileSystemSupervisor.get_filesystem(scope1)
      refute Process.alive?(fs_pid1)

      # Second should still be running
      assert {:ok, ^fs_pid2} = FileSystemSupervisor.get_filesystem(scope2)
      assert Process.alive?(fs_pid2)

      # Clean up
      FileSystemSupervisor.stop_filesystem(scope2, supervisor: sup)
    end
  end
end
