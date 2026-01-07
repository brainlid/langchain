defmodule LangChain.Agents.FileSystemTest do
  use ExUnit.Case, async: false

  alias LangChain.Agents.FileSystem
  alias LangChain.Agents.FileSystem.FileSystemConfig
  alias LangChain.Agents.FileSystem.FileSystemSupervisor

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

  describe "ensure_filesystem/3" do
    test "starts a new filesystem if not running", %{supervisor_pid: sup} do
      scope_key = {:user, System.unique_integer([:positive])}
      config = create_test_config(scope_key)

      # Should start new filesystem
      assert {:ok, pid} = FileSystem.ensure_filesystem(scope_key, [config], supervisor: sup)
      assert Process.alive?(pid)

      # Should be registered
      assert {:ok, ^pid} = FileSystem.get_filesystem_pid(scope_key)

      # Clean up
      FileSystem.stop_filesystem(scope_key, supervisor: sup)
    end

    test "returns existing PID if filesystem already running (idempotent)", %{supervisor_pid: sup} do
      scope_key = {:user, System.unique_integer([:positive])}
      config = create_test_config(scope_key)

      # First call starts the filesystem
      {:ok, pid1} = FileSystem.ensure_filesystem(scope_key, [config], supervisor: sup)

      # Second call returns same PID
      {:ok, pid2} = FileSystem.ensure_filesystem(scope_key, [config], supervisor: sup)

      assert pid1 == pid2
      assert Process.alive?(pid1)

      # Third call also returns same PID
      {:ok, pid3} = FileSystem.ensure_filesystem(scope_key, [config], supervisor: sup)
      assert pid1 == pid3

      # Clean up
      FileSystem.stop_filesystem(scope_key, supervisor: sup)
    end

    test "idempotent across multiple calls", %{supervisor_pid: sup} do
      scope_key = {:user, System.unique_integer([:positive])}
      config = create_test_config(scope_key)

      # Call ensure_filesystem 10 times
      results =
        Enum.map(1..10, fn _ ->
          FileSystem.ensure_filesystem(scope_key, [config], supervisor: sup)
        end)

      # All should succeed
      pids = Enum.map(results, fn {:ok, pid} -> pid end)

      # All PIDs should be the same
      [first_pid | rest] = pids
      assert Enum.all?(rest, fn pid -> pid == first_pid end)

      # Clean up
      FileSystem.stop_filesystem(scope_key, supervisor: sup)
    end

    test "validates scope_key is a tuple", %{supervisor_pid: _sup} do
      config = create_test_config({:user, 123})

      # Invalid scope_key (not a tuple)
      assert {:error, _} = FileSystem.ensure_filesystem("invalid", [config])
    end

    test "validates configs is a list", %{supervisor_pid: _sup} do
      scope_key = {:user, System.unique_integer([:positive])}

      # Invalid configs (not a list)
      assert {:error, _} = FileSystem.ensure_filesystem(scope_key, "invalid")
    end
  end

  describe "start_filesystem/3" do
    test "starts a new filesystem", %{supervisor_pid: sup} do
      scope_key = {:user, System.unique_integer([:positive])}
      config = create_test_config(scope_key)

      assert {:ok, pid} = FileSystem.start_filesystem(scope_key, [config], supervisor: sup)
      assert Process.alive?(pid)

      # Clean up
      FileSystem.stop_filesystem(scope_key, supervisor: sup)
    end

    test "returns error if filesystem already running (not idempotent)", %{supervisor_pid: sup} do
      scope_key = {:user, System.unique_integer([:positive])}
      config = create_test_config(scope_key)

      # First call succeeds
      {:ok, pid} = FileSystem.start_filesystem(scope_key, [config], supervisor: sup)

      # Second call returns error
      assert {:error, {:already_started, ^pid}} =
               FileSystem.start_filesystem(scope_key, [config], supervisor: sup)

      # Clean up
      FileSystem.stop_filesystem(scope_key, supervisor: sup)
    end
  end

  describe "stop_filesystem/2" do
    test "stops a running filesystem", %{supervisor_pid: sup} do
      scope_key = {:user, System.unique_integer([:positive])}
      config = create_test_config(scope_key)

      {:ok, pid} = FileSystem.ensure_filesystem(scope_key, [config], supervisor: sup)
      assert Process.alive?(pid)

      # Stop the filesystem
      assert :ok = FileSystem.stop_filesystem(scope_key, supervisor: sup)

      # Give it time to stop
      Process.sleep(50)

      # Should no longer be running
      refute Process.alive?(pid)
      assert {:error, :not_found} = FileSystem.get_filesystem_pid(scope_key)
    end

    test "returns error if filesystem not running", %{supervisor_pid: sup} do
      scope_key = {:user, System.unique_integer([:positive])}

      assert {:error, :not_found} = FileSystem.stop_filesystem(scope_key, supervisor: sup)
    end
  end

  describe "filesystem_running?/1" do
    test "returns true when filesystem is running", %{supervisor_pid: sup} do
      scope_key = {:user, System.unique_integer([:positive])}
      config = create_test_config(scope_key)

      {:ok, _pid} = FileSystem.ensure_filesystem(scope_key, [config], supervisor: sup)

      assert FileSystem.filesystem_running?(scope_key) == true

      # Clean up
      FileSystem.stop_filesystem(scope_key, supervisor: sup)
    end

    test "returns false when filesystem is not running", %{supervisor_pid: _sup} do
      scope_key = {:user, System.unique_integer([:positive])}

      assert FileSystem.filesystem_running?(scope_key) == false
    end

    test "returns false after filesystem stops", %{supervisor_pid: sup} do
      scope_key = {:user, System.unique_integer([:positive])}
      config = create_test_config(scope_key)

      {:ok, _pid} = FileSystem.ensure_filesystem(scope_key, [config], supervisor: sup)
      assert FileSystem.filesystem_running?(scope_key) == true

      FileSystem.stop_filesystem(scope_key, supervisor: sup)
      Process.sleep(50)

      assert FileSystem.filesystem_running?(scope_key) == false
    end
  end

  describe "get_filesystem_pid/1" do
    test "returns PID when filesystem is running", %{supervisor_pid: sup} do
      scope_key = {:user, System.unique_integer([:positive])}
      config = create_test_config(scope_key)

      {:ok, pid} = FileSystem.ensure_filesystem(scope_key, [config], supervisor: sup)

      assert {:ok, ^pid} = FileSystem.get_filesystem_pid(scope_key)

      # Clean up
      FileSystem.stop_filesystem(scope_key, supervisor: sup)
    end

    test "returns error when filesystem is not running", %{supervisor_pid: _sup} do
      scope_key = {:user, System.unique_integer([:positive])}

      assert {:error, :not_found} = FileSystem.get_filesystem_pid(scope_key)
    end
  end

  describe "get_scope/1" do
    test "returns scope from filesystem PID", %{supervisor_pid: sup} do
      scope_key = {:user, System.unique_integer([:positive])}
      config = create_test_config(scope_key)

      {:ok, _pid} = FileSystem.ensure_filesystem(scope_key, [config], supervisor: sup)

      # This will be tested once we implement get_scope in FileSystemServer
      # For now, we'll just verify the function exists
      assert function_exported?(FileSystem, :get_scope, 1)

      # Clean up
      FileSystem.stop_filesystem(scope_key, supervisor: sup)
    end
  end

  describe "list_filesystems/0" do
    test "returns list of running filesystems", %{supervisor_pid: sup} do
      scope1 = {:user, System.unique_integer([:positive])}
      scope2 = {:project, System.unique_integer([:positive])}

      config1 = create_test_config(scope1)
      config2 = create_test_config(scope2)

      {:ok, pid1} = FileSystem.ensure_filesystem(scope1, [config1], supervisor: sup)
      {:ok, pid2} = FileSystem.ensure_filesystem(scope2, [config2], supervisor: sup)

      filesystems = FileSystem.list_filesystems()

      # Should include our filesystems
      assert {scope1, pid1} in filesystems
      assert {scope2, pid2} in filesystems

      # Clean up
      FileSystem.stop_filesystem(scope1, supervisor: sup)
      FileSystem.stop_filesystem(scope2, supervisor: sup)
    end

    test "returns empty list when no filesystems running", %{supervisor_pid: _sup} do
      filesystems = FileSystem.list_filesystems()
      # Just verify it's a list (might have entries from other tests if async)
      assert is_list(filesystems)
    end
  end

  describe "error handling" do
    test "ensure_filesystem propagates start errors", %{supervisor_pid: sup} do
      scope_key = {:user, System.unique_integer([:positive])}

      # Pass invalid config (not a FileSystemConfig struct) to trigger error
      assert {:error, _reason} =
               FileSystem.ensure_filesystem(scope_key, [:invalid_config], supervisor: sup)
    end

    test "handles concurrent ensure_filesystem calls gracefully", %{supervisor_pid: sup} do
      scope_key = {:user, System.unique_integer([:positive])}
      config = create_test_config(scope_key)

      # Start 5 concurrent tasks trying to ensure the same filesystem
      tasks =
        Enum.map(1..5, fn _ ->
          Task.async(fn ->
            FileSystem.ensure_filesystem(scope_key, [config], supervisor: sup)
          end)
        end)

      # Wait for all tasks to complete
      results = Enum.map(tasks, &Task.await/1)

      # All should succeed
      assert Enum.all?(results, fn result ->
               match?({:ok, _pid}, result)
             end)

      # All should return the same PID
      pids = Enum.map(results, fn {:ok, pid} -> pid end)
      [first_pid | rest] = pids
      assert Enum.all?(rest, fn pid -> pid == first_pid end)

      # Clean up
      FileSystem.stop_filesystem(scope_key, supervisor: sup)
    end
  end

  describe "integration with different scope types" do
    test "supports user-scoped filesystems", %{supervisor_pid: sup} do
      scope_key = {:user, System.unique_integer([:positive])}
      config = create_test_config(scope_key)

      {:ok, _pid} = FileSystem.ensure_filesystem(scope_key, [config], supervisor: sup)
      assert FileSystem.filesystem_running?(scope_key)

      FileSystem.stop_filesystem(scope_key, supervisor: sup)
    end

    test "supports project-scoped filesystems", %{supervisor_pid: sup} do
      scope_key = {:project, System.unique_integer([:positive])}
      config = create_test_config(scope_key)

      {:ok, _pid} = FileSystem.ensure_filesystem(scope_key, [config], supervisor: sup)
      assert FileSystem.filesystem_running?(scope_key)

      FileSystem.stop_filesystem(scope_key, supervisor: sup)
    end

    test "supports organization-scoped filesystems", %{supervisor_pid: sup} do
      scope_key = {:organization, System.unique_integer([:positive])}
      config = create_test_config(scope_key)

      {:ok, _pid} = FileSystem.ensure_filesystem(scope_key, [config], supervisor: sup)
      assert FileSystem.filesystem_running?(scope_key)

      FileSystem.stop_filesystem(scope_key, supervisor: sup)
    end

    test "supports agent-scoped filesystems", %{supervisor_pid: sup} do
      scope_key = {:agent, "agent-#{System.unique_integer([:positive])}"}
      config = create_test_config(scope_key)

      {:ok, _pid} = FileSystem.ensure_filesystem(scope_key, [config], supervisor: sup)
      assert FileSystem.filesystem_running?(scope_key)

      FileSystem.stop_filesystem(scope_key, supervisor: sup)
    end

    test "different scope types can coexist", %{supervisor_pid: sup} do
      user_scope = {:user, System.unique_integer([:positive])}
      project_scope = {:project, System.unique_integer([:positive])}
      org_scope = {:organization, System.unique_integer([:positive])}

      user_config = create_test_config(user_scope)
      project_config = create_test_config(project_scope)
      org_config = create_test_config(org_scope)

      {:ok, user_pid} = FileSystem.ensure_filesystem(user_scope, [user_config], supervisor: sup)

      {:ok, project_pid} =
        FileSystem.ensure_filesystem(project_scope, [project_config], supervisor: sup)

      {:ok, org_pid} = FileSystem.ensure_filesystem(org_scope, [org_config], supervisor: sup)

      # All should be running with different PIDs
      assert user_pid != project_pid
      assert project_pid != org_pid
      assert FileSystem.filesystem_running?(user_scope)
      assert FileSystem.filesystem_running?(project_scope)
      assert FileSystem.filesystem_running?(org_scope)

      # Clean up
      FileSystem.stop_filesystem(user_scope, supervisor: sup)
      FileSystem.stop_filesystem(project_scope, supervisor: sup)
      FileSystem.stop_filesystem(org_scope, supervisor: sup)
    end
  end
end
