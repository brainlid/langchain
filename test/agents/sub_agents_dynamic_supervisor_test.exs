defmodule LangChain.Agents.SubAgentsDynamicSupervisorTest do
  use ExUnit.Case, async: false

  alias LangChain.Agents.SubAgentsDynamicSupervisor

  setup do
    # Note: Registry is started globally in test_helper.exs
    :ok
  end

  describe "start_link/1" do
    test "starts supervisor with agent_id" do
      agent_id = "test-agent-#{System.unique_integer([:positive])}"

      assert {:ok, sup_pid} = SubAgentsDynamicSupervisor.start_link(agent_id: agent_id)
      assert Process.alive?(sup_pid)

      # Verify it's registered
      assert SubAgentsDynamicSupervisor.whereis(agent_id) == sup_pid

      # Clean up
      DynamicSupervisor.stop(sup_pid)
    end

    test "starts with custom name" do
      agent_id = "test-agent-#{System.unique_integer([:positive])}"
      name = :"custom_name_#{System.unique_integer([:positive])}"

      assert {:ok, sup_pid} =
               SubAgentsDynamicSupervisor.start_link(
                 agent_id: agent_id,
                 name: name
               )

      assert Process.whereis(name) == sup_pid

      # Clean up
      DynamicSupervisor.stop(sup_pid)
    end

    test "raises error if agent_id is not provided" do
      assert_raise KeyError, fn ->
        SubAgentsDynamicSupervisor.start_link([])
      end
    end

    test "multiple supervisors for different agents" do
      agent_id_1 = "test-agent-1-#{System.unique_integer([:positive])}"
      agent_id_2 = "test-agent-2-#{System.unique_integer([:positive])}"

      {:ok, sup_pid_1} = SubAgentsDynamicSupervisor.start_link(agent_id: agent_id_1)
      {:ok, sup_pid_2} = SubAgentsDynamicSupervisor.start_link(agent_id: agent_id_2)

      assert sup_pid_1 != sup_pid_2
      assert SubAgentsDynamicSupervisor.whereis(agent_id_1) == sup_pid_1
      assert SubAgentsDynamicSupervisor.whereis(agent_id_2) == sup_pid_2

      # Clean up
      DynamicSupervisor.stop(sup_pid_1)
      DynamicSupervisor.stop(sup_pid_2)
    end
  end

  describe "whereis/1" do
    test "returns pid when supervisor exists" do
      agent_id = "test-agent-#{System.unique_integer([:positive])}"

      {:ok, sup_pid} = SubAgentsDynamicSupervisor.start_link(agent_id: agent_id)

      assert SubAgentsDynamicSupervisor.whereis(agent_id) == sup_pid

      # Clean up
      DynamicSupervisor.stop(sup_pid)
    end

    test "returns nil when supervisor does not exist" do
      agent_id = "nonexistent-agent-#{System.unique_integer([:positive])}"

      assert SubAgentsDynamicSupervisor.whereis(agent_id) == nil
    end

    test "returns nil after supervisor stops" do
      agent_id = "test-agent-#{System.unique_integer([:positive])}"

      {:ok, sup_pid} = SubAgentsDynamicSupervisor.start_link(agent_id: agent_id)
      assert SubAgentsDynamicSupervisor.whereis(agent_id) == sup_pid

      DynamicSupervisor.stop(sup_pid)

      # Give it time to stop
      Process.sleep(50)

      assert SubAgentsDynamicSupervisor.whereis(agent_id) == nil
    end
  end

  describe "dynamic child management" do
    test "can start and stop children dynamically" do
      agent_id = "test-agent-#{System.unique_integer([:positive])}"

      {:ok, sup_pid} = SubAgentsDynamicSupervisor.start_link(agent_id: agent_id)

      # Initially no children
      assert DynamicSupervisor.count_children(sup_pid) == %{
               active: 0,
               specs: 0,
               supervisors: 0,
               workers: 0
             }

      # Start a simple child (Task for testing)
      child_spec = %{
        id: :test_task,
        start: {Task, :start_link, [fn -> Process.sleep(:infinity) end]},
        restart: :temporary
      }

      {:ok, child_pid} = DynamicSupervisor.start_child(sup_pid, child_spec)
      assert Process.alive?(child_pid)

      # Now we have one child
      assert DynamicSupervisor.count_children(sup_pid).active == 1

      # Terminate the child
      DynamicSupervisor.terminate_child(sup_pid, child_pid)

      # Give it time to terminate
      Process.sleep(50)

      # Back to no children
      assert DynamicSupervisor.count_children(sup_pid).active == 0

      # Clean up
      DynamicSupervisor.stop(sup_pid)
    end

    test "uses :one_for_one strategy (child crash doesn't affect other children)" do
      agent_id = "test-agent-#{System.unique_integer([:positive])}"

      {:ok, sup_pid} = SubAgentsDynamicSupervisor.start_link(agent_id: agent_id)

      # Start two children
      child_spec = %{
        id: :test_task,
        start: {Task, :start_link, [fn -> Process.sleep(:infinity) end]},
        restart: :temporary
      }

      {:ok, child_pid_1} = DynamicSupervisor.start_child(sup_pid, child_spec)
      {:ok, child_pid_2} = DynamicSupervisor.start_child(sup_pid, child_spec)

      assert DynamicSupervisor.count_children(sup_pid).active == 2

      # Kill one child
      Process.exit(child_pid_1, :kill)

      # Give it time to handle the exit
      Process.sleep(50)

      # Since restart is :temporary, the killed child is not restarted
      # But the other child should still be alive
      assert Process.alive?(child_pid_2)

      # We should have one active child
      assert DynamicSupervisor.count_children(sup_pid).active == 1

      # Clean up
      DynamicSupervisor.stop(sup_pid)
    end
  end

  describe "integration with Registry" do
    test "supervisor is properly registered in Registry" do
      agent_id = "test-agent-#{System.unique_integer([:positive])}"

      {:ok, sup_pid} = SubAgentsDynamicSupervisor.start_link(agent_id: agent_id)

      # Check Registry lookup
      assert [{^sup_pid, _}] =
               Registry.lookup(LangChain.Agents.Registry, {:sub_agents_supervisor, agent_id})

      # Clean up
      DynamicSupervisor.stop(sup_pid)
    end

    test "supervisor is unregistered after stop" do
      agent_id = "test-agent-#{System.unique_integer([:positive])}"

      {:ok, sup_pid} = SubAgentsDynamicSupervisor.start_link(agent_id: agent_id)
      DynamicSupervisor.stop(sup_pid)

      # Give it time to stop
      Process.sleep(50)

      # Should not be in Registry anymore
      assert [] = Registry.lookup(LangChain.Agents.Registry, {:sub_agents_supervisor, agent_id})
    end
  end
end
