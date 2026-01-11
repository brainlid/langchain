defmodule LangChain.Agents.AgentSupervisorTest do
  use LangChain.BaseCase, async: false
  use Mimic

  alias LangChain.Agents.Agent
  alias LangChain.Agents.AgentSupervisor
  alias LangChain.Agents.AgentServer
  alias LangChain.Agents.SubAgentsDynamicSupervisor
  alias LangChain.Agents.State
  alias LangChain.ChatModels.ChatAnthropic
  alias LangChain.Message

  setup :set_mimic_global
  setup :verify_on_exit!

  setup do
    # Mock ChatAnthropic.call to prevent real API calls
    stub(ChatAnthropic, :call, fn _model, _messages, _callbacks ->
      {:ok, [Message.new_assistant!("Mock response")]}
    end)

    :ok
  end

  describe "start_link/1" do
    test "starts supervisor with minimal config" do
      agent = create_test_agent()

      assert {:ok, sup_pid} = AgentSupervisor.start_link(agent: agent)
      assert Process.alive?(sup_pid)

      # Verify all children are started (AgentServer and SubAgentsDynamicSupervisor only)
      children = Supervisor.which_children(sup_pid)
      assert length(children) == 2

      # Verify AgentServer is running
      assert AgentServer.get_pid(agent.agent_id) != nil

      # Verify SubAgentsDynamicSupervisor is running
      assert SubAgentsDynamicSupervisor.whereis(agent.agent_id) != nil

      # Clean up
      Supervisor.stop(sup_pid)
    end

    test "starts supervisor with initial state" do
      agent = create_test_agent()
      agent_id = agent.agent_id
      initial_state = State.new!(%{messages: [Message.new_user!("Hello")]})

      assert {:ok, sup_pid} =
               AgentSupervisor.start_link(
                 agent: agent,
                 initial_state: initial_state
               )

      # Find the AgentServer child
      children = Supervisor.which_children(sup_pid)
      {_, agent_server_pid, _, _} = Enum.find(children, fn {id, _, _, _} -> id == AgentServer end)

      assert agent_server_pid != nil

      # Verify initial state was passed using agent_id
      state = AgentServer.get_state(agent_id)
      assert length(state.messages) == 1

      # Clean up
      Supervisor.stop(sup_pid)
    end

    test "starts with named registration" do
      agent = create_test_agent()
      name = :"test_supervisor_#{System.unique_integer([:positive])}"

      assert {:ok, sup_pid} =
               AgentSupervisor.start_link(
                 agent: agent,
                 name: name
               )

      assert Process.whereis(name) == sup_pid

      # Clean up
      Supervisor.stop(sup_pid)
    end

    test "raises error if agent is not provided" do
      # The supervisor will exit when start_link fails
      Process.flag(:trap_exit, true)

      assert {:error, {%KeyError{key: :agent}, _stacktrace}} =
               AgentSupervisor.start_link([])
    end

    test "raises error if agent is not an Agent struct" do
      # The supervisor will exit when start_link fails
      Process.flag(:trap_exit, true)

      assert {:error, {%ArgumentError{message: msg}, _stacktrace}} =
               AgentSupervisor.start_link(agent: %{not: "an agent"})

      assert msg =~ "must be a LangChain.Agents.Agent struct"
    end

    test "raises error if agent has no agent_id" do
      # Create an agent struct without going through new! (to bypass validation)
      invalid_agent = %Agent{
        agent_id: nil,
        model: mock_model(),
        base_system_prompt: "Test"
      }

      # The supervisor will exit when start_link fails
      Process.flag(:trap_exit, true)

      assert {:error, {%ArgumentError{message: msg}, _stacktrace}} =
               AgentSupervisor.start_link(agent: invalid_agent)

      assert msg =~ "must have a valid agent_id"
    end
  end

  describe "supervision strategy (:rest_for_one)" do
    test "AgentServer crash restarts SubAgentsDynamicSupervisor" do
      agent = create_test_agent()

      {:ok, sup_pid} = AgentSupervisor.start_link(agent: agent)

      # Get original child PIDs
      children_before = Supervisor.which_children(sup_pid)

      {_, agent_server_pid_before, _, _} =
        Enum.find(children_before, fn {id, _, _, _} -> id == AgentServer end)

      sub_sup_pid_before = SubAgentsDynamicSupervisor.whereis(agent.agent_id)

      # Kill AgentServer
      Process.exit(agent_server_pid_before, :kill)

      # Give supervisor time to restart
      Process.sleep(100)

      # Both should restart with new PIDs
      children_after = Supervisor.which_children(sup_pid)

      {_, agent_server_pid_after, _, _} =
        Enum.find(children_after, fn {id, _, _, _} -> id == AgentServer end)

      sub_sup_pid_after = SubAgentsDynamicSupervisor.whereis(agent.agent_id)

      # Both should have restarted (different PIDs)
      assert agent_server_pid_after != agent_server_pid_before
      assert sub_sup_pid_after != sub_sup_pid_before

      # Clean up
      Supervisor.stop(sup_pid)
    end

    test "SubAgentsDynamicSupervisor crash only restarts itself" do
      agent = create_test_agent()

      {:ok, sup_pid} = AgentSupervisor.start_link(agent: agent)

      # Get original child PIDs
      children_before = Supervisor.which_children(sup_pid)

      {_, agent_server_pid_before, _, _} =
        Enum.find(children_before, fn {id, _, _, _} -> id == AgentServer end)

      sub_sup_pid_before = SubAgentsDynamicSupervisor.whereis(agent.agent_id)

      # Kill SubAgentsDynamicSupervisor
      Process.exit(sub_sup_pid_before, :kill)

      # Give supervisor time to restart
      Process.sleep(100)

      # AgentServer should be the same
      children_after = Supervisor.which_children(sup_pid)

      {_, agent_server_pid_after, _, _} =
        Enum.find(children_after, fn {id, _, _, _} -> id == AgentServer end)

      sub_sup_pid_after = SubAgentsDynamicSupervisor.whereis(agent.agent_id)

      # AgentServer survives (same PID)
      assert agent_server_pid_after == agent_server_pid_before

      # SubAgentsDynamicSupervisor restarts (different PID)
      assert sub_sup_pid_after != sub_sup_pid_before

      # Clean up
      Supervisor.stop(sup_pid)
    end
  end

  describe "child process accessibility" do
    test "can access SubAgentsDynamicSupervisor via agent_id" do
      agent = create_test_agent()

      {:ok, sup_pid} = AgentSupervisor.start_link(agent: agent)

      # SubAgentsDynamicSupervisor should be accessible by agent_id
      sub_sup_pid = SubAgentsDynamicSupervisor.whereis(agent.agent_id)
      assert sub_sup_pid != nil
      assert Process.alive?(sub_sup_pid)

      # Clean up
      Supervisor.stop(sup_pid)
    end

    test "can access AgentServer from supervisor children" do
      agent = create_test_agent()
      agent_id = agent.agent_id

      {:ok, sup_pid} = AgentSupervisor.start_link(agent: agent)

      # Find AgentServer in children
      children = Supervisor.which_children(sup_pid)
      {_, agent_server_pid, _, _} = Enum.find(children, fn {id, _, _, _} -> id == AgentServer end)

      assert agent_server_pid != nil
      assert Process.alive?(agent_server_pid)

      # Should be able to interact with AgentServer using agent_id
      assert :idle == AgentServer.get_status(agent_id)

      # Clean up
      Supervisor.stop(sup_pid)
    end
  end

  describe "stop/2" do
    test "stops supervisor and all children by agent_id" do
      agent = create_test_agent()
      agent_id = agent.agent_id

      # Start with registered name
      {:ok, sup_pid} =
        AgentSupervisor.start_link(
          agent: agent,
          name: AgentSupervisor.get_name(agent_id)
        )

      # Get all child PIDs
      children = Supervisor.which_children(sup_pid)
      {_, agent_server_pid, _, _} = Enum.find(children, fn {id, _, _, _} -> id == AgentServer end)
      sub_sup_pid = SubAgentsDynamicSupervisor.whereis(agent_id)

      # Verify everything is running
      assert Process.alive?(sup_pid)
      assert Process.alive?(agent_server_pid)
      assert Process.alive?(sub_sup_pid)

      # Stop using agent_id
      assert :ok = AgentSupervisor.stop(agent_id)

      # Give processes time to shutdown
      Process.sleep(100)

      # Verify everything is stopped
      refute Process.alive?(sup_pid)
      refute Process.alive?(agent_server_pid)
      refute Process.alive?(sub_sup_pid)
    end

    test "returns error when agent_id not found" do
      assert {:error, :not_found} = AgentSupervisor.stop("non-existent-agent-id")
    end

    test "stops supervisor with custom timeout" do
      agent = create_test_agent()
      agent_id = agent.agent_id

      {:ok, sup_pid} =
        AgentSupervisor.start_link(
          agent: agent,
          name: AgentSupervisor.get_name(agent_id)
        )

      assert Process.alive?(sup_pid)

      # Stop with custom timeout
      assert :ok = AgentSupervisor.stop(agent_id, 10_000)

      Process.sleep(100)
      refute Process.alive?(sup_pid)
    end
  end

  describe "start_link_sync/1" do
    test "starts supervisor and waits for AgentServer to be ready" do
      agent = create_test_agent()

      assert {:ok, sup_pid} = AgentSupervisor.start_link_sync(agent: agent)
      assert Process.alive?(sup_pid)

      # AgentServer should be immediately accessible without any delays
      assert AgentServer.get_pid(agent.agent_id) != nil
      assert :idle == AgentServer.get_status(agent.agent_id)

      # Verify all children are started (AgentServer and SubAgentsDynamicSupervisor)
      children = Supervisor.which_children(sup_pid)
      assert length(children) == 2

      # Clean up
      Supervisor.stop(sup_pid)
    end

    test "allows immediate interaction with AgentServer after start_link_sync returns" do
      agent = create_test_agent()

      assert {:ok, sup_pid} = AgentSupervisor.start_link_sync(agent: agent)

      # Should be able to interact with AgentServer immediately without race conditions
      # These calls would fail if AgentServer wasn't registered yet
      assert :idle == AgentServer.get_status(agent.agent_id)
      state = AgentServer.get_state(agent.agent_id)
      assert state.agent_id == agent.agent_id

      # Should be able to add messages immediately
      message = Message.new_user!("test")
      assert :ok == AgentServer.add_message(agent.agent_id, message)

      # Clean up
      Supervisor.stop(sup_pid)
    end

    test "works with custom startup timeout" do
      agent = create_test_agent()

      assert {:ok, sup_pid} =
               AgentSupervisor.start_link_sync(
                 agent: agent,
                 startup_timeout: 10_000
               )

      assert Process.alive?(sup_pid)
      assert AgentServer.get_pid(agent.agent_id) != nil

      # Clean up
      Supervisor.stop(sup_pid)
    end

    test "handles already started supervisor" do
      agent = create_test_agent()
      agent_id = agent.agent_id

      # Start first time with registered name
      {:ok, sup_pid1} =
        AgentSupervisor.start_link_sync(
          agent: agent,
          name: AgentSupervisor.get_name(agent_id)
        )

      # Try to start again with same name
      result =
        AgentSupervisor.start_link_sync(
          agent: agent,
          name: AgentSupervisor.get_name(agent_id)
        )

      # Should handle already_started gracefully
      case result do
        {:ok, sup_pid2} ->
          # Some supervisors return the existing pid
          assert sup_pid1 == sup_pid2 or Process.alive?(sup_pid2)

        {:error, {:already_started, sup_pid2}} ->
          # Also acceptable - supervisor already running
          assert sup_pid1 == sup_pid2
      end

      # AgentServer should still be accessible
      assert AgentServer.get_pid(agent_id) != nil

      # Clean up
      Supervisor.stop(sup_pid1)
    end

    test "returns error if AgentServer fails to start within timeout" do
      # This test verifies the timeout mechanism by using a very short timeout
      # In normal conditions, agent starts quickly, but we can at least verify
      # the error format is correct when we get a timeout
      agent = create_test_agent()

      # Use a ridiculously short timeout to potentially trigger timeout
      # (though in practice the agent usually starts fast enough)
      result =
        AgentSupervisor.start_link_sync(
          agent: agent,
          startup_timeout: 1
        )

      case result do
        {:ok, sup_pid} ->
          # Agent started fast enough - still valid
          assert Process.alive?(sup_pid)
          Supervisor.stop(sup_pid)

        {:error, {:agent_startup_timeout, agent_id}} ->
          # Timeout occurred - verify error format
          assert agent_id == agent.agent_id
      end
    end
  end

  describe "inactivity timeout integration" do
    test "supervisor passes inactivity_timeout to AgentServer" do
      agent = create_test_agent()
      agent_id = agent.agent_id

      {:ok, sup_pid} =
        AgentSupervisor.start_link(
          agent: agent,
          inactivity_timeout: 600_000,
          name: AgentSupervisor.get_name(agent_id)
        )

      # Verify AgentServer received the timeout
      status = AgentServer.get_inactivity_status(agent_id)
      assert status.inactivity_timeout == 600_000
      assert status.timer_active == true

      # Clean up
      Supervisor.stop(sup_pid)
    end

    test "supervisor passes nil inactivity_timeout to disable" do
      agent = create_test_agent()
      agent_id = agent.agent_id

      {:ok, sup_pid} =
        AgentSupervisor.start_link(
          agent: agent,
          inactivity_timeout: nil,
          name: AgentSupervisor.get_name(agent_id)
        )

      # Verify AgentServer has timeout disabled
      status = AgentServer.get_inactivity_status(agent_id)
      assert status.inactivity_timeout == nil
      assert status.timer_active == false

      # Clean up
      Supervisor.stop(sup_pid)
    end

    test "supervisor uses default timeout when not specified" do
      agent = create_test_agent()
      agent_id = agent.agent_id

      {:ok, sup_pid} =
        AgentSupervisor.start_link(
          agent: agent,
          name: AgentSupervisor.get_name(agent_id)
        )

      # Verify AgentServer got default 5 minute timeout
      status = AgentServer.get_inactivity_status(agent_id)
      assert status.inactivity_timeout == 300_000

      # Clean up
      Supervisor.stop(sup_pid)
    end

    test "inactivity timeout triggers supervisor shutdown" do
      agent = create_test_agent()
      agent_id = agent.agent_id

      {:ok, sup_pid} =
        AgentSupervisor.start_link(
          agent: agent,
          # shutdown after 10ms
          inactivity_timeout: 10,
          # allow 100ms for the supervisor to shutdown
          shutdown_delay: 100,
          name: AgentSupervisor.get_name(agent_id)
        )

      # Get child PIDs
      children = Supervisor.which_children(sup_pid)
      {_, agent_server_pid, _, _} = Enum.find(children, fn {id, _, _, _} -> id == AgentServer end)
      sub_sup_pid = SubAgentsDynamicSupervisor.whereis(agent_id)

      # Monitor the supervisor
      ref = Process.monitor(sup_pid)

      # Wait for inactivity timeout and shutdown
      assert_receive {:DOWN, ^ref, :process, ^sup_pid, _reason}, 1_000

      # Verify all children stopped
      refute Process.alive?(sup_pid)
      refute Process.alive?(agent_server_pid)
      refute Process.alive?(sub_sup_pid)
    end
  end
end
