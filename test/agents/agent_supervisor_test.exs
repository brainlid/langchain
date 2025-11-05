defmodule LangChain.Agents.AgentSupervisorTest do
  use ExUnit.Case, async: false
  use Mimic

  alias LangChain.Agents.{Agent, AgentSupervisor, AgentServer, FileSystemServer, SubAgentsDynamicSupervisor}
  alias LangChain.Agents.FileSystem.FileSystemConfig
  alias LangChain.Agents.State
  alias LangChain.ChatModels.ChatAnthropic
  alias LangChain.Message

  setup :set_mimic_global
  setup :verify_on_exit!

  # Helper to create a mock model
  defp mock_model do
    ChatAnthropic.new!(%{
      model: "claude-3-5-sonnet-20241022",
      api_key: "test_key"
    })
  end

  # Helper to create a test agent
  defp create_test_agent(agent_id \\ "test-agent-#{System.unique_integer([:positive])}") do
    Agent.new!(
      agent_id: agent_id,
      model: mock_model(),
      system_prompt: "Test agent",
      replace_default_middleware: true,
      middleware: []
    )
  end

  describe "start_link/1" do
    test "starts supervisor with minimal config" do
      agent = create_test_agent()

      assert {:ok, sup_pid} = AgentSupervisor.start_link(agent: agent)
      assert Process.alive?(sup_pid)

      # Verify all children are started
      children = Supervisor.which_children(sup_pid)
      assert length(children) == 3

      # Verify FileSystemServer is running
      assert FileSystemServer.whereis(agent.agent_id) != nil

      # Verify SubAgentsDynamicSupervisor is running
      assert SubAgentsDynamicSupervisor.whereis(agent.agent_id) != nil

      # Clean up
      Supervisor.stop(sup_pid)
    end

    test "starts supervisor with persistence configs" do
      agent = create_test_agent()

      # Note: Using a mock module here since we don't have an actual persistence module
      config =
        FileSystemConfig.new!(%{
          base_directory: "TestDir",
          persistence_module: LangChain.Agents.FileSystem.Persistence.Disk,
          debounce_ms: 1000,
          storage_opts: [path: "/tmp/test"]
        })

      assert {:ok, sup_pid} =
               AgentSupervisor.start_link(
                 agent: agent,
                 persistence_configs: [config]
               )

      assert Process.alive?(sup_pid)

      # Verify FileSystemServer received the config
      fs_pid = FileSystemServer.whereis(agent.agent_id)
      assert fs_pid != nil

      configs = FileSystemServer.get_persistence_configs(agent.agent_id)
      assert map_size(configs) == 1
      assert configs["TestDir"].base_directory == "TestDir"

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
        system_prompt: "Test"
      }

      # The supervisor will exit when start_link fails
      Process.flag(:trap_exit, true)

      assert {:error, {%ArgumentError{message: msg}, _stacktrace}} =
               AgentSupervisor.start_link(agent: invalid_agent)

      assert msg =~ "must have a valid agent_id"
    end
  end

  describe "supervision strategy (:rest_for_one)" do
    test "FileSystemServer crash restarts all children" do
      agent = create_test_agent()

      {:ok, sup_pid} = AgentSupervisor.start_link(agent: agent)

      # Get original child PIDs
      fs_pid_before = FileSystemServer.whereis(agent.agent_id)
      children_before = Supervisor.which_children(sup_pid)
      {_, agent_server_pid_before, _, _} = Enum.find(children_before, fn {id, _, _, _} -> id == AgentServer end)
      sub_sup_pid_before = SubAgentsDynamicSupervisor.whereis(agent.agent_id)

      # Kill FileSystemServer
      Process.exit(fs_pid_before, :kill)

      # Give supervisor time to restart
      Process.sleep(100)

      # All children should have restarted (new PIDs)
      fs_pid_after = FileSystemServer.whereis(agent.agent_id)
      children_after = Supervisor.which_children(sup_pid)
      {_, agent_server_pid_after, _, _} = Enum.find(children_after, fn {id, _, _, _} -> id == AgentServer end)
      sub_sup_pid_after = SubAgentsDynamicSupervisor.whereis(agent.agent_id)

      assert fs_pid_after != fs_pid_before
      assert agent_server_pid_after != agent_server_pid_before
      assert sub_sup_pid_after != sub_sup_pid_before

      # Clean up
      Supervisor.stop(sup_pid)
    end

    test "AgentServer crash restarts AgentServer and SubAgentsDynamicSupervisor but not FileSystemServer" do
      agent = create_test_agent()

      {:ok, sup_pid} = AgentSupervisor.start_link(agent: agent)

      # Get original child PIDs
      fs_pid_before = FileSystemServer.whereis(agent.agent_id)
      children_before = Supervisor.which_children(sup_pid)
      {_, agent_server_pid_before, _, _} = Enum.find(children_before, fn {id, _, _, _} -> id == AgentServer end)
      sub_sup_pid_before = SubAgentsDynamicSupervisor.whereis(agent.agent_id)

      # Kill AgentServer
      Process.exit(agent_server_pid_before, :kill)

      # Give supervisor time to restart
      Process.sleep(100)

      # FileSystemServer should be the same, but AgentServer and SubAgentsDynamicSupervisor should be new
      fs_pid_after = FileSystemServer.whereis(agent.agent_id)
      children_after = Supervisor.which_children(sup_pid)
      {_, agent_server_pid_after, _, _} = Enum.find(children_after, fn {id, _, _, _} -> id == AgentServer end)
      sub_sup_pid_after = SubAgentsDynamicSupervisor.whereis(agent.agent_id)

      # FileSystemServer survives (same PID)
      assert fs_pid_after == fs_pid_before

      # AgentServer and SubAgentsDynamicSupervisor restart (different PIDs)
      assert agent_server_pid_after != agent_server_pid_before
      assert sub_sup_pid_after != sub_sup_pid_before

      # Clean up
      Supervisor.stop(sup_pid)
    end

    test "SubAgentsDynamicSupervisor crash only restarts itself" do
      agent = create_test_agent()

      {:ok, sup_pid} = AgentSupervisor.start_link(agent: agent)

      # Get original child PIDs
      fs_pid_before = FileSystemServer.whereis(agent.agent_id)
      children_before = Supervisor.which_children(sup_pid)
      {_, agent_server_pid_before, _, _} = Enum.find(children_before, fn {id, _, _, _} -> id == AgentServer end)
      sub_sup_pid_before = SubAgentsDynamicSupervisor.whereis(agent.agent_id)

      # Kill SubAgentsDynamicSupervisor
      Process.exit(sub_sup_pid_before, :kill)

      # Give supervisor time to restart
      Process.sleep(100)

      # FileSystemServer and AgentServer should be the same
      fs_pid_after = FileSystemServer.whereis(agent.agent_id)
      children_after = Supervisor.which_children(sup_pid)
      {_, agent_server_pid_after, _, _} = Enum.find(children_after, fn {id, _, _, _} -> id == AgentServer end)
      sub_sup_pid_after = SubAgentsDynamicSupervisor.whereis(agent.agent_id)

      # FileSystemServer and AgentServer survive (same PIDs)
      assert fs_pid_after == fs_pid_before
      assert agent_server_pid_after == agent_server_pid_before

      # SubAgentsDynamicSupervisor restarts (different PID)
      assert sub_sup_pid_after != sub_sup_pid_before

      # Clean up
      Supervisor.stop(sup_pid)
    end
  end

  describe "child process accessibility" do
    test "can access FileSystemServer via agent_id" do
      agent = create_test_agent()

      {:ok, sup_pid} = AgentSupervisor.start_link(agent: agent)

      # FileSystemServer should be accessible by agent_id
      fs_pid = FileSystemServer.whereis(agent.agent_id)
      assert fs_pid != nil
      assert Process.alive?(fs_pid)

      # Should be able to interact with FileSystemServer
      assert :ok == FileSystemServer.write_file(agent.agent_id, "/test.txt", "content")

      # Clean up
      Supervisor.stop(sup_pid)
    end

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
end
