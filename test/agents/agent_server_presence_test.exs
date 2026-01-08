defmodule LangChain.Agents.AgentServerPresenceTest do
  use LangChain.BaseCase, async: false
  use Mimic

  alias LangChain.Agents.{Agent, AgentServer, AgentSupervisor, State}
  alias LangChain.Message

  setup :set_mimic_global
  setup :verify_on_exit!

  setup_all do
    Mimic.copy(Agent)
    :ok
  end

  # Cross-process presence mock using Elixir's Agent
  defmodule TestPresence do
    def start_link(_opts \\ []) do
      Elixir.Agent.start_link(fn -> %{} end, name: __MODULE__)
    end

    def stop do
      if Process.whereis(__MODULE__) do
        Elixir.Agent.stop(__MODULE__)
      end
    end

    def set_viewers(topic, viewers) do
      Elixir.Agent.update(__MODULE__, fn state -> Map.put(state, topic, viewers) end)
    end

    def clear_viewers(topic) do
      Elixir.Agent.update(__MODULE__, fn state -> Map.delete(state, topic) end)
    end

    # Phoenix.Presence compatible interface
    def track(_pid, _topic, _id, _metadata) do
      {:ok, make_ref()}
    end

    def untrack(_pid, _topic, _id) do
      :ok
    end

    def list(topic) do
      Elixir.Agent.get(__MODULE__, fn state -> Map.get(state, topic, %{}) end)
    end
  end

  setup do
    {:ok, _} = TestPresence.start_link()
    on_exit(fn -> TestPresence.stop() end)
    :ok
  end

  # Mock agent execution to return success immediately
  defp mock_quick_execution do
    Agent
    |> expect(:execute, fn _agent, _state, _callbacks ->
      {:ok, State.new!()}
    end)
  end

  # Short delays for fast tests
  @check_delay 10
  @shutdown_delay 10

  describe "presence-based shutdown" do
    test "agent shuts down when idle with no viewers" do
      agent = create_test_agent()
      agent_id = agent.agent_id
      initial_state = State.new!()
      presence_topic = "conversation:test-no-viewers"

      TestPresence.set_viewers(presence_topic, %{})
      mock_quick_execution()

      {:ok, _pid} =
        AgentSupervisor.start_link(
          agent: agent,
          initial_state: initial_state,
          name: AgentSupervisor.get_name(agent_id),
          shutdown_delay: @shutdown_delay,
          presence_tracking: [
            enabled: true,
            presence_module: TestPresence,
            topic: presence_topic,
            check_delay: @check_delay
          ]
        )

      agent_pid = AgentServer.get_pid(agent_id)
      assert is_pid(agent_pid)

      ref = Process.monitor(agent_pid)

      assert :ok = AgentServer.add_message(agent_id, Message.new_user!("test"))

      assert_receive {:DOWN, ^ref, :process, ^agent_pid, _reason}, 500

      refute Process.alive?(agent_pid)
    end

    test "agent stays alive when idle with viewers present" do
      agent = create_test_agent()
      agent_id = agent.agent_id
      initial_state = State.new!()
      presence_topic = "conversation:test-with-viewers"

      TestPresence.set_viewers(presence_topic, %{
        "user-1" => %{metas: [%{joined_at: System.system_time(:second)}]},
        "user-2" => %{metas: [%{joined_at: System.system_time(:second)}]}
      })

      mock_quick_execution()

      {:ok, _pid} =
        AgentSupervisor.start_link(
          agent: agent,
          initial_state: initial_state,
          name: AgentSupervisor.get_name(agent_id),
          shutdown_delay: @shutdown_delay,
          presence_tracking: [
            enabled: true,
            presence_module: TestPresence,
            topic: presence_topic,
            check_delay: @check_delay
          ]
        )

      agent_pid = AgentServer.get_pid(agent_id)
      assert is_pid(agent_pid)

      assert :ok = AgentServer.add_message(agent_id, Message.new_user!("test"))

      # Wait past the check delay
      Process.sleep(50)

      # Agent should still be alive (viewers present)
      assert Process.alive?(agent_pid)

      AgentSupervisor.stop(agent_id)
    end

    test "agent shuts down after viewers leave" do
      agent = create_test_agent()
      agent_id = agent.agent_id
      initial_state = State.new!()
      presence_topic = "conversation:test-viewers-leave"

      TestPresence.set_viewers(presence_topic, %{
        "user-1" => %{metas: [%{joined_at: System.system_time(:second)}]}
      })

      mock_quick_execution()

      {:ok, _pid} =
        AgentSupervisor.start_link(
          agent: agent,
          initial_state: initial_state,
          name: AgentSupervisor.get_name(agent_id),
          shutdown_delay: @shutdown_delay,
          presence_tracking: [
            enabled: true,
            presence_module: TestPresence,
            topic: presence_topic,
            check_delay: @check_delay
          ]
        )

      agent_pid = AgentServer.get_pid(agent_id)
      ref = Process.monitor(agent_pid)

      # Trigger execution - should stay alive due to viewers
      assert :ok = AgentServer.add_message(agent_id, Message.new_user!("test"))
      Process.sleep(50)
      assert Process.alive?(agent_pid)

      # Now remove viewers and trigger another execution
      TestPresence.set_viewers(presence_topic, %{})

      mock_quick_execution()
      assert :ok = AgentServer.add_message(agent_id, Message.new_user!("another message"))

      # Should now shut down
      assert_receive {:DOWN, ^ref, :process, ^agent_pid, _reason}, 500
      refute Process.alive?(agent_pid)
    end

    test "agent uses standard inactivity timeout when presence disabled" do
      agent = create_test_agent()
      agent_id = agent.agent_id
      initial_state = State.new!()

      mock_quick_execution()

      {:ok, _pid} =
        AgentSupervisor.start_link(
          agent: agent,
          initial_state: initial_state,
          name: AgentSupervisor.get_name(agent_id),
          inactivity_timeout: 10_000
        )

      agent_pid = AgentServer.get_pid(agent_id)
      assert is_pid(agent_pid)

      assert :ok = AgentServer.add_message(agent_id, Message.new_user!("test"))

      Process.sleep(50)

      # Agent should still be alive (no presence-based shutdown)
      assert Process.alive?(agent_pid)

      AgentSupervisor.stop(agent_id)
    end

    test "presence config is properly passed through supervisor to server" do
      agent = create_test_agent()
      agent_id = agent.agent_id
      initial_state = State.new!()
      presence_topic = "conversation:test-config"

      {:ok, _pid} =
        AgentSupervisor.start_link(
          agent: agent,
          initial_state: initial_state,
          name: AgentSupervisor.get_name(agent_id),
          presence_tracking: [
            enabled: true,
            presence_module: TestPresence,
            topic: presence_topic,
            check_delay: 500
          ]
        )

      agent_pid = AgentServer.get_pid(agent_id)
      assert is_pid(agent_pid)

      state = :sys.get_state(agent_pid)

      assert state.presence_config != nil
      assert state.presence_config.enabled == true
      assert state.presence_config.presence_module == TestPresence
      assert state.presence_config.topic == presence_topic
      assert state.presence_config.check_delay == 500

      AgentSupervisor.stop(agent_id)
    end
  end
end
