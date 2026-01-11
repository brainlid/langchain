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
      case Process.whereis(__MODULE__) do
        nil -> :ok
        pid when is_pid(pid) -> Elixir.Agent.stop(__MODULE__, :normal, :infinity)
      end
    catch
      :exit, _ -> :ok
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

  # Tracking presence module that records calls for verification
  defmodule AgentPresence do
    def start_link(_opts \\ []) do
      Elixir.Agent.start_link(fn -> %{tracked: [], updates: []} end, name: __MODULE__)
    end

    def stop do
      case Process.whereis(__MODULE__) do
        nil -> :ok
        pid when is_pid(pid) -> Elixir.Agent.stop(__MODULE__, :normal, :infinity)
      end
    catch
      :exit, _ -> :ok
    end

    def get_tracked do
      Elixir.Agent.get(__MODULE__, fn state -> state.tracked end)
    end

    def get_updates do
      Elixir.Agent.get(__MODULE__, fn state -> state.updates end)
    end

    # Phoenix.Presence compatible interface
    def track(pid, topic, id, metadata) do
      Elixir.Agent.update(__MODULE__, fn state ->
        %{state | tracked: [{pid, topic, id, metadata} | state.tracked]}
      end)

      {:ok, make_ref()}
    end

    def untrack(_pid, _topic, _id) do
      :ok
    end

    def list(_topic) do
      # Return empty for now, not used in these tests
      %{}
    end
  end

  describe "agent presence tracking" do
    setup do
      {:ok, _} = AgentPresence.start_link()
      on_exit(fn -> AgentPresence.stop() end)
      :ok
    end

    test "agent tracks presence on startup when presence_module is configured" do
      agent = create_test_agent()
      agent_id = agent.agent_id
      initial_state = State.new!()

      {:ok, _pid} =
        AgentSupervisor.start_link(
          agent: agent,
          initial_state: initial_state,
          name: AgentSupervisor.get_name(agent_id),
          presence_module: AgentPresence
        )

      agent_pid = AgentServer.get_pid(agent_id)
      assert is_pid(agent_pid)

      # Wait a moment for the handle_continue to complete
      Process.sleep(10)

      # Verify presence was tracked
      tracked = AgentPresence.get_tracked()
      assert length(tracked) == 1

      {pid, topic, tracked_id, metadata} = hd(tracked)
      assert pid == agent_pid
      assert topic == "agent_server:presence"
      assert tracked_id == agent_id
      assert metadata.status == :idle
      assert is_struct(metadata.started_at, DateTime)

      AgentSupervisor.stop(agent_id)
    end

    test "agent does not track presence when presence_module is nil" do
      agent = create_test_agent()
      agent_id = agent.agent_id
      initial_state = State.new!()

      {:ok, _pid} =
        AgentSupervisor.start_link(
          agent: agent,
          initial_state: initial_state,
          name: AgentSupervisor.get_name(agent_id)
          # No presence_module specified
        )

      agent_pid = AgentServer.get_pid(agent_id)
      assert is_pid(agent_pid)

      Process.sleep(10)

      # No presence should be tracked
      tracked = AgentPresence.get_tracked()
      assert tracked == []

      AgentSupervisor.stop(agent_id)
    end

    test "presence_module is stored in server state" do
      agent = create_test_agent()
      agent_id = agent.agent_id
      initial_state = State.new!()

      {:ok, _pid} =
        AgentSupervisor.start_link(
          agent: agent,
          initial_state: initial_state,
          name: AgentSupervisor.get_name(agent_id),
          presence_module: AgentPresence
        )

      agent_pid = AgentServer.get_pid(agent_id)
      state = :sys.get_state(agent_pid)

      assert state.presence_module == AgentPresence

      AgentSupervisor.stop(agent_id)
    end

    test "agent includes conversation_id in presence metadata" do
      agent = create_test_agent()
      agent_id = agent.agent_id
      initial_state = State.new!()
      conversation_id = "conv-123"

      {:ok, _pid} =
        AgentSupervisor.start_link(
          agent: agent,
          initial_state: initial_state,
          name: AgentSupervisor.get_name(agent_id),
          presence_module: AgentPresence,
          conversation_id: conversation_id
        )

      _agent_pid = AgentServer.get_pid(agent_id)
      Process.sleep(10)

      tracked = AgentPresence.get_tracked()
      assert length(tracked) == 1

      {_pid, _topic, _id, metadata} = hd(tracked)
      assert metadata.conversation_id == conversation_id

      AgentSupervisor.stop(agent_id)
    end

    test "agent includes filesystem_scope in presence metadata" do
      # Create agent with filesystem_scope
      agent = create_test_agent(filesystem_scope: {:project_id, "proj-456"})
      agent_id = agent.agent_id
      initial_state = State.new!()

      {:ok, _pid} =
        AgentSupervisor.start_link(
          agent: agent,
          initial_state: initial_state,
          name: AgentSupervisor.get_name(agent_id),
          presence_module: AgentPresence
        )

      _agent_pid = AgentServer.get_pid(agent_id)
      Process.sleep(10)

      tracked = AgentPresence.get_tracked()
      assert length(tracked) == 1

      {_pid, _topic, _id, metadata} = hd(tracked)
      assert metadata.project_id == "proj-456"

      AgentSupervisor.stop(agent_id)
    end

    test "agent includes last_activity_at in initial presence metadata" do
      agent = create_test_agent()
      agent_id = agent.agent_id
      initial_state = State.new!()

      before_start = DateTime.utc_now()

      {:ok, _pid} =
        AgentSupervisor.start_link(
          agent: agent,
          initial_state: initial_state,
          name: AgentSupervisor.get_name(agent_id),
          presence_module: AgentPresence
        )

      _agent_pid = AgentServer.get_pid(agent_id)
      Process.sleep(10)

      assert [tracked] = AgentPresence.get_tracked()

      {_pid, _topic, _id, metadata} = tracked
      assert is_struct(metadata.last_activity_at, DateTime)
      # last_activity_at should be around the same time as started_at
      assert DateTime.compare(metadata.last_activity_at, before_start) in [:gt, :eq]
      assert metadata.last_activity_at == metadata.started_at

      AgentSupervisor.stop(agent_id)
    end
  end

  # Helper to get metadata from real presence
  defp get_presence_metadata(agent_id) do
    presences = LangChain.TestPresence.list("agent_server:presence")

    case Map.get(presences, agent_id) do
      %{metas: [meta | _]} -> meta
      _ -> nil
    end
  end

  describe "presence activity updates (Phase 2)" do
    test "touch updates last_activity_at in presence metadata" do
      agent = create_test_agent()
      agent_id = agent.agent_id
      initial_state = State.new!()

      {:ok, _pid} =
        AgentSupervisor.start_link(
          agent: agent,
          initial_state: initial_state,
          name: AgentSupervisor.get_name(agent_id),
          presence_module: LangChain.TestPresence
        )

      agent_pid = AgentServer.get_pid(agent_id)
      assert is_pid(agent_pid)
      Process.sleep(10)

      # Get initial last_activity_at
      initial_metadata = get_presence_metadata(agent_id)
      assert initial_metadata != nil
      initial_activity = initial_metadata.last_activity_at

      # Wait a moment then touch
      Process.sleep(50)
      AgentServer.touch(agent_id)
      Process.sleep(10)

      # Verify last_activity_at was updated
      updated_metadata = get_presence_metadata(agent_id)
      assert DateTime.compare(updated_metadata.last_activity_at, initial_activity) == :gt

      AgentSupervisor.stop(agent_id)
    end

    test "status change updates last_activity_at in presence metadata" do
      agent = create_test_agent()
      agent_id = agent.agent_id
      initial_state = State.new!()

      mock_quick_execution()

      {:ok, _pid} =
        AgentSupervisor.start_link(
          agent: agent,
          initial_state: initial_state,
          name: AgentSupervisor.get_name(agent_id),
          presence_module: LangChain.TestPresence
        )

      agent_pid = AgentServer.get_pid(agent_id)
      assert is_pid(agent_pid)
      Process.sleep(10)

      # Get initial last_activity_at
      initial_metadata = get_presence_metadata(agent_id)
      assert initial_metadata != nil
      initial_activity = initial_metadata.last_activity_at

      # Trigger execution which changes status
      Process.sleep(50)
      AgentServer.add_message(agent_id, Message.new_user!("test"))
      Process.sleep(50)

      # Verify last_activity_at was updated
      updated_metadata = get_presence_metadata(agent_id)
      assert DateTime.compare(updated_metadata.last_activity_at, initial_activity) == :gt

      AgentSupervisor.stop(agent_id)
    end
  end
end
