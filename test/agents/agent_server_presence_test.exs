defmodule LangChain.Agents.AgentServerPresenceTest do
  use LangChain.BaseCase, async: false
  use Mimic

  alias LangChain.Agents.{Agent, AgentServer, AgentSupervisor, State}
  alias LangChain.ChatModels.ChatAnthropic
  alias LangChain.Message

  setup :set_mimic_global
  setup :verify_on_exit!

  setup_all do
    # Copy Agent module for mocking
    Mimic.copy(Agent)
    :ok
  end

  # Mock Presence module for testing
  defmodule TestPresence do
    def track(_pid, _topic, _id, _metadata) do
      {:ok, make_ref()}
    end

    def untrack(_pid, _topic, _id) do
      :ok
    end

    def list(topic) do
      # Check if we have viewers stored in the test process
      case Process.get({:test_viewers, topic}) do
        nil -> %{}
        viewers -> viewers
      end
    end
  end

  # Helper to set viewers for a topic
  defp set_viewers(topic, viewers) do
    Process.put({:test_viewers, topic}, viewers)
  end

  # Helper to create a mock model
  defp mock_model do
    ChatAnthropic.new!(%{
      model: "claude-3-5-sonnet-20241022",
      api_key: "test_key"
    })
  end

  # Helper to create a simple agent
  defp create_test_agent(opts \\ []) do
    agent_id = Keyword.get(opts, :agent_id, "test-agent-#{System.unique_integer([:positive])}")
    opts = Keyword.put(opts, :agent_id, agent_id)

    Agent.new!(
      Map.merge(
        %{
          model: mock_model(),
          base_system_prompt: "Test agent",
          replace_default_middleware: true,
          middleware: []
        },
        Enum.into(opts, %{})
      )
    )
  end

  # Helper to mock agent execution - just returns success immediately
  defp mock_quick_execution do
    Agent
    |> expect(:execute, fn _agent, _state, _callbacks ->
      # Just return success immediately
      {:ok, State.new!()}
    end)
  end

  describe "presence-based shutdown" do
    test "agent shuts down when idle with no viewers" do
      agent = create_test_agent()
      agent_id = agent.agent_id
      initial_state = State.new!()

      presence_topic = "conversation:test-no-viewers"

      # Set no viewers for this topic
      set_viewers(presence_topic, %{})

      # Mock quick execution
      mock_quick_execution()

      # Start agent with presence tracking enabled and short shutdown delay
      {:ok, _pid} =
        AgentSupervisor.start_link(
          agent: agent,
          initial_state: initial_state,
          name: AgentSupervisor.get_name(agent_id),
          shutdown_delay: 100,
          presence_tracking: [
            enabled: true,
            presence_module: TestPresence,
            topic: presence_topic
          ]
        )

      # Wait for agent to be ready
      agent_pid = AgentServer.get_pid(agent_id)
      assert is_pid(agent_pid)

      # Monitor the agent process to detect when it dies
      ref = Process.monitor(agent_pid)

      # Add a message to trigger execution
      assert :ok = AgentServer.add_message(agent_id, Message.new_user!("test"))

      # Wait for the agent to shut down
      # The shutdown happens after: execution completes + 1s delay + shutdown_delay
      assert_receive {:DOWN, ^ref, :process, ^agent_pid, _reason}, 3000

      # Agent should have shut down
      refute Process.alive?(agent_pid)
    end

    test "agent stays alive when idle with viewers present" do
      agent = create_test_agent()
      agent_id = agent.agent_id
      initial_state = State.new!()

      presence_topic = "conversation:test-with-viewers"

      # Set some viewers for this topic
      set_viewers(presence_topic, %{
        "user-1" => %{metas: [%{joined_at: System.system_time(:second)}]},
        "user-2" => %{metas: [%{joined_at: System.system_time(:second)}]}
      })

      # Mock quick execution
      mock_quick_execution()

      # Start agent with presence tracking enabled
      {:ok, _pid} =
        AgentSupervisor.start_link(
          agent: agent,
          initial_state: initial_state,
          name: AgentSupervisor.get_name(agent_id),
          presence_tracking: [
            enabled: true,
            presence_module: TestPresence,
            topic: presence_topic
          ]
        )

      # Wait for agent to be ready
      agent_pid = AgentServer.get_pid(agent_id)
      assert is_pid(agent_pid)

      # Add a message to trigger execution
      assert :ok = AgentServer.add_message(agent_id, Message.new_user!("test"))

      # Wait a bit (but less than the shutdown delay)
      Process.sleep(1500)

      # Agent should still be alive (viewers present)
      assert Process.alive?(agent_pid)

      # Cleanup
      AgentSupervisor.stop(agent_id)
    end

    test "agent uses standard inactivity timeout when presence disabled" do
      agent = create_test_agent()
      agent_id = agent.agent_id
      initial_state = State.new!()

      # Mock quick execution
      mock_quick_execution()

      # Start agent WITHOUT presence tracking
      {:ok, _pid} =
        AgentSupervisor.start_link(
          agent: agent,
          initial_state: initial_state,
          name: AgentSupervisor.get_name(agent_id),
          inactivity_timeout: 10_000
          # No presence_tracking config
        )

      # Wait for agent to be ready
      agent_pid = AgentServer.get_pid(agent_id)
      assert is_pid(agent_pid)

      # Add a message to trigger execution
      assert :ok = AgentServer.add_message(agent_id, Message.new_user!("test"))

      # Wait for execution to complete
      Process.sleep(100)

      # Agent should still be alive (no presence-based shutdown, relies on inactivity timeout)
      assert Process.alive?(agent_pid)

      # Cleanup
      AgentSupervisor.stop(agent_id)
    end

    test "presence config is properly passed through supervisor to server" do
      agent = create_test_agent()
      agent_id = agent.agent_id
      initial_state = State.new!()

      presence_topic = "conversation:test-config"

      # Start agent with presence tracking (no need to execute, just check config)
      {:ok, _pid} =
        AgentSupervisor.start_link(
          agent: agent,
          initial_state: initial_state,
          name: AgentSupervisor.get_name(agent_id),
          presence_tracking: [
            enabled: true,
            presence_module: TestPresence,
            topic: presence_topic
          ]
        )

      # Wait for agent to be ready
      agent_pid = AgentServer.get_pid(agent_id)
      assert is_pid(agent_pid)

      # Get server state to verify presence config
      state = :sys.get_state(agent_pid)

      assert state.presence_config != nil
      assert state.presence_config.enabled == true
      assert state.presence_config.presence_module == TestPresence
      assert state.presence_config.topic == presence_topic

      # Cleanup
      AgentSupervisor.stop(agent_id)
    end
  end
end
