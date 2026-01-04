defmodule LangChain.Agents.AgentServerDebugPubSubTest do
  use ExUnit.Case, async: false

  alias LangChain.Agents.{Agent, AgentServer, State, Middleware}
  alias LangChain.ChatModels.ChatAnthropic
  alias LangChain.Message

  setup do
    # Start PubSub for testing
    pubsub_name = :"test_pubsub_#{System.unique_integer([:positive])}"
    debug_pubsub_name = :"test_debug_pubsub_#{System.unique_integer([:positive])}"

    {:ok, _} =
      Phoenix.PubSub.Supervisor.start_link(name: pubsub_name, adapter_name: Phoenix.PubSub.PG2)

    {:ok, _} =
      Phoenix.PubSub.Supervisor.start_link(
        name: debug_pubsub_name,
        adapter_name: Phoenix.PubSub.PG2
      )

    %{
      pubsub_name: pubsub_name,
      debug_pubsub_name: debug_pubsub_name
    }
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

  # Simple test middleware that can trigger state updates
  defmodule TestMiddleware do
    @behaviour Middleware

    @impl true
    def init(opts) do
      # Convert opts to map
      config = Enum.into(opts, %{})
      {:ok, config}
    end

    @impl true
    def before_model(_state, _middleware_state), do: {:ok, %{}}

    @impl true
    def after_model(_state, _middleware_state), do: {:ok, %{}}

    @impl true
    def handle_message(:test_message, state, _middleware_state) do
      # Update state metadata
      updated_state = State.put_metadata(state, "test_key", "test_value")
      {:ok, updated_state}
    end

    @impl true
    def handle_message(_message, state, _middleware_state) do
      {:ok, state}
    end
  end

  describe "debug pubsub configuration" do
    test "starts server with debug pubsub enabled", %{
      pubsub_name: pubsub_name,
      debug_pubsub_name: debug_pubsub_name
    } do
      agent = create_test_agent()
      agent_id = agent.agent_id

      assert {:ok, _pid} =
               AgentServer.start_link(
                 agent: agent,
                 pubsub: {Phoenix.PubSub, pubsub_name},
                 debug_pubsub: {Phoenix.PubSub, debug_pubsub_name}
               )

      # Verify the server is running
      assert AgentServer.get_status(agent_id) == :idle
    end

    test "starts server without debug pubsub", %{pubsub_name: pubsub_name} do
      agent = create_test_agent()
      agent_id = agent.agent_id

      assert {:ok, _pid} =
               AgentServer.start_link(
                 agent: agent,
                 pubsub: {Phoenix.PubSub, pubsub_name}
               )

      # Verify subscribe_debug returns error
      assert {:error, :no_debug_pubsub} = AgentServer.subscribe_debug(agent_id)
    end
  end

  describe "subscribe_debug/1" do
    test "subscribes to debug events successfully", %{
      pubsub_name: pubsub_name,
      debug_pubsub_name: debug_pubsub_name
    } do
      agent = create_test_agent()
      agent_id = agent.agent_id

      {:ok, _pid} =
        AgentServer.start_link(
          agent: agent,
          pubsub: {Phoenix.PubSub, pubsub_name},
          debug_pubsub: {Phoenix.PubSub, debug_pubsub_name}
        )

      assert :ok = AgentServer.subscribe_debug(agent_id)
    end

    test "returns error when debug pubsub not configured", %{pubsub_name: pubsub_name} do
      agent = create_test_agent()
      agent_id = agent.agent_id

      {:ok, _pid} =
        AgentServer.start_link(
          agent: agent,
          pubsub: {Phoenix.PubSub, pubsub_name}
        )

      assert {:error, :no_debug_pubsub} = AgentServer.subscribe_debug(agent_id)
    end
  end

  describe "debug event broadcasting" do
    test "broadcasts agent_state_update on middleware message with broadcast: true", %{
      pubsub_name: pubsub_name,
      debug_pubsub_name: debug_pubsub_name
    } do
      # Create agent with test middleware
      agent =
        create_test_agent(
          middleware: [
            {TestMiddleware, []}
          ]
        )

      agent_id = agent.agent_id

      {:ok, _pid} =
        AgentServer.start_link(
          agent: agent,
          pubsub: {Phoenix.PubSub, pubsub_name},
          debug_pubsub: {Phoenix.PubSub, debug_pubsub_name}
        )

      # Subscribe to debug events
      :ok = AgentServer.subscribe_debug(agent_id)

      # Send a middleware message (use module name as ID)
      :ok = AgentServer.send_middleware_message(agent_id, TestMiddleware, :test_message)

      # Should receive debug event wrapped in {:debug, event} tuple
      assert_receive {:debug, {:agent_state_update, TestMiddleware, %State{}}}, 100
    end

    test "does not broadcast to regular pubsub when using debug events", %{
      pubsub_name: pubsub_name,
      debug_pubsub_name: debug_pubsub_name
    } do
      # Create agent with test middleware
      agent =
        create_test_agent(
          middleware: [
            {TestMiddleware, []}
          ]
        )

      agent_id = agent.agent_id

      {:ok, _pid} =
        AgentServer.start_link(
          agent: agent,
          pubsub: {Phoenix.PubSub, pubsub_name},
          debug_pubsub: {Phoenix.PubSub, debug_pubsub_name}
        )

      # Subscribe to regular events (not debug)
      :ok = AgentServer.subscribe(agent_id)

      # Send a middleware message
      :ok = AgentServer.send_middleware_message(agent_id, TestMiddleware, :test_message)

      # Should NOT receive agent_state_update on regular pubsub (neither wrapped nor unwrapped)
      refute_receive {:debug, {:agent_state_update, _, _}}, 100
      refute_receive {:agent_state_update, _, _}, 100
    end
  end

  describe "debug event topic" do
    test "uses correct debug topic format", %{
      pubsub_name: pubsub_name,
      debug_pubsub_name: debug_pubsub_name
    } do
      agent = create_test_agent()
      agent_id = agent.agent_id

      {:ok, _pid} =
        AgentServer.start_link(
          agent: agent,
          pubsub: {Phoenix.PubSub, pubsub_name},
          debug_pubsub: {Phoenix.PubSub, debug_pubsub_name}
        )

      # Get debug pubsub info
      {debug_pubsub, returned_debug_pubsub_name, debug_topic} =
        GenServer.call(AgentServer.get_name(agent_id), :get_debug_pubsub_info)

      assert debug_pubsub == Phoenix.PubSub
      assert returned_debug_pubsub_name == debug_pubsub_name
      assert debug_topic == "agent_server:debug:#{agent_id}"
    end

    test "returns nil when debug pubsub not configured", %{pubsub_name: pubsub_name} do
      agent = create_test_agent()
      agent_id = agent.agent_id

      {:ok, _pid} =
        AgentServer.start_link(
          agent: agent,
          pubsub: {Phoenix.PubSub, pubsub_name}
        )

      # Get debug pubsub info
      result = GenServer.call(AgentServer.get_name(agent_id), :get_debug_pubsub_info)

      assert result == nil
    end
  end

  describe "debug events with state restoration" do
    test "debug pubsub can be configured when restoring state", %{
      pubsub_name: pubsub_name,
      debug_pubsub_name: debug_pubsub_name
    } do
      # Create and export state
      agent = create_test_agent()
      agent_id = agent.agent_id
      initial_state = State.new!(%{messages: [Message.new_user!("Test message")]})

      {:ok, _pid} =
        AgentServer.start_link(
          agent: agent,
          initial_state: initial_state,
          pubsub: nil
        )

      exported_state = AgentServer.export_state(agent_id)
      :ok = AgentServer.stop(agent_id)

      # Give it a moment to fully stop
      Process.sleep(100)

      # Restore with debug pubsub enabled
      new_agent_id = "restored-agent-#{System.unique_integer([:positive])}"
      restored_agent = create_test_agent(agent_id: new_agent_id)

      assert {:ok, _pid} =
               AgentServer.start_link_from_state(
                 exported_state,
                 agent: restored_agent,
                 agent_id: new_agent_id,
                 pubsub: {Phoenix.PubSub, pubsub_name},
                 debug_pubsub: {Phoenix.PubSub, debug_pubsub_name}
               )

      # Should be able to subscribe to debug events
      assert :ok = AgentServer.subscribe_debug(new_agent_id)
    end
  end

  describe "multiple agents with debug pubsub" do
    test "multiple agents can use same debug pubsub instance", %{
      pubsub_name: pubsub_name,
      debug_pubsub_name: debug_pubsub_name
    } do
      # Create two agents
      agent1 = create_test_agent()
      agent2 = create_test_agent()

      {:ok, _pid1} =
        AgentServer.start_link(
          agent: agent1,
          pubsub: {Phoenix.PubSub, pubsub_name},
          debug_pubsub: {Phoenix.PubSub, debug_pubsub_name}
        )

      {:ok, _pid2} =
        AgentServer.start_link(
          agent: agent2,
          pubsub: {Phoenix.PubSub, pubsub_name},
          debug_pubsub: {Phoenix.PubSub, debug_pubsub_name}
        )

      # Both should be able to subscribe to their respective debug events
      assert :ok = AgentServer.subscribe_debug(agent1.agent_id)
      assert :ok = AgentServer.subscribe_debug(agent2.agent_id)
    end

    test "debug events are isolated per agent", %{
      pubsub_name: pubsub_name,
      debug_pubsub_name: debug_pubsub_name
    } do
      # Create two agents with middleware
      agent1 =
        create_test_agent(
          middleware: [
            {TestMiddleware, []}
          ]
        )

      agent2 =
        create_test_agent(
          middleware: [
            {TestMiddleware, []}
          ]
        )

      {:ok, _pid1} =
        AgentServer.start_link(
          agent: agent1,
          pubsub: {Phoenix.PubSub, pubsub_name},
          debug_pubsub: {Phoenix.PubSub, debug_pubsub_name}
        )

      {:ok, _pid2} =
        AgentServer.start_link(
          agent: agent2,
          pubsub: {Phoenix.PubSub, pubsub_name},
          debug_pubsub: {Phoenix.PubSub, debug_pubsub_name}
        )

      # Subscribe to agent1's debug events only
      :ok = AgentServer.subscribe_debug(agent1.agent_id)

      # Send message to agent2
      :ok = AgentServer.send_middleware_message(agent2.agent_id, TestMiddleware, :test_message)

      # Should NOT receive debug events from agent2
      refute_receive {:debug, {:agent_state_update, _, _}}, 100

      # Send message to agent1
      :ok = AgentServer.send_middleware_message(agent1.agent_id, TestMiddleware, :test_message)

      # Should receive debug events from agent1 wrapped in {:debug, event} tuple
      assert_receive {:debug, {:agent_state_update, TestMiddleware, %State{}}}, 100
    end
  end
end
