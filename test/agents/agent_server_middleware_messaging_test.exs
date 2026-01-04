defmodule LangChain.Agents.AgentServerMiddlewareMessagingTest do
  use ExUnit.Case, async: false

  alias LangChain.Agents.{Agent, AgentServer, State, Middleware, MiddlewareEntry}
  alias LangChain.ChatModels.ChatAnthropic

  setup do
    # Start PubSub for each test
    start_supervised({Phoenix.PubSub, name: :langchain_pubsub})
    :ok
  end

  # Test middleware that implements handle_message
  defmodule TestMiddleware do
    @behaviour Middleware

    @impl true
    def init(opts) do
      # Convert opts to map and preserve all options including :id
      config =
        opts
        |> Enum.into(%{})
        |> Map.put_new(:test_key, "default_value")

      {:ok, config}
    end

    @impl true
    def handle_message({:test_message, value}, state, config) do
      # Update state metadata with the value
      updated_state = State.put_metadata(state, "test_value", value)

      # Notify test process
      if config[:test_pid] do
        send(config[:test_pid], {:middleware_handled, value})
      end

      {:ok, updated_state}
    end

    @impl true
    def handle_message({:test_error}, _state, _config) do
      {:error, :intentional_test_error}
    end

    @impl true
    def handle_message(_message, state, _config) do
      {:ok, state}
    end
  end

  # Another test middleware to test multiple instances
  defmodule AnotherTestMiddleware do
    @behaviour Middleware

    @impl true
    def init(opts) do
      # Convert opts to map and preserve all options including :id
      config =
        opts
        |> Enum.into(%{})
        |> Map.put_new(:instance_name, "default")

      {:ok, config}
    end

    @impl true
    def handle_message({:instance_message, value}, state, config) do
      updated_state = State.put_metadata(state, "instance_#{config.instance_name}", value)

      if config[:test_pid] do
        send(config[:test_pid], {:instance_handled, config.instance_name, value})
      end

      {:ok, updated_state}
    end

    @impl true
    def handle_message(_message, state, _config) do
      {:ok, state}
    end
  end

  # Helper to create a mock model
  defp mock_model do
    ChatAnthropic.new!(%{
      model: "claude-3-5-sonnet-20241022",
      api_key: "test_key"
    })
  end

  # Helper to create a test agent with middleware
  defp create_test_agent_with_middleware(middleware_specs, opts \\ []) do
    agent_id = Keyword.get(opts, :agent_id, "test-agent-#{System.unique_integer([:positive])}")

    Agent.new!(%{
      agent_id: agent_id,
      model: mock_model(),
      base_system_prompt: "Test agent",
      replace_default_middleware: true,
      middleware: middleware_specs
    })
  end

  describe "middleware registry initialization" do
    test "builds middleware registry with default module name IDs" do
      middleware_specs = [
        {TestMiddleware, [test_key: "value1"]},
        {AnotherTestMiddleware, [instance_name: "first"]}
      ]

      agent = create_test_agent_with_middleware(middleware_specs)
      agent_id = agent.agent_id

      {:ok, _pid} =
        AgentServer.start_link(
          agent: agent,
          name: AgentServer.get_name(agent_id),
          pubsub: nil
        )

      # Get the server state to inspect the registry
      state = :sys.get_state(AgentServer.get_name(agent_id))

      # Verify agent.middleware is a list of MiddlewareEntry structs
      assert Enum.all?(state.agent.middleware, &match?(%MiddlewareEntry{}, &1))

      # Verify middleware entries have IDs
      assert Enum.all?(state.agent.middleware, fn %LangChain.Agents.MiddlewareEntry{id: id} ->
               id != nil
             end)

      # Verify registry has entries for both middleware
      assert Map.has_key?(state.middleware_registry, TestMiddleware)
      assert Map.has_key?(state.middleware_registry, AnotherTestMiddleware)

      # Verify entry structure
      entry = state.middleware_registry[TestMiddleware]
      assert entry.id == TestMiddleware
      assert entry.module == TestMiddleware
      # ID is stored in the entry struct, not in config
      assert is_map(entry.config) or is_list(entry.config)
    end

    test "builds middleware registry with custom IDs" do
      middleware_specs = [
        {TestMiddleware, [id: "custom_test_1", test_key: "value1"]},
        {TestMiddleware, [id: "custom_test_2", test_key: "value2"]}
      ]

      agent = create_test_agent_with_middleware(middleware_specs)
      agent_id = agent.agent_id

      {:ok, _pid} =
        AgentServer.start_link(
          agent: agent,
          name: AgentServer.get_name(agent_id),
          pubsub: nil
        )

      # Get the server state to inspect the registry
      state = :sys.get_state(AgentServer.get_name(agent_id))

      # Verify registry has entries with custom IDs
      assert Map.has_key?(state.middleware_registry, "custom_test_1")
      assert Map.has_key?(state.middleware_registry, "custom_test_2")

      # Verify both instances use the same module but different configs
      entry1 = state.middleware_registry["custom_test_1"]
      entry2 = state.middleware_registry["custom_test_2"]

      assert entry1.module == TestMiddleware
      assert entry2.module == TestMiddleware
      assert entry1.config.test_key == "value1"
      assert entry2.config.test_key == "value2"
    end
  end

  describe "middleware message routing" do
    test "routes messages to correct middleware by module name" do
      test_pid = self()

      middleware_specs = [
        {TestMiddleware, [test_pid: test_pid]}
      ]

      agent = create_test_agent_with_middleware(middleware_specs)
      agent_id = agent.agent_id

      {:ok, server_pid} =
        AgentServer.start_link(
          agent: agent,
          name: AgentServer.get_name(agent_id),
          pubsub: nil
        )

      # Send middleware message
      send(server_pid, {:middleware_message, TestMiddleware, {:test_message, "hello"}})

      # Wait for middleware to handle the message
      assert_receive {:middleware_handled, "hello"}, 1000

      # Verify state was updated
      state = AgentServer.get_state(agent_id)
      assert State.get_metadata(state, "test_value") == "hello"
    end

    test "routes messages to correct middleware by custom ID" do
      test_pid = self()

      middleware_specs = [
        {TestMiddleware, [id: "custom_middleware", test_pid: test_pid]}
      ]

      agent = create_test_agent_with_middleware(middleware_specs)
      agent_id = agent.agent_id

      {:ok, server_pid} =
        AgentServer.start_link(
          agent: agent,
          name: AgentServer.get_name(agent_id),
          pubsub: nil
        )

      # Send middleware message using custom ID
      send(server_pid, {:middleware_message, "custom_middleware", {:test_message, "world"}})

      # Wait for middleware to handle the message
      assert_receive {:middleware_handled, "world"}, 1000

      # Verify state was updated
      state = AgentServer.get_state(agent_id)
      assert State.get_metadata(state, "test_value") == "world"
    end

    test "handles messages to multiple middleware instances correctly" do
      test_pid = self()

      middleware_specs = [
        {AnotherTestMiddleware, [id: "instance_1", instance_name: "first", test_pid: test_pid]},
        {AnotherTestMiddleware, [id: "instance_2", instance_name: "second", test_pid: test_pid]}
      ]

      agent = create_test_agent_with_middleware(middleware_specs)
      agent_id = agent.agent_id

      {:ok, server_pid} =
        AgentServer.start_link(
          agent: agent,
          name: AgentServer.get_name(agent_id),
          pubsub: nil
        )

      # Send messages to both instances
      send(server_pid, {:middleware_message, "instance_1", {:instance_message, "value1"}})
      send(server_pid, {:middleware_message, "instance_2", {:instance_message, "value2"}})

      # Wait for both middleware to handle messages
      assert_receive {:instance_handled, "first", "value1"}, 1000
      assert_receive {:instance_handled, "second", "value2"}, 1000

      # Verify both states were updated independently
      state = AgentServer.get_state(agent_id)
      assert State.get_metadata(state, "instance_first") == "value1"
      assert State.get_metadata(state, "instance_second") == "value2"
    end

    test "handles unknown middleware ID gracefully" do
      middleware_specs = [
        {TestMiddleware, []}
      ]

      agent = create_test_agent_with_middleware(middleware_specs)
      agent_id = agent.agent_id

      {:ok, server_pid} =
        AgentServer.start_link(
          agent: agent,
          name: AgentServer.get_name(agent_id),
          pubsub: nil
        )

      # Send message to non-existent middleware
      send(server_pid, {:middleware_message, :unknown_middleware, {:test_message, "test"}})

      # Give it a moment to process
      Process.sleep(100)

      # Server should still be alive and state unchanged
      assert Process.alive?(server_pid)
      state = AgentServer.get_state(agent_id)
      assert State.get_metadata(state, "test_value") == nil
    end

    test "handles middleware errors gracefully" do
      test_pid = self()

      middleware_specs = [
        {TestMiddleware, [test_pid: test_pid]}
      ]

      agent = create_test_agent_with_middleware(middleware_specs)
      agent_id = agent.agent_id

      {:ok, server_pid} =
        AgentServer.start_link(
          agent: agent,
          name: AgentServer.get_name(agent_id),
          pubsub: nil
        )

      # Send message that will cause an error
      send(server_pid, {:middleware_message, TestMiddleware, {:test_error}})

      # Give it a moment to process
      Process.sleep(100)

      # Server should still be alive
      assert Process.alive?(server_pid)
    end
  end

  describe "middleware state broadcast" do
    test "broadcasts state update to debug channel on middleware state changes" do
      test_pid = self()

      middleware_specs = [
        {TestMiddleware, [test_pid: test_pid]}
      ]

      agent = create_test_agent_with_middleware(middleware_specs)
      agent_id = agent.agent_id

      # Start with PubSub and debug PubSub enabled
      {:ok, server_pid} =
        AgentServer.start_link(
          agent: agent,
          name: AgentServer.get_name(agent_id),
          pubsub: {Phoenix.PubSub, :langchain_pubsub},
          debug_pubsub: {Phoenix.PubSub, :langchain_pubsub}
        )

      # Subscribe to agent debug events
      Phoenix.PubSub.subscribe(:langchain_pubsub, "agent_server:debug:#{agent_id}")

      # Send message that updates state
      send(
        server_pid,
        {:middleware_message, TestMiddleware, {:test_message, "test_value"}}
      )

      # Wait for middleware to handle the message
      assert_receive {:middleware_handled, "test_value"}, 1000

      # Should receive broadcast event on debug channel wrapped in {:debug, event} tuple
      assert_receive {:debug, {:agent_state_update, TestMiddleware, _state}}, 1000
    end
  end
end
