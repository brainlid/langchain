defmodule LangChain.Integration.DebugFlowTest do
  @moduledoc """
  Integration tests for the complete debug flow.

  These tests verify the end-to-end debugging workflow including:
  - Presence-based agent discovery
  - Debug event broadcasting
  - Sub-agent event propagation
  - Late-connecting debugger scenarios

  Note: These tests use mocks for LLM calls to avoid external API dependencies.
  """
  use LangChain.BaseCase, async: false
  use Mimic

  alias LangChain.Agents.{AgentServer, State, SubAgent, SubAgentServer}
  alias LangChain.Chains.LLMChain
  alias LangChain.Message

  @moduletag :integration

  setup :set_mimic_global
  setup :verify_on_exit!

  setup_all do
    Mimic.copy(LLMChain)
    :ok
  end

  setup do
    # Start unique PubSub instances for each test
    pubsub_name = :"test_pubsub_#{System.unique_integer([:positive])}"
    debug_pubsub_name = :"test_debug_pubsub_#{System.unique_integer([:positive])}"
    presence_name = :"test_presence_#{System.unique_integer([:positive])}"

    {:ok, _} =
      Phoenix.PubSub.Supervisor.start_link(name: pubsub_name, adapter_name: Phoenix.PubSub.PG2)

    {:ok, _} =
      Phoenix.PubSub.Supervisor.start_link(
        name: debug_pubsub_name,
        adapter_name: Phoenix.PubSub.PG2
      )

    # Start a test Presence module
    {:ok, _} = start_test_presence(presence_name, pubsub_name)

    %{
      pubsub_name: pubsub_name,
      debug_pubsub_name: debug_pubsub_name,
      presence_name: presence_name
    }
  end

  # Helper to start a test Presence module
  # Note: For testing, we use a simplified presence tracking approach
  # Real presence modules require compile-time definition
  defp start_test_presence(_presence_name, _pubsub_name) do
    # For these integration tests, we skip actual presence module testing
    # The presence functionality is tested in agent_server_presence_test.exs
    # Here we focus on the debug event flow
    {:ok, nil}
  end

  # Note: Presence-based discovery tests are in agent_server_presence_test.exs
  # These require a compile-time defined Presence module which can't be
  # dynamically created in integration tests. The tests here focus on
  # debug event flow which doesn't require presence.

  describe "debug event broadcasting" do
    test "debug events broadcast on debug topic", context do
      agent = create_test_agent()

      {:ok, _pid} =
        AgentServer.start_link(
          agent: agent,
          pubsub: {Phoenix.PubSub, context.pubsub_name},
          debug_pubsub: {Phoenix.PubSub, context.debug_pubsub_name}
        )

      # Subscribe to debug topic
      debug_topic = "agent_server:debug:#{agent.agent_id}"
      Phoenix.PubSub.subscribe(context.debug_pubsub_name, debug_topic)

      # Mock LLM response
      assistant_message = Message.new_assistant!(%{content: "Test response"})
      LLMChain
      |> stub(:run, fn chain ->
        updated_chain =
          chain
          |> Map.put(:messages, chain.messages ++ [assistant_message])
          |> Map.put(:last_message, assistant_message)
          |> Map.put(:needs_response, false)

        {:ok, updated_chain}
      end)

      # Trigger execution
      AgentServer.add_message(agent.agent_id, Message.new_user!("Hello"))

      # Should receive debug events
      assert_receive {:agent, {:debug, _event}}, 100
    end

    test "no debug events when debug_pubsub not configured", context do
      agent = create_test_agent()

      {:ok, _pid} =
        AgentServer.start_link(
          agent: agent,
          pubsub: {Phoenix.PubSub, context.pubsub_name}
          # No debug_pubsub configured
        )

      # Subscribe anyway
      debug_topic = "agent_server:debug:#{agent.agent_id}"
      Phoenix.PubSub.subscribe(context.debug_pubsub_name, debug_topic)

      # Mock LLM response
      assistant_message = Message.new_assistant!(%{content: "Test"})
      LLMChain
      |> stub(:run, fn chain ->
        updated_chain =
          chain
          |> Map.put(:messages, chain.messages ++ [assistant_message])
          |> Map.put(:last_message, assistant_message)
          |> Map.put(:needs_response, false)

        {:ok, updated_chain}
      end)

      # Trigger execution
      AgentServer.add_message(agent.agent_id, Message.new_user!("Hello"))

      # Should NOT receive debug events
      refute_receive {:agent, {:debug, _event}}, 100
    end
  end

  describe "sub-agent event propagation" do
    test "sub-agent events broadcast via parent debug topic", context do
      parent_agent = create_test_agent()

      {:ok, _pid} =
        AgentServer.start_link(
          agent: parent_agent,
          pubsub: {Phoenix.PubSub, context.pubsub_name},
          debug_pubsub: {Phoenix.PubSub, context.debug_pubsub_name}
        )

      # Subscribe to parent's debug topic
      debug_topic = "agent_server:debug:#{parent_agent.agent_id}"
      Phoenix.PubSub.subscribe(context.debug_pubsub_name, debug_topic)

      # Create and start a subagent
      subagent =
        SubAgent.new_from_config(
          parent_agent_id: parent_agent.agent_id,
          instructions: "Test task",
          agent_config: create_test_agent(),
          parent_state: State.new!(%{})
        )

      {:ok, _pid} = SubAgentServer.start_link(subagent: subagent)

      # Should receive subagent_started event
      assert_receive {:agent, {:debug, {:subagent, sub_id, {:subagent_started, data}}}}, 100

      assert sub_id == subagent.id
      assert data.parent_id == parent_agent.agent_id
    end

    test "sub-agent completion broadcasts via parent", context do
      parent_agent = create_test_agent()

      {:ok, _pid} =
        AgentServer.start_link(
          agent: parent_agent,
          pubsub: {Phoenix.PubSub, context.pubsub_name},
          debug_pubsub: {Phoenix.PubSub, context.debug_pubsub_name}
        )

      debug_topic = "agent_server:debug:#{parent_agent.agent_id}"
      Phoenix.PubSub.subscribe(context.debug_pubsub_name, debug_topic)

      subagent =
        SubAgent.new_from_config(
          parent_agent_id: parent_agent.agent_id,
          instructions: "Test task",
          agent_config: create_test_agent(),
          parent_state: State.new!(%{})
        )

      {:ok, _pid} = SubAgentServer.start_link(subagent: subagent)

      # Consume started event
      assert_receive {:agent, {:debug, {:subagent, _, {:subagent_started, _}}}}, 100

      # Mock LLM success
      assistant_message = Message.new_assistant!(%{content: "Done"})
      LLMChain
      |> stub(:run, fn chain ->
        updated_chain =
          chain
          |> Map.put(:messages, chain.messages ++ [assistant_message])
          |> Map.put(:last_message, assistant_message)
          |> Map.put(:needs_response, false)

        {:ok, updated_chain}
      end)

      # Execute subagent
      {:ok, result} = SubAgentServer.execute(subagent.id)
      assert result == "Done"

      # Should receive status and completion events
      assert_receive {:agent, {:debug, {:subagent, _, {:subagent_status_changed, :running}}}}, 100
      assert_receive {:agent, {:debug, {:subagent, _, {:subagent_completed, data}}}}, 100

      assert data.result == "Done"
      assert is_integer(data.duration_ms)
    end

    test "sub-agent error broadcasts via parent", context do
      parent_agent = create_test_agent()

      {:ok, _pid} =
        AgentServer.start_link(
          agent: parent_agent,
          pubsub: {Phoenix.PubSub, context.pubsub_name},
          debug_pubsub: {Phoenix.PubSub, context.debug_pubsub_name}
        )

      debug_topic = "agent_server:debug:#{parent_agent.agent_id}"
      Phoenix.PubSub.subscribe(context.debug_pubsub_name, debug_topic)

      subagent =
        SubAgent.new_from_config(
          parent_agent_id: parent_agent.agent_id,
          instructions: "Test task",
          agent_config: create_test_agent(),
          parent_state: State.new!(%{})
        )

      {:ok, _pid} = SubAgentServer.start_link(subagent: subagent)

      # Consume started event
      assert_receive {:agent, {:debug, {:subagent, _, {:subagent_started, _}}}}, 100

      # Mock LLM failure
      LLMChain
      |> stub(:run, fn chain -> {:error, chain, "API error"} end)

      # Execute subagent (should fail)
      {:error, reason} = SubAgentServer.execute(subagent.id)
      assert reason == "API error"

      # Should receive error event
      assert_receive {:agent, {:debug, {:subagent, _, {:subagent_status_changed, :running}}}}, 100
      assert_receive {:agent, {:debug, {:subagent, _, {:subagent_error, error}}}}, 100
      assert error == "API error"
    end
  end

  describe "late-connecting debugger" do
    test "can get current state after connecting late", context do
      agent = create_test_agent()

      {:ok, _pid} =
        AgentServer.start_link(
          agent: agent,
          pubsub: {Phoenix.PubSub, context.pubsub_name},
          debug_pubsub: {Phoenix.PubSub, context.debug_pubsub_name}
        )

      # Mock LLM response
      assistant_message = Message.new_assistant!(%{content: "First response"})
      LLMChain
      |> stub(:run, fn chain ->
        updated_chain =
          chain
          |> Map.put(:messages, chain.messages ++ [assistant_message])
          |> Map.put(:last_message, assistant_message)
          |> Map.put(:needs_response, false)

        {:ok, updated_chain}
      end)

      # Agent does some work BEFORE debugger connects
      AgentServer.add_message(agent.agent_id, Message.new_user!("Hello"))
      Process.sleep(200)

      # NOW connect debugger (late)
      debug_topic = "agent_server:debug:#{agent.agent_id}"
      Phoenix.PubSub.subscribe(context.debug_pubsub_name, debug_topic)

      # Can get current state (missed events are gone, but state is available)
      state = AgentServer.get_state(agent.agent_id)
      assert length(state.messages) > 0

      # Future events ARE received
      second_response = Message.new_assistant!(%{content: "Second response"})
      LLMChain
      |> stub(:run, fn chain ->
        updated_chain =
          chain
          |> Map.put(:messages, chain.messages ++ [second_response])
          |> Map.put(:last_message, second_response)
          |> Map.put(:needs_response, false)

        {:ok, updated_chain}
      end)

      AgentServer.add_message(agent.agent_id, Message.new_user!("Another message"))

      # Should receive events for this new activity
      assert_receive {:agent, {:debug, _event}}, 100
    end
  end

  describe "edge cases" do
    test "sub-agent shutdown broadcasts via terminate callback", context do
      parent_agent = create_test_agent()

      {:ok, _pid} =
        AgentServer.start_link(
          agent: parent_agent,
          pubsub: {Phoenix.PubSub, context.pubsub_name},
          debug_pubsub: {Phoenix.PubSub, context.debug_pubsub_name}
        )

      debug_topic = "agent_server:debug:#{parent_agent.agent_id}"
      Phoenix.PubSub.subscribe(context.debug_pubsub_name, debug_topic)

      subagent =
        SubAgent.new_from_config(
          parent_agent_id: parent_agent.agent_id,
          instructions: "Test task",
          agent_config: create_test_agent(),
          parent_state: State.new!(%{})
        )

      {:ok, pid} = SubAgentServer.start_link(subagent: subagent)

      # Consume started event
      assert_receive {:agent, {:debug, {:subagent, _, {:subagent_started, _}}}}, 100

      # Stop the subagent gracefully (triggers terminate with :normal reason)
      GenServer.stop(pid, :normal)
      Process.sleep(100)

      # :normal termination should NOT broadcast error event
      refute_receive {:agent, {:debug, {:subagent, _, {:subagent_error, _}}}}, 100
    end

    test "agent without presence_module still works", context do
      agent = create_test_agent()

      # Start without presence_module - this should work without errors
      {:ok, pid} =
        AgentServer.start_link(
          agent: agent,
          pubsub: {Phoenix.PubSub, context.pubsub_name}
          # No presence_module
        )

      # Agent should start and be accessible
      assert Process.alive?(pid)
      assert AgentServer.get_status(agent.agent_id) == :idle

      # Can get state
      state = AgentServer.get_state(agent.agent_id)
      assert state != nil

      # Can stop cleanly
      assert :ok = AgentServer.stop(agent.agent_id)
    end
  end
end
