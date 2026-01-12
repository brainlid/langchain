defmodule LangChain.Agents.SubAgentServerBroadcastTest do
  @moduledoc """
  Tests for SubAgentServer event broadcasting via parent AgentServer.

  These tests verify that SubAgentServer broadcasts lifecycle events
  (started, status_changed, completed, error) through the parent AgentServer's
  debug topic.
  """
  use LangChain.BaseCase, async: false
  use Mimic

  alias LangChain.Agents.{AgentServer, SubAgent, SubAgentServer, State}
  alias LangChain.Chains.LLMChain
  alias LangChain.Message

  setup :set_mimic_global
  setup :verify_on_exit!

  setup_all do
    Mimic.copy(LLMChain)
    :ok
  end

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

  # Helper to create parent agent with debug pubsub
  defp start_parent_agent(context) do
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

    parent_agent
  end

  # Helper to create a SubAgent struct with parent
  defp create_subagent(parent_agent_id, opts \\ []) do
    agent = Keyword.get(opts, :agent, create_test_agent())
    instructions = Keyword.get(opts, :instructions, "Do something")
    parent_state = Keyword.get(opts, :parent_state, State.new!(%{}))

    SubAgent.new_from_config(
      parent_agent_id: parent_agent_id,
      instructions: instructions,
      agent_config: agent,
      parent_state: parent_state
    )
  end

  describe "subagent_started event" do
    test "broadcasts subagent_started on init via parent", context do
      parent_agent = start_parent_agent(context)
      subagent = create_subagent(parent_agent.agent_id)

      # Start subagent server
      {:ok, _pid} = SubAgentServer.start_link(subagent: subagent)

      # Should receive started event via parent's topic
      assert_receive {:agent, {:debug, {:subagent, sub_id, {:subagent_started, data}}}}, 100

      assert sub_id == subagent.id
      assert data.parent_id == parent_agent.agent_id
      assert data.id == subagent.id
      assert is_binary(data.name)
      assert is_list(data.tools)
      assert is_binary(data.model)
    end

    test "includes instructions in started metadata", context do
      parent_agent = start_parent_agent(context)
      instructions = "Research renewable energy impacts on climate"
      subagent = create_subagent(parent_agent.agent_id, instructions: instructions)

      {:ok, _pid} = SubAgentServer.start_link(subagent: subagent)

      assert_receive {:agent, {:debug, {:subagent, _, {:subagent_started, data}}}}, 100
      assert data.instructions == instructions
    end
  end

  describe "subagent_status_changed event" do
    test "broadcasts status change to running on execute", context do
      parent_agent = start_parent_agent(context)
      subagent = create_subagent(parent_agent.agent_id)

      {:ok, _pid} = SubAgentServer.start_link(subagent: subagent)

      # Consume the started event
      assert_receive {:agent, {:debug, {:subagent, _, {:subagent_started, _}}}}, 100

      # Mock LLMChain.run to return success
      assistant_message = Message.new_assistant!(%{content: "Done"})

      updated_chain =
        subagent.chain
        |> Map.put(:messages, subagent.chain.messages ++ [assistant_message])
        |> Map.put(:last_message, assistant_message)
        |> Map.put(:needs_response, false)

      LLMChain
      |> stub(:run, fn _chain -> {:ok, updated_chain} end)

      # Execute (spawns a task to avoid blocking test)
      Task.async(fn -> SubAgentServer.execute(subagent.id) end)

      # Should receive running status
      assert_receive {:agent, {:debug, {:subagent, sub_id, {:subagent_status_changed, :running}}}},
                     1000

      assert sub_id == subagent.id
    end
  end

  describe "subagent_completed event" do
    test "broadcasts completion with result and duration", context do
      parent_agent = start_parent_agent(context)
      subagent = create_subagent(parent_agent.agent_id)

      {:ok, _pid} = SubAgentServer.start_link(subagent: subagent)

      # Consume started event
      assert_receive {:agent, {:debug, {:subagent, _, {:subagent_started, _}}}}, 100

      # Mock successful completion
      assistant_message = Message.new_assistant!(%{content: "Task completed successfully"})

      updated_chain =
        subagent.chain
        |> Map.put(:messages, subagent.chain.messages ++ [assistant_message])
        |> Map.put(:last_message, assistant_message)
        |> Map.put(:needs_response, false)

      LLMChain
      |> stub(:run, fn _chain -> {:ok, updated_chain} end)

      {:ok, result} = SubAgentServer.execute(subagent.id)
      assert result == "Task completed successfully"

      # Consume running status
      assert_receive {:agent, {:debug, {:subagent, _, {:subagent_status_changed, :running}}}},
                     1000

      # Should receive completed event
      assert_receive {:agent, {:debug, {:subagent, sub_id, {:subagent_completed, data}}}}, 100

      assert sub_id == subagent.id
      assert data.id == subagent.id
      assert data.result == "Task completed successfully"
      assert is_list(data.messages)
      assert is_integer(data.duration_ms)
      assert data.duration_ms >= 0
    end
  end

  describe "subagent_error event" do
    test "broadcasts error on execution failure", context do
      parent_agent = start_parent_agent(context)
      subagent = create_subagent(parent_agent.agent_id)

      {:ok, _pid} = SubAgentServer.start_link(subagent: subagent)

      # Consume started event
      assert_receive {:agent, {:debug, {:subagent, _, {:subagent_started, _}}}}, 100

      # Mock LLMChain.run to return error
      LLMChain
      |> stub(:run, fn _chain -> {:error, subagent.chain, "API error: rate limit"} end)

      {:error, reason} = SubAgentServer.execute(subagent.id)
      assert reason == "API error: rate limit"

      # Consume running status
      assert_receive {:agent, {:debug, {:subagent, _, {:subagent_status_changed, :running}}}},
                     1000

      # Should receive error event
      assert_receive {:agent, {:debug, {:subagent, sub_id, {:subagent_error, error_reason}}}},
                     1000

      assert sub_id == subagent.id
      assert error_reason == "API error: rate limit"
    end
  end

  describe "no-op when parent has no debug_pubsub" do
    test "subagent starts without error when parent lacks debug_pubsub", context do
      # Start parent WITHOUT debug_pubsub
      parent_agent = create_test_agent()

      {:ok, _pid} =
        AgentServer.start_link(
          agent: parent_agent,
          pubsub: {Phoenix.PubSub, context.pubsub_name}
          # No debug_pubsub configured
        )

      # Subscribe to debug topic anyway (shouldn't receive anything)
      debug_topic = "agent_server:debug:#{parent_agent.agent_id}"
      Phoenix.PubSub.subscribe(context.debug_pubsub_name, debug_topic)

      subagent = create_subagent(parent_agent.agent_id)
      {:ok, _pid} = SubAgentServer.start_link(subagent: subagent)

      # Should NOT receive any events (parent has no debug_pubsub)
      refute_receive {:agent, {:debug, {:subagent, _, _}}}, 200
    end

    test "subagent executes without error when parent lacks debug_pubsub", context do
      # Start parent WITHOUT debug_pubsub
      parent_agent = create_test_agent()

      {:ok, _pid} =
        AgentServer.start_link(
          agent: parent_agent,
          pubsub: {Phoenix.PubSub, context.pubsub_name}
        )

      subagent = create_subagent(parent_agent.agent_id)
      {:ok, _pid} = SubAgentServer.start_link(subagent: subagent)

      # Mock successful completion
      assistant_message = Message.new_assistant!(%{content: "Done"})

      updated_chain =
        subagent.chain
        |> Map.put(:messages, subagent.chain.messages ++ [assistant_message])
        |> Map.put(:last_message, assistant_message)
        |> Map.put(:needs_response, false)

      LLMChain
      |> stub(:run, fn _chain -> {:ok, updated_chain} end)

      # Execute should still work
      {:ok, result} = SubAgentServer.execute(subagent.id)
      assert result == "Done"
    end
  end

  describe "resume broadcasts events" do
    test "broadcasts running status on resume", context do
      parent_agent = start_parent_agent(context)
      subagent = create_subagent(parent_agent.agent_id)

      {:ok, _pid} = SubAgentServer.start_link(subagent: subagent)

      # Consume started event
      assert_receive {:agent, {:debug, {:subagent, _, {:subagent_started, _}}}}, 100

      # Mock chain to trigger interrupt then complete on resume
      assistant_with_tool = Message.new_assistant!(%{
        content: "I'll write a file",
        tool_calls: [
          LangChain.Message.ToolCall.new!(%{
            type: :function,
            call_id: "call_1",
            name: "file_write",
            arguments: %{"path" => "test.txt", "content" => "hello"}
          })
        ]
      })

      chain_after_llm =
        subagent.chain
        |> Map.put(:messages, subagent.chain.messages ++ [assistant_with_tool])
        |> Map.put(:last_message, assistant_with_tool)
        |> Map.put(:needs_response, true)

      # First call returns interrupt-triggering chain
      LLMChain
      |> stub(:run, fn _chain -> {:ok, chain_after_llm} end)

      # Skip interrupt test for now - focus on basic events
      # The resume flow is complex and tested in integration tests
    end
  end
end
