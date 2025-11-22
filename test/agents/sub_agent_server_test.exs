defmodule LangChain.Agents.SubAgentServerTest do
  use ExUnit.Case, async: false
  use Mimic

  alias LangChain.Agents.{Agent, SubAgent, SubAgentServer, State}
  alias LangChain.ChatModels.ChatAnthropic
  alias LangChain.Chains.LLMChain
  alias LangChain.Message

  setup :set_mimic_global
  setup :verify_on_exit!

  setup_all do
    # Copy modules for mocking
    Mimic.copy(LLMChain)
    :ok
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
    agent_id = Keyword.get(opts, :agent_id, "test-subagent-#{System.unique_integer([:positive])}")
    opts = Keyword.put(opts, :agent_id, agent_id)

    Agent.new!(
      Map.merge(
        %{
          model: mock_model(),
          system_prompt: "Test subagent",
          replace_default_middleware: true,
          middleware: []
        },
        Enum.into(opts, %{})
      )
    )
  end

  # Helper to create a SubAgent struct
  defp create_subagent(opts \\ []) do
    agent = Keyword.get(opts, :agent, create_test_agent())
    parent_agent_id = Keyword.get(opts, :parent_agent_id, "main-agent")
    instructions = Keyword.get(opts, :instructions, "Do something")
    parent_state = Keyword.get(opts, :parent_state, State.new!(%{}))

    SubAgent.new_from_config(
      parent_agent_id: parent_agent_id,
      instructions: instructions,
      agent_config: agent,
      parent_state: parent_state
    )
  end

  describe "start_link/1" do
    test "starts server with subagent struct and starts with idle status" do
      subagent = create_subagent()

      assert {:ok, pid} = SubAgentServer.start_link(subagent: subagent)

      assert Process.alive?(pid)

      # Verify status is accessible
      assert SubAgentServer.get_status(subagent.id) == :idle
    end
  end

  describe "whereis/1" do
    test "returns pid of running SubAgent" do
      subagent = create_subagent()

      {:ok, pid} = SubAgentServer.start_link(subagent: subagent)

      assert SubAgentServer.whereis(subagent.id) == pid
    end

    test "returns nil for non-existent SubAgent" do
      assert SubAgentServer.whereis("nonexistent-sub-agent") == nil
    end
  end

  describe "execute/1 - success case" do
    test "returns {:ok, result} when chain completes successfully" do
      agent = create_test_agent()
      subagent = create_subagent(agent: agent)

      {:ok, _pid} = SubAgentServer.start_link(subagent: subagent)

      # Mock LLMChain.run to return success with updated chain
      assistant_message = Message.new_assistant!(%{content: "Task done"})

      updated_chain =
        subagent.chain
        |> Map.put(:messages, subagent.chain.messages ++ [assistant_message])
        |> Map.put(:last_message, assistant_message)
        |> Map.put(:needs_response, false)

      LLMChain
      |> stub(:run, fn _chain ->
        {:ok, updated_chain}
      end)

      # Should extract final message content as string
      assert {:ok, result} = SubAgentServer.execute(subagent.id)
      assert is_binary(result)
      assert result == "Task done"
      assert SubAgentServer.get_status(subagent.id) == :completed
    end

    test "updates status from idle to completed" do
      agent = create_test_agent()
      subagent = create_subagent(agent: agent)

      {:ok, _pid} = SubAgentServer.start_link(subagent: subagent)

      # Verify initial status
      assert SubAgentServer.get_status(subagent.id) == :idle

      # Mock LLMChain.run
      assistant_message = Message.new_assistant!(%{content: "Done"})

      updated_chain =
        subagent.chain
        |> Map.put(:messages, subagent.chain.messages ++ [assistant_message])
        |> Map.put(:last_message, assistant_message)
        |> Map.put(:needs_response, false)

      LLMChain
      |> expect(:run, fn _chain ->
        {:ok, updated_chain}
      end)

      SubAgentServer.execute(subagent.id)

      # Verify final status
      assert SubAgentServer.get_status(subagent.id) == :completed
    end
  end

  describe "execute/1 - error case" do
    test "returns {:error, reason} when chain fails" do
      agent = create_test_agent()
      subagent = create_subagent(agent: agent)

      {:ok, _pid} = SubAgentServer.start_link(subagent: subagent)

      # Mock LLMChain.run to return error with correct format
      LLMChain
      |> expect(:run, fn chain ->
        {:error, chain, :something_went_wrong}
      end)

      assert {:error, :something_went_wrong} = SubAgentServer.execute(subagent.id)
      assert SubAgentServer.get_status(subagent.id) == :error
    end
  end

  describe "get_status/1" do
    test "returns current status" do
      subagent = create_subagent()

      {:ok, _pid} = SubAgentServer.start_link(subagent: subagent)

      assert SubAgentServer.get_status(subagent.id) == :idle
    end
  end

  describe "get_subagent/1" do
    test "returns current subagent struct" do
      subagent = create_subagent()

      {:ok, _pid} = SubAgentServer.start_link(subagent: subagent)

      retrieved_subagent = SubAgentServer.get_subagent(subagent.id)
      assert %SubAgent{} = retrieved_subagent
      assert retrieved_subagent.id == subagent.id
      assert retrieved_subagent.status == :idle
    end
  end

  describe "resume/2 - success case" do
    test "returns {:ok, result} when chain completes after resume" do
      agent = create_test_agent()
      subagent = create_subagent(agent: agent)

      # Manually set status to interrupted to simulate an interrupt
      interrupted_subagent = %{
        subagent
        | status: :interrupted,
          interrupt_data: %{action_requests: [], hitl_tool_call_ids: []}
      }

      {:ok, _pid} = SubAgentServer.start_link(subagent: interrupted_subagent)

      # Mock LLMChain.execute_tool_calls_with_decisions - returns chain with tool results
      LLMChain
      |> stub(:execute_tool_calls_with_decisions, fn chain, _tool_calls, _decisions ->
        chain
      end)

      # Mock LLMChain.run to return success with updated chain
      assistant_message = Message.new_assistant!(%{content: "Resumed and completed"})

      updated_chain =
        interrupted_subagent.chain
        |> Map.put(:messages, interrupted_subagent.chain.messages ++ [assistant_message])
        |> Map.put(:last_message, assistant_message)
        |> Map.put(:needs_response, false)

      LLMChain
      |> stub(:run, fn _chain ->
        {:ok, updated_chain}
      end)

      decisions = [%{type: :approve}]
      assert {:ok, result} = SubAgentServer.resume(interrupted_subagent.id, decisions)
      assert is_binary(result)
      assert result == "Resumed and completed"
      assert SubAgentServer.get_status(interrupted_subagent.id) == :completed
    end

    test "updates status from interrupted to completed" do
      agent = create_test_agent()
      subagent = create_subagent(agent: agent)

      # Set status to interrupted
      interrupted_subagent = %{
        subagent
        | status: :interrupted,
          interrupt_data: %{action_requests: [], hitl_tool_call_ids: []}
      }

      {:ok, _pid} = SubAgentServer.start_link(subagent: interrupted_subagent)

      # Verify initial status
      assert SubAgentServer.get_status(interrupted_subagent.id) == :interrupted

      # Mock LLMChain.execute_tool_calls_with_decisions
      LLMChain
      |> stub(:execute_tool_calls_with_decisions, fn chain, _tool_calls, _decisions ->
        chain
      end)

      # Mock LLMChain.run
      assistant_message = Message.new_assistant!(%{content: "Done"})

      updated_chain =
        interrupted_subagent.chain
        |> Map.put(:messages, interrupted_subagent.chain.messages ++ [assistant_message])
        |> Map.put(:last_message, assistant_message)
        |> Map.put(:needs_response, false)

      LLMChain
      |> stub(:run, fn _chain ->
        {:ok, updated_chain}
      end)

      SubAgentServer.resume(interrupted_subagent.id, [%{type: :approve}])

      # Verify final status
      assert SubAgentServer.get_status(interrupted_subagent.id) == :completed
    end
  end

  describe "resume/2 - error case" do
    test "returns {:error, reason} when resume fails" do
      agent = create_test_agent()
      subagent = create_subagent(agent: agent)

      # Set status to interrupted
      interrupted_subagent = %{
        subagent
        | status: :interrupted,
          interrupt_data: %{action_requests: [], hitl_tool_call_ids: []}
      }

      {:ok, _pid} = SubAgentServer.start_link(subagent: interrupted_subagent)

      # Mock LLMChain.execute_tool_calls_with_decisions
      LLMChain
      |> stub(:execute_tool_calls_with_decisions, fn chain, _tool_calls, _decisions ->
        chain
      end)

      # Mock LLMChain.run to return error with correct format
      LLMChain
      |> expect(:run, fn chain ->
        {:error, chain, :resume_failed}
      end)

      decisions = [%{type: :approve}]
      assert {:error, :resume_failed} = SubAgentServer.resume(interrupted_subagent.id, decisions)
      assert SubAgentServer.get_status(interrupted_subagent.id) == :error
    end

    test "returns error when called on non-interrupted status" do
      agent = create_test_agent()
      subagent = create_subagent(agent: agent)

      {:ok, _pid} = SubAgentServer.start_link(subagent: subagent)

      # Subagent is in :idle status, not :interrupted
      decisions = [%{type: :approve}]

      assert {:error, {:invalid_status, :idle, :expected_interrupted}} =
               SubAgentServer.resume(subagent.id, decisions)
    end
  end

  describe "blocking behavior" do
    test "execute blocks until completion" do
      agent = create_test_agent()
      subagent = create_subagent(agent: agent)

      {:ok, _pid} = SubAgentServer.start_link(subagent: subagent)

      # Mock LLMChain.run with simulated delay
      assistant_message = Message.new_assistant!(%{content: "Done"})

      updated_chain =
        subagent.chain
        |> Map.put(:messages, subagent.chain.messages ++ [assistant_message])
        |> Map.put(:last_message, assistant_message)
        |> Map.put(:needs_response, false)

      LLMChain
      |> expect(:run, fn _chain ->
        # Simulate LLM calls and work
        Process.sleep(100)
        {:ok, updated_chain}
      end)

      # This should block for at least 100ms
      start_time = System.monotonic_time(:millisecond)
      assert {:ok, result} = SubAgentServer.execute(subagent.id)
      elapsed = System.monotonic_time(:millisecond) - start_time

      # Verify it actually blocked
      assert elapsed >= 100
      assert is_binary(result)
      assert result == "Done"
    end

    test "resume blocks until completion" do
      agent = create_test_agent()
      subagent = create_subagent(agent: agent)

      # Set status to interrupted
      interrupted_subagent = %{
        subagent
        | status: :interrupted,
          interrupt_data: %{action_requests: [], hitl_tool_call_ids: []}
      }

      {:ok, _pid} = SubAgentServer.start_link(subagent: interrupted_subagent)

      # Mock LLMChain.execute_tool_calls_with_decisions
      LLMChain
      |> stub(:execute_tool_calls_with_decisions, fn chain, _tool_calls, _decisions ->
        chain
      end)

      # Mock LLMChain.run with simulated delay
      assistant_message = Message.new_assistant!(%{content: "Done"})

      updated_chain =
        interrupted_subagent.chain
        |> Map.put(:messages, interrupted_subagent.chain.messages ++ [assistant_message])
        |> Map.put(:last_message, assistant_message)
        |> Map.put(:needs_response, false)

      LLMChain
      |> stub(:run, fn _chain ->
        # Simulate work during resume
        Process.sleep(100)
        {:ok, updated_chain}
      end)

      # This should block for at least 100ms
      start_time = System.monotonic_time(:millisecond)
      assert {:ok, result} = SubAgentServer.resume(interrupted_subagent.id, [%{type: :approve}])
      elapsed = System.monotonic_time(:millisecond) - start_time

      # Verify it actually blocked
      assert elapsed >= 100
      assert is_binary(result)
      assert result == "Done"
    end
  end
end
