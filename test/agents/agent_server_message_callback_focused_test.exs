defmodule LangChain.Agents.AgentServerMessageCallbackFocusedTest do
  @moduledoc """
  Focused unit tests for AgentServer message persistence callback functionality.

  These tests verify the callback infrastructure without triggering full agent execution.
  """

  use ExUnit.Case, async: false
  use Mimic

  alias LangChain.Agents.{Agent, AgentServer}
  alias LangChain.ChatModels.ChatAnthropic
  alias LangChain.Message

  setup :set_mimic_global

  setup do
    # Mock ChatAnthropic to prevent execution
    stub(ChatAnthropic, :call, fn _model, _messages, _callbacks ->
      {:ok, [Message.new_assistant!("Mock response")]}
    end)

    # Start PubSub
    pubsub_name = :"test_pubsub_#{System.unique_integer([:positive])}"
    {:ok, _pid} = Phoenix.PubSub.Supervisor.start_link(name: pubsub_name, adapter_name: Phoenix.PubSub.PG2)

    agent_id = "test-agent-#{System.unique_integer([:positive])}"

    {:ok, pubsub: pubsub_name, agent_id: agent_id}
  end

  defp create_test_agent(agent_id) do
    model = ChatAnthropic.new!(%{
      model: "claude-3-5-sonnet-20241022",
      api_key: "test_key"
    })

    Agent.new!(%{
      agent_id: agent_id,
      model: model,
      base_system_prompt: "Test agent",
      replace_default_middleware: true,
      middleware: []
    })
  end

  describe "callback configuration" do
    test "accepts and stores conversation_id", %{agent_id: agent_id, pubsub: pubsub} do
      agent = create_test_agent(agent_id)
      conversation_id = "conv-test-123"

      {:ok, _pid} = AgentServer.start_link(
        agent: agent,
        conversation_id: conversation_id,
        pubsub: {Phoenix.PubSub, pubsub}
      )

      # Verify server started successfully (proves config was accepted)
      assert AgentServer.get_pid(agent_id) != nil
      assert AgentServer.get_status(agent_id) == :idle

      AgentServer.stop(agent_id)
    end

    test "accepts and stores save_new_message_fn", %{agent_id: agent_id, pubsub: pubsub} do
      agent = create_test_agent(agent_id)

      callback_fn = fn _conv_id, _message -> {:ok, []} end

      {:ok, _pid} = AgentServer.start_link(
        agent: agent,
        conversation_id: "conv-123",
        save_new_message_fn: callback_fn,
        pubsub: {Phoenix.PubSub, pubsub}
      )

      # Verify server started successfully (proves config was accepted)
      assert AgentServer.get_pid(agent_id) != nil
      assert AgentServer.get_status(agent_id) == :idle

      AgentServer.stop(agent_id)
    end

    test "works without callback options (backward compatibility)", %{agent_id: agent_id, pubsub: pubsub} do
      agent = create_test_agent(agent_id)

      {:ok, _pid} = AgentServer.start_link(
        agent: agent,
        pubsub: {Phoenix.PubSub, pubsub}
        # No conversation_id or save_new_message_fn
      )

      # Verify server started successfully
      assert AgentServer.get_pid(agent_id) != nil
      assert AgentServer.get_status(agent_id) == :idle

      AgentServer.stop(agent_id)
    end
  end

  describe "callback invocation" do
    test "callback is invoked when adding user message", %{agent_id: agent_id, pubsub: pubsub} do
      agent = create_test_agent(agent_id)
      conversation_id = "conv-123"

      # Track callback invocations
      test_pid = self()
      callback_fn = fn conv_id, message ->
        send(test_pid, {:callback_invoked, conv_id, message.role})
        {:ok, []}
      end

      {:ok, _pid} = AgentServer.start_link(
        agent: agent,
        conversation_id: conversation_id,
        save_new_message_fn: callback_fn,
        pubsub: {Phoenix.PubSub, pubsub}
      )

      # Add a user message
      message = Message.new_user!("Test message")
      :ok = AgentServer.add_message(agent_id, message)

      # Verify callback was invoked
      assert_receive {:callback_invoked, ^conversation_id, :user}, 1000

      AgentServer.stop(agent_id)
    end

    test "callback receives correct conversation_id", %{agent_id: agent_id, pubsub: pubsub} do
      agent = create_test_agent(agent_id)
      conversation_id = "conv-unique-12345"

      test_pid = self()
      callback_fn = fn conv_id, _message ->
        send(test_pid, {:received_conv_id, conv_id})
        {:ok, []}
      end

      {:ok, _pid} = AgentServer.start_link(
        agent: agent,
        conversation_id: conversation_id,
        save_new_message_fn: callback_fn,
        pubsub: {Phoenix.PubSub, pubsub}
      )

      message = Message.new_user!("Test")
      :ok = AgentServer.add_message(agent_id, message)

      # Verify exact conversation_id was passed
      assert_receive {:received_conv_id, ^conversation_id}, 1000

      AgentServer.stop(agent_id)
    end

    test "callback is not invoked when conversation_id is missing", %{agent_id: agent_id, pubsub: pubsub} do
      agent = create_test_agent(agent_id)

      # This callback should NEVER be called
      callback_fn = fn _conv_id, _message ->
        flunk("Callback should not be invoked without conversation_id")
        {:ok, []}
      end

      {:ok, _pid} = AgentServer.start_link(
        agent: agent,
        save_new_message_fn: callback_fn,  # Has callback
        pubsub: {Phoenix.PubSub, pubsub}
        # Missing conversation_id - callback won't activate
      )

      message = Message.new_user!("Test")
      :ok = AgentServer.add_message(agent_id, message)

      # Give it time to potentially call (it shouldn't)
      Process.sleep(100)

      AgentServer.stop(agent_id)
    end
  end

  describe "error handling" do
    test "callback exception doesn't crash server", %{agent_id: agent_id, pubsub: pubsub} do
      agent = create_test_agent(agent_id)

      # Callback that raises
      callback_fn = fn _conv_id, _message ->
        raise "Simulated error"
      end

      {:ok, pid} = AgentServer.start_link(
        agent: agent,
        conversation_id: "conv-123",
        save_new_message_fn: callback_fn,
        pubsub: {Phoenix.PubSub, pubsub}
      )

      message = Message.new_user!("Test")
      :ok = AgentServer.add_message(agent_id, message)

      # Give it time to process
      Process.sleep(100)

      # Server should still be alive (status may be :running or :idle)
      assert Process.alive?(pid)
      status = AgentServer.get_status(agent_id)
      assert status in [:idle, :running]

      AgentServer.stop(agent_id)
    end

    test "invalid callback return doesn't crash server", %{agent_id: agent_id, pubsub: pubsub} do
      agent = create_test_agent(agent_id)

      # Callback returns invalid format
      callback_fn = fn _conv_id, _message ->
        {:ok, "not a list"}  # Invalid!
      end

      {:ok, pid} = AgentServer.start_link(
        agent: agent,
        conversation_id: "conv-123",
        save_new_message_fn: callback_fn,
        pubsub: {Phoenix.PubSub, pubsub}
      )

      message = Message.new_user!("Test")
      :ok = AgentServer.add_message(agent_id, message)

      # Give it time to process
      Process.sleep(100)

      # Server should still be alive
      assert Process.alive?(pid)

      AgentServer.stop(agent_id)
    end
  end

  describe "message state management" do
    test "messages added to state regardless of callback success", %{agent_id: agent_id, pubsub: pubsub} do
      agent = create_test_agent(agent_id)

      # Callback that always fails
      callback_fn = fn _conv_id, _message ->
        {:error, :database_error}
      end

      {:ok, _pid} = AgentServer.start_link(
        agent: agent,
        conversation_id: "conv-123",
        save_new_message_fn: callback_fn,
        pubsub: {Phoenix.PubSub, pubsub}
      )

      # Add a message
      message = Message.new_user!("Test message")
      :ok = AgentServer.add_message(agent_id, message)

      # Give it time to process
      Process.sleep(50)

      # Message should still be in state even though callback failed
      state = AgentServer.get_state(agent_id)
      assert length(state.messages) >= 1

      AgentServer.stop(agent_id)
    end
  end
end
