defmodule LangChain.Agents.AgentServerMessageCallbackTest do
  @moduledoc """
  Tests for AgentServer message persistence callback functionality.

  This tests the feature where AgentServer:
  1. Accepts a `conversation_id` and `save_new_message_fn` callback
  2. Calls the callback when new messages are added
  3. Broadcasts `{:display_message_saved, display_msg}` events for saved messages
  4. Also broadcasts `{:llm_message, message}` events (both with and without callback)
  5. Does not broadcast message events when callback fails or raises exceptions
  """

  use LangChain.BaseCase, async: false
  use Mimic

  alias LangChain.Agents.AgentServer
  alias LangChain.ChatModels.ChatAnthropic
  alias LangChain.Message
  alias LangChain.TestingHelpers

  # Make mocks global since agent execution happens in a Task
  setup :set_mimic_global

  setup do
    # Mock ChatAnthropic.call to prevent real API calls
    stub(ChatAnthropic, :call, fn _model, _messages, _callbacks ->
      {:ok, [Message.new_assistant!("Mock response")]}
    end)

    :ok
  end

  describe "message persistence callback configuration" do
    test "accepts conversation_id and save_new_message_fn options" do
      agent_id = TestingHelpers.generate_test_agent_id()
      conversation_id = "conv-123"

      # Define a callback that tracks calls and returns display messages
      test_pid = self()

      callback_fn = fn conv_id, message ->
        send(test_pid, {:callback_called, conv_id, message.role})
        # Return a display message to trigger broadcast
        display_msg = %{
          id: System.unique_integer([:positive]),
          content: "Saved: #{inspect(message.role)}",
          content_type: "text",
          role: Atom.to_string(message.role)
        }

        {:ok, [display_msg]}
      end

      # Start agent with callback configuration using helper
      {:ok, %{agent_id: ^agent_id}} =
        TestingHelpers.start_test_agent(
          agent_id: agent_id,
          pubsub: {Phoenix.PubSub, :test_pubsub},
          conversation_id: conversation_id,
          save_new_message_fn: callback_fn
        )

      # Add a message
      message = Message.new_user!("Test message")
      :ok = AgentServer.add_message(agent_id, message)

      # Verify callback was called for user message
      assert_receive {:callback_called, ^conversation_id, :user}, 100

      # Verify display_message_saved event was broadcast for user message
      assert_receive {:display_message_saved, user_display_msg}, 100
      assert user_display_msg.role == "user"
      assert user_display_msg.content =~ "user"

      # Agent executes and generates assistant response
      # Verify callback called for assistant message
      assert_receive {:callback_called, ^conversation_id, :assistant}, 100

      # Verify display_message_saved event was broadcast for assistant message
      assert_receive {:display_message_saved, assistant_display_msg}, 100
      assert assistant_display_msg.role == "assistant"
      assert assistant_display_msg.content =~ "assistant"

      # Cleanup
      TestingHelpers.stop_test_agent(agent_id)
    end

    test "works without callback configuration (fallback mode)" do
      agent_id = TestingHelpers.generate_test_agent_id()
      # Start agent WITHOUT callback configuration
      {:ok, %{agent_id: ^agent_id}} =
        TestingHelpers.start_test_agent(
          agent_id: agent_id,
          pubsub: {Phoenix.PubSub, :test_pubsub}
          # No conversation_id or save_new_message_fn
        )

      # Add a message
      message = Message.new_user!("Test message")
      :ok = AgentServer.add_message(agent_id, message)

      # Should receive :llm_message event (fallback) for both user and assistant
      assert_receive {:llm_message, user_msg}, 100
      assert user_msg.role == :user

      assert_receive {:llm_message, assistant_msg}, 100
      assert assistant_msg.role == :assistant

      # Cleanup
      TestingHelpers.stop_test_agent(agent_id)
    end

    test "requires both conversation_id and save_new_message_fn for callback to activate" do
      agent_id = TestingHelpers.generate_test_agent_id()
      # Callback that should NOT be called
      callback_fn = fn _conv_id, _message ->
        flunk("Callback should not be called when conversation_id is missing")
        {:ok, []}
      end

      # Start with callback but NO conversation_id
      {:ok, %{agent_id: ^agent_id}} =
        TestingHelpers.start_test_agent(
          agent_id: agent_id,
          pubsub: {Phoenix.PubSub, :test_pubsub},
          save_new_message_fn: callback_fn
          # Missing conversation_id
        )

      # Add a message
      message = Message.new_user!("Test message")
      :ok = AgentServer.add_message(agent_id, message)

      # Should fallback to :llm_message (callback not activated) for both messages
      assert_receive {:llm_message, user_msg}, 100
      assert user_msg.role == :user

      assert_receive {:llm_message, assistant_msg}, 100
      assert assistant_msg.role == :assistant

      # Cleanup
      TestingHelpers.stop_test_agent(agent_id)
    end
  end

  describe "callback invocation and PubSub broadcasting" do
    test "broadcasts {:display_message_saved} events for each saved display message" do
      agent_id = TestingHelpers.generate_test_agent_id()
      conversation_id = "conv-123"

      {:ok, %{agent_id: ^agent_id}} =
        TestingHelpers.start_test_agent(
          agent_id: agent_id,
          pubsub: {Phoenix.PubSub, :test_pubsub},
          conversation_id: conversation_id,
          save_new_message_fn: &TestingHelpers.basic_process_to_display_data/2
        )

      # Add a message
      message = Message.new_user!("Test")
      :ok = AgentServer.add_message(agent_id, message)

      # First receives the user message events
      assert_receive {:display_message_saved, %{role: "user", content: "Test"}}, 100
      assert_receive {:llm_message, user_msg}, 100
      assert user_msg.role == :user

      # Then receives the assistant message events
      assert_receive {:display_message_saved, assistant_display_msg}, 100
      assert assistant_display_msg.role == "assistant"
      assert assistant_display_msg.content == "Mock response"

      assert_receive {:llm_message, assistant_msg}, 100
      assert assistant_msg.role == :assistant
      assert assistant_msg.content == [%LangChain.Message.ContentPart{type: :text, content: "Mock response"}]

      # Cleanup
      TestingHelpers.stop_test_agent(agent_id)
    end

    test "handles callback returning error without broadcasting" do
      agent_id = TestingHelpers.generate_test_agent_id()
      conversation_id = "conv-123"

      # Callback that returns error
      failing_callback_fn = fn _conv_id, _message ->
        {:error, :database_connection_failed}
      end

      {:ok, %{agent_id: ^agent_id}} =
        TestingHelpers.start_test_agent(
          agent_id: agent_id,
          pubsub: {Phoenix.PubSub, :test_pubsub},
          conversation_id: conversation_id,
          save_new_message_fn: failing_callback_fn
        )

      # Add a message
      message = Message.new_user!("Test")
      :ok = AgentServer.add_message(agent_id, message)

      # Should NOT broadcast any message events when callback fails
      refute_receive {:llm_message, _}, 500
      refute_receive {:display_message_saved, _}, 500

      # Cleanup
      TestingHelpers.stop_test_agent(agent_id)
    end

    test "handles callback returning empty list" do
      agent_id = TestingHelpers.generate_test_agent_id()
      conversation_id = "conv-123"

      # Track callback calls
      test_pid = self()

      callback_fn = fn _conv_id, _message ->
        send(test_pid, :callback_called)
        # Empty list - no displayable content
        {:ok, []}
      end

      {:ok, %{agent_id: ^agent_id}} =
        TestingHelpers.start_test_agent(
          agent_id: agent_id,
          pubsub: {Phoenix.PubSub, :test_pubsub},
          conversation_id: conversation_id,
          save_new_message_fn: callback_fn
        )

      # Add a message
      message = Message.new_user!("Test")
      :ok = AgentServer.add_message(agent_id, message)

      # Callback should be called
      assert_receive :callback_called, 100

      # Should not broadcast display_message_saved (empty list)
      refute_receive {:display_message_saved, _}, 100

      # Should still broadcast :llm_message event (callback succeeded, even if empty)
      assert_receive {:llm_message, llm_msg}, 100
      assert llm_msg.role == :user

      # Cleanup
      TestingHelpers.stop_test_agent(agent_id)
    end
  end

  describe "callback with different message types" do
    test "calls callback for user messages" do
      agent_id = TestingHelpers.generate_test_agent_id()
      conversation_id = "conv-123"

      test_pid = self()

      callback_fn = fn _conv_id, message ->
        send(test_pid, {:callback_message_role, message.role})
        {:ok, [%{id: 1, content_type: "text", role: "user", content: "Saved"}]}
      end

      {:ok, %{agent_id: ^agent_id}} =
        TestingHelpers.start_test_agent(
          agent_id: agent_id,
          pubsub: {Phoenix.PubSub, :test_pubsub},
          conversation_id: conversation_id,
          save_new_message_fn: callback_fn
        )

      # Add user message
      :ok = AgentServer.add_message(agent_id, Message.new_user!("Hello"))

      assert_receive {:callback_message_role, :user}, 100
      assert_receive {:display_message_saved, _}, 100

      # Cleanup
      TestingHelpers.stop_test_agent(agent_id)
    end

    test "calls callback for assistant messages" do
      agent_id = TestingHelpers.generate_test_agent_id()
      conversation_id = "conv-123"

      test_pid = self()

      callback_fn = fn _conv_id, message ->
        send(test_pid, {:callback_message_role, message.role})
        {:ok, [%{id: 2, content_type: "text", role: "assistant", content: "Saved"}]}
      end

      {:ok, %{agent_id: ^agent_id}} =
        TestingHelpers.start_test_agent(
          agent_id: agent_id,
          pubsub: {Phoenix.PubSub, :test_pubsub},
          conversation_id: conversation_id,
          save_new_message_fn: callback_fn
        )

      # Add assistant message
      :ok = AgentServer.add_message(agent_id, Message.new_assistant!(%{content: "Response"}))

      assert_receive {:callback_message_role, :assistant}, 100
      assert_receive {:display_message_saved, _}, 100

      # Cleanup
      TestingHelpers.stop_test_agent(agent_id)
    end
  end

  describe "integration with AgentServer.add_message" do
    test "callback is invoked when add_message is called" do
      agent_id = TestingHelpers.generate_test_agent_id()
      conversation_id = "conv-123"

      # Track all callback invocations
      test_pid = self()

      callback_fn = fn conv_id, message ->
        send(test_pid, {:callback_invoked, conv_id, message})
        {:ok, [%{content_type: "text", role: "user", content: "Saved"}]}
      end

      {:ok, %{agent_id: ^agent_id}} =
        TestingHelpers.start_test_agent(
          agent_id: agent_id,
          pubsub: {Phoenix.PubSub, :test_pubsub},
          conversation_id: conversation_id,
          save_new_message_fn: callback_fn
        )

      # Add first message
      msg1 = Message.new_user!("Message 1")
      :ok = AgentServer.add_message(agent_id, msg1)

      # Verify callback called for first message
      assert_receive {:callback_invoked, ^conversation_id, user_msg1}, 100
      assert user_msg1.role == :user
      # Verify we got back the assistant response
      assert_receive {:callback_invoked, ^conversation_id, ai_msg1}, 100
      assert ai_msg1.role == :assistant

      # brief pause before sending next message (async)
      Process.sleep(50)

      # Add second message
      msg2 = Message.new_user!("Message 2")
      :ok = AgentServer.add_message(agent_id, msg2)

      # Verify callback called for second message
      assert_receive {:callback_invoked, ^conversation_id, user_msg2}, 100
      assert user_msg2.role == :user
      # Verify we got back the assistant response
      assert_receive {:callback_invoked, ^conversation_id, ai_msg2}, 100
      assert ai_msg2.role == :assistant

      # Cleanup
      TestingHelpers.stop_test_agent(agent_id)
    end

    test "messages are added to state regardless of callback success" do
      agent_id = TestingHelpers.generate_test_agent_id()
      conversation_id = "conv-123"

      # Callback that fails
      callback_fn = fn _conv_id, _message ->
        {:error, :persistence_failed}
      end

      {:ok, %{agent_id: ^agent_id}} =
        TestingHelpers.start_test_agent(
          agent_id: agent_id,
          pubsub: {Phoenix.PubSub, :test_pubsub},
          conversation_id: conversation_id,
          save_new_message_fn: callback_fn
        )

      # Add messages
      :ok = AgentServer.add_message(agent_id, Message.new_user!("Message 1"))

      # brief pause before sending next message (async)
      Process.sleep(50)

      :ok = AgentServer.add_message(agent_id, Message.new_user!("Message 2"))
      # brief pause for processing second message
      Process.sleep(50)

      # Messages should still be in state (includes the two assistant responses)
      state = AgentServer.get_state(agent_id)
      assert length(state.messages) == 4

      # Cleanup
      TestingHelpers.stop_test_agent(agent_id)
    end
  end

  describe "conversation_id propagation" do
    test "passes correct conversation_id to callback" do
      agent_id = TestingHelpers.generate_test_agent_id()
      conversation_id = "conv-unique-12345"

      test_pid = self()

      callback_fn = fn conv_id, _message ->
        send(test_pid, {:received_conversation_id, conv_id})
        {:ok, []}
      end

      {:ok, %{agent_id: ^agent_id}} =
        TestingHelpers.start_test_agent(
          agent_id: agent_id,
          pubsub: {Phoenix.PubSub, :test_pubsub},
          conversation_id: conversation_id,
          save_new_message_fn: callback_fn
        )

      # Add message
      :ok = AgentServer.add_message(agent_id, Message.new_user!("Test"))

      # Verify exact conversation_id was passed
      assert_receive {:received_conversation_id, ^conversation_id}, 100

      # Cleanup
      TestingHelpers.stop_test_agent(agent_id)
    end

    test "different agents with different conversation_ids call callbacks correctly" do
      # Create two agents with different conversation IDs
      agent_id_1 = TestingHelpers.generate_test_agent_id()
      agent_id_2 = TestingHelpers.generate_test_agent_id()

      conversation_id_1 = "conv-111"
      conversation_id_2 = "conv-222"

      test_pid = self()

      callback_fn = fn conv_id, message ->
        send(test_pid, {:callback_called, conv_id, message.role})
        {:ok, [%{id: 1, content_type: "text", role: "user", content: "Saved"}]}
      end

      # Start both agents (helper handles subscription)
      {:ok, %{agent_id: ^agent_id_1}} =
        TestingHelpers.start_test_agent(
          agent_id: agent_id_1,
          pubsub: {Phoenix.PubSub, :test_pubsub},
          conversation_id: conversation_id_1,
          save_new_message_fn: callback_fn
        )

      {:ok, %{agent_id: ^agent_id_2}} =
        TestingHelpers.start_test_agent(
          agent_id: agent_id_2,
          pubsub: {Phoenix.PubSub, :test_pubsub},
          conversation_id: conversation_id_2,
          save_new_message_fn: callback_fn
        )

      # Add messages to both agents
      :ok = AgentServer.add_message(agent_id_1, Message.new_user!("Agent 1 message"))
      :ok = AgentServer.add_message(agent_id_2, Message.new_user!("Agent 2 message"))

      # Verify callbacks called with correct conversation_ids
      assert_receive {:callback_called, ^conversation_id_1, :user}, 100
      assert_receive {:callback_called, ^conversation_id_2, :user}, 100

      # Cleanup
      TestingHelpers.stop_test_agent(agent_id_1)
      TestingHelpers.stop_test_agent(agent_id_2)
    end
  end

  describe "error handling" do
    test "logs error but doesn't crash when callback raises exception" do
      agent_id = TestingHelpers.generate_test_agent_id()
      conversation_id = "conv-123"

      # Callback that raises
      callback_fn = fn _conv_id, _message ->
        raise "Simulated callback error"
      end

      {:ok, %{agent_id: ^agent_id, pid: pid}} =
        TestingHelpers.start_test_agent(
          agent_id: agent_id,
          pubsub: {Phoenix.PubSub, :test_pubsub},
          conversation_id: conversation_id,
          save_new_message_fn: callback_fn
        )

      # Add message - should not crash the server
      :ok = AgentServer.add_message(agent_id, Message.new_user!("Test"))

      # Server should still be alive
      Process.sleep(100)
      assert Process.alive?(pid)

      # Should NOT broadcast any events when callback raises
      refute_receive {:llm_message, _}, 100
      refute_receive {:display_message_saved, _}, 100

      # Cleanup
      TestingHelpers.stop_test_agent(agent_id)
    end

    test "handles callback returning invalid format" do
      agent_id = TestingHelpers.generate_test_agent_id()
      conversation_id = "conv-123"

      # Callback that returns invalid format
      callback_fn = fn _conv_id, _message ->
        # Invalid - should be a list
        {:ok, "not a list"}
      end

      {:ok, %{agent_id: ^agent_id, pid: pid}} =
        TestingHelpers.start_test_agent(
          agent_id: agent_id,
          pubsub: {Phoenix.PubSub, :test_pubsub},
          conversation_id: conversation_id,
          save_new_message_fn: callback_fn
        )

      # Add message - should handle gracefully
      :ok = AgentServer.add_message(agent_id, Message.new_user!("Test"))

      # Server should still be alive
      Process.sleep(100)
      assert Process.alive?(pid)

      # Should NOT broadcast any events when callback returns invalid format
      refute_receive {:llm_message, _}, 100
      refute_receive {:display_message_saved, _}, 100

      # Cleanup
      TestingHelpers.stop_test_agent(agent_id)
    end
  end
end
