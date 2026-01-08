defmodule LangChain.Agents.Middleware.ConversationTitleIntegrationTest do
  use LangChain.BaseCase, async: false
  use Mimic

  alias LangChain.Agents.{Agent, AgentServer, State}
  alias LangChain.Agents.Middleware.ConversationTitle
  alias LangChain.ChatModels.ChatAnthropic
  alias LangChain.Message
  alias LangChain.Message.ContentPart

  # Because we are mocking ChatAnthropic and it is being executed by the
  # AgentServer -> Task, we are making it global and running with `async:
  # false`.
  setup :set_mimic_global

  setup do
    start_supervised({Phoenix.PubSub, name: :langchain_pubsub})
    :ok
  end

  defp mock_model do
    ChatAnthropic.new!(%{
      model: "claude-3-5-haiku-latest",
      api_key: "test_key",
      stream: false
    })
  end

  defp create_agent_with_title_middleware(opts \\ []) do
    chat_model = Keyword.get(opts, :chat_model, mock_model())
    fallbacks = Keyword.get(opts, :fallbacks, [])
    agent_id = Keyword.get(opts, :agent_id, "test-agent-#{System.unique_integer([:positive])}")

    middleware_config = [
      chat_model: chat_model,
      fallbacks: fallbacks
    ]

    Agent.new!(%{
      agent_id: agent_id,
      model: mock_model(),
      base_system_prompt: "Test agent",
      replace_default_middleware: true,
      middleware: [{ConversationTitle, middleware_config}]
    })
  end

  # TODO: ISSUES:
  # - replace_default_middleware: true and only the ConversationTitle middleware still adds all the file system tools.

  describe "middleware integration with AgentServer" do
    test "full integration: generates title after user message" do
      # This test should verify the FULL integration flow:
      # 1. User sends a message to the agent
      # 2. Agent processes the message and responds
      # 3. ConversationTitle middleware detects it's a new conversation
      # 4. Middleware spawns async task to generate title
      # 5. Title is generated via LLM call and stored in metadata

      agent = create_agent_with_title_middleware()
      agent_id = agent.agent_id

      {:ok, _server_pid} =
        AgentServer.start_link(
          agent: agent,
          name: AgentServer.get_name(agent_id),
          pubsub: {Phoenix.PubSub, :langchain_pubsub},
          debug_pubsub: {Phoenix.PubSub, :langchain_pubsub}
        )

      AgentServer.subscribe(agent_id)
      AgentServer.subscribe_debug(agent_id)

      # Made NOT LIVE here
      #
      # Mock ChatAnthropic responses here
      # - Mock the main agent's response to the user message
      # - Mock the ConversationTitle middleware's LLM call for title generation
      #
      # Expecting 2 calls using ChatAnthropic.
      # - FIRST request is for the answer to the weather question
      # - SECOND request is for the answer to the title
      expect(ChatAnthropic, :call, 2, fn _model, messages, _tools ->
        # Pattern match on the last message to differentiate when it's the
        # user's message vs the request to generate a title.
        case List.last(messages) do
          %Message{content: [%ContentPart{content: "What's the weather today?"}]} ->
            {:ok, [Message.new_assistant!("The weather is balmy and icky.")]}

          # For the generated conversation title response
          _other ->
            {:ok, [Message.new_assistant!("Asking about the weather")]}
        end
      end)

      # Send a user message through the proper API
      :ok = AgentServer.add_message(agent_id, Message.new_user!("What's the weather today?"))

      # Should receive broadcast debug event wrapped in {:agent, {:debug, event}} tuple
      assert_receive {:agent, {:debug, {:agent_state_update, _received_state}}}, 100

      # Should receive PubSub event of title generation
      assert_receive {:agent, {:conversation_title_generated, title, _agent_id}}, 100
      assert title == "Asking about the weather"

      # Verify title was generated and stored
      state = AgentServer.get_state(agent_id)
      assert State.get_metadata(state, "conversation_title") == "Asking about the weather"
    end

    test "title generation failure doesn't break agent" do
      # Integration test: When the title generation LLM call fails,
      # the agent should continue working normally. TextToTitleChain will
      # return a fallback title, but the important thing is the agent doesn't crash.

      agent = create_agent_with_title_middleware()
      agent_id = agent.agent_id

      {:ok, _server_pid} =
        AgentServer.start_link(
          agent: agent,
          name: AgentServer.get_name(agent_id),
          pubsub: {Phoenix.PubSub, :langchain_pubsub},
          debug_pubsub: {Phoenix.PubSub, :langchain_pubsub}
        )

      AgentServer.subscribe(agent_id)
      AgentServer.subscribe_debug(agent_id)

      # Mock ChatAnthropic responses:
      # - First call (agent response) succeeds
      # - Second call (title generation) fails
      expect(ChatAnthropic, :call, 2, fn _model, messages, _tools ->
        case List.last(messages) do
          %Message{content: [%ContentPart{content: "Tell me a joke"}]} ->
            {:ok, [Message.new_assistant!("Why did the chicken cross the road?")]}

          # Title generation request fails
          _other ->
            {:error, "API timeout"}
        end
      end)

      # Send a user message
      :ok = AgentServer.add_message(agent_id, Message.new_user!("Tell me a joke"))

      # Wait for title generation event (will use fallback title)
      assert_receive {:agent, {:conversation_title_generated, fallback_title, _agent_id}}, 200

      # Verify fallback title was generated (TextToTitleChain returns "New topic" by default on error)
      state = AgentServer.get_state(agent_id)
      assert State.get_metadata(state, "conversation_title") == fallback_title
      assert fallback_title == "New topic"

      # Verify agent is still functional - can send another message
      # Only 1 LLM call expected (no title generation since title exists)
      expect(ChatAnthropic, :call, 1, fn _model, _messages, _tools ->
        {:ok, [Message.new_assistant!("Another response")]}
      end)

      :ok = AgentServer.add_message(agent_id, Message.new_user!("Another question"))

      # Should NOT receive another title generation event
      # Builds in the delay as well
      refute_receive {:agent, {:conversation_title_generated, _title, _agent_id}}, 100
    end

    test "title generation skipped when title already exists" do
      # Integration test: When a title already exists in metadata,
      # the middleware should NOT generate a new title

      agent = create_agent_with_title_middleware()
      agent_id = agent.agent_id

      {:ok, _server_pid} =
        AgentServer.start_link(
          agent: agent,
          name: AgentServer.get_name(agent_id),
          pubsub: {Phoenix.PubSub, :langchain_pubsub},
          debug_pubsub: {Phoenix.PubSub, :langchain_pubsub}
        )

      AgentServer.subscribe(agent_id)
      AgentServer.subscribe_debug(agent_id)

      # Pre-populate the conversation with an existing title by sending a middleware message
      existing_title = "Existing Conversation Title"

      AgentServer.send_middleware_message(
        agent_id,
        ConversationTitle,
        {:title_generated, existing_title}
      )

      # Wait for the initial title to be set
      Process.sleep(50)

      # Mock only ONE ChatAnthropic call - for the agent's response
      # No second call should happen for title generation (verifies middleware skips title gen)
      expect(ChatAnthropic, :call, 1, fn _model, _messages, _tools ->
        {:ok, [Message.new_assistant!("Sure, I can help with that.")]}
      end)

      # Send a user message
      :ok = AgentServer.add_message(agent_id, Message.new_user!("Can you help me?"))

      # Should NOT receive a title generation event (key assertion: middleware didn't trigger)
      # Builds in the delay as well
      refute_receive {:agent, {:conversation_title_generated, _title, _agent_id}}, 100

      # Verify title is still the original one (not regenerated)
      final_state = AgentServer.get_state(agent_id)
      assert State.get_metadata(final_state, "conversation_title") == existing_title
    end

    test "only first message triggers title generation" do
      # Integration test: In a new conversation with multiple messages,
      # only the first message should trigger title generation

      agent = create_agent_with_title_middleware()
      agent_id = agent.agent_id

      {:ok, _server_pid} =
        AgentServer.start_link(
          agent: agent,
          name: AgentServer.get_name(agent_id),
          pubsub: {Phoenix.PubSub, :langchain_pubsub},
          debug_pubsub: {Phoenix.PubSub, :langchain_pubsub}
        )

      AgentServer.subscribe(agent_id)
      AgentServer.subscribe_debug(agent_id)

      # First message: expect 2 calls (agent response + title generation)
      expect(ChatAnthropic, :call, 2, fn _model, messages, _tools ->
        case List.last(messages) do
          %Message{content: [%ContentPart{content: "Hello!"}]} ->
            {:ok, [Message.new_assistant!("Hi there!")]}

          _other ->
            {:ok, [Message.new_assistant!("Initial Greeting Conversation")]}
        end
      end)

      # Send first user message
      :ok = AgentServer.add_message(agent_id, Message.new_user!("Hello!"))

      # Wait for title generation event (key signal: middleware triggered)
      assert_receive {:agent, {:conversation_title_generated, title, _agent_id}}, 200
      assert title == "Initial Greeting Conversation"

      # Verify title was stored
      state = AgentServer.get_state(agent_id)
      assert State.get_metadata(state, "conversation_title") == "Initial Greeting Conversation"

      # Second message: expect only 1 call (agent response only, no title generation)
      # This mock expectation proves the middleware didn't try to generate a title
      expect(ChatAnthropic, :call, 1, fn _model, _messages, _tools ->
        {:ok, [Message.new_assistant!("Sure, how can I help?")]}
      end)

      # Send second user message
      :ok = AgentServer.add_message(agent_id, Message.new_user!("Can you help me?"))

      # Should NOT receive another title generation event (key assertion)
      # Builds in the delay as well
      refute_receive {:agent, {:conversation_title_generated, _title, _agent_id}}, 100

      # Verify title is unchanged (no regeneration)
      final_state = AgentServer.get_state(agent_id)

      assert State.get_metadata(final_state, "conversation_title") ==
               "Initial Greeting Conversation"
    end
  end
end
