defmodule LangChain.Agents.Middleware.ConversationTitleTest do
  use ExUnit.Case, async: true

  alias LangChain.Agents.{State, Middleware}
  alias LangChain.Agents.Middleware.ConversationTitle
  alias LangChain.ChatModels.ChatAnthropic

  # Mock model for testing
  defp mock_model do
    ChatAnthropic.new!(%{
      model: "claude-3-5-haiku-latest",
      api_key: "test_key"
    })
  end

  describe "init/1" do
    test "initializes with required chat_model" do
      assert {:ok, config} = ConversationTitle.init(chat_model: mock_model())
      assert config.chat_model != nil
      assert config.fallbacks == []
      assert config.examples != nil
    end

    test "returns error when chat_model is missing" do
      assert {:error, msg} = ConversationTitle.init([])
      assert msg =~ "requires :chat_model"
    end

    test "accepts optional fallbacks configuration" do
      fallback = mock_model()

      assert {:ok, config} =
               ConversationTitle.init(chat_model: mock_model(), fallbacks: [fallback])

      assert config.fallbacks == [fallback]
    end

    test "accepts optional custom prompt_template" do
      custom_prompt = "Generate a title for this conversation"

      assert {:ok, config} =
               ConversationTitle.init(chat_model: mock_model(), prompt_template: custom_prompt)

      assert config.prompt_template == custom_prompt
    end

    test "accepts optional custom examples" do
      examples = ["Example 1", "Example 2"]
      assert {:ok, config} = ConversationTitle.init(chat_model: mock_model(), examples: examples)
      assert config.examples == examples
    end

    test "accepts optional custom id for multiple instances" do
      assert {:ok, config} =
               ConversationTitle.init(chat_model: mock_model(), id: "custom_title_gen")

      assert config.id == "custom_title_gen"
    end
  end

  describe "handle_message/3" do
    test "handles :title_generated message and updates state" do
      state = State.new!()
      config = %{chat_model: mock_model()}

      assert {:ok, updated_state} =
               ConversationTitle.handle_message({:title_generated, "My Title"}, state, config)

      assert State.get_metadata(updated_state, "conversation_title") == "My Title"
    end

    test "handles :title_generation_failed message gracefully" do
      state = State.new!()
      config = %{chat_model: mock_model()}

      assert {:ok, ^state} =
               ConversationTitle.handle_message(
                 {:title_generation_failed, :timeout},
                 state,
                 config
               )

      # State should be unchanged
      refute State.get_metadata(state, "conversation_title")
    end

    test "handle_message updates different titles independently" do
      state = State.new!()
      config = %{chat_model: mock_model()}

      # First title
      {:ok, state1} =
        ConversationTitle.handle_message({:title_generated, "Title 1"}, state, config)

      assert State.get_metadata(state1, "conversation_title") == "Title 1"

      # Update with new title
      {:ok, state2} =
        ConversationTitle.handle_message({:title_generated, "Title 2"}, state1, config)

      assert State.get_metadata(state2, "conversation_title") == "Title 2"
    end
  end

  describe "middleware behavior implementation" do
    test "can be initialized through Middleware.init_middleware/1" do
      middleware_spec = {ConversationTitle, [chat_model: mock_model()]}

      entry = Middleware.init_middleware(middleware_spec)

      assert entry.module == ConversationTitle
      assert entry.id == ConversationTitle
      assert entry.config.chat_model != nil
    end

    test "can be initialized with custom ID" do
      middleware_spec =
        {ConversationTitle,
         [
           chat_model: mock_model(),
           id: "custom_title"
         ]}

      entry = Middleware.init_middleware(middleware_spec)

      assert entry.module == ConversationTitle
      assert entry.id == "custom_title"
      assert entry.config.id == "custom_title"
    end
  end

  describe "configuration validation" do
    test "preserves all configuration options in config" do
      fallback = mock_model()
      examples = ["Example 1", "Example 2"]
      prompt = "Custom prompt"

      {:ok, config} =
        ConversationTitle.init(
          chat_model: mock_model(),
          fallbacks: [fallback],
          prompt_template: prompt,
          examples: examples,
          id: "test_id"
        )

      assert config.chat_model != nil
      assert config.fallbacks == [fallback]
      assert config.prompt_template == prompt
      assert config.examples == examples
      assert config.id == "test_id"
    end
  end
end
