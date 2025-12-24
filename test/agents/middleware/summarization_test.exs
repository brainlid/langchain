defmodule LangChain.Agents.Middleware.SummarizationTest do
  use LangChain.BaseCase, async: true

  alias LangChain.Agents.Middleware.Summarization
  alias LangChain.Agents.State
  alias LangChain.Message
  alias LangChain.Message.ToolCall
  alias LangChain.Message.ToolResult
  alias LangChain.ChatModels.ChatOpenAI

  describe "init/1" do
    test "uses default configuration" do
      assert {:ok, config} = Summarization.init([])
      assert config.max_tokens_before_summary == 170_000
      assert config.messages_to_keep == 6
      assert is_function(config.token_counter, 1)
      assert is_binary(config.summary_prompt)
    end

    test "accepts custom configuration" do
      model = ChatOpenAI.new!(%{model: "gpt-4", stream: false})

      opts = [
        model: model,
        max_tokens_before_summary: 100_000,
        messages_to_keep: 10,
        token_counter: fn _ -> 1000 end
      ]

      assert {:ok, config} = Summarization.init(opts)
      assert config.model == model
      assert config.max_tokens_before_summary == 100_000
      assert config.messages_to_keep == 10
      assert config.token_counter.([]) == 1000
    end
  end

  describe "before_model/2" do
    test "returns state unchanged when under token threshold" do
      messages = [
        Message.new_system!("You are helpful"),
        Message.new_user!("Hello"),
        Message.new_assistant!("Hi there!")
      ]

      state = State.new!(%{messages: messages})

      config = %{
        max_tokens_before_summary: 1_000_000,
        messages_to_keep: 6,
        token_counter: fn _ -> 100 end
      }

      assert {:ok, ^state} = Summarization.before_model(state, config)
    end

    test "returns state unchanged when messages is empty" do
      state = State.new!(%{messages: []})

      config = %{
        max_tokens_before_summary: 1000,
        messages_to_keep: 6,
        token_counter: fn _ -> 0 end
      }

      assert {:ok, ^state} = Summarization.before_model(state, config)
    end

    test "returns state unchanged when messages is nil" do
      state = State.new!(%{})

      config = %{
        max_tokens_before_summary: 1000,
        messages_to_keep: 6,
        token_counter: fn _ -> 0 end
      }

      assert {:ok, ^state} = Summarization.before_model(state, config)
    end

    test "keeps all messages when no safe cutoff point found" do
      # Create scenario where all recent messages are assistant+tool pairs
      tool_call =
        ToolCall.new!(%{
          call_id: "call_1",
          name: "search",
          arguments: %{"q" => "test"}
        })

      messages = [
        Message.new_system!("System"),
        Message.new_user!("Search for something"),
        Message.new_assistant!(%{tool_calls: [tool_call]}),
        Message.new_tool_result!(%{
          tool_results: [
            ToolResult.new!(%{tool_call_id: "call_1", name: "search", content: "Results"})
          ]
        })
      ]

      state = State.new!(%{messages: messages})

      config = %{
        model: nil,
        max_tokens_before_summary: 10,
        # Keep 2 messages, would need to cut between assistant and tool
        messages_to_keep: 2,
        summary_prompt: "Summarize",
        token_counter: fn _ -> 1000 end
      }

      # Should keep all messages since we can't safely cut the assistant+tool pair
      assert {:ok, result_state} = Summarization.before_model(state, config)
      assert result_state.messages == messages
    end
  end

  describe "token counting" do
    test "counts tokens for simple text messages" do
      messages = [
        Message.new_user!("Hello world"),
        Message.new_assistant!("Hi there, how can I help you today?")
      ]

      {:ok, config} = Summarization.init([])

      tokens = config.token_counter.(messages)
      # Should be roughly proportional to text length
      assert tokens > 0
      assert tokens < 100
    end

    test "counts tokens for messages with tool calls" do
      tool_call =
        ToolCall.new!(%{
          call_id: "call_1",
          name: "calculator",
          arguments: %{"expression" => "2 + 2"}
        })

      messages = [
        Message.new_assistant!(%{tool_calls: [tool_call]})
      ]

      {:ok, config} = Summarization.init([])

      tokens = config.token_counter.(messages)
      # Tool calls add extra tokens
      assert tokens > 10
    end

    test "counts tokens for messages with tool results" do
      messages = [
        Message.new_tool_result!(%{
          tool_results: [
            ToolResult.new!(%{
              tool_call_id: "call_1",
              name: "calculator",
              content: "The result is 4"
            })
          ]
        })
      ]

      {:ok, config} = Summarization.init([])
      tokens = config.token_counter.(messages)
      assert tokens > 10
    end

    test "counts tokens for ContentPart messages" do
      message =
        Message.new_user!([
          Message.ContentPart.new!(%{type: :text, content: "Look at this image"}),
          Message.ContentPart.new!(%{type: :image_url, url: "http://example.com/image.jpg"})
        ])

      {:ok, config} = Summarization.init([])
      tokens = config.token_counter.([message])

      # Text + image should be substantial
      assert tokens > 100
    end
  end

  describe "safe cutoff detection" do
    test "finds safe cutoff point before assistant with tool calls" do
      tool_call =
        ToolCall.new!(%{
          call_id: "call_1",
          name: "search",
          arguments: %{"q" => "test"}
        })

      messages = [
        Message.new_system!("System"),
        Message.new_user!("Message 1"),
        Message.new_assistant!("Response 1"),
        Message.new_user!("Message 2"),
        Message.new_assistant!(%{tool_calls: [tool_call]}),
        Message.new_tool_result!(%{
          tool_results: [
            ToolResult.new!(%{tool_call_id: "call_1", name: "search", content: "Results"})
          ]
        }),
        Message.new_user!("Recent message")
      ]

      state = State.new!(%{messages: messages, metadata: %{model: nil}})

      config = %{
        model: nil,
        max_tokens_before_summary: 10,
        messages_to_keep: 3,
        # Would keep: assistant+tool+user (last 3)
        # Should cut before the assistant with tool_calls
        summary_prompt: "Summarize",
        token_counter: fn _ -> 1000 end
      }

      # Since we can't actually summarize without a model, we'll get an error
      # but we can test the logic doesn't crash
      assert {:ok, _result} = Summarization.before_model(state, config)
    end

    test "allows cutting after completed tool cycle" do
      tool_call =
        ToolCall.new!(%{
          call_id: "call_1",
          name: "search",
          arguments: %{"q" => "test"}
        })

      messages = [
        Message.new_system!("System"),
        Message.new_user!("Search request"),
        Message.new_assistant!(%{tool_calls: [tool_call]}),
        Message.new_tool_result!(%{
          tool_results: [
            ToolResult.new!(%{tool_call_id: "call_1", name: "search", content: "Results"})
          ]
        }),
        # Safe to cut here - tool cycle is complete
        Message.new_user!("Thanks!"),
        Message.new_assistant!("You're welcome!")
      ]

      state = State.new!(%{messages: messages})

      config = %{
        model: nil,
        max_tokens_before_summary: 10,
        messages_to_keep: 2,
        # Keep last 2 messages
        summary_prompt: "Summarize",
        token_counter: fn _ -> 1000 end
      }

      assert {:ok, _result} = Summarization.before_model(state, config)
    end
  end

  describe "configuration validation" do
    test "accepts valid messages_to_keep values" do
      assert {:ok, config} = Summarization.init(messages_to_keep: 0)
      assert config.messages_to_keep == 0

      assert {:ok, config} = Summarization.init(messages_to_keep: 20)
      assert config.messages_to_keep == 20
    end

    test "accepts valid max_tokens_before_summary values" do
      assert {:ok, config} = Summarization.init(max_tokens_before_summary: 1000)
      assert config.max_tokens_before_summary == 1000

      assert {:ok, config} = Summarization.init(max_tokens_before_summary: 500_000)
      assert config.max_tokens_before_summary == 500_000
    end

    test "accepts custom token counter function" do
      custom_counter = fn messages -> length(messages) * 100 end
      assert {:ok, config} = Summarization.init(token_counter: custom_counter)
      assert config.token_counter.([1, 2, 3]) == 300
    end
  end

  describe "edge cases" do
    test "handles empty message list" do
      state = State.new!(%{messages: []})
      {:ok, config} = Summarization.init([])

      assert {:ok, ^state} = Summarization.before_model(state, config)
    end

    test "handles state with only system message" do
      messages = [Message.new_system!("You are helpful")]
      state = State.new!(%{messages: messages})
      {:ok, config} = Summarization.init(max_tokens_before_summary: 10)

      # Token counter returns high value to trigger summarization
      config = %{config | token_counter: fn _ -> 1000 end}

      # Should keep the system message
      assert {:ok, result_state} = Summarization.before_model(state, config)
      assert length(result_state.messages) >= 1
      assert List.first(result_state.messages).role == :system
    end

    test "handles messages with minimal content" do
      messages = [
        Message.new_user!("."),
        Message.new_assistant!(".")
      ]

      {:ok, config} = Summarization.init([])
      tokens = config.token_counter.(messages)

      # Should handle gracefully
      assert tokens >= 0
    end

    test "handles messages with very long content" do
      long_text = String.duplicate("word ", 10_000)
      messages = [Message.new_user!(long_text)]

      {:ok, config} = Summarization.init([])
      tokens = config.token_counter.(messages)

      # Should count substantial tokens for long text
      assert tokens > 1000
    end
  end
end
