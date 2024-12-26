defmodule LangChain.Chains.SummarizeConversationChainTest do
  use ExUnit.Case
  use Mimic

  doctest LangChain.Chains.SummarizeConversationChain

  alias LangChain.Chains.SummarizeConversationChain
  alias LangChain.Chains.LLMChain
  alias LangChain.Message
  alias LangChain.Message.ToolCall
  alias LangChain.Message.ToolResult
  alias LangChain.Message.ContentPart
  alias LangChain.ChatModels.ChatAnthropic
  alias LangChain.ChatModels.ChatOpenAI
  alias LangChain.LangChainError

  @test_anthropic_model "claude-3-5-haiku-latest"
  @test_openai_model "gpt-4o-mini"

  setup do
    llm_anthropic =
      ChatAnthropic.new!(%{model: @test_anthropic_model})

    llm_openai = ChatOpenAI.new!(%{model: @test_openai_model})

    data = %{llm: llm_anthropic, keep_count: 2, threshold_count: 6}
    summarizer = SummarizeConversationChain.new!(data)
    # basic chain to be summarized (no messages yet)
    chain = LLMChain.new!(%{llm: llm_anthropic})

    %{
      chain: chain,
      data: data,
      llm_anthropic: llm_anthropic,
      llm_openai: llm_openai,
      summarizer: summarizer
    }
  end

  describe "new/1" do
    test "defines a summarizer", %{data: data} do
      assert {:ok, summarizer} = SummarizeConversationChain.new(data)
      assert summarizer.keep_count == 2
      assert summarizer.threshold_count == 6
      assert %SummarizeConversationChain{} = summarizer
    end

    test "requires llm, keep_count, and threshold_count" do
      assert {:error, changeset} =
               SummarizeConversationChain.new(%{keep_count: nil, threshold_count: nil})

      refute changeset.valid?
      assert {"can't be blank", _} = changeset.errors[:llm]
      assert {"can't be blank", _} = changeset.errors[:keep_count]
      assert {"can't be blank", _} = changeset.errors[:threshold_count]
    end
  end

  describe "combine_messages_for_summary_text/2" do
    setup do
      test_messages = [
        Message.new_system!("System"),
        Message.new_user!("Question 1"),
        Message.new_assistant!("Answer 1"),
        Message.new_user!("Question 2"),
        Message.new_assistant!("Answer 2")
      ]

      %{test_messages: test_messages}
    end

    test "correctly combines basic messages up to the keep count", %{
      llm_anthropic: llm,
      chain: chain,
      test_messages: test_messages
    } do
      summarizer =
        SummarizeConversationChain.new!(%{llm: llm, threshold_count: 2, keep_count: 2})

      chain = LLMChain.add_messages(chain, test_messages)
      result = SummarizeConversationChain.combine_messages_for_summary_text(summarizer, chain)
      assert result == "<user>Question 1</user>\n<AI>Answer 1</AI>"

      # when leaving no messages
      summarizer_2 =
        SummarizeConversationChain.new!(%{llm: llm, threshold_count: 2, keep_count: 0})

      result = SummarizeConversationChain.combine_messages_for_summary_text(summarizer_2, chain)

      assert result ==
               "<user>Question 1</user>\n<AI>Answer 1</AI>\n<user>Question 2</user>\n<AI>Answer 2</AI>"
    end

    test "returns nil when no summary text needed yet", %{llm_anthropic: llm, chain: chain} do
      summarizer = SummarizeConversationChain.new!(%{llm: llm, threshold_count: 20})
      long_chain = LLMChain.add_messages(chain, get_full_conversation())

      assert nil ==
               SummarizeConversationChain.combine_messages_for_summary_text(
                 summarizer,
                 long_chain
               )
    end

    test "handles when the keep_count is greater than the number of messages", %{
      llm_anthropic: llm,
      chain: chain,
      test_messages: test_messages
    } do
      summarizer =
        SummarizeConversationChain.new!(%{llm: llm, threshold_count: 2, keep_count: 10})

      chain = LLMChain.add_messages(chain, test_messages)

      # told to keep 10 and there are only 4, so nothing gets summarized yet.
      assert nil ==
               SummarizeConversationChain.combine_messages_for_summary_text(summarizer, chain)
    end

    test "correctly handles tool calls, tool results and content part messages", %{
      chain: chain,
      llm_anthropic: llm
    } do
      summarizer = SummarizeConversationChain.new!(%{llm: llm, threshold_count: 2, keep_count: 0})

      chain =
        LLMChain.add_messages(chain, [
          Message.new_system!("System"),
          Message.new_user!([
            ContentPart.text!("Hi <user>you</user>"),
            ContentPart.image_url!("https://example.com/image.jpg")
          ]),
          Message.new_assistant!("basic answer"),
          Message.new_user!("question 2"),
          Message.new_assistant!(%{
            tool_calls: [ToolCall.new!(%{call_id: "123", name: "testing"})]
          }),
          Message.new_tool_result!(%{
            tool_results: [
              ToolResult.new!(%{tool_call_id: "123", name: "testing", content: "Stuff"})
            ]
          }),
          Message.new_assistant!("All taken care of.")
        ])

      result = SummarizeConversationChain.combine_messages_for_summary_text(summarizer, chain)

      assert result ==
               "<user>Hi you\n(Omitted URL to an image)</user>\n<AI>basic answer</AI>\n<user>question 2</user>\n<AI>Using tools\n- Tool 'testing' called</AI>\n<tool>Tool results\n- Tool 'testing' SUCCEEDED: Stuff</tool>\n<AI>All taken care of.</AI>"
    end
  end

  describe "create_summary_messages/1" do
    test "returns empty list when no summary text" do
      assert [] == SummarizeConversationChain.create_summary_messages(nil)
    end

    test "returns user and assistant message with summary" do
      summary_text = ~s|- You asked about why the sun is described as yellow
- I explained the science|

      [user_message, assistant_message] =
        SummarizeConversationChain.create_summary_messages(summary_text)

      assert user_message.role == :user

      assert user_message.content ==
               "Summarize our conversation to this point for future reference."

      assert assistant_message.role == :assistant
      assert assistant_message.content == summary_text
    end
  end

  describe "splice_messages_with_summary/3" do
    setup do
      test_messages = [
        Message.new_system!("System"),
        Message.new_user!("Question 1"),
        Message.new_assistant!("Answer 1"),
        Message.new_user!("Question 2"),
        Message.new_assistant!("Answer 2")
      ]

      %{test_messages: test_messages}
    end

    test "keeps the system message, creates summarized user and assistant messages, keeps last messages",
         %{
           summarizer: summarizer,
           chain: chain,
           test_messages: test_messages
         } do
      chain = LLMChain.add_messages(chain, test_messages)
      assert summarizer

      updated_chain =
        SummarizeConversationChain.splice_messages_with_summary(
          summarizer,
          chain,
          "fake summary text"
        )

      [system, summary_1, summary_2, user_2, ai_2] = updated_chain.messages
      assert system.content == "System"
      assert summary_1.role == :user
      assert summary_1.content == "Summarize our conversation to this point for future reference."
      assert summary_2.role == :assistant
      assert summary_2.content == "fake summary text"
      assert user_2.content == "Question 2"
      assert ai_2.content == "Answer 2"
    end

    test "correctly handles when keeping 0 messages", %{
      summarizer: summarizer,
      chain: chain,
      test_messages: test_messages
    } do
      chain = LLMChain.add_messages(chain, test_messages)
      assert summarizer

      updated_chain =
        SummarizeConversationChain.splice_messages_with_summary(
          %SummarizeConversationChain{summarizer | keep_count: 0},
          chain,
          "fake summary text"
        )

      [system, summary_1, summary_2] = updated_chain.messages
      assert system.content == "System"
      assert summary_1.role == :user
      assert summary_1.content == "Summarize our conversation to this point for future reference."
      assert summary_2.role == :assistant
      assert summary_2.content == "fake summary text"
    end

    test "correctly handles when no system message",
         %{
           summarizer: summarizer,
           chain: chain,
           test_messages: [_system | test_messages] = _test_messages
         } do
      chain = LLMChain.add_messages(chain, test_messages)
      assert summarizer

      updated_chain =
        SummarizeConversationChain.splice_messages_with_summary(
          summarizer,
          chain,
          "fake summary text"
        )

      [summary_1, summary_2, user_2, ai_2] = updated_chain.messages
      assert summary_1.role == :user
      assert summary_1.content == "Summarize our conversation to this point for future reference."
      assert summary_2.role == :assistant
      assert summary_2.content == "fake summary text"
      assert user_2.content == "Question 2"
      assert ai_2.content == "Answer 2"
    end
  end

  describe "run/2" do
    @tag live_call: true, live_anthropic: true
    test "runs the summarizer chain to create summary", %{llm_anthropic: llm, chain: chain} do
      summarizer = SummarizeConversationChain.new!(%{llm: llm, threshold_count: 6, keep_count: 2})
      chain = LLMChain.add_messages(chain, get_full_conversation())

      text_to_summarize =
        SummarizeConversationChain.combine_messages_for_summary_text(summarizer, chain)

      {:ok, result_chain} = SummarizeConversationChain.run(summarizer, text_to_summarize)
      summary_text = result_chain.last_message.content
      # IO.puts(summary_text)

      assert String.starts_with?(summary_text, "- User")
    end

    test "uses explicitly provided messages", %{
      llm_anthropic: llm,
      chain: chain
    } do
      # Made NOT LIVE here
      expect(ChatAnthropic, :call, fn _model, _messages, _tools ->
        {:ok, Message.new_assistant!("- Fake OpenAI summary")}
      end)

      summarizer =
        SummarizeConversationChain.new!(%{
          llm: llm,
          threshold_count: 30,
          keep_count: 2,
          messages: [
            Message.new_system!("Custom system message"),
            Message.new_user!("Custom user message"),
            Message.new_assistant!("Custom assistant message")
          ]
        })

      original_chain = LLMChain.add_messages(chain, get_full_conversation())

      {:ok, used_chain} = SummarizeConversationChain.run(summarizer, original_chain)

      [system, user, assistant, returned] = used_chain.messages
      assert %Message{role: :system, content: "Custom system message"} = system
      assert %Message{role: :user, content: "Custom user message"} = user
      assert %Message{role: :assistant, content: "Custom assistant message"} = assistant
      assert %Message{role: :assistant, content: "- Fake OpenAI summary"} = returned
    end
  end

  describe "summarize/3" do
    @tag live_call: true, live_anthropic: true
    test "correctly summarizes chain", %{llm_anthropic: llm, chain: chain} do
      summarizer = SummarizeConversationChain.new!(%{llm: llm, threshold_count: 6, keep_count: 2})
      chain = LLMChain.add_messages(chain, get_full_conversation())

      %LLMChain{} = summarized_chain = SummarizeConversationChain.summarize(summarizer, chain)
      # IO.inspect(summarized_chain.messages)

      [system, summary_1, summary_2, keep_1, keep_2] = summarized_chain.messages
      assert system.role == :system
      assert String.starts_with?(system.content, "You are a helpful travel assistant.")
      assert summary_1.role == :user
      assert summary_1.content == "Summarize our conversation to this point for future reference."
      assert summary_2.role == :assistant
      assert keep_1.role == :user
      assert keep_2.role == :assistant
    end

    test "set last_message correctly when keep_count is 0", %{llm_anthropic: llm, chain: chain} do
      # Made NOT LIVE here
      expect(ChatAnthropic, :call, fn _model, _messages, _tools ->
        {:ok, Message.new_assistant!("- Fake OpenAI summary")}
      end)

      summarizer = SummarizeConversationChain.new!(%{llm: llm, threshold_count: 6, keep_count: 0})
      chain = LLMChain.add_messages(chain, get_full_conversation())

      summarized_chain = SummarizeConversationChain.summarize(summarizer, chain)

      [_system, _summary_1, summary_2] = summarized_chain.messages
      assert summary_2.role == :assistant
      assert summary_2.content == "- Fake OpenAI summary"

      assert summarized_chain.last_message == summary_2
    end

    test "returns unmodified chain when LLM operation fails", %{
      llm_anthropic: llm,
      chain: chain
    } do
      # Made NOT LIVE here
      expect(ChatAnthropic, :call, fn _model, _messages, _tools ->
        {:error, LangChainError.exception(type: "overloaded", message: "Overloaded")}
      end)

      summarizer = SummarizeConversationChain.new!(%{llm: llm, threshold_count: 6, keep_count: 2})
      original_chain = LLMChain.add_messages(chain, get_full_conversation())

      assert original_chain == SummarizeConversationChain.summarize(summarizer, original_chain)
    end

    test "supports with_fallback option", %{llm_anthropic: llm, chain: chain} do
      # Made NOT LIVE here
      expect(ChatAnthropic, :call, fn _model, _messages, _tools ->
        {:error, LangChainError.exception(type: "overloaded", message: "Overloaded")}
      end)

      # Made NOT LIVE here
      expect(ChatOpenAI, :call, fn _model, _messages, _tools ->
        {:ok, Message.new_assistant!("- Fake OpenAI summary")}
      end)

      summarizer = SummarizeConversationChain.new!(%{llm: llm, threshold_count: 6, keep_count: 2})
      chain = LLMChain.add_messages(chain, get_full_conversation())

      summarized_chain =
        SummarizeConversationChain.summarize(summarizer, chain,
          with_fallbacks: [ChatOpenAI.new!(%{model: @test_openai_model, stream: false})]
        )

      # IO.inspect(summarized_chain.messages)

      [system, summary_1, summary_2, _keep_1, _keep_2] = summarized_chain.messages
      assert system.role == :system
      assert String.starts_with?(system.content, "You are a helpful travel assistant.")
      assert summary_1.role == :user
      assert summary_1.content == "Summarize our conversation to this point for future reference."
      assert summary_2.role == :assistant
      assert summary_2.content == "- Fake OpenAI summary"
    end

    test "returns unmodified chain when threshold not yet reached", %{
      llm_anthropic: llm,
      chain: chain
    } do
      summarizer =
        SummarizeConversationChain.new!(%{llm: llm, threshold_count: 30, keep_count: 2})

      original_chain = LLMChain.add_messages(chain, get_full_conversation())

      assert original_chain == SummarizeConversationChain.summarize(summarizer, original_chain)
    end

    @tag live_call: true, live_anthropic: true
    test "successfully summarizes a chain that was previously summarized", %{
      llm_anthropic: llm,
      chain: chain
    } do
      summarizer =
        SummarizeConversationChain.new!(%{llm: llm, threshold_count: 6, keep_count: 0})

      original_chain = LLMChain.add_messages(chain, previously_summarized_conversation())

      summarized_chain = SummarizeConversationChain.summarize(summarizer, original_chain)

      # IO.inspect(summarized_chain.messages)
      # IO.inspect(summarized_chain.last_message)

      assert summarized_chain.last_message.role == :assistant
      assert String.starts_with?(summarized_chain.last_message.content, "- ")
    end
  end

  describe "for_summary_text/1" do
    test "returns a basic user message" do
      user = Message.new_user!("a user message")
      result = SummarizeConversationChain.for_summary_text(user)
      assert result == "<user>a user message</user>"
    end

    test "returns a basic assistant message" do
      assistant = Message.new_assistant!("a basic assistant message")
      result = SummarizeConversationChain.for_summary_text(assistant)
      assert result == "<AI>a basic assistant message</AI>"
    end

    test "returns a ToolCall description" do
      tool_call =
        Message.new_assistant!(%{
          tool_calls: [
            ToolCall.new!(%{
              call_id: "call_abc123",
              name: "my_fun",
              arguments: Jason.encode!(%{parameters: %{name: "Sally", age: 44}})
            })
          ],
          status: :complete
        })

      result = SummarizeConversationChain.for_summary_text(tool_call)
      assert result == "<AI>Using tools\n- Tool 'my_fun' called</AI>"
    end

    test "returns a ToolResult description" do
      tool_response =
        Message.new_tool_result!(%{
          tool_results: [
            ToolResult.new!(%{
              tool_call_id: "123abc",
              name: "special_tool",
              is_error: false,
              content: Jason.encode!(%{status: "success", value: "green"})
            }),
            ToolResult.new!(%{
              tool_call_id: "234bcd",
              name: "special_tool",
              is_error: true,
              content: Jason.encode!(%{status: "failure", message: "I'm angry"})
            })
          ]
        })

      result = SummarizeConversationChain.for_summary_text(tool_response)

      assert result ==
               "<tool>Tool results\n- Tool 'special_tool' SUCCEEDED: {\"status\":\"success\",\"value\":\"green\"}\n- Tool 'special_tool' ERRORED: {\"message\":\"I'm angry\",\"status\":\"failure\"}</tool>"
    end

    test "returns for content parts" do
      user_message =
        Message.new_user!([
          ContentPart.text!("Hi"),
          ContentPart.image_url!("https://example.com/image.jpg"),
          ContentPart.image!("BASE64 ENCODED IMAGE DATA")
        ])

      result = SummarizeConversationChain.for_summary_text(user_message)

      assert result == "<user>Hi\n(Omitted URL to an image)\n(Omitted image data)</user>"
    end

    test "removes used XML tags" do
      user =
        Message.new_user!(
          "A user message with a <user>malicious</user> <AI>set of text</AI> and <tool>stuff</tool>."
        )

      result = SummarizeConversationChain.for_summary_text(user)
      assert result == "<user>A user message with a malicious set of text and stuff.</user>"
    end
  end

  ###
  ### Test factory conversation
  ###
  defp get_full_conversation() do
    [
      Message.new_system!("You are a helpful travel assistant."),
      Message.new_user!("Hi! I need help planning a vacation to Italy."),
      Message.new_assistant!(
        "Of course! I'd be happy to help you plan your Italian vacation. When are you thinking of traveling, and how long do you plan to stay?"
      ),
      Message.new_user!(
        "I'm planning to go for 2 weeks in September. I'm interested in both cities and coastal areas."
      ),
      Message.new_assistant!(
        "September is a wonderful time to visit Italy! For a 2-week trip combining cities and coast, I'd suggest starting in Rome, then Florence, followed by some time on the Amalfi Coast. Would you like a more detailed breakdown of how to split your time?"
      ),
      Message.new_user!("Yes, that would be great! Also, should I rent a car or rely on trains?"),
      Message.new_assistant!(
        ~s|For cities like Rome and Florence, trains are definitely the way to go - they're fast, efficient, and you won't have to worry about parking. However, for the Amalfi Coast, a car could be useful, though the roads are quite winding. I'd suggest this split:
- Rome: 4 days
- Florence: 4 days
- Amalfi Coast: 5 days
Would you like specific hotel recommendations for each location?|
      ),
      Message.new_user!(
        "Yes please! I prefer mid-range hotels with good locations. My budget is around $200 per night."
      ),
      Message.new_assistant!(~s|Here are some great options within your budget:

Rome: Hotel De Russie - close to the Spanish Steps
Florence: Hotel Davanzati - near the Ponte Vecchio
Amalfi: Hotel Marina Riviera - beautiful sea views

Would you like me to check current availability and rates for your September dates?|),
      Message.new_user!(
        "That would be helpful! My dates are September 10-24. Could you also suggest some must-see attractions in each city?"
      ),
      Message.new_assistant!(
        ~s|I'll check those dates for you. Meanwhile, here are the must-see attractions:

Rome:
- Colosseum
- Vatican Museums
- Roman Forum
- Trevi Fountain

Florence:
- Uffizi Gallery
- Duomo
- Ponte Vecchio
- Accademia (David)

Amalfi:
- Positano
- Ravello Gardens
- Boat tour to Capri

Would you like me to help you with advance tickets for any of these attractions?|
      ),
      Message.new_user!(
        "Yes, please! I'd definitely want skip-the-line tickets for the Vatican and Uffizi. How far in advance should I book these?"
      ),
      Message.new_assistant!(
        ~s|For September, I'd recommend booking the Vatican and Uffizi tickets at least 2-3 months in advance. They're very popular attractions! For the Vatican, early morning tours (8 AM) are best to avoid crowds. The Uffizi is less crowded in the late afternoon.

I can provide you with the official booking websites for both. Would you also like me to suggest some good guided tour options? They often provide more context and historical background.|
      )
    ]
  end

  def previously_summarized_conversation() do
    [
      Message.new_system!("You are a helpful travel assistant."),
      Message.new_user!("Summarize our conversation to this point for future reference."),
      Message.new_assistant!(
        "- User seeking help planning a 2-week Italy vacation in September\n- Proposed travel route: Rome (4 days), Florence (4 days), Amalfi Coast (5 days)\n- Transportation recommendations:\n  * Trains for city travel (Rome and Florence)\n  * Potential car rental for Amalfi Coast\n- Conversation focused on itinerary planning and transportation logistics\n- User interested in both urban and coastal experiences\n- Detailed location and time allocation discussed\n- Awaiting further details on hotel preferences and specific interests in each destination"
      ),
      Message.new_user!(
        "Yes please! I prefer mid-range hotels with good locations. My budget is around $200 per night."
      ),
      Message.new_assistant!(~s|Here are some great options within your budget:

Rome: Hotel De Russie - close to the Spanish Steps
Florence: Hotel Davanzati - near the Ponte Vecchio
Amalfi: Hotel Marina Riviera - beautiful sea views

Would you like me to check current availability and rates for your September dates?|),
      Message.new_user!(
        "That would be helpful! My dates are September 10-24. Could you also suggest some must-see attractions in each city?"
      ),
      Message.new_assistant!(
        ~s|I'll check those dates for you. Meanwhile, here are the must-see attractions:

Rome:
- Colosseum
- Vatican Museums
- Roman Forum
- Trevi Fountain

Florence:
- Uffizi Gallery
- Duomo
- Ponte Vecchio
- Accademia (David)

Amalfi:
- Positano
- Ravello Gardens
- Boat tour to Capri

Would you like me to help you with advance tickets for any of these attractions?|
      ),
      Message.new_user!(
        "Yes, please! I'd definitely want skip-the-line tickets for the Vatican and Uffizi. How far in advance should I book these?"
      ),
      Message.new_assistant!(
        ~s|For September, I'd recommend booking the Vatican and Uffizi tickets at least 2-3 months in advance. They're very popular attractions! For the Vatican, early morning tours (8 AM) are best to avoid crowds. The Uffizi is less crowded in the late afternoon.

I can provide you with the official booking websites for both. Would you also like me to suggest some good guided tour options? They often provide more context and historical background.|
      )
    ]
  end
end
