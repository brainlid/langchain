defmodule LangChain.MessageExpansionTest do
  use LangChain.BaseCase
  use Mimic

  alias LangChain.ChatModels.ChatOpenAI
  alias LangChain.Chains.LLMChain
  alias LangChain.Function
  alias LangChain.LangChainError
  alias LangChain.Message
  alias LangChain.Message.ContentPart
  alias LangChain.Message.ToolCall
  alias LangChain.Message.ToolResult
  alias LangChain.MessageExpansion

  setup :verify_on_exit!

  @records "POLICY SECTION 4: refunds are issued within 30 days."
  @summary "Loaded 1 document."
  @anchor "Answer my question using the policy above."

  defp established(records, anchor) do
    [Message.new_assistant!(records), Message.new_user!(anchor)]
  end

  describe "expand/3" do
    test "the result carries the fallback so a mode without the step still shows it" do
      assert {:ok, %ToolResult{} = result} =
               MessageExpansion.expand(
                 @summary <> "\n\n" <> @records,
                 established(@records, @anchor),
                 result_content: @summary
               )

      assert [%ContentPart{content: text}] = result.content
      assert text =~ @summary
      assert text =~ @records
    end

    test "inserts exactly the messages it was given, in order" do
      assert {:ok, %ToolResult{message_expansion: expansion}} =
               MessageExpansion.expand(@records, established(@records, @anchor))

      assert [%Message{role: :assistant} = records, %Message{role: :user} = anchor] =
               expansion.messages

      assert [%ContentPart{content: @records}] = records.content
      assert [%ContentPart{content: @anchor}] = anchor.content
    end

    test "expands into a single message" do
      assert {:ok, %ToolResult{message_expansion: expansion}} =
               MessageExpansion.expand(@records, [Message.new_user!(@records)])

      assert [%Message{role: :user}] = expansion.messages
    end

    test "expands into as many messages as the tool needs" do
      messages = [
        Message.new_assistant!("one"),
        Message.new_user!("two"),
        Message.new_assistant!("three"),
        Message.new_user!("four"),
        Message.new_assistant!("five"),
        Message.new_user!("six")
      ]

      assert {:ok, %ToolResult{message_expansion: expansion}} =
               MessageExpansion.expand("1-6", messages)

      assert expansion.messages == messages
    end

    test "accepts content parts as the fallback" do
      parts = [ContentPart.text!(@summary), ContentPart.text!(@records)]

      assert {:ok, %ToolResult{content: ^parts}} =
               MessageExpansion.expand(parts, established(@records, @anchor))
    end

    test "carries what the result should keep once expanded" do
      assert {:ok, %ToolResult{message_expansion: expansion}} =
               MessageExpansion.expand(@records, established(@records, @anchor),
                 result_content: @summary
               )

      assert expansion.result_content == @summary
    end

    test "defaults what the result keeps, so the payload never stays in both places" do
      assert {:ok, %ToolResult{message_expansion: expansion}} =
               MessageExpansion.expand(@records, established(@records, @anchor))

      assert is_binary(expansion.result_content)
      refute expansion.result_content =~ @records
    end

    test "an explicit nil result_content leaves the fallback in place" do
      assert {:ok, %ToolResult{message_expansion: expansion}} =
               MessageExpansion.expand(@records, established(@records, @anchor),
                 result_content: nil
               )

      assert expansion.result_content == nil
    end

    test "refuses a system role, which a conversation can only hold one of" do
      assert_raise LangChainError, ~r/Cannot expand into a message with role :system/, fn ->
        MessageExpansion.expand(@records, [Message.new_system!(@records)])
      end
    end

    test "refuses a tool role, which needs a call to answer" do
      tool_message =
        Message.new_tool_result!(%{
          content: nil,
          tool_results: [
            ToolResult.new!(%{tool_call_id: "call_1", name: "search", content: "found"})
          ]
        })

      assert_raise LangChainError, ~r/Cannot expand into a message with role :tool/, fn ->
        MessageExpansion.expand(@records, [tool_message])
      end
    end

    test "refuses an empty expansion, and says what to do instead" do
      assert_raise LangChainError, ~r/needs at least one message/, fn ->
        MessageExpansion.expand(@records, [])
      end
    end

    test "refuses anything that is not a message" do
      assert_raise LangChainError, ~r/Expected a %LangChain.Message{}/, fn ->
        MessageExpansion.expand(@records, ["just text"])
      end

      assert_raise LangChainError, ~r/Expected a list of %LangChain.Message{}/, fn ->
        MessageExpansion.expand(@records, Message.new_user!("not a list"))
      end
    end

    test "leaves the call identity for execute_tool_call/3 to fill in" do
      assert {:ok, %ToolResult{tool_call_id: nil, name: nil, display_text: nil}} =
               MessageExpansion.expand(@records, established(@records, @anchor))
    end
  end

  describe "expandable?/1" do
    test "true for a result carrying an expansion" do
      assert {:ok, result} = MessageExpansion.expand(@records, established(@records, @anchor))
      assert MessageExpansion.expandable?(result)
    end

    test "false for a result with no expansion" do
      result = ToolResult.new!(%{tool_call_id: "call_1", name: "search", content: "found"})
      refute MessageExpansion.expandable?(result)
    end

    test "false for an interrupted result, whose turn is about to stop" do
      assert {:ok, result} = MessageExpansion.expand(@records, established(@records, @anchor))
      refute MessageExpansion.expandable?(%ToolResult{result | is_interrupt: true})
    end
  end

  describe "an expanding tool under a running chain" do
    setup do
      load_reference =
        Function.new!(%{
          name: "load_reference",
          description: "Load the reference material",
          display_text: "Loading reference",
          function: fn _args, _context ->
            MessageExpansion.expand(
              @summary <> "\n\n" <> @records,
              established(@records, @anchor),
              result_content: @summary
            )
          end
        })

      search =
        Function.new!(%{
          name: "search",
          description: "Search",
          function: fn _args, _context -> {:ok, "no results"} end
        })

      {:ok, chat} = ChatOpenAI.new(%{temperature: 0})

      chain =
        LLMChain.new!(%{llm: chat})
        |> LLMChain.add_tools([load_reference, search])
        |> LLMChain.add_message(Message.new_user!("What does the policy say?"))

      %{chain: chain}
    end

    test "the model's next call already contains the material", %{chain: chain} do
      test_pid = self()

      # Turn 1: call the loading tool.
      expect(ChatOpenAI, :call, fn _model, messages, _tools ->
        send(test_pid, {:turn, 1, messages})
        {:ok, [Message.new_assistant!(%{tool_calls: [call("call_1", "load_reference")]})]}
      end)

      # Turn 2: keep working instead of stopping. This is the behaviour that
      # made a run-boundary hand-off fail: by now the material has to be here.
      expect(ChatOpenAI, :call, fn _model, messages, _tools ->
        send(test_pid, {:turn, 2, messages})
        {:ok, [Message.new_assistant!(%{tool_calls: [call("call_2", "search")]})]}
      end)

      # Turn 3: answer.
      expect(ChatOpenAI, :call, fn _model, messages, _tools ->
        send(test_pid, {:turn, 3, messages})
        {:ok, [Message.new_assistant!("The policy allows 30 days.")]}
      end)

      assert {:ok, _final} = LLMChain.run(chain, mode: :while_needs_response)

      assert_received {:turn, 2, turn_2}

      assert Enum.any?(turn_2, fn message ->
               message.role == :assistant and text_of(message) =~ @records
             end)

      assert %Message{role: :user} = List.last(turn_2)
      assert text_of(List.last(turn_2)) =~ @anchor
    end

    test "the bulky payload is out of the result by the next call", %{chain: chain} do
      test_pid = self()

      expect(ChatOpenAI, :call, fn _model, _messages, _tools ->
        {:ok, [Message.new_assistant!(%{tool_calls: [call("call_1", "load_reference")]})]}
      end)

      expect(ChatOpenAI, :call, fn _model, messages, _tools ->
        send(test_pid, {:turn, 2, messages})
        {:ok, [Message.new_assistant!("The policy allows 30 days.")]}
      end)

      assert {:ok, _final} = LLMChain.run(chain, mode: :while_needs_response)

      assert_received {:turn, 2, turn_2}
      tool_message = Enum.find(turn_2, &(&1.role == :tool))

      assert [%ToolResult{content: [%ContentPart{content: @summary}]}] = tool_message.tool_results

      # The material appears exactly once in what the model is shown.
      assert 1 = Enum.count(turn_2, &(text_of(&1) =~ @records))
    end

    test "the function's call identity is stamped onto the expanded result", %{chain: chain} do
      expect(ChatOpenAI, :call, fn _model, _messages, _tools ->
        {:ok, [Message.new_assistant!(%{tool_calls: [call("call_1", "load_reference")]})]}
      end)

      assert {:ok, stepped} = LLMChain.run(chain, mode: :step)
      assert {:ok, stepped} = LLMChain.run(stepped, mode: :step)

      assert [
               %ToolResult{
                 tool_call_id: "call_1",
                 name: "load_reference",
                 display_text: "Loading reference"
               }
             ] = stepped.last_message.tool_results
    end

    test "fail-open: a mode without the step still shows the material", %{chain: chain} do
      expect(ChatOpenAI, :call, fn _model, _messages, _tools ->
        {:ok, [Message.new_assistant!(%{tool_calls: [call("call_1", "load_reference")]})]}
      end)

      # `:step` executes the tool calls and stops without composing the
      # expansion step, so the result is what the model would read.
      assert {:ok, stepped} = LLMChain.run(chain, mode: :step)
      assert {:ok, stepped} = LLMChain.run(stepped, mode: :step)

      assert [%ToolResult{content: [%ContentPart{content: text}]}] =
               stepped.last_message.tool_results

      assert text =~ @records
    end
  end

  defp call(call_id, name) do
    ToolCall.new!(%{call_id: call_id, name: name, arguments: %{}})
  end

  defp text_of(%Message{content: content}) when is_list(content) do
    content
    |> Enum.filter(&match?(%ContentPart{type: :text}, &1))
    |> Enum.map_join(" ", & &1.content)
  end

  defp text_of(%Message{content: content}) when is_binary(content), do: content
  defp text_of(%Message{}), do: ""
end
