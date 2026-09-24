defmodule LangChain.ChatModels.ChatOpenAIResponsesPhaseTest do
  @moduledoc """
  The Responses API labels each assistant message item with a `phase`. These
  tests decode recorded responses and streams, and check that the label becomes
  the content part's utterance marker on the way in and the item's `phase` on
  the way out.
  """
  use LangChain.BaseCase
  use Mimic

  alias LangChain.Chains.LLMChain
  alias LangChain.ChatModels.ChatOpenAIResponses
  alias LangChain.Message
  alias LangChain.Message.ContentPart
  alias LangChain.Message.ToolCall
  alias LangChain.MessageDelta

  setup :verify_on_exit!

  @fixture_dir Path.expand("../fixtures/openai_phase", __DIR__)

  setup do
    %{model: ChatOpenAIResponses.new!(%{model: "gpt-5.4"})}
  end

  defp fixture(name), do: @fixture_dir |> Path.join(name) |> File.read!() |> Jason.decode!()

  defp output_messages(response),
    do: Enum.filter(response["output"], &(&1["type"] == "message"))

  # Feeds a recorded SSE body through the same decode path a live stream takes.
  defp stream_to_message(model, sse_body) do
    {events, _rest} = ChatOpenAIResponses.decode_stream({sse_body, ""})

    deltas =
      model
      |> ChatOpenAIResponses.do_process_response(events)
      |> Enum.filter(&match?(%MessageDelta{}, &1))

    {:ok, message} = deltas |> MessageDelta.merge_deltas() |> MessageDelta.to_message()
    message
  end

  defp sse(events) do
    Enum.map_join(events, fn event ->
      "event: #{event["type"]}\ndata: #{Jason.encode!(event)}\n\n"
    end)
  end

  defp text_markers(%Message{content: parts}) do
    for %ContentPart{type: :text} = part <- parts, do: {part.content, ContentPart.utterance(part)}
  end

  describe "non-streaming decode" do
    test "commentary followed by a final answer yields two marked parts", %{model: model} do
      response = fixture("commentary_then_answer.json")
      [commentary, answer] = output_messages(response)

      assert %Message{} = message = ChatOpenAIResponses.do_process_response(model, response)

      assert [narration_part, answer_part] = message.content
      assert narration_part.content == hd(commentary["content"])["text"]
      assert ContentPart.utterance(narration_part) == "narration"
      assert answer_part.content == hd(answer["content"])["text"]
      assert ContentPart.utterance(answer_part) == "answer"

      refute Message.narration?(message)
      assert Message.answer_content(message) == answer_part.content
    end

    test "a commentary-only response is narration", %{model: model} do
      response =
        fixture("commentary_then_answer.json")
        |> Map.update!("output", fn output ->
          Enum.reject(output, &(&1["phase"] == "final_answer"))
        end)

      message = ChatOpenAIResponses.do_process_response(model, response)

      assert [%ContentPart{type: :text} = part] = message.content
      assert ContentPart.narration?(part)
      assert Message.narration?(message)
      assert Message.answer_content(message) == nil
    end

    test "commentary sent with tool calls is a tool call, not narration", %{model: model} do
      message =
        ChatOpenAIResponses.do_process_response(model, fixture("commentary_with_tool_calls.json"))

      assert [%ContentPart{} = part] = message.content
      assert ContentPart.narration?(part)
      assert [%ToolCall{} | _] = message.tool_calls
      assert Message.is_tool_call?(message)
      refute Message.narration?(message)
    end

    test "separate commentary items stay separate parts", %{model: model} do
      response = fixture("two_commentary_items.json")
      message = ChatOpenAIResponses.do_process_response(model, response)

      assert [
               %ContentPart{type: :text} = first,
               %ContentPart{type: :unsupported},
               %ContentPart{type: :text} = second
             ] = message.content

      assert ContentPart.narration?(first)
      assert ContentPart.narration?(second)
    end

    test "a message item without phase is left unmarked", %{model: model} do
      response = %{
        "status" => "completed",
        "output" => [
          %{"type" => "message", "content" => [%{"type" => "output_text", "text" => "Hi"}]}
        ]
      }

      message = ChatOpenAIResponses.do_process_response(model, response)

      assert [%ContentPart{options: []} = part] = message.content
      assert ContentPart.utterance(part) == nil
      refute Message.narration?(message)
    end
  end

  describe "streaming decode" do
    test "a recorded stream marks the same parts the completed response does", %{model: model} do
      body = File.read!(Path.join(@fixture_dir, "commentary_with_tool_calls_stream.txt"))
      streamed = stream_to_message(model, body)

      {events, _} = ChatOpenAIResponses.decode_stream({body, ""})
      %{"response" => completed} = Enum.find(events, &(&1["type"] == "response.completed"))
      decoded = ChatOpenAIResponses.do_process_response(model, completed)

      assert [{_text, "narration"}] = text_markers(streamed)
      assert text_markers(streamed) == text_markers(decoded)
      assert length(streamed.tool_calls) == length(decoded.tool_calls)
    end

    test "the marker is set before the first text delta arrives", %{model: model} do
      added = %{
        "type" => "response.output_item.added",
        "output_index" => 0,
        "item" => %{"type" => "message", "phase" => "commentary", "content" => []}
      }

      assert %MessageDelta{index: 0, content: %ContentPart{type: :text, content: nil} = marker} =
               marker_delta = ChatOpenAIResponses.do_process_response(model, added)

      assert ContentPart.narration?(marker)

      text =
        ChatOpenAIResponses.do_process_response(model, %{
          "type" => "response.output_text.delta",
          "output_index" => 0,
          "delta" => "Checking"
        })

      merged = MessageDelta.merge_deltas([marker_delta, text])
      assert [%ContentPart{content: "Checking"} = part] = merged.merged_content
      assert ContentPart.narration?(part)
    end

    test "commentary and final answer stream into separate parts", %{model: model} do
      item = fn index, phase ->
        %{"output_index" => index, "item" => %{"type" => "message", "phase" => phase}}
      end

      delta = fn index, text ->
        %{"type" => "response.output_text.delta", "output_index" => index, "delta" => text}
      end

      events = [
        Map.put(item.(0, "commentary"), "type", "response.output_item.added"),
        delta.(0, "Checking "),
        delta.(0, "now."),
        Map.put(item.(0, "commentary"), "type", "response.output_item.done"),
        Map.put(item.(1, "final_answer"), "type", "response.output_item.added"),
        delta.(1, "All "),
        delta.(1, "clear."),
        Map.put(item.(1, "final_answer"), "type", "response.output_item.done"),
        %{"type" => "response.completed", "response" => %{"id" => "resp_1"}}
      ]

      message = stream_to_message(model, sse(events))

      assert text_markers(message) == [{"Checking now.", "narration"}, {"All clear.", "answer"}]
      refute Message.narration?(message)
    end

    test "a message item without phase is skipped", %{model: model} do
      event = %{
        "type" => "response.output_item.added",
        "output_index" => 0,
        "item" => %{"type" => "message", "content" => []}
      }

      assert :skip == ChatOpenAIResponses.do_process_response(model, event)

      assert :skip ==
               ChatOpenAIResponses.do_process_response(model, %{
                 event
                 | "type" => "response.output_item.done"
               })
    end
  end

  describe "outbound" do
    test "narration then answer become two items with their phases", %{model: model} do
      message =
        Message.new_assistant!([
          ContentPart.narration!("Checking."),
          ContentPart.narration!("Still checking."),
          ContentPart.answer!("Done.")
        ])

      assert [
               %{
                 "type" => "message",
                 "role" => "assistant",
                 "phase" => "commentary",
                 "content" => [%{"text" => "Checking."}, %{"text" => "Still checking."}]
               },
               %{
                 "type" => "message",
                 "role" => "assistant",
                 "phase" => "final_answer",
                 "content" => [%{"text" => "Done."}]
               }
             ] = ChatOpenAIResponses.for_api(model, message)
    end

    test "unmarked text is one item with no phase, exactly as before", %{model: model} do
      message = Message.new_assistant!([ContentPart.text!("One"), ContentPart.text!("Two")])

      assert ChatOpenAIResponses.for_api(model, message) == [
               %{
                 "role" => "assistant",
                 "type" => "message",
                 "content" => [
                   %{"type" => "output_text", "text" => "One", "annotations" => []},
                   %{"type" => "output_text", "text" => "Two", "annotations" => []}
                 ]
               }
             ]
    end

    test "grouping keeps order when markers alternate", %{model: model} do
      message =
        Message.new_assistant!([
          ContentPart.narration!("A"),
          ContentPart.answer!("B"),
          ContentPart.narration!("C")
        ])

      assert [
               %{"phase" => "commentary", "content" => [%{"text" => "A"}]},
               %{"phase" => "final_answer", "content" => [%{"text" => "B"}]},
               %{"phase" => "commentary", "content" => [%{"text" => "C"}]}
             ] = ChatOpenAIResponses.for_api(model, message)
    end

    test "narration with tool calls sends the commentary item before the calls", %{model: model} do
      message =
        Message.new_assistant!(%{
          content: [ContentPart.narration!("Checking.")],
          tool_calls: [ToolCall.new!(%{call_id: "call_1", name: "inspect", arguments: %{}})]
        })

      assert [
               %{"type" => "message", "phase" => "commentary"},
               %{"type" => "function_call", "call_id" => "call_1"}
             ] = ChatOpenAIResponses.for_api(model, message)
    end
  end

  describe "round trip" do
    test "decoded message items re-encode with their original phases", %{model: model} do
      response = fixture("commentary_then_answer.json")
      message = ChatOpenAIResponses.do_process_response(model, response)

      original =
        for item <- output_messages(response),
            do: {item["phase"], Enum.map(item["content"], & &1["text"])}

      replayed =
        for item <- ChatOpenAIResponses.for_api(model, message),
            item["type"] == "message",
            do: {item["phase"], Enum.map(item["content"], & &1["text"])}

      assert replayed == original
    end

    test "items re-encode in the order the API produced them", %{model: model} do
      response = fixture("two_commentary_items.json")

      shape = fn items ->
        for item <- items do
          case item do
            %{"type" => "message"} -> {"message", item["phase"]}
            %{"type" => "reasoning"} -> {"reasoning", item["id"]}
            %{"type" => "function_call"} -> {"function_call", item["call_id"]}
          end
        end
      end

      assert [
               {"message", "commentary"},
               {"reasoning", _},
               {"message", "commentary"} | _calls
             ] = original = shape.(response["output"])

      decoded = ChatOpenAIResponses.do_process_response(model, response)
      assert shape.(ChatOpenAIResponses.for_api(model, decoded)) == original

      streamed = stream_to_message(model, LangChain.ScriptedResponsesAdapter.sse(response))
      assert shape.(ChatOpenAIResponses.for_api(model, streamed)) == original
    end

    test "a streamed message re-encodes with its phase", %{model: model} do
      body = File.read!(Path.join(@fixture_dir, "commentary_with_tool_calls_stream.txt"))
      message = stream_to_message(model, body)

      assert [%{"type" => "message", "phase" => "commentary"} | calls] =
               ChatOpenAIResponses.for_api(model, message)

      assert Enum.all?(calls, &(&1["type"] == "function_call"))
    end
  end

  describe "a response cut off by max_output_tokens" do
    # The API reports a response it stopped early with `status: "incomplete"`
    # and the reason in `incomplete_details`. Text that finished streaming
    # before the cut-off does not make the response complete.
    setup do
      response =
        "../fixtures/openai_phase/synthesized_tool_loop.json"
        |> Path.expand(__DIR__)
        |> File.read!()
        |> Jason.decode!()
        |> Map.fetch!("truncated_after_commentary")

      %{response: response}
    end

    test "decodes as :length", %{model: model, response: response} do
      assert %Message{status: :length} = ChatOpenAIResponses.do_process_response(model, response)
    end

    test "a filtered response decodes as :content_filtered", %{model: model, response: response} do
      filtered = %{response | "incomplete_details" => %{"reason" => "content_filter"}}

      assert %Message{status: :content_filtered} =
               ChatOpenAIResponses.do_process_response(model, filtered)
    end

    test "streams as :length", %{model: model, response: response} do
      message = stream_to_message(model, LangChain.ScriptedResponsesAdapter.sse(response))

      assert message.status == :length

      assert [%ContentPart{type: :text} = part] =
               for(%{type: :text} = p <- message.content, do: p)

      assert ContentPart.narration?(part)
    end
  end

  describe "LLMChain" do
    test "a commentary-only turn is followed by another call that answers", %{model: model} do
      commentary_only =
        fixture("commentary_then_answer.json")
        |> Map.update!("output", fn output ->
          Enum.reject(output, &(&1["phase"] == "final_answer"))
        end)

      answer_only =
        fixture("commentary_then_answer.json")
        |> Map.update!("output", fn output ->
          Enum.reject(output, &(&1["phase"] == "commentary"))
        end)

      expect(ChatOpenAIResponses, :call, fn _model, _messages, _tools ->
        {:ok, [ChatOpenAIResponses.do_process_response(model, commentary_only)]}
      end)

      expect(ChatOpenAIResponses, :call, fn _model, messages, _tools ->
        # The narration is part of the history the second call replays.
        assert [%Message{role: :user}, %Message{role: :assistant} = narration] = messages
        assert Message.narration?(narration)
        {:ok, [ChatOpenAIResponses.do_process_response(model, answer_only)]}
      end)

      assert {:ok, chain} =
               %{llm: model}
               |> LLMChain.new!()
               |> LLMChain.add_message(Message.new_user!("Why did the deploy fail?"))
               |> LLMChain.run(mode: :while_needs_response)

      assert [%Message{role: :user}, narration, answer] = chain.messages
      assert Message.narration?(narration)
      refute Message.narration?(answer)
      assert chain.last_message == answer
      assert [%ContentPart{} = part] = answer.content
      assert ContentPart.utterance(part) == "answer"
    end
  end
end
