if Code.ensure_loaded?(ReqLLM) do
  defmodule LangChain.ChatModels.ChatReqLLMPhaseTest do
    @moduledoc """
    The narration marker, decoded from recorded Responses API bodies through
    `req_llm` rather than from structs built by hand.

    `req_llm` reports the assistant `phase` on each content part, and on each
    streamed text chunk along with the index of the item it belongs to. Going
    through its own decoders, streamed and not, is what pins down what these
    bodies produce and what `ChatReqLLM` makes of it.
    """
    use LangChain.BaseCase
    use Mimic

    alias LangChain.ChatModels.ChatReqLLM
    alias LangChain.Message
    alias LangChain.Message.ContentPart
    alias LangChain.MessageDelta

    setup :verify_on_exit!

    @fixture_dir Path.expand("../fixtures/openai_phase", __DIR__)

    setup do
      %{model: ChatReqLLM.new!(%{model: "openai:gpt-5.4"})}
    end

    # Runs a recorded body through the provider's own decode step, so the
    # message these tests read is the one a live call would produce.
    defp decode_fixture(name) do
      body = @fixture_dir |> Path.join(name) |> File.read!() |> Jason.decode!()
      model = %LLMDB.Model{provider: :openai, id: body["model"]}

      req = Req.new(url: "/responses")
      req = %{req | options: Map.merge(req.options, %{model: model, operation: :chat})}
      resp = %Req.Response{status: 200, body: body, headers: %{}}

      {_req, decoded} = ReqLLM.Providers.OpenAI.ResponsesAPI.decode_response({req, resp})
      decoded.body
    end

    defp fixture_output(name) do
      @fixture_dir |> Path.join(name) |> File.read!() |> Jason.decode!() |> Map.fetch!("output")
    end

    defp message_item_texts(name) do
      for item <- fixture_output(name),
          item["type"] == "message",
          do: Enum.map_join(item["content"], "", & &1["text"])
    end

    defp fake_stream_response(chunks) do
      %ReqLLM.StreamResponse{
        stream: chunks,
        metadata_handle: self(),
        cancel: fn -> :ok end,
        model: nil,
        context: ReqLLM.Context.new([])
      }
    end

    # The server-sent events of a recorded body, streamed through the
    # provider's own stream decoder: each message item is announced with its
    # phase, its text streams, and the completed response closes the stream.
    defp stream_fixture_chunks(name) do
      body = @fixture_dir |> Path.join(name) |> File.read!() |> Jason.decode!()
      model = %LLMDB.Model{provider: :openai, id: body["model"]}
      decoder = ReqLLM.Providers.OpenAI.ResponsesAPI

      item_events =
        body["output"]
        |> Enum.with_index()
        |> Enum.flat_map(fn
          {%{"type" => "message"} = item, index} ->
            text = Enum.map_join(item["content"], "", & &1["text"])

            [
              %{
                "type" => "response.output_item.added",
                "output_index" => index,
                "item" => Map.put(item, "content", [])
              },
              %{
                "type" => "response.output_text.delta",
                "output_index" => index,
                "content_index" => 0,
                "item_id" => item["id"],
                "delta" => text
              },
              %{"type" => "response.output_item.done", "output_index" => index, "item" => item}
            ]

          {item, index} ->
            [
              %{"type" => "response.output_item.added", "output_index" => index, "item" => item},
              %{"type" => "response.output_item.done", "output_index" => index, "item" => item}
            ]
        end)

      events = item_events ++ [%{"type" => "response.completed", "response" => body}]

      {chunks, _state} =
        Enum.flat_map_reduce(events, decoder.init_stream_state(), fn data, state ->
          decoder.decode_stream_event(%{data: data}, model, state)
        end)

      chunks
    end

    defp stream_fixture(model, name) do
      chunks = stream_fixture_chunks(name)

      stub(ReqLLM, :stream_text, fn _model, _context, _opts ->
        {:ok, fake_stream_response(chunks)}
      end)

      %{model | stream: true}
      |> ChatReqLLM.do_api_request([Message.new_user!("why did it fail")], [], 3)
      |> MessageDelta.merge_deltas()
    end

    # The assistant message items req_llm's Responses encoder sends for a
    # message, as `{phase, text}`.
    defp sent_assistant_items(%Message{} = message) do
      context = ChatReqLLM.messages_to_req_llm_context([Message.new_user!("hi"), message])
      body = ReqLLM.Providers.OpenAI.ResponsesAPI.build_request_body(context, "gpt-5.4", [], nil)

      for %{"role" => "assistant", "content" => content} = item <- body["input"] do
        {item["phase"], Enum.map_join(content, "", & &1["text"])}
      end
    end

    describe "non-streaming, from recorded bodies" do
      test "a commentary item and an answer item arrive as separate labeled parts",
           %{model: model} do
        # Without the labels, the preamble would read as the start of the answer.
        response = decode_fixture("commentary_then_answer.json")
        [commentary, answer] = message_item_texts("commentary_then_answer.json")

        assert [
                 %ReqLLM.Message.ContentPart{text: ^commentary, metadata: %{phase: "commentary"}},
                 %ReqLLM.Message.ContentPart{text: ^answer, metadata: %{phase: "final_answer"}}
               ] = response.message.content

        message = ChatReqLLM.do_process_response(model, response)

        assert [narration_part, answer_part] = message.content
        assert narration_part.content == commentary
        assert ContentPart.utterance(narration_part) == "narration"
        assert answer_part.content == answer
        assert ContentPart.utterance(answer_part) == "answer"

        # The answer is reachable on its own, which is what a caller parsing
        # structured output needs.
        refute Message.narration?(message)
        assert Message.answer_content(message) == answer
      end

      test "several commentary items in one response stay separate parts", %{model: model} do
        response = decode_fixture("two_commentary_items.json")
        texts = message_item_texts("two_commentary_items.json")

        assert length(texts) == 2

        message = ChatReqLLM.do_process_response(model, response)

        text_parts = for %ContentPart{type: :text} = part <- message.content, do: part

        assert Enum.map(text_parts, & &1.content) == texts
        assert Enum.map(text_parts, &ContentPart.utterance/1) == ["narration", "narration"]
      end

      test "commentary sent with tool calls is a tool call, not narration", %{model: model} do
        response = decode_fixture("commentary_with_tool_calls.json")

        assert [%ReqLLM.Message.ContentPart{metadata: %{phase: "commentary"}}] =
                 response.message.content

        message = ChatReqLLM.do_process_response(model, response)

        assert [%ContentPart{type: :text} = part] = message.content
        assert ContentPart.narration?(part)
        assert Message.is_tool_call?(message)
        refute Message.narration?(message)
      end

      test "a recorded turn re-encodes to the phases it arrived with", %{model: model} do
        message =
          ChatReqLLM.do_process_response(model, decode_fixture("commentary_then_answer.json"))

        [commentary, answer] = message_item_texts("commentary_then_answer.json")

        assert sent_assistant_items(message) == [
                 {"commentary", commentary},
                 {"final_answer", answer}
               ]
      end
    end

    describe "streaming, from recorded bodies" do
      test "a commentary item and an answer item stream as separate labeled parts",
           %{model: model} do
        [commentary, answer] = message_item_texts("commentary_then_answer.json")

        merged = stream_fixture(model, "commentary_then_answer.json")

        assert %MessageDelta{status: :complete} = merged
        assert {:ok, message} = MessageDelta.to_message(merged)

        assert [narration_part, answer_part] = message.content
        assert narration_part.content == commentary
        assert ContentPart.utterance(narration_part) == "narration"
        assert answer_part.content == answer
        assert ContentPart.utterance(answer_part) == "answer"
        refute Message.narration?(message)
      end

      test "several commentary items stream as separate parts", %{model: model} do
        texts = message_item_texts("two_commentary_items.json")

        assert {:ok, message} =
                 model
                 |> stream_fixture("two_commentary_items.json")
                 |> MessageDelta.to_message()

        text_parts = for %ContentPart{type: :text} = part <- message.content, do: part

        assert Enum.map(text_parts, & &1.content) == texts
        assert Enum.map(text_parts, &ContentPart.utterance/1) == ["narration", "narration"]
      end

      test "a streamed turn re-encodes to the phases it arrived with", %{model: model} do
        [commentary, answer] = message_item_texts("commentary_then_answer.json")

        assert {:ok, message} =
                 model
                 |> stream_fixture("commentary_then_answer.json")
                 |> MessageDelta.to_message()

        assert sent_assistant_items(message) == [
                 {"commentary", commentary},
                 {"final_answer", answer}
               ]
      end
    end
  end
end
