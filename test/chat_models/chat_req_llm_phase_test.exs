if Code.ensure_loaded?(ReqLLM) do
  defmodule LangChain.ChatModels.ChatReqLLMPhaseTest do
    @moduledoc """
    The narration marker, decoded from recorded Responses API bodies through
    `req_llm` rather than from structs built by hand.

    `req_llm` reports the assistant `phase` differently depending on its
    version. Going through its own decoder is what pins down which shape these
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

    defp stream_to_parts(model, chunks) do
      stub(ReqLLM, :stream_text, fn _model, _context, _opts ->
        {:ok, fake_stream_response(chunks)}
      end)

      model
      |> ChatReqLLM.do_api_request([Message.new_user!("why did it fail")], [], 3)
      |> MessageDelta.merge_deltas()
      |> Map.fetch!(:merged_content)
    end

    describe "non-streaming, from recorded bodies" do
      test "a commentary item and an answer item arrive as one joined part", %{model: model} do
        # The shape this repair exists for. The provider reports the two items'
        # text joined into a single part, with no separator between them, and
        # records the items themselves alongside it.
        response = decode_fixture("commentary_then_answer.json")
        [commentary, answer] = message_item_texts("commentary_then_answer.json")

        assert [%ReqLLM.Message.ContentPart{type: :text, text: joined}] = response.message.content
        assert joined == commentary <> answer

        assert %{phase_items: [%{"phase" => "commentary"}, %{"phase" => "final_answer"}]} =
                 response.message.metadata

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

        assert %{phase: "commentary"} = response.message.metadata

        message = ChatReqLLM.do_process_response(model, response)

        assert [%ContentPart{type: :text} = part] = message.content
        assert ContentPart.narration?(part)
        assert Message.is_tool_call?(message)
        refute Message.narration?(message)
      end

      test "a recorded turn re-encodes to the phases it arrived with", %{model: model} do
        [req_msg] =
          model
          |> ChatReqLLM.do_process_response(decode_fixture("commentary_then_answer.json"))
          |> ChatReqLLM.message_to_req_llm_messages()

        assert Enum.map(req_msg.content, & &1.metadata) == [
                 %{phase: "commentary"},
                 %{phase: "final_answer"}
               ]

        assert %{phase_items: [%{"phase" => "commentary"}, %{"phase" => "final_answer"}]} =
                 req_msg.metadata
      end
    end

    describe "streaming, with the terminal metadata a recorded body produces" do
      # Streamed text arrives with nothing marking where one message item ends,
      # so it merges into a single part before the terminal chunk names the
      # items. These tests use the terminal metadata the recorded bodies
      # produce, against that single merged part.
      defp terminal_chunk(fixture) do
        metadata =
          fixture
          |> decode_fixture()
          |> Map.fetch!(:message)
          |> Map.fetch!(:metadata)
          |> Map.take([:phase, :phase_items])
          |> Map.merge(%{finish_reason: :stop, terminal?: true})

        %ReqLLM.StreamChunk{type: :meta, metadata: metadata}
      end

      test "items that share a phase mark the merged part", %{model: model} do
        model = %{model | stream: true}

        chunks = [
          %ReqLLM.StreamChunk{type: :content, text: "Checking the deployment."},
          terminal_chunk("two_commentary_items.json")
        ]

        assert [part] = stream_to_parts(model, chunks)
        assert ContentPart.utterance(part) == "narration"
      end

      test "items that disagree leave the merged part unmarked", %{model: model} do
        model = %{model | stream: true}

        # The merged part holds an item of narration and an item of answer, so
        # no single marker is true of it.
        chunks = [
          %ReqLLM.StreamChunk{type: :content, text: "Progress report.The answer."},
          terminal_chunk("commentary_then_answer.json")
        ]

        assert [part] = stream_to_parts(model, chunks)
        assert ContentPart.utterance(part) == nil
      end

      test "a single reported phase marks the merged part", %{model: model} do
        model = %{model | stream: true}

        chunks = [
          %ReqLLM.StreamChunk{type: :content, text: "Looking into it."},
          terminal_chunk("commentary_with_tool_calls.json")
        ]

        assert [part] = stream_to_parts(model, chunks)
        assert ContentPart.utterance(part) == "narration"
      end

      test "the terminal chunk still closes the turn", %{model: model} do
        model = %{model | stream: true}

        chunks = [
          %ReqLLM.StreamChunk{type: :content, text: "Looking into it."},
          terminal_chunk("two_commentary_items.json")
        ]

        stub(ReqLLM, :stream_text, fn _model, _context, _opts ->
          {:ok, fake_stream_response(chunks)}
        end)

        deltas = ChatReqLLM.do_api_request(model, [Message.new_user!("hi")], [], 3)

        assert List.last(deltas).status == :complete
      end
    end
  end
end
