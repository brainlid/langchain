defmodule LangChain.Chains.NarrationToolLoopTest do
  @moduledoc """
  A long tool loop with narration between the tool calls, run through
  `LLMChain` against each adapter that speaks the Responses API, streaming and
  not.

  The fixture's `turns` are six responses shaped like the recorded ones:
  turns that narrate and call tools in one response, a turn whose only output
  is a commentary item, a turn holding two commentary items and no tool calls,
  and a closing answer. Its `after_narration` responses are what a call made
  after a commentary-only turn can return instead of acting: a reasoning item
  alone, no output at all, or a message with no `phase`.

  Only the HTTP layer is replaced. `ChatOpenAIResponses` requests reach
  `LangChain.ScriptedResponsesAdapter`, which replays each body, as SSE when
  the request streams. `ChatReqLLM` gets the same bodies run through req_llm's
  own response and stream decoders. Both adapters therefore decode what a live
  call would hand them.
  """
  use LangChain.BaseCase
  use Mimic

  alias LangChain.Chains.LLMChain
  alias LangChain.ChatModels.ChatOpenAIResponses
  alias LangChain.Function
  alias LangChain.LangChainError
  alias LangChain.Message
  alias LangChain.Message.ContentPart
  alias LangChain.Message.ToolCall
  alias LangChain.ScriptedResponsesAdapter

  setup :verify_on_exit!

  @fixture "../fixtures/openai_phase/synthesized_tool_loop.json"
           |> Path.expand(__DIR__)
           |> File.read!()
           |> Jason.decode!()

  @turns @fixture["turns"]
  @after_narration @fixture["after_narration"]
  @truncated @fixture["truncated"]

  # The first three turns end on the commentary-only response.
  @through_narration Enum.take(@turns, 3)
  @answer List.last(@turns)

  @paths [:responses, :responses_stream] ++
           if(Code.ensure_loaded?(ReqLLM), do: [:req_llm, :req_llm_stream], else: [])

  @tool_names ~w(list_deployments search_logs get_metrics inspect_resource read_file)

  defp tools(test_pid) do
    for name <- @tool_names do
      Function.new!(%{
        name: name,
        description: "Look something up for the investigation (#{name}).",
        parameters_schema: %{type: "object", properties: %{}},
        function: fn args, _context ->
          send(test_pid, {:tool_ran, name, args})
          {:ok, "#{name} result"}
        end
      })
    end
  end

  # Script `bodies`, run the chain on `path`, and return the result. The mode
  # defaults to `:while_needs_response`.
  defp run_path(path, bodies, opts \\ []) do
    ScriptedResponsesAdapter.script(bodies)

    %{llm: model(path)}
    |> LLMChain.new!()
    |> LLMChain.add_tools(tools(self()))
    |> LLMChain.add_message(
      Message.new_user!("Checkout has been failing since this morning. Why?")
    )
    |> LLMChain.run(Keyword.put_new(opts, :mode, :while_needs_response))
  end

  defp model(:responses), do: responses_model(false)
  defp model(:responses_stream), do: responses_model(true)
  defp model(:req_llm), do: req_llm_model(false)
  defp model(:req_llm_stream), do: req_llm_model(true)

  defp responses_model(stream) do
    ChatOpenAIResponses.new!(%{
      model: "gpt-5.4",
      api_key: "test",
      stream: stream,
      req_config: %{adapter: ScriptedResponsesAdapter, retry: false}
    })
  end

  if Code.ensure_loaded?(ReqLLM) do
    alias LangChain.ChatModels.ChatReqLLM
    alias ReqLLM.Providers.OpenAI.ResponsesAPI

    defp req_llm_model(false = stream) do
      stub(ReqLLM, :generate_text, fn _model, context, _opts ->
        {:ok, context |> ScriptedResponsesAdapter.next_body() |> req_llm_decode()}
      end)

      ChatReqLLM.new!(%{model: "openai:gpt-5.4", stream: stream})
    end

    defp req_llm_model(true = stream) do
      stub(ReqLLM, :stream_text, fn _model, context, _opts ->
        {:ok, context |> ScriptedResponsesAdapter.next_body() |> req_llm_stream()}
      end)

      ChatReqLLM.new!(%{model: "openai:gpt-5.4", stream: stream})
    end

    defp llmdb_model, do: %LLMDB.Model{provider: :openai, id: "gpt-5.4"}

    defp req_llm_decode(body) do
      req = Req.new(url: "/responses")
      req = %{req | options: Map.merge(req.options, %{model: llmdb_model(), operation: :chat})}
      resp = %Req.Response{status: 200, body: body, headers: %{}}
      {_req, decoded} = ResponsesAPI.decode_response({req, resp})
      decoded.body
    end

    defp req_llm_stream(body) do
      {chunks, _state} =
        body
        |> ScriptedResponsesAdapter.sse_events()
        |> Enum.flat_map_reduce(nil, fn event, state ->
          ResponsesAPI.decode_stream_event(
            %{event: event["type"], data: event},
            llmdb_model(),
            state
          )
        end)

      # req_llm collects the finish reason from the terminal chunk in a
      # separate process, which `ChatReqLLM` asks when the chunks it translated
      # did not close the turn.
      terminal =
        Enum.find_value(chunks, %{}, fn
          %ReqLLM.StreamChunk{type: :meta, metadata: %{terminal?: true} = meta} -> meta
          _chunk -> nil
        end)

      {:ok, handle} = ReqLLM.StreamResponse.MetadataHandle.start_link(fn -> terminal end)

      %ReqLLM.StreamResponse{
        stream: chunks,
        metadata_handle: handle,
        cancel: fn -> :ok end,
        model: nil,
        context: ReqLLM.Context.new([])
      }
    end
  else
    defp req_llm_model(_stream), do: raise("req_llm is not available")
  end

  # The label on the last assistant message a request sent, in each client's
  # own vocabulary.
  defp last_sent_phase(path, sent) when path in [:responses, :responses_stream] do
    case List.last(sent["input"]) do
      %{"type" => "message", "role" => "assistant"} = item -> item["phase"]
      other -> {:not_an_assistant_message, other}
    end
  end

  defp last_sent_phase(_req_llm_path, context) do
    case List.last(context.messages) do
      %{role: :assistant, metadata: metadata} -> metadata[:phase]
      other -> {:not_an_assistant_message, other}
    end
  end

  # One entry per assistant message: whether the chain continued past it (on
  # tool calls, on narration, or on a reported open turn) or stopped, and the
  # utterance label on each text part.
  defp trajectory(%LLMChain{messages: messages}) do
    for %Message{role: :assistant} = message <- messages do
      cond do
        Message.is_tool_call?(message) ->
          {:tools, length(message.tool_calls), text_labels(message)}

        Message.continues_turn?(message) and Message.narration?(message) ->
          {:narration, text_labels(message)}

        Message.continues_turn?(message) ->
          {:continue, text_labels(message)}

        true ->
          {:stop, text_labels(message)}
      end
    end
  end

  defp text_labels(%Message{content: parts}) when is_list(parts) do
    for %ContentPart{type: :text} = part <- parts, do: ContentPart.utterance(part)
  end

  defp text_labels(_message), do: []

  # The last assistant text a user would see.
  defp last_visible_text(%LLMChain{messages: messages}) do
    messages
    |> Enum.filter(&(&1.role == :assistant))
    |> Enum.flat_map(fn %Message{content: parts} ->
      for %ContentPart{type: :text, content: text} = part <- parts || [],
          is_binary(text) and text != "",
          do: {text, ContentPart.utterance(part)}
    end)
    |> List.last()
  end

  # Forward the no-answer report to the test process.
  defp attach_no_answer_handler(context) do
    handler_id = "no-answer-#{inspect(context.test)}"
    test_pid = self()

    :telemetry.attach(
      handler_id,
      [:langchain, :chain, :turn, :no_answer],
      fn _event, _measurements, metadata, _config -> send(test_pid, {:no_answer, metadata}) end,
      nil
    )

    on_exit(fn -> :telemetry.detach(handler_id) end)
  end

  describe "a tool loop with narration" do
    setup :attach_no_answer_handler

    for path <- @paths do
      @tag path: path
      test "runs to the answer through commentary-only turns (#{path})", %{path: path} do
        assert {:ok, chain} = run_path(path, @turns)
        assert ScriptedResponsesAdapter.remaining() == []

        # Streamed through req_llm, the text of the two commentary items in
        # turn four arrives without an item boundary and merges into one part.
        fourth_turn_labels =
          if path == :req_llm_stream, do: ["narration"], else: ["narration", "narration"]

        assert trajectory(chain) == [
                 {:tools, 3, ["narration"]},
                 {:tools, 2, ["narration"]},
                 {:narration, ["narration"]},
                 {:narration, fourth_turn_labels},
                 {:tools, 1, ["narration"]},
                 {:stop, ["answer"]}
               ]

        assert Message.answer_content(chain.last_message) =~ "Restore the 30s timeout"

        for _call <- 1..6, do: assert_received({:tool_ran, _name, _args})
        refute_received {:tool_ran, _name, _args}

        # The request after the commentary-only turn ends on it, labelled.
        assert [_, _, _, after_commentary_only | _] = ScriptedResponsesAdapter.sent()
        assert last_sent_phase(path, after_commentary_only) == "commentary"

        refute_received {:no_answer, _}
      end
    end

    for path <- @paths do
      @tag path: path
      test "an empty response in a run that never narrated is not reported (#{path})", %{
        path: path
      } do
        assert {:ok, chain} = run_path(path, [@after_narration["empty"]])
        assert [{:stop, []}] = trajectory(chain)

        refute_received {:no_answer, _}
      end
    end
  end

  describe "a run never finishes on a commentary-only message" do
    @commentary_only Enum.at(@turns, 2)

    for path <- @paths, mode <- [:while_needs_response, :until_success] do
      @tag path: path, mode: mode
      test "it calls the model again and finishes on the answer (#{mode}, #{path})", %{
        path: path,
        mode: mode
      } do
        assert {:ok, chain} = run_path(path, [@commentary_only, @answer], mode: mode)
        assert ScriptedResponsesAdapter.remaining() == []

        assert [{:narration, ["narration"]}, {:stop, ["answer"]}] = trajectory(chain)
        assert Message.answer_content(chain.last_message) =~ "Restore the 30s timeout"
      end

      @tag path: path, mode: mode
      test "a model that only narrates exhausts max_runs (#{mode}, #{path})", %{
        path: path,
        mode: mode
      } do
        bodies = List.duplicate(@commentary_only, 3)

        assert {:error, chain, %LangChainError{type: "exceeded_max_runs"}} =
                 run_path(path, bodies, mode: mode, max_runs: 3)

        assert ScriptedResponsesAdapter.remaining() == []
        assert Message.narration?(chain.last_message)
      end
    end
  end

  describe "the call after a commentary-only turn returns no text" do
    # These follow-up shapes are synthesized, not captured: no recorded
    # response has been only a reasoning item or had no output at all. They
    # describe what the chain does if one arrives. The last message holds no
    # text, so it is neither narration nor an answer, and the run finishes on
    # it with the narration before it as the last visible text. The scripted
    # answer is never requested, and the run is reported.
    setup :attach_no_answer_handler

    for path <- @paths, shape <- ["reasoning_only", "empty"] do
      @tag path: path, shape: shape
      test "the run finishes on the empty message and is reported (#{shape}, #{path})", %{
        path: path,
        shape: shape
      } do
        bodies = @through_narration ++ [@after_narration[shape], @answer]

        assert {:ok, chain} = run_path(path, bodies)
        assert ScriptedResponsesAdapter.remaining() == [@answer]

        assert [
                 {:tools, 3, ["narration"]},
                 {:tools, 2, ["narration"]},
                 {:narration, ["narration"]},
                 {:stop, []}
               ] = trajectory(chain)

        assert {text, "narration"} = last_visible_text(chain)

        # The report carries the shapes but not the text.
        assert_received {:no_answer, %{message: message, last_narration: narration}}
        assert %{tool_call_count: 0, response_id: _} = message
        refute Enum.any?(message.parts, &(&1.type == :text))

        assert [%{type: :text, utterance: "narration", length: length}] =
                 Enum.filter(narration.parts, &(&1.type == :text))

        assert length == String.length(text)
        refute inspect(message) =~ "Before I call"
      end
    end

    # A message item with no `phase` is unmarked, which reads as an answer.
    for path <- @paths do
      @tag path: path
      test "an unlabelled message ends the run (#{path})", %{path: path} do
        bodies = @through_narration ++ [@after_narration["unphased"], @answer]

        assert {:ok, chain} = run_path(path, bodies)
        assert ScriptedResponsesAdapter.remaining() == [@answer]

        assert [_, _, {:narration, ["narration"]}, {:stop, [nil]}] = trajectory(chain)
        assert {"Let me check the gateway call timings.", nil} = last_visible_text(chain)

        # Unlabelled text reads as an answer, so the run is not reported.
        refute_received {:no_answer, _}
      end
    end
  end

  describe "a response cut off mid-tool-call" do
    # A response that reaches `max_output_tokens` or the context window comes
    # back with `status: "incomplete"`, possibly after its commentary item and
    # partway through a function call's arguments. The run ends as an error on
    # it: the tool call is partial and never runs, and the commentary is not
    # the model's finished turn. The truncated message stays in the chain.
    for path <- @paths do
      @tag path: path
      test "the run ends as response_truncated (#{path})", %{path: path} do
        assert {:error, chain, %LangChainError{type: "response_truncated"}} =
                 run_path(path, [@truncated])

        assert ScriptedResponsesAdapter.remaining() == []
        refute_received {:tool_ran, _name, _args}

        assert %Message{role: :assistant, status: :length} = message = chain.last_message
        assert [%ToolCall{status: :incomplete, name: "search_logs"}] = message.tool_calls
        refute Message.is_tool_call?(message)
      end
    end
  end

  describe "a completed response that reports end_turn" do
    # The report decides the turn boundary ahead of the narration marker, and
    # reaches the message through every adapter and decode path.
    for path <- @paths do
      @tag path: path
      test "end_turn true ends the run on a commentary-only turn (#{path})", %{path: path} do
        [first, second, commentary_only] = @through_narration
        bodies = [first, second, Map.put(commentary_only, "end_turn", true), @answer]

        assert {:ok, chain} = run_path(path, bodies)
        assert ScriptedResponsesAdapter.remaining() == [@answer]

        refute chain.needs_response
        assert [_, _, {:stop, ["narration"]}] = trajectory(chain)
        assert Message.end_turn(chain.last_message) == true
      end

      @tag path: path
      test "end_turn false continues past an unlabelled message (#{path})", %{path: path} do
        unphased = Map.put(@after_narration["unphased"], "end_turn", false)
        bodies = @through_narration ++ [unphased, @answer]

        assert {:ok, chain} = run_path(path, bodies)
        assert ScriptedResponsesAdapter.remaining() == []

        assert [_, _, {:narration, _}, {:continue, [nil]}, {:stop, ["answer"]}] =
                 trajectory(chain)

        assert Message.answer_content(chain.last_message) =~ "Restore the 30s timeout"
      end
    end
  end
end
