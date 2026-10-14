if Code.ensure_loaded?(ReqLLM) do
  defmodule LangChain.ChatModels.ChatReqLLMToolSearchTest do
    use LangChain.BaseCase
    use Mimic

    alias LangChain.Chains.LLMChain
    alias LangChain.ChatModels.ChatReqLLM
    alias LangChain.Function
    alias LangChain.LangChainError
    alias LangChain.Message
    alias LangChain.Message.ContentPart
    alias LangChain.Message.ToolCall
    alias LangChain.Message.ToolResult
    alias LangChain.MessageDelta
    alias LangChain.NativeTool
    alias ReqLLM.Providers.Anthropic

    @anthropic "anthropic:claude-sonnet-5-5"

    @server_tool_use %{
      "type" => "server_tool_use",
      "id" => "srvtoolu_01ABC123",
      "name" => "tool_search_tool_bm25",
      "input" => %{"query" => "devices"}
    }

    @search_result %{
      "type" => "tool_search_tool_result",
      "tool_use_id" => "srvtoolu_01ABC123",
      "content" => %{
        "type" => "tool_search_tool_search_result",
        "tool_references" => [%{"type" => "tool_reference", "tool_name" => "devices_list"}]
      }
    }

    @tool_use %{
      "type" => "tool_use",
      "id" => "toolu_1",
      "name" => "devices_list",
      "input" => %{"site" => "hq"}
    }

    defp function(name, opts \\ []) do
      Function.new!(%{
        name: name,
        description: "Tool #{name}",
        parameters_schema: %{"type" => "object", "properties" => %{}},
        function: fn _args, _context -> {:ok, "done"} end,
        options: opts
      })
    end

    defp req_llm_model(spec \\ @anthropic) do
      {:ok, model} = ReqLLM.model(spec)
      model
    end

    # Runs `call/3` with `ReqLLM.generate_text` stubbed, returning the context
    # and options ChatReqLLM handed to req_llm.
    defp captured_request(model_spec, messages, tools) do
      test_pid = self()

      stub(ReqLLM, :generate_text, fn _spec, context, opts ->
        send(test_pid, {:request, context, opts})
        {:error, :stop_here}
      end)

      ChatReqLLM.call(ChatReqLLM.new!(%{model: model_spec}), messages, tools)
      assert_received {:request, context, opts}
      {context, opts}
    end

    # The JSON body req_llm's Anthropic provider would send for that request.
    defp anthropic_body(messages, tools) do
      {context, opts} = captured_request(@anthropic, messages, tools)

      {:ok, request} =
        Anthropic.prepare_request(:chat, @anthropic, context, Keyword.put(opts, :api_key, "k"))

      request
      |> Anthropic.encode_body()
      |> then(& &1.options[:json])
      |> Jason.encode!()
      |> Jason.decode!()
    end

    defp decode(content, stop_reason \\ "tool_use") do
      {:ok, response} =
        Anthropic.Response.decode_response(
          %{
            "id" => "msg_1",
            "type" => "message",
            "role" => "assistant",
            "model" => "claude-sonnet-5-5",
            "content" => content,
            "stop_reason" => stop_reason,
            "usage" => %{"input_tokens" => 10, "output_tokens" => 5}
          },
          req_llm_model()
        )

      ChatReqLLM.do_process_response(ChatReqLLM.new!(%{model: @anthropic}), response)
    end

    describe "request" do
      test "a deferred function carries defer_loading for the Anthropic API" do
        %{"tools" => tools} =
          anthropic_body([Message.new_user!("hi")], [
            function("get_time"),
            function("devices_list", defer_loading: true)
          ])

        assert %{"name" => "devices_list", "defer_loading" => true} =
                 Enum.find(tools, &(&1["name"] == "devices_list"))

        refute Map.has_key?(Enum.find(tools, &(&1["name"] == "get_time")), "defer_loading")
      end

      test "the search tool becomes Anthropic's bm25 or regex tool" do
        deferred = function("devices_list", defer_loading: true)

        %{"tools" => tools} =
          anthropic_body([Message.new_user!("hi")], [NativeTool.tool_search(), deferred])

        assert %{"type" => "tool_search_tool_bm25_20251119"} =
                 Enum.find(tools, &(&1["name"] == "tool_search_tool_bm25"))

        %{"tools" => tools} =
          anthropic_body([Message.new_user!("hi")], [
            NativeTool.tool_search(variant: :regex),
            deferred
          ])

        assert Enum.any?(tools, &(&1["type"] == "tool_search_tool_regex_20251119"))
      end

      test "the search option joins provider options set through provider_opts" do
        test_pid = self()

        stub(ReqLLM, :generate_text, fn _spec, _context, opts ->
          send(test_pid, {:opts, opts})
          {:error, :stop_here}
        end)

        model =
          ChatReqLLM.new!(%{model: @anthropic, provider_opts: %{provider_options: [top_k: 5]}})

        ChatReqLLM.call(model, [Message.new_user!("hi")], [
          NativeTool.tool_search(),
          function("devices_list", defer_loading: true)
        ])

        assert_received {:opts, opts}
        assert opts[:provider_options][:top_k] == 5
        assert opts[:provider_options][:tool_search] == %{variant: :bm25}
      end

      test "an OpenAI model gets the hosted search tool in its tools" do
        {_context, opts} =
          captured_request("openai:gpt-5.4", [Message.new_user!("hi")], [
            NativeTool.tool_search(),
            function("devices_list", defer_loading: true)
          ])

        assert %{"type" => "tool_search"} in opts[:tools]

        assert %ReqLLM.Tool{} = tool = Enum.find(opts[:tools], &match?(%ReqLLM.Tool{}, &1))
        assert ReqLLM.Tool.provider_options(tool, :openai) == %{defer_loading: true}
      end

      test "a provider without tool search drops the search tool" do
        {_context, opts} =
          captured_request("groq:llama-3.3-70b-versatile", [Message.new_user!("hi")], [
            NativeTool.tool_search(),
            function("devices_list", defer_loading: true)
          ])

        assert [%ReqLLM.Tool{name: "devices_list"}] = opts[:tools]
        refute opts[:provider_options][:tool_search]
      end

      test "every tool deferred on an Anthropic model is an error before sending" do
        reject(&ReqLLM.generate_text/3)

        assert {:error, %LangChainError{type: "invalid_tools"}} =
                 ChatReqLLM.call(
                   ChatReqLLM.new!(%{model: @anthropic}),
                   [Message.new_user!("hi")],
                   [
                     function("devices_list", defer_loading: true)
                   ]
                 )
      end

      test "another native tool is an error" do
        assert {:error, %LangChainError{message: message}} =
                 ChatReqLLM.call(
                   ChatReqLLM.new!(%{model: @anthropic}),
                   [Message.new_user!("hi")],
                   [
                     NativeTool.new!(%{name: "web_search", configuration: %{}})
                   ]
                 )

        assert message =~ "does not support the native tool"
      end
    end

    describe "non-streamed response" do
      test "search blocks become provider items in order, and text after them stays apart" do
        message =
          decode([
            %{"type" => "text", "text" => "Searching."},
            @server_tool_use,
            @search_result,
            %{"type" => "text", "text" => "Found it."},
            @tool_use
          ])

        assert %Message{
                 content: [
                   %ContentPart{type: :text, content: "Searching."},
                   %ContentPart{type: :provider_item} = use_part,
                   %ContentPart{type: :provider_item} = result_part,
                   %ContentPart{type: :text, content: "Found it."}
                 ],
                 tool_calls: [%ToolCall{call_id: "toolu_1", name: "devices_list"}]
               } = message

        assert ContentPart.provider_item?(use_part, "anthropic")
        assert ContentPart.provider_item_block(use_part) == @server_tool_use
        assert ContentPart.provider_item_block(result_part) == @search_result
      end

      test "the replayed assistant turn carries the blocks unchanged and in place" do
        message =
          decode([
            %{"type" => "text", "text" => "Searching."},
            @server_tool_use,
            @search_result,
            %{"type" => "text", "text" => "Found it."},
            @tool_use
          ])

        %{"messages" => [_user, %{"role" => "assistant", "content" => content} | _]} =
          anthropic_body(
            [
              Message.new_user!("hi"),
              message,
              Message.new_tool_result!(%{
                tool_results: [
                  ToolResult.new!(%{tool_call_id: "toolu_1", name: "devices_list", content: "17"})
                ]
              })
            ],
            [NativeTool.tool_search(), function("devices_list", defer_loading: true)]
          )

        assert [
                 %{"type" => "text", "text" => "Searching."},
                 @server_tool_use,
                 @search_result,
                 %{"type" => "text", "text" => "Found it."},
                 %{"type" => "tool_use", "id" => "toolu_1"}
               ] = content
      end

      test "pause_turn completes the message and keeps the turn open" do
        message = decode([@server_tool_use], "pause_turn")

        assert %Message{status: :complete, metadata: %{end_turn: false}} = message
        assert Message.continues_turn?(message)
      end
    end

    describe "streamed response" do
      # Feeds raw Anthropic stream events through req_llm's decoder, then
      # through ChatReqLLM's stream handling, and returns the merged message.
      defp stream_message(events) do
        model = req_llm_model()

        {chunks, state} =
          Enum.reduce(events, {[], nil}, fn event, {acc, state} ->
            {chunks, state} = Anthropic.Response.decode_stream_event(%{data: event}, model, state)
            {acc ++ chunks, state}
          end)

        {flushed, _state} = Anthropic.Response.flush_stream_state(model, state)

        stub(ReqLLM, :stream_text, fn _spec, _context, _opts ->
          {:ok,
           %ReqLLM.StreamResponse{
             stream: chunks ++ flushed,
             metadata_handle: self(),
             cancel: fn -> :ok end,
             model: nil,
             context: ReqLLM.Context.new([])
           }}
        end)

        llm = ChatReqLLM.new!(%{model: @anthropic, stream: true})
        {:ok, deltas} = ChatReqLLM.call(llm, [Message.new_user!("hi")], [])

        {:ok, message} = deltas |> MessageDelta.merge_deltas() |> MessageDelta.to_message()
        message
      end

      defp start_event do
        %{
          "type" => "message_start",
          "message" => %{
            "id" => "msg_1",
            "type" => "message",
            "role" => "assistant",
            "model" => "claude-sonnet-5-5",
            "content" => [],
            "usage" => %{"input_tokens" => 10, "output_tokens" => 1}
          }
        }
      end

      defp text_block(index, text) do
        [
          %{
            "type" => "content_block_start",
            "index" => index,
            "content_block" => %{"type" => "text", "text" => ""}
          },
          %{
            "type" => "content_block_delta",
            "index" => index,
            "delta" => %{"type" => "text_delta", "text" => text}
          },
          %{"type" => "content_block_stop", "index" => index}
        ]
      end

      defp search_events(first_index) do
        [
          %{
            "type" => "content_block_start",
            "index" => first_index,
            "content_block" => Map.put(@server_tool_use, "input", %{})
          },
          %{
            "type" => "content_block_delta",
            "index" => first_index,
            "delta" => %{"type" => "input_json_delta", "partial_json" => ~s({"query":)}
          },
          %{
            "type" => "content_block_delta",
            "index" => first_index,
            "delta" => %{"type" => "input_json_delta", "partial_json" => ~s("devices"})}
          },
          %{"type" => "content_block_stop", "index" => first_index},
          %{
            "type" => "content_block_start",
            "index" => first_index + 1,
            "content_block" => @search_result
          },
          %{"type" => "content_block_stop", "index" => first_index + 1}
        ]
      end

      defp stop_events(stop_reason) do
        [
          %{
            "type" => "message_delta",
            "delta" => %{"stop_reason" => stop_reason},
            "usage" => %{"output_tokens" => 20}
          },
          %{"type" => "message_stop"}
        ]
      end

      test "search blocks keep their place between the text around them" do
        events =
          [start_event()] ++
            text_block(0, "Searching.") ++
            search_events(1) ++ text_block(3, "Found it.") ++ stop_events("end_turn")

        assert %Message{
                 status: :complete,
                 content: [
                   %ContentPart{type: :text, content: "Searching."},
                   %ContentPart{type: :provider_item} = use_part,
                   %ContentPart{type: :provider_item} = result_part,
                   %ContentPart{type: :text, content: "Found it."}
                 ],
                 tool_calls: []
               } = stream_message(events)

        assert ContentPart.provider_item_block(use_part) == @server_tool_use
        assert ContentPart.provider_item_block(result_part) == @search_result
      end

      defp thinking_block(index, text, signature) do
        [
          %{
            "type" => "content_block_start",
            "index" => index,
            "content_block" => %{"type" => "thinking", "thinking" => "", "signature" => ""}
          },
          %{
            "type" => "content_block_delta",
            "index" => index,
            "delta" => %{"type" => "thinking_delta", "thinking" => text}
          },
          %{
            "type" => "content_block_delta",
            "index" => index,
            "delta" => %{"type" => "signature_delta", "signature" => signature}
          },
          %{"type" => "content_block_stop", "index" => index}
        ]
      end

      test "thinking after the search blocks replays in place, once" do
        events =
          [start_event()] ++
            thinking_block(0, "I should search.", "sig_1") ++
            search_events(1) ++
            thinking_block(3, "Use devices_list.", "sig_2") ++
            text_block(4, "Found it.") ++ stop_events("end_turn")

        message = stream_message(events)

        # The later thinking streams for display and is also kept whole as a
        # provider item, which is what goes back.
        assert [
                 %ContentPart{type: :thinking, content: "I should search."},
                 %ContentPart{type: :provider_item},
                 %ContentPart{type: :provider_item},
                 %ContentPart{type: :thinking, content: "Use devices_list.", options: display},
                 %ContentPart{type: :provider_item} = later_thinking,
                 %ContentPart{type: :text, content: "Found it."}
               ] = Enum.reject(message.content, &match?(%ContentPart{type: :unsupported}, &1))

        assert Keyword.get(display, :stream_only) == true

        assert %{"type" => "thinking", "thinking" => "Use devices_list.", "signature" => "sig_2"} =
                 ContentPart.provider_item_block(later_thinking)

        %{"messages" => [_user, %{"role" => "assistant", "content" => content} | _]} =
          anthropic_body(
            [Message.new_user!("hi"), message, Message.new_user!("And the warehouse?")],
            [NativeTool.tool_search(), function("devices_list", defer_loading: true)]
          )

        assert [
                 %{
                   "type" => "thinking",
                   "thinking" => "I should search.",
                   "signature" => "sig_1"
                 },
                 @server_tool_use,
                 @search_result,
                 %{
                   "type" => "thinking",
                   "thinking" => "Use devices_list.",
                   "signature" => "sig_2"
                 },
                 %{"type" => "text", "text" => "Found it."}
               ] = content
      end

      test "pause_turn on the stream keeps the turn open" do
        events = [start_event()] ++ search_events(0) ++ stop_events("pause_turn")

        assert %Message{status: :complete, metadata: %{end_turn: false}} =
                 message = stream_message(events)

        assert Message.continues_turn?(message)
      end
    end

    describe "thinking after a search block" do
      test "streams for display and is skipped on replay without a warning" do
        part =
          ContentPart.new!(%{
            type: :thinking,
            content: "Use the device tool.",
            options: [stream_only: true]
          })

        log =
          ExUnit.CaptureLog.capture_log(fn ->
            assert ChatReqLLM.content_part_to_req_llm(part) == nil
          end)

        refute log =~ "without signature"
      end

      test "a stream-only thinking chunk becomes a marked thinking part" do
        chunk =
          ReqLLM.StreamChunk.thinking("Use the device tool.", %{stream_only?: true})

        stub(ReqLLM, :stream_text, fn _spec, _context, _opts ->
          {:ok,
           %ReqLLM.StreamResponse{
             stream: [chunk, ReqLLM.StreamChunk.meta(%{finish_reason: :stop, terminal?: true})],
             metadata_handle: self(),
             cancel: fn -> :ok end,
             model: nil,
             context: ReqLLM.Context.new([])
           }}
        end)

        llm = ChatReqLLM.new!(%{model: @anthropic, stream: true})
        {:ok, deltas} = ChatReqLLM.call(llm, [Message.new_user!("hi")], [])
        {:ok, message} = deltas |> MessageDelta.merge_deltas() |> MessageDelta.to_message()

        assert %Message{content: [%ContentPart{type: :thinking, options: options}]} = message
        assert Keyword.get(options, :stream_only) == true
      end
    end

    describe "provider items on replay" do
      test "an item for a provider req_llm does not know is dropped" do
        part = ContentPart.provider_item!("no_such_provider_xyz", %{"type" => "x"})
        assert ChatReqLLM.content_part_to_req_llm(part) == nil
      end

      test "an item from ChatAnthropic's stream, with its input in content, replays whole" do
        part =
          ContentPart.new!(%{
            type: :provider_item,
            content: ~s({"query":"devices"}),
            options: [provider: "anthropic", item: Map.put(@server_tool_use, "input", %{})]
          })

        assert %ReqLLM.Message.ContentPart{type: :provider_block, data: @server_tool_use} =
                 ChatReqLLM.content_part_to_req_llm(part)
      end
    end

    describe "tool references" do
      defp load_result do
        Message.new_tool_result!(%{
          tool_results: [
            ToolResult.new!(%{
              tool_call_id: "toolu_1",
              name: "load_tools",
              content: [
                ContentPart.text!("Loaded the devices tools."),
                ContentPart.tool_reference!("devices_list")
              ]
            }),
            ToolResult.new!(%{tool_call_id: "toolu_2", name: "get_time", content: "noon"})
          ]
        })
      end

      defp load_call do
        Message.new_assistant!(%{
          tool_calls: [
            ToolCall.new!(%{
              call_id: "toolu_1",
              name: "load_tools",
              arguments: %{},
              status: :complete
            }),
            ToolCall.new!(%{
              call_id: "toolu_2",
              name: "get_time",
              arguments: %{},
              status: :complete
            })
          ]
        })
      end

      test "on an Anthropic model, fill the tool result, with the other content after" do
        %{"messages" => messages} =
          anthropic_body(
            [Message.new_user!("hi"), load_call(), load_result()],
            [function("load_tools"), function("devices_list", defer_loading: true)]
          )

        # req_llm encodes one tool message per result and merges them into one
        # user turn. The text follows as its own user message, which the API
        # joins to the same turn.
        assert [
                 _user,
                 _assistant,
                 %{
                   "role" => "user",
                   "content" => [
                     %{
                       "type" => "tool_result",
                       "tool_use_id" => "toolu_1",
                       "content" => [%{"type" => "tool_reference", "tool_name" => "devices_list"}]
                     },
                     %{"type" => "tool_result", "tool_use_id" => "toolu_2"}
                   ]
                 },
                 %{"role" => "user", "content" => "Loaded the devices tools."}
               ] = messages
      end

      test "on another provider, read as text inside the tool result" do
        context =
          ChatReqLLM.messages_to_req_llm_context(
            [load_result()],
            ChatReqLLM.new!(%{model: "openai:gpt-5.4"})
          )

        assert [
                 %ReqLLM.Message{role: :tool, content: content},
                 %ReqLLM.Message{role: :tool}
               ] = context.messages

        assert Enum.map(content, & &1.text) == [
                 "Loaded the devices tools.",
                 "Tool available: devices_list"
               ]
      end
    end

    describe "anthropic_model?/1" do
      test "recognizes Anthropic models directly and on Bedrock" do
        assert ChatReqLLM.anthropic_model?("anthropic:claude-sonnet-5-5")
        assert ChatReqLLM.anthropic_model?("amazon_bedrock:anthropic.claude-sonnet-5-5")
        assert ChatReqLLM.anthropic_model?("amazon_bedrock:us.anthropic.claude-sonnet-5-5")
        refute ChatReqLLM.anthropic_model?("amazon_bedrock:meta.llama3")
        refute ChatReqLLM.anthropic_model?("openai:gpt-5.4")
        refute ChatReqLLM.anthropic_model?(nil)
      end
    end

    describe "pause_turn in a chain" do
      test "the chain calls the model again with the paused content" do
        paused_response = fn ->
          {:ok, response} =
            Anthropic.Response.decode_response(
              %{
                "id" => "msg_1",
                "type" => "message",
                "role" => "assistant",
                "model" => "claude-sonnet-5-5",
                "content" => [@server_tool_use],
                "stop_reason" => "pause_turn",
                "usage" => %{"input_tokens" => 10, "output_tokens" => 5}
              },
              req_llm_model()
            )

          response
        end

        ReqLLM
        |> expect(:generate_text, fn _spec, _context, _opts -> {:ok, paused_response.()} end)
        |> expect(:generate_text, fn _spec, context, _opts ->
          assert %ReqLLM.Message{role: :assistant, content: [%{type: :provider_block}]} =
                   List.last(context.messages)

          {:ok, response} =
            Anthropic.Response.decode_response(
              %{
                "id" => "msg_2",
                "type" => "message",
                "role" => "assistant",
                "model" => "claude-sonnet-5-5",
                "content" => [%{"type" => "text", "text" => "All done."}],
                "stop_reason" => "end_turn",
                "usage" => %{"input_tokens" => 10, "output_tokens" => 5}
              },
              req_llm_model()
            )

          {:ok, response}
        end)

        {:ok, chain} =
          %{llm: ChatReqLLM.new!(%{model: @anthropic})}
          |> LLMChain.new!()
          |> LLMChain.add_message(Message.new_user!("Find the device tools."))
          |> LLMChain.run(mode: :while_needs_response)

        assert ContentPart.content_to_string(chain.last_message.content) == "All done."
      end
    end
  end
end
