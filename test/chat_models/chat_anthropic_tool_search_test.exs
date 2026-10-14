defmodule LangChain.ChatModels.ChatAnthropicToolSearchTest do
  use LangChain.BaseCase
  use Mimic

  import ExUnit.CaptureLog

  alias LangChain.Chains.LLMChain
  alias LangChain.ChatModels.ChatAnthropic
  alias LangChain.Function
  alias LangChain.LangChainError
  alias LangChain.Message
  alias LangChain.Message.ContentPart
  alias LangChain.Message.ToolCall
  alias LangChain.Message.ToolResult
  alias LangChain.MessageDelta
  alias LangChain.NativeTool

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

  defp function(name, opts \\ []) do
    Function.new!(%{
      name: name,
      description: "Tool #{name}",
      parameters_schema: %{"type" => "object", "properties" => %{}},
      function: fn _args, _context -> {:ok, "done"} end,
      options: opts
    })
  end

  defp tools_body(tools) do
    %{tools: tools} =
      ChatAnthropic.for_api(ChatAnthropic.new!(%{}), [Message.new_user!("hi")], tools)

    tools
  end

  describe "request encoding" do
    test "a deferred function carries defer_loading" do
      assert [
               %{"name" => "devices_list", "defer_loading" => true},
               %{"name" => "get_time"} = plain
             ] =
               tools_body([
                 function("devices_list", defer_loading: true),
                 function("get_time")
               ])

      refute Map.has_key?(plain, "defer_loading")
    end

    test "the search tool encodes as the bm25 or regex variant" do
      deferred = function("devices_list", defer_loading: true)

      assert [%{"type" => "tool_search_tool_bm25_20251119", "name" => "tool_search_tool_bm25"}, _] =
               tools_body([NativeTool.tool_search(), deferred])

      assert [
               %{"type" => "tool_search_tool_regex_20251119", "name" => "tool_search_tool_regex"},
               _
             ] = tools_body([NativeTool.tool_search(variant: :regex), deferred])
    end

    test "another native tool is sent as its configuration with its name" do
      tool =
        NativeTool.new!(%{
          name: "web_search",
          configuration: %{"type" => "web_search_20250305", "max_uses" => 3}
        })

      assert [%{"type" => "web_search_20250305", "name" => "web_search", "max_uses" => 3}] =
               tools_body([tool])
    end

    test "a native tool without a type raises" do
      assert_raise LangChainError, ~r/needs a "type"/, fn ->
        tools_body([NativeTool.new!(%{name: "mystery", configuration: %{}})])
      end
    end

    test "the order of tools is kept" do
      tools = [
        function("a"),
        NativeTool.tool_search(),
        function("b", defer_loading: true),
        function("c")
      ]

      assert ["a", "tool_search_tool_bm25", "b", "c"] = Enum.map(tools_body(tools), & &1["name"])
    end

    test "every tool deferred raises before the request is sent" do
      assert_raise LangChainError, ~r/Every tool is deferred/, fn ->
        tools_body([function("a", defer_loading: true), function("b", defer_loading: true)])
      end
    end

    test "call/3 reports every tool deferred as an error" do
      model = ChatAnthropic.new!(%{})

      assert {:error, %LangChainError{type: "invalid_tools"}} =
               ChatAnthropic.call(model, [Message.new_user!("hi")], [
                 function("a", defer_loading: true)
               ])
    end

    test "cache_control is dropped from a deferred function, with a warning" do
      log =
        capture_log(fn ->
          assert [deferred, cached] =
                   tools_body([
                     function("devices_list", defer_loading: true, cache_control: true),
                     function("get_time", cache_control: true)
                   ])

          refute Map.has_key?(deferred, "cache_control")
          assert deferred["defer_loading"] == true
          assert cached["cache_control"] == %{"type" => "ephemeral"}
        end)

      assert log =~ "Deferred function \"devices_list\" has cache_control set"
    end
  end

  describe "non-streamed response" do
    defp response(content, stop_reason \\ "tool_use") do
      %{
        "role" => "assistant",
        "type" => "message",
        "content" => content,
        "stop_reason" => stop_reason,
        "usage" => %{"input_tokens" => 10, "output_tokens" => 5}
      }
    end

    test "search blocks become provider items in content order, never tool calls" do
      tool_use = %{
        "type" => "tool_use",
        "id" => "toolu_1",
        "name" => "devices_list",
        "input" => %{"site" => "hq"}
      }

      result =
        ChatAnthropic.do_process_response(
          ChatAnthropic.new!(%{}),
          response([
            %{"type" => "text", "text" => "Searching."},
            @server_tool_use,
            @search_result,
            %{"type" => "text", "text" => "Found it."},
            tool_use
          ])
        )

      assert %Message{
               content: [
                 %ContentPart{type: :text, content: "Searching."},
                 %ContentPart{type: :provider_item} = use_part,
                 %ContentPart{type: :provider_item} = result_part,
                 %ContentPart{type: :text, content: "Found it."}
               ],
               tool_calls: [%ToolCall{call_id: "toolu_1", name: "devices_list"}]
             } = result

      assert ContentPart.provider_item?(use_part, "anthropic")
      assert Keyword.get(use_part.options, :item) == @server_tool_use
      assert Keyword.get(result_part.options, :item) == @search_result
    end

    test "a replayed message sends the blocks back unchanged and in place" do
      thinking = %{"type" => "thinking", "thinking" => "Let me look.", "signature" => "sig1"}
      later_thinking = %{"type" => "thinking", "thinking" => "Use it.", "signature" => "sig2"}

      tool_use = %{
        "type" => "tool_use",
        "id" => "toolu_1",
        "name" => "devices_list",
        "input" => %{"site" => "hq"}
      }

      content = [thinking, @server_tool_use, @search_result, later_thinking, tool_use]

      message = ChatAnthropic.do_process_response(ChatAnthropic.new!(%{}), response(content))

      assert %{"role" => "assistant", "content" => ^content} =
               ChatAnthropic.message_for_api(message)
    end

    test "a search error result is kept as a provider item" do
      error_result = %{
        "type" => "tool_search_tool_result",
        "tool_use_id" => "srvtoolu_01ABC123",
        "content" => %{
          "type" => "tool_search_tool_result_error",
          "error_code" => "invalid_tool_input"
        }
      }

      assert %Message{content: [_use, %ContentPart{type: :provider_item} = part]} =
               ChatAnthropic.do_process_response(
                 ChatAnthropic.new!(%{}),
                 response([@server_tool_use, error_result], "end_turn")
               )

      assert Keyword.get(part.options, :item) == error_result
    end

    test "an unrecognized block is kept rather than raising" do
      block = %{"type" => "some_future_block", "data" => %{"x" => 1}}

      assert %Message{content: [%ContentPart{type: :provider_item} = part]} =
               ChatAnthropic.do_process_response(
                 ChatAnthropic.new!(%{}),
                 response([block], "end_turn")
               )

      assert Keyword.get(part.options, :item) == block
    end

    test "pause_turn completes the message but keeps the turn open" do
      message =
        ChatAnthropic.do_process_response(
          ChatAnthropic.new!(%{}),
          response([@server_tool_use], "pause_turn")
        )

      assert %Message{status: :complete, metadata: %{end_turn: false}} = message
      assert Message.continues_turn?(message)
    end
  end

  describe "provider items on replay" do
    test "an item from another provider is left out" do
      message =
        Message.new_assistant!(%{
          content: [
            ContentPart.text!("Hi"),
            ContentPart.provider_item!("openai", %{"type" => "tool_search_call"})
          ]
        })

      assert %{"content" => [%{"type" => "text", "text" => "Hi"}]} =
               ChatAnthropic.message_for_api(message)
    end

    test "streamed input in content replaces the item's input" do
      part =
        ContentPart.new!(%{
          type: :provider_item,
          content: ~s({"query":"devices"}),
          options: [provider: "anthropic", item: Map.put(@server_tool_use, "input", %{})]
        })

      assert ChatAnthropic.content_part_for_api(part) == @server_tool_use
    end
  end

  describe "streamed response" do
    # Encodes events as the API's server-sent events and splits them into
    # chunks at the given byte offsets, so an event can span two chunks.
    defp sse(events) do
      Enum.map_join(events, fn event ->
        "event: #{event["type"]}\ndata: #{Jason.encode!(event)}\n\n"
      end)
    end

    defp stream(events, split_at) do
      body = sse(events)

      chunks =
        split_at
        |> Enum.reduce({[], 0}, fn at, {acc, from} ->
          {[binary_part(body, from, at - from) | acc], at}
        end)
        |> then(fn {acc, from} ->
          Enum.reverse([binary_part(body, from, byte_size(body) - from) | acc])
        end)

      model = ChatAnthropic.new!(%{stream: true})

      {deltas, _buffer} =
        Enum.reduce(chunks, {[], ""}, fn chunk, {acc, buffer} ->
          {events, buffer} = ChatAnthropic.decode_stream(model, {chunk, buffer})

          deltas =
            events
            |> Enum.map(&ChatAnthropic.do_process_response(model, &1))
            |> List.flatten()

          {acc ++ deltas, buffer}
        end)

      {:ok, message} =
        deltas
        |> MessageDelta.merge_deltas()
        |> MessageDelta.to_message()

      message
    end

    defp search_stream_events do
      [
        %{
          "type" => "message_start",
          "message" => %{
            "type" => "message",
            "role" => "assistant",
            "content" => [],
            "usage" => %{"input_tokens" => 10, "output_tokens" => 1}
          }
        },
        %{
          "type" => "content_block_start",
          "index" => 0,
          "content_block" => %{"type" => "text", "text" => ""}
        },
        %{
          "type" => "content_block_delta",
          "index" => 0,
          "delta" => %{"type" => "text_delta", "text" => "Searching."}
        },
        %{"type" => "content_block_stop", "index" => 0},
        %{
          "type" => "content_block_start",
          "index" => 1,
          "content_block" => Map.put(@server_tool_use, "input", %{})
        },
        %{
          "type" => "content_block_delta",
          "index" => 1,
          "delta" => %{"type" => "input_json_delta", "partial_json" => ~s({"que)}
        },
        %{
          "type" => "content_block_delta",
          "index" => 1,
          "delta" => %{"type" => "input_json_delta", "partial_json" => ~s(ry":"dev)}
        },
        %{
          "type" => "content_block_delta",
          "index" => 1,
          "delta" => %{"type" => "input_json_delta", "partial_json" => ~s(ices"})}
        },
        %{"type" => "content_block_stop", "index" => 1},
        %{"type" => "content_block_start", "index" => 2, "content_block" => @search_result},
        %{"type" => "content_block_stop", "index" => 2},
        %{
          "type" => "content_block_start",
          "index" => 3,
          "content_block" => %{
            "type" => "tool_use",
            "id" => "toolu_1",
            "name" => "devices_list",
            "input" => %{}
          }
        },
        %{
          "type" => "content_block_delta",
          "index" => 3,
          "delta" => %{"type" => "input_json_delta", "partial_json" => ~s({"site":"hq"})}
        },
        %{"type" => "content_block_stop", "index" => 3},
        %{
          "type" => "message_delta",
          "delta" => %{"stop_reason" => "tool_use"},
          "usage" => %{"output_tokens" => 20}
        },
        %{"type" => "message_stop"}
      ]
    end

    test "search blocks stream into provider items, and only tool_use becomes a call" do
      events = search_stream_events()
      body = sse(events)

      # Split inside the second input fragment's event, so the fragments of the
      # server tool block arrive in a later chunk than its start.
      split = :binary.match(body, "ry\\\":\\\"dev") |> elem(0)

      message = stream(events, [split])

      assert %Message{
               status: :complete,
               content: [
                 %ContentPart{type: :text, content: "Searching."},
                 %ContentPart{type: :provider_item} = use_part,
                 %ContentPart{type: :provider_item} = result_part
               ],
               tool_calls: [
                 %ToolCall{
                   call_id: "toolu_1",
                   name: "devices_list",
                   arguments: %{"site" => "hq"}
                 }
               ]
             } = message

      assert ChatAnthropic.content_part_for_api(use_part) == @server_tool_use
      assert ChatAnthropic.content_part_for_api(result_part) == @search_result
    end

    test "the same events decode the same way in one chunk" do
      message = stream(search_stream_events(), [])

      assert %Message{content: [_text, use_part, _result], tool_calls: [_call]} = message
      assert ChatAnthropic.content_part_for_api(use_part) == @server_tool_use
    end

    test "the buffer is a plain string while no server tool block is open" do
      model = ChatAnthropic.new!(%{stream: true})
      [start, text_start | _] = search_stream_events()

      assert {_events, ""} = ChatAnthropic.decode_stream(model, {sse([start, text_start]), ""})
    end

    test "pause_turn on the stream keeps the turn open" do
      events = [
        %{
          "type" => "message_start",
          "message" => %{
            "type" => "message",
            "role" => "assistant",
            "content" => [],
            "usage" => %{"input_tokens" => 10, "output_tokens" => 1}
          }
        },
        %{
          "type" => "content_block_start",
          "index" => 0,
          "content_block" => %{"type" => "text", "text" => "Working on it."}
        },
        %{
          "type" => "message_delta",
          "delta" => %{"stop_reason" => "pause_turn"},
          "usage" => %{"output_tokens" => 5}
        }
      ]

      message = stream(events, [])

      assert %Message{status: :complete, metadata: %{end_turn: false}} = message
      assert Message.continues_turn?(message)
    end
  end

  describe "pause_turn in a chain" do
    test "the chain sends the paused content back and the model continues" do
      paused =
        Message.new_assistant!(%{
          content: [ContentPart.provider_item!("anthropic", @server_tool_use)],
          status: :complete,
          metadata: %{end_turn: false}
        })

      finished = Message.new_assistant!("All done.")

      ChatAnthropic
      |> expect(:call, fn _model, _messages, _tools -> {:ok, paused} end)
      |> expect(:call, fn _model, messages, _tools ->
        assert %Message{role: :assistant, content: [%ContentPart{type: :provider_item}]} =
                 List.last(messages)

        {:ok, finished}
      end)

      {:ok, chain} =
        %{llm: ChatAnthropic.new!(%{})}
        |> LLMChain.new!()
        |> LLMChain.add_message(Message.new_user!("Find the device tools."))
        |> LLMChain.run(mode: :while_needs_response)

      assert chain.last_message.content == [ContentPart.text!("All done.")]
    end
  end

  describe "tool references" do
    test "fill the tool result, and the result's other content follows the tool results" do
      message =
        Message.new_tool_result!(%{
          tool_results: [
            ToolResult.new!(%{
              tool_call_id: "toolu_1",
              name: "load_tools",
              content: [
                ContentPart.text!("Loaded the devices tools."),
                ContentPart.tool_reference!("devices_list"),
                ContentPart.tool_reference!("devices_get")
              ]
            }),
            ToolResult.new!(%{tool_call_id: "toolu_2", name: "get_time", content: "noon"})
          ]
        })

      # A tool_result holding references cannot hold anything else, and every
      # tool_result comes before other content in the user message.
      assert %{
               "role" => "user",
               "content" => [
                 %{
                   "type" => "tool_result",
                   "tool_use_id" => "toolu_1",
                   "content" => [
                     %{"type" => "tool_reference", "tool_name" => "devices_list"},
                     %{"type" => "tool_reference", "tool_name" => "devices_get"}
                   ]
                 },
                 %{
                   "type" => "tool_result",
                   "tool_use_id" => "toolu_2",
                   "content" => [%{"type" => "text", "text" => "noon"}]
                 },
                 %{"type" => "text", "text" => "Loaded the devices tools."}
               ]
             } = ChatAnthropic.message_for_api(message)
    end

    test "a result made only of references sends nothing after the tool results" do
      message =
        Message.new_tool_result!(%{
          tool_results: [
            ToolResult.new!(%{
              tool_call_id: "toolu_1",
              name: "load_tools",
              content: [ContentPart.tool_reference!("devices_list")]
            })
          ]
        })

      assert %{"content" => [%{"type" => "tool_result"}]} = ChatAnthropic.message_for_api(message)
    end

    test "read as text outside a tool result, with a warning" do
      log =
        capture_log(fn ->
          assert %{"type" => "text", "text" => "Tool available: devices_list"} =
                   ChatAnthropic.content_part_for_api(ContentPart.tool_reference!("devices_list"))
        end)

      assert log =~ "outside a tool result"
    end
  end
end
