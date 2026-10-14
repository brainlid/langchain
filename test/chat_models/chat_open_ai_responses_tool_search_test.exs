defmodule LangChain.ChatModels.ChatOpenAIResponsesToolSearchTest do
  use LangChain.BaseCase

  alias LangChain.ChatModels.ChatOpenAIResponses
  alias LangChain.Function
  alias LangChain.LangChainError
  alias LangChain.Message
  alias LangChain.Message.ContentPart
  alias LangChain.Message.ToolCall
  alias LangChain.MessageDelta
  alias LangChain.NativeTool

  # Items as the Responses API returns them for a hosted search that loads
  # `devices_list`, a function outside any namespace.
  @search_call %{
    "arguments" => %{"paths" => ["devices_list"]},
    "call_id" => nil,
    "execution" => "server",
    "id" => "tsc_0578",
    "status" => "completed",
    "type" => "tool_search_call"
  }

  @search_output %{
    "call_id" => nil,
    "execution" => "server",
    "id" => "tso_0578",
    "status" => "completed",
    "tools" => [
      %{
        "defer_loading" => true,
        "description" => "List managed devices at a site, with the count.",
        "name" => "devices_list",
        "output_schema" => nil,
        "parameters" => %{
          "additionalProperties" => false,
          "properties" => %{"site" => %{"type" => "string"}},
          "required" => ["site"],
          "type" => "object"
        },
        "strict" => true,
        "type" => "function"
      }
    ],
    "type" => "tool_search_output"
  }

  @function_call %{
    "arguments" => ~s({"site":"hq"}),
    "call_id" => "call_Hx2A",
    "id" => "fc_0578",
    "name" => "devices_list",
    "namespace" => "devices_list",
    "status" => "completed",
    "type" => "function_call"
  }

  setup do
    %{model: ChatOpenAIResponses.new!(%{model: "gpt-5.4"})}
  end

  defp function(name, opts \\ []) do
    Function.new!(%{
      name: name,
      description: "Tool #{name}",
      parameters_schema: %{"type" => "object", "properties" => %{}},
      function: fn _args, _context -> {:ok, "done"} end,
      options: opts
    })
  end

  defp tools_body(model, tools) do
    ChatOpenAIResponses.for_api(model, [Message.new_user!("hi")], tools).tools
  end

  defp stream_to_message(model, events) do
    %MessageDelta{} =
      merged =
      events
      |> Enum.map(&ChatOpenAIResponses.do_process_response(model, &1))
      |> List.flatten()
      |> Enum.reject(&(&1 == :skip))
      |> MessageDelta.merge_deltas()

    {:ok, message} = MessageDelta.to_message(%MessageDelta{merged | status: :complete})
    message
  end

  describe "request tools" do
    test "a deferred function carries defer_loading", %{model: model} do
      assert [
               %{"type" => "tool_search"},
               %{"name" => "get_time"} = plain,
               %{"name" => "devices_list", "defer_loading" => true}
             ] =
               tools_body(model, [
                 NativeTool.tool_search(),
                 function("get_time"),
                 function("devices_list", defer_loading: true)
               ])

      refute Map.has_key?(plain, "defer_loading")
    end

    test "the search tool is sent without its variant", %{model: model} do
      for variant <- [:bm25, :regex] do
        assert [search, _deferred] =
                 tools_body(model, [
                   NativeTool.tool_search(variant: variant),
                   function("devices_list", defer_loading: true)
                 ])

        assert search == %{"type" => "tool_search"}
      end
    end

    test "functions sharing a namespace become one namespace tool in place of the first",
         %{model: model} do
      assert [
               %{"name" => "get_time"},
               %{
                 "type" => "namespace",
                 "name" => "devices",
                 "description" => "Inspect and manage devices.",
                 "tools" => [
                   %{"name" => "list", "defer_loading" => true},
                   %{"name" => "restart", "defer_loading" => true}
                 ]
               },
               %{"type" => "tool_search"},
               %{"type" => "namespace", "name" => "billing", "tools" => [%{"name" => "invoices"}]}
             ] =
               tools_body(model, [
                 function("get_time"),
                 function("list", defer_loading: true, namespace: "devices"),
                 NativeTool.tool_search(),
                 function("invoices",
                   defer_loading: true,
                   namespace: "billing",
                   namespace_description: "Billing."
                 ),
                 function("restart",
                   defer_loading: true,
                   namespace: "devices",
                   namespace_description: "Inspect and manage devices."
                 )
               ])
    end

    test "a namespace without a description gets one", %{model: model} do
      assert [%{"type" => "namespace", "description" => "Tools for devices."}] =
               tools_body(model, [function("list", namespace: "devices")])
    end

    test "a deferred function without the search tool is an error", %{model: model} do
      assert_raise LangChainError, ~r/OpenAI requires the search tool/, fn ->
        tools_body(model, [function("get_time"), function("devices_list", defer_loading: true)])
      end
    end

    test "a namespace needs no search tool when nothing is deferred", %{model: model} do
      assert [%{"type" => "namespace"}] =
               tools_body(model, [function("list", namespace: "devices")])
    end
  end

  describe "non-streamed response" do
    test "search items become provider items and the call keeps its namespace",
         %{model: model} do
      response = %{
        "status" => "completed",
        "output" => [@search_call, @search_output, @function_call]
      }

      assert %Message{content: content, tool_calls: [call]} =
               ChatOpenAIResponses.do_process_response(model, response)

      assert [
               %ContentPart{
                 type: :provider_item,
                 options: [provider: "openai", item: @search_call]
               },
               %ContentPart{
                 type: :provider_item,
                 options: [provider: "openai", item: @search_output]
               }
             ] = content

      assert %ToolCall{
               name: "devices_list",
               call_id: "call_Hx2A",
               metadata: %{"namespace" => "devices_list"}
             } = call
    end

    test "the message replays the search items in place and the call with its namespace",
         %{model: model} do
      response = %{
        "status" => "completed",
        "output" => [@search_call, @search_output, @function_call]
      }

      message = ChatOpenAIResponses.do_process_response(model, response)

      assert [
               @search_call,
               @search_output,
               %{
                 "type" => "function_call",
                 "call_id" => "call_Hx2A",
                 "name" => "devices_list",
                 "namespace" => "devices_list",
                 "arguments" => ~s({"site":"hq"})
               }
             ] = ChatOpenAIResponses.for_api(model, message)
    end

    test "a call in a namespace keeps the namespace's name", %{model: model} do
      response = %{
        "status" => "completed",
        "output" => [%{@function_call | "name" => "list", "namespace" => "devices"}]
      }

      assert %Message{tool_calls: [%ToolCall{name: "list"} = call]} =
               ChatOpenAIResponses.do_process_response(model, response)

      assert %{"namespace" => "devices"} =
               ChatOpenAIResponses.for_api(model, call)
    end

    test "a call without a namespace replays without one", %{model: model} do
      response = %{"status" => "completed", "output" => [Map.delete(@function_call, "namespace")]}

      assert %Message{tool_calls: [%ToolCall{metadata: nil} = call]} =
               ChatOpenAIResponses.do_process_response(model, response)

      refute Map.has_key?(ChatOpenAIResponses.for_api(model, call), "namespace")
    end

    test "a namespace restored under an atom key replays", %{model: model} do
      call = ToolCall.new!(%{call_id: "c1", name: "list", metadata: %{namespace: "devices"}})
      assert %{"namespace" => "devices"} = ChatOpenAIResponses.for_api(model, call)
    end

    test "another provider's items are not replayed", %{model: model} do
      message =
        Message.new_assistant!(%{
          content: [
            ContentPart.provider_item!("anthropic", %{"type" => "server_tool_use"}),
            ContentPart.text!("Checking.")
          ]
        })

      assert [%{"type" => "message", "role" => "assistant"}] =
               ChatOpenAIResponses.for_api(model, message)
    end
  end

  describe "streamed response" do
    # The events the API streams for a search followed by a function call,
    # with the argument deltas trimmed to two.
    @events [
      %{"type" => "response.created", "response" => %{}},
      %{
        "type" => "response.output_item.added",
        "output_index" => 0,
        "item" => %{@search_call | "arguments" => %{}, "status" => "in_progress"}
      },
      %{"type" => "response.output_item.done", "output_index" => 0, "item" => @search_call},
      %{
        "type" => "response.output_item.added",
        "output_index" => 1,
        "item" => %{@search_output | "status" => "in_progress"}
      },
      %{"type" => "response.output_item.done", "output_index" => 1, "item" => @search_output},
      %{
        "type" => "response.output_item.added",
        "output_index" => 2,
        "item" => %{@function_call | "arguments" => "", "status" => "in_progress"}
      },
      %{
        "type" => "response.function_call_arguments.delta",
        "output_index" => 2,
        "item_id" => "fc_0578",
        "delta" => ~s({"site":)
      },
      %{
        "type" => "response.function_call_arguments.delta",
        "output_index" => 2,
        "item_id" => "fc_0578",
        "delta" => ~s("hq"})
      },
      %{
        "type" => "response.function_call_arguments.done",
        "output_index" => 2,
        "arguments" => ~s({"site":"hq"})
      },
      %{"type" => "response.output_item.done", "output_index" => 2, "item" => @function_call}
    ]

    test "search items arrive whole and the call keeps its namespace", %{model: model} do
      message = stream_to_message(model, @events)

      assert [
               %ContentPart{
                 type: :provider_item,
                 options: [provider: "openai", item: @search_call]
               },
               %ContentPart{
                 type: :provider_item,
                 options: [provider: "openai", item: @search_output]
               }
             ] = message.content

      assert [
               %ToolCall{
                 name: "devices_list",
                 arguments: %{"site" => "hq"},
                 metadata: %{"namespace" => "devices_list"}
               }
             ] = message.tool_calls
    end

    test "the streamed message replays as the items the API sent", %{model: model} do
      message = stream_to_message(model, @events)

      assert [
               @search_call,
               @search_output,
               %{"type" => "function_call", "namespace" => "devices_list"}
             ] = ChatOpenAIResponses.for_api(model, message)
    end

    test "text before the search keeps its place", %{model: model} do
      text_events = [
        %{
          "type" => "response.output_item.added",
          "output_index" => 0,
          "item" => %{"type" => "message", "role" => "assistant", "content" => []}
        },
        %{
          "type" => "response.output_text.delta",
          "output_index" => 0,
          "content_index" => 0,
          "delta" => "Let me look that up."
        }
      ]

      shifted =
        Enum.map(@events, fn
          %{"output_index" => index} = event -> %{event | "output_index" => index + 1}
          event -> event
        end)

      message = stream_to_message(model, text_events ++ shifted)

      assert [
               %ContentPart{type: :text, content: "Let me look that up."},
               %ContentPart{
                 type: :provider_item,
                 options: [provider: "openai", item: @search_call]
               },
               %ContentPart{
                 type: :provider_item,
                 options: [provider: "openai", item: @search_output]
               }
             ] = message.content

      assert [
               %{"type" => "message", "role" => "assistant"},
               @search_call,
               @search_output,
               %{"type" => "function_call"}
             ] = ChatOpenAIResponses.for_api(model, message)
    end
  end
end
