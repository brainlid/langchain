defmodule LangChain.ChatModels.ChatCompletionsToolSearchFallbackTest do
  @moduledoc """
  A conversation that used tool search on Anthropic and on the OpenAI
  Responses API, replayed through the Chat Completions models. The search
  items are left out, every deferred function is an ordinary tool, and the
  real tool calls and results go through unchanged.
  """
  use ExUnit.Case, async: true

  alias LangChain.ChatModels.ChatAnthropic
  alias LangChain.ChatModels.ChatOpenAI
  alias LangChain.ChatModels.ChatOpenAICompatible
  alias LangChain.ChatModels.ChatOpenAIResponses
  alias LangChain.Function
  alias LangChain.LangChainError
  alias LangChain.Message
  alias LangChain.Message.ContentPart
  alias LangChain.Message.ToolResult
  alias LangChain.NativeTool

  defp function(name, opts) do
    Function.new!(%{
      name: name,
      description: "Tool #{name}",
      parameters_schema: %{"type" => "object", "properties" => %{}},
      function: fn _args, _context -> {:ok, "done"} end,
      options: opts
    })
  end

  defp tools do
    [
      NativeTool.tool_search(),
      function("get_time", []),
      function("devices_list", defer_loading: true, namespace: "devices"),
      function("billing_balance", defer_loading: true)
    ]
  end

  # An assistant turn as ChatAnthropic decodes a server-side search followed
  # by a call to the tool it loaded.
  defp anthropic_turn do
    ChatAnthropic.do_process_response(ChatAnthropic.new!(%{}), %{
      "role" => "assistant",
      "type" => "message",
      "stop_reason" => "tool_use",
      "usage" => %{"input_tokens" => 10, "output_tokens" => 5},
      "content" => [
        %{"type" => "text", "text" => "Let me find the right tool."},
        %{
          "type" => "server_tool_use",
          "id" => "srvtoolu_1",
          "name" => "tool_search_tool_bm25",
          "input" => %{"query" => "devices"}
        },
        %{
          "type" => "tool_search_tool_result",
          "tool_use_id" => "srvtoolu_1",
          "content" => %{
            "type" => "tool_search_tool_search_result",
            "tool_references" => [%{"type" => "tool_reference", "tool_name" => "devices_list"}]
          }
        },
        %{"type" => "text", "text" => "Checking hq."},
        %{
          "type" => "tool_use",
          "id" => "toolu_1",
          "name" => "devices_list",
          "input" => %{"site" => "hq"}
        }
      ]
    })
  end

  # An assistant turn as ChatOpenAIResponses decodes a hosted search that
  # loaded a namespaced function, followed by the call.
  defp responses_turn do
    ChatOpenAIResponses.do_process_response(ChatOpenAIResponses.new!(%{model: "gpt-5.4"}), %{
      "status" => "completed",
      "output" => [
        %{
          "type" => "tool_search_call",
          "id" => "tsc_1",
          "execution" => "server",
          "call_id" => nil,
          "status" => "completed",
          "arguments" => %{"paths" => ["devices"]}
        },
        %{
          "type" => "tool_search_output",
          "id" => "tso_1",
          "execution" => "server",
          "call_id" => nil,
          "status" => "completed",
          "tools" => [%{"type" => "namespace", "name" => "devices", "tools" => []}]
        },
        %{
          "type" => "function_call",
          "id" => "fc_1",
          "call_id" => "call_2",
          "name" => "devices_list",
          "namespace" => "devices",
          "arguments" => ~s({"site":"warehouse"}),
          "status" => "completed"
        }
      ]
    })
  end

  defp conversation do
    [
      Message.new_system!("You are an IT assistant."),
      Message.new_user!("How many devices are at hq?"),
      anthropic_turn(),
      Message.new_tool_result!(%{
        tool_results: [
          ToolResult.new!(%{
            tool_call_id: "toolu_1",
            name: "devices_list",
            content: [
              ContentPart.text!("17 devices."),
              ContentPart.tool_reference!("devices_get")
            ]
          })
        ]
      }),
      Message.new_assistant!("There are 17 devices at hq."),
      Message.new_user!("And at the warehouse?"),
      responses_turn(),
      Message.new_tool_result!(%{
        tool_results: [
          ToolResult.new!(%{tool_call_id: "call_2", name: "devices_list", content: "42 devices."})
        ]
      })
    ]
  end

  # Collects every string an encoded message carries, wherever it sits.
  defp strings(value) when is_binary(value), do: [value]
  defp strings(value) when is_map(value), do: value |> Map.values() |> Enum.flat_map(&strings/1)
  defp strings(value) when is_list(value), do: Enum.flat_map(value, &strings/1)
  defp strings(_value), do: []

  for {label, model} <- [
        {"ChatOpenAI", quote(do: ChatOpenAI.new!(%{model: "gpt-5.4"}))},
        {"ChatOpenAICompatible",
         quote(
           do:
             ChatOpenAICompatible.new!(%{
               endpoint: "http://localhost:8000/v1/chat/completions",
               model: "Qwen/Qwen3-8B"
             })
         )}
      ] do
    describe label do
      setup do
        %{model: unquote(model)}
      end

      test "sends every function as an ordinary tool and leaves out the search tool",
           %{model: model} do
        body = model.__struct__.for_api(model, [Message.new_user!("hi")], tools())

        assert ["get_time", "devices_list", "billing_balance"] =
                 Enum.map(body.tools, & &1["function"]["name"])

        refute Enum.any?(strings(body.tools), &(&1 in ["tool_search", "devices"]))
      end

      test "replays the conversation without search items, keeping the real calls and results",
           %{model: model} do
        %{messages: messages} = model.__struct__.for_api(model, conversation(), tools())

        assert [
                 %{"role" => :system},
                 %{"role" => :user},
                 %{"role" => :assistant, "tool_calls" => [anthropic_call]} = anthropic_turn,
                 %{"role" => :tool, "tool_call_id" => "toolu_1"} = anthropic_result,
                 %{"role" => :assistant},
                 %{"role" => :user},
                 %{"role" => :assistant, "tool_calls" => [responses_call]} = responses_turn,
                 %{"role" => :tool, "tool_call_id" => "call_2"}
               ] = messages

        assert %{"id" => "toolu_1", "function" => %{"name" => "devices_list"}} = anthropic_call
        assert %{"id" => "call_2", "function" => %{"name" => "devices_list"}} = responses_call

        assert "Let me find the right tool." in strings(anthropic_turn)
        assert "Checking hq." in strings(anthropic_turn)
        assert "Tool available: devices_get" in strings(anthropic_result)

        for item <- [anthropic_turn, responses_turn],
            text <- strings(item) do
          refute text =~ ~r/srvtoolu_|tool_search|tsc_|tso_/
        end

        refute Map.has_key?(responses_call, "namespace")
      end

      test "raises a clear error for any other native tool", %{model: model} do
        assert_raise LangChainError, ~r/not supported by Chat Completions/, fn ->
          model.__struct__.for_api(model, [Message.new_user!("hi")], [
            NativeTool.new!(%{name: "web_search"})
          ])
        end
      end
    end
  end
end
