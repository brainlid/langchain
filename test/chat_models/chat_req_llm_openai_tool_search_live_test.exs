if Code.ensure_loaded?(ReqLLM) do
  defmodule LangChain.ChatModels.ChatReqLLMOpenAIToolSearchLiveTest do
    @moduledoc """
    OpenAI hosted tool search through `ChatReqLLM` and req_llm, against the
    live Responses API with `store: false`.

    - A model finds a deferred function with the hosted search tool and calls
      it. The `tool_search_call` and `tool_search_output` items, and the call's
      `namespace`, replay on the next turn, where the API accepts them only
      together.
    - A model finds a function inside a namespace. The call comes back under
      the function's own name, with the namespace in the call's metadata.
    """
    use ExUnit.Case, async: true

    import LangChain.ToolSearchFixtures

    alias LangChain.Chains.LLMChain
    alias LangChain.ChatModels.ChatReqLLM
    alias LangChain.Message
    alias LangChain.Message.ContentPart
    alias LangChain.Message.ToolCall
    alias LangChain.NativeTool

    @moduletag live_call: true, live_open_ai: true, timeout: 600_000

    defp run(chain) do
      case LLMChain.run(chain, mode: :while_needs_response) do
        {:ok, chain} ->
          {:ok, chain}

        {:error, chain, error} ->
          flunk("""
          Run failed: #{inspect(error, limit: :infinity)}
          Last message: #{inspect(chain.last_message, limit: :infinity)}
          """)
      end
    end

    defp reply_text(%LLMChain{last_message: message}),
      do: ContentPart.content_to_string(message.content) || ""

    defp llm(stream) do
      ChatReqLLM.new!(%{
        model: "openai:gpt-5.4",
        stream: stream,
        receive_timeout: 120_000,
        provider_opts: %{provider_options: [store: false]}
      })
    end

    defp ask(llm, tools, question) do
      %{llm: llm}
      |> LLMChain.new!()
      |> LLMChain.add_message(Message.new_system!(system_prompt()))
      |> LLMChain.add_tools(tools)
      |> LLMChain.add_message(Message.new_user!(question))
      |> run()
    end

    for stream <- [false, true] do
      test "openai:gpt-5.4 (stream: #{stream}): searches for a deferred tool and replays the search" do
        test_pid = self()
        tools = [NativeTool.tool_search() | device_tools(test_pid) ++ other_tools()]

        {:ok, chain} =
          ask(llm(unquote(stream)), tools, "How many managed devices are at the hq site?")

        assert_received {:tool_ran, "devices_list", _site}
        assert reply_text(chain) =~ hq_count()

        assert ["tool_search_call", "tool_search_output" | _] =
                 provider_item_types(chain.messages)

        assert [%ToolCall{metadata: %{"namespace" => "devices_list"}} | _] =
                 tool_calls(chain.messages, "devices_list")

        turn_two_start = length(chain.messages)

        {:ok, chain} =
          chain
          |> LLMChain.add_message(Message.new_user!("And at the warehouse site?"))
          |> run()

        assert reply_text(chain) =~ warehouse_count()
        assert [_ | _] = tool_calls(Enum.drop(chain.messages, turn_two_start), "devices_list")
      end
    end

    for stream <- [false, true] do
      test "openai:gpt-5.4 (stream: #{stream}): finds a function inside a namespace" do
        test_pid = self()

        devices =
          device_tools(test_pid,
            namespace: "devices",
            namespace_description: "List, inspect, and restart the managed devices at a site."
          )

        tools = [NativeTool.tool_search() | devices ++ other_tools()]

        {:ok, chain} =
          ask(llm(unquote(stream)), tools, "How many managed devices are at the hq site?")

        assert_received {:tool_ran, "devices_list", _site}
        assert reply_text(chain) =~ hq_count()

        assert [%ToolCall{metadata: %{"namespace" => "devices"}} | _] =
                 tool_calls(chain.messages, "devices_list")

        {:ok, chain} =
          chain
          |> LLMChain.add_message(Message.new_user!("And at the warehouse site?"))
          |> run()

        assert reply_text(chain) =~ warehouse_count()
      end
    end
  end
end
