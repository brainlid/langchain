if Code.ensure_loaded?(ReqLLM) do
  defmodule LangChain.ChatModels.ChatReqLLMOpenAIToolSearchLiveTest do
    @moduledoc """
    OpenAI hosted tool search through `ChatReqLLM` and req_llm, against the
    live Responses API with `store: false`.

    req_llm sends deferred functions and the hosted `tool_search` tool, but it
    does not decode the `tool_search_call` and `tool_search_output` items or a
    function call's `namespace`. A replayed history therefore holds the call to
    a deferred function without the search that loaded it. The API accepts
    that, and the model can call the same tool again on a later turn.
    """
    use ExUnit.Case, async: true

    import LangChain.ToolSearchFixtures

    alias LangChain.Chains.LLMChain
    alias LangChain.ChatModels.ChatReqLLM
    alias LangChain.Message
    alias LangChain.Message.ContentPart
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

    for stream <- [false, true] do
      test "openai:gpt-5.4 (stream: #{stream}): finds a deferred tool and calls it on later turns" do
        stream = unquote(stream)
        test_pid = self()

        llm =
          ChatReqLLM.new!(%{
            model: "openai:gpt-5.4",
            stream: stream,
            receive_timeout: 120_000,
            provider_opts: %{provider_options: [store: false]}
          })

        tools = [NativeTool.tool_search() | device_tools(test_pid) ++ other_tools()]

        {:ok, chain} =
          %{llm: llm}
          |> LLMChain.new!()
          |> LLMChain.add_message(Message.new_system!(system_prompt()))
          |> LLMChain.add_tools(tools)
          |> LLMChain.add_message(
            Message.new_user!("How many managed devices are at the hq site?")
          )
          |> run()

        assert_received {:tool_ran, "devices_list", _site}
        assert reply_text(chain) =~ hq_count()

        # req_llm drops the search items, so nothing reaches the message.
        assert [] = provider_item_types(chain.messages)

        turn_two_start = length(chain.messages)

        {:ok, chain} =
          chain
          |> LLMChain.add_message(Message.new_user!("And at the warehouse site?"))
          |> run()

        assert reply_text(chain) =~ warehouse_count()
        assert [_ | _] = tool_calls(Enum.drop(chain.messages, turn_two_start), "devices_list")
      end
    end
  end
end
