defmodule LangChain.ChatModels.ToolSearchPartsTest do
  @moduledoc """
  A conversation that used tool search holds `:provider_item` parts in its
  assistant messages and `:tool_reference` parts in its tool results. Every
  chat model encodes such a conversation: provider items from another provider
  are left out, and tool references read as text naming the tool.
  """
  use ExUnit.Case

  alias LangChain.ChatModels.ChatAnthropic
  alias LangChain.ChatModels.ChatAwsMantle
  alias LangChain.ChatModels.ChatDeepSeek
  alias LangChain.ChatModels.ChatGoogleAI
  alias LangChain.ChatModels.ChatGrok
  alias LangChain.ChatModels.ChatMistralAI
  alias LangChain.ChatModels.ChatOllamaAI
  alias LangChain.ChatModels.ChatOpenAI
  alias LangChain.ChatModels.ChatOpenAICompatible
  alias LangChain.ChatModels.ChatOpenAIResponses
  alias LangChain.ChatModels.ChatOrq
  alias LangChain.ChatModels.ChatPerplexity
  alias LangChain.ChatModels.ChatVertexAI
  alias LangChain.Message
  alias LangChain.Message.ContentPart
  alias LangChain.Message.ToolCall
  alias LangChain.Message.ToolResult

  @search_item %{
    "type" => "tool_search_tool_result",
    "tool_use_id" => "srvtoolu_01ABC123",
    "content" => %{
      "type" => "tool_search_tool_search_result",
      "tool_references" => [%{"type" => "tool_reference", "tool_name" => "devices_list"}]
    }
  }

  @reference_text "Tool available: devices_list"

  defp conversation do
    [
      Message.new_system!("You manage devices."),
      Message.new_user!("List my devices."),
      Message.new_assistant!(%{
        content: [
          ContentPart.text!("Loading the device tools."),
          ContentPart.provider_item!("other_provider", @search_item)
        ],
        tool_calls: [
          ToolCall.new!(%{
            call_id: "call_1",
            name: "load_tools",
            arguments: %{"ability" => "devices"},
            status: :complete
          })
        ]
      }),
      Message.new_tool_result!(%{
        tool_results: [
          ToolResult.new!(%{
            tool_call_id: "call_1",
            name: "load_tools",
            content: [ContentPart.tool_reference!("devices_list")]
          })
        ]
      })
    ]
  end

  # Each entry encodes the conversation into the request body the model sends.
  # `references?` is false for a model whose request carries no tool results.
  defp encoders do
    [
      {"ChatAnthropic", fn msgs -> ChatAnthropic.for_api(ChatAnthropic.new!(%{}), msgs, []) end,
       true},
      {"ChatOpenAI", fn msgs -> ChatOpenAI.for_api(ChatOpenAI.new!(%{}), msgs, []) end, true},
      {"ChatOpenAICompatible",
       fn msgs ->
         model = ChatOpenAICompatible.new!(%{endpoint: "http://localhost:1/v1", model: "m"})
         ChatOpenAICompatible.for_api(model, msgs, [])
       end, true},
      {"ChatAwsMantle",
       fn msgs ->
         model =
           ChatAwsMantle.new!(%{model: "moonshotai.kimi-k2", region: "us-west-2", api_key: "k"})

         ChatAwsMantle.for_api(model, msgs, [])
       end, true},
      {"ChatOpenAIResponses",
       fn msgs -> ChatOpenAIResponses.for_api(ChatOpenAIResponses.new!(%{}), msgs, []) end, true},
      {"ChatGoogleAI",
       fn msgs ->
         ChatGoogleAI.for_api(ChatGoogleAI.new!(%{model: "gemini-pro", api_key: "k"}), msgs, [])
       end, true},
      {"ChatVertexAI",
       fn msgs ->
         model = ChatVertexAI.new!(%{model: "gemini-pro", endpoint: "http://localhost:1/"})
         ChatVertexAI.for_api(model, msgs, [])
       end, true},
      {"ChatGrok", fn msgs -> ChatGrok.for_api(ChatGrok.new!(%{model: "grok-4"}), msgs, []) end,
       true},
      {"ChatMistralAI",
       fn msgs ->
         ChatMistralAI.for_api(ChatMistralAI.new!(%{model: "mistral-tiny"}), msgs, [])
       end, true},
      {"ChatOllamaAI",
       fn msgs -> ChatOllamaAI.for_api(ChatOllamaAI.new!(%{model: "llama2"}), msgs, []) end,
       true},
      {"ChatOrq", fn msgs -> ChatOrq.for_api(ChatOrq.new!(%{key: "k"}), msgs, []) end, true},
      {"ChatDeepSeek", fn msgs -> ChatDeepSeek.for_api(ChatDeepSeek.new!(%{}), msgs, []) end,
       true},
      {"ChatPerplexity",
       fn msgs -> ChatPerplexity.for_api(ChatPerplexity.new!(%{}), msgs, []) end, false}
    ]
  end

  @model_names ~w(ChatAnthropic ChatOpenAI ChatOpenAICompatible ChatAwsMantle
                  ChatOpenAIResponses ChatGoogleAI ChatVertexAI ChatGrok ChatMistralAI
                  ChatOllamaAI ChatOrq ChatDeepSeek ChatPerplexity)

  for name <- @model_names do
    test "#{name} leaves out a foreign provider item and reads a tool reference as text" do
      {_name, encode, references?} = List.keyfind(encoders(), unquote(name), 0)

      body = encode.(conversation())
      encoded = Jason.encode!(body)

      refute encoded =~ "tool_search_tool_result"
      refute encoded =~ "srvtoolu_01ABC123"

      if references? do
        assert encoded =~ @reference_text
      end
    end
  end

  test "ChatVertexAI encodes every part as a map, with no nested lists" do
    model = ChatVertexAI.new!(%{model: "gemini-pro", endpoint: "http://localhost:1/"})
    %{"contents" => contents} = ChatVertexAI.for_api(model, conversation(), [])

    for %{"parts" => parts} <- contents, is_list(parts) do
      assert Enum.all?(parts, &is_map/1)
    end
  end

  if Code.ensure_loaded?(ReqLLM) do
    alias LangChain.ChatModels.ChatReqLLM

    test "ChatReqLLM leaves out a foreign provider item and reads a tool reference as text" do
      context = ChatReqLLM.messages_to_req_llm_context(conversation())
      rendered = inspect(context, limit: :infinity, printable_limit: :infinity)

      refute rendered =~ "tool_search_tool_result"
      assert rendered =~ @reference_text
    end
  end
end
