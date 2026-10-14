defmodule LangChain.ChatModels.ToolSearchFallbackTest do
  @moduledoc """
  A chain configured for tool search runs on every chat model without tool
  search: the search tool is left out of the request and deferred functions
  are sent as ordinary tools.
  """
  use ExUnit.Case, async: true

  alias LangChain.ChatModels.ChatAwsMantle
  alias LangChain.ChatModels.ChatDeepSeek
  alias LangChain.ChatModels.ChatGoogleAI
  alias LangChain.ChatModels.ChatGrok
  alias LangChain.ChatModels.ChatMistralAI
  alias LangChain.ChatModels.ChatOllamaAI
  alias LangChain.ChatModels.ChatOrq
  alias LangChain.ChatModels.ChatPerplexity
  alias LangChain.ChatModels.ChatVertexAI
  alias LangChain.Function
  alias LangChain.Message
  alias LangChain.NativeTool

  defp deferred do
    Function.new!(%{
      name: "devices_list",
      description: "List the managed devices at a site.",
      parameters_schema: %{"type" => "object", "properties" => %{}},
      function: fn _args, _context -> {:ok, "17"} end,
      options: [defer_loading: true, namespace: "devices"]
    })
  end

  defp tools, do: [NativeTool.tool_search(), deferred()]

  for {label, model} <- [
        {"ChatDeepSeek", quote(do: ChatDeepSeek.new!(%{}))},
        {"ChatGoogleAI", quote(do: ChatGoogleAI.new!(%{}))},
        {"ChatGrok", quote(do: ChatGrok.new!(%{}))},
        {"ChatMistralAI", quote(do: ChatMistralAI.new!(%{model: "mistral-large-latest"}))},
        {"ChatOllamaAI", quote(do: ChatOllamaAI.new!(%{}))},
        {"ChatOrq", quote(do: ChatOrq.new!(%{key: "key"}))},
        {"ChatPerplexity", quote(do: ChatPerplexity.new!(%{}))},
        {"ChatVertexAI",
         quote(do: ChatVertexAI.new!(%{endpoint: "https://example.com", model: "gemini-2.5-pro"}))},
        {"ChatAwsMantle",
         quote(
           do: ChatAwsMantle.new!(%{endpoint: "https://example.com", model: "m", api_key: "k"})
         )}
      ] do
    test "#{label} leaves out the search tool and sends the deferred function" do
      model = unquote(model)

      body =
        model.__struct__.for_api(model, [Message.new_user!("hi")], tools())
        |> Jason.encode!()

      refute body =~ "tool_search"
      assert body =~ "devices_list"
    end
  end

  describe "NativeTool.drop_tool_search/1" do
    test "removes only the search tool and keeps the order of the rest" do
      google = NativeTool.new!(%{name: "google_search", configuration: %{}})
      fun = deferred()

      assert [^google, ^fun] =
               NativeTool.drop_tool_search([google, NativeTool.tool_search(), fun])
    end

    test "returns the list unchanged when there is no search tool" do
      fun = deferred()
      assert [^fun] = NativeTool.drop_tool_search([fun])
    end

    test "passes nil through" do
      assert NativeTool.drop_tool_search(nil) == nil
    end
  end
end
