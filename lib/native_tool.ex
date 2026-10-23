defmodule LangChain.NativeTool do
  @moduledoc """
  Represents built-in tools available from AI/LLM services that can be used within the LangChain framework.

  Native tools are functionalities provided directly by language model services (like Google AI, OpenAI)
  that can be invoked by LLMs to perform specific actions or retrieve information. They are
  "native" because they're built into the AI service itself, rather than implemented externally.

  Each native tool has:
  - A unique name that identifies it (e.g., "google_search", "code_interpreter")
  - A configuration map that contains tool-specific settings

  ## Google Search Grounding

  Starting with Gemini 2.0, Google Search is available as a native tool. This enables the model
  to decide when to use Google Search to enhance the factuality and recency of responses.

  Google Search as a tool enables:
  - More accurate and up-to-date answers with grounding sources
  - Retrieving information from the web for further analysis
  - Finding relevant images, videos, or other media for multimodal reasoning
  - Supporting specialized tasks like coding, technical troubleshooting, etc.

  ### Example with Google Search

  ```elixir
  alias LangChain.Chains.LLMChain
  alias LangChain.Message
  alias LangChain.NativeTool

  model = ChatGoogleAI.new!(%{temperature: 0, stream: false, model: "gemini-2.0-flash"})

  {:ok, updated_chain} =
     %{llm: model, verbose: false, stream: false}
     |> LLMChain.new!()
     |> LLMChain.add_message(
       Message.new_user!("What is the current Google stock price?")
     )
     |> LLMChain.add_tools(NativeTool.new!(%{name: "google_search", configuration: %{}}))
     |> LLMChain.run()
  ```

  ## Tool Search

  `tool_search/1` builds a provider-neutral search tool. Add it to a chain
  alongside functions marked `defer_loading: true` (see "Tool search options"
  in `LangChain.Function`). Deferred functions are declared on every request
  but kept out of the model's context until the model searches for them, so
  the request prefix never changes and prompt caching keeps working.

  ```elixir
  chain
  |> LLMChain.add_tools(NativeTool.tool_search())
  |> LLMChain.add_tools(deferred_functions)
  ```

  Keep at least one function, normally the most-used ones, not deferred.

  A chat model that offers tool search translates the tool for its provider.
  A chat model without tool search falls back: it leaves the search tool out
  of the request and sends deferred functions as ordinary tools, so every tool
  is available from the start. Search items another model added to the
  conversation are left out, and `:tool_reference` parts are sent as text. A
  chain configured for tool search therefore runs on any chat model, which
  keeps fallbacks and model swaps working.

  | Chat model | Tool search | Namespaces | Verified |
  |---|---|---|---|
  | `LangChain.ChatModels.ChatAnthropic` | Server-side search (`:bm25` or `:regex`), and client-side loading with `:tool_reference` parts | Not used | Live |
  | `LangChain.ChatModels.ChatOpenAIResponses` | OpenAI's hosted search; `:variant` does not apply | Yes | Live |
  | `LangChain.ChatModels.ChatReqLLM`, `anthropic:` models | Server-side and client-side, as `ChatAnthropic` | Not used | Live |
  | `LangChain.ChatModels.ChatReqLLM`, Anthropic models on `amazon_bedrock:` | Server-side and client-side | Not used | Request encoding only |
  | `LangChain.ChatModels.ChatReqLLM`, `openai:` models | OpenAI's hosted search; `:variant` does not apply | Yes | Live |
  | `LangChain.ChatModels.ChatReqLLM`, any other provider | Falls back | | Request encoding |
  | `LangChain.ChatModels.ChatOpenAI`, `LangChain.ChatModels.ChatOpenAICompatible`, `LangChain.ChatModels.ChatAwsMantle` | Falls back. The Chat Completions API rejects the search tool. | | Request encoding and replay |
  | `LangChain.ChatModels.ChatDeepSeek`, `LangChain.ChatModels.ChatGoogleAI`, `LangChain.ChatModels.ChatVertexAI`, `LangChain.ChatModels.ChatGrok`, `LangChain.ChatModels.ChatMistralAI`, `LangChain.ChatModels.ChatOllamaAI`, `LangChain.ChatModels.ChatOrq`, `LangChain.ChatModels.ChatPerplexity` | Falls back | | Request encoding |
  | `LangChain.ChatModels.ChatBumblebee` | Falls back | | Untested |

  "Live" means a model searched for a deferred tool, called it, and accepted
  the replayed conversation on a later turn. "Request encoding" means unit
  tests of the request body only.

  A chat model without tool search uses `drop_tool_search/1` to leave the
  search tool out.
  """
  use Ecto.Schema
  import Ecto.Changeset
  require Logger

  alias __MODULE__
  alias LangChain.LangChainError

  embedded_schema do
    field :name, :string
    field :configuration, :map
  end

  @type t :: %NativeTool{}
  @type configuration :: %{String.t() => any()}

  @create_fields [
    :name,
    :configuration
  ]
  @required_fields [:name]

  @doc """
  Build a new native tool.
  """
  @spec new(attrs :: map()) :: {:ok, t} | {:error, Ecto.Changeset.t()}
  def new(attrs \\ %{}) do
    %NativeTool{}
    |> cast(attrs, @create_fields)
    |> common_validation()
    |> apply_action(:insert)
  end

  @doc """
  Build a new native tool and return it or raise an error if invalid.
  """
  @spec new!(attrs :: map()) :: t() | no_return()
  def new!(attrs \\ %{}) do
    case new(attrs) do
      {:ok, native_tool} ->
        native_tool

      {:error, changeset} ->
        raise LangChainError, changeset
    end
  end

  @tool_search_name "tool_search"
  @tool_search_variants [:bm25, :regex]

  @doc """
  Build the provider-neutral tool search tool.

  ## Options

  * `:variant` - How the model writes its search, on providers that offer a
    choice. `:bm25` (default) searches with natural language queries; `:regex`
    searches with regular expressions. Providers with a single kind of search
    ignore it.
  """
  @spec tool_search(keyword()) :: t()
  def tool_search(opts \\ []) do
    variant = Keyword.get(opts, :variant, :bm25)

    unless variant in @tool_search_variants do
      raise LangChainError,
            "Unsupported tool search variant #{inspect(variant)}. " <>
              "Expected one of #{inspect(@tool_search_variants)}."
    end

    new!(%{name: @tool_search_name, configuration: %{"variant" => Atom.to_string(variant)}})
  end

  @doc """
  Return `true` when the tool is the tool search tool built by `tool_search/1`.
  """
  @spec tool_search?(any()) :: boolean()
  def tool_search?(%NativeTool{name: @tool_search_name}), do: true
  def tool_search?(_tool), do: false

  @doc """
  Remove the tool search tool from a list of tools.

  A chat model without tool search calls this before encoding its tools, so a
  chain configured for tool search runs on it with every function mounted.
  Deferred functions are left in the list and sent as ordinary tools. A debug
  message is logged when the search tool is removed.
  """
  @spec drop_tool_search(nil | list()) :: nil | list()
  def drop_tool_search(nil), do: nil

  def drop_tool_search(tools) when is_list(tools) do
    case Enum.split_with(tools, &tool_search?/1) do
      {[], _tools} ->
        tools

      {_search_tools, rest} ->
        Logger.debug(fn ->
          "This chat model has no tool search; the search tool is left out and " <>
            "deferred functions are sent as ordinary tools."
        end)

        rest
    end
  end

  defp common_validation(changeset) do
    changeset
    |> validate_required(@required_fields)
  end
end
