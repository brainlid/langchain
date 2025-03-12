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
  """
  use Ecto.Schema
  import Ecto.Changeset

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

  defp common_validation(changeset) do
    changeset
    |> validate_required(@required_fields)
  end
end
