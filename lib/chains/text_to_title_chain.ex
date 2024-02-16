defmodule LangChain.Chains.TextToTitleChain do
  @moduledoc """
  A convenience chain for turning a user's prompt text into a summarized title
  for the anticipated conversation.
  """
  use Ecto.Schema
  import Ecto.Changeset
  require Logger
  alias LangChain.PromptTemplate
  alias __MODULE__
  alias LangChain.Chains.LLMChain
  alias LangChain.LangChainError
  alias LangChain.Message
  alias LangChain.Utils
  alias LangChain.Utils.ChainResult

  @primary_key false
  embedded_schema do
    field :llm, :any, virtual: true
    field :input_text, :string
    field :fallback_title, :string, default: "New topic"
    field :verbose, :boolean, default: false
  end

  @type t :: %TextToTitleChain{}

  @create_fields [:llm, :input_text, :fallback_title, :verbose]
  @required_fields [:llm, :input_text]

  @doc """
  Start a new LLMChain configuration.

      {:ok, chain} = LLMChain.new(%{
        llm: %ChatOpenAI{model: "gpt-3.5-turbo", stream: false},
        input_text: "Let's create a marketing blog post about our new product 'Fuzzy Furries'"
      })
  """
  @spec new(attrs :: map()) :: {:ok, t} | {:error, Ecto.Changeset.t()}
  def new(attrs \\ %{}) do
    %TextToTitleChain{}
    |> cast(attrs, @create_fields)
    |> common_validation()
    |> apply_action(:insert)
  end

  @doc """
  Start a new TextToTitleChain and return it or raise an error if invalid.

      chain = TextToTitleChain.new!(%{
        llm: %ChatOpenAI{model: "gpt-3.5-turbo", stream: false},
        input_text: "Let's create a marketing blog post about our new product 'Fuzzy Furries'"
      })
  """
  @spec new!(attrs :: map()) :: t() | no_return()
  def new!(attrs \\ %{}) do
    case new(attrs) do
      {:ok, chain} ->
        chain

      {:error, changeset} ->
        raise LangChainError, changeset
    end
  end

  defp common_validation(changeset) do
    changeset
    |> validate_required(@required_fields)
    |> Utils.validate_llm_is_struct()
  end

  @doc """
  Run a simple LLMChain to summarize the user's prompt into a title for the
  conversation. Uses the provided model. Recommend faster, simpler LLMs without
  streaming.

  If it fails to summarize to a title, it returns the default text.

      new_title = TextToTitleChain.new!(%{
        llm: %ChatOpenAI{model: "gpt-3.5-turbo", stream: false},
        input_text: "Let's create a marketing blog post about our new product 'Fuzzy Furries'"
      })
      |> TextToTitleChain.run()

  """
  @spec run(t(), Keyword.t()) :: String.t() | no_return()
  def run(%TextToTitleChain{} = chain, opts \\ []) do
    messages =
      [
        Message.new_system!(
          "You expertly summarize a user's prompt into a short title or phrase to represent a conversation."
        ),
        PromptTemplate.new!(%{
          role: :user,
          text: "Generate and return the title and nothing else. Prompt:\n<%= @input %>"
        })
      ]
      |> PromptTemplate.to_messages!(%{input: chain.input_text})

    %{llm: chain.llm, verbose: chain.verbose}
    |> LLMChain.new!()
    |> LLMChain.add_messages(messages)
    |> LLMChain.run(opts)
  end

  @doc """
  Runs the TextToTitleChain and evaluates the result to return the final answer.
  If it was unable to generate a title, the `fallback_title` is returned.
  """
  @spec evaluate(t(), Keyword.t()) :: String.t()
  def evaluate(%TextToTitleChain{} = chain, opts \\ []) do
    chain
    |> run(opts)
    |> ChainResult.to_string()
    |> case do
      {:ok, title} ->
        Logger.debug("TextToTitleChain generated #{inspect(title)}")
        if chain.verbose, do: IO.inspect(title, label: "TITLE GENERATED")
        title

      {:error, reason} ->
        Logger.error("TextToTitleChain failed. Reason: #{inspect(reason)}. Returning DEFAULT")
        chain.fallback_title
    end
  end
end
