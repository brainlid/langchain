defmodule LangChain.Chains.TextToTitleChain do
  @moduledoc """
  A convenience chain for turning a user's prompt text into a summarized title
  for the anticipated conversation.

  ## Basic Examples
  A basic example that generates a title

      llm = ChatOpenAI.new!(%{model: "gpt-3.5-turbo", stream: false, seed: 0})
      user_text = "Let's start a new blog post about the magical properties of pineapple cookies."

      %{
        llm: llm,
        input_text: user_text
      }
      |> TextToTitleChain.new!()
      |> TextToTitleChain.evaluate()

      #=> "Magical Properties of Pineapple Cookies Blog Post"

  ## Examples using Title Examples
  Want to get more consistent titles?

  LLMs are pretty bad at following instructions for text length. However, we can
  provide examples titles for the LLM to follow in format style and length. We
  get the added benefit of getting more consistently formatted titles.

  This is the same example, however now we provide other title examples to the
  LLM to follow for consistency.

     llm = ChatOpenAI.new!(%{model: "gpt-3.5-turbo", stream: false, seed: 0})
      user_text = "Let's start a new blog post about the magical properties of
      pineapple cookies."

      %{
        llm: llm,
        input_text: user_text,
        examples: [
          "Blog Post: Making Delicious and Healthy Smoothies",
          "System Email: Notifying Users of Planned Downtime"
        ]
      }
      |> TextToTitleChain.new!()
      |> TextToTitleChain.evaluate()

      #=> "Blog Post: Exploring the Magic of Pineapple Cookies"

  ## Overriding the System Prompt
  For more explicit control of how titles are generated, an `override_system_prompt` can be provided.

      %{
        llm: llm,
        input_text: user_text,
        override_system_prompt: ~s|
          You expertly summarize the User Text into a short 3 or 4 word title to represent a conversation in a positive way.|
      }
      |> TextToTitleChain.new!()
      |> TextToTitleChain.evaluate()

  ## Using a Fallback
  If the primary LLM fails to respond successfully, one or more fallback LLMs can be specified.

      %{
        llm: primary_llm,
        input_text: user_text
      }
      |> TextToTitleChain.new!()
      |> TextToTitleChain.evaluate(with_fallbacks: [fallback_llm])

  """
  use Ecto.Schema
  import Ecto.Changeset
  require Logger
  alias __MODULE__
  alias LangChain.PromptTemplate
  alias LangChain.Chains.LLMChain
  alias LangChain.LangChainError
  alias LangChain.Utils
  alias LangChain.Utils.ChainResult

  @primary_key false
  embedded_schema do
    field :llm, :any, virtual: true
    field :input_text, :string
    field :fallback_title, :string, default: "New topic"
    field :examples, {:array, :string}, default: []
    field :override_system_prompt, :string
    field :verbose, :boolean, default: false
  end

  @type t :: %TextToTitleChain{}

  @create_fields [
    :llm,
    :input_text,
    :fallback_title,
    :examples,
    :override_system_prompt,
    :verbose
  ]
  @required_fields [:llm, :input_text]

  @default_system_prompt ~s|
You expertly summarize the User Text into a short title or phrase to represent a conversation.

<%= if @examples != [] do %>Follow the style, approximate length, and format of the following examples:
<%= for example <- @examples do %>- <%= example %>
<% end %><% end %>|

  @doc """
  Start a new TextToTitleChain configuration.

      {:ok, chain} = TextToTitleChain.new(%{
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
  @spec run(t(), Keyword.t()) :: {:ok, LLMChain.t()} | {:error, LLMChain.t(), LangChainError.t()}
  def run(%TextToTitleChain{} = chain, opts \\ []) do
    messages =
      [
        PromptTemplate.new!(%{
          role: :system,
          text: chain.override_system_prompt || @default_system_prompt
        }),
        PromptTemplate.new!(%{
          role: :user,
          text: "Generate and return the title and nothing else. User Text:\n<%= @input %>"
        })
      ]
      |> PromptTemplate.to_messages!(%{input: chain.input_text, examples: chain.examples})

    %{llm: chain.llm, verbose: chain.verbose}
    |> LLMChain.new!()
    |> LLMChain.add_messages(messages)
    |> LLMChain.run(opts)
  end

  @doc """
  Runs the TextToTitleChain and evaluates the result to return the final answer.

  If unable to generate a title, the `fallback_title` is returned.

  ## Option
  - `:with_fallbacks` - Supports the `with_fallbacks: [fallback_llm]` where one or more additional LLMs can be specified as a backup when the preferred LLM fails.
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

      {:error, _chain, _reason} ->
        chain.fallback_title
    end
  end
end
