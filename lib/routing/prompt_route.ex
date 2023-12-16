defmodule Routing.PromptRoute do
  @moduledoc """
  Defines a route or direction a prompting interaction with an LLM can take.

  This helps add complexity and specificity to a the LLM's instructions without
  using many tokens and trying to combine many things into a single prompt
  setup. This works by letting the user's initial prompt define how the next
  prompt is setup.

  To better understand, let's see example of taking the user's input and
  classifying what supported activity or specialty it is best aligned with.

  User prompt:
  > Let's create a marketing focused blog post comparing three types of our
  > trailer hitch products.

  We want our assistant to use different prompts for the different types of
  activities it's setup to do. Let's define the activities like this:

      [
        PromptRoute.new!(%{
          name: "marketing_email",
          description: "Create marketing focused emails",
          chain: marketing_email_chain,
          final: true
        }),
        PromptRoute.new!(%{
          name: "blog_post",
          description: "Create a blog post that will be linked from the company's landing page",
          chain: blog_post_chain
        }),
        PromptRoute.new!(%{
          name: "support_triage",
          description: "Triage a customer support request for severity and requested resolution",
          chain: support_triage_chain, # not the final chain. Execution is not displayed?
          final: false
        }),
      ]

  Given the user's prompt and the routes we provided, it will choose
  "marketing_email". This leads to an LLMChain being setup for that purpose
  where the user's initial prompt can be re-run against the prompts we define
  for that route.

  """
  use Ecto.Schema
  import Ecto.Changeset
  require Logger
  alias __MODULE__
  alias LangChain.LangChainError

  @primary_key false
  embedded_schema do
    field :name, :string
    field :description, :string

    # The LLMChain to use if the route matches.
    # Contains prompt templates, model config, and functions
    field :chain, :any, virtual: true

    # if the chain is the final one or not. If true, the callback_fn should be
    # executed while running for realtime display. If false, the execution of
    # the chain may lead to future chains which should instead be displayed as
    # realtime execution.
    field :final, :boolean, default: true

    # field :prompts, {:array, :any}, default: []
    # field :functions, {:array, :any}, default: []
  end

  @type t :: %PromptRoute{}

  @create_fields [
    :name,
    :description,
    :chain,
    :final
    # :prompts,
    # :functions
  ]
  @required_fields [:name, :description, :chain]

  @doc """
  Build a new PromptRoute struct.
  """
  @spec new(attrs :: map()) :: {:ok, t} | {:error, Ecto.Changeset.t()}
  def new(attrs \\ %{}) do
    %PromptRoute{}
    |> cast(attrs, @create_fields)
    |> common_validation()
    |> apply_action(:insert)
  end

  @doc """
  Build a new `PromptRoute` struct and return it or raise an error if invalid.
  """
  @spec new!(attrs :: map()) :: t() | no_return()
  def new!(attrs \\ %{}) do
    case new(attrs) do
      {:ok, param} ->
        param

      {:error, changeset} ->
        raise LangChainError, changeset
    end
  end

  defp common_validation(changeset) do
    changeset
    |> validate_required(@required_fields)
  end

  @doc """
  Return the selected route based on the route name.
  """
  @spec get_selected([t()], route_name :: String.t()) :: nil | t()
  def get_selected(routes, route_name) do
    Enum.find(routes, &(&1.name == route_name))
  end
end
