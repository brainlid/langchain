defmodule LangChain.Routing.PromptRoute do
  @moduledoc """
  Defines a route or direction a prompting interaction with an LLM can take.

  This helps add complexity and specificity to an LLM's instructions without
  using many tokens and helps avoid combining many things into a single prompt
  setup. This works by letting the user's initial prompt define how the next
  prompt is setup. A first pass is run on the prompt to determine which of the
  provided PromptRoutes is the best match.

  To better understand, let's see example of taking the user's input and
  classifying what supported activity or specialty it is best matched with.

  User prompt:
  > Let's create a marketing focused blog post comparing three types of our
  > trailer hitch products.

  We want our assistant to use different prompts for the different types of
  activities it's capable of doing. Let's define the activities like this:

      [
        PromptRoute.new!(%{
          name: "marketing_email",
          description: "Create a marketing focused email",
          chain: marketing_email_chain
        }),
        PromptRoute.new!(%{
          name: "blog_post",
          description: "Create a blog post that will be linked from the company's landing page",
          chain: blog_post_chain
        }),
        PromptRoute.new!(%{
          name: "support_triage",
          description: "Triage a customer support request for severity and requested resolution",
          chain: support_triage_chain
        })
      ]

  Given the user's prompt and the routes we provided, it will choose
  "blog_post". This leads to an LLMChain being setup for that purpose where the
  user's initial prompt can be re-run against the prompts we define for that
  route.

  When a `chain` is linked to each route, we can specify the model to use, the
  prompt templates we want, and associate functions that can be used to
  accomplish the task.

  The selected LLMChain can then be run through a UI for chat interactions where
  the user and AI work together.

  The `name` will be returned by the LLM when a particular route is selected.

  The `description` is given to the LLM to help it determine if the route should
  be selected based on the current input prompt.
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
    # Contains prompt templates, model config, and any desired functions.
    field :chain, :any, virtual: true
  end

  @type t :: %PromptRoute{}

  @create_fields [
    :name,
    :description,
    :chain
  ]
  @required_fields [:name]

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
  @spec get_selected(route_name :: String.t(), [t()]) :: nil | t()
  def get_selected(route_name, routes) when is_binary(route_name) and is_list(routes) do
    Enum.find(routes, &(&1.name == route_name))
  end
end
