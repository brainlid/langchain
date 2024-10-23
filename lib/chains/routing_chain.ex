defmodule LangChain.Chains.RoutingChain do
  @moduledoc """
  Run a router based on a user's initial prompt to determine what category best
  matches from the given options. If there is no good match, the value "DEFAULT"
  is returned.

  Here's an example:

      routes = [
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
      ]

      selected_route =
        RoutingChain.new!(%{
          llm: ChatOpenAI.new!(%{model: "gpt-40-mini", stream: false}),
          input_text: "Let's create a marketing blog post about our new product 'Fuzzy Furies'",
          routes: routes,
          default_route: PromptRoute.new!(%{name: "DEFAULT", chain: fallback_chain})
        })
        |> RoutingChain.evaluate()

      # The PromptRoute for the `blog_post` should be returned as the `selected_route`.

  The `llm` is the model used to make the determination of which route is the
  best match. A smaller, faster LLM may be a great choice for the routing
  decision, then a more complex LLM may be used for a selected route.

  The `default_route` is required and is used as a fallback if the user's prompt
  doesn't match any of the specified routes. It may also be used in some
  fallback error situations as well.
  """
  use Ecto.Schema
  import Ecto.Changeset
  require Logger
  alias LangChain.Routing.PromptRoute
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
    field :routes, {:array, :any}, virtual: true
    field :default_route, :any, virtual: true
    field :verbose, :boolean, default: false
  end

  @type t :: %RoutingChain{}

  @create_fields [:llm, :input_text, :routes, :default_route, :verbose]
  @required_fields [:llm, :input_text, :routes, :default_route]

  @doc """
  Start a new RoutingChain.
  """
  @spec new(attrs :: map()) :: {:ok, t()} | {:error, Ecto.Changeset.t()}
  def new(attrs \\ %{}) do
    %RoutingChain{}
    |> cast(attrs, @create_fields)
    |> common_validation()
    |> apply_action(:insert)
  end

  @doc """
  Start a new RoutingChain and return it or raise an error if invalid.
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
    |> validate_default_route()
    |> Utils.validate_llm_is_struct()
  end

  @doc """
  Run a simple RoutingChain to analyze the input_text and determine which of the
  given routes is the best match.

  A simpler, faster LLM may be a great fit for running the analysis. If it fails
  to find a good match, the `default_route` is used. The `default_route`'s name
  is supplied to the LLM as well. The name "DEFAULT" is suggested for this
  route.
  """
  @spec run(t(), Keyword.t()) ::
          {:ok, LLMChain.t(), Message.t() | [Message.t()]} | {:error, String.t()}
  def run(%RoutingChain{} = chain, opts \\ []) do
    default_name = chain.default_route.name

    messages =
      [
        Message.new_system!("""
        You analyze the INPUT from the user to identify which category
        it best applies to. If no category seems to be a good fit, assign
        the category #{default_name}. Respond only with the category name.

        REMEMBER: The category MUST be one of the candidate category names
        specified below OR it can be "#{default_name}" if the input is not well
        suited for any of the candidate categories.
        """),
        PromptTemplate.new!(%{
          role: :user,
          text: """
          << CANDIDATE CATEGORIES >>
          <%= for route <- @routes do %>- <%= route.name %><%= if route.description do %>: <%= route.description %><% end %>
          <% end %>

          << INPUT >>
          <%= @input %>
          """
        })
      ]
      |> PromptTemplate.to_messages!(%{input: chain.input_text, routes: chain.routes})

    %{llm: chain.llm, verbose: chain.verbose}
    |> LLMChain.new!()
    |> LLMChain.add_messages(messages)
    |> LLMChain.run(opts)
  end

  @doc """
  Runs the RoutingChain and evaluates the result to return the selected chain.
  """
  @spec evaluate(t(), Keyword.t()) :: PromptRoute.t()
  def evaluate(%RoutingChain{} = chain, opts \\ []) do
    selected_name =
      chain
      |> run(opts)
      |> ChainResult.to_string()
      |> case do
        {:ok, name} ->
          Logger.debug("RoutingChain selected #{inspect(name)}")
          if chain.verbose, do: IO.inspect(name, label: "SELECTED ROUTE NAME")
          name

        {:error, _chain, reason} ->
          Logger.warning("RoutingChain failed. Reason: #{inspect(reason)}")
          if chain.verbose, do: IO.puts("RoutingChain FAILED IN EXECUTION - USING DEFAULT")
          "DEFAULT"
      end

    # use selected route name to return the matching chain
    if selected_name == "DEFAULT" do
      chain.default_route
    else
      selected_name
      |> PromptRoute.get_selected(chain.routes)
      |> case do
        %PromptRoute{} = route ->
          route

        nil ->
          # log, verbose
          Logger.warning("No matching route found. Returning default chain.")
          if chain.verbose, do: IO.puts("NO MATCHING ROUTE FOUND: USING DEFAULT")
          chain.default_route
      end
    end
  end

  defp validate_default_route(changeset) do
    case get_field(changeset, :default_route) do
      nil ->
        changeset

      %PromptRoute{} ->
        changeset

      _other ->
        add_error(
          changeset,
          :default_route,
          "must be a PromptRoute"
        )
    end
  end
end
