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

      selected_chain =
        RoutingChain.new(%{
          llm: ChatOpenAI.new(%{model: "gpt-3.5-turbo", stream: false}),
          input_text: "Let's create a marketing blog post about our new product 'Fuzzy Furries'",
          routes: routes,
          default_chain: fallback_chain
        })
        |> RoutingChain.evaluate()

      # The `blog_post_chain` should be returned as the `selected_chain`.

  The `llm` is the model used to make the determination of which route is the
  best match. A smaller, faster LLM may be a great choice for the routing
  decision, then a more complex LLM may be used for a selected route.

  The `default_chain` is required and is used as a fallback if the user's prompt
  doesn't match any of the specified routes.
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
    field :default_chain, :any, virtual: true
    field :verbose, :boolean, default: false
  end

  @type t :: %RoutingChain{}

  @create_fields [:llm, :input_text, :routes, :default_chain, :verbose]
  @required_fields [:llm, :input_text, :routes, :default_chain]

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
    |> Utils.validate_llm_is_struct()
  end

  @doc """
  Run a simple RoutingChain to summarize the user's prompt into a title for the
  conversation. Uses the provided model. Recommend faster, simpler LLMs without
  streaming.

  If it fails to summarize to a title, it returns the default text.

      new_title = RoutingChain.new!(%{
        llm: %ChatOpenAI{model: "gpt-3.5-turbo", stream: false},
        text_to_summarize: "Let's create a marketing blog post about our new product 'Fuzzy Furries'"
      })
      |> RoutingChain.run()

  """
  @spec run(t(), Keyword.t()) ::
          {:ok, LLMChain.t(), Message.t() | [Message.t()]} | {:error, String.t()}
  def run(%RoutingChain{} = chain, opts \\ []) do
    messages =
      [
        Message.new_system!("""
        You analyze the INPUT from the user to identify which category
        it best applies to. If no category seems to be a good fit, assign
        the category DEFAULT. Respond only with the category name.

        REMEMBER: The category MUST be one of the candidate category names
        specified below OR it can be "DEFAULT" if the input is not well
        suited for any of the candidate categories.
        """),
        PromptTemplate.new!(%{
          role: :user,
          text: """
          << CANDIDATE CATEGORIES >>
          <%= for route <- @routes do %>- <%= route.name %>: <%= route.description %>
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
  @spec evaluate(t(), Keyword.t()) :: any()
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

        {:error, reason} ->
          Logger.warning("RoutingChain failed. Reason: #{inspect(reason)}")
          if chain.verbose, do: IO.puts("RoutingChain FAILED IN EXECUTION - USING DEFAULT")
          "DEFAULT"
      end

    # use selected route name to return the matching chain
    if selected_name == "DEFAULT" do
      chain.default_chain
    else
      selected_name
      |> PromptRoute.get_selected(chain.routes)
      |> case do
        %PromptRoute{chain: selected_chain} ->
          selected_chain

        nil ->
          # log, verbose
          Logger.warning("No matching route found. Returning default chain.")
          if chain.verbose, do: IO.puts("NO MATCHING ROUTE FOUND: USING DEFAULT")
          chain.default_chain
      end
    end
  end
end
