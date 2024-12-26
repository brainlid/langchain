defmodule LangChain.Document.Loaders.Github do
  @moduledoc """
  Currently this module only supports grabbing issues.

  Extending this to support other resources (like PRs, commits, etc) will require
  a little more work, but at the moment I am not using those resources.
  """

  @behaviour LangChain.Document.Loader

  use Ecto.Schema
  require Logger
  import Ecto.Changeset
  alias __MODULE__
  alias LangChain.Config
  alias LangChain.Document
  alias LangChain.LangChainError

  # allow up to 1 minute for response.
  @receive_timeout 60_000

  @primary_key false
  embedded_schema do
    # API endpoint to use. Defaults to Github's API
    field :endpoint, :string, default: "https://api.github.com"
    field :api_key, :string
    field :receive_timeout, :integer, default: @receive_timeout
  end

  @type t :: %Github{}

  @create_fields [
    :endpoint,
    :api_key,
    :receive_timeout
  ]
  @required_fields [:endpoint, :receive_timeout]

  @spec load(t(), map()) :: [Document.t()] | []
  def load(%Github{} = github, %{type: :issue} = options) do
    make_request(github, options[:repo])
    |> to_documents()
  end

  def load(options) do
    raise LangChainError, "Unsupported type: #{inspect(options[:type])}"
  end

  @spec to_documents(issues :: [map()]) :: [Document.t()]
  def to_documents(issues) do
    Enum.map(issues, fn issue ->
      %Document{
        content: issue.body,
        metadata: %{
          id: issue.id,
          title: issue.title
        },
        type: "github_issue"
      }
    end)
  end

  @spec make_request(t(), String.t()) :: [map()] | no_return()
  def make_request(github, repo) do
    make_request(github, repo, 1, 3, [])
  end

  @spec make_request(t(), String.t(), integer(), integer(), [map()]) :: [map()] | no_return()
  def make_request(_gihub, _repo, _page, 0, _acc) do
    raise LangChainError, "Retries exceeded. Connection failed."
  end

  def make_request(%Github{} = github, repo, page, retry_count, acc) do
    req =
      Req.new(
        url: "#{github.endpoint}/repos/#{repo}/issues?page=#{page}",
        headers: headers(get_api_key(github)),
        receive_timeout: github.receive_timeout,
        retry: :transient,
        max_retries: 3,
        retry_delay: fn attempt -> 300 * attempt end
      )

    req
    |> Req.get()
    # parse the body and return it as parsed structs
    |> case do
      {:ok, %Req.Response{body: data, headers: _headers} = _response} ->
        case process_response(data) do
          {:error, reason} ->
            {:error, reason}

          result ->
            # @TODO check the headers and see if we need to do some pagination
            # if so, call this function recursively with the next page
            result
        end

      {:error, %Req.TransportError{reason: :timeout}} ->
        {:error, "Request timed out"}

      {:error, %Req.TransportError{reason: :closed}} ->
        # Force a retry by making a recursive call decrementing the counter
        Logger.debug(fn -> "Mint connection closed: retry count = #{inspect(retry_count)}" end)
        make_request(github, repo, page, retry_count - 1, acc)

      other ->
        Logger.error("Unexpected and unhandled API response! #{inspect(other)}")
        other
    end
  end

  @spec get_api_key(t()) :: String.t()
  defp get_api_key(%Github{api_key: api_key}) do
    api_key || Config.resolve(:github_key, "")
  end

  @doc """
  Setup a Github client configuration.
  """
  @spec new(attrs :: map()) :: {:ok, t} | {:error, Ecto.Changeset.t()}
  def new(%{} = attrs \\ %{}) do
    %Github{}
    |> cast(attrs, @create_fields)
    |> common_validation()
    |> apply_action(:insert)
  end

  @doc """
  Setup a Guthub client configuration and return it or raise an error if invalid.
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
  end

  defp headers("") do
    %{}
  end

  defp headers(api_key) do
    %{
      "Authorization" => "Bearer #{api_key}"
    }
  end

  def process_response(response) do
    Enum.map(response, fn issue ->
      %{
        :id => issue["id"],
        :title => issue["title"],
        :body => issue["body"]
      }
    end)
  end
end
