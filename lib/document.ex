defmodule Langchain.Document do
  use Ecto.Schema
  import Ecto.Changeset
  require Logger
  alias __MODULE__
  alias Langchain.LangchainError

  @primary_key false
  embedded_schema do
    field :page_content, :string
    field :metadata, :map
  end

  @type t :: %Document{}

  @create_fields [:page_content, :metadata]
  @required_fields [:page_content]

  @doc """
  Build a new document and return an `:ok`/`:error` tuple with the result.
  """
  @spec new(attrs :: map()) :: {:ok, t()} | {:error, Ecto.Changeset.t()}
  def new(attrs \\ %{}) do
    %Document{}
    |> cast(attrs, @create_fields)
    |> apply_action(:insert)
  end
end