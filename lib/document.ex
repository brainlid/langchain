defmodule LangChain.Document do
  use Ecto.Schema
  import Ecto.Changeset
  require Logger
  alias __MODULE__
  alias LangChain.LangChainError

  @type t :: %__MODULE__{}

  @primary_key false
  embedded_schema do
    field :page_content, :string
    field :metadata, :map
  end

  @create_fields [:page_content, :metadata]
  @required_fields [:page_content]

  @doc """
  Build a new document and return an `:ok`/`:error` tuple with the result.
  """
  @spec new(attrs :: map()) :: {:ok, t()} | {:error, Ecto.Changeset.t()}
  def new(attrs \\ %{}) do
    %Document{}
    |> cast(attrs, @create_fields)
    |> common_validations()
    |> apply_action(:insert)
  end

  defp common_validations(changeset) do
    changeset
    |> validate_required(@required_fields)
  end
end
