defmodule LangChain.Document do
  use Ecto.Schema
  import Ecto.Changeset
  require Logger
  alias __MODULE__

  @type t :: %__MODULE__{}

  @primary_key false
  embedded_schema do
    field :content, :string
    field :metadata, :map
    field :type, :string
  end

  @create_fields [:content, :metadata, :type]
  @required_fields [:content, :type]

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

  @doc """
    Build a new document and error out if the changeset is invalid
  """
  @spec new!(attrs :: map()) :: t()
  def new!(attrs \\ %{}) do
    case new(attrs) do
      {:ok, doc} ->
        doc

      {:error, changeset} ->
        raise LangChainError, changeset
    end
  end

  defp common_validations(changeset) do
    changeset
    |> validate_required(@required_fields)
  end
end
