defmodule LangChain.NativeTool do
  use Ecto.Schema
  import Ecto.Changeset

  alias __MODULE__
  alias LangChain.LangChainError

  embedded_schema do
    field :name, :string
    field :configuration, :map
  end

  @type t :: %NativeTool{}
  @type configuration :: %{String.t() => any()}

  @create_fields [
    :name,
    :configuration
  ]
  @required_fields [:name]

  @doc """
  Build a new native tool.
  """
  @spec new(attrs :: map()) :: {:ok, t} | {:error, Ecto.Changeset.t()}
  def new(attrs \\ %{}) do
    %NativeTool{}
    |> cast(attrs, @create_fields)
    |> common_validation()
    |> apply_action(:insert)
  end

  @doc """
  Build a new native tool and return it or raise an error if invalid.
  """
  @spec new!(attrs :: map()) :: t() | no_return()
  def new!(attrs \\ %{}) do
    case new(attrs) do
      {:ok, native_tool} ->
        native_tool

      {:error, changeset} ->
        raise LangChainError, changeset
    end
  end

  defp common_validation(changeset) do
    changeset
    |> validate_required(@required_fields)
  end
end
