# {:ok, p_name} = Langchain.Functions.FunctionParameter.new(%{name: "name", type: "string", description: "The name of a person"})
# {:ok, p_age} = Langchain.Functions.FunctionParameter.new(%{name: "age", type: "integer", description: "The age of the person"})
defmodule Langchain.Functions.FunctionParameter do
  use Ecto.Schema
  import Ecto.Changeset
  alias __MODULE__

  @primary_key false
  embedded_schema do
    field(:name, :string)
    field(:type, :string)
    field(:description, :string)
    field(:enum, {:array, :string})
  end

  @type t :: %FunctionParameter{}

  @create_fields [:name, :description, :type, :enum]
  @required_fields [:name, :type]

  @spec new(attrs :: map()) :: {:ok, t} | {:error, Ecto.Changeset.t()}
  def new(attrs \\ %{}) do
    %FunctionParameter{}
    |> changeset(attrs)
    |> apply_action(:insert)
  end

  def changeset(struct, attrs) do
    struct
    |> cast(attrs, @create_fields)
    |> common_validation()
  end

  def common_validation(changeset) do
    changeset
    |> validate_required(@required_fields)
    |> validate_length(:name, max: 64)
  end

  def for_api(%FunctionParameter{} = param, parameters_map \\ %{}) do
    p_def =
      %{
        "type" => param.type,
        "description" => param.description
      }
      |> put_enum(param)

    Map.put(parameters_map, param.name, p_def)
  end

  def put_enum(data, %FunctionParameter{enum: []} = _param), do: data
  def put_enum(data, %FunctionParameter{enum: nil} = _param), do: data

  def put_enum(data, %FunctionParameter{enum: enum} = _param) do
    Map.put(data, "enum", enum)
  end
end
