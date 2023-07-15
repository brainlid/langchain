# {:ok, f} = Langchain.Function.new(%{name: "register_person", description: "Register a new person in the system", required: ["name"], parameters: [p_name, p_age]})
# NOTE: New in OpenAI - https://openai.com/blog/function-calling-and-other-api-updates
#  - 13 June 2023
# NOTE: Pretty much takes the place of a Langchain "Tool".
defmodule Langchain.Functions.Function do
  use Ecto.Schema
  import Ecto.Changeset
  alias __MODULE__
  alias Langchain.Functions.FunctionParameter

  @primary_key false
  embedded_schema do
    field :name, :string
    field :description, :string
    # flag if the function should be auto-evaluated. Defaults to `false`
    # requiring an explicit step to perform the evaluation.
    # field :auto_evaluate, :boolean, default: false
    field :function, :any, virtual: true
    # parameters is a map used to express a JSONSchema structure of inputs and what's required
    field :parameters_schema, :map

    # embeds_many(:parameters, FunctionParameter)
  end

  @type t :: %Function{}

  @create_fields [:name, :description, :parameters_schema, :function]
  @required_fields [:name]

  @spec new(attrs :: map()) :: {:ok, t} | {:error, Ecto.Changeset.t()}
  def new(attrs \\ %{}) do
    %Function{}
    |> cast(attrs, @create_fields)
    |> common_validation()
    |> apply_action(:insert)
  end

  def common_validation(changeset) do
    changeset
    |> validate_required(@required_fields)
    |> validate_length(:name, max: 64)
  end
end

defimpl Langchain.ForOpenAIApi, for: Langchain.Functions.Function do
  alias Langchain.Functions.Function

  def for_api(%Function{} = fun) do
    %{
      "name" => fun.name,
      "description" => fun.description,
      "parameters" => get_parameters(fun)
    }
  end

  defp get_parameters(%Function{parameters_schema: nil} = _fun) do
    %{
      "type" => "object",
      "properties" => %{}
    }
  end

  defp get_parameters(%Function{parameters_schema: schema} = _fun) do
    schema
  end
end
