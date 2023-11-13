defmodule LangChain.FunctionParam do
  @moduledoc """
  Define a function parameter as a struct. Used to generate the expected
  JSONSchema data for describing one or more arguments being passed to a
  `LangChain.Function`.

  Note: This is not intended to be a fully compliant implementation of
  [JSONSchema
  types](https://json-schema.org/understanding-json-schema/reference/type). This
  is intended to be a convenience for working with the most common situations
  when working with an LLM that understands JSONSchema.

  Supports:

  * simple values - string, integer, number, boolean
  * enum values - `enum: ["alpha", "beta"]`. The values can be strings,
    integers, etc.
  * array values - `type: :array` couples with `item_type: "string"` to express
    it is an array of.
    * `item_type` is optional. When omitted, it can be a mixed array.
    * `item_type: "object"` allows for creating an array of objects. Use
      `object_properties: [...]` to describe the structure of the objects.
  * objects - Define the object's expected values or supported structure using
    `object_properties`.

  The function `to_parameters_schema/1` is used to convert a list of
  `FunctionParam` structs into a JSONSchema formatted data map.
  """
  use Ecto.Schema
  import Ecto.Changeset
  require Logger
  alias __MODULE__
  alias LangChain.LangChainError

  @primary_key false
  embedded_schema do
    field :name, :string
    field :type, Ecto.Enum, values: [:string, :integer, :number, :boolean, :array, :object]
    field :item_type, :string
    field :enum, {:array, :any}, default: []
    field :description, :string
    field :required, :boolean, default: false
    # list of object properties. Only used for objects
    field :object_properties, {:array, :any}, default: []
  end

  @type t :: %FunctionParam{}

  @create_fields [
    :name,
    :type,
    :item_type,
    :enum,
    :description,
    :required,
    :object_properties
  ]
  @required_fields [:name, :type]

  @doc """
  Build a new FunctionParam struct.
  """
  @spec new(attrs :: map()) :: {:ok, t} | {:error, Ecto.Changeset.t()}
  def new(attrs \\ %{}) do
    %FunctionParam{}
    |> cast(attrs, @create_fields)
    # |> Ecto.Changeset.put_embed(:object_properties, value: )
    |> common_validation()
    |> apply_action(:insert)
  end

  @doc """
  Build a new `FunctionParam` struct and return it or raise an error if invalid.
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
    |> validate_enum()
    |> validate_array_type()
    |> validate_object_type()
  end

  defp validate_enum(changeset) do
    values = get_field(changeset, :enum, [])
    type = get_field(changeset, :type)

    cond do
      type in [:string, :integer, :number] and !Enum.empty?(values) ->
        changeset

      # not an :enum field but gave enum, error
      !Enum.empty?(values) ->
        add_error(changeset, :enum, "not allowed for type #{inspect(type)}")

      # no enum given
      true ->
        changeset
    end
  end

  defp validate_array_type(changeset) do
    item = get_field(changeset, :item_type)
    type = get_field(changeset, :type)

    cond do
      # can only use item_type field when an array
      type != :array and item != nil ->
        add_error(changeset, :item_type, "not allowed for type #{inspect(type)}")

      # okay
      true ->
        changeset
    end
  end

  defp validate_object_type(changeset) do
    props = get_field(changeset, :object_properties)
    item = get_field(changeset, :item_type)
    type = get_field(changeset, :type)

    cond do
      # allowed case for object_properties
      type == :object and !Enum.empty?(props) ->
        changeset

      # allowed case for object_properties
      type == :array and item == "object" and !Enum.empty?(props) ->
        changeset

      # object type but missing the properties. Add error
      type == :object ->
        add_error(changeset, :object_properties, "is required for object type")

      # when an array of objects, object_properties is required
      type == :array and item == "object" and Enum.empty?(props) ->
        add_error(changeset, :object_properties, "required when array type of object is used")

      # has object_properties but not one of the allowed cases
      !Enum.empty?(props) and (!(type == :array and item == "object") and !(type == :object)) ->
        add_error(changeset, :object_properties, "not allowed for type #{inspect(type)}")

      # not an object and didn't give object_properties
      true ->
        changeset
    end
  end

  @doc """
  Return the list of required property names.
  """
  @spec required_properties(params :: [t()]) :: [String.t()]
  def required_properties(params) when is_list(params) do
    params
    |> Enum.reduce([], fn p, acc ->
      if p.required do
        [p.name | acc]
      else
        acc
      end
    end)
    |> Enum.reverse()
  end

  @doc """
  Transform a list of `FunctionParam` structs into a map expressing the structure
  in a JSONSchema compatible way.
  """
  @spec to_parameters_schema([t()]) :: %{String.t() => any()}
  def to_parameters_schema(params) when is_list(params) do
    %{
      "type" => "object",
      "properties" => Enum.reduce(params, %{}, &to_json_schema(&2, &1)),
      "required" => required_properties(params)
    }
  end

  @doc """
  Transform a `FunctionParam` to a JSONSchema compatible definition that is
  added to the passed in `data` map.
  """
  @spec to_json_schema(data :: map(), t()) :: map()
  def to_json_schema(%{} = data, %FunctionParam{type: type} = param)
      when type in [:string, :integer, :number, :boolean] do
    settings =
      %{"type" => to_string(type)}
      |> include_enum_value(param)
      |> description_for_schema(param.description)

    Map.put(data, param.name, settings)
  end

  def to_json_schema(%{} = data, %FunctionParam{type: :array, item_type: nil} = param) do
    settings =
      %{"type" => "array"}
      |> description_for_schema(param.description)

    Map.put(data, param.name, settings)
  end

  def to_json_schema(%{} = data, %FunctionParam{type: :array, item_type: "object"} = param) do
    settings =
      %{"type" => "array", "items" => to_parameters_schema(param.object_properties)}
      |> description_for_schema(param.description)

    Map.put(data, param.name, settings)
  end

  def to_json_schema(%{} = data, %FunctionParam{type: :array, item_type: item_type} = param) do
    settings =
      %{"type" => "array", "items" => %{"type" => item_type}}
      |> description_for_schema(param.description)

    Map.put(data, param.name, settings)
  end

  def to_json_schema(%{} = data, %FunctionParam{type: :object, object_properties: props} = param) do
    settings =
      props
      |> to_parameters_schema()
      |> description_for_schema(param.description)

    Map.put(data, param.name, settings)
  end

  # conditionally add the description field if set
  defp description_for_schema(data, nil), do: data

  defp description_for_schema(data, description) when is_binary(description) do
    Map.put(data, "description", description)
  end

  defp include_enum_value(data, %FunctionParam{type: type, enum: values} = _param)
       when type in [:string, :integer, :number] and values != [] do
    Map.put(data, "enum", values)
  end

  defp include_enum_value(data, %FunctionParam{} = _param), do: data
end
