defmodule LangChain.Utils.GoogleSchema do
  @moduledoc """
  Removes JSON Schema keywords that Google's Gemini APIs reject.

  Google's function declarations and response schemas accept a
  [select subset](https://ai.google.dev/api/caching#Schema) of an OpenAPI 3.0
  schema object rather than full JSON Schema. The `Schema` type is a protobuf
  message, so a keyword it has no field for is rejected by the parser before
  any semantic validation runs:

      Invalid JSON payload received. Unknown name "additionalProperties" at
      'tools[0].function_declarations[0].parameters': Cannot find field.

  That fails the entire request, not the single tool, and the message names a
  field path rather than the tool, so the cause is not obvious.

  `additionalProperties` is the keyword that matters most in practice, because
  `LangChain.FunctionParam.to_parameters_schema/1` adds it to every schema it
  generates. Without this sanitizing step, any tool declared the ordinary way —
  with a `parameters:` list of `LangChain.FunctionParam` structs — is rejected
  by Gemini.

  ## Semantic loss

  One removal changes what the API enforces. Dropping `additionalProperties:
  false` means Gemini does not reject unexpected properties, so a tool call's
  arguments may contain fields the schema did not declare. Where that matters,
  validate the arguments inside the tool's own function.

  Schemas built around `$ref` are also flattened, since neither `$ref` nor
  `$defs` has a Google equivalent. Inline such schemas before sending them.
  """

  # Keywords that are valid JSON Schema but are not fields of Google's `Schema`
  # type. Supported keywords are deliberately not enumerated: a denylist lets a
  # field Google adds later keep working instead of being silently dropped.
  @unsupported_keywords ~w(
    additionalProperties unevaluatedProperties patternProperties propertyNames
    $schema $ref $defs $comment $id $anchor $dynamicRef $dynamicAnchor $vocabulary
    definitions
    additionalItems prefixItems unevaluatedItems uniqueItems
    contains minContains maxContains
    dependencies dependentRequired dependentSchemas
    oneOf allOf not if then else
    const examples exclusiveMinimum exclusiveMaximum multipleOf
    readOnly writeOnly deprecated
    contentEncoding contentMediaType contentSchema
  )

  # Keys whose values hold nested schemas. Recursion is limited to these
  # because every other value is instance data — `enum` members, a `default`,
  # an `example` — and filtering those would corrupt a value that happens to
  # use a keyword as one of its own field names.
  @schema_bearing_keys ~w(items anyOf)

  @doc """
  Remove the schema keywords Google does not support, recursing through nested
  schemas.

  Non-map input is returned unchanged, so this is safe to apply to a `nil`
  response schema or a schema fragment that is a bare boolean.

  ## Example

      iex> LangChain.Utils.GoogleSchema.sanitize(%{
      ...>   "type" => "object",
      ...>   "additionalProperties" => false,
      ...>   "properties" => %{"city" => %{"type" => "string"}}
      ...> })
      %{"type" => "object", "properties" => %{"city" => %{"type" => "string"}}}
  """
  @spec sanitize(map()) :: map()
  @spec sanitize(any()) :: any()
  def sanitize(%{} = schema) do
    schema
    |> Enum.reject(fn {key, _value} -> to_string(key) in @unsupported_keywords end)
    |> Map.new(fn {key, value} -> {key, sanitize_value(to_string(key), value)} end)
  end

  def sanitize(schema), do: schema

  # `properties` maps caller-chosen names to schemas. Its keys are property
  # names rather than schema keywords, so only the values are sanitized. A
  # parameter named `const` or `examples` has to survive.
  defp sanitize_value("properties", %{} = properties) do
    Map.new(properties, fn {name, sub_schema} -> {name, sanitize(sub_schema)} end)
  end

  # `items` holds a schema; `anyOf` holds a list of them.
  defp sanitize_value(key, value) when key in @schema_bearing_keys do
    sanitize_nested(value)
  end

  defp sanitize_value(_key, value), do: value

  defp sanitize_nested(value) when is_list(value), do: Enum.map(value, &sanitize_nested/1)
  defp sanitize_nested(%{} = value), do: sanitize(value)
  defp sanitize_nested(value), do: value

  @doc """
  Return true when the schema describes an object with no properties.

  Google rejects such a schema with "should be non-empty for OBJECT type", so a
  function declaration that produces one has to omit its parameters entirely
  rather than send an empty object.

  ## Example

      iex> LangChain.Utils.GoogleSchema.empty_object?(%{"type" => "object", "properties" => %{}})
      true

      iex> LangChain.Utils.GoogleSchema.empty_object?(%{"type" => "object", "properties" => %{"a" => %{}}})
      false
  """
  @spec empty_object?(any()) :: boolean()
  def empty_object?(%{"type" => "object", "properties" => properties})
      when map_size(properties) == 0,
      do: true

  def empty_object?(_schema), do: false
end
