defmodule LangChain.Function do
  @moduledoc """
  Defines a "function" that can be provided to an LLM for the LLM to optionally
  execute and pass argument data to.

  A function is defined using a schema.

  * `name` - The name of the function given to the LLM.
  * `description` - A description of the function provided to the LLM. This
    should describe what the function is used for or what it returns. This
    information is used by the LLM to decide which function to call and for what
    purpose.
  * ` parameters` - A list of `Function.FunctionParam` structs that are
    converted to a JSONSchema format. (Use in place of `parameters_schema`)
  * ` parameters_schema` - A [JSONSchema
    structure](https://json-schema.org/learn/getting-started-step-by-step.html)
    that describes the required data structure format for how arguments are
    passed to the function. (Use if greater control or unsupported features are
    needed.)
  * `function` - An Elixir function to execute when an LLM requests to execute
    the function. The function should return `{:ok, "and a text response"}`,
    `{:error, "and text explanation of the error"}` or just plain `"text
    response"`, which is returned to the LLM.
  * `async` - Boolean value that flags if this can function can be executed
    asynchronously, potentially concurrently with other calls to the same
    function. Defaults to `true`.

  When passing arguments from an LLM to a function, they go through a single
  `map` argument. This allows for multiple keys or named parameters.

  ## Example

  This example defines a function that an LLM can execute for performing basic
  math calculations. **NOTE:** This is a partial implementation of the
  `LangChain.Tools.Calculator`.

      Function.new(%{
        name: "calculator",
        description: "Perform basic math calculations",
        parameters_schema: %{
          type: "object",
          properties: %{
            expression: %{type: "string", description: "A simple mathematical expression."}
          },
          required: ["expression"]
        },
        function:
          fn(%{"expression" => expr} = _args, _context) ->
            {:ok, "42?"}
          end)
      })

  The `function` attribute is an Elixir function that can be executed when the
  function is "called" by the LLM.

  The `args` argument is the JSON data passed by the LLM after being parsed to a
  map.

  The `context` argument is passed through as the `context` on a
  `LangChain.Chains.LLMChain`. This is whatever context data is needed for the
  function to do it's work.

  Context examples may be user_id, account_id, account struct, billing level,
  etc.

  ## Function Parameters

  The `parameters` field is a list of `LangChain.FunctionParam` structs. This is
  a convenience for defining the parameters to the function. If it does not work
  for more complex use-cases, then use the `parameters_schema` to declare it as
  needed.

  The `parameters_schema` is an Elixir map that follows a
  [JSONSchema](https://json-schema.org/learn/getting-started-step-by-step.html)
  structure. It is used to define the required data structure format for
  receiving data to the function from the LLM.

  NOTE: Only use `parameters` or `parameters_schema`, not both.

  ## Expanded Parameter Examples

  Function with no arguments:

      alias LangChain.Function

      Function.new!(%{name: "get_current_user_info"})

  Function that takes a simple required argument:

      alias LangChain.FunctionParam

      Function.new!(%{name: "set_user_name", parameters: [
        FunctionParam.new!(%{name: "user_name", type: :string, required: true})
      ]})

  Function that takes an array of strings:

      Function.new!(%{name: "set_tags", parameters: [
        FunctionParam.new!(%{name: "tags", type: :array, item_type: "string"})
      ]})

  Function that takes two arguments and one is an object/map:

      Function.new!(%{name: "update_preferences", parameters: [
        FunctionParam.new!(%{name: "unique_code", type: :string, required: true})
        FunctionParam.new!(%{name: "data", type: :object, object_properties: [
          FunctionParam.new!(%{name: "auto_complete_email", type: :boolean}),
          FunctionParam.new!(%{name: "items_per_page", type: :integer}),
        ]})
      ]})

  The `LangChain.FunctionParam` is nestable allowing for arrays of object and
  objects with nested objects.
  """
  use Ecto.Schema
  import Ecto.Changeset
  require Logger
  alias __MODULE__
  alias LangChain.LangChainError

  @primary_key false
  embedded_schema do
    field :name, :string
    field :description, :string
    # Optional text the UI can display for when the function is executed.
    field :display_text, :string
    # Optional flag to indicate if the function should be executed in strict mode.
    # Defaults to `false`.
    field :strict, :boolean, default: false
    # flag if the function should be auto-evaluated. Defaults to `false`
    # requiring an explicit step to perform the evaluation.
    # field :auto_evaluate, :boolean, default: false
    field :function, :any, virtual: true

    # Track if the function can be executed async. Defaults to `true`.
    field :async, :boolean, default: true

    # parameters_schema is a map used to express a JSONSchema structure of inputs and what's required
    field :parameters_schema, :map
    # parameters is a list of `LangChain.FunctionParam` structs.
    field :parameters, {:array, :any}, default: []
  end

  @type t :: %Function{}
  @type arguments :: %{String.t() => any()}
  @type context :: nil | %{atom() => any()}

  @create_fields [
    :name,
    :description,
    :display_text,
    :strict,
    :parameters_schema,
    :parameters,
    :function,
    :async
  ]
  @required_fields [:name]

  @doc """
  Build a new function.
  """
  @spec new(attrs :: map()) :: {:ok, t} | {:error, Ecto.Changeset.t()}
  def new(attrs \\ %{}) do
    %Function{}
    |> cast(attrs, @create_fields)
    |> common_validation()
    |> apply_action(:insert)
  end

  @doc """
  Build a new function and return it or raise an error if invalid.
  """
  @spec new!(attrs :: map()) :: t() | no_return()
  def new!(attrs \\ %{}) do
    case new(attrs) do
      {:ok, function} ->
        function

      {:error, changeset} ->
        raise LangChainError, changeset
    end
  end

  defp common_validation(changeset) do
    changeset
    |> validate_required(@required_fields)
    |> validate_length(:name, max: 64)
    |> ensure_single_parameter_option()
    |> validate_function_and_arity()
  end

  @doc """
  Execute the function passing in arguments and additional optional context.
  This is called by a `LangChain.Chains.LLMChain` when a `Function` execution is
  requested by the LLM.
  """
  @spec execute(t(), arguments(), context()) :: any() | no_return()
  def execute(%Function{function: fun} = function, arguments, context) do
    Logger.debug("Executing function #{inspect(function.name)}")

    try do
      # execute the function and normalize the results. Want :ok/:error tuples
      case fun.(arguments, context) do
        {:ok, result} ->
          # successful execution.
          {:ok, result}

        {:error, reason} when is_binary(reason) ->
          {:error, reason}

        {:error, reason} ->
          # turn the error response into a string.
          {:error, "#{inspect(reason)}"}

        text when is_binary(text) ->
          {:ok, text}

        other ->
          Logger.error(
            "Function #{function.name} unexpectedly returned #{inspect(other)}. Expect a string. Unable to present as response to LLM."
          )

          {:error, "An unexpected response was returned from the tool."}
      end
    rescue
      err ->
        Logger.error("Function #{function.name} failed in execution. Exception: #{inspect(err)}")
        {:error, inspect(err)}
    end
  end

  defp validate_function_and_arity(changeset) do
    function = get_field(changeset, :function)

    if is_function(function) do
      case Elixir.Function.info(function, :arity) do
        {:arity, 2} ->
          changeset

        {:arity, n} ->
          add_error(changeset, :function, "expected arity of 2 but has arity #{inspect(n)}")
      end
    else
      add_error(changeset, :function, "is not an Elixir function")
    end
  end

  defp ensure_single_parameter_option(changeset) do
    params_list = get_field(changeset, :parameters)
    schema_map = get_field(changeset, :parameters_schema)

    cond do
      # can't have both
      is_map(schema_map) and !Enum.empty?(params_list) ->
        add_error(changeset, :parameters, "Cannot use both parameters and parameters_schema")

      true ->
        changeset
    end
  end

  @doc """
  Given a list of functions, return the `display_text` for the named function.
  If it not found, return the fallback text.
  """
  @spec get_display_text([t()], String.t(), String.t()) :: String.t()
  def get_display_text(functions, function_name, fallback_text \\ "Perform action") do
    case Enum.find(functions, &(&1.name == function_name)) do
      nil -> fallback_text
      %Function{} = func -> func.display_text
    end
  end
end
