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
    the function. The function can return a string, a tuple, or a
    `%ToolResult{}` struct for advanced control. Returning a ToolResult allows
    for multi-modal responses (list of ContentParts), cache control, and
    processed_content.
  * `parse_args` - An optional 1-arity function that runs **before** `function`
    to parse, coerce, and validate the raw arguments handed back by the LLM.
    See "Parsing arguments before execution" below for the full contract.
  * `async` - Boolean value that flags if this can function can be executed
    asynchronously, potentially concurrently with other calls to the same
    function. Defaults to `false`.
  * `options` - A Keyword list of options that can be passed to the LLM. For
    example, this can be used for passing caching config to Anthropic.

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

  Context examples could be data like user_id, account_id, account struct,
  billing level, etc.

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

  ## Example that also stores the Elixir result

  Sometimes we want to process a `ToolCall` from the LLM and keep the processed
  Elixir data for ourselves. This is particularly useful when using an LLM to
  perform structured data extraction. Our Elixir function may even process that
  data into a newly created Ecto Schema database entry. The result of the
  `ToolCall` that goes back to the LLM must be in a string form. That typically
  means returning a JSON string of the result data.

  To make it easier to process the data, return a string response to the LLM,
  but **keep** the original Elixir data as well, our Elixir function can return
  a 3-tuple result.

      Function.new!(%{name: "create_invoice",
        parameters: [
          FunctionParam.new!(%{name: "vendor_name", type: :string, required: true})
          FunctionParam.new!(%{name: "total_amount", type: :string, required: true})
        ],
        function: &execute_create_invoice/2
      })

      # ...

      def execute_create_invoice(args, %{account_id: account_id} = _context) do
        case MyApp.Invoices.create_invoice(account_id, args) do
          {:ok, invoice} ->
            {:ok, "SUCCESS", invoice}

          {:error, changeset} ->
            {:error, "ERROR: " <> LangChain.Utils.changeset_error_to_string(changeset)}
        end
      end

  In this example, the `LangChain.Function` is tied to the
  `MyApp.Invoices.create_invoice/2` function in our application.

  The Elixir function returns a 3-tuple result. The `"SUCCESS"` is returned to
  the LLM. In our scenario, we don't care to return a JSON version of the
  invoice. The important part is we return the actual
  `%MyApp.Invoices.Invoice{}` struct in the tuple. This is stored on the
  `LangChain.ToolResult`'s `processed_content` field.

  This is really helpful when all we want is the final, fully processed Elixir
  result. This pairs well with the `LLMChain.run(chain, mode: :until_success)`.
  This is when we want the LLM to perform some data extraction and it should be
  re-run until it succeeds and we have our final, processed result in the
  `ToolResult`.

  Note: The LLM may issue one or more `ToolCall`s in a single assistant message.
  Each Elixir function's `ToolResult` may contain a `processed_content`.

  ## Explicit ToolResult Control

  For advanced use cases where you need explicit control over the `ToolResult`
  structure or want to set LLM-specific options, your Elixir function can return
  a fully constructed `%ToolResult{}` struct. The `content` field can be a list
  of ContentParts for multi-modal responses.

  This approach is particularly useful when you need to:

  * Set LLM-specific options (like Anthropic's `cache_control`)
  * Set custom error states beyond simple string responses
  * Customize the `display_text` for the ToolResult
  * Provide detailed metadata in the `options` field

  The `options` field can contain any LLM-specific configuration that gets
  passed through to the chat model's API conversion layer. If an LLM does not
  support it, it will be ignored.

  ## Parsing arguments before execution

  LangChain's built-in `parameters: [%FunctionParam{}]` declaration only does a
  required-key presence check at execute time. `parameters_schema:` is treated
  as a hint sent to the LLM and is not enforced on the way back in. Provider
  "strict mode" closes the gap somewhat, but is best-effort and varies by
  provider.

  The optional `:parse_args` callback fills this gap. It runs **after** the
  built-in required-key check and **before** the user-supplied `function`. The
  parser receives the raw, string-keyed arguments map from the LLM and returns
  one of three shapes:

      :ok                                # arguments are fine, hand them to `function` as-is
      {:ok, parsed_arguments :: map()}   # use these parsed/coerced arguments instead
      {:error, reason :: String.t()}     # reject the call, return `reason` to the LLM

  On rejection, the tool's body is **not** run. The error string flows through
  as the tool's response, so the model sees a structured "your args were wrong"
  message and can self-correct. Tool-execution callbacks (e.g.
  `:on_tool_response_created`) and the `[:langchain, :tool, :call]` telemetry
  span still fire, meaning failed parses are observable for telemetry, token
  accounting, and trajectory analysis.

  This is a "parse, don't validate" hook: tools that need typed/coerced
  arguments parse once here and pattern-match the parsed result in `function`,
  rather than re-parsing inside the body.

  LangChain takes no dependency on any specific schema library. Adapters for
  `Zoi`, `NimbleOptions`, `Ecto.Changeset`, `JSV`, or hand-rolled checks all
  conform to the same `:ok | {:ok, map()} | {:error, String.t()}` contract.

      defp parse_args(args) do
        case Zoi.parse(@params, args) do
          {:ok, parsed} -> {:ok, parsed}
          {:error, errors} -> {:error, format_zoi_errors(errors)}
        end
      end

      Function.new!(%{
        name: "...",
        parameters_schema: ReqLLM.Schema.to_json(@params),
        parse_args: &parse_args/1,
        function: &execute/2
      })
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

    # Track if the function can be executed async. Defaults to `false`.
    field :async, :boolean, default: false

    # parameters_schema is a map used to express a JSONSchema structure of inputs and what's required
    field :parameters_schema, :map
    # parameters is a list of `LangChain.FunctionParam` structs.
    field :parameters, {:array, :any}, default: []

    # Optional pre-execution argument parser. See module doc for the contract.
    field :parse_args, :any, virtual: true

    field :options, :any, virtual: true, default: []
  end

  @type t :: %Function{}
  @type arguments :: %{String.t() => any()}
  @type context :: nil | %{atom() => any()}

  @typedoc """
  Return shape for a `:parse_args` callback. See module doc for full details.
  """
  @type parse_result ::
          :ok
          | {:ok, parsed_arguments :: map()}
          | {:error, reason :: String.t()}

  @typedoc """
  Pre-execution argument parser. A 1-arity function that takes the raw
  arguments map handed back by the LLM and returns a `t:parse_result/0`.
  """
  @type parse_args :: (arguments() -> parse_result())

  @create_fields [
    :name,
    :description,
    :display_text,
    :strict,
    :parameters_schema,
    :parameters,
    :function,
    :parse_args,
    :async,
    :options
  ]
  @required_fields [:name]

  @doc """
  Build a new function.
  """
  @spec new(attrs :: map()) :: {:ok, t()} | {:error, Ecto.Changeset.t()}
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

  @spec common_validation(Ecto.Changeset.t()) :: Ecto.Changeset.t()
  defp common_validation(changeset) do
    changeset
    |> validate_required(@required_fields)
    |> validate_length(:name, max: 64)
    |> validate_parameter_exclusivity()
    |> validate_function_arity()
    |> validate_parse_args()
  end

  @doc """
  Execute the function passing in arguments and additional optional context.
  This is called by a `LangChain.Chains.LLMChain` when a `Function` execution is
  requested by the LLM.
  """
  @spec execute(t(), arguments(), context()) :: any() | no_return()
  def execute(%Function{function: fun} = function, arguments, context) do
    Logger.debug("Executing function #{inspect(function.name)}")

    with :ok <- validate_required_params(function, arguments),
         {:ok, parsed_arguments} <- run_parse_args(function, arguments) do
      execute_with_error_handling(function, fun, parsed_arguments, context)
    end
  end

  # Invokes the optional `:parse_args` callback. When absent, passes the
  # arguments through unchanged. When present, accepts `:ok`, `{:ok, map}`, or
  # `{:error, reason}`. Other return shapes — and exceptions raised by the
  # parser — are normalized to `{:error, reason}` so the calling tool path
  # produces a `ToolResult{is_error: true}` rather than crashing. This keeps
  # `:on_tool_response_created` callbacks and `[:langchain, :tool, :call]`
  # telemetry firing on parse failures, which downstream consumers rely on for
  # token usage accounting and trajectory analysis.
  @spec run_parse_args(t(), arguments()) :: {:ok, map()} | {:error, String.t()}
  defp run_parse_args(%Function{parse_args: nil}, arguments), do: {:ok, arguments}

  defp run_parse_args(%Function{parse_args: parser, name: name}, arguments)
       when is_function(parser, 1) do
    try do
      parser.(arguments)
      |> normalize_parse_result(name, arguments)
    rescue
      err ->
        Logger.warning(fn ->
          "Function #{name} :parse_args raised an exception. " <>
            LangChainError.format_exception(err, __STACKTRACE__)
        end)

        {:error, "ERROR: #{LangChainError.format_exception(err, __STACKTRACE__, :short)}"}
    end
  end

  defp normalize_parse_result(:ok, _name, arguments), do: {:ok, arguments}
  defp normalize_parse_result({:ok, %{} = parsed}, _name, _arguments), do: {:ok, parsed}

  defp normalize_parse_result({:error, reason}, _name, _arguments) when is_binary(reason),
    do: {:error, reason}

  defp normalize_parse_result({:error, reason}, _name, _arguments),
    do: {:error, "#{inspect(reason)}"}

  defp normalize_parse_result(other, name, _arguments) do
    Logger.warning(
      "Function #{name} :parse_args returned an unexpected shape: #{inspect(other)}. " <>
        "Expected :ok | {:ok, map} | {:error, reason}."
    )

    {:error, "parse_args returned unexpected shape: #{inspect(other)}"}
  end

  @doc """
  Given a list of functions, return the `display_text` for the named function.
  If it not found, return the fallback text.
  """
  @spec get_display_text([t()], String.t(), String.t()) :: String.t()
  def get_display_text(functions, function_name, fallback_text \\ "Perform action")

  def get_display_text(functions, function_name, fallback_text) do
    case Enum.find(functions, &(&1.name == function_name)) do
      nil -> fallback_text
      %Function{display_text: display_text} -> display_text
    end
  end

  # Validates that the function field contains a function with arity 2
  @spec validate_function_arity(Ecto.Changeset.t()) :: Ecto.Changeset.t()
  defp validate_function_arity(changeset) do
    changeset
    |> get_field(:function)
    |> do_validate_function_arity(changeset)
  end

  @spec do_validate_function_arity(any(), Ecto.Changeset.t()) :: Ecto.Changeset.t()
  defp do_validate_function_arity(function, changeset) when is_function(function, 2) do
    changeset
  end

  defp do_validate_function_arity(function, changeset) when is_function(function) do
    {:arity, arity} = Elixir.Function.info(function, :arity)
    add_error(changeset, :function, "expected arity of 2 but has arity #{inspect(arity)}")
  end

  defp do_validate_function_arity(_not_a_function, changeset) do
    add_error(changeset, :function, "is not an Elixir function")
  end

  # Validates that :parse_args, if set, is a 1-arity function.
  @spec validate_parse_args(Ecto.Changeset.t()) :: Ecto.Changeset.t()
  defp validate_parse_args(changeset) do
    case get_field(changeset, :parse_args) do
      nil ->
        changeset

      parser when is_function(parser, 1) ->
        changeset

      parser when is_function(parser) ->
        {:arity, arity} = Elixir.Function.info(parser, :arity)
        add_error(changeset, :parse_args, "expected arity of 1 but has arity #{inspect(arity)}")

      _other ->
        add_error(changeset, :parse_args, "is not an Elixir function")
    end
  end

  # Validates that only one of parameters or parameters_schema is provided
  @spec validate_parameter_exclusivity(Ecto.Changeset.t()) :: Ecto.Changeset.t()
  defp validate_parameter_exclusivity(changeset) do
    params_list = get_field(changeset, :parameters)
    schema_map = get_field(changeset, :parameters_schema)

    do_validate_parameter_exclusivity(changeset, params_list, schema_map)
  end

  @spec do_validate_parameter_exclusivity(Ecto.Changeset.t(), list(), map() | nil) ::
          Ecto.Changeset.t()
  defp do_validate_parameter_exclusivity(changeset, params, schema)
       when is_map(schema) and is_list(params) and params != [] do
    add_error(changeset, :parameters, "Cannot use both parameters and parameters_schema")
  end

  defp do_validate_parameter_exclusivity(changeset, _params, _schema), do: changeset

  @spec execute_with_error_handling(t(), function(), arguments(), context()) ::
          {:ok, any()}
          | {:ok, any(), any()}
          | {:interrupt, String.t(), any()}
          | {:error, String.t()}
  defp execute_with_error_handling(function, fun, arguments, context) do
    fun.(arguments, context)
    |> normalize_execution_result(function)
  rescue
    err ->
      Logger.warning(fn ->
        "Function! #{function.name} failed in execution. Exception: #{LangChainError.format_exception(err, __STACKTRACE__)}"
      end)

      {:error, "ERROR: #{LangChainError.format_exception(err, __STACKTRACE__, :short)}"}
  end

  # Normalizes the various return types from function execution into consistent tagged tuples
  @spec normalize_execution_result(any(), t()) ::
          {:ok, any()}
          | {:ok, any(), any()}
          | {:interrupt, String.t(), any()}
          | {:error, String.t()}
  defp normalize_execution_result({:ok, llm_result, processed_content}, _function) do
    {:ok, llm_result, processed_content}
  end

  defp normalize_execution_result({:ok, result}, _function) do
    {:ok, result}
  end

  defp normalize_execution_result({:interrupt, message, data}, _function)
       when is_binary(message) do
    {:interrupt, message, data}
  end

  defp normalize_execution_result({:error, reason}, _function) when is_binary(reason) do
    {:error, reason}
  end

  defp normalize_execution_result({:error, reason}, _function) do
    {:error, "#{inspect(reason)}"}
  end

  defp normalize_execution_result(text, _function) when is_binary(text) do
    {:ok, text}
  end

  defp normalize_execution_result(parts, _function) when is_list(parts) do
    {:ok, parts}
  end

  defp normalize_execution_result(other, function) do
    Logger.warning(
      "Function #{function.name} unexpectedly returned #{inspect(other)}. Expect a string. Unable to present as response to LLM."
    )

    {:error, "An unexpected response was returned from the tool."}
  end

  # Validates that all required parameters are present in the arguments
  @spec validate_required_params(t(), arguments()) :: :ok | {:error, String.t()}
  defp validate_required_params(%Function{parameters: params}, arguments)
       when is_list(params) and params != [] do
    params
    |> collect_required_param_names()
    |> find_missing_params(arguments)
    |> format_missing_params_result()
  end

  defp validate_required_params(_function, _arguments), do: :ok

  # Extracts names of required parameters from the parameter list
  @spec collect_required_param_names([struct()]) :: [String.t()]
  defp collect_required_param_names(params) do
    Enum.reduce(params, [], fn param, acc ->
      if param.required, do: [param.name | acc], else: acc
    end)
  end

  # Finds parameters that are required but missing from the arguments
  @spec find_missing_params([String.t()], arguments()) :: [String.t()]
  defp find_missing_params([], _arguments), do: []

  defp find_missing_params(required_names, arguments) do
    Enum.reject(required_names, &Map.has_key?(arguments, &1))
  end

  # Formats the result based on whether any parameters are missing
  @spec format_missing_params_result([String.t()]) :: :ok | {:error, String.t()}
  defp format_missing_params_result([]), do: :ok

  @missing_params_error_template """
  Missing required parameters for this tool.

  Required parameters: ~s

  Ensure you're passing the correct parameter names as defined in the tool schema.
  """

  defp format_missing_params_result(missing_params) do
    expected = Enum.join(missing_params, ", ")
    {:error, :io_lib.format(@missing_params_error_template, [expected]) |> IO.iodata_to_binary()}
  end
end
