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

  Unless a `:parse_args` parser is supplied, both parameter declarations,
  `parameters: [%FunctionParam{}]` and `parameters_schema:`, get a **top-level
  required-key presence check** at execute time. Nothing else is enforced:
  types, enums, formats, and nested object shapes are all passed through to the
  tool as the LLM sent them. Provider "strict mode" closes that remaining gap
  somewhat, but is best-effort and varies by provider.

  When the check fails, the returned error names the required parameters, the
  missing ones, any unrecognized argument names that were sent, and the full
  list of accepted parameters. This matters because a model that renames an
  argument (sending `file_path` where the tool declared `path`) will otherwise
  read a raw exception as a transient fault and retry the same call verbatim,
  with each retry teaching itself the wrong calling convention.

  ### `:parse_args` owns argument validation outright

  The optional `:parse_args` callback replaces the built-in check rather than
  layering on top of it. **When a parser is supplied, LangChain performs no
  argument validation of its own**: the required-key check is skipped, and an
  exception raised by the tool body is reported with its original formatting
  instead of being reinterpreted as an argument-name problem.

  This keeps one voice and one round trip. A parser such as `Zoi` reports
  missing keys *and* type violations together in a single message, formatted
  the way you wrote it, rather than having LangChain answer the missing-key
  case in a different format and hide the rest until the next turn. It also
  avoids second-guessing a parser that legitimately coerces keys or injects
  defaults, since the arguments reaching the tool body no longer have to match
  the declared schema.

  The trade-off is that tools using a parser do not get LangChain's
  "unrecognized parameter / did you mean" diagnostic. Parsers that want it can
  build the same message from `required_param_names/1` and
  `accepted_param_names/1`.

  The parser runs **before** the user-supplied `function` and receives the raw,
  string-keyed arguments map from the LLM. It returns one of three shapes:

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
  alias LangChain.FunctionParam
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

  @doc """
  Return the names of the function's required top-level parameters.

  Works for both declaration styles. For `parameters:` it reads the `required`
  flag from each `LangChain.FunctionParam`. For `parameters_schema:` it reads
  the schema's `required` list.

  Returns `[]` when the function declares no parameters or declares none as
  required.

      iex> alias LangChain.{Function, FunctionParam}
      iex> fun = Function.new!(%{
      ...>   name: "demo",
      ...>   function: fn _args, _context -> {:ok, "ok"} end,
      ...>   parameters: [
      ...>     FunctionParam.new!(%{name: "path", type: :string, required: true}),
      ...>     FunctionParam.new!(%{name: "limit", type: :integer})
      ...>   ]
      ...> })
      iex> Function.required_param_names(fun)
      ["path"]
  """
  @spec required_param_names(t()) :: [String.t()]
  def required_param_names(%Function{parameters: params})
      when is_list(params) and params != [] do
    FunctionParam.required_properties(params)
  end

  def required_param_names(%Function{parameters_schema: schema}) when is_map(schema) do
    schema
    |> schema_get(:required)
    |> normalize_names()
  end

  def required_param_names(%Function{}), do: []

  @doc """
  Return the names of every top-level parameter the function accepts.

  Works for both declaration styles. Returns `[]` when the function's
  declaration doesn't enumerate its parameters, which happens for a
  `parameters_schema:` without a `properties` map and for a function declaring
  no parameters at all. Callers should treat `[]` as "unknown", not as "accepts
  nothing", since an empty list carries no information about which argument
  names are valid.

      iex> alias LangChain.Function
      iex> fun = Function.new!(%{
      ...>   name: "demo",
      ...>   function: fn _args, _context -> {:ok, "ok"} end,
      ...>   parameters_schema: %{
      ...>     type: "object",
      ...>     properties: %{path: %{type: "string"}, limit: %{type: "integer"}},
      ...>     required: ["path"]
      ...>   }
      ...> })
      iex> fun |> Function.accepted_param_names() |> Enum.sort()
      ["limit", "path"]
  """
  @spec accepted_param_names(t()) :: [String.t()]
  def accepted_param_names(%Function{parameters: params})
      when is_list(params) and params != [] do
    Enum.map(params, & &1.name)
  end

  def accepted_param_names(%Function{parameters_schema: schema}) when is_map(schema) do
    case schema_get(schema, :properties) do
      properties when is_map(properties) ->
        properties |> Map.keys() |> normalize_names()

      _not_a_map ->
        []
    end
  end

  def accepted_param_names(%Function{}), do: []

  # Schemas in the wild are written with atom keys as often as string keys.
  # `LangChain.Tools.Calculator` uses atom keys for both the schema keys and the
  # property names, while other schemas use strings throughout. Look for the
  # atom first, then fall back to its string form.
  @spec schema_get(map(), atom()) :: any()
  defp schema_get(schema, key) when is_map(schema) and is_atom(key) do
    case Map.fetch(schema, key) do
      {:ok, value} -> value
      :error -> Map.get(schema, Atom.to_string(key))
    end
  end

  # Coerce a list of parameter names to strings, dropping anything that isn't
  # name-shaped. Only converts atoms to strings, never the reverse, so nothing
  # here can create atoms from LLM-supplied data.
  @spec normalize_names(any()) :: [String.t()]
  defp normalize_names(names) when is_list(names) do
    Enum.flat_map(names, fn
      name when is_binary(name) ->
        [name]

      name when is_atom(name) and not is_nil(name) and not is_boolean(name) ->
        [Atom.to_string(name)]

      _other ->
        []
    end)
  end

  defp normalize_names(_other), do: []

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

      case argument_error_message(err, function, fun, arguments) do
        nil -> {:error, "ERROR: #{LangChainError.format_exception(err, __STACKTRACE__, :short)}"}
        message -> {:error, message}
      end
  end

  # The required-parameter check can't reach a tool that reads an *optional*
  # argument with `Map.fetch!/2` or pattern-matches one in its head, so those
  # still blow up here. Left alone, the model sees a stack frame from the tool's
  # source and reads it as a transient system fault rather than as "you used the
  # wrong argument name", and retries the identical call. Recognize the two
  # exception shapes that mean exactly that and say so plainly instead.
  #
  # Returns nil for every other exception, leaving the existing formatting in
  # place.
  @spec argument_error_message(Exception.t(), t(), function(), arguments()) :: String.t() | nil
  defp argument_error_message(_err, %Function{parse_args: parser}, _fun, _arguments)
       when is_function(parser, 1) do
    # The parser owns the argument contract. Two reasons to stay out of the way
    # here: `arguments` at this point is the *parsed* map, which may have
    # coerced keys or injected defaults that no longer match the declared
    # schema, so calling a key "unrecognized" would be wrong. And if a parser
    # approved the arguments and the body still can't read them, that is a bug
    # in the tool rather than a mistake by the model -- a stack frame is the
    # right signal for it.
    nil
  end

  defp argument_error_message(%KeyError{term: term, key: key}, function, _fun, arguments)
       when is_map(term) do
    # Only when the KeyError is about the arguments map itself, not about some
    # unrelated map the tool touched. `arguments` here is post-`:parse_args`,
    # which is the same map handed to the tool body.
    if term == arguments do
      build_argument_error_message(function, arguments, key)
    else
      nil
    end
  end

  defp argument_error_message(%FunctionClauseError{} = err, function, fun, arguments) do
    # A tool declared as `def execute(%{"path" => path}, _context)` raises here
    # when the argument name doesn't match. Confirm the clause error came from
    # the declared tool function itself rather than from something it called.
    if raised_by_tool_function?(err, fun) do
      build_argument_error_message(function, arguments, nil)
    else
      nil
    end
  end

  defp argument_error_message(_err, _function, _fun, _arguments), do: nil

  @spec raised_by_tool_function?(FunctionClauseError.t(), function()) :: boolean()
  defp raised_by_tool_function?(%FunctionClauseError{} = err, fun) when is_function(fun) do
    info = Elixir.Function.info(fun)

    # Match on identity rather than on `info[:type]`. A capture written inside
    # the module that defines it reports `type: :local` even when the target is
    # public, so `:external` would reject most real tools. An anonymous
    # function reports a mangled name like :"-caps/0-fun-0-", which still
    # matches when that same anonymous function is the one that failed to match
    # its argument, and that is precisely the case worth reporting. Anything
    # the tool merely *called* has a different module/name/arity and falls
    # through.
    err.module == info[:module] and
      err.function == info[:name] and
      err.arity == info[:arity]
  end

  defp raised_by_tool_function?(_err, _fun), do: false

  @spec build_argument_error_message(t(), arguments(), any()) :: String.t() | nil
  defp build_argument_error_message(%Function{} = function, arguments, missing_key) do
    args = if is_map(arguments), do: arguments, else: %{}

    case accepted_param_names(function) do
      # Without a declared parameter list there is nothing useful to say. Fall
      # back rather than assert the argument names were wrong when we have no
      # idea which names are right.
      [] ->
        nil

      accepted ->
        unrecognized = unrecognized_arg_names(args, accepted)

        lead =
          cond do
            unrecognized != [] ->
              "The tool was called with an argument name it does not accept."

            is_binary(missing_key) or is_atom(missing_key) ->
              "The tool needs the #{inspect(missing_key)} argument, which was not provided."

            true ->
              "The tool could not read the arguments it was given."
          end

        details =
          [
            {"Accepted parameters", accepted},
            {"Unrecognized parameters", describe_unrecognized(unrecognized, args, accepted)},
            {"Received", args |> Map.keys() |> normalize_names() |> Enum.sort()}
          ]
          |> Enum.reject(fn {_label, values} -> values == [] end)
          |> Enum.map_join(" ", fn {label, values} -> "#{label}: #{Enum.join(values, ", ")}." end)

        # No stack frame. The tool's source location is noise to the model and
        # leaks the calling project's file paths into the conversation.
        "ERROR: #{lead} #{details}"
    end
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

  # Validates that all required top-level parameters are present in the
  # arguments. Applies to both the `parameters:` and `parameters_schema:`
  # declaration styles.
  #
  # Unrecognized argument names are never grounds for rejection on their own;
  # extra arguments are passed through to the tool as they always have been.
  # They are only *named* in the error when a required parameter is already
  # missing, which is exactly the situation a renamed argument produces.
  @spec validate_required_params(t(), arguments()) :: :ok | {:error, String.t()}
  defp validate_required_params(%Function{parse_args: parser}, _arguments)
       when is_function(parser, 1) do
    # A supplied parser owns argument validation outright. Answering the
    # missing-key case here would split one tool's errors across two formats
    # and would keep the parser from reporting missing keys and type
    # violations together in a single round trip.
    :ok
  end

  defp validate_required_params(%Function{} = function, arguments) do
    # An LLM can hand back `nil` instead of an empty map for a no-argument
    # call. Treat that as `%{}` rather than letting `Map.has_key?/2` raise a
    # BadMapError that surfaces to the model as an internal error.
    args = if is_map(arguments), do: arguments, else: %{}

    case function |> required_param_names() |> Enum.reject(&Map.has_key?(args, &1)) do
      [] -> :ok
      missing -> {:error, format_missing_params_error(function, args, missing)}
    end
  end

  @missing_params_error_intro "Missing required parameters for this tool."
  @missing_params_error_outro "Ensure you're passing the correct parameter names as defined in the tool schema."

  # Builds the model-facing message. The model reads this string as the tool's
  # response, so it has to say what was wrong *and* what to send instead.
  @spec format_missing_params_error(t(), arguments(), [String.t()]) :: String.t()
  defp format_missing_params_error(%Function{} = function, arguments, missing) do
    required = required_param_names(function)
    accepted = accepted_param_names(function)
    unrecognized = unrecognized_arg_names(arguments, accepted)

    details =
      [
        {"Required parameters", required},
        {"Missing parameters", missing},
        {"Unrecognized parameters", describe_unrecognized(unrecognized, arguments, accepted)},
        {"Accepted parameters", accepted}
      ]
      |> Enum.reject(fn {_label, values} -> values == [] end)
      |> Enum.map_join("\n", fn {label, values} -> "#{label}: #{Enum.join(values, ", ")}" end)

    Enum.join([@missing_params_error_intro, details, @missing_params_error_outro], "\n\n")
  end

  # Argument names the function doesn't declare. Returns `[]` when the accepted
  # list is unknown, since we can't call a name unrecognized without knowing
  # which names are recognized.
  @spec unrecognized_arg_names(arguments(), [String.t()]) :: [String.t()]
  defp unrecognized_arg_names(_arguments, []), do: []

  defp unrecognized_arg_names(arguments, accepted) do
    arguments
    |> Map.keys()
    |> normalize_names()
    |> Enum.reject(&(&1 in accepted))
    |> Enum.sort()
  end

  # Annotates each unrecognized name with the accepted name it most likely
  # meant, when one is close enough.
  @spec describe_unrecognized([String.t()], arguments(), [String.t()]) :: [String.t()]
  defp describe_unrecognized(unrecognized, arguments, accepted) do
    # Only suggest names the caller didn't already supply. A parameter that was
    # correctly provided is never what a misnamed argument was reaching for.
    candidates = Enum.reject(accepted, &Map.has_key?(arguments, &1))

    Enum.map(unrecognized, fn name ->
      case suggest_param_name(name, candidates) do
        nil -> name
        suggestion -> "#{name} (did you mean \"#{suggestion}\"?)"
      end
    end)
  end

  # Jaro alone is not enough here. `String.jaro_distance("file_path", "path")`
  # is 0.0: the matching window is `max(9, 4) / 2 - 1 = 3` and every character
  # of "path" sits 5 positions away inside "file_path", so nothing matches.
  # Renames of that shape (a qualifier added to or dropped from the front) are
  # the common case, so containment is checked first and scored by how much of
  # the longer name the shorter one accounts for.
  @jaro_threshold 0.7
  @min_containment_length 3

  @spec suggest_param_name(String.t(), [String.t()]) :: String.t() | nil
  defp suggest_param_name(unknown, candidates) do
    candidates
    |> Enum.map(&{&1, name_similarity(unknown, &1)})
    |> Enum.reject(fn {_candidate, score} -> score == 0.0 end)
    |> case do
      [] -> nil
      scored -> scored |> Enum.max_by(fn {_candidate, score} -> score end) |> elem(0)
    end
  end

  @spec name_similarity(String.t(), String.t()) :: float()
  defp name_similarity(unknown, candidate) do
    left = String.downcase(unknown)
    right = String.downcase(candidate)

    cond do
      left == right ->
        1.0

      contains_name?(left, right) ->
        {shorter, longer} =
          if String.length(left) <= String.length(right), do: {left, right}, else: {right, left}

        0.7 + 0.3 * (String.length(shorter) / String.length(longer))

      true ->
        case String.jaro_distance(left, right) do
          score when score > @jaro_threshold -> score
          _too_far -> 0.0
        end
    end
  end

  # Containment only counts when the shorter name is substantial. Without the
  # floor, a single-character parameter would be a substring of nearly every
  # argument name and would win every suggestion.
  @spec contains_name?(String.t(), String.t()) :: boolean()
  defp contains_name?(left, right) do
    min(String.length(left), String.length(right)) >= @min_containment_length and
      (String.contains?(left, right) or String.contains?(right, left))
  end
end
