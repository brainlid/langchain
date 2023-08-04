defmodule Langchain.Chains.LLMChain do
  @doc """
  Define an LLMChain.

  The chain deals with functions, a function map, delta tracking, last_message
  tracking, conversation messages, and verbose logging. This helps by separating
  these responsibilities from the LLM making it easier to support additional
  LLMs because the focus is on communication and formats instead of all the
  extra logic.

  """
  use Ecto.Schema
  import Ecto.Changeset
  require Logger
  alias Langchain.PromptTemplate
  alias __MODULE__
  alias Langchain.Message
  alias Langchain.MessageDelta
  alias Langchain.Function
  alias Langchain.LangchainError

  @primary_key false
  embedded_schema do
    field :llm, :any, virtual: true
    field :verbose, :boolean, default: false
    field :functions, {:array, :any}, default: [], virtual: true
    # set and managed privately through functions
    field :function_map, :map, default: %{}, virtual: true

    # List of `Message` structs for creating the conversation with the LLM.
    field :messages, {:array, :any}, default: [], virtual: true

    # Custom context data made available to functions when executed.
    # Could include information like account ID, user data, etc.
    field :custom_context, :any, virtual: true

    # Track the current merged `%MessageDelta{}` struct received when streamed.
    # Set to `nil` when there is no current delta being tracked. This happens
    # when the final delta is received that completes the message. At that point,
    # the delta is converted to a message and the delta is set to nil.
    field :delta, :any, virtual: true
    # Track the last `%Message{}` received in the chain.
    field :last_message, :any, virtual: true
    # Track if the state of the chain expects a response from the LLM. This
    # happens after sending a user message or when a function_call is received,
    # we've provided a function response and the LLM needs to respond.
    field :needs_response, :boolean, default: false

    # A callback function to execute when messages are added. Don't allow caller
    # to setup in `.new` function. Want to set it from the `.run` function to
    # avoid multiple chain instances (across processes) from both firing
    # callbacks.
    field :callback_fn, :any, virtual: true
  end

  # Note: A Langchain "Tool" is pretty much expressed by an OpenAI Function.
  # TODO: Toolkit is a list of Tools/Functions. Makes it easy to define a set of
  # functions for a specific service.

  # TODO: Ability to receive a message executing a function and execute it. Add
  # a message with the function response.

  # TODO: Create a State structure that supports processing responses, executing
  # functions, and adding more to the state object (like the function result of
  # the execution)

  @type t :: %LLMChain{}

  @create_fields [:llm, :functions, :custom_context, :verbose]
  @required_fields [:llm]

  @doc """
  Start a new LLMChain configuration.
  """
  @spec new(attrs :: map()) :: {:ok, t} | {:error, Ecto.Changeset.t()}
  def new(attrs \\ %{}) do
    %LLMChain{}
    |> cast(attrs, @create_fields)
    |> common_validation()
    |> apply_action(:insert)
  end

  @doc """
  Start a new LLMChain configuration and return it or raise an error if invalid.
  """
  @spec new!(attrs :: map()) :: t() | no_return()
  def new!(attrs \\ %{}) do
    case new(attrs) do
      {:ok, chain} ->
        chain

      {:error, changeset} ->
        raise LangchainError, changeset
    end
  end

  def common_validation(changeset) do
    changeset
    |> validate_required(@required_fields)
    |> validate_llm_is_struct()
    |> build_functions_map_from_functions()
  end

  defp validate_llm_is_struct(changeset) do
    case get_change(changeset, :llm) do
      nil -> changeset
      llm when is_struct(llm) -> changeset
      _other -> add_error(changeset, :llm, "LLM must be a struct")
    end
  end

  # TODO:
  # Figure out the definition for a tool. I believe it should define 1 - many functions.
  # Enum.tools()
  # A Tool is basically a Function. If %Function{}, keep it. If a Toolkit, look for functions and flatten it.

  @doc false
  def build_functions_map_from_functions(changeset) do
    functions = get_field(changeset, :functions, [])

    # get a list of all the functions indexed into a map by name
    fun_map =
      Enum.reduce(functions, %{}, fn f, acc ->
        Map.put(acc, f.name, f)
      end)

    put_change(changeset, :function_map, fun_map)
  end

  @doc """
  Add more functions to an LLMChain.
  """
  @spec add_functions(t(), Function.t() | [Function.t()]) :: t() | no_return()
  def add_functions(%LLMChain{} = chain, %Function{} = function) do
    add_functions(chain, [function])
  end

  def add_functions(%LLMChain{functions: existing} = chain, functions) when is_list(functions) do
    updated = existing ++ functions

    chain
    |> change()
    |> cast(%{functions: updated}, [:functions])
    |> build_functions_map_from_functions()
    |> apply_action!(:update)
  end

  # TODO: Callback needed here if streaming. Setup the callback on the LLM? That's where the stream option is set.

  # TODO: The wrapping up of functions, function map, delta tracking, last_message tracking, messages, and verbose logging is handy to separate from the LLM.
  #       - makes it easier to implement another LLM because it doesn't DO as much.

  @doc """
  Run the chain on the LLM using messages and any registered functions. This
  formats the request for a ChatLLMChain where messages are passed to the API.

  When successful, it returns `{:ok, updated_chain, message_or_messages}`

  ## Options

  - `:while_needs_response` - repeatedly evaluates functions and submits to the
    LLM so long as we still expect to get a response.
  - `:callback_fn` - the callback function to execute as messages are received.

  """
  @spec run(t(), Keyword.t()) :: {:ok, t(), Message.t() | [Message.t()]} | {:error, String.t()}
  def run(chain, opts \\ [])

  def run(%LLMChain{} = chain, opts) do
    # set the callback function on the chain
    chain = %LLMChain{chain | callback_fn: Keyword.get(opts, :callback_fn)}

    if chain.verbose, do: IO.inspect(chain.llm, label: "LLM")

    if chain.verbose, do: IO.inspect(chain.messages, label: "MESSAGES")

    functions = chain.functions
    if chain.verbose, do: IO.inspect(functions, label: "FUNCTIONS")

    if Keyword.get(opts, :while_needs_response, false) do
      run_while_needs_response(chain)
    else
      # run the chain and format the return
      case do_run(chain) do
        {:ok, chain} ->
          {:ok, chain, chain.last_message}

        {:error, _reason} = error ->
          error
      end
    end
  end

  # Repeatedly run the chain while `needs_response` is true. This will execute
  # functions and re-submit the function result to the LLM giving the LLM an
  # opportunity to execute more functions or return a response.
  @spec run_while_needs_response(t()) :: {:ok, t(), Message.t()} | {:error, String.t()}
  defp run_while_needs_response(%LLMChain{needs_response: false} = chain) do
    {:ok, chain, chain.last_message}
  end

  defp run_while_needs_response(%LLMChain{needs_response: true} = chain) do
    chain
    |> execute_function()
    |> do_run()
    |> case do
      {:ok, updated_chain} ->
        run_while_needs_response(updated_chain)

      {:error, reason} ->
        {:error, reason}
    end
  end

  # internal reusable function for running the chain
  @spec do_run(t()) :: {:ok, t()} | {:error, String.t()}
  defp do_run(%LLMChain{} = chain) do
    # submit to LLM. The "llm" is a struct. Match to get the name of the module
    # then execute the `.call` function on that module.
    %module{} = chain.llm

    # handle and output response
    case module.call(chain.llm, chain.messages, chain.functions, chain.callback_fn) do
      {:ok, [%Message{} = message]} ->
        if chain.verbose, do: IO.inspect(message, label: "SINGLE MESSAGE RESPONSE")
        {:ok, add_message(chain, message)}

      {:ok, [%Message{} = message, _others] = messages} ->
        if chain.verbose, do: IO.inspect(messages, label: "MULTIPLE MESSAGE RESPONSE")
        # return the list of message responses. Happens when multiple
        # "choices" are returned from LLM by request.
        {:ok, add_message(chain, message)}

      {:ok, [[%MessageDelta{} | _] | _] = deltas} ->
        if chain.verbose, do: IO.inspect(deltas, label: "DELTA MESSAGE LIST RESPONSE")
        {:ok, apply_deltas(chain, deltas)}

      {:error, reason} ->
        if chain.verbose, do: IO.inspect(reason, label: "ERROR")
        Logger.error("Error during chat call. Reason: #{inspect(reason)}")
        {:error, reason}
    end
  end

  defp do_run(%LLMChain{needs_response: false} = chain) do
    {:ok, chain, chain.last_message}
  end

  @doc """
  Apply a received MessageDelta struct to the chain. The LLMChain tracks the
  current merged MessageDelta state. When the final delta is received that
  completes the message, the LLMChain is updated to clear the `delta` and the
  `last_message` and list of messages are updated.
  """
  @spec apply_delta(t(), MessageDelta.t()) :: t()
  def apply_delta(%LLMChain{delta: nil} = chain, %MessageDelta{} = new_delta) do
    %LLMChain{chain | delta: new_delta}
  end

  def apply_delta(%LLMChain{delta: %MessageDelta{} = delta} = chain, %MessageDelta{} = new_delta) do
    merged = MessageDelta.merge_delta(delta, new_delta)

    # if the merged delta is now complete, updates as a message.
    if merged.complete do
      case MessageDelta.to_message(merged) do
        {:ok, %Message{} = message} ->
          IO.puts "APPLY_DELTA MERGED AND FIRING CALLBACK"
          fire_callback(chain, message)
          add_message(%LLMChain{chain | delta: nil}, message)

        {:error, reason} ->
          # should not have failed, but it did. Log the error and return
          # the chain unmodified.
          Logger.warning("Error applying delta message. Reason: #{inspect(reason)}")
          chain
      end
    else
      # the delta message is not yet complete. Update the delta with the merged
      # result.
      %LLMChain{chain | delta: merged}
    end
  end

  @doc """
  Apply a list of deltas to the chain.
  """
  @spec apply_deltas(t(), list()) :: t()
  def apply_deltas(%LLMChain{} = chain, deltas) when is_list(deltas) do
    deltas
    |> List.flatten()
    |> Enum.reduce(chain, fn d, acc -> apply_delta(acc, d) end)
  end

  @doc """
  Add a received Message struct to the chain. The LLMChain tracks the
  `last_message` received and the complete list of messages exchanged. Depending
  on the message role, the chain may be in a pending or incomplete state where
  a response from the LLM is anticipated.
  """
  @spec add_message(t(), Message.t()) :: t()
  def add_message(%LLMChain{} = chain, %Message{} = new_message) do
    needs_response =
      case new_message do
        %Message{role: role} when role in [:user, :function_call, :function] ->
          true

        %Message{role: role} when role in [:system, :assistant] ->
          false
      end

    %LLMChain{
      chain
      | messages: chain.messages ++ [new_message],
        last_message: new_message,
        needs_response: needs_response
    }
  end

  @doc """
  Add a set of Message structs to the chain. This enables quickly building a chain
  for submitting to an LLM.
  """
  @spec add_messages(t(), [Message.t()]) :: t()
  def add_messages(%LLMChain{} = chain, messages) do
    Enum.reduce(messages, chain, fn msg, acc ->
      add_message(acc, msg)
    end)
  end

  @doc """
  Apply a set of PromptTemplates to the chain. The list of templates can also
  include Messages with no templates. Provide the inputs to apply to the
  templates for rendering as a message. The prepared messages are applied to the
  chain.
  """
  @spec apply_prompt_templates(t(), [Message.t() | PromptTemplate.t()], %{atom() => any()}) ::
          t() | no_return()
  def apply_prompt_templates(%LLMChain{} = chain, templates, %{} = inputs) do
    messages = PromptTemplate.to_messages(templates, inputs)
    add_messages(chain, messages)
  end

  @doc """
  Convenience function for setting the prompt text for the LLMChain using
  prepared text.
  """
  @spec quick_prompt(t(), String.t()) :: t()
  def quick_prompt(%LLMChain{} = chain, text) do
    messages = [
      Message.new_system!(),
      Message.new_user!(text)
    ]

    add_messages(chain, messages)
  end

  @doc """
  If the `last_message` is a `%Message{role: :function_call}`, then the linked
  function is executed. If there is no `last_message` or the `last_message` is
  not a `:function_call`, the LLMChain is returned with no action performed.
  This makes it safe to call any time.

  The `context` is additional data that will be passed to the executed function.
  The value given here will override any `custom_context` set on the LLMChain.
  If not set, the global `custom_context` is used.

  https://platform.openai.com/docs/guides/gpt/function-calling
  """
  @spec execute_function(t(), context :: any()) :: t()
  def execute_function(chain, context \\ nil)
  def execute_function(%LLMChain{last_message: nil} = chain, _context), do: chain

  def execute_function(
        %LLMChain{last_message: %Message{role: :function_call} = message} = chain,
        context
      ) do
    # context to use
    use_context = context || chain.custom_context

    # find and execute the linked function
    case chain.function_map[message.function_name] do
      %Function{} = function ->
        if chain.verbose, do: IO.inspect(function.name, label: "EXECUTING FUNCTION")

        # execute the function
        result = Function.execute(function, message.arguments, use_context)
        if chain.verbose, do: IO.inspect(result, label: "FUNCTION RESULT")

        # add the :function response to the chain
        function_result = Message.new_function!(function.name, result)
        # fire the callback as this is newly generated message
        fire_callback(chain, function_result)
        LLMChain.add_message(chain, function_result)

      nil ->
        Logger.warning(
          "Received function_call for missing function #{inspect(message.function_name)}"
        )

        chain
    end
  end

  # Either not a function_call or an incomplete function_call, do nothing.
  def execute_function(%LLMChain{last_message: %Message{}} = chain, _context), do: chain

  # Fire the callback if set.
  defp fire_callback(%LLMChain{callback_fn: nil}, _data), do: :ok

  # OPTIONAL: Execute callback function
  defp fire_callback(%LLMChain{callback_fn: callback_fn}, data) when is_function(callback_fn) do
    case data do
      value when is_list(value) ->
        value
        |> List.flatten()
        |> Enum.each(fn item -> callback_fn.(item) end)

        :ok

      # not a list, pass the item as-is
      item ->
        callback_fn.(item)
        :ok
    end
  end
end
