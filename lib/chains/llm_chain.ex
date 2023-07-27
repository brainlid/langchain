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
  alias Langchain.Functions.Function

  @primary_key false
  embedded_schema do
    field :llm, :any, virtual: true
    field :stream, :boolean, default: false
    field :verbose, :boolean, default: false
    field :functions, {:array, :any}, default: [], virtual: true
    # set and managed privately through functions
    field :function_map, :map, default: %{}, virtual: true

    # List of `Message` structs for creating the conversation with the LLM.
    field :messages, {:array, :any}, default: [], virtual: true

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
  end

  # Note: A Langchain "Tool" is pretty much expressed by an OpenAI Function.
  # TODO: Toolkit is a list of Tools/Functions. Makes it easy to define a set of
  # functions for a specific service.

  # TODO: Ability to receive a message executing a function and execute it. Add
  # a message with the function response.

  # TODO: Create a State structure that supports processing responses, executing
  # functions, and adding more to the state object (like the function result of
  # the execution)

  # TODO: function that reduces all messages or prompts to single text string. USAGE with LLM and not ChatLLM.

  @type t :: %LLMChain{}

  @create_fields [:llm, :messages, :functions, :stream, :verbose]
  @required_fields [:llm]

  @spec new(attrs :: map()) :: {:ok, t} | {:error, Ecto.Changeset.t()}
  def new(attrs \\ %{}) do
    %LLMChain{}
    |> cast(attrs, @create_fields)
    |> common_validation()
    |> apply_action(:insert)
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

  def build_functions_map_from_functions(changeset) do
    functions = get_field(changeset, :functions, [])

    # get a list of all the functions from all the functions
    # funs = Enum.flat_map(functions, & &1.functions)

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
  """
  @spec run(t()) :: {:ok, t(), Message.t() | [Message.t()]} | {:error, String.t()}
  def run(%LLMChain{} = chain) do
    if chain.verbose, do: IO.inspect(chain.llm, label: "LLM")

    if chain.verbose, do: IO.inspect(chain.messages, label: "MESSAGES")

    functions = chain.functions
    if chain.verbose, do: IO.inspect(functions, label: "FUNCTIONS")

    # submit to LLM. The "llm" is a struct. Match to get the name of the module
    # then execute the `.call` function on that module.
    %module{} = chain.llm

    # handle and output response
    case module.call(chain.llm, chain.messages, functions) do
      {:ok, [%Message{} = message]} ->
        if chain.verbose, do: IO.inspect(message, label: "SINGLE MESSAGE RESPONSE")
        {:ok, apply_message(chain, message), message}

      {:ok, [%Message{} = message, _others] = messages} ->
        if chain.verbose, do: IO.inspect(messages, label: "MULTIPLE MESSAGE RESPONSE")
        # return the list of message responses. Happens when multiple
        # "choices" are returned from LLM by request.
        {:ok, apply_message(chain, message), messages}

      {:error, reason} ->
        if chain.verbose, do: IO.inspect(reason, label: "ERROR")
        Logger.error("Error during chat call. Reason: #{inspect(reason)}")
        {:error, reason}
    end
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
          apply_message(%LLMChain{chain | delta: nil}, message)

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
  Apply a received Message struct to the chain. The LLMChain tracks the
  `last_message` received and the complete list of messages exchanged. Depending
  on the message role, the chain may be in a pending or incomplete state where
  an response from the LLM is anticipated.
  """
  @spec apply_message(t(), Message.t()) :: t()
  def apply_message(%LLMChain{} = chain, %Message{} = new_message) do
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
  Apply a set of Message structs to the chain. This enables quickly building a chain
  for submitting to an LLM.
  """
  @spec apply_messages(t(), [Message.t()]) :: t()
  def apply_messages(%LLMChain{} = chain, messages) do
    Enum.reduce(messages, chain, fn msg, acc ->
      apply_message(acc, msg)
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
    apply_messages(chain, messages)
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

    apply_messages(chain, messages)
  end

  @doc """
  If the `last_message` is a `%Message{role: :function_call}`, then the linked
  function is executed. If there is no `last_message` or the `last_message` is
  not a `:function_call`, the LLMChain is returned with no action performed.
  This makes it safe to call any time.
  """
  @spec execute_function(t()) :: t()
  def execute_function(%LLMChain{last_message: nil} = chain), do: chain

  def execute_function(%LLMChain{last_message: %Message{role: :function_call} = message} = chain) do
    # TODO: execute the linked function

    # TODO: How to handle this when the function function to execute will be slow??? Want it to be async!
    #   - function has an `:async` flag? Execute helper?
    #   - how to handle receiving the function result?
  end

  # Either not a function_call or an incomplete function_call, do nothing.
  def execute_function(%LLMChain{last_message: %Message{}} = chain), do: chain
end
