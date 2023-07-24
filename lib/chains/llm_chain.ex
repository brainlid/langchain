defmodule Langchain.Chains.LLMChain do
  @doc """
  Define an LLMChain
  """
  use Ecto.Schema
  import Ecto.Changeset
  require Logger
  alias Langchain.PromptTemplate
  alias __MODULE__
  alias Langchain.Message
  alias Langchain.Functions.Function

  @primary_key false
  embedded_schema do
    field(:llm, :any, virtual: true)
    field(:prompt, :any, virtual: true)
    field(:stream, :boolean, default: false)
    field(:verbose, :boolean, default: false)
    field(:functions, {:array, :any}, default: [], virtual: true)
    # set and managed privately through functions
    field(:function_map, :map, default: %{}, virtual: true)
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

  @create_fields [:prompt, :llm, :functions, :stream, :verbose]
  @required_fields [:prompt, :llm]

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
    |> validate_prompt()
    |> build_functions_map_from_functions()
  end

  defp validate_llm_is_struct(changeset) do
    case get_change(changeset, :llm) do
      nil -> changeset
      llm when is_struct(llm) -> changeset
      _other -> add_error(changeset, :llm, "LLM must be a struct")
    end
  end

  defp validate_prompt(changeset) do
    # TODO: validate the types? List of String, Message, or PromptTemplate?
    # TODO: Change the type to require a list? Make it a {:array, :any}? Don't accept a single text or prompt template?
    # prompt =
    changeset
  end

  @doc """
  Prepare the prompt for actual submission. It's quite versatile in what it accepts.

  Accepts a Message, String, PromptTemplate or a list of those.
  """
  @doc """
  Prepares the prompt.
  """
  def prepare_prompt(%LLMChain{prompt: nil} = chain), do: ""

  def prepare_prompt(%LLMChain{prompt: prompt} = chain) when not is_list(prompt) do
    prepare_prompt(%LLMChain{chain | prompt: [prompt]})
  end

  def prepare_prompt(%LLMChain{prompt: prompt} = chain) when is_list(prompt) do
    # TODO: Process the prompt
    prepare_prompt(%LLMChain{chain | prompt: [prompt]})
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

  # NOTE: This isn't something I care about currently.
  # @doc """
  # Call the chain combining the inputs to generate a final combined text prompt
  # submitted to the LLM to be evaluated. This submits as a single block of text
  # and receives a block of text.
  # """
  # @spec call_chat(t(), inputs :: map()) :: {:ok, any()} | {:error, String.t()}
  # def call_text(%LLMChain{} = chain, %{} = inputs \\ %{}) do
  #   if chain.verbose, do: IO.inspect(chain.llm, label: "LLM")

  #   #TODO:
  #   # build final combined text prompt
  #   messages = PromptTemplate.to_messages(chain.prompt, inputs)
  #   if chain.verbose, do: IO.inspect(messages, label: "TEXT")

  #   functions = []
  #   if chain.verbose, do: IO.inspect(functions, label: "FUNCTIONS")

  #   # submit to LLM. The "llm" is a struct. Match to get the name of the module
  #   # then execute the `.call` function on that module.
  #   %module{} = chain.llm

  #   #TODO: If includes function executions, does it perform the evaluation?

  #   # handle and output response
  #   case module.call(chain.llm, messages, functions) do
  #     {:ok, %Message{role: :assistant, content: content} = message} ->
  #       if chain.verbose, do: IO.inspect(message, label: "MESSAGE RESPONSE")

  #       {:ok, %{text: content}}
  #   end
  # end

  @doc """
  Call the chain combining the inputs to generate the final prompt submitted to
  the LLM to be evaluated. This formats the request for a ChatLLMChain where
  messages are passed to the API.
  """
  @spec call_chat(t(), inputs :: map()) :: {:ok, Message.t() | [Message.t()]} | {:error, String.t()}
  def call_chat(%LLMChain{} = chain, %{} = inputs \\ %{}) do
    if chain.verbose, do: IO.inspect(chain.llm, label: "LLM")

    if is_list(chain.prompt) do
      # build final prompt, a list of messages
      messages = PromptTemplate.to_messages(chain.prompt, inputs)
      if chain.verbose, do: IO.inspect(messages, label: "MESSAGES")

      functions = chain.functions
      if chain.verbose, do: IO.inspect(functions, label: "FUNCTIONS")

      # submit to LLM. The "llm" is a struct. Match to get the name of the module
      # then execute the `.call` function on that module.
      %module{} = chain.llm

      # TODO: If includes function executions, does it perform the evaluation?

      # handle and output response
      case module.call(chain.llm, messages, functions) do
        {:ok, [%Message{} = message]} ->
          if chain.verbose, do: IO.inspect(message, label: "SINGLE MESSAGE RESPONSE")
          {:ok, message}

        {:ok, [%Message{}, _others] = messages} ->
          if chain.verbose, do: IO.inspect(messages, label: "MULTIPLE MESSAGE RESPONSE")
          # return the list of message responses. Happens when multiple
          # "choices" are returned from LLM by request.
          {:ok, messages}

        {:error, reason} ->
          if chain.verbose, do: IO.inspect(reason, label: "ERROR")
          Logger.error("Error during chat call. Reason: #{inspect(reason)}")
          {:error, reason}
      end
    else
      {:error, "LLM prompt should be a list when using call_chat"}
    end
  end
end
