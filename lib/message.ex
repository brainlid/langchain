defmodule LangChain.Message do
  @moduledoc """
  Models a complete `Message` for a chat LLM.

  ## Roles

  - `:system` - a system message. Typically just one and it occurs first as a
    primer for how the LLM should behave.

  - `:user` - The user or application responses. Typically represents the
    "human" element of the exchange.

  - `:assistant` - Responses coming back from the LLM.

  - `:arguments` - The `arguments` can be set as a map where each key is an
    "argument". If set as a String, it is expected to be a JSON formatted string
    and will be parsed to a map. If there is an error parsing the arguments to
    JSON, it is considered an error.

    An empty map `%{}` means no arguments are passed.

  - `:function` - A message for returning the result of executing a
    `function_call`.

  ## Functions

  A `function_call` comes from the `:assistant` role. The `function_name`
  identifies the named function to execute.

  Create a message of role `:function` to provide the function response.

  """
  use Ecto.Schema
  import Ecto.Changeset
  require Logger
  alias __MODULE__
  alias LangChain.LangChainError

  @primary_key false
  embedded_schema do
    field :content, :string
    field :index, :integer
    field :status, Ecto.Enum, values: [:complete, :cancelled, :length], default: :complete

    field :role, Ecto.Enum,
      values: [:system, :user, :assistant, :function],
      default: :user

    field :function_name, :string
    field :arguments, :any, virtual: true
  end

  @type t :: %Message{}
  @type status :: :complete | :cancelled | :length

  @update_fields [:role, :content, :status, :function_name, :arguments, :index]
  @create_fields @update_fields
  @required_fields [:role]

  @doc """
  Build a new message and return an `:ok`/`:error` tuple with the result.
  """
  @spec new(attrs :: map()) :: {:ok, t()} | {:error, Ecto.Changeset.t()}
  def new(attrs \\ %{}) do
    %Message{}
    |> cast(attrs, @create_fields)
    |> parse_arguments()
    |> common_validations()
    |> apply_action(:insert)
  end

  @doc """
  Build a new message and return it or raise an error if invalid.
  """
  @spec new!(attrs :: map()) :: t() | no_return()
  def new!(attrs \\ %{}) do
    case new(attrs) do
      {:ok, message} ->
        message

      {:error, changeset} ->
        raise LangChainError, changeset
    end
  end

  @doc false
  def changeset(message, attrs) do
    message
    |> cast(attrs, @update_fields)
    |> common_validations()
  end

  defp changeset_is_function?(changeset) do
    get_field(changeset, :role) == :assistant and
      is_binary(get_field(changeset, :function_name)) and
      get_field(changeset, :status) == :complete
  end

  defp parse_arguments(changeset) do
    args = get_field(changeset, :arguments)
    is_function = changeset_is_function?(changeset)

    # only
    cond do
      is_function && is_binary(args) ->
        # decode the arguments
        case Jason.decode(args) do
          {:ok, parsed} when is_map(parsed) ->
            put_change(changeset, :arguments, parsed)

          {:ok, parsed} ->
            Logger.warning(
              "Parsed unexpected function argument format. Expected a map but received: #{inspect(parsed)}"
            )

            add_error(changeset, :arguments, "unexpected JSON arguments format")

          {:error, error} ->
            Logger.warning("Received invalid argument JSON data. Error: #{inspect(error)}")
            add_error(changeset, :arguments, "invalid JSON function arguments")
        end

      true ->
        changeset
    end
  end

  defp common_validations(changeset) do
    changeset
    |> validate_required(@required_fields)
    |> validate_content()
    |> validate_function_name_for_role()
    |> validate_function_name_when_args()
  end

  # validate that a "user" and "system" message has content. Allow an
  # "assistant" message to be created where we don't have content yet because it
  # can be streamed in through deltas from an LLM and not yet receive the
  # content.
  defp validate_content(changeset) do
    case fetch_field!(changeset, :role) do
      role when role in [:system, :user] ->
        validate_required(changeset, [:content])

      _other ->
        changeset
    end
  end

  # validate that "function_name" can only be set if the role is "function_call"
  # for requesting execution or "function" for the returning a function result.
  #
  # The function_name is required for those message types.
  defp validate_function_name_for_role(changeset) do
    case fetch_field!(changeset, :role) do
      role when role in [:function] ->
        validate_required(changeset, [:function_name])

      role when role in [:system, :user] ->
        if get_field(changeset, :function_name) == nil do
          changeset
        else
          add_error(changeset, :function_name, "can't be set with role #{inspect(role)}")
        end

      _other ->
        changeset
    end
  end

  # validate that "function_name" is required if arguments are set.
  defp validate_function_name_when_args(changeset) do
    function_name = get_field(changeset, :function_name)

    if get_field(changeset, :arguments) != nil && is_nil(function_name) do
      add_error(changeset, :function_name, "is required when arguments are given")
    else
      changeset
    end
  end

  @doc """
  Create a new system message which can prime the AI/Assistant for how to
  respond.
  """
  @spec new_system(content :: String.t()) :: {:ok, t()} | {:error, Ecto.Changeset.t()}
  def new_system(content \\ "You are a helpful assistant.") do
    new(%{role: :system, content: content, status: :complete})
  end

  @doc """
  Create a new system message which can prime the AI/Assistant for how to
  respond.
  """
  @spec new_system!(content :: String.t()) :: t() | no_return()
  def new_system!(content \\ "You are a helpful assistant.") do
    case new_system(content) do
      {:ok, msg} ->
        msg

      {:error, %Ecto.Changeset{} = changeset} ->
        raise LangChainError, changeset
    end
  end

  @doc """
  Create a new user message which represents a human message or a message from
  the application.
  """
  @spec new_user(content :: String.t()) :: {:ok, t()} | {:error, Ecto.Changeset.t()}
  def new_user(content) do
    new(%{role: :user, content: content, status: :complete})
  end

  @doc """
  Create a new user message which represents a human message or a message from
  the application.
  """
  @spec new_user!(content :: String.t()) :: t() | no_return()
  def new_user!(content) do
    case new_user(content) do
      {:ok, msg} ->
        msg

      {:error, %Ecto.Changeset{} = changeset} ->
        raise LangChainError, changeset
    end
  end

  @doc """
  Create a new assistant message which represents a response from the AI or LLM.
  """
  @spec new_assistant(content :: String.t(), status()) ::
          {:ok, t()} | {:error, Ecto.Changeset.t()}
  def new_assistant(content, status \\ :complete) do
    new(%{role: :assistant, content: content, status: status})
  end

  @doc """
  Create a new assistant message which represents a response from the AI or LLM.
  """
  @spec new_assistant!(content :: String.t(), status()) :: t() | no_return()
  def new_assistant!(content, status \\ :complete) do
    case new_assistant(content, status) do
      {:ok, msg} ->
        msg

      {:error, %Ecto.Changeset{} = changeset} ->
        raise LangChainError, changeset
    end
  end

  @doc """
  Create a new function_call message to represent the request for a function to
  be executed.
  """
  @spec new_function_call(name :: String.t(), raw_args :: String.t()) ::
          {:ok, t()} | {:error, Ecto.Changeset.t()}
  def new_function_call(name, raw_args) do
    case Jason.decode(raw_args) do
      {:ok, parsed} ->
        new(%{role: :assistant, function_name: name, arguments: parsed, status: :complete})

      {:error, %Jason.DecodeError{data: reason}} ->
        {:error,
         %Message{role: :assistant, function_name: name}
         |> change()
         |> add_error(:arguments, "Failed to parse arguments: #{inspect(reason)}")}
    end
  end

  @doc """
  Create a new function_call message to represent the request for a function to
  be executed.
  """
  @spec new_function_call!(name :: String.t(), raw_args :: String.t()) :: t() | no_return()
  def new_function_call!(name, raw_args) do
    case new_function_call(name, raw_args) do
      {:ok, msg} ->
        msg

      {:error, %Ecto.Changeset{} = changeset} ->
        raise LangChainError, changeset
    end
  end

  @doc """
  Create a new function message to represent the result of an executed
  function.
  """
  @spec new_function(name :: String.t(), result :: any()) ::
          {:ok, t()} | {:error, Ecto.Changeset.t()}
  def new_function(name, result) do
    new(%{role: :function, function_name: name, content: serialize_result(result)})
  end

  @spec serialize_result(result :: any()) :: String.t()
  defp serialize_result(result) when is_binary(result), do: result
  defp serialize_result(result) when is_map(result), do: Jason.encode!(result)

  @doc """
  Create a new function message to represent the result of an executed
  function.
  """
  @spec new_function!(name :: String.t(), result :: any()) :: t() | no_return()
  def new_function!(name, result) do
    case new_function(name, result) do
      {:ok, msg} ->
        msg

      {:error, %Ecto.Changeset{} = changeset} ->
        raise LangChainError, changeset
    end
  end

  @doc """
  Return if a Message is a function_call.
  """
  def is_function_call?(%Message{role: :assistant, status: :complete, function_name: fun_name})
      when is_binary(fun_name),
      do: true

  def is_function_call?(%Message{}), do: false
end

defimpl LangChain.ForOpenAIApi, for: LangChain.Message do
  alias LangChain.Message

  def for_api(%Message{role: :assistant, function_name: fun_name} = fun)
      when is_binary(fun_name) do
    %{
      "role" => :assistant,
      "function_call" => %{
        "arguments" => Jason.encode!(fun.arguments),
        "name" => fun.function_name
      },
      "content" => fun.content
    }
  end

  def for_api(%Message{role: :function} = fun) do
    %{
      "role" => :function,
      "name" => fun.function_name,
      "content" => fun.content
    }
  end

  def for_api(%Message{} = fun) do
    %{
      "role" => fun.role,
      "content" => fun.content
    }
  end
end
