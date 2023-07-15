defmodule Langchain.Message do
  @moduledoc """
  Models a `Message` for chat LLM.

  ## Roles

  - `:system` - a system message. Typically just one and it tends to occur first
    as a primer for how the LLM should behave.
  - `:user` - The user or application responses. Typically represents the
    "human" element of the exchange.
  - `:assistant` - Responses coming back from the LLM.
  - `:function_call` - A message from the LLM expressing the intent to execute a
    function that was previously declared available to it.

    The `arguments` will be the parsed JSON values passed to the function.

  - `:function` - A message for returning the results of executing a
    `function_call` if there is a response to give.

  """
  use Ecto.Schema
  import Ecto.Changeset
  alias __MODULE__
  alias Langchain.LangchainError

  @primary_key false
  embedded_schema do
    field(:content, :string)
    field(:index, :integer)

    field(:role, Ecto.Enum,
      values: [:system, :user, :assistant, :function, :function_call],
      default: :user
    )

    field(:function_name, :string)
    field :arguments, :any, virtual: true
  end

  @type t :: %Message{}

  @create_fields [:role, :content, :function_name, :arguments]
  @required_fields [:role]

  @spec new(attrs :: map()) :: {:ok, t()} | {:error, Ecto.Changeset.t()}
  def new(attrs \\ %{}) do
    %Message{}
    |> cast(attrs, @create_fields)
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
        raise LangchainError, changeset
    end
  end

  defp common_validations(changeset) do
    changeset
    |> validate_required(@required_fields)
    |> validate_content()
    |> validate_function_name()
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
  defp validate_function_name(changeset) do
    case fetch_field!(changeset, :role) do
      role when role in [:function_call, :function] ->
        validate_required(changeset, [:function_name])

      role when role in [:system, :user, :assistant] ->
        if get_field(changeset, :function_name) == nil do
          changeset
        else
          add_error(changeset, :function_name, "can't be set with role #{inspect(role)}")
        end

      _other ->
        changeset
    end
  end

  @doc """
  Create a new system message which can prime the AI/Assistant for how to
  respond.
  """
  @spec new_system(content :: String.t()) :: {:ok, t()} | {:error, Ecto.Changeset.t()}
  def new_system(content \\ "You are a helpful assistant.") do
    new(%{role: :system, content: content})
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
        raise LangchainError, changeset
    end
  end

  @doc """
  Create a new user message which represents a human message or a message from
  the application.
  """
  @spec new_user(content :: String.t()) :: {:ok, t()} | {:error, Ecto.Changeset.t()}
  def new_user(content) do
    new(%{role: :user, content: content})
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
        raise LangchainError, changeset
    end
  end

  @doc """
  Create a new assistant message which represents a response from the AI or LLM.
  """
  @spec new_assistant(content :: String.t()) :: {:ok, t()} | {:error, Ecto.Changeset.t()}
  def new_assistant(content) do
    new(%{role: :assistant, content: content})
  end

  @doc """
  Create a new assistant message which represents a response from the AI or LLM.
  """
  @spec new_assistant!(content :: String.t()) :: t() | no_return()
  def new_assistant!(content) do
    case new_assistant(content) do
      {:ok, msg} ->
        msg

      {:error, %Ecto.Changeset{} = changeset} ->
        raise LangchainError, changeset
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
        new(%{role: :function_call, function_name: name, arguments: parsed})

      {:error, %Jason.DecodeError{data: reason}} ->
        {:error,
         %Message{role: :function_call, function_name: name}
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
        raise LangchainError, changeset
    end
  end

  @doc """
  Create a new function message to represent the result of an executed
  function.
  """
  @spec new_function(name :: String.t(), result :: any()) :: {:ok, t()} | {:error, Ecto.Changeset.t()}
  def new_function(name, result) do
    new(%{role: :function, function_name: name, content: result})
  end

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
        raise LangchainError, changeset
    end
  end
end

defimpl Langchain.ForOpenAIApi, for: Langchain.Message do
  alias Langchain.Message

  def for_api(%Message{} = fun) do
    %{
      "role" => fun.role,
      "content" => fun.content
    }
  end
end
