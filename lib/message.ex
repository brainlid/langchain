defmodule Langchain.Message do
  use Ecto.Schema
  import Ecto.Changeset
  alias __MODULE__
  alias Langchain.LangchainError

  @primary_key false
  embedded_schema do
    field :content, :string
    field :index, :integer
    field :role, Ecto.Enum, values: [:system, :user, :assistant, :function], default: :user
  end

  @type t :: %Message{}

  @create_fields [:role, :content]
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
  Create a new function message to represent the result of an executed
  function.
  """
  @spec new_function(content :: String.t()) :: {:ok, t()} | {:error, Ecto.Changeset.t()}
  def new_function(content) do
    new(%{role: :function, content: content})
  end

  @doc """
  Create a new function message to represent the result of an executed
  function.
  """
  @spec new_function!(content :: String.t()) :: t() | no_return()
  def new_function!(content) do
    case new_function(content) do
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
