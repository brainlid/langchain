defmodule Langchain.LangchainError do
  @moduledoc """
  Exception used for raising Langchain specific errors.

  It stores the `:message`. Passing an Ecto.Changeset with an error
  converts the error into a string message.

      raise LangchainError, changeset

      raise LangchainError, "Message text"

  """
  import Langchain.Utils, only: [changeset_error_to_string: 1]
  alias __MODULE__

  @type t :: %LangchainError{}

  defexception [:message]

  @doc """
  Create the exception using either a message or a changeset who's errors are
  converted to a message.
  """
  @spec exception(message :: String.t() | Ecto.Changeset.t()) :: t()
  def exception(message) when is_binary(message) do
    %LangchainError{message: message}
  end

  def exception(%Ecto.Changeset{} = changeset) do
    text_reason = changeset_error_to_string(changeset)
    %LangchainError{message: text_reason}
  end
end
