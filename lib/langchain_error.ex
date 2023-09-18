defmodule LangChain.LangChainError do
  @moduledoc """
  Exception used for raising LangChain specific errors.

  It stores the `:message`. Passing an Ecto.Changeset with an error
  converts the error into a string message.

      raise LangChainError, changeset

      raise LangChainError, "Message text"

  """
  import LangChain.Utils, only: [changeset_error_to_string: 1]
  alias __MODULE__

  @type t :: %LangChainError{}

  defexception [:message]

  @doc """
  Create the exception using either a message or a changeset who's errors are
  converted to a message.
  """
  @spec exception(message :: String.t() | Ecto.Changeset.t()) :: t()
  def exception(message) when is_binary(message) do
    %LangChainError{message: message}
  end

  def exception(%Ecto.Changeset{} = changeset) do
    text_reason = changeset_error_to_string(changeset)
    %LangChainError{message: text_reason}
  end
end
