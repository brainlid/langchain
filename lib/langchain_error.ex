defmodule LangChain.LangChainError do
  @moduledoc """
  Exception used for raising LangChain specific errors.

  It stores the `:message`. Passing an Ecto.Changeset with an error
  converts the error into a string message.

      raise LangChainError, changeset

      raise LangChainError, "Message text"

      raise LangChainError, type: "overloaded_error", message: "Message text"

  The error struct contains the following keys:

  - `:type` - A string code to make detecting and responding to specific errors easier. This may have values like "length" or "overloaded_error". The specific meaning of the type is dependent on the service or model.

  - `:message` - A string representation or explanation of the error.

  - `:original` - If a exception was caught and wrapped into a LangChainError, this may be the original message that was encountered.
  """
  import LangChain.Utils, only: [changeset_error_to_string: 1]
  alias __MODULE__

  @type t :: %LangChainError{}

  defexception [:type, :message, :original]

  @doc """
  Create the exception using either a message or a changeset who's errors are
  converted to a message.
  """
  @spec exception(message :: String.t() | Ecto.Changeset.t()) :: t() | no_return()
  def exception(message) when is_binary(message) do
    %LangChainError{message: message}
  end

  def exception(%Ecto.Changeset{} = changeset) do
    text_reason = changeset_error_to_string(changeset)
    %LangChainError{type: "changeset", message: text_reason}
  end

  def exception(opts) when is_list(opts) do
    %LangChainError{
      message: Keyword.fetch!(opts, :message),
      type: Keyword.get(opts, :type),
      original: Keyword.get(opts, :original),
    }
  end
end
