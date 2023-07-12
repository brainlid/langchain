defmodule Langchain.LangchainError do
  import Langchain.Utils, only: [changeset_error_to_string: 1]
  alias __MODULE__

  defexception [:message]

  def exception(message) when is_binary(message) do
    %LangchainError{message: message}
  end

  def exception(%Ecto.Changeset{} = changeset) do
    text_reason = changeset_error_to_string(changeset)
    %LangchainError{message: text_reason}
  end
end
