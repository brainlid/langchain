defmodule LangChain.ChatModels.ChatModel do
  alias LangChain.Message
  alias LangChain.MessageDelta
  alias LangChain.Function

  @type call_response :: {:ok, Message.t() | [Message.t()] | [MessageDelta.t()]} | {:error, String.t()}

  @type tool :: Function.t()
  @type tools :: [tool()]

  @type t :: Ecto.Schema.t()

  @callback call(
              t(),
              String.t() | [Message.t()],
              [LangChain.Function.t()],
              nil | (Message.t() | MessageDelta.t() -> any())
            ) :: call_response()
end
