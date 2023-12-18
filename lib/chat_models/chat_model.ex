defmodule LangChain.ChatModels.ChatModel do
  @type call_response :: {:ok, Message.t() | [Message.t()]} | {:error, String.t()}

  @type t :: Ecto.Schema.t()

  @callback call(
              t(),
              String.t() | [Message.t()],
              [LangChain.Function.t()],
              nil | (Message.t() | MessageDelta.t() -> any())
            ) :: call_response()
end
