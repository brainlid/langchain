defmodule LangChain.ChatModels.ChatModel do
  alias LangChain.Message
  alias LangChain.MessageDelta

  @type call_response ::
          {:ok, Message.t() | [Message.t()] | [MessageDelta.t()]} | {:error, String.t()}

  @type t :: Ecto.Schema.t()

  @callback call(
              t(),
              String.t() | [Message.t()],
              [LangChain.Function.t()],
              nil | (Message.t() | MessageDelta.t() -> any())
            ) :: call_response()
end
