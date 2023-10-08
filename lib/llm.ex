defmodule LangChain.LLM do
  @moduledoc """
  A behaviour for LLM
  """

  alias LangChain.Message
  alias LangChain.MessageDelta

  @type call_response :: {:ok, Message.t() | [Message.t()]} | {:error, String.t()}

  @callback call(
              struct(),
              String.t() | [Message.t()],
              [LangChain.Function.t()],
              nil | (Message.t() | MessageDelta.t() -> any())
            ) :: call_response()

  def call(%model{} = config, messages, functions, callback_fn) do
    model.call(config, messages, functions, callback_fn)
  end
end
