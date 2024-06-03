defmodule LangChain.ChatModels.LLMCallbacks do
  @moduledoc """
  Defines the callbacks fired by an LLM module.

  A callback handler is a map that defines the specific callback event with a
  function to execute for that event.

  ## Example

  A sample configured callback handler that forwards received data to a specific
  LiveView.

      live_view_pid = self()

      my_handlers = %{
        on_llm_new_message: fn new_message -> send(live_view_pid, {:received_message, new_message})
      }

      model = SomeLLM.new!(%{callbacks: [my_handlers]})
      chain = LLMChain.new!(%{llm: model})

  """
  alias LangChain.Message
  alias LangChain.MessageDelta

  @typedoc """
  Executed when an LLM is streaming a response and a new MessageDelta (or token)
  was received.

  - `:index` is optionally present if the LLM supports sending `n` versions of a
    response.

  The return value is discarded.

  ## Example

  A function declaration that matches the signature.

      def handle_llm_new_delta(_chat_model, _index, delta) do
        IO.write(delta)
      end
  """
  @type llm_new_delta :: (model :: struct(), MessageDelta.t() -> any())

  @typedoc """
  Executed when an LLM is not streaming and a full message was received.

  The return value is discarded.

  ## Example

  A function declaration that matches the signature.

      def handle_llm_new_message(_chat_model, message) do
        IO.inspect(message)
      end
  """
  @type llm_new_message :: (model :: struct(), Message.t() -> any())

  @typedoc """
  The supported set of callbacks for an LLM module.
  """
  @type llm_callback_handler :: %{
          on_llm_new_delta: llm_new_delta(),
          on_llm_new_message: llm_new_message()
        }
end
