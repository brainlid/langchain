defmodule LangChain.Chains.ChainCallbacks do
  @moduledoc """
  Defines the callbacks fired by an LLMChain.

  A callback handler is a map that defines the specific callback event with a
  function to execute for that event.

  ## Example

  A sample configured callback handler that forwards received data to a specific
  LiveView.

      live_view_pid = self()

      my_handlers = %{
        handle_chain_error_message_created: fn new_message -> send(live_view_pid, {:received_message, new_message})
      }

      model = SomeLLM.new!(%{callbacks: [my_handlers]})
      chain = LLMChain.new!(%{llm: model})

  """
  alias LangChain.Chains.LLMChain
  alias LangChain.Message

  @typedoc """
  Executed when an LLMChain has completed processing a received assistant
  message.

  The handler's return value is discarded.

  ## Example

  A function declaration that matches the signature.

      def handle_chain_message_processed(chain, message) do
        IO.inspect(message)
      end
  """
  @type chain_message_processed :: (LLMChain.t(), Message.t() -> any())

  @typedoc """
  Executed when an LLMChain, in response to an error from the LLM, generates a
  new, automated response message intended to be returned to the LLM.

  The handler's return value is discarded.

  ## Example

  A function declaration that matches the signature.

      def handles_chain_error_message_created(chain, new_message) do
        IO.inspect(new_message)
      end
  """
  @type chain_error_message_created :: (LLMChain.t(), Message.t() -> any())

  @typedoc """
  Executed when processing a received message errors or fails. The erroring
  message is included in the callback with the state of processing that was
  completed before erroring.

  The handler's return value is discarded.

  ## Example

  A function declaration that matches the signature.

      def handle_chain_message_processing_error(chain, new_message) do
        IO.inspect(new_message)
      end
  """
  @type chain_message_processing_error :: (LLMChain.t(), Message.t() -> any())

  @typedoc """
  Executed when the chain uses one or more tools and the resulting ToolResults
  are generated as part of a tool response message.

  The handler's return value is discarded.

  ## Example

  A function declaration that matches the signature.

      def handle_chain_tool_response_created(chain, new_message) do
        IO.inspect(new_message)
      end
  """
  @type chain_tool_response_created :: (LLMChain.t(), Message.t() -> any())

  @typedoc """
  Executed when the chain failed multiple times used up the `max_retry_count`
  resulting in the process aborting and returning an error.

  The handler's return value is discarded.

  ## Example

  A function declaration that matches the signature.

      def handle_retries_exceeded(chain) do
        IO.inspect(chain)
      end
  """
  @type chain_retries_exceeded :: (LLMChain.t() -> any())

  @typedoc """
  The supported set of callbacks for an LLM module.
  """
  @type chain_callback_handler :: %{
          on_message_processed: chain_message_processed(),
          on_message_processing_error: chain_message_processing_error(),
          on_error_message_created: chain_error_message_created(),
          on_tool_response_created: chain_tool_response_created(),
          on_retries_exceeded: chain_retries_exceeded()
        }
end
