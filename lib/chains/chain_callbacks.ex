defmodule LangChain.Chains.ChainCallbacks do
  @moduledoc """
  Defines the callbacks fired by an LLMChain and LLM module.

  A callback handler is a map that defines the specific callback event with a
  function to execute for that event.

  ## Example

  A sample configured callback handler that forwards received data to a specific
  LiveView.

      live_view_pid = self()

      my_handlers = %{
        on_llm_new_delta: fn _chain, new_deltas -> send(live_view_pid, {:received_delta, new_deltas}) end,
        on_message_processed: fn _chain, new_message -> send(live_view_pid, {:received_message, new_message}) end,
        on_error_message_created: fn _chain, new_message -> send(live_view_pid, {:received_message, new_message}) end
      }

      model = SomeLLM.new!(%{...})

      chain =
        %{llm: model}
        |> LLMChain.new!()
        |> LLMChain.add_callback(my_handlers)

  """

  alias LangChain.Chains.LLMChain
  alias LangChain.Function
  alias LangChain.Message
  alias LangChain.Message.ToolCall
  alias LangChain.Message.ToolResult
  alias LangChain.MessageDelta
  alias LangChain.TokenUsage

  @typedoc """
  Executed when an LLM is streaming a response and a new MessageDelta (or token)
  was received.

  - `:index` is optionally present if the LLM supports sending `n` versions of a
    response.

  The return value is discarded.
  """
  @type llm_new_delta :: (LLMChain.t(), [MessageDelta.t()] -> any())

  @typedoc """
  Executed when an LLM is not streaming and a full message was received.

  The return value is discarded.
  """
  @type llm_new_message :: (LLMChain.t(), Message.t() -> any())

  @typedoc """
  Executed when an LLM (typically a service) responds with rate limiting
  information.

  The specific rate limit information depends on the LLM. It returns a map with
  all the available information included.

  The return value is discarded.
  """
  @type llm_ratelimit_info :: (LLMChain.t(), info :: %{String.t() => any()} -> any())

  @typedoc """
  Executed when an LLM response reports the token usage in a
  `LangChain.TokenUsage` struct. The data returned depends on the LLM.

  The return value is discarded.
  """
  @type llm_token_usage :: (LLMChain.t(), TokenUsage.t() -> any())

  @typedoc """
  Executed when an LLM response is received through an HTTP response. The entire
  set of raw response headers can be received and processed.

  The return value is discarded.

  ## Example

  A function declaration that matches the signature.

      def handle_llm_response_headers(chain, response_headers) do
        # This demonstrates how to send the response headers to a
        # LiveView assuming the LiveView's pid was stored in the chain's
        # custom_context.
        send(chain.custom_context.live_view_pid, {:req_response_headers, response_headers})

        IO.inspect(response_headers)
      end
  """
  @type llm_response_headers :: (LLMChain.t(), response_headers :: map() -> any())

  @typedoc """
  Executed when an LLMChain has completed processing a received assistant
  message. This fires when a message is complete either after assembling
  streaming deltas or when a full message is received when not streaming.

  This is the best way to be notified when a message is "done" and should be
  handled by the application.

  The handler's return value is discarded.
  """
  @type chain_message_processed :: (LLMChain.t(), Message.t() -> any())

  @typedoc """
  Executed when an LLMChain, in response to an error from the LLM, generates a
  new, automated response message intended to be returned to the LLM.

  """
  @type chain_error_message_created :: (LLMChain.t(), Message.t() -> any())

  @typedoc """
  Executed when processing a received message errors or fails. The erroring
  message is included in the callback with the state of processing that was
  completed before erroring.

  The handler's return value is discarded.
  """
  @type chain_message_processing_error :: (LLMChain.t(), Message.t() -> any())

  @typedoc """
  Executed when the chain begins executing a tool call.

  This fires immediately before tool execution starts, allowing UIs to show
  real-time feedback like "Searching the web..." or "Creating file...".

  - First argument: LLMChain.t()
  - Second argument: ToolCall struct being executed
  - Third argument: Function struct for the tool (includes display_text)

  The handler's return value is discarded.
  """
  @type chain_tool_execution_started :: (LLMChain.t(), ToolCall.t(), Function.t() -> any())

  @typedoc """
  Executed when a single tool execution completes successfully.

  Fires after individual tool execution, before results are aggregated.
  Useful for showing per-tool success indicators.

  - First argument: LLMChain.t()
  - Second argument: ToolCall that was executed
  - Third argument: ToolResult that was generated

  The handler's return value is discarded.
  """
  @type chain_tool_execution_completed :: (LLMChain.t(), ToolCall.t(), ToolResult.t() -> any())

  @typedoc """
  Executed when a single tool execution fails.

  Fires when tool execution raises an exception or returns an error result.

  - First argument: LLMChain.t()
  - Second argument: ToolCall that failed
  - Third argument: Error reason or exception

  The handler's return value is discarded.
  """
  @type chain_tool_execution_failed :: (LLMChain.t(), ToolCall.t(), term() -> any())

  @typedoc """
  Executed when the chain uses one or more tools and the resulting ToolResults
  are generated as part of a tool response message.

  The handler's return value is discarded.
  """
  @type chain_tool_response_created :: (LLMChain.t(), Message.t() -> any())

  @typedoc """
  Executed when the chain failed multiple times used up the `max_retry_count`
  resulting in the process aborting and returning an error.

  The handler's return value is discarded.
  """
  @type chain_retries_exceeded :: (LLMChain.t() -> any())

  @typedoc """
  The supported set of callbacks for an LLM module.
  """
  @type chain_callback_handler :: %{
          # model-level callbacks
          optional(:on_llm_new_delta) => llm_new_delta(),
          optional(:on_llm_new_message) => llm_new_message(),
          optional(:on_llm_ratelimit_info) => llm_ratelimit_info(),
          optional(:on_llm_token_usage) => llm_token_usage(),
          optional(:on_llm_response_headers) => llm_response_headers(),

          # Chain-level callbacks
          optional(:on_message_processed) => chain_message_processed(),
          optional(:on_message_processing_error) => chain_message_processing_error(),
          optional(:on_error_message_created) => chain_error_message_created(),
          optional(:on_tool_execution_started) => chain_tool_execution_started(),
          optional(:on_tool_execution_completed) => chain_tool_execution_completed(),
          optional(:on_tool_execution_failed) => chain_tool_execution_failed(),
          optional(:on_tool_response_created) => chain_tool_response_created(),
          optional(:on_retries_exceeded) => chain_retries_exceeded()
        }
end
