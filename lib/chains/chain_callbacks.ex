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
  alias LangChain.LangChainError
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
  Executed when a tool call is identified during streaming, before execution begins.

  This fires as soon as we have enough information to identify the tool (at minimum, the `name` field).
  The tool call may be incomplete - `call_id` might not be available yet, and `arguments` may be partial.

  This callback provides early notification for UI feedback like "Searching web..." while the LLM
  is still streaming the complete tool call.

  Timing:
  - Fires: As soon as tool name is detected in streaming deltas
  - Before: Tool arguments are fully received
  - Before: Tool execution begins

  Arguments:
  - First: LLMChain.t() - Current chain state
  - Second: ToolCall.t() - Tool call struct (may be incomplete, but has name)
  - Third: Function.t() - Function definition (includes display_text)

  The handler's return value is discarded.

  ## Example

      callback_handler = %{
        on_tool_call_identified: fn _chain, tool_call, func ->
          IO.puts("Tool identified: \#{func.display_text || tool_call.name}")
        end
      }

  """
  @type chain_tool_call_identified :: (LLMChain.t(), ToolCall.t(), Function.t() -> any())

  @typedoc """
  Executed when the chain begins executing a tool call.

  This fires immediately before tool execution starts, allowing UIs to show
  real-time feedback like "Searching the web..." or "Creating file...".

  Note: This callback fires in the **parent chain process**, before any per-tool
  async Task is spawned. For code that must run *inside* the per-tool process
  (e.g. propagating tenancy/OTel/Sentry context across the async boundary), use
  `:on_tool_pre_execution` instead.

  It fires only for a call that is going to run. An `:on_tool_call_review`
  handler settles its call before this point, so a call it denied, or held to ask
  the user about, is reported to `:on_tool_execution_failed` or
  `:on_tool_interrupted` without ever being announced as started. A UI tracking tool state must reach those from
  `:on_tool_call_identified` as well as from here.

  The ToolCall carries whatever arguments review left it with, which is what the
  tool is actually given.

  - First argument: LLMChain.t()
  - Second argument: ToolCall struct being executed
  - Third argument: Function struct for the tool (includes display_text)

  The handler's return value is discarded.
  """
  @type chain_tool_execution_started :: (LLMChain.t(), ToolCall.t(), Function.t() -> any())

  @typedoc """
  Executed inside the process that will run the tool, immediately before the
  tool function is invoked.

  Unlike `:on_tool_execution_started` (which fires in the parent chain process
  before any async Task is spawned), `:on_tool_pre_execution` fires in whichever
  process actually runs the tool:

  - For `async: true` tools — fires inside the spawned `Task.async/1`.
  - For `async: false` tools — fires in the chain's own process.
  - For tools executed via `execute_tool_calls_with_decisions/3` — fires in
    the chain's own process.

  This is the correct hook for code that depends on per-process state — for
  example, re-applying tenant/observability context that lives in the process
  dictionary across an async Task boundary.

  - First argument: LLMChain.t()
  - Second argument: ToolCall struct about to be executed
  - Third argument: Function struct for the tool

  The handler's return value is discarded.
  """
  @type chain_tool_pre_execution :: (LLMChain.t(), ToolCall.t(), Function.t() -> any())

  @typedoc """
  Executed before a tool call is announced or run, to decide whether it may
  proceed.

  This is the one tool callback whose return value is used. It fires in the
  parent chain process ahead of `:on_tool_execution_started`, before any async
  `Task` is spawned, so a call that is turned away is never announced as
  running and never reaches the tool function.

  It fires on every path that executes a tool, including
  `LLMChain.execute_tool_calls_with_decisions/3`. A human approving a call in a
  Human-in-the-Loop workflow does not exempt it from review.

  It does not fire for a call naming a tool the chain does not have. There is no
  `Function` to hand a handler, so there is nothing to allow or deny. A handler
  auditing what the model attempts sees only calls that resolve to a real tool.

  - First argument: LLMChain.t()
  - Second argument: ToolCall struct under review
  - Third argument: Function struct for the tool
  - Fourth argument: a review context map describing the circumstances

  ## The review context

  The fourth argument describes the circumstances of the call rather than the
  call itself.

  - `:human_decision` is `nil` while the model's call is being run directly. It
    is `:approve` or `:edit` once the user has decided on this call through
    `LLMChain.execute_tool_calls_with_decisions/3`, which is the only thing that
    gives it any other value.
  - `:custom_context` is the context the tool will be given: the context passed
    to `LLMChain.execute_tool_calls/2` when one is supplied, and
    `chain.custom_context` otherwise. A handler scoped to a tenant reads this
    rather than `chain.custom_context`, which is not always the context the call
    will run under.

  On a chain whose `custom_context` is `%{tenant_id: "acme"}`, a handler running
  against the model's own call is given:

      %{human_decision: nil, custom_context: %{tenant_id: "acme"}}

  Match on the keys a handler cares about rather than the whole map. It gains
  keys as the chain learns more about a call.

  ## Return values

  - `:ok` - express no opinion; the call proceeds to the next handler
  - `{:update_arguments, map}` - rewrite the arguments and keep going. Later
    handlers review the rewritten call, and the tool runs with it.
  - `{:deny, reason}` - refuse the call. The tool never runs and `reason` is
    returned to the model as the tool result.
  - `{:interrupt, message, interrupt_data}` - refuse the call for now and
    interrupt, producing a `ToolResult` with `is_interrupt: true`. Use this to
    put the call in front of the user before it runs.

  Handlers are consulted in the order their maps were added to the chain. The
  first `{:deny, _}` or `{:interrupt, _, _}` settles the call and the remaining
  handlers are skipped.

  An exception raised by a handler aborts the whole batch of tool calls before
  any of them runs, including calls an earlier handler already cleared. A
  handler that cannot reach the system it consults should decide the call, by
  denying it, rather than raise.

  ## Asking the user to confirm

  A handler decides from the arguments the model chose and what the chain knows
  about the user, so the same tool can run without comment in one case and be
  worth stopping on in another. `{:interrupt, _, _}` puts that call in front of
  the user before it runs.

  Review runs again for the call once the user answers, so a handler has to
  recognize their answer coming back. `:human_decision` is what it reads. A
  handler that ignores it asks the same question a second time and the call
  never runs.

  Take a file deletion, on a chain whose `custom_context` carries the project
  the user is working in:

      on_tool_call_review: fn _chain, call, _func, review ->
        cond do
          # The user already answered for this call. Their answer stands.
          review.human_decision ->
            :ok

          call.name == "delete_file" and
              not inside?(call.arguments["path"], review.custom_context.workspace_root) ->
            {:interrupt, "That file is outside your project. Delete it anyway?",
             %{path: call.arguments["path"]}}

          true ->
            :ok
        end
      end

  A delete inside the project runs without comment. A delete outside it stops
  and asks. Withholding the tool cannot draw that line, because which case a
  call falls into depends on the path the model chose.

  ### First pass, on the model's call

  The handler is given
  `%{human_decision: nil, custom_context: %{workspace_root: "/home/sam/project"}}`.
  The user has not been asked anything yet, the path is outside the project, and
  the handler interrupts. The tool never runs, and the call is answered by a
  stand-in result:

      %ToolResult{
        tool_call_id: "call_abc123",
        name: "delete_file",
        display_text: "Deleting the file",
        content: [%ContentPart{type: :text, content: "That file is outside your project. Delete it anyway?"}],
        is_error: false,
        is_interrupt: true,
        interrupt_data: %{path: "/home/sam/notes.md"}
      }

  The message becomes `content`, as a list of ContentParts rather than the string
  the handler returned. `interrupt_data` is carried through untouched and is
  never shown to the model. It is for the code that builds the question, which
  needs the path here to show the user what they are agreeing to.

  `:on_tool_interrupted` fires with that result, and `LLMChain.run/2` returns:

      {:interrupt, chain, %{path: "/home/sam/notes.md", tool_call_id: "call_abc123"}}

  The chain adds `:tool_call_id` to whatever the handler returned, so a handler
  does not need to put the call id in `interrupt_data` itself. That id is what
  ties the answer back to the call it came from.

  ### Second pass, once the user has answered

  The host resumes through `LLMChain.execute_tool_calls_with_decisions/3` with
  the answer it collected. Review runs again on the same call, and this time the
  handler is given:

      %{human_decision: :approve, custom_context: %{workspace_root: "/home/sam/project"}}

  The first branch matches, the handler returns `:ok`, and the tool runs. The
  call is answered the way any other cleared call is:

      %ToolResult{
        tool_call_id: "call_abc123",
        name: "delete_file",
        display_text: "Deleting the file",
        content: [%ContentPart{type: :text, content: "deleted"}],
        is_error: false,
        is_interrupt: false,
        interrupt_data: nil
      }

  An `:edit` answer reaches the handler the same way, with
  `human_decision: :edit` and the call already carrying the arguments the user
  supplied. A `:reject` answer does not reach review at all, because the call
  does not run.

  An interrupt raised on the model's call leaves the chain holding a ToolResult
  that already answers it, so a resumed run replaces that result with
  `LangChain.Message.replace_tool_result/3` rather than adding a second one for
  the same call.

  ## Rewritten arguments and the transcript

  `{:update_arguments, map}` changes what the tool receives and what
  `:on_tool_execution_started` reports. It does not change the assistant message
  already recorded in `chain.messages`, which keeps the arguments the model
  produced, and it does not change what `:on_tool_call_identified` reported
  during streaming. A UI or audit trail that reads arguments off the assistant
  message shows what was asked for, not what ran.

  ## Denials and the failure counter

  A denied call is reported with `is_error: false`. It is a decision about the
  call, not a fault in it, so it leaves the chain's failure counter alone and
  cannot exhaust `max_retry_count`. A policy that turns down many calls in a row
  will not abort the run.

  That counter is also the only bound on a tool-calling loop, so a model that
  keeps reaching for a tool that keeps being denied is not stopped by it. Write
  the `reason` so the model can act on it, naming what to do instead of
  retrying, since the reason is the whole of what the model learns.

  ## Example

      callback_handler = %{
        on_tool_call_review: fn _chain, tool_call, _func, _review ->
          case Policy.check(tool_call.name, tool_call.arguments) do
            :allowed -> :ok
            {:refused, why} -> {:deny, why}
          end
        end
      }

  """
  @type chain_tool_call_review ::
          (LLMChain.t(), ToolCall.t(), Function.t(), map() ->
             :ok
             | {:update_arguments, map()}
             | {:deny, String.t()}
             | {:interrupt, String.t(), map()})

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

  Fires when tool execution raises an exception, returns an error result, names a
  tool that doesn't exist, is rejected during human review, or is denied by an
  `:on_tool_call_review` handler.

  A denial and a human rejection both report a `ToolResult` with
  `is_error: false`, because each is a decision about the call rather than a
  fault in it. A handler that counts failures should read the `ToolResult` rather
  than assume every call it hears about here errored.

  - First argument: LLMChain.t()
  - Second argument: ToolCall that failed
  - Third argument: The error content returned to the model

  The handler's return value is discarded.
  """
  @type chain_tool_execution_failed :: (LLMChain.t(), ToolCall.t(), term() -> any())

  @typedoc """
  Executed when a tool execution raises an exception that LangChain rescues.

  Fires in addition to `:on_tool_execution_failed`, not instead of it. That
  callback receives the message the model sees; this one receives the exception
  itself, which is what an error tracker needs in order to group and fingerprint
  the failure.

  Does not fire for `{:error, reason}` results, tool calls naming a tool that
  doesn't exist, human rejections, or interrupts. None of those involve a rescued
  exception.

  - First argument: LLMChain.t()
  - Second argument: ToolCall that raised
  - Third argument: The rescued exception
  - Fourth argument: The exception's stacktrace

  Fires after the tool's OpenTelemetry span has closed, so attributes set here land
  on the enclosing chain span rather than on `execute_tool`.

  Exceptions and stacktraces can carry application data from the frames they
  captured. LangChain never sends them to the model, but handlers should not expose
  them to untrusted clients.

  The handler's return value is discarded.
  """
  @type chain_tool_execution_exception ::
          (LLMChain.t(), ToolCall.t(), Exception.t(), Exception.stacktrace() -> any())

  @typedoc """
  Executed when one or more tools return an interrupt signal.

  Fires once per tool execution batch with all interrupted results.
  The tool is paused and awaiting external input to continue.

  - First argument: LLMChain.t()
  - Second argument: List of ToolResult structs with `is_interrupt: true`

  The handler's return value is discarded.
  """
  @type chain_tool_interrupted :: (LLMChain.t(), [ToolResult.t()] -> any())

  @typedoc """
  Executed when the chain uses one or more tools and the resulting ToolResults
  are generated as part of a tool response message.

  The handler's return value is discarded.
  """
  @type chain_tool_response_created :: (LLMChain.t(), Message.t() -> any())

  @typedoc """
  Executed when an individual LLM API call fails with an error.

  This fires on **every** LLM call failure, including transient errors that may
  be retried or recovered from via fallbacks. It provides visibility into errors
  that would otherwise be invisible when retries succeed.

  Use this callback for diagnostic/observational purposes -- logging, metrics,
  debug dashboards. The chain may continue executing after this callback fires.

  ## Examples

  Common scenarios where this fires:
  - Rate limit errors (may be retried)
  - Overloaded/server errors (may fall back to another model)
  - Authentication errors (terminal)
  - Network timeouts (may be retried)

  In a retry loop: fires once per failed attempt, not just when retries are
  exhausted. In a fallback chain: fires for each model that fails before the
  next one is tried.

      callback_handler = %{
        on_llm_error: fn _chain, error ->
          Logger.warning("LLM call failed: \#{inspect(error)}")
        end
      }

  - First argument: LLMChain.t() - Current chain state
  - Second argument: LangChainError.t() - The error from the LLM call

  The handler's return value is discarded.
  """
  @type chain_llm_error :: (LLMChain.t(), LangChainError.t() -> any())

  @typedoc """
  Executed when the chain encounters a terminal error and is returning an error
  result to the caller.

  Unlike `on_llm_error` which fires on every individual LLM failure (including
  transient ones), this callback fires exactly **once** when the chain has
  exhausted all recovery options (retries, fallbacks) and is giving up.

  This is the chain-level "final answer is an error" signal. Use this for
  application-level error handling -- updating UI state, notifying users,
  recording failures.

  ## Examples

  Scenarios where this fires:
  - All retry attempts exhausted
  - All fallback models failed
  - Unrecoverable error (e.g., invalid request)
  - Rescued exception during chain execution

      callback_handler = %{
        on_error: fn _chain, error ->
          send(live_view_pid, {:chain_error, error})
        end
      }

  - First argument: LLMChain.t() - Chain state at time of failure
  - Second argument: LangChainError.t() - The terminal error

  The handler's return value is discarded.
  """
  @type chain_error :: (LLMChain.t(), LangChainError.t() -> any())

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
          optional(:on_tool_call_identified) => chain_tool_call_identified(),
          optional(:on_tool_execution_started) => chain_tool_execution_started(),
          optional(:on_tool_pre_execution) => chain_tool_pre_execution(),
          optional(:on_tool_call_review) => chain_tool_call_review(),
          optional(:on_tool_execution_completed) => chain_tool_execution_completed(),
          optional(:on_tool_execution_failed) => chain_tool_execution_failed(),
          optional(:on_tool_execution_exception) => chain_tool_execution_exception(),
          optional(:on_tool_interrupted) => chain_tool_interrupted(),
          optional(:on_tool_response_created) => chain_tool_response_created(),
          optional(:on_llm_error) => chain_llm_error(),
          optional(:on_error) => chain_error(),
          optional(:on_retries_exceeded) => chain_retries_exceeded()
        }
end
