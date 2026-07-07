# LangChain Message Assembly and Event Emission

LLMChain handles LLM responses through two distinct paths that differ in process ownership and event flow. Understanding these paths is critical for debugging issues and knowing where certain operations should occur.

## Non-Streaming Path

When `stream: false`, the chat model makes a single API call and returns a complete `Message` struct:

1. `LLMChain.run/2` calls the chat model's `call/3` function
2. Chat model returns `{:ok, Message}` — a fully constructed message
3. `:on_llm_new_message` callback fires with the complete message
4. The chain runs message processors (JSON parsing, validation, etc.)
5. Chain fires `:on_message_processed` callback with final message
6. Message is added to chain's message list

**Process context**: When called directly, runs synchronously in the calling process. However, in LiveView/GenServer contexts, the chain is typically still run in an async Task to keep the main process free for handling messages and allowing cancellation of long-running calls. The `:on_llm_new_message` callback sends the complete message to the main process, which then adds it to its locally-maintained LLMChain.

## Streaming Path

When `stream: true`, responses arrive as partial `MessageDelta` structs over time. This typically involves **two separate processes**:

**HTTP Streaming Context (Req Task):**
1. Raw bytes arrive from LLM API stream
2. Chat model decodes and transforms bytes into `MessageDelta` structs
3. `:on_llm_new_delta` callback fires **immediately** for each delta batch
4. Deltas accumulate in the HTTP response body

**Main Process Context (LLMChain caller):**
1. After stream ends, LLMChain receives accumulated deltas
2. `apply_deltas/2` merges them using `MessageDelta.merge_delta/2`
3. When delta status becomes `:complete`, converts to `Message`
4. Chain fires `:on_message_processed` with completed message

**Key insight**: The `:on_llm_new_delta` callback runs in the HTTP task context, not the main process. For LiveView/GenServer integrations, these callbacks should send messages to the main process for state updates.

## Delta Merging

`MessageDelta` has two content fields:
- `content` — raw data from current chunk (cleared after merge)
- `merged_content` — accumulated `ContentPart` list (source of truth)

The merge process:
- Converts string content to `ContentPart` structs for consistency
- Merges content parts at their specified indices
- Tool calls are matched and merged by their `index` field (allowing parallel streaming of multiple tool calls)
- Status only upgrades (`:incomplete` → `:complete`), never downgrades
- Token usage accumulates across deltas

## LiveView/GenServer Integration Pattern

When running LLMChain from a LiveView or GenServer, use an async Task to keep the main process responsive. This applies to **both** streaming and non-streaming modes:

```elixir
# In LiveView: start chain in a Task
task = Task.async(fn ->
  LLMChain.run(chain, mode: :while_needs_response)
end)

# Store task reference to allow cancellation if needed
socket = assign(socket, :llm_task, task)
```

**For non-streaming**, the complete message arrives via `:on_llm_new_message`:

```elixir
# Callback sends complete message to LiveView
on_llm_new_message: fn _chain, message ->
  send(live_view_pid, {:llm_message, message})
end

# LiveView adds message to its local chain
def handle_info({:llm_message, message}, socket) do
  updated_chain =
    socket.assigns.llm_chain
    |> LLMChain.drop_delta()
    |> LLMChain.add_message(message)

  # Message is complete — safe to persist
  {:ok, _} = Messages.create_message(conversation_id, message)

  {:noreply, assign(socket, llm_chain: updated_chain, streaming_text: nil)}
end
```

**For streaming**, maintain a separate LLMChain in the main process and merge deltas as they arrive:

```elixir
# Callback sends deltas to LiveView
on_llm_new_delta: fn _chain, deltas ->
  send(live_view_pid, {:llm_delta, deltas})
end

# LiveView merges into its own chain
def handle_info({:llm_delta, deltas}, socket) do
  updated_chain = LLMChain.merge_deltas(socket.assigns.llm_chain, deltas)

  # Display partial content from the delta
  current_text = MessageDelta.content_to_string(updated_chain.delta, :text)

  {:noreply, assign(socket, llm_chain: updated_chain, streaming_text: current_text)}
end
```

This pattern allows:
- The Task to be cancelled if the user navigates away or submits a new message
- The LiveView to remain responsive to new events while streaming
- Real-time UI updates as deltas arrive

## Callback Types

**LLM-Level Callbacks** (fired by chat model during API interaction):
- `:on_llm_new_delta` — Each streaming delta batch (HTTP task context)
- `:on_llm_new_message` — Full non-streaming message received
- `:on_llm_token_usage` — Token usage information
- `:on_llm_ratelimit_info` — Rate limiting headers from API

**Chain-Level Callbacks** (fired by LLMChain after processing):
- `:on_message_processed` — Message complete and processed (use for persistence)
- `:on_tool_call_identified` — Tool name detected during streaming (args may be incomplete)
- `:on_tool_execution_started/completed/failed` — Tool execution lifecycle
- `:on_tool_response_created` — Tool results compiled into message
- `:on_message_processing_error` — Message processor failure

**Critical distinction**: Use `:on_llm_new_delta` for real-time UI updates, but wait for `:on_message_processed` before persisting to database or treating the message as complete.

## Tool Call display_text During Streaming

Each `ToolCall` has a `display_text` field for UI-friendly labels (e.g., "Reading file" instead of `file_read`). The library handles this automatically — **consumers of `:on_llm_new_delta` receive deltas with `display_text` already set**. No per-consumer augmentation is needed.

Resolution order: `Function.display_text` if defined on the tool, otherwise `Utils.humanize_tool_name/1` (e.g., `"file_read"` → `"File read"`).

### Why this is important

Streaming deltas flow through two separate paths that process the same data at different times:

1. **Callback path** — `:on_llm_new_delta` fires immediately as chunks arrive from the LLM. The library enriches tool calls with `display_text` here via `rewrap_callbacks_for_model`, so the UI can show tool names the moment they appear.

2. **Post-streaming path** — After the HTTP stream completes, `apply_deltas` processes the accumulated raw deltas through `merge_delta`, which independently sets `display_text` and fires `:on_tool_call_identified`.

These paths operate on separate copies of the deltas and don't interfere. Both use `display_text != nil` as an idempotent "already processed" guard.

Without callback-level enrichment, the UI would show blank tool labels during the entire streaming phase. By the time the post-streaming path runs, the tool may already be executing or finished.

## Observability: Telemetry & OpenTelemetry

LangChain emits `:telemetry` events for LLM calls, chain runs, and tool calls. This is separate from the callbacks above: callbacks drive application behavior (UI, persistence), while telemetry is for monitoring and tracing. There are two layers.

### Layer 1 — `LangChain.Telemetry` (`:telemetry` events, no extra deps)

Events follow `[:langchain, component, operation, stage]`. The lifecycle triples are:

- `[:langchain, :llm, :call, :start | :stop | :exception]`
- `[:langchain, :chain, :execute, :start | :stop | :exception]`
- `[:langchain, :tool, :call, :start | :stop | :exception]`

Plus content-bearing events `[:langchain, :llm, :prompt]` and `[:langchain, :llm, :response]`.

Rules when consuming these events:

- **Correlate with `:call_id`.** Every `:start`/`:stop`/`:exception` triple shares a `:call_id` UUID in its metadata.
- **Read duration from `:stop` and `:exception`.** Both carry a `duration` measurement (native time units) — failed operations are visible to latency metrics, not just successes.
- **`:token_usage`** is a `%LangChain.TokenUsage{}` on LLM `:stop` events (when the model reports it) and on chain `:stop` events, where it is **aggregated across all assistant messages** in the run (multi-turn/tool-calling safe). It can be `nil`.
- **`:provider`** (`"openai"`, `"anthropic"`, `"xai"`, …) is on LLM call events, sourced from the `ChatModel.provider/0` callback. Custom chat models that don't implement the optional callback get a provider derived from the module name via `ChatModel.provider/1`.
- **The chain metadata key is `:tools_count`** (plural, matching `:message_count`) — not `:tool_count`.
- **`:request_options`** is a map of the model's standard request parameters (`:temperature`, `:max_tokens`, `:top_p`, `:seed`, …) on LLM call events, extracted from the model struct by `ChatModel.request_options/1`. Absent parameters are omitted; an empty map means none were captured. LLM call events also carry `:output_type` (`"text"`/`"json"`) and `:endpoint` (when the model exposes one).
- **`:custom_context`** (your `LLMChain.custom_context`) is on chain and tool events, but intentionally **not** on LLM-level events — correlate via `:call_id` instead.
- **Privacy:** lifecycle events never carry message content. Content is only on the opt-in `[:langchain, :llm, :prompt]` / `[:langchain, :llm, :response]` events.

### Layer 2 — `LangChain.OpenTelemetry` (opt-in)

An optional integration that turns the above events into OpenTelemetry spans/metrics using a subset of the GenAI Semantic Conventions. Add `:opentelemetry_api` (+ `:opentelemetry`, `:opentelemetry_exporter`) to your deps, then call `LangChain.OpenTelemetry.setup/1` once at startup.

- Spans nest automatically for synchronous work: `invoke_agent {chain}` → `chat {model}` → `execute_tool {name}`.
- **Request parameters are emitted automatically** as `gen_ai.request.*` (temperature, max_tokens, top_p, seed, …) from `:request_options`, plus `gen_ai.output.type` (`"json"` for structured-output requests), `server.address`/`server.port` (from the endpoint), `gen_ai.response.finish_reasons`, `gen_ai.tool.description`, best-effort cache/reasoning token counts, and `gen_ai.conversation.id` (from a `:conversation_id` / `:langfuse_session_id` in `custom_context`). No config needed; only values the model actually set appear.
- **Message/argument/result capture is off by default** (PII). Enable per-flag via `setup/1`: `capture_input_messages`, `capture_output_messages`, `capture_tool_arguments`, `capture_tool_results`.
- **`enable_metrics: true` (default) does not record histograms directly** — it re-emits `[:langchain, :otel, :operation, :duration]`, `[:langchain, :otel, :token, :usage]`, and `[:langchain, :otel, :operation, :time_to_first_token]` events. You must attach a consumer (`Telemetry.Metrics` + reporter, PromEx) to record them.
- **Streaming time-to-first-token** is captured automatically: a `gen_ai.response.time_to_first_token` span attribute + `gen_ai.first_token` span event on the LLM span, and the metric event above. Best-effort — only for providers that stream through the shared streaming path.
- **Async tools** (`async: true`) run in a separate `Task`. `LLMChain`'s built-in tool executor now **propagates the OTel context into the Task automatically**, so async tool spans nest under the chain span with no extra work. If you spawn your own processes running LangChain ops, re-attach the context yourself (e.g. inside the `:on_tool_pre_execution` callback).
- Use `LangChain.OpenTelemetry.without_tracing/1` to suppress spans **and metrics** for utility sub-chains (process-scoped; not inherited by spawned processes).

See the [Observability guide](guides/observability.md) for full details and Langfuse integration.