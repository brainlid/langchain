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