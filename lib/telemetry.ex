defmodule LangChain.Telemetry do
  @moduledoc """
  Telemetry events for LangChain.

  This module defines telemetry events that other applications can attach to.
  It provides a standardized way to emit events for various operations in the
  LangChain library without implementing tracing functionality.

  ## Event Naming

  Events follow the convention: `[:langchain, component, operation, stage]`

  ## Core Events

  * `[:langchain, :llm, :call, :start]` - Emitted when an LLM call starts
  * `[:langchain, :llm, :call, :stop]` - Emitted when an LLM call completes
  * `[:langchain, :llm, :call, :exception]` - Emitted when an LLM call raises an exception
  * `[:langchain, :llm, :prompt]` - Emitted when a prompt is sent to an LLM
  * `[:langchain, :llm, :response]` - Emitted when a response is received from an LLM
  * `[:langchain, :llm, :response, :non_streaming]` - Emitted when a non-streaming response is received from an LLM
  * `[:langchain, :llm, :response, :streaming]` - Emitted when a streaming response is received from an LLM
  * `[:langchain, :chain, :execute, :start]` - Emitted when a chain execution starts
  * `[:langchain, :chain, :execute, :stop]` - Emitted when a chain execution completes
  * `[:langchain, :chain, :execute, :exception]` - Emitted when a chain execution raises an exception
  * `[:langchain, :message, :process, :start]` - Emitted when message processing starts
  * `[:langchain, :message, :process, :stop]` - Emitted when message processing completes
  * `[:langchain, :message, :process, :exception]` - Emitted when message processing raises an exception
  * `[:langchain, :tool, :call, :start]` - Emitted when a tool call starts
  * `[:langchain, :tool, :call, :stop]` - Emitted when a tool call completes
  * `[:langchain, :tool, :call, :exception]` - Emitted when a tool call raises an exception

  ## Metadata Fields

  The following metadata fields are automatically injected or available in events:

  * `:call_id` - A UUID (via `Ecto.UUID.generate/0`) that correlates start, stop, and
    exception events within a single `span/3` or `start_event/2` call. Automatically
    injected via `Map.put_new/3`, so callers can supply their own ID to override.

  * `:provider` - The LLM provider name (e.g. `"openai"`, `"anthropic"`, `"google"`).
    Included in LLM call metadata by each chat model implementation via the
    `ChatModel.provider/0` callback.

  * `:custom_context` - User-supplied context data from `LLMChain.custom_context`.
    Included in chain execution and tool call metadata. Not included in LLM-level
    telemetry (correlate via `call_id` instead).

  * `:token_usage` - A `%TokenUsage{}` struct with input/output token counts. Included
    in LLM call `:stop` events and chain execution `:stop` events when available (via
    the `:enrich_stop` callback). `nil` when the model does not report usage.

  * `:last_message` - The final assembled `%Message{}` from the LLM response. Included
    in chain execution `:stop` events. For streaming responses this is the fully
    assembled message (not individual deltas).

  ## Expected Metadata Shape by Event

  * **LLM call `:start`**:
    `%{model: String.t(), provider: String.t(), message_count: integer(), tools_count: integer(), call_id: String.t()}`

  * **LLM call `:stop`** (includes enriched fields):
    `%{model: String.t(), provider: String.t(), message_count: integer(), tools_count: integer(), call_id: String.t(), token_usage: TokenUsage.t() | nil, result: term()}`

  * **Chain execution `:start`**:
    `%{chain_type: String.t(), mode: term(), message_count: integer(), tools_count: integer(), custom_context: term(), call_id: String.t()}`

  * **Chain execution `:stop`** (includes enriched fields):
    `%{chain_type: String.t(), mode: term(), message_count: integer(), tools_count: integer(), custom_context: term(), call_id: String.t(), last_message: Message.t() | nil, token_usage: TokenUsage.t() | nil, result: term()}`

  * **Tool call** (`:start` / `:stop` / `:exception`):
    `%{tool_name: String.t(), tool_call_id: String.t(), async: boolean(), custom_context: term(), call_id: String.t()}`

  ## Usage

  To attach to these events in your application:

  ```elixir
  :telemetry.attach(
    "my-handler-id",
    [:langchain, :llm, :call, :stop],
    &MyApp.handle_llm_call/4,
    nil
  )

  def handle_llm_call(_event_name, measurements, metadata, _config) do
    # Process the event
    IO.inspect(measurements)
    IO.inspect(metadata)
  end
  ```
  """

  @doc """
  Emits a telemetry event with the given name, measurements, and metadata.

  ## Parameters

    * `event_name` - The name of the event as a list of atoms
    * `measurements` - A map of measurements for the event
    * `metadata` - A map of metadata for the event

  ## Examples

      iex> LangChain.Telemetry.emit_event([:langchain, :llm, :call, :start], %{system_time: System.system_time()}, %{model: "gpt-4"})
  """
  @spec emit_event(list(atom()), map(), map()) :: :ok
  def emit_event(event_name, measurements, metadata) do
    :telemetry.execute(event_name, measurements, metadata)
  end

  @doc """
  Emits a start event and returns a function to emit the corresponding stop event.

  This is useful for span-like events where you want to measure the duration of an operation.

  ## Parameters

    * `event_prefix` - The prefix for the event name as a list of atoms
    * `metadata` - A map of metadata for the event

  ## Returns

    A function that accepts additional metadata to be merged with the original metadata
    and emits the stop event with the duration measurement.

  ## Examples

      iex> stop_fun = LangChain.Telemetry.start_event([:langchain, :llm, :call], %{model: "gpt-4"})
      iex> # Do some work
      iex> stop_fun.(%{result: "success"})
  """
  @spec start_event(list(atom()), map()) :: (map() -> :ok)
  def start_event(event_prefix, metadata) do
    start_time = System.monotonic_time()
    start_system_time = System.system_time()

    # Inject a call_id if not already present, so start and stop events share the same ID
    metadata = Map.put_new(metadata, :call_id, Ecto.UUID.generate())

    emit_event(event_prefix ++ [:start], %{system_time: start_system_time}, metadata)

    fn additional_metadata ->
      end_time = System.monotonic_time()
      duration = end_time - start_time

      emit_event(
        event_prefix ++ [:stop],
        %{duration: duration, system_time: System.system_time()},
        Map.merge(metadata, additional_metadata)
      )
    end
  end

  @doc """
  Wraps a function call with start and stop telemetry events.

  ## Parameters

    * `event_prefix` - The prefix for the event name as a list of atoms
    * `metadata` - A map of metadata for the event
    * `fun` - The function to execute
    * `opts` - Optional keyword list:
      * `:enrich_stop` - A 1-arity function that receives the result and returns
        a map of additional metadata to merge into the stop event. Useful for
        extracting data (e.g. token usage) from the result into top-level metadata.

  ## Returns

    The result of the function call.

  ## Examples

      iex> LangChain.Telemetry.span([:langchain, :llm, :call], %{model: "gpt-4"}, fn ->
      ...>   # Call the LLM
      ...>   {:ok, "response"}
      ...> end)

      # With enrich_stop to surface token usage:
      iex> LangChain.Telemetry.span([:langchain, :llm, :call], %{model: "gpt-4"}, fn ->
      ...>   {:ok, response}
      ...> end, enrich_stop: fn {:ok, msg} -> %{token_usage: msg.metadata[:usage]} end)
  """
  @spec span(list(atom()), map(), (-> result), keyword()) :: result when result: any()
  def span(event_prefix, metadata, fun, opts \\ []) do
    # Inject call_id once here so it's shared across start, stop, and exception events.
    # start_event/2 also calls put_new, but since we set it first, the same ID is reused.
    metadata = Map.put_new(metadata, :call_id, Ecto.UUID.generate())

    stop = start_event(event_prefix, metadata)

    try do
      result = fun.()

      additional_metadata =
        case Keyword.get(opts, :enrich_stop) do
          nil ->
            %{result: result}

          enrich_fn when is_function(enrich_fn, 1) ->
            try do
              Map.merge(%{result: result}, enrich_fn.(result))
            rescue
              _ -> %{result: result}
            end
        end

      stop.(additional_metadata)
      result
    rescue
      exception ->
        stacktrace = __STACKTRACE__

        emit_event(
          event_prefix ++ [:exception],
          %{system_time: System.system_time()},
          Map.merge(metadata, %{
            kind: :error,
            error: exception,
            stacktrace: stacktrace
          })
        )

        reraise exception, stacktrace
    end
  end

  # LLM Events

  @doc """
  Emits an LLM call start event.
  """
  @spec llm_call_start(map()) :: (map() -> :ok)
  def llm_call_start(metadata) do
    start_event([:langchain, :llm, :call], metadata)
  end

  @doc """
  Emits an LLM prompt event.
  """
  @spec llm_prompt(map(), map()) :: :ok
  def llm_prompt(measurements, metadata) do
    emit_event([:langchain, :llm, :prompt], measurements, metadata)
  end

  @doc """
  Emits an LLM response event.
  """
  @spec llm_response(map(), map()) :: :ok
  def llm_response(measurements, metadata) do
    emit_event([:langchain, :llm, :response], measurements, metadata)
  end

  # Chain Events

  @doc """
  Emits a chain execution start event.
  """
  @spec chain_execute_start(map()) :: (map() -> :ok)
  def chain_execute_start(metadata) do
    start_event([:langchain, :chain, :execute], metadata)
  end

  # Message Events

  @doc """
  Emits a message processing start event.
  """
  @spec message_process_start(map()) :: (map() -> :ok)
  def message_process_start(metadata) do
    start_event([:langchain, :message, :process], metadata)
  end

  # Tool Events

  @doc """
  Emits a tool call start event.
  """
  @spec tool_call_start(map()) :: (map() -> :ok)
  def tool_call_start(metadata) do
    start_event([:langchain, :tool, :call], metadata)
  end

  @doc """
  Emits a tool call event.
  """
  @spec tool_call(map(), map()) :: :ok
  def tool_call(measurements, metadata) do
    emit_event([:langchain, :tool, :call], measurements, metadata)
  end

  # Memory Events

  @doc """
  Emits a memory read start event.
  """
  @spec memory_read_start(map()) :: (map() -> :ok)
  def memory_read_start(metadata) do
    start_event([:langchain, :memory, :read], metadata)
  end

  @doc """
  Emits a memory write start event.
  """
  @spec memory_write_start(map()) :: (map() -> :ok)
  def memory_write_start(metadata) do
    start_event([:langchain, :memory, :write], metadata)
  end

  # Retriever Events

  @doc """
  Emits a retriever get relevant documents start event.
  """
  @spec retriever_get_relevant_documents_start(map()) :: (map() -> :ok)
  def retriever_get_relevant_documents_start(metadata) do
    start_event([:langchain, :retriever, :get_relevant_documents], metadata)
  end
end
