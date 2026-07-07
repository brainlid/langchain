# Guarded on the `:opentelemetry` module from the `opentelemetry_api` optional
# dep (see `LangChain.OpenTelemetry` for the full rationale) — not the SDK app.
if Code.ensure_loaded?(:opentelemetry) do
  defmodule LangChain.OpenTelemetry.SpanHandler do
    @moduledoc """
    Telemetry handler that creates OpenTelemetry spans from LangChain telemetry events.

    Spans follow a subset of the GenAI Semantic Conventions (v1.40+) — see
    `LangChain.OpenTelemetry.Attributes` for the attributes that are emitted:

    | Telemetry Event                       | OTel Span Name              | Kind       | `gen_ai.operation.name` |
    |---------------------------------------|-----------------------------|------------|------------------------|
    | `[:langchain, :chain, :execute, ...]` | `invoke_agent llm_chain`    | `:internal` | `invoke_agent`          |
    | `[:langchain, :llm, :call, ...]`      | `chat {model}`              | `:client`  | `chat`                 |
    | `[:langchain, :tool, :call, ...]`     | `execute_tool {tool_name}`  | `:internal` | `execute_tool`          |

    Span hierarchy is automatic for synchronous work: chain → LLM call → tool call
    run in the same process, so parent context propagation works via the process
    dictionary.

    > #### Async tools {: .info}
    >
    > Tools declared with `async: true` execute in a separate `Task` process, and
    > the OpenTelemetry context is **not** inherited across a process boundary. For
    > tools run by `LLMChain`'s built-in executor this is handled automatically —
    > the chain captures the current context before spawning each async tool and
    > re-attaches it inside the `Task`, so async tool spans nest under the chain
    > span. If you spawn your own processes running LangChain operations, do the
    > same yourself: capture `OpenTelemetry.Ctx.get_current/0` before spawning and
    > `OpenTelemetry.Ctx.attach/1` inside the process (e.g. via the
    > `:on_tool_pre_execution` callback).

    ## Usage

    This module is used internally by `LangChain.OpenTelemetry.setup/1`. You
    typically don't need to interact with it directly.
    """

    alias LangChain.OpenTelemetry.Attributes
    alias LangChain.OpenTelemetry.Config
    alias LangChain.OpenTelemetry.MessageSerializer

    require OpenTelemetry.Tracer, as: Tracer
    require Logger

    @handler_prefix "langchain-otel-span"

    @doc """
    Returns the list of telemetry events this handler attaches to.
    """
    @spec events() :: [list(atom())]
    def events do
      [
        [:langchain, :llm, :call, :start],
        [:langchain, :llm, :call, :stop],
        [:langchain, :llm, :call, :exception],
        [:langchain, :llm, :prompt],
        [:langchain, :llm, :stream, :first_token],
        [:langchain, :chain, :execute, :start],
        [:langchain, :chain, :execute, :stop],
        [:langchain, :chain, :execute, :exception],
        [:langchain, :tool, :call, :start],
        [:langchain, :tool, :call, :stop],
        [:langchain, :tool, :call, :exception]
      ]
    end

    @doc """
    Returns the telemetry handler ID prefix used for attaching/detaching.
    """
    @spec handler_id() :: String.t()
    def handler_id, do: @handler_prefix

    @doc """
    Telemetry handler callback. Dispatches to the appropriate handler based on the event.
    """
    @spec handle_event(list(atom()), map(), map(), Config.t()) :: :ok
    def handle_event(event, measurements, metadata, config) do
      # `:telemetry` permanently detaches a handler that raises (VM-wide, for the
      # rest of the run). A single bad payload — e.g. a non-JSON-encodable message
      # reaching the serializer — must never silently disable tracing for every
      # subsequent request, so we trap and log instead.
      do_handle_event(event, measurements, metadata, config)
    rescue
      exception ->
        Logger.warning(fn ->
          "[LangChain.OpenTelemetry] span handler failed for #{inspect(event)} and was " <>
            "skipped (tracing remains attached): " <>
            Exception.format(:error, exception, __STACKTRACE__)
        end)

        :ok
    end

    defp do_handle_event(event, measurements, metadata, config)

    # --- LLM call events ---

    defp do_handle_event(
           [:langchain, :llm, :call, :start],
           _measurements,
           metadata,
           %Config{} = config
         ) do
      span_name = "chat #{metadata[:model] || "unknown"}"
      attrs = Attributes.llm_call_start(metadata, config)

      start_span(metadata, span_name, attrs, :client)
    end

    defp do_handle_event(
           [:langchain, :llm, :call, :stop],
           _measurements,
           metadata,
           %Config{} = config
         ) do
      stop_attrs =
        safe_stop_attributes([:langchain, :llm, :call, :stop], fn ->
          Attributes.llm_call_stop(metadata, config)
        end)

      end_span(metadata, stop_attrs)
    end

    defp do_handle_event(
           [:langchain, :llm, :call, :exception],
           _measurements,
           metadata,
           %Config{}
         ) do
      end_span_on_exception(metadata)
    end

    # --- LLM prompt event (opt-in message capture) ---

    defp do_handle_event(
           [:langchain, :llm, :prompt],
           _measurements,
           metadata,
           %Config{} = config
         ) do
      if config.capture_input_messages do
        case {OpenTelemetry.Tracer.current_span_ctx(), metadata[:messages]} do
          # No active span to attach to (e.g. the `:prompt` event fired outside
          # an LLM call span). Nothing to record.
          {:undefined, _} ->
            :ok

          {span_ctx, [_ | _] = messages} ->
            OpenTelemetry.Span.set_attributes(span_ctx, [
              {"gen_ai.input.messages", MessageSerializer.serialize_input(messages)}
            ])

          _ ->
            :ok
        end
      end

      :ok
    end

    # --- LLM streaming: time-to-first-token ---

    defp do_handle_event(
           [:langchain, :llm, :stream, :first_token],
           measurements,
           _metadata,
           %Config{}
         ) do
      record_first_token(measurements[:duration])
    end

    # --- Chain execute events ---

    defp do_handle_event(
           [:langchain, :chain, :execute, :start],
           _measurements,
           metadata,
           %Config{}
         ) do
      chain_type = metadata[:chain_type] || "unknown"
      span_name = "invoke_agent #{chain_type}"
      attrs = Attributes.chain_start(metadata)

      start_span(metadata, span_name, attrs, :internal)
    end

    defp do_handle_event(
           [:langchain, :chain, :execute, :stop],
           _measurements,
           metadata,
           %Config{} = config
         ) do
      stop_attrs =
        safe_stop_attributes([:langchain, :chain, :execute, :stop], fn ->
          Attributes.chain_stop(metadata, config)
        end)

      end_span(metadata, stop_attrs)
    end

    defp do_handle_event(
           [:langchain, :chain, :execute, :exception],
           _measurements,
           metadata,
           %Config{}
         ) do
      end_span_on_exception(metadata)
    end

    # --- Tool call events ---

    defp do_handle_event(
           [:langchain, :tool, :call, :start],
           _measurements,
           metadata,
           %Config{} = config
         ) do
      tool_name = metadata[:tool_name] || "unknown"
      span_name = "execute_tool #{tool_name}"
      attrs = Attributes.tool_call(metadata, config)

      start_span(metadata, span_name, attrs, :internal)
    end

    defp do_handle_event(
           [:langchain, :tool, :call, :stop],
           _measurements,
           metadata,
           %Config{} = config
         ) do
      stop_attrs =
        safe_stop_attributes([:langchain, :tool, :call, :stop], fn ->
          Attributes.tool_call_stop(metadata, config)
        end)

      end_span(metadata, stop_attrs)
    end

    defp do_handle_event(
           [:langchain, :tool, :call, :exception],
           _measurements,
           metadata,
           %Config{}
         ) do
      end_span_on_exception(metadata)
    end

    # --- Private helpers ---

    # Records time-to-first-token on the active LLM span. Streaming decode runs in
    # the same process as the `chat` span, so the current span context IS that
    # span. Adds a timestamped span event (backends can place it on the trace
    # timeline) plus a numeric `gen_ai.response.time_to_first_token` attribute in
    # seconds. A non-recording current span (e.g. inside `without_tracing/1`) makes
    # both calls no-ops at the SDK level, so this stays correct there too.
    defp record_first_token(nil), do: :ok

    defp record_first_token(duration_native) do
      case OpenTelemetry.Tracer.current_span_ctx() do
        :undefined ->
          :ok

        span_ctx ->
          seconds =
            System.convert_time_unit(duration_native, :native, :microsecond) / 1_000_000

          OpenTelemetry.Span.add_event(span_ctx, "gen_ai.first_token", %{})

          OpenTelemetry.Span.set_attributes(span_ctx, [
            {"gen_ai.response.time_to_first_token", seconds}
          ])

          :ok
      end
    end

    defp start_span(metadata, span_name, attributes, kind) do
      call_id = metadata[:call_id]

      parent_ctx = OpenTelemetry.Ctx.get_current()

      span_ctx =
        Tracer.start_span(span_name, %{
          kind: kind,
          attributes: attributes
        })

      new_ctx = OpenTelemetry.Tracer.set_current_span(parent_ctx, span_ctx)
      token = OpenTelemetry.Ctx.attach(new_ctx)

      if call_id do
        Process.put({__MODULE__, call_id}, {span_ctx, token})
      end

      :ok
    end

    defp end_span(metadata, extra_attributes) do
      call_id = metadata[:call_id]

      case pop_span(call_id) do
        {span_ctx, token} ->
          # Ending the span and detaching its context MUST happen even if
          # `set_attributes` raises — otherwise the span leaks and every later
          # span in this process nests under the never-closed one. Setting the
          # attributes is best-effort; closing the span is not.
          try do
            if extra_attributes != [] do
              OpenTelemetry.Span.set_attributes(span_ctx, extra_attributes)
            end
          after
            OpenTelemetry.Span.end_span(span_ctx)
            OpenTelemetry.Ctx.detach(token)
          end

          :ok

        nil ->
          :ok
      end
    end

    # Build a stop event's attributes without letting a failure abort the span's
    # lifecycle. Attribute construction serializes message content (when the
    # `capture_*_messages` flags are on), and serialization can raise on a
    # non-JSON-encodable payload — e.g. an invalid-UTF-8 binary in a model or
    # tool response. If that happened before `end_span/2`, the span would never
    # be ended and its context never detached, silently corrupting the parent
    # hierarchy of every subsequent trace in this (often long-lived) process. So
    # trap here and fall back to no extra attributes: the span still closes and
    # at most the enrichment attributes are dropped for this one event.
    defp safe_stop_attributes(event, fun) do
      fun.()
    rescue
      exception ->
        Logger.warning(fn ->
          "[LangChain.OpenTelemetry] failed to build stop attributes for #{inspect(event)}; " <>
            "ending the span without them: " <>
            Exception.format(:error, exception, __STACKTRACE__)
        end)

        []
    end

    defp end_span_on_exception(metadata) do
      call_id = metadata[:call_id]

      case pop_span(call_id) do
        {span_ctx, token} ->
          # Ending the span and detaching its context MUST happen even if setting
          # the error status/attributes raises (e.g. a custom exception whose
          # `Exception.message/1` blows up) — otherwise the span leaks and every
          # later span in this (often long-lived) process nests under the
          # never-closed one. Recording the error is best-effort; closing is not.
          # Mirrors `end_span/2`.
          try do
            error = metadata[:error]

            status_message =
              if error, do: Exception.message(error), else: "exception"

            OpenTelemetry.Span.set_status(span_ctx, OpenTelemetry.status(:error, status_message))

            # `error.type` lets GenAI-semconv backends (Langfuse, etc.) group and
            # filter errors by kind. Use the exception's module name when available.
            OpenTelemetry.Span.set_attributes(span_ctx, [{"error.type", error_type(error)}])

            if error do
              OpenTelemetry.Span.record_exception(
                span_ctx,
                error,
                metadata[:stacktrace] || []
              )
            end
          after
            OpenTelemetry.Span.end_span(span_ctx)
            OpenTelemetry.Ctx.detach(token)
          end

          :ok

        nil ->
          :ok
      end
    end

    defp error_type(%module{}), do: inspect(module)
    defp error_type(_), do: "error"

    defp pop_span(nil), do: nil

    defp pop_span(call_id) do
      Process.delete({__MODULE__, call_id})
    end
  end
end
