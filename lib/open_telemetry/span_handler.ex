if Code.ensure_loaded?(:opentelemetry) do
  defmodule LangChain.OpenTelemetry.SpanHandler do
    @moduledoc """
    Telemetry handler that creates OpenTelemetry spans from LangChain telemetry events.

    Spans follow the GenAI Semantic Conventions (v1.40+):

    | Telemetry Event                       | OTel Span Name              | Kind       | `gen_ai.operation.name` |
    |---------------------------------------|-----------------------------|------------|------------------------|
    | `[:langchain, :chain, :execute, ...]` | `invoke_agent llm_chain`    | `:internal` | `invoke_agent`          |
    | `[:langchain, :llm, :call, ...]`      | `chat {model}`              | `:client`  | `chat`                 |
    | `[:langchain, :tool, :call, ...]`     | `execute_tool {tool_name}`  | `:internal` | `execute_tool`          |

    Span hierarchy is automatic for synchronous work: chain → LLM call → tool call
    run in the same process, so parent context propagation works via the process
    dictionary.

    > #### Async tools {: .warning}
    >
    > Tools declared with `async: true` execute in a separate `Task` process. The
    > OpenTelemetry context lives in the parent process's dictionary and is **not**
    > inherited by the spawned process, so an async tool's span would otherwise be
    > orphaned (it becomes its own root span rather than a child of the chain span).
    > To keep async tool spans attached, propagate the parent context into the Task
    > via the `:on_tool_pre_execution` callback (which fires inside the spawned
    > process) — capture `OpenTelemetry.Ctx.get_current/0` before execution and
    > `OpenTelemetry.Ctx.attach/1` inside the callback.

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
      stop_attrs = Attributes.llm_call_stop(metadata, config)
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
      stop_attrs = Attributes.chain_stop(metadata, config)
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
      stop_attrs = Attributes.tool_call_stop(metadata, config)
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
          if extra_attributes != [] do
            OpenTelemetry.Span.set_attributes(span_ctx, extra_attributes)
          end

          OpenTelemetry.Span.end_span(span_ctx)
          OpenTelemetry.Ctx.detach(token)
          :ok

        nil ->
          :ok
      end
    end

    defp end_span_on_exception(metadata) do
      call_id = metadata[:call_id]

      case pop_span(call_id) do
        {span_ctx, token} ->
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

          OpenTelemetry.Span.end_span(span_ctx)
          OpenTelemetry.Ctx.detach(token)
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
