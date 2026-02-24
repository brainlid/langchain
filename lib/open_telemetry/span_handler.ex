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

    Span hierarchy is automatic: chain → LLM call → tool call. All run synchronously
    in the same process, so parent context propagation works via the process dictionary.

    ## Usage

    This module is used internally by `LangChain.OpenTelemetry.setup/1`. You
    typically don't need to interact with it directly.
    """

    alias LangChain.OpenTelemetry.Attributes
    alias LangChain.OpenTelemetry.Config

    require OpenTelemetry.Tracer, as: Tracer

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
    def handle_event(event, measurements, metadata, config)

    # --- LLM call events ---

    def handle_event(
          [:langchain, :llm, :call, :start],
          _measurements,
          metadata,
          _config
        ) do
      span_name = "chat #{metadata[:model] || "unknown"}"
      attrs = Attributes.llm_call_start(metadata)

      start_span(metadata, span_name, attrs, :client)
    end

    def handle_event(
          [:langchain, :llm, :call, :stop],
          _measurements,
          metadata,
          _config
        ) do
      stop_attrs = Attributes.llm_call_stop(metadata)
      end_span(metadata, stop_attrs)
    end

    def handle_event(
          [:langchain, :llm, :call, :exception],
          _measurements,
          metadata,
          _config
        ) do
      end_span_on_exception(metadata)
    end

    # --- Chain execute events ---

    def handle_event(
          [:langchain, :chain, :execute, :start],
          _measurements,
          metadata,
          _config
        ) do
      chain_type = metadata[:chain_type] || "unknown"
      span_name = "invoke_agent #{chain_type}"
      attrs = Attributes.chain_start(metadata)

      start_span(metadata, span_name, attrs, :internal)
    end

    def handle_event(
          [:langchain, :chain, :execute, :stop],
          _measurements,
          metadata,
          _config
        ) do
      end_span(metadata, [])
    end

    def handle_event(
          [:langchain, :chain, :execute, :exception],
          _measurements,
          metadata,
          _config
        ) do
      end_span_on_exception(metadata)
    end

    # --- Tool call events ---

    def handle_event(
          [:langchain, :tool, :call, :start],
          _measurements,
          metadata,
          _config
        ) do
      tool_name = metadata[:tool_name] || "unknown"
      span_name = "execute_tool #{tool_name}"
      attrs = Attributes.tool_call(metadata)

      start_span(metadata, span_name, attrs, :internal)
    end

    def handle_event(
          [:langchain, :tool, :call, :stop],
          _measurements,
          metadata,
          _config
        ) do
      end_span(metadata, [])
    end

    def handle_event(
          [:langchain, :tool, :call, :exception],
          _measurements,
          metadata,
          _config
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

    defp pop_span(nil), do: nil

    defp pop_span(call_id) do
      Process.delete({__MODULE__, call_id})
    end
  end
end
