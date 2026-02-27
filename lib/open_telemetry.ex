if Code.ensure_loaded?(:opentelemetry) do
  defmodule LangChain.OpenTelemetry do
    @moduledoc """
    OpenTelemetry integration for LangChain.

    Attaches to LangChain's `:telemetry` events and translates them into
    OpenTelemetry spans and metrics following the
    [GenAI Semantic Conventions](https://opentelemetry.io/docs/specs/semconv/gen-ai/) (v1.40+).

    ## Setup

    Add the optional dependencies to your `mix.exs`:

        {:opentelemetry_api, "~> 1.4"},
        {:opentelemetry, "~> 1.5"},
        {:opentelemetry_exporter, "~> 1.8"}

    Then call `setup/1` in your application startup (e.g., `Application.start/2`):

        LangChain.OpenTelemetry.setup()

    ## Options

    Options are passed through to `LangChain.OpenTelemetry.Config`:

      * `:capture_input_messages` - Record input messages as span attributes. Default: `false`.
        When enabled, serializes `LangChain.Message` structs sent to the LLM into
        `gen_ai.input.messages` as a JSON string. **May contain sensitive data.**
      * `:capture_output_messages` - Record output messages as span attributes. Default: `false`.
        When enabled, serializes the LLM response into `gen_ai.output.messages`.
        **May contain sensitive data.**
      * `:capture_tool_arguments` - Record tool call arguments. Default: `false`.
        Serializes tool call arguments into `gen_ai.tool.call.arguments`.
      * `:capture_tool_results` - Record tool call results. Default: `false`.
        Captures tool return values into `gen_ai.tool.call.result`.
      * `:enable_metrics` - Record OTel histogram metrics. Default: `true`.

    ## Span Mapping

    | Telemetry Event                       | OTel Span Name              | Kind       |
    |---------------------------------------|-----------------------------|------------|
    | `[:langchain, :chain, :execute, ...]` | `invoke_agent llm_chain`    | `:internal` |
    | `[:langchain, :llm, :call, ...]`      | `chat {model}`              | `:client`  |
    | `[:langchain, :tool, :call, ...]`     | `execute_tool {tool_name}`  | `:internal` |

    ## Langfuse Integration

    [Langfuse](https://langfuse.com/) can ingest OpenTelemetry traces via its
    OTLP-compatible endpoint. Configure the OTel exporter to point at your
    Langfuse instance:

        # config/runtime.exs
        config :opentelemetry_exporter,
          otlp_protocol: :http_protobuf,
          otlp_endpoint: "https://your-langfuse-host/api/public/otel",
          otlp_headers: [
            {"Authorization", "Basic " <> Base.encode64("pk-lf-...:sk-lf-...")}
          ]

    ### Langfuse-specific attributes via `custom_context`

    Set `custom_context` on your `LLMChain` to propagate Langfuse trace attributes:

        chain =
          %{llm: llm, messages: messages}
          |> LLMChain.new!()
          |> Map.put(:custom_context, %{
            langfuse_user_id: current_user.id,
            langfuse_session_id: session_id,
            langfuse_tags: ["production", "v2"],
            langfuse_metadata: %{env: "prod", feature: "chat"}
          })

    These are mapped to span attributes on the root chain span:

    | `custom_context` key    | Span attribute                     |
    |-------------------------|------------------------------------|
    | `:langfuse_trace_name`  | `langfuse.trace.name`              |
    | `:langfuse_user_id`     | `langfuse.user.id`                 |
    | `:langfuse_session_id`  | `langfuse.session.id`              |
    | `:langfuse_tags`        | `langfuse.trace.tags`              |
    | `:langfuse_metadata`    | `langfuse.trace.metadata.*`        |
    """

    alias LangChain.OpenTelemetry.Config
    alias LangChain.OpenTelemetry.SpanHandler
    alias LangChain.OpenTelemetry.MetricsHandler

    @doc """
    Attaches OpenTelemetry handlers to LangChain telemetry events.

    ## Examples

        LangChain.OpenTelemetry.setup()
        LangChain.OpenTelemetry.setup(capture_input_messages: true, enable_metrics: false)
    """
    @spec setup(keyword()) :: :ok
    def setup(opts \\ []) do
      config = Config.new(opts)

      :telemetry.attach_many(
        SpanHandler.handler_id(),
        SpanHandler.events(),
        &SpanHandler.handle_event/4,
        config
      )

      if config.enable_metrics do
        :telemetry.attach_many(
          MetricsHandler.handler_id(),
          MetricsHandler.events(),
          &MetricsHandler.handle_event/4,
          config
        )
      end

      :ok
    end

    @doc """
    Executes a function with OpenTelemetry tracing suppressed.

    Any LangChain operations inside the function will NOT create OTEL spans or traces.
    Useful for lightweight utility chains (translation, topic generation) that should
    not appear as separate traces.

    ## Example

        LangChain.OpenTelemetry.without_tracing(fn ->
          LLM.chain() |> LLMChain.run()
        end)
    """
    @spec without_tracing((-> result)) :: result when result: any()
    def without_tracing(fun) when is_function(fun, 0) do
      # Create a non-recording span context and attach it.
      # Child spans inherit the non-recording flag and are silently dropped
      # by the SDK, so no traces are exported for operations inside this block.
      trace_id = apply(:otel_id_generator, :generate_trace_id, [])
      span_id = apply(:otel_id_generator, :generate_span_id, [])
      span_ctx = :otel_tracer.non_recording_span(trace_id, span_id, 0)

      ctx = OpenTelemetry.Ctx.get_current()
      new_ctx = OpenTelemetry.Tracer.set_current_span(ctx, span_ctx)
      token = OpenTelemetry.Ctx.attach(new_ctx)

      try do
        fun.()
      after
        OpenTelemetry.Ctx.detach(token)
      end
    end

    @doc """
    Detaches all OpenTelemetry handlers from LangChain telemetry events.
    """
    @spec teardown() :: :ok
    def teardown do
      for id <- [SpanHandler.handler_id(), MetricsHandler.handler_id()] do
        case :telemetry.detach(id) do
          :ok -> :ok
          {:error, :not_found} -> :ok
        end
      end

      :ok
    end
  end
end
