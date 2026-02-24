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
      * `:capture_output_messages` - Record output messages as span attributes. Default: `false`.
      * `:capture_tool_arguments` - Record tool call arguments. Default: `false`.
      * `:capture_tool_results` - Record tool call results. Default: `false`.
      * `:enable_metrics` - Record OTel histogram metrics. Default: `true`.

    ## Span Mapping

    | Telemetry Event                       | OTel Span Name              | Kind       |
    |---------------------------------------|-----------------------------|------------|
    | `[:langchain, :chain, :execute, ...]` | `invoke_agent llm_chain`    | `:internal` |
    | `[:langchain, :llm, :call, ...]`      | `chat {model}`              | `:client`  |
    | `[:langchain, :tool, :call, ...]`     | `execute_tool {tool_name}`  | `:internal` |
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
    Detaches all OpenTelemetry handlers from LangChain telemetry events.
    """
    @spec teardown() :: :ok
    def teardown do
      :telemetry.detach(SpanHandler.handler_id())
      :telemetry.detach(MetricsHandler.handler_id())
      :ok
    rescue
      # detach raises if handler not found; ignore
      _ -> :ok
    end
  end
end
