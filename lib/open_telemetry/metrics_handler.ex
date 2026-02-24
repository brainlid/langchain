if Code.ensure_loaded?(:opentelemetry) do
  defmodule LangChain.OpenTelemetry.MetricsHandler do
    @moduledoc """
    Telemetry handler that re-emits LangChain telemetry events as GenAI
    Semantic Convention-aligned metric events.

    Emits the following telemetry events:

    * `[:langchain, :otel, :operation, :duration]` — with `%{duration_s: float()}`
      measurement and GenAI attributes as metadata
    * `[:langchain, :otel, :token, :usage]` — with `%{tokens: integer()}` measurement
      and GenAI attributes (including `gen_ai.token.type`) as metadata

    These events can be consumed by any `:telemetry`-based metrics library
    (e.g., `Telemetry.Metrics`, `PromEx`, `OpenTelemetry.Metrics` when available).

    ## Usage

    This module is used internally by `LangChain.OpenTelemetry.setup/1` when
    `enable_metrics: true` (the default). You typically don't need to interact
    with it directly.
    """

    alias LangChain.OpenTelemetry.ProviderMapping

    @handler_prefix "langchain-otel-metrics"

    @doc """
    Returns the list of telemetry events this handler attaches to.
    """
    @spec events() :: [list(atom())]
    def events do
      [
        [:langchain, :llm, :call, :stop],
        [:langchain, :chain, :execute, :stop],
        [:langchain, :tool, :call, :stop]
      ]
    end

    @doc """
    Returns the telemetry handler ID prefix used for attaching/detaching.
    """
    @spec handler_id() :: String.t()
    def handler_id, do: @handler_prefix

    @doc """
    Telemetry handler callback. Re-emits duration and token usage metric events.
    """
    @spec handle_event(list(atom()), map(), map(), term()) :: :ok
    def handle_event(event, measurements, metadata, _config)

    def handle_event(
          [:langchain, :llm, :call, :stop],
          measurements,
          metadata,
          _config
        ) do
      common_attrs = common_attributes("chat", metadata)
      emit_duration(measurements, common_attrs)
      emit_token_usage(metadata, common_attrs)
      :ok
    end

    def handle_event(
          [:langchain, :chain, :execute, :stop],
          measurements,
          metadata,
          _config
        ) do
      common_attrs = common_attributes("invoke_agent", metadata)
      emit_duration(measurements, common_attrs)
      :ok
    end

    def handle_event(
          [:langchain, :tool, :call, :stop],
          measurements,
          metadata,
          _config
        ) do
      common_attrs = common_attributes("execute_tool", metadata)
      emit_duration(measurements, common_attrs)
      :ok
    end

    defp emit_duration(measurements, attrs) do
      case measurements[:duration] do
        nil ->
          :ok

        duration_native ->
          duration_s =
            System.convert_time_unit(duration_native, :native, :microsecond) / 1_000_000

          :telemetry.execute(
            [:langchain, :otel, :operation, :duration],
            %{duration_s: duration_s},
            attrs
          )
      end
    end

    defp emit_token_usage(metadata, common_attrs) do
      case metadata[:token_usage] do
        %{input: input, output: output} ->
          if input do
            attrs = Map.put(common_attrs, "gen_ai.token.type", "input")

            :telemetry.execute(
              [:langchain, :otel, :token, :usage],
              %{tokens: input},
              attrs
            )
          end

          if output do
            attrs = Map.put(common_attrs, "gen_ai.token.type", "output")

            :telemetry.execute(
              [:langchain, :otel, :token, :usage],
              %{tokens: output},
              attrs
            )
          end

          :ok

        _ ->
          :ok
      end
    end

    defp common_attributes(operation_name, metadata) do
      attrs = %{"gen_ai.operation.name" => operation_name}

      attrs =
        case metadata[:provider] do
          nil -> attrs
          provider -> Map.put(attrs, "gen_ai.provider.name", ProviderMapping.to_otel(provider))
        end

      case metadata[:model] do
        nil -> attrs
        model -> Map.put(attrs, "gen_ai.request.model", model)
      end
    end
  end
end
