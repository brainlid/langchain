# Guarded on the `:opentelemetry` module from the `opentelemetry_api` optional
# dep (see `LangChain.OpenTelemetry` for the full rationale) — not the SDK app.
if Code.ensure_loaded?(:opentelemetry) do
  defmodule LangChain.OpenTelemetry.MetricsHandler do
    @moduledoc """
    Telemetry handler that re-emits LangChain telemetry events as GenAI
    Semantic Convention-aligned intermediary metric events.

    **Important:** This module does not directly record OpenTelemetry histograms
    or counters. It emits `:telemetry.execute/3` events that must be consumed by
    a metrics library to become actual OTel metrics. Without a consumer attached
    to these events, `enable_metrics: true` has no visible effect.

    To produce actual OTel metrics, attach a consumer such as `Telemetry.Metrics`
    with an OpenTelemetry reporter, `PromEx`, or equivalent to the events below.

    ## Emitted events

    * `[:langchain, :otel, :operation, :duration]` — with `%{duration_s: float()}`
      measurement and GenAI attributes as metadata. Emitted for both successful
      (`:stop`) and failed (`:exception`) operations; failures additionally carry
      an `error.type` attribute so error rate is observable alongside latency.
    * `[:langchain, :otel, :token, :usage]` — with `%{tokens: integer()}` measurement
      and GenAI attributes (including `gen_ai.token.type`) as metadata
    * `[:langchain, :otel, :operation, :time_to_first_token]` — with
      `%{duration_s: float()}` measurement and GenAI attributes as metadata.
      Emitted once per streaming LLM call, measuring the time from request start to
      the first streamed chunk (aligns with the semantic-convention
      `gen_ai.server.time_to_first_token` metric).

    ## Usage

    This module is used internally by `LangChain.OpenTelemetry.setup/1` when
    `enable_metrics: true` (the default). You typically don't need to interact
    with it directly.
    """

    alias LangChain.OpenTelemetry.ProviderMapping

    require Logger

    @handler_prefix "langchain-otel-metrics"

    @doc """
    Returns the list of telemetry events this handler attaches to.
    """
    @spec events() :: [list(atom())]
    def events do
      [
        [:langchain, :llm, :call, :stop],
        [:langchain, :llm, :call, :exception],
        [:langchain, :llm, :stream, :first_token],
        [:langchain, :chain, :execute, :stop],
        [:langchain, :chain, :execute, :exception],
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
    Telemetry handler callback. Re-emits duration and token usage metric events.
    """
    @spec handle_event(list(atom()), map(), map(), term()) :: :ok
    def handle_event(event, measurements, metadata, config) do
      # Like `SpanHandler`: `:telemetry` permanently detaches a handler that
      # raises (VM-wide, for the rest of the run). A single bad payload must never
      # silently disable metrics for every subsequent request, so we trap and log.
      #
      # Skip operations running inside `without_tracing/1`: spans are dropped by
      # the SDK's non-recording context, but this handler has no span context, so
      # it needs the explicit flag to stay consistent (no metrics for utility
      # chains either).
      if LangChain.OpenTelemetry.telemetry_suppressed?() do
        :ok
      else
        do_handle_event(event, measurements, metadata, config)
      end
    rescue
      exception ->
        Logger.warning(fn ->
          "[LangChain.OpenTelemetry] metrics handler failed for #{inspect(event)} and was " <>
            "skipped (metrics remain attached): " <>
            Exception.format(:error, exception, __STACKTRACE__)
        end)

        :ok
    end

    defp do_handle_event(event, measurements, metadata, config)

    defp do_handle_event(
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

    defp do_handle_event(
           [:langchain, :chain, :execute, :stop],
           measurements,
           metadata,
           _config
         ) do
      common_attrs = common_attributes("invoke_agent", metadata)
      emit_duration(measurements, common_attrs)
      :ok
    end

    defp do_handle_event(
           [:langchain, :tool, :call, :stop],
           measurements,
           metadata,
           _config
         ) do
      common_attrs = common_attributes("execute_tool", metadata)
      emit_duration(measurements, common_attrs)
      :ok
    end

    defp do_handle_event(
           [:langchain, :llm, :stream, :first_token],
           measurements,
           metadata,
           _config
         ) do
      case measurements[:duration] do
        nil ->
          :ok

        duration_native ->
          seconds =
            System.convert_time_unit(duration_native, :native, :microsecond) / 1_000_000

          :telemetry.execute(
            [:langchain, :otel, :operation, :time_to_first_token],
            %{duration_s: seconds},
            common_attributes("chat", metadata)
          )
      end

      :ok
    end

    # Failed operations: emit a duration metric tagged with `error.type` so error
    # rate and error latency are observable alongside successes. No token usage is
    # available on a failure.
    defp do_handle_event(
           [:langchain, component, operation, :exception],
           measurements,
           metadata,
           _config
         ) do
      operation_name = operation_name_for(component, operation)

      attrs =
        operation_name
        |> common_attributes(metadata)
        |> Map.put("error.type", error_type(metadata[:error]))

      emit_duration(measurements, attrs)
      :ok
    end

    defp operation_name_for(:llm, :call), do: "chat"
    defp operation_name_for(:chain, :execute), do: "invoke_agent"
    defp operation_name_for(:tool, :call), do: "execute_tool"

    defp error_type(%module{}), do: inspect(module)
    defp error_type(_), do: "error"

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
