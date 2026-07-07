defmodule LangChain.OpenTelemetry.WithoutTracingTest do
  use ExUnit.Case, async: false

  require Record

  Record.defrecord(
    :span,
    Record.extract(:span, from_lib: "opentelemetry/include/otel_span.hrl")
  )

  alias LangChain.OpenTelemetry

  setup do
    :application.stop(:opentelemetry)

    :application.set_env(:opentelemetry, :processors, [
      {:otel_batch_processor, %{scheduled_delay_ms: 1}}
    ])

    {:ok, _} = :application.ensure_all_started(:opentelemetry)

    tid = :ets.new(:test_spans, [:bag, :public])
    :otel_batch_processor.set_exporter(:otel_exporter_tab, tid)

    OpenTelemetry.setup(capture_input_messages: true, enable_metrics: false)

    on_exit(fn ->
      OpenTelemetry.teardown()

      try do
        :ets.delete(tid)
      rescue
        _ -> :ok
      end
    end)

    %{tid: tid}
  end

  defp flush_spans(tid) do
    Process.sleep(50)

    :otel_batch_processor.force_flush(%{
      reg_name: {:via, :gproc, {:n, :l, {:otel_batch_processor, :global}}}
    })

    Process.sleep(50)

    :ets.tab2list(tid)
    |> Enum.map(fn record -> span(record, :name) end)
  end

  defp emit_llm_span(call_id) do
    :telemetry.execute(
      [:langchain, :llm, :call, :start],
      %{system_time: System.system_time()},
      %{call_id: call_id, model: "gpt-4o", provider: "openai"}
    )

    :telemetry.execute(
      [:langchain, :llm, :call, :stop],
      %{duration: System.convert_time_unit(1, :second, :native)},
      %{call_id: call_id, model: "gpt-4o", provider: "openai"}
    )
  end

  describe "without_tracing/1" do
    test "returns the function's value" do
      assert OpenTelemetry.without_tracing(fn -> :result end) == :result
    end

    test "suppresses spans for operations inside the block", %{tid: tid} do
      OpenTelemetry.without_tracing(fn ->
        emit_llm_span(Ecto.UUID.generate())
      end)

      assert flush_spans(tid) == []
    end

    test "tracing resumes after the block returns", %{tid: tid} do
      OpenTelemetry.without_tracing(fn ->
        emit_llm_span(Ecto.UUID.generate())
      end)

      emit_llm_span(Ecto.UUID.generate())

      assert "chat gpt-4o" in flush_spans(tid)
    end

    test "restores the previous context even if the function raises", %{tid: tid} do
      assert_raise RuntimeError, fn ->
        OpenTelemetry.without_tracing(fn -> raise "boom" end)
      end

      # Spans created after a raising block are still recorded.
      emit_llm_span(Ecto.UUID.generate())
      assert "chat gpt-4o" in flush_spans(tid)
    end
  end

  describe "prompt event with no active span" do
    test "is a no-op and does not raise when no LLM span is active" do
      # `:prompt` fired with capture enabled but outside any LLM call span.
      assert :ok =
               :telemetry.execute(
                 [:langchain, :llm, :prompt],
                 %{},
                 %{messages: [LangChain.Message.new_user!("hello")]}
               )
    end
  end

  describe "metrics suppression inside without_tracing/1" do
    setup do
      test_pid = self()
      handler_id = "wt-metrics-#{System.unique_integer([:positive])}"

      :telemetry.attach_many(
        handler_id,
        [
          [:langchain, :otel, :operation, :duration],
          [:langchain, :otel, :token, :usage]
        ],
        fn event, measurements, metadata, _ ->
          send(test_pid, {:metric, event, measurements, metadata})
        end,
        nil
      )

      on_exit(fn -> :telemetry.detach(handler_id) end)
      :ok
    end

    defp emit_llm_metric do
      LangChain.OpenTelemetry.MetricsHandler.handle_event(
        [:langchain, :llm, :call, :stop],
        %{duration: System.convert_time_unit(1, :second, :native)},
        %{provider: "openai", model: "gpt-4o", token_usage: %{input: 1, output: 1}},
        nil
      )
    end

    test "emits metric events outside the block (baseline)" do
      emit_llm_metric()
      assert_received {:metric, [:langchain, :otel, :operation, :duration], _, _}
    end

    test "does NOT emit metric events for operations inside the block" do
      OpenTelemetry.without_tracing(fn -> emit_llm_metric() end)

      refute_received {:metric, [:langchain, :otel, :operation, :duration], _, _}
      refute_received {:metric, [:langchain, :otel, :token, :usage], _, _}
    end

    test "metrics resume after the block returns" do
      OpenTelemetry.without_tracing(fn -> emit_llm_metric() end)
      emit_llm_metric()

      assert_received {:metric, [:langchain, :otel, :operation, :duration], _, _}
    end
  end
end
