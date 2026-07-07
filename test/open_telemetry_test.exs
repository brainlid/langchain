defmodule LangChain.OpenTelemetryTest do
  use ExUnit.Case

  alias LangChain.OpenTelemetry

  describe "setup/1 and teardown/0" do
    test "attaches and detaches telemetry handlers" do
      OpenTelemetry.setup()

      handlers = :telemetry.list_handlers([:langchain, :llm, :call, :start])
      span_handler = Enum.find(handlers, &(&1.id == "langchain-otel-span"))
      assert span_handler != nil

      metrics_handlers = :telemetry.list_handlers([:langchain, :llm, :call, :stop])
      metrics_handler = Enum.find(metrics_handlers, &(&1.id == "langchain-otel-metrics"))
      assert metrics_handler != nil

      OpenTelemetry.teardown()

      handlers_after = :telemetry.list_handlers([:langchain, :llm, :call, :start])
      assert Enum.find(handlers_after, &(&1.id == "langchain-otel-span")) == nil
    end

    test "setup with enable_metrics: false skips metrics handler" do
      OpenTelemetry.setup(enable_metrics: false)

      handlers = :telemetry.list_handlers([:langchain, :llm, :call, :stop])
      metrics_handler = Enum.find(handlers, &(&1.id == "langchain-otel-metrics"))
      assert metrics_handler == nil

      # Span handler should still be attached
      span_handlers = :telemetry.list_handlers([:langchain, :llm, :call, :start])
      span_handler = Enum.find(span_handlers, &(&1.id == "langchain-otel-span"))
      assert span_handler != nil

      OpenTelemetry.teardown()
    end

    test "teardown is idempotent" do
      # Should not raise when called without setup
      assert :ok = OpenTelemetry.teardown()
    end
  end
end
