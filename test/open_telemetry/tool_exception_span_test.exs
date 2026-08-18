defmodule LangChain.OpenTelemetry.ToolExceptionSpanTest do
  use ExUnit.Case, async: false

  require Record

  Record.defrecord(
    :span,
    Record.extract(:span, from_lib: "opentelemetry/include/otel_span.hrl")
  )

  # NOTE: intentionally NOT aliasing `LangChain.OpenTelemetry` — doing so would
  # shadow the real `OpenTelemetry` SDK module this test drives directly.
  alias LangChain.Chains.LLMChain
  alias LangChain.ChatModels.ChatAnthropic
  alias LangChain.Function
  alias LangChain.Message
  alias LangChain.Message.ToolCall

  setup do
    :application.stop(:opentelemetry)

    :application.set_env(:opentelemetry, :processors, [
      {:otel_batch_processor, %{scheduled_delay_ms: 1}}
    ])

    {:ok, _} = :application.ensure_all_started(:opentelemetry)

    tid = :ets.new(:test_spans, [:bag, :public])
    :otel_batch_processor.set_exporter(:otel_exporter_tab, tid)

    LangChain.OpenTelemetry.setup(enable_metrics: false)

    on_exit(fn ->
      LangChain.OpenTelemetry.teardown()

      try do
        :ets.delete(tid)
      rescue
        _ -> :ok
      end
    end)

    %{tid: tid}
  end

  defp flush_spans(tid) do
    Process.sleep(100)

    :otel_batch_processor.force_flush(%{
      reg_name: {:via, :gproc, {:n, :l, {:otel_batch_processor, :global}}}
    })

    Process.sleep(100)

    Enum.map(:ets.tab2list(tid), fn record ->
      %{
        name: span(record, :name),
        status: span(record, :status),
        events: span(record, :events),
        attributes: :otel_attributes.map(span(record, :attributes))
      }
    end)
  end

  defp run_tool(tool) do
    LLMChain.new!(%{
      llm: ChatAnthropic.new!(%{model: "claude-sonnet-4-5-20250929"}),
      tools: [tool]
    })
    |> LLMChain.add_message(
      Message.new_assistant!(%{
        tool_calls: [ToolCall.new!(%{call_id: "c1", name: tool.name, arguments: %{}})]
      })
    )
    |> LLMChain.execute_tool_calls()
  end

  test "a rescued tool exception closes the tool span with an error status", %{tid: tid} do
    tool =
      Function.new!(%{
        name: "raising_tool",
        function: fn _args, _ctx -> raise RuntimeError, "span boom" end
      })

    run_tool(tool)

    assert [tool_span] = flush_spans(tid)
    assert tool_span.name == "execute_tool raising_tool"
    assert tool_span.status != nil
    assert tool_span.attributes["error.type"] == "RuntimeError"
  end

  test "the rescued exception is recorded as a span event", %{tid: tid} do
    tool =
      Function.new!(%{
        name: "raising_tool",
        function: fn _args, _ctx -> raise RuntimeError, "recorded boom" end
      })

    run_tool(tool)

    assert [tool_span] = flush_spans(tid)
    assert {:events, _, _, _, _, [{:event, _time, :exception, attrs_record}]} = tool_span.events

    attrs = :otel_attributes.map(attrs_record)
    assert attrs[:"exception.type"] == "Elixir.RuntimeError"
    assert attrs[:"exception.message"] == "recorded boom"

    # The real stacktrace reaches the backend, which is the point of the feature:
    # error trackers group by frames, not by a formatted message.
    assert attrs[:"exception.stacktrace"] =~ "tool_exception_span_test.exs"
    assert attrs[:"exception.stacktrace"] =~ "execute_with_error_handling"
  end

  test "an error the tool returns deliberately leaves the span unset", %{tid: tid} do
    tool =
      Function.new!(%{
        name: "returns_error",
        function: fn _args, _ctx -> {:error, "not found"} end
      })

    run_tool(tool)

    assert [tool_span] = flush_spans(tid)
    assert tool_span.name == "execute_tool returns_error"
    assert tool_span.status == :undefined
    refute Map.has_key?(tool_span.attributes, "error.type")
  end
end
