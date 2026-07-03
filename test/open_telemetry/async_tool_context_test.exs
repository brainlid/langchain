defmodule LangChain.OpenTelemetry.AsyncToolContextTest do
  use ExUnit.Case, async: false

  require Record
  require OpenTelemetry.Tracer, as: Tracer

  Record.defrecord(
    :span,
    Record.extract(:span, from_lib: "opentelemetry/include/otel_span.hrl")
  )

  # NOTE: intentionally NOT aliasing `LangChain.OpenTelemetry` — doing so would
  # shadow the real `OpenTelemetry` SDK module this test drives directly.
  alias LangChain.Chains.LLMChain
  alias LangChain.ChatModels.ChatOpenAI
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

    :ets.tab2list(tid)
    |> Enum.map(fn record ->
      %{
        name: span(record, :name),
        span_id: span(record, :span_id),
        parent_span_id: span(record, :parent_span_id)
      }
    end)
  end

  defp chain_with_async_tool do
    func =
      Function.new!(%{
        name: "get_thing",
        async: true,
        function: fn _args, _context -> {:ok, "done"} end
      })

    tool_call = ToolCall.new!(%{call_id: "call_1", name: "get_thing", arguments: %{}})

    LLMChain.new!(%{llm: ChatOpenAI.new!(%{stream: false})})
    |> LLMChain.add_tools(func)
    |> LLMChain.add_message(Message.new_system!())
    |> LLMChain.add_message(Message.new_assistant!(%{status: :complete, tool_calls: [tool_call]}))
  end

  test "async tool span nests under the surrounding OpenTelemetry context", %{tid: tid} do
    chain = chain_with_async_tool()

    # Open a span that stands in for the chain span, and make it current — exactly
    # the situation the async tool runs inside during a real `LLMChain.run/2`.
    parent_ctx = OpenTelemetry.Ctx.get_current()
    parent_span = Tracer.start_span("parent", %{kind: :internal})

    token =
      OpenTelemetry.Ctx.attach(OpenTelemetry.Tracer.set_current_span(parent_ctx, parent_span))

    try do
      %LLMChain{} = LLMChain.execute_tool_calls(chain)
    after
      OpenTelemetry.Span.end_span(parent_span)
      OpenTelemetry.Ctx.detach(token)
    end

    spans = flush_spans(tid)
    parent = Enum.find(spans, &(&1.name == "parent"))
    tool = Enum.find(spans, &String.starts_with?(&1.name, "execute_tool"))

    assert parent, "expected the parent span to be exported"
    assert tool, "expected the async tool span to be exported"

    # The propagated context makes the async tool span a child of the parent
    # span, not an orphaned root.
    assert tool.parent_span_id == parent.span_id
  end

  test "without a surrounding context, the async tool span is a root", %{tid: tid} do
    chain = chain_with_async_tool()

    # No parent span attached in this process.
    %LLMChain{} = LLMChain.execute_tool_calls(chain)

    spans = flush_spans(tid)
    tool = Enum.find(spans, &String.starts_with?(&1.name, "execute_tool"))

    assert tool, "expected the async tool span to be exported"
    # An undefined parent is represented as 0 by the OTel SDK.
    assert tool.parent_span_id in [:undefined, 0]
  end
end
