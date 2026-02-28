defmodule LangChain.OpenTelemetry.SpanHandlerTest do
  use ExUnit.Case

  require Record

  # Extract the span record from the OTel SDK header
  Record.defrecord(
    :span,
    Record.extract(:span, from_lib: "opentelemetry/include/otel_span.hrl")
  )

  alias LangChain.OpenTelemetry

  setup do
    # Set up OTel SDK with an ETS-based exporter for capturing spans
    :application.stop(:opentelemetry)

    :application.set_env(:opentelemetry, :processors, [
      {:otel_batch_processor, %{scheduled_delay_ms: 1}}
    ])

    {:ok, _} = :application.ensure_all_started(:opentelemetry)

    # Create ETS table for the exporter
    tid = :ets.new(:test_spans, [:bag, :public])

    :otel_batch_processor.set_exporter(:otel_exporter_tab, tid)

    OpenTelemetry.setup(enable_metrics: false)

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
    # Give the batch processor time to export
    Process.sleep(100)

    :otel_batch_processor.force_flush(%{
      reg_name: {:via, :gproc, {:n, :l, {:otel_batch_processor, :global}}}
    })

    Process.sleep(100)

    :ets.tab2list(tid)
    |> Enum.map(fn record -> span_to_map(record) end)
  end

  defp span_to_map(record) do
    %{
      name: span(record, :name),
      kind: span(record, :kind),
      attributes: extract_attributes(span(record, :attributes)),
      parent_span_id: span(record, :parent_span_id),
      trace_id: span(record, :trace_id),
      span_id: span(record, :span_id),
      status: span(record, :status)
    }
  end

  defp extract_attributes(attrs) when is_tuple(attrs) do
    :otel_attributes.map(attrs)
  end

  defp extract_attributes(_), do: %{}

  describe "LLM call spans" do
    test "creates a span for LLM call start/stop", %{tid: tid} do
      call_id = Ecto.UUID.generate()

      :telemetry.execute(
        [:langchain, :llm, :call, :start],
        %{system_time: System.system_time()},
        %{
          call_id: call_id,
          model: "gpt-4o",
          provider: "openai",
          message_count: 2,
          tools_count: 0
        }
      )

      :telemetry.execute(
        [:langchain, :llm, :call, :stop],
        %{duration: 1_000_000, system_time: System.system_time()},
        %{
          call_id: call_id,
          model: "gpt-4o",
          provider: "openai",
          token_usage: %{input: 100, output: 50}
        }
      )

      spans = flush_spans(tid)
      assert [llm_span] = spans

      assert llm_span.name == "chat gpt-4o"
      assert llm_span.kind == :client

      assert llm_span.attributes["gen_ai.operation.name"] == "chat"
      assert llm_span.attributes["gen_ai.request.model"] == "gpt-4o"
      assert llm_span.attributes["gen_ai.provider.name"] == "openai"
      assert llm_span.attributes["gen_ai.usage.input_tokens"] == 100
      assert llm_span.attributes["gen_ai.usage.output_tokens"] == 50
    end

    test "includes gen_ai.response.model on stop", %{tid: tid} do
      call_id = Ecto.UUID.generate()

      :telemetry.execute(
        [:langchain, :llm, :call, :start],
        %{system_time: System.system_time()},
        %{call_id: call_id, model: "gpt-4o", provider: "openai"}
      )

      :telemetry.execute(
        [:langchain, :llm, :call, :stop],
        %{duration: 1_000_000, system_time: System.system_time()},
        %{
          call_id: call_id,
          model: "gpt-4o-2024-05-13",
          token_usage: %{input: 10, output: 5}
        }
      )

      spans = flush_spans(tid)
      assert [llm_span] = spans
      assert llm_span.attributes["gen_ai.response.model"] == "gpt-4o-2024-05-13"
    end
  end

  describe "LLM call spans with capture options" do
    setup do
      OpenTelemetry.teardown()

      OpenTelemetry.setup(
        enable_metrics: false,
        capture_input_messages: true,
        capture_output_messages: true
      )

      :ok
    end

    test "captures input messages from prompt event when configured", %{tid: tid} do
      call_id = Ecto.UUID.generate()
      messages = [LangChain.Message.new_user!("Hello")]

      :telemetry.execute(
        [:langchain, :llm, :call, :start],
        %{system_time: System.system_time()},
        %{call_id: call_id, model: "gpt-4o", provider: "openai"}
      )

      # Messages are now sent via the prompt event, not the start event
      :telemetry.execute(
        [:langchain, :llm, :prompt],
        %{system_time: System.system_time()},
        %{model: "gpt-4o", messages: messages}
      )

      :telemetry.execute(
        [:langchain, :llm, :call, :stop],
        %{duration: 1_000_000, system_time: System.system_time()},
        %{call_id: call_id}
      )

      spans = flush_spans(tid)
      assert [llm_span] = spans

      json = llm_span.attributes["gen_ai.input.messages"]
      assert json != nil
      assert [%{"role" => "user", "content" => "Hello"}] = Jason.decode!(json)
    end

    test "does not capture input messages from prompt event when not configured", %{tid: tid} do
      # Teardown and re-setup without capture_input_messages
      OpenTelemetry.teardown()
      OpenTelemetry.setup(enable_metrics: false, capture_output_messages: true)

      call_id = Ecto.UUID.generate()
      messages = [LangChain.Message.new_user!("Hello")]

      :telemetry.execute(
        [:langchain, :llm, :call, :start],
        %{system_time: System.system_time()},
        %{call_id: call_id, model: "gpt-4o", provider: "openai"}
      )

      :telemetry.execute(
        [:langchain, :llm, :prompt],
        %{system_time: System.system_time()},
        %{model: "gpt-4o", messages: messages}
      )

      :telemetry.execute(
        [:langchain, :llm, :call, :stop],
        %{duration: 1_000_000, system_time: System.system_time()},
        %{call_id: call_id}
      )

      spans = flush_spans(tid)
      assert [llm_span] = spans

      refute Map.has_key?(llm_span.attributes, "gen_ai.input.messages")
    end

    test "captures output messages when configured", %{tid: tid} do
      call_id = Ecto.UUID.generate()
      msg = LangChain.Message.new_assistant!(%{content: "Hi there!"})

      :telemetry.execute(
        [:langchain, :llm, :call, :start],
        %{system_time: System.system_time()},
        %{call_id: call_id, model: "gpt-4o", provider: "openai"}
      )

      :telemetry.execute(
        [:langchain, :llm, :call, :stop],
        %{duration: 1_000_000, system_time: System.system_time()},
        %{call_id: call_id, result: {:ok, msg}}
      )

      spans = flush_spans(tid)
      assert [llm_span] = spans

      json = llm_span.attributes["gen_ai.output.messages"]
      assert json != nil
      assert [%{"role" => "assistant", "content" => "Hi there!"}] = Jason.decode!(json)
    end
  end

  describe "chain execute spans" do
    test "creates a span for chain execution", %{tid: tid} do
      call_id = Ecto.UUID.generate()

      :telemetry.execute(
        [:langchain, :chain, :execute, :start],
        %{system_time: System.system_time()},
        %{
          call_id: call_id,
          chain_type: "llm_chain",
          mode: :run,
          message_count: 1,
          tools_count: 0
        }
      )

      :telemetry.execute(
        [:langchain, :chain, :execute, :stop],
        %{duration: 2_000_000, system_time: System.system_time()},
        %{call_id: call_id}
      )

      spans = flush_spans(tid)
      assert [chain_span] = spans

      assert chain_span.name == "invoke_agent llm_chain"
      assert chain_span.kind == :internal
      assert chain_span.attributes["gen_ai.operation.name"] == "invoke_agent"
      assert chain_span.attributes["gen_ai.agent.name"] == "llm_chain"
    end

    test "includes langfuse attributes from custom_context", %{tid: tid} do
      call_id = Ecto.UUID.generate()

      :telemetry.execute(
        [:langchain, :chain, :execute, :start],
        %{system_time: System.system_time()},
        %{
          call_id: call_id,
          chain_type: "llm_chain",
          custom_context: %{
            langfuse_user_id: "user-abc",
            langfuse_session_id: "sess-xyz"
          }
        }
      )

      :telemetry.execute(
        [:langchain, :chain, :execute, :stop],
        %{duration: 1_000_000, system_time: System.system_time()},
        %{call_id: call_id}
      )

      spans = flush_spans(tid)
      assert [chain_span] = spans

      assert chain_span.attributes["langfuse.user.id"] == "user-abc"
      assert chain_span.attributes["langfuse.session.id"] == "sess-xyz"
    end
  end

  describe "tool call spans" do
    test "creates a span for tool call", %{tid: tid} do
      call_id = Ecto.UUID.generate()

      :telemetry.execute(
        [:langchain, :tool, :call, :start],
        %{system_time: System.system_time()},
        %{call_id: call_id, tool_name: "calculator", tool_call_id: "tc-1"}
      )

      :telemetry.execute(
        [:langchain, :tool, :call, :stop],
        %{duration: 500_000, system_time: System.system_time()},
        %{call_id: call_id}
      )

      spans = flush_spans(tid)
      assert [tool_span] = spans

      assert tool_span.name == "execute_tool calculator"
      assert tool_span.kind == :internal
      assert tool_span.attributes["gen_ai.operation.name"] == "execute_tool"
      assert tool_span.attributes["gen_ai.tool.name"] == "calculator"
      assert tool_span.attributes["gen_ai.tool.call.id"] == "tc-1"
      assert tool_span.attributes["gen_ai.tool.type"] == "function"
    end
  end

  describe "tool call spans with capture options" do
    setup do
      OpenTelemetry.teardown()

      OpenTelemetry.setup(
        enable_metrics: false,
        capture_tool_arguments: true,
        capture_tool_results: true
      )

      :ok
    end

    test "captures tool arguments when configured", %{tid: tid} do
      call_id = Ecto.UUID.generate()

      :telemetry.execute(
        [:langchain, :tool, :call, :start],
        %{system_time: System.system_time()},
        %{
          call_id: call_id,
          tool_name: "calculator",
          tool_call_id: "tc-1",
          arguments: %{"x" => 1, "y" => 2}
        }
      )

      :telemetry.execute(
        [:langchain, :tool, :call, :stop],
        %{duration: 500_000, system_time: System.system_time()},
        %{call_id: call_id}
      )

      spans = flush_spans(tid)
      assert [tool_span] = spans

      json = tool_span.attributes["gen_ai.tool.call.arguments"]
      assert json != nil
      assert %{"x" => 1, "y" => 2} = Jason.decode!(json)
    end

    test "captures tool results when configured", %{tid: tid} do
      call_id = Ecto.UUID.generate()

      :telemetry.execute(
        [:langchain, :tool, :call, :start],
        %{system_time: System.system_time()},
        %{call_id: call_id, tool_name: "calculator", tool_call_id: "tc-1"}
      )

      :telemetry.execute(
        [:langchain, :tool, :call, :stop],
        %{duration: 500_000, system_time: System.system_time()},
        %{call_id: call_id, tool_result: %{content: "42"}}
      )

      spans = flush_spans(tid)
      assert [tool_span] = spans
      assert tool_span.attributes["gen_ai.tool.call.result"] == "42"
    end
  end

  describe "span hierarchy" do
    test "chain -> LLM call creates parent-child relationship", %{tid: tid} do
      chain_call_id = Ecto.UUID.generate()
      llm_call_id = Ecto.UUID.generate()

      # Start chain span
      :telemetry.execute(
        [:langchain, :chain, :execute, :start],
        %{system_time: System.system_time()},
        %{call_id: chain_call_id, chain_type: "llm_chain"}
      )

      # Start LLM span (child of chain)
      :telemetry.execute(
        [:langchain, :llm, :call, :start],
        %{system_time: System.system_time()},
        %{call_id: llm_call_id, model: "gpt-4o", provider: "openai"}
      )

      # End LLM span
      :telemetry.execute(
        [:langchain, :llm, :call, :stop],
        %{duration: 1_000_000, system_time: System.system_time()},
        %{call_id: llm_call_id}
      )

      # End chain span
      :telemetry.execute(
        [:langchain, :chain, :execute, :stop],
        %{duration: 2_000_000, system_time: System.system_time()},
        %{call_id: chain_call_id}
      )

      spans = flush_spans(tid)
      assert length(spans) == 2

      chain_span = Enum.find(spans, &(&1.name == "invoke_agent llm_chain"))
      llm_span = Enum.find(spans, &(&1.name == "chat gpt-4o"))

      assert chain_span != nil
      assert llm_span != nil

      # Same trace
      assert chain_span.trace_id == llm_span.trace_id

      # LLM span is a child of chain span
      assert llm_span.parent_span_id == chain_span.span_id
    end
  end

  describe "exception handling" do
    test "records error status on exception", %{tid: tid} do
      call_id = Ecto.UUID.generate()

      :telemetry.execute(
        [:langchain, :llm, :call, :start],
        %{system_time: System.system_time()},
        %{call_id: call_id, model: "gpt-4o", provider: "openai"}
      )

      :telemetry.execute(
        [:langchain, :llm, :call, :exception],
        %{system_time: System.system_time()},
        %{
          call_id: call_id,
          kind: :error,
          error: %RuntimeError{message: "connection timeout"},
          stacktrace: []
        }
      )

      spans = flush_spans(tid)
      assert [error_span] = spans

      assert error_span.name == "chat gpt-4o"
      # Status should indicate error
      assert error_span.status != nil
    end
  end
end
