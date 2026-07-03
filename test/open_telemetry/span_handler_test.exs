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

    test "sibling LLM and tool spans both parent to the chain and leave no leaked context",
         %{tid: tid} do
      chain_call_id = Ecto.UUID.generate()
      llm_call_id = Ecto.UUID.generate()
      tool_call_id = Ecto.UUID.generate()
      chain_dict_key = {LangChain.OpenTelemetry.SpanHandler, chain_call_id}

      # Chain opens, then an LLM call fully opens and closes inside it...
      :telemetry.execute(
        [:langchain, :chain, :execute, :start],
        %{system_time: System.system_time()},
        %{call_id: chain_call_id, chain_type: "llm_chain"}
      )

      :telemetry.execute(
        [:langchain, :llm, :call, :start],
        %{system_time: System.system_time()},
        %{call_id: llm_call_id, model: "gpt-4o", provider: "openai"}
      )

      :telemetry.execute(
        [:langchain, :llm, :call, :stop],
        %{duration: 1_000_000, system_time: System.system_time()},
        %{call_id: llm_call_id}
      )

      # ...then a tool call opens. If the LLM span's context wasn't detached in LIFO
      # order on its :stop, this tool span would mis-parent to the (closed) LLM span
      # instead of the chain.
      :telemetry.execute(
        [:langchain, :tool, :call, :start],
        %{system_time: System.system_time()},
        %{call_id: tool_call_id, tool_name: "get_weather", tool_call_id: "tc_1"}
      )

      :telemetry.execute(
        [:langchain, :tool, :call, :stop],
        %{duration: 500_000, system_time: System.system_time()},
        %{call_id: tool_call_id}
      )

      :telemetry.execute(
        [:langchain, :chain, :execute, :stop],
        %{duration: 3_000_000, system_time: System.system_time()},
        %{call_id: chain_call_id}
      )

      # The chain's stored span/token was popped on :stop — nothing leaked.
      refute Process.get(chain_dict_key)

      spans = flush_spans(tid)
      assert length(spans) == 3

      chain_span = Enum.find(spans, &(&1.name == "invoke_agent llm_chain"))
      llm_span = Enum.find(spans, &(&1.name == "chat gpt-4o"))
      tool_span = Enum.find(spans, &(&1.name == "execute_tool get_weather"))

      assert chain_span != nil
      assert llm_span != nil
      assert tool_span != nil

      # One trace; both children hang directly off the chain (siblings), proving the
      # LLM span's context was correctly detached before the tool span opened.
      assert llm_span.trace_id == chain_span.trace_id
      assert tool_span.trace_id == chain_span.trace_id
      assert llm_span.parent_span_id == chain_span.span_id
      assert tool_span.parent_span_id == chain_span.span_id
    end
  end

  describe "async tool span propagation" do
    # Async tools (`async: true`) run in a spawned `Task`. The OpenTelemetry
    # context lives in the parent process's dictionary and is NOT inherited by the
    # spawned process. The documented remedy is to capture the parent context
    # before the run and re-attach it inside the `:on_tool_pre_execution` callback
    # (which fires inside the Task, before the tool span opens). These tests
    # exercise that contract at the span level — the propagated case parents the
    # tool span to the chain; the un-propagated case reproduces the orphan symptom
    # the docs warn about (a new root span in a separate trace).

    test "an async tool span parents to the chain when the parent ctx is propagated",
         %{tid: tid} do
      chain_call_id = Ecto.UUID.generate()
      tool_call_id = Ecto.UUID.generate()

      :telemetry.execute(
        [:langchain, :chain, :execute, :start],
        %{system_time: System.system_time()},
        %{call_id: chain_call_id, chain_type: "llm_chain"}
      )

      # Captured in the parent process before spawning — exactly what a caller does
      # before `LLMChain.run`, then re-applies inside `:on_tool_pre_execution`.
      # `:otel_ctx` is what `OpenTelemetry.Ctx` delegates to; used directly here
      # because `alias LangChain.OpenTelemetry` shadows the `OpenTelemetry` prefix.
      parent_ctx = :otel_ctx.get_current()

      task =
        Task.async(fn ->
          # Mirrors the `:on_tool_pre_execution` callback re-attaching the captured
          # context inside the spawned Task.
          :otel_ctx.attach(parent_ctx)

          :telemetry.execute(
            [:langchain, :tool, :call, :start],
            %{system_time: System.system_time()},
            %{call_id: tool_call_id, tool_name: "get_weather", tool_call_id: "tc_1"}
          )

          :telemetry.execute(
            [:langchain, :tool, :call, :stop],
            %{duration: 500_000, system_time: System.system_time()},
            %{call_id: tool_call_id}
          )
        end)

      Task.await(task)

      :telemetry.execute(
        [:langchain, :chain, :execute, :stop],
        %{duration: 2_000_000, system_time: System.system_time()},
        %{call_id: chain_call_id}
      )

      spans = flush_spans(tid)
      chain_span = Enum.find(spans, &(&1.name == "invoke_agent llm_chain"))
      tool_span = Enum.find(spans, &(&1.name == "execute_tool get_weather"))

      assert chain_span != nil
      assert tool_span != nil
      # Same trace, and the tool span hangs off the chain despite running in a
      # different process.
      assert tool_span.trace_id == chain_span.trace_id
      assert tool_span.parent_span_id == chain_span.span_id
    end

    test "an async tool span is orphaned into its own trace without ctx propagation",
         %{tid: tid} do
      chain_call_id = Ecto.UUID.generate()
      tool_call_id = Ecto.UUID.generate()

      :telemetry.execute(
        [:langchain, :chain, :execute, :start],
        %{system_time: System.system_time()},
        %{call_id: chain_call_id, chain_type: "llm_chain"}
      )

      # No context propagation: the spawned Task begins with an empty OTel context
      # (process dictionaries are not inherited across `Task.async`).
      task =
        Task.async(fn ->
          :telemetry.execute(
            [:langchain, :tool, :call, :start],
            %{system_time: System.system_time()},
            %{call_id: tool_call_id, tool_name: "get_weather", tool_call_id: "tc_1"}
          )

          :telemetry.execute(
            [:langchain, :tool, :call, :stop],
            %{duration: 500_000, system_time: System.system_time()},
            %{call_id: tool_call_id}
          )
        end)

      Task.await(task)

      :telemetry.execute(
        [:langchain, :chain, :execute, :stop],
        %{duration: 2_000_000, system_time: System.system_time()},
        %{call_id: chain_call_id}
      )

      spans = flush_spans(tid)
      chain_span = Enum.find(spans, &(&1.name == "invoke_agent llm_chain"))
      tool_span = Enum.find(spans, &(&1.name == "execute_tool get_weather"))

      assert chain_span != nil
      assert tool_span != nil
      # The orphan symptom: the tool span is a root of its own trace, not a child
      # of the chain.
      assert tool_span.trace_id != chain_span.trace_id
      refute tool_span.parent_span_id == chain_span.span_id
    end
  end

  describe "request parameters on the exported span" do
    test "gen_ai.request.*, output.type, server.*, and finish_reasons land on the span",
         %{tid: tid} do
      # Unit tests in AttributesTest prove these are built correctly; this asserts
      # they survive the SpanHandler and OTel SDK onto the actually-exported span.
      call_id = Ecto.UUID.generate()

      :telemetry.execute(
        [:langchain, :llm, :call, :start],
        %{system_time: System.system_time()},
        %{
          call_id: call_id,
          model: "gpt-4o",
          provider: "openai",
          output_type: "json",
          endpoint: "https://api.openai.com/v1/chat/completions",
          request_options: %{
            temperature: 0.7,
            max_tokens: 512,
            top_p: 0.9,
            seed: 42,
            stream: false
          }
        }
      )

      :telemetry.execute(
        [:langchain, :llm, :call, :stop],
        %{duration: 1_000_000, system_time: System.system_time()},
        %{
          call_id: call_id,
          model: "gpt-4o",
          token_usage: %{input: 10, output: 5},
          result: {:ok, LangChain.Message.new_assistant!(%{content: "done"})}
        }
      )

      assert [span] = flush_spans(tid)
      attrs = span.attributes

      assert attrs["gen_ai.request.temperature"] == 0.7
      assert attrs["gen_ai.request.max_tokens"] == 512
      assert attrs["gen_ai.request.top_p"] == 0.9
      assert attrs["gen_ai.request.seed"] == 42
      assert attrs["gen_ai.request.stream"] == false
      assert attrs["gen_ai.output.type"] == "json"
      assert attrs["server.address"] == "api.openai.com"
      assert attrs["server.port"] == 443
      assert attrs["gen_ai.response.finish_reasons"] == ["stop"]
    end
  end

  describe "events with no started span" do
    test "a :stop for an unknown call_id is a no-op (no crash, no span)", %{tid: tid} do
      :telemetry.execute(
        [:langchain, :llm, :call, :stop],
        %{duration: 1_000_000, system_time: System.system_time()},
        %{call_id: Ecto.UUID.generate(), model: "gpt-4o"}
      )

      assert flush_spans(tid) == []
    end

    test "an :exception for an unknown call_id is a no-op (no crash, no span)", %{tid: tid} do
      :telemetry.execute(
        [:langchain, :llm, :call, :exception],
        %{system_time: System.system_time()},
        %{
          call_id: Ecto.UUID.generate(),
          kind: :error,
          error: %RuntimeError{message: "orphan"},
          stacktrace: []
        }
      )

      assert flush_spans(tid) == []
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

    test "sets error.type attribute to the exception module", %{tid: tid} do
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

      assert [error_span] = flush_spans(tid)
      assert error_span.attributes["error.type"] == "RuntimeError"
    end

    test "ends the span (no leak) when the wrapped function throws", %{tid: tid} do
      call_id = Ecto.UUID.generate()
      dict_key = {LangChain.OpenTelemetry.SpanHandler, call_id}

      caught =
        try do
          LangChain.Telemetry.span(
            [:langchain, :llm, :call],
            %{call_id: call_id, model: "gpt-4o", provider: "openai"},
            fn -> throw(:boom) end
          )
        catch
          :throw, reason -> reason
        end

      assert caught == :boom
      # The span was opened on :start; the :exception emitted on the throw must
      # have popped and ended it. A lingering process-dict entry would mean the
      # span (and its attached context) leaked.
      assert Process.get(dict_key) == nil

      assert [error_span] = flush_spans(tid)
      assert error_span.name == "chat gpt-4o"
      assert error_span.status != nil
    end

    test "ends the span (no leak) when the wrapped function exits", %{tid: tid} do
      call_id = Ecto.UUID.generate()
      dict_key = {LangChain.OpenTelemetry.SpanHandler, call_id}

      caught =
        try do
          LangChain.Telemetry.span(
            [:langchain, :llm, :call],
            %{call_id: call_id, model: "gpt-4o", provider: "openai"},
            fn -> exit(:down) end
          )
        catch
          :exit, reason -> reason
        end

      assert caught == :down
      assert Process.get(dict_key) == nil

      assert [error_span] = flush_spans(tid)
      assert error_span.name == "chat gpt-4o"
      assert error_span.status != nil
    end

    test "closes a chain execute span with error status and error.type on :exception",
         %{tid: tid} do
      call_id = Ecto.UUID.generate()

      :telemetry.execute(
        [:langchain, :chain, :execute, :start],
        %{system_time: System.system_time()},
        %{call_id: call_id, chain_type: "llm_chain"}
      )

      :telemetry.execute(
        [:langchain, :chain, :execute, :exception],
        %{system_time: System.system_time()},
        %{
          call_id: call_id,
          kind: :error,
          error: %RuntimeError{message: "chain blew up"},
          stacktrace: []
        }
      )

      assert [error_span] = flush_spans(tid)
      assert error_span.name == "invoke_agent llm_chain"
      assert error_span.status != nil
      assert error_span.attributes["error.type"] == "RuntimeError"
    end

    test "closes a tool call span with error status and error.type on :exception",
         %{tid: tid} do
      call_id = Ecto.UUID.generate()

      :telemetry.execute(
        [:langchain, :tool, :call, :start],
        %{system_time: System.system_time()},
        %{call_id: call_id, tool_name: "calculator", tool_call_id: "tc-1"}
      )

      :telemetry.execute(
        [:langchain, :tool, :call, :exception],
        %{system_time: System.system_time()},
        %{
          call_id: call_id,
          kind: :error,
          error: %ArgumentError{message: "bad tool arg"},
          stacktrace: []
        }
      )

      assert [error_span] = flush_spans(tid)
      assert error_span.name == "execute_tool calculator"
      assert error_span.status != nil
      assert error_span.attributes["error.type"] == "ArgumentError"
    end

    test "sets a generic error.type and status when no exception struct is present",
         %{tid: tid} do
      # A `throw`/`exit` carries `error: nil` (there is no exception struct). The
      # span must still close with an error status and a generic `error.type`, and
      # must not attempt to record an exception from `nil`.
      call_id = Ecto.UUID.generate()

      :telemetry.execute(
        [:langchain, :llm, :call, :start],
        %{system_time: System.system_time()},
        %{call_id: call_id, model: "gpt-4o", provider: "openai"}
      )

      :telemetry.execute(
        [:langchain, :llm, :call, :exception],
        %{system_time: System.system_time()},
        %{call_id: call_id, kind: :exit, error: nil, reason: :down, stacktrace: []}
      )

      assert [error_span] = flush_spans(tid)
      assert error_span.name == "chat gpt-4o"
      assert error_span.status != nil
      assert error_span.attributes["error.type"] == "error"
    end
  end

  describe "handler resilience" do
    # `:telemetry` permanently detaches any handler that raises. A serialization
    # failure on one bad payload must not disable tracing for the whole VM, so the
    # handler traps its own exceptions.
    test "a failing event is skipped without detaching the handler", %{tid: tid} do
      # A capture-enabled config plus a message whose content cannot be JSON
      # encoded forces the serializer to raise inside the prompt handler.
      OpenTelemetry.teardown()
      OpenTelemetry.setup(enable_metrics: false, capture_input_messages: true)

      call_id = Ecto.UUID.generate()

      :telemetry.execute(
        [:langchain, :llm, :call, :start],
        %{system_time: System.system_time()},
        %{call_id: call_id, model: "gpt-4o", provider: "openai"}
      )

      unencodable = %LangChain.Message{
        role: :user,
        content: [%LangChain.Message.ContentPart{type: :text, content: <<0xFFFF::16>>}]
      }

      # This would raise inside the handler (invalid UTF-8 → Jason.encode! error).
      # The handler must swallow it and stay attached.
      :telemetry.execute(
        [:langchain, :llm, :prompt],
        %{system_time: System.system_time()},
        %{model: "gpt-4o", messages: [unencodable]}
      )

      # Handler is still attached: a subsequent normal stop still produces a span.
      :telemetry.execute(
        [:langchain, :llm, :call, :stop],
        %{duration: 1_000_000, system_time: System.system_time()},
        %{call_id: call_id, model: "gpt-4o", provider: "openai"}
      )

      assert [llm_span] = flush_spans(tid)
      assert llm_span.name == "chat gpt-4o"

      # The handler must still be registered after the failure.
      assert Enum.any?(:telemetry.list_handlers([:langchain, :llm, :prompt]), fn h ->
               h.id == OpenTelemetry.SpanHandler.handler_id()
             end)
    end
  end
end
