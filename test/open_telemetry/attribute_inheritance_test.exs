defmodule LangChain.OpenTelemetry.AttributeInheritanceTest do
  @moduledoc """
  Covers `custom_context[:otel_attributes]` reaching the spans nested under a chain.

  These assert on the *exported* spans rather than on the attribute-building
  functions, because the whole point of inheritance is what ends up in the backend.
  The `chat` span in particular cannot be covered any other way: it carries no
  `custom_context` at all, so nothing about its metadata reveals whether inheritance
  worked.
  """
  use ExUnit.Case, async: false

  require Record

  Record.defrecord(
    :span,
    Record.extract(:span, from_lib: "opentelemetry/include/otel_span.hrl")
  )

  # NOTE: intentionally NOT aliasing `LangChain.OpenTelemetry` — that would shadow
  # the real `OpenTelemetry` SDK module this test drives directly.

  setup do
    :application.stop(:opentelemetry)

    :application.set_env(:opentelemetry, :processors, [
      {:otel_batch_processor, %{scheduled_delay_ms: 1}}
    ])

    {:ok, _} = :application.ensure_all_started(:opentelemetry)

    tid = :ets.new(:test_spans, [:bag, :public])
    :otel_batch_processor.set_exporter(:otel_exporter_tab, tid)

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
        attributes: extract_attributes(span(record, :attributes))
      }
    end)
  end

  defp extract_attributes(attrs) when is_tuple(attrs), do: :otel_attributes.map(attrs)
  defp extract_attributes(_), do: %{}

  # Drives a full chain -> llm -> tool nesting through the telemetry events, with the
  # given `custom_context` on the chain (and on the tool, matching what LLMChain does).
  defp run_nested_chain(custom_context) do
    chain_call_id = Ecto.UUID.generate()
    llm_call_id = Ecto.UUID.generate()
    tool_call_id = Ecto.UUID.generate()

    :telemetry.execute(
      [:langchain, :chain, :execute, :start],
      %{system_time: System.system_time()},
      %{call_id: chain_call_id, chain_type: "llm_chain", custom_context: custom_context}
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

    :telemetry.execute(
      [:langchain, :tool, :call, :start],
      %{system_time: System.system_time()},
      %{
        call_id: tool_call_id,
        tool_name: "lookup",
        tool_call_id: "tc-1",
        custom_context: custom_context
      }
    )

    :telemetry.execute(
      [:langchain, :tool, :call, :stop],
      %{duration: 500_000, system_time: System.system_time()},
      %{call_id: tool_call_id}
    )

    :telemetry.execute(
      [:langchain, :chain, :execute, :stop],
      %{duration: 2_000_000, system_time: System.system_time()},
      %{call_id: chain_call_id}
    )
  end

  defp find_span(spans, name), do: Enum.find(spans, &(&1.name == name))

  describe "inheritance to nested spans" do
    setup do
      LangChain.OpenTelemetry.setup(enable_metrics: false)
      :ok
    end

    test "otel_attributes reach the chain, chat, and execute_tool spans", %{tid: tid} do
      run_nested_chain(%{
        otel_attributes: %{"user.id" => "u-1", "organization.id" => "org-9"}
      })

      spans = flush_spans(tid)

      for name <- ["invoke_agent llm_chain", "chat gpt-4o", "execute_tool lookup"] do
        found = find_span(spans, name)
        assert found, "expected a #{name} span"
        assert found.attributes["user.id"] == "u-1", "user.id missing from #{name}"

        assert found.attributes["organization.id"] == "org-9",
               "organization.id missing from #{name}"
      end
    end

    test "gen_ai.conversation.id is inherited so spans can be filtered by session",
         %{tid: tid} do
      run_nested_chain(%{conversation_id: "conv-77"})

      spans = flush_spans(tid)

      assert find_span(spans, "chat gpt-4o").attributes["gen_ai.conversation.id"] == "conv-77"

      assert find_span(spans, "execute_tool lookup").attributes["gen_ai.conversation.id"] ==
               "conv-77"
    end

    test "native value types survive inheritance rather than being stringified",
         %{tid: tid} do
      run_nested_chain(%{otel_attributes: %{"myapp.retry" => 3, "myapp.beta" => true}})

      spans = flush_spans(tid)
      chat = find_span(spans, "chat gpt-4o")

      assert chat.attributes["myapp.retry"] === 3
      assert chat.attributes["myapp.beta"] === true
    end

    # The failure this guards against is quiet and bad: a caller sets a broad
    # attribute map on the chain, and it silently replaces the real per-call model on
    # every chat span — worst exactly when a fallback model engaged and the trace is
    # the only record of which model actually answered.
    test "an inherited value never overwrites an attribute the span derived itself",
         %{tid: tid} do
      run_nested_chain(%{otel_attributes: %{"gen_ai.request.model" => "wrong-model"}})

      spans = flush_spans(tid)

      assert find_span(spans, "chat gpt-4o").attributes["gen_ai.request.model"] == "gpt-4o"
      # The chain span has no model of its own, so there the caller's value stands.
      assert find_span(spans, "invoke_agent llm_chain").attributes["gen_ai.request.model"] ==
               "wrong-model"
    end

    test "attributes do not leak into a sibling chain started afterwards", %{tid: tid} do
      run_nested_chain(%{otel_attributes: %{"user.id" => "u-1"}})
      run_nested_chain(%{})

      spans = flush_spans(tid)
      chats = Enum.filter(spans, &(&1.name == "chat gpt-4o"))

      assert [_, _] = chats
      assert Enum.count(chats, &(&1.attributes["user.id"] == "u-1")) == 1
    end

    test "a nested chain merges its own attributes over the enclosing chain's",
         %{tid: tid} do
      outer_id = Ecto.UUID.generate()
      inner_id = Ecto.UUID.generate()
      llm_id = Ecto.UUID.generate()

      :telemetry.execute(
        [:langchain, :chain, :execute, :start],
        %{system_time: System.system_time()},
        %{
          call_id: outer_id,
          chain_type: "llm_chain",
          custom_context: %{otel_attributes: %{"user.id" => "u-1", "depth" => "outer"}}
        }
      )

      :telemetry.execute(
        [:langchain, :chain, :execute, :start],
        %{system_time: System.system_time()},
        %{
          call_id: inner_id,
          chain_type: "llm_chain",
          custom_context: %{otel_attributes: %{"depth" => "inner"}}
        }
      )

      :telemetry.execute(
        [:langchain, :llm, :call, :start],
        %{system_time: System.system_time()},
        %{call_id: llm_id, model: "gpt-4o", provider: "openai"}
      )

      for {event, id} <- [
            {[:langchain, :llm, :call, :stop], llm_id},
            {[:langchain, :chain, :execute, :stop], inner_id},
            {[:langchain, :chain, :execute, :stop], outer_id}
          ] do
        :telemetry.execute(event, %{duration: 1, system_time: System.system_time()}, %{
          call_id: id
        })
      end

      spans = flush_spans(tid)
      chat = find_span(spans, "chat gpt-4o")

      # Inherited from the outer chain, which the inner one did not override.
      assert chat.attributes["user.id"] == "u-1"
      # The innermost chain wins where both set the same key.
      assert chat.attributes["depth"] == "inner"
    end
  end

  describe "inherit_attributes: false" do
    setup do
      LangChain.OpenTelemetry.setup(enable_metrics: false, inherit_attributes: false)
      :ok
    end

    test "keeps attributes on the chain span only", %{tid: tid} do
      run_nested_chain(%{otel_attributes: %{"user.id" => "u-1"}})

      spans = flush_spans(tid)

      # The passthrough still applies to the spans whose own metadata carries
      # custom_context; only the inheritance to `chat` is switched off.
      assert find_span(spans, "invoke_agent llm_chain").attributes["user.id"] == "u-1"
      assert find_span(spans, "execute_tool lookup").attributes["user.id"] == "u-1"
      refute Map.has_key?(find_span(spans, "chat gpt-4o").attributes, "user.id")
    end
  end

  describe "Enrich" do
    setup do
      LangChain.OpenTelemetry.setup(enable_metrics: false)
      :ok
    end

    test "set_current_span_attributes/1 enriches the innermost open span", %{tid: tid} do
      chain_id = Ecto.UUID.generate()
      llm_id = Ecto.UUID.generate()

      :telemetry.execute(
        [:langchain, :chain, :execute, :start],
        %{system_time: System.system_time()},
        %{call_id: chain_id, chain_type: "llm_chain"}
      )

      :telemetry.execute(
        [:langchain, :llm, :call, :start],
        %{system_time: System.system_time()},
        %{call_id: llm_id, model: "gpt-4o", provider: "openai"}
      )

      LangChain.OpenTelemetry.Enrich.set_current_span_attributes(%{"myapp.cache" => "hit"})

      :telemetry.execute(
        [:langchain, :llm, :call, :stop],
        %{duration: 1, system_time: System.system_time()},
        %{call_id: llm_id}
      )

      :telemetry.execute(
        [:langchain, :chain, :execute, :stop],
        %{duration: 2, system_time: System.system_time()},
        %{call_id: chain_id}
      )

      spans = flush_spans(tid)

      assert find_span(spans, "chat gpt-4o").attributes["myapp.cache"] == "hit"
      refute Map.has_key?(find_span(spans, "invoke_agent llm_chain").attributes, "myapp.cache")
    end

    test "put_inherited_attributes/1 seeds attributes for spans opened afterwards",
         %{tid: tid} do
      LangChain.OpenTelemetry.Enrich.put_inherited_attributes(%{"organization.id" => "org-5"})

      run_nested_chain(%{})

      spans = flush_spans(tid)

      for name <- ["invoke_agent llm_chain", "chat gpt-4o", "execute_tool lookup"] do
        assert find_span(spans, name).attributes["organization.id"] == "org-5",
               "organization.id missing from #{name}"
      end
    end

    test "coerces values so a non-encodable payload cannot drop the span", %{tid: tid} do
      chain_id = Ecto.UUID.generate()

      :telemetry.execute(
        [:langchain, :chain, :execute, :start],
        %{system_time: System.system_time()},
        %{call_id: chain_id, chain_type: "llm_chain"}
      )

      LangChain.OpenTelemetry.Enrich.set_current_span_attributes(%{"myapp.nested" => %{a: 1}})

      :telemetry.execute(
        [:langchain, :chain, :execute, :stop],
        %{duration: 1, system_time: System.system_time()},
        %{call_id: chain_id}
      )

      spans = flush_spans(tid)
      chain = find_span(spans, "invoke_agent llm_chain")

      assert chain, "the span survived the nested-map attribute"
      assert chain.attributes["myapp.nested"] == ~s({"a":1})
    end

    test "no-ops safely when no span is open" do
      assert :ok = LangChain.OpenTelemetry.Enrich.set_current_span_attributes(%{"a" => "b"})
    end
  end
end
