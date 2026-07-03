defmodule LangChain.OpenTelemetry.MetricsHandlerTest do
  use ExUnit.Case, async: false

  alias LangChain.OpenTelemetry.MetricsHandler

  # Attach a capturing handler to the intermediary metric events the
  # MetricsHandler re-emits, forwarding them to the test process.
  setup do
    test_pid = self()
    handler_id = "metrics-handler-test-#{System.unique_integer([:positive])}"

    events = [
      [:langchain, :otel, :operation, :duration],
      [:langchain, :otel, :token, :usage],
      [:langchain, :otel, :operation, :time_to_first_token]
    ]

    :telemetry.attach_many(
      handler_id,
      events,
      fn event, measurements, metadata, _ ->
        send(test_pid, {:metric, event, measurements, metadata})
      end,
      nil
    )

    on_exit(fn -> :telemetry.detach(handler_id) end)
    :ok
  end

  defp native(seconds), do: System.convert_time_unit(seconds, :second, :native)

  describe "duration emission" do
    test "emits operation duration in seconds with operation/provider/model attrs" do
      MetricsHandler.handle_event(
        [:langchain, :llm, :call, :stop],
        %{duration: native(2)},
        %{provider: "openai", model: "gpt-4o"},
        nil
      )

      assert_received {:metric, [:langchain, :otel, :operation, :duration], %{duration_s: 2.0},
                       attrs}

      assert attrs["gen_ai.operation.name"] == "chat"
      assert attrs["gen_ai.provider.name"] == "openai"
      assert attrs["gen_ai.request.model"] == "gpt-4o"
    end

    test "uses invoke_agent operation name for chain stop events" do
      MetricsHandler.handle_event(
        [:langchain, :chain, :execute, :stop],
        %{duration: native(1)},
        %{},
        nil
      )

      assert_received {:metric, [:langchain, :otel, :operation, :duration], %{duration_s: 1.0},
                       attrs}

      assert attrs["gen_ai.operation.name"] == "invoke_agent"
    end

    test "uses execute_tool operation name for tool stop events" do
      MetricsHandler.handle_event(
        [:langchain, :tool, :call, :stop],
        %{duration: native(1)},
        %{},
        nil
      )

      assert_received {:metric, [:langchain, :otel, :operation, :duration], _measurements, attrs}
      assert attrs["gen_ai.operation.name"] == "execute_tool"
    end

    test "skips duration emission when no duration measurement is present" do
      MetricsHandler.handle_event(
        [:langchain, :llm, :call, :stop],
        %{},
        %{provider: "openai", model: "gpt-4o"},
        nil
      )

      refute_received {:metric, [:langchain, :otel, :operation, :duration], _, _}
    end
  end

  describe "time-to-first-token emission" do
    test "emits time_to_first_token in seconds with chat/provider/model attrs" do
      MetricsHandler.handle_event(
        [:langchain, :llm, :stream, :first_token],
        %{duration: native(1)},
        %{provider: "openai", model: "gpt-4o"},
        nil
      )

      assert_received {:metric, [:langchain, :otel, :operation, :time_to_first_token],
                       %{duration_s: 1.0}, attrs}

      assert attrs["gen_ai.operation.name"] == "chat"
      assert attrs["gen_ai.provider.name"] == "openai"
      assert attrs["gen_ai.request.model"] == "gpt-4o"
    end

    test "skips time_to_first_token when no duration measurement is present" do
      MetricsHandler.handle_event(
        [:langchain, :llm, :stream, :first_token],
        %{},
        %{provider: "openai", model: "gpt-4o"},
        nil
      )

      refute_received {:metric, [:langchain, :otel, :operation, :time_to_first_token], _, _}
    end
  end

  describe "token usage emission" do
    test "emits separate input and output token events" do
      MetricsHandler.handle_event(
        [:langchain, :llm, :call, :stop],
        %{duration: native(1)},
        %{provider: "openai", model: "gpt-4o", token_usage: %{input: 10, output: 5}},
        nil
      )

      assert_received {:metric, [:langchain, :otel, :token, :usage], %{tokens: 10}, input_attrs}
      assert input_attrs["gen_ai.token.type"] == "input"
      assert input_attrs["gen_ai.provider.name"] == "openai"

      assert_received {:metric, [:langchain, :otel, :token, :usage], %{tokens: 5}, output_attrs}
      assert output_attrs["gen_ai.token.type"] == "output"
    end

    test "skips a token type when its count is nil" do
      MetricsHandler.handle_event(
        [:langchain, :llm, :call, :stop],
        %{duration: native(1)},
        %{token_usage: %{input: 10, output: nil}},
        nil
      )

      assert_received {:metric, [:langchain, :otel, :token, :usage], %{tokens: 10}, input_attrs}
      assert input_attrs["gen_ai.token.type"] == "input"

      # No output event since the output count was nil.
      refute_received {:metric, [:langchain, :otel, :token, :usage], _, _}
    end

    test "does not emit token events when token_usage is absent" do
      MetricsHandler.handle_event(
        [:langchain, :llm, :call, :stop],
        %{duration: native(1)},
        %{provider: "openai", model: "gpt-4o"},
        nil
      )

      refute_received {:metric, [:langchain, :otel, :token, :usage], _, _}
    end

    test "chain and tool stop events do not emit token usage" do
      MetricsHandler.handle_event(
        [:langchain, :chain, :execute, :stop],
        %{duration: native(1)},
        %{token_usage: %{input: 10, output: 5}},
        nil
      )

      refute_received {:metric, [:langchain, :otel, :token, :usage], _, _}
    end
  end

  describe "failure (exception) emission" do
    test "emits a duration metric tagged with error.type for a failed LLM call" do
      MetricsHandler.handle_event(
        [:langchain, :llm, :call, :exception],
        %{duration: native(3)},
        %{provider: "openai", model: "gpt-4o", error: %RuntimeError{message: "boom"}},
        nil
      )

      assert_received {:metric, [:langchain, :otel, :operation, :duration], %{duration_s: 3.0},
                       attrs}

      assert attrs["gen_ai.operation.name"] == "chat"
      assert attrs["gen_ai.provider.name"] == "openai"
      assert attrs["error.type"] == "RuntimeError"
    end

    test "uses invoke_agent operation name for a failed chain execution" do
      MetricsHandler.handle_event(
        [:langchain, :chain, :execute, :exception],
        %{duration: native(1)},
        %{error: %ArgumentError{}},
        nil
      )

      assert_received {:metric, [:langchain, :otel, :operation, :duration], _, attrs}
      assert attrs["gen_ai.operation.name"] == "invoke_agent"
      assert attrs["error.type"] == "ArgumentError"
    end

    test "uses execute_tool operation name for a failed tool call" do
      MetricsHandler.handle_event(
        [:langchain, :tool, :call, :exception],
        %{duration: native(1)},
        %{error: %RuntimeError{}},
        nil
      )

      assert_received {:metric, [:langchain, :otel, :operation, :duration], _, attrs}
      assert attrs["gen_ai.operation.name"] == "execute_tool"
      assert attrs["error.type"] == "RuntimeError"
    end

    test "falls back to a generic error.type when no exception struct is present" do
      MetricsHandler.handle_event(
        [:langchain, :llm, :call, :exception],
        %{duration: native(1)},
        %{provider: "openai", model: "gpt-4o"},
        nil
      )

      assert_received {:metric, [:langchain, :otel, :operation, :duration], _, attrs}
      assert attrs["error.type"] == "error"
    end

    test "does not emit token usage on failure" do
      MetricsHandler.handle_event(
        [:langchain, :llm, :call, :exception],
        %{duration: native(1)},
        %{error: %RuntimeError{}},
        nil
      )

      refute_received {:metric, [:langchain, :otel, :token, :usage], _, _}
    end
  end

  describe "attributes" do
    test "omits provider/model attributes when not provided" do
      MetricsHandler.handle_event(
        [:langchain, :llm, :call, :stop],
        %{duration: native(1)},
        %{},
        nil
      )

      assert_received {:metric, [:langchain, :otel, :operation, :duration], _, attrs}
      refute Map.has_key?(attrs, "gen_ai.provider.name")
      refute Map.has_key?(attrs, "gen_ai.request.model")
    end
  end

  describe "resilience" do
    # Mirrors the SpanHandler resilience guarantee: `:telemetry` permanently
    # detaches a handler that raises, so a single bad payload must not disable
    # metrics VM-wide. The handler must trap, log, and stay attached.
    test "traps a raising payload, logs, and stays attached" do
      event = [:langchain, :llm, :call, :stop]

      handler_id =
        MetricsHandler.handler_id() <> "-resilience-#{System.unique_integer([:positive])}"

      :telemetry.attach(handler_id, event, &MetricsHandler.handle_event/4, nil)
      on_exit(fn -> :telemetry.detach(handler_id) end)

      # A non-integer duration makes System.convert_time_unit/3 raise inside the
      # handler. Without the trap, :telemetry would detach it after this event.
      log =
        ExUnit.CaptureLog.capture_log(fn ->
          :telemetry.execute(event, %{duration: :not_a_number}, %{provider: "openai"})
        end)

      assert log =~ "metrics handler failed"

      # Still registered after the failure...
      assert Enum.any?(:telemetry.list_handlers(event), fn h -> h.id == handler_id end)

      # ...and still functioning: a subsequent valid event re-emits normally.
      :telemetry.execute(event, %{duration: native(1)}, %{provider: "openai", model: "gpt-4o"})

      assert_received {:metric, [:langchain, :otel, :operation, :duration], %{duration_s: 1.0},
                       _attrs}

      :telemetry.detach(handler_id)
    end
  end
end
