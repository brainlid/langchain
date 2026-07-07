defmodule LangChain.TelemetryTest do
  use ExUnit.Case, async: false

  alias LangChain.Telemetry

  describe "span/4 exit and throw handling" do
    setup do
      test_pid = self()
      id = "telemetry-span-exception-#{System.unique_integer([:positive])}"

      :telemetry.attach(
        id,
        [:langchain, :test, :op, :exception],
        fn _event, measurements, metadata, _config ->
          send(test_pid, {:exception, measurements, metadata})
        end,
        nil
      )

      on_exit(fn -> :telemetry.detach(id) end)
      :ok
    end

    test "emits :exception (with duration) and re-throws when the function throws" do
      assert catch_throw(Telemetry.span([:langchain, :test, :op], %{}, fn -> throw(:nope) end)) ==
               :nope

      assert_received {:exception, measurements, metadata}
      # A duration must be present so error latency stays observable.
      assert is_integer(measurements.duration)
      assert metadata.kind == :throw
      assert metadata.reason == :nope
      # No exception struct exists for a throw.
      assert metadata.error == nil
    end

    test "emits :exception (with duration) and re-exits when the function exits" do
      assert catch_exit(Telemetry.span([:langchain, :test, :op], %{}, fn -> exit(:down) end)) ==
               :down

      assert_received {:exception, measurements, metadata}
      assert is_integer(measurements.duration)
      assert metadata.kind == :exit
      assert metadata.reason == :down
      assert metadata.error == nil
    end

    test "still routes raised exceptions through the rescue path" do
      assert_raise RuntimeError, "boom", fn ->
        Telemetry.span([:langchain, :test, :op], %{}, fn -> raise "boom" end)
      end

      assert_received {:exception, _measurements, metadata}
      assert metadata.kind == :error
      assert %RuntimeError{message: "boom"} = metadata.error
    end
  end

  describe "span/4 enrich_stop" do
    setup do
      test_pid = self()
      id = "telemetry-span-stop-#{System.unique_integer([:positive])}"

      :telemetry.attach(
        id,
        [:langchain, :test, :enrich, :stop],
        fn _event, _measurements, metadata, _config ->
          send(test_pid, {:stop, metadata})
        end,
        nil
      )

      on_exit(fn -> :telemetry.detach(id) end)
      :ok
    end

    test "merges enrich_stop metadata into the :stop event" do
      result =
        Telemetry.span([:langchain, :test, :enrich], %{}, fn -> {:ok, :value} end,
          enrich_stop: fn {:ok, v} -> %{extra: v} end
        )

      assert result == {:ok, :value}
      assert_received {:stop, metadata}
      assert metadata.result == {:ok, :value}
      assert metadata.extra == :value
    end

    test "falls back to result-only metadata when enrich_stop raises" do
      # enrich_stop is best-effort: if it raises, span/4 must still emit :stop with
      # just %{result: result} and return the wrapped value unchanged, never
      # letting a bad enrichment crash the operation.
      result =
        Telemetry.span([:langchain, :test, :enrich], %{}, fn -> {:ok, :value} end,
          enrich_stop: fn _ -> raise "boom in enrich" end
        )

      assert result == {:ok, :value}
      assert_received {:stop, metadata}
      assert metadata.result == {:ok, :value}
      refute Map.has_key?(metadata, :extra)
    end
  end

  describe "lifecycle helper wrappers" do
    # These are public helpers on the `LangChain.Telemetry` API surface but are not
    # (yet) emitted anywhere inside the library. This locks in their event names and
    # span-style contract — emit a `:start` event and return a stop function that
    # emits the matching `:stop` — so a rename or arity change is caught.
    test "each start helper emits its :start event and returns a working stop fn" do
      test_pid = self()

      helpers = [
        {&Telemetry.message_process_start/1, [:langchain, :message, :process]},
        {&Telemetry.memory_read_start/1, [:langchain, :memory, :read]},
        {&Telemetry.memory_write_start/1, [:langchain, :memory, :write]},
        {&Telemetry.retriever_get_relevant_documents_start/1,
         [:langchain, :retriever, :get_relevant_documents]}
      ]

      for {start_fun, prefix} <- helpers do
        start_event = prefix ++ [:start]
        stop_event = prefix ++ [:stop]
        id = "telemetry-wrapper-#{Enum.join(prefix, "-")}"

        :telemetry.attach_many(
          id,
          [start_event, stop_event],
          fn event, _measurements, _metadata, _config -> send(test_pid, {:event, event}) end,
          nil
        )

        stop = start_fun.(%{})
        assert is_function(stop, 1)
        assert_received {:event, ^start_event}

        stop.(%{})
        assert_received {:event, ^stop_event}

        :telemetry.detach(id)
      end
    end
  end
end
