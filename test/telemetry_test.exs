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
end
