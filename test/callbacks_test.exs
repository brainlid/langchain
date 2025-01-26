defmodule LangChain.CallbacksTest do
  use LangChain.BaseCase

  doctest LangChain.Callbacks
  alias LangChain.Callbacks
  alias LangChain.LangChainError

  describe "fire/3" do
    test "handles when no callbacks provided" do
      assert :ok == Callbacks.fire([], :custom, ["123"])
    end

    test "fires a single callback handler" do
      handlers = %{custom: fn value -> send(self(), {:callback_custom, value}) end}
      assert :ok == Callbacks.fire([handlers], :custom, ["123"])
      assert_received {:callback_custom, "123"}
    end

    test "does nothing when callback_name not found" do
      handlers = %{custom: fn value -> send(self(), {:callback_custom, value}) end}
      assert :ok == Callbacks.fire([handlers], :missing_name, ["123"])
      refute_received {:callback_custom, "123"}
    end

    test "fires multiple callback handlers" do
      handler1 = %{custom: fn value -> send(self(), {:callback_custom1, value}) end}
      handler2 = %{custom: fn value -> send(self(), {:callback_custom2, value}) end}
      assert :ok == Callbacks.fire([handler1, handler2], :custom, ["123"])
      assert_received {:callback_custom1, "123"}
      assert_received {:callback_custom2, "123"}
    end

    test "handles when a handler errors" do
      handlers = %{custom: fn _value -> raise ArgumentError, "BOOM!" end}

      assert_raise LangChainError,
                   "Callback handler for :custom raised an exception: (ArgumentError) BOOM! at test/callbacks_test.exs:#{__ENV__.line - 3}: anonymous fn/1 in LangChain.CallbacksTest.\"test fire/3 handles when a handler errors\"/1",
                   fn ->
                     Callbacks.fire([handlers], :custom, ["123"])
                   end
    end

    test "raises error when handler is not a function" do
      handlers = %{custom: "invalid"}

      assert_raise LangChainError,
                   "Unexpected callback handler. Callback :custom was assigned \"invalid\"",
                   fn ->
                     Callbacks.fire([handlers], :custom, ["123"])
                   end
    end
  end
end
