defmodule LangChain.LangChainErrorTest do
  use ExUnit.Case
  doctest LangChain.LangChainError

  alias LangChain.LangChainError

  describe "exception/1" do
    test "supports creating with keyword list" do
      original = RuntimeError.exception("testing")

      error = LangChainError.exception(type: "test", message: "A test error", original: original)

      assert error.type == "test"
      assert error.message == "A test error"
      assert error.original == original
    end
  end
end
