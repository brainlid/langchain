defmodule LangChain.TokenUsageTest do
  use ExUnit.Case
  doctest LangChain.TokenUsage

  alias LangChain.TokenUsage
  alias LangChain.LangChainError

  describe "new/1" do
    test "accepts valid data" do
      assert {:ok, %TokenUsage{} = usage} = TokenUsage.new(%{"input" => 1, "output" => 2})

      assert usage.input == 1
      assert usage.output == 2
    end

    test "returns error when invalid" do
      assert {:error, changeset} = TokenUsage.new(%{"input" => -1, "output" => nil})

      refute changeset.valid?
      assert {"must be greater than or equal to %{number}", _} = changeset.errors[:input]
      assert {"can't be blank", _} = changeset.errors[:output]
    end
  end

  describe "new!/1" do
    test "accepts valid data" do
      assert {:ok, %TokenUsage{} = usage} = TokenUsage.new(%{"input" => 1, "output" => 2})

      assert usage.input == 1
      assert usage.output == 2
    end

    test "raises exception when invalid" do
      assert_raise LangChainError, "output: can't be blank", fn ->
        TokenUsage.new!(%{input: 1})
      end
    end
  end

  describe "total/1" do
    test "returns the sum" do
      usage = TokenUsage.new!(%{input: 1, output: 10})
      assert 11 == TokenUsage.total(usage)
    end
  end
end
