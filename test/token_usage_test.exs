defmodule LangChain.TokenUsageTest do
  use ExUnit.Case
  doctest LangChain.TokenUsage, import: true

  alias LangChain.TokenUsage
  alias LangChain.LangChainError

  describe "new/1" do
    test "accepts valid data" do
      assert {:ok, %TokenUsage{} = usage} =
               TokenUsage.new(%{"input" => 1, "output" => 2, "raw" => %{"total_tokens" => 29}})

      assert usage.input == 1
      assert usage.output == 2
      assert usage.raw == %{"total_tokens" => 29}
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

  describe "add/2" do
    test "combines two token usages" do
      usage1 = TokenUsage.new!(%{input: 10, output: 20, raw: %{"total_tokens" => 30}})
      usage2 = TokenUsage.new!(%{input: 5, output: 15, raw: %{"total_tokens" => 20}})

      combined = TokenUsage.add(usage1, usage2)

      assert combined.input == 15
      assert combined.output == 35
      assert combined.raw["total_tokens"] == 50
    end

    test "handles nil values gracefully" do
      usage1 = TokenUsage.new!(%{input: nil, output: 20, raw: %{"total_tokens" => 30}})
      usage2 = TokenUsage.new!(%{input: 5, output: 15, raw: %{"total_tokens" => 20}})

      combined = TokenUsage.add(usage1, usage2)

      assert combined.input == 5
      assert combined.output == 35
      assert combined.raw["total_tokens"] == 50
    end

    test "merges raw values correctly" do
      usage1 = TokenUsage.new!(%{
        input: 55,
        output: 4,
        raw: %{
          "cache_creation_input_tokens" => 0,
          "cache_read_input_tokens" => 0,
          "input_tokens" => 55,
          "output_tokens" => 4
        }
      })
      usage2 = TokenUsage.new!(%{
        input: 30,
        output: 2,
        raw: %{
          "cache_creation_input_tokens" => 10,
          "cache_read_input_tokens" => 5,
          "input_tokens" => 30,
          "output_tokens" => 2
        }
      })

      combined = TokenUsage.add(usage1, usage2)

      assert combined.input == 85
      assert combined.output == 6
      assert combined.raw["cache_creation_input_tokens"] == 10
      assert combined.raw["cache_read_input_tokens"] == 5
      assert combined.raw["input_tokens"] == 85
      assert combined.raw["output_tokens"] == 6
    end
  end
end
