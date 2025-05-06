defmodule LangChain.TokenUsageTest do
  use ExUnit.Case
  doctest LangChain.TokenUsage, import: true

  alias LangChain.TokenUsage
  alias LangChain.Message
  alias LangChain.MessageDelta

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
      # allow output to be nil because it can come in deltas
      assert nil == changeset.errors[:output]
    end
  end

  describe "new!/1" do
    test "accepts valid data" do
      assert {:ok, %TokenUsage{} = usage} = TokenUsage.new(%{"input" => 1, "output" => 2})

      assert usage.input == 1
      assert usage.output == 2
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
      usage1 =
        TokenUsage.new!(%{
          input: 55,
          output: 4,
          raw: %{
            "cache_creation_input_tokens" => 0,
            "cache_read_input_tokens" => 0,
            "input_tokens" => 55,
            "output_tokens" => 4
          }
        })

      usage2 =
        TokenUsage.new!(%{
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

    test "handles nil arguments" do
      usage = TokenUsage.new!(%{input: 10, output: 20})

      assert TokenUsage.add(nil, nil) == nil
      assert TokenUsage.add(usage, nil) == usage
      assert TokenUsage.add(nil, usage) == usage
    end
  end

  describe "get/1" do
    test "extracts token usage from message metadata" do
      usage = TokenUsage.new!(%{input: 10, output: 20})
      message = %LangChain.Message{metadata: %{usage: usage}}

      assert TokenUsage.get(message) == usage
    end

    test "extracts token usage from message delta metadata" do
      usage = TokenUsage.new!(%{input: 10, output: 20})
      delta = %MessageDelta{metadata: %{usage: usage}}

      assert TokenUsage.get(delta) == usage
    end

    test "returns nil when no usage in metadata" do
      message = %LangChain.Message{metadata: %{}}
      assert TokenUsage.get(message) == nil
    end

    test "returns nil when metadata is nil" do
      message = %LangChain.Message{metadata: nil}
      assert TokenUsage.get(message) == nil
    end

    test "returns nil for invalid struct" do
      assert TokenUsage.get(%{}) == nil
      assert TokenUsage.get(%{metadata: %{}}) == nil
      assert TokenUsage.get(%{metadata: %{usage: "not a token usage"}}) == nil
    end
  end

  describe "set/2" do
    test "sets the token usage on a message" do
      message = %Message{metadata: %{}}
      token_usage = %TokenUsage{input: 10, output: 20}

      assert TokenUsage.set(message, token_usage) == %Message{
               metadata: %{usage: token_usage}
             }
    end

    test "sets the token usage on a message delta" do
      delta = %MessageDelta{metadata: %{}}
      token_usage = %TokenUsage{input: 10, output: 20}

      assert TokenUsage.set(delta, token_usage) == %MessageDelta{
               metadata: %{usage: token_usage}
             }
    end

    test "handles when metadata is nil" do
      message = %Message{metadata: nil}
      token_usage = %TokenUsage{input: 10, output: 20}

      assert TokenUsage.set(message, token_usage) == %Message{
               metadata: %{usage: token_usage}
             }

      #  works on message delta too
      message = %MessageDelta{metadata: nil}
      token_usage = %TokenUsage{input: 10, output: 20}

      assert TokenUsage.set(message, token_usage) == %MessageDelta{
               metadata: %{usage: token_usage}
             }
    end

    test "when no TokenUsage information, returns the original struct" do
      message = %Message{metadata: %{}}
      assert TokenUsage.set(message, nil) == message
    end

    test "doesn't alter any other existing metadata when setting token usage" do
      message = %Message{metadata: %{other: "metadata"}}

      assert TokenUsage.set(message, %TokenUsage{input: 10, output: 20}) == %Message{
               metadata: %{other: "metadata", usage: %TokenUsage{input: 10, output: 20}}
             }
    end
  end

  describe "set_wrapped/2" do
    test "works on :ok wrapped structs when setting token usage" do
      message = {:ok, %Message{metadata: %{}}}

      assert TokenUsage.set_wrapped(message, %TokenUsage{input: 10, output: 20}) ==
               {:ok,
                %Message{
                  metadata: %{usage: %TokenUsage{input: 10, output: 20}}
                }}
    end

    test "works on :error wrapped structs when setting token usage" do
      message = {:error, %{}}
      assert TokenUsage.set_wrapped(message, %TokenUsage{input: 10, output: 20}) == {:error, %{}}
    end
  end
end
