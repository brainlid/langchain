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

    test "sums nested raw detail maps per-key at every depth" do
      usage1 =
        TokenUsage.new!(%{
          input: 100,
          output: 10,
          raw: %{
            "prompt_tokens" => 100,
            "prompt_tokens_details" => %{"cached_tokens" => 64, "audio_tokens" => 0},
            "completion_tokens_details" => %{"reasoning_tokens" => 4}
          }
        })

      usage2 =
        TokenUsage.new!(%{
          input: 50,
          output: 5,
          raw: %{
            "prompt_tokens" => 50,
            "prompt_tokens_details" => %{"cached_tokens" => 32, "audio_tokens" => 0},
            "completion_tokens_details" => %{"reasoning_tokens" => 1}
          }
        })

      combined = TokenUsage.add(usage1, usage2)

      assert combined.raw["prompt_tokens"] == 150
      assert combined.raw["prompt_tokens_details"]["cached_tokens"] == 96
      assert combined.raw["prompt_tokens_details"]["audio_tokens"] == 0
      assert combined.raw["completion_tokens_details"]["reasoning_tokens"] == 5
    end

    test "keeps nested keys that only one side reports" do
      usage1 = TokenUsage.new!(%{raw: %{"details" => %{"a" => 1}}})
      usage2 = TokenUsage.new!(%{raw: %{"details" => %{"b" => 2}}})

      combined = TokenUsage.add(usage1, usage2)

      assert combined.raw["details"] == %{"a" => 1, "b" => 2}
    end

    test "a cumulative usage supersedes the accumulator instead of adding to it" do
      running = TokenUsage.new!(%{input: 2679, output: 3, raw: %{"input_tokens" => 2679}})

      total =
        TokenUsage.new!(%{
          input: 10_682,
          output: 510,
          raw: %{"input_tokens" => 10_682},
          cumulative: true
        })

      combined = TokenUsage.add(running, total)

      assert combined.input == 10_682
      assert combined.output == 510
      assert combined.raw["input_tokens"] == 10_682
      assert combined.cumulative
    end

    test "a cumulative usage carries forward the fields it does not report" do
      running =
        TokenUsage.new!(%{
          input: 25,
          output: 1,
          raw: %{"input_tokens" => 25, "cache_read_input_tokens" => 128}
        })

      total = TokenUsage.new!(%{output: 15, raw: %{"output_tokens" => 15}, cumulative: true})

      combined = TokenUsage.add(running, total)

      assert combined.input == 25
      assert combined.output == 15
      assert combined.raw["cache_read_input_tokens"] == 128
      assert combined.raw["output_tokens"] == 15
    end

    test "a cumulative reading that reports zero does not erase an earlier count" do
      # Some adapters normalize an unreported class to zero rather than omitting
      # it. A running total never decreases, so the earlier count stands.
      running =
        TokenUsage.new!(%{
          input: 25,
          output: 1,
          raw: %{"input_tokens" => 25, "cache_creation_input_tokens" => 4096}
        })

      total =
        TokenUsage.new!(%{
          input: 0,
          output: 15,
          raw: %{"input_tokens" => 0, "cache_creation_input_tokens" => 0, "output_tokens" => 15},
          cumulative: true
        })

      combined = TokenUsage.add(running, total)

      assert combined.input == 25
      assert combined.output == 15
      assert combined.raw["input_tokens"] == 25
      assert combined.raw["cache_creation_input_tokens"] == 4096
      assert combined.raw["output_tokens"] == 15
    end

    test "a cumulative reading advances nested counts without summing them" do
      running = TokenUsage.new!(%{raw: %{"details" => %{"cached_tokens" => 64}}})

      total =
        TokenUsage.new!(%{raw: %{"details" => %{"cached_tokens" => 96}}, cumulative: true})

      combined = TokenUsage.add(running, total)

      assert combined.raw["details"]["cached_tokens"] == 96
    end
  end

  describe "clear_cumulative/1" do
    test "clears the flag and passes other values through" do
      usage = TokenUsage.new!(%{input: 10, output: 20, cumulative: true})

      assert %TokenUsage{cumulative: false, input: 10, output: 20} =
               TokenUsage.clear_cumulative(usage)

      assert TokenUsage.clear_cumulative(nil) == nil
    end
  end

  describe "add_total/2" do
    test "sums per-message totals even when they are marked cumulative" do
      first = TokenUsage.new!(%{input: 100, output: 20, cumulative: true})
      second = TokenUsage.new!(%{input: 150, output: 35, cumulative: true})

      total = TokenUsage.add_total(first, second)

      assert total.input == 250
      assert total.output == 55
      refute total.cumulative
    end

    test "handles nil on either side" do
      usage = TokenUsage.new!(%{input: 10, output: 20, cumulative: true})

      assert TokenUsage.add_total(nil, nil) == nil
      assert %TokenUsage{input: 10, cumulative: false} = TokenUsage.add_total(nil, usage)
      assert %TokenUsage{input: 10, cumulative: false} = TokenUsage.add_total(usage, nil)
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
