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

    test "treats an unreported count as contributing nothing" do
      # Anthropic's closing `message_delta` event carries only `output_tokens`,
      # so a usage built from it alone has no input count to add.
      assert 93 == TokenUsage.total(TokenUsage.new!(%{output: 93}))
      assert 25 == TokenUsage.total(TokenUsage.new!(%{input: 25}))
      assert 0 == TokenUsage.total(TokenUsage.new!(%{}))
    end
  end

  describe "add/2" do
    test "keeps the larger reading of each count" do
      # Two readings of one message: the second reports a further-along picture
      # of the same counters, not a second message's worth of tokens.
      opening = TokenUsage.new!(%{input: 10, output: 1, raw: %{"total_tokens" => 11}})
      closing = TokenUsage.new!(%{input: 10, output: 20, raw: %{"total_tokens" => 30}})

      combined = TokenUsage.add(opening, closing)

      assert combined.input == 10
      assert combined.output == 20
      assert combined.raw["total_tokens"] == 30
    end

    test "handles nil counts on either side" do
      opening = TokenUsage.new!(%{input: nil, output: 20, raw: %{"total_tokens" => 30}})
      closing = TokenUsage.new!(%{input: 5, output: 15, raw: %{"total_tokens" => 20}})

      combined = TokenUsage.add(opening, closing)

      assert combined.input == 5
      assert combined.output == 20
      assert combined.raw["total_tokens"] == 30
    end

    test "handles nil arguments" do
      usage = TokenUsage.new!(%{input: 10, output: 20})

      assert TokenUsage.add(nil, nil) == nil
      assert TokenUsage.add(usage, nil) == usage
      assert TokenUsage.add(nil, usage) == usage
    end

    test "carries forward a class the later reading does not report" do
      # Anthropic's closing `message_delta` can carry only `output_tokens`.
      opening =
        TokenUsage.new!(%{
          input: 25,
          output: 1,
          raw: %{"input_tokens" => 25, "cache_read_input_tokens" => 128}
        })

      closing = TokenUsage.new!(%{output: 15, raw: %{"output_tokens" => 15}})

      combined = TokenUsage.add(opening, closing)

      assert combined.input == 25
      assert combined.output == 15
      assert combined.raw["cache_read_input_tokens"] == 128
      assert combined.raw["output_tokens"] == 15
    end

    test "a reading that reports zero does not erase an earlier count" do
      # Some adapters normalize an unreported class to zero rather than omitting
      # it. A count never decreases while a message generates, so the earlier
      # reading stands.
      opening =
        TokenUsage.new!(%{
          input: 25,
          output: 1,
          raw: %{"input_tokens" => 25, "cache_creation_input_tokens" => 4096}
        })

      closing =
        TokenUsage.new!(%{
          input: 0,
          output: 15,
          raw: %{"input_tokens" => 0, "cache_creation_input_tokens" => 0, "output_tokens" => 15}
        })

      combined = TokenUsage.add(opening, closing)

      assert combined.input == 25
      assert combined.output == 15
      assert combined.raw["input_tokens"] == 25
      assert combined.raw["cache_creation_input_tokens"] == 4096
      assert combined.raw["output_tokens"] == 15
    end

    test "no count exceeds the largest single reading that reports it" do
      readings = [
        TokenUsage.new!(%{
          input: 10,
          output: 1,
          raw: %{"input_tokens" => 10, "cache_creation_input_tokens" => 31_350}
        }),
        TokenUsage.new!(%{
          input: 10,
          output: 93,
          raw: %{"input_tokens" => 10, "cache_creation_input_tokens" => 31_350}
        })
      ]

      combined = Enum.reduce(readings, nil, &TokenUsage.add(&2, &1))

      for key <- ["input_tokens", "cache_creation_input_tokens"] do
        largest = readings |> Enum.map(& &1.raw[key]) |> Enum.max()

        assert combined.raw[key] == largest
      end
    end

    test "advances nested raw detail maps per-key at every depth" do
      opening =
        TokenUsage.new!(%{
          input: 100,
          output: 10,
          raw: %{
            "prompt_tokens" => 100,
            "prompt_tokens_details" => %{"cached_tokens" => 64, "audio_tokens" => 0},
            "completion_tokens_details" => %{"reasoning_tokens" => 4}
          }
        })

      closing =
        TokenUsage.new!(%{
          input: 100,
          output: 15,
          raw: %{
            "prompt_tokens" => 100,
            "prompt_tokens_details" => %{"cached_tokens" => 64, "audio_tokens" => 0},
            "completion_tokens_details" => %{"reasoning_tokens" => 9}
          }
        })

      combined = TokenUsage.add(opening, closing)

      assert combined.raw["prompt_tokens"] == 100
      assert combined.raw["prompt_tokens_details"]["cached_tokens"] == 64
      assert combined.raw["prompt_tokens_details"]["audio_tokens"] == 0
      assert combined.raw["completion_tokens_details"]["reasoning_tokens"] == 9
    end

    test "keeps nested keys that only one side reports" do
      usage1 = TokenUsage.new!(%{raw: %{"details" => %{"a" => 1}}})
      usage2 = TokenUsage.new!(%{raw: %{"details" => %{"b" => 2}}})

      combined = TokenUsage.add(usage1, usage2)

      assert combined.raw["details"] == %{"a" => 1, "b" => 2}
    end

    test "takes the later value for a struct rather than merging it key-by-key" do
      # Structs are maps, so recursing into one would build a malformed struct
      # out of two. Whatever a provider parked in `:raw` arrives intact.
      usage1 = TokenUsage.new!(%{raw: %{"at" => ~D[2024-01-01], "count" => 1}})
      usage2 = TokenUsage.new!(%{raw: %{"at" => ~D[2024-06-01], "count" => 2}})

      combined = TokenUsage.add(usage1, usage2)

      assert combined.raw["at"] == ~D[2024-06-01]
      assert combined.raw["count"] == 2
    end

    test "takes a list-valued detail from the later usage whole" do
      # Gemini reports the per-modality breakdown as a list of objects rather
      # than a keyed map, which offers nothing to advance against.
      details = fn n -> [%{"modality" => "TEXT", "tokenCount" => n}] end

      usage1 =
        TokenUsage.new!(%{
          raw: %{"promptTokenCount" => 10, "promptTokensDetails" => details.(10)}
        })

      usage2 =
        TokenUsage.new!(%{
          raw: %{"promptTokenCount" => 20, "promptTokensDetails" => details.(20)}
        })

      combined = TokenUsage.add(usage1, usage2)

      assert combined.raw["promptTokenCount"] == 20
      assert combined.raw["promptTokensDetails"] == details.(20)
    end

    test "a raw value derived within one reading does not survive the merge" do
      # An adapter that computes `total_tokens` as this reading's input + output
      # describes only that reading. Keeping the larger of two such values is
      # not the total of the merged counts, which is what `total/1` reports.
      opening = TokenUsage.new!(%{input: 25, output: 1, raw: %{"total_tokens" => 26}})
      closing = TokenUsage.new!(%{input: 0, output: 15, raw: %{"total_tokens" => 15}})

      combined = TokenUsage.add(opening, closing)

      assert TokenUsage.total(combined) == 40
      assert combined.raw["total_tokens"] == 26
    end
  end

  describe "add_total/2" do
    test "sums the counts of two messages" do
      first = TokenUsage.new!(%{input: 100, output: 20})
      second = TokenUsage.new!(%{input: 150, output: 35})

      total = TokenUsage.add_total(first, second)

      assert total.input == 250
      assert total.output == 55
    end

    test "handles nil on either side" do
      usage = TokenUsage.new!(%{input: 10, output: 20})

      assert TokenUsage.add_total(nil, nil) == nil
      assert %TokenUsage{input: 10} = TokenUsage.add_total(nil, usage)
      assert %TokenUsage{input: 10} = TokenUsage.add_total(usage, nil)
    end

    test "treats an unreported count as contributing nothing" do
      first = TokenUsage.new!(%{input: nil, output: 20})
      second = TokenUsage.new!(%{input: 5, output: 15})

      total = TokenUsage.add_total(first, second)

      assert total.input == 5
      assert total.output == 35
    end

    test "sums raw values, including nested detail maps, at every depth" do
      first =
        TokenUsage.new!(%{
          input: 100,
          output: 10,
          raw: %{
            "prompt_tokens" => 100,
            "prompt_tokens_details" => %{"cached_tokens" => 64, "audio_tokens" => 0},
            "completion_tokens_details" => %{"reasoning_tokens" => 4}
          }
        })

      second =
        TokenUsage.new!(%{
          input: 50,
          output: 5,
          raw: %{
            "prompt_tokens" => 50,
            "prompt_tokens_details" => %{"cached_tokens" => 32, "audio_tokens" => 0},
            "completion_tokens_details" => %{"reasoning_tokens" => 1}
          }
        })

      total = TokenUsage.add_total(first, second)

      assert total.raw["prompt_tokens"] == 150
      assert total.raw["prompt_tokens_details"]["cached_tokens"] == 96
      assert total.raw["prompt_tokens_details"]["audio_tokens"] == 0
      assert total.raw["completion_tokens_details"]["reasoning_tokens"] == 5
    end

    test "takes the later value for a struct rather than merging it key-by-key" do
      first = TokenUsage.new!(%{raw: %{"at" => ~D[2024-01-01], "count" => 1}})
      second = TokenUsage.new!(%{raw: %{"at" => ~D[2024-06-01], "count" => 2}})

      total = TokenUsage.add_total(first, second)

      assert total.raw["at"] == ~D[2024-06-01]
      assert total.raw["count"] == 3
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
