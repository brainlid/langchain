defmodule LangChain.ChatModels.ChatAwsMantleTest do
  use LangChain.BaseCase

  alias LangChain.ChatModels.ChatAwsMantle
  alias LangChain.Function
  alias LangChain.FunctionParam
  alias LangChain.LangChainError
  alias LangChain.Message
  alias LangChain.Message.ContentPart
  alias LangChain.Message.ToolCall
  alias LangChain.MessageDelta
  alias LangChain.TokenUsage

  @kimi_model "moonshotai.kimi-k2.5"

  describe "new/1 — schema and validation" do
    test "succeeds with model + region + api_key (Bearer auth)" do
      assert {:ok, %ChatAwsMantle{} = m} =
               ChatAwsMantle.new(%{
                 model: @kimi_model,
                 region: "us-east-1",
                 api_key: "test-key"
               })

      assert m.model == @kimi_model
      assert m.region == "us-east-1"
      assert m.api_key == "test-key"
      assert m.credentials == nil
      assert m.receive_timeout == 120_000
      assert m.temperature == 1.0
      assert m.stream == false
    end

    test "succeeds with model + region + credentials (SigV4 auth)" do
      creds_fn = fn ->
        [access_key_id: "AKIA...", secret_access_key: "secret"]
      end

      assert {:ok, %ChatAwsMantle{} = m} =
               ChatAwsMantle.new(%{
                 model: @kimi_model,
                 region: "us-east-1",
                 credentials: creds_fn
               })

      assert m.credentials == creds_fn
      assert m.api_key == nil
    end

    test "fails when neither api_key nor credentials is set" do
      assert {:error, changeset} =
               ChatAwsMantle.new(%{model: @kimi_model, region: "us-east-1"})

      assert {"must set either :api_key (Bearer) or :credentials (SigV4)", _} =
               changeset.errors[:api_key]
    end

    test "fails when both api_key and credentials are set" do
      assert {:error, changeset} =
               ChatAwsMantle.new(%{
                 model: @kimi_model,
                 region: "us-east-1",
                 api_key: "k",
                 credentials: fn -> [] end
               })

      assert {"cannot set both :api_key and :credentials" <> _, _} =
               changeset.errors[:api_key]
    end

    test "fails when neither :endpoint nor :region is set" do
      assert {:error, changeset} =
               ChatAwsMantle.new(%{model: @kimi_model, api_key: "k"})

      assert {_, _} = changeset.errors[:region]
    end

    test "succeeds with :endpoint override and no :region" do
      assert {:ok, %ChatAwsMantle{}} =
               ChatAwsMantle.new(%{
                 model: @kimi_model,
                 endpoint: "https://example.com/v1/chat/completions",
                 api_key: "k"
               })
    end

    test "rejects an invalid :reasoning_effort value" do
      assert {:error, changeset} =
               ChatAwsMantle.new(%{
                 model: @kimi_model,
                 region: "us-east-1",
                 api_key: "k",
                 reasoning_effort: "extreme"
               })

      assert {"must be one of: low, medium, high", _} = changeset.errors[:reasoning_effort]
    end

    test "accepts valid reasoning_effort values" do
      for effort <- ~w(low medium high) do
        assert {:ok, %ChatAwsMantle{reasoning_effort: ^effort}} =
                 ChatAwsMantle.new(%{
                   model: @kimi_model,
                   region: "us-east-1",
                   api_key: "k",
                   reasoning_effort: effort
                 })
      end
    end

    test "new!/1 raises on validation failure" do
      assert_raise LangChainError, fn -> ChatAwsMantle.new!(%{}) end
    end
  end

  describe "url/1" do
    test "builds the Mantle URL from :region when no endpoint override" do
      m = ChatAwsMantle.new!(%{model: @kimi_model, region: "us-west-2", api_key: "k"})

      assert ChatAwsMantle.url(m) ==
               "https://bedrock-mantle.us-west-2.api.aws/v1/chat/completions"
    end

    test "honors :endpoint override" do
      m =
        ChatAwsMantle.new!(%{
          model: @kimi_model,
          endpoint: "https://custom.example/v1/chat",
          api_key: "k"
        })

      assert ChatAwsMantle.url(m) == "https://custom.example/v1/chat"
    end
  end

  describe "auth_opts/1" do
    test "returns Bearer auth when api_key is set" do
      m = ChatAwsMantle.new!(%{model: @kimi_model, region: "us-east-1", api_key: "secret"})

      assert [auth: {:bearer, "secret"}] = ChatAwsMantle.auth_opts(m)
    end

    test "returns SigV4 keyword list when credentials are set" do
      creds = fn ->
        [access_key_id: "AKIA", secret_access_key: "S", token: "T"]
      end

      m = ChatAwsMantle.new!(%{model: @kimi_model, region: "ap-south-1", credentials: creds})

      assert [aws_sigv4: opts] = ChatAwsMantle.auth_opts(m)
      assert opts[:region] == "ap-south-1"
      assert opts[:service] == :bedrock
      assert opts[:access_key_id] == "AKIA"
      assert opts[:secret_access_key] == "S"
      assert opts[:token] == "T"
    end
  end

  describe "for_api/3" do
    setup do
      m =
        ChatAwsMantle.new!(%{
          model: @kimi_model,
          region: "us-east-1",
          api_key: "k",
          temperature: 0.0,
          max_tokens: 64
        })

      %{model: m}
    end

    test "produces an OpenAI-shaped request body", %{model: m} do
      body = ChatAwsMantle.for_api(m, [Message.new_user!("hi")], [])

      assert body.model == @kimi_model
      assert body.stream == false
      assert body.temperature == 0.0
      assert body.max_tokens == 64
      assert is_list(body.messages)
      assert [%{} = msg] = body.messages
      assert msg["role"] == :user
    end

    test "includes :reasoning_effort when set", %{model: %ChatAwsMantle{} = m} do
      m_with_reasoning = %ChatAwsMantle{m | reasoning_effort: "high"}
      body = ChatAwsMantle.for_api(m_with_reasoning, [Message.new_user!("hi")], [])
      assert body.reasoning_effort == "high"
    end

    test "omits :reasoning_effort when not set", %{model: m} do
      body = ChatAwsMantle.for_api(m, [Message.new_user!("hi")], [])
      refute Map.has_key?(body, :reasoning_effort)
    end

    test "passes :top_p, :frequency_penalty, :presence_penalty through to the body when set", %{
      model: m
    } do
      updated = %{m | top_p: 0.9, frequency_penalty: 0.5, presence_penalty: 0.2}
      body = ChatAwsMantle.for_api(updated, [Message.new_user!("hi")], [])
      assert body.top_p == 0.9
      assert body.frequency_penalty == 0.5
      assert body.presence_penalty == 0.2
    end

    test "omits the sampling knobs when not set", %{model: m} do
      body = ChatAwsMantle.for_api(m, [Message.new_user!("hi")], [])
      refute Map.has_key?(body, :top_p)
      refute Map.has_key?(body, :frequency_penalty)
      refute Map.has_key?(body, :presence_penalty)
    end

    test "strips :thinking ContentParts from assistant messages before serialization", %{model: m} do
      # Mantle's wire format has no representation for thinking blocks.
      # ChatAwsMantle surfaces them from delta.reasoning for UI display, but
      # on the way back out (when a multi-turn conversation re-sends the
      # assistant message as history), they must be filtered or ChatOpenAI's
      # content_part_for_api/2 crashes (no clause for :thinking).
      history = [
        Message.new_user!("What model are you?"),
        %Message{
          role: :assistant,
          status: :complete,
          content: [
            ContentPart.thinking!("I should answer with my model name. Let me think..."),
            ContentPart.text!("I'm gpt-oss-120b.")
          ]
        },
        Message.new_user!("Great, what files do I have?")
      ]

      # Should not raise — crashes pre-fix because the assistant message has
      # a thinking part that ChatOpenAI can't serialize.
      body = ChatAwsMantle.for_api(m, history, [])

      assistant_serialized = Enum.at(body.messages, 1)

      content =
        Map.get(assistant_serialized, "content") || Map.get(assistant_serialized, :content)

      # Thinking was stripped; text survived.
      assert [%{"type" => "text", "text" => "I'm gpt-oss-120b."}] = content
    end

    test "strips :unsupported ContentParts (e.g. redacted_thinking) from assistant messages before serialization",
         %{model: m} do
      # Anthropic returns `redacted_thinking` blocks when extended thinking is
      # enabled but the content is encrypted. LangChain stores these as
      # %ContentPart{type: :unsupported, options: [type: "redacted_thinking"]}.
      # If such a message is sent back to Mantle as history, the :unsupported
      # part must be stripped — ChatOpenAI.content_part_for_api/2 has no clause
      # for it and will crash.
      redacted_thinking = %ContentPart{
        type: :unsupported,
        content: "<encrypted_thinking_data>",
        options: [type: "redacted_thinking"]
      }

      history = [
        Message.new_user!("Think hard about this."),
        %Message{
          role: :assistant,
          status: :complete,
          content: [
            redacted_thinking,
            ContentPart.text!("The answer is 42.")
          ]
        },
        Message.new_user!("Are you sure?")
      ]

      body = ChatAwsMantle.for_api(m, history, [])

      assistant_serialized = Enum.at(body.messages, 1)

      content =
        Map.get(assistant_serialized, "content") || Map.get(assistant_serialized, :content)

      # :unsupported was stripped; text survived.
      assert [%{"type" => "text", "text" => "The answer is 42."}] = content
    end
  end

  describe "new/1 — sampling knob validation" do
    test "rejects out-of-range :frequency_penalty" do
      assert {:error, changeset} =
               ChatAwsMantle.new(%{
                 model: @kimi_model,
                 region: "us-east-1",
                 api_key: "k",
                 frequency_penalty: 3.0
               })

      assert changeset.errors[:frequency_penalty]
    end

    test "rejects out-of-range :top_p" do
      assert {:error, changeset} =
               ChatAwsMantle.new(%{
                 model: @kimi_model,
                 region: "us-east-1",
                 api_key: "k",
                 top_p: 1.5
               })

      assert changeset.errors[:top_p]
    end
  end

  describe "do_process_response/2 — reasoning extraction" do
    setup do
      m = ChatAwsMantle.new!(%{model: @kimi_model, region: "us-east-1", api_key: "k"})
      %{model: m}
    end

    test "extracts message.reasoning into a leading thinking ContentPart", %{model: m} do
      body = %{
        "choices" => [
          %{
            "finish_reason" => "stop",
            "index" => 0,
            "message" => %{
              "role" => "assistant",
              "content" => "The answer is Friday.",
              "reasoning" => "100 mod 7 = 2, so Wednesday + 2 = Friday."
            }
          }
        ],
        "usage" => %{"prompt_tokens" => 49, "completion_tokens" => 30, "total_tokens" => 79}
      }

      assert [%Message{} = msg] = ChatAwsMantle.do_process_response(m, body)
      assert msg.role == :assistant

      assert [
               %ContentPart{
                 type: :thinking,
                 content: "100 mod 7 = 2, so Wednesday + 2 = Friday."
               },
               %ContentPart{type: :text, content: "The answer is Friday."}
             ] = msg.content

      assert %TokenUsage{input: 49, output: 30} = msg.metadata.usage
    end

    test "leaves messages without reasoning unchanged", %{model: m} do
      body = %{
        "choices" => [
          %{
            "finish_reason" => "stop",
            "index" => 0,
            "message" => %{"role" => "assistant", "content" => "Hi!"}
          }
        ],
        "usage" => %{"prompt_tokens" => 5, "completion_tokens" => 2, "total_tokens" => 7}
      }

      assert [%Message{} = msg] = ChatAwsMantle.do_process_response(m, body)
      assert [%ContentPart{type: :text, content: "Hi!"}] = msg.content
      refute Enum.any?(msg.content, &(&1.type == :thinking))
    end

    test "returns an error tuple on Mantle error envelope", %{model: m} do
      body = %{"error" => %{"message" => "bad token", "type" => "auth_error"}}

      assert {:error, %LangChainError{message: "bad token"}} =
               ChatAwsMantle.do_process_response(m, body)
    end
  end

  # ---------------------------------------------------------------------------
  # Live integration smoke test — exercises the full module against Mantle.
  # Mirrors the first smoke test from aws_mantle_smoke_test.exs but routed
  # through ChatAwsMantle to prove the new module works end-to-end.
  # ---------------------------------------------------------------------------
  describe "live: end-to-end through ChatAwsMantle" do
    @tag live_call: true, live_aws_mantle: true, timeout: 180_000
    test "non-streaming completion returns expected text" do
      api_key = System.fetch_env!("AWS_BEARER_TOKEN_BEDROCK")

      model =
        ChatAwsMantle.new!(%{
          model: @kimi_model,
          region: "us-east-1",
          api_key: api_key,
          temperature: 0.0,
          max_tokens: 32,
          verbose_api: true
        })

      assert {:ok, [%Message{role: :assistant, content: content} = msg]} =
               ChatAwsMantle.call(model, [Message.new_user!("Reply with the single word 'pong'.")])

      IO.inspect(msg, label: "ChatAwsMantle MESSAGE")

      text = ContentPart.parts_to_string(content)
      assert text =~ ~r/pong/i
      assert %TokenUsage{} = msg.metadata.usage
    end

    @tag live_call: true, live_aws_mantle: true, timeout: 180_000
    test "with reasoning_effort: high, response includes a thinking ContentPart" do
      api_key = System.fetch_env!("AWS_BEARER_TOKEN_BEDROCK")

      model =
        ChatAwsMantle.new!(%{
          model: @kimi_model,
          region: "us-east-1",
          api_key: api_key,
          temperature: 1.0,
          max_tokens: 512,
          reasoning_effort: "high",
          verbose_api: true
        })

      assert {:ok, [%Message{role: :assistant} = msg]} =
               ChatAwsMantle.call(model, [
                 Message.new_user!(
                   "If today is Wednesday, what day is it in 100 days? Show your work."
                 )
               ])

      IO.inspect(msg.content, label: "ChatAwsMantle MULTIPART CONTENT")

      thinking_parts = Enum.filter(msg.content, &(&1.type == :thinking))
      text_parts = Enum.filter(msg.content, &(&1.type == :text))

      assert length(thinking_parts) >= 1, "expected at least one :thinking content part"
      assert length(text_parts) >= 1, "expected at least one :text content part"

      [thinking | _] = thinking_parts
      assert is_binary(thinking.content)
      assert String.length(thinking.content) > 20

      text = ContentPart.parts_to_string(text_parts)
      assert text =~ ~r/friday/i
    end
  end

  # ---------------------------------------------------------------------------
  # Unit tests — streaming delta parsing. No network calls; we feed raw SSE
  # JSON structures through do_process_response/2 and verify the MessageDelta
  # structures produced (and the merged Message after running the full delta
  # sequence through MessageDelta.merge_deltas/1).
  # ---------------------------------------------------------------------------
  describe "do_process_response/2 — streaming delta parsing" do
    setup do
      m = ChatAwsMantle.new!(%{model: @kimi_model, region: "us-east-1", api_key: "k"})
      %{model: m}
    end

    test "role-only opening delta produces a role MessageDelta", %{model: m} do
      chunk = %{
        "choices" => [
          %{"delta" => %{"role" => "assistant", "content" => nil}, "index" => 0}
        ]
      }

      assert %MessageDelta{role: :assistant, status: :incomplete, content: nil} =
               ChatAwsMantle.do_process_response(m, chunk)
    end

    test "content delta places text at index 1", %{model: m} do
      chunk = %{
        "choices" => [%{"delta" => %{"content" => "Hello"}, "index" => 0}]
      }

      assert %MessageDelta{content: "Hello", index: 1, status: :incomplete} =
               ChatAwsMantle.do_process_response(m, chunk)
    end

    test "reasoning delta becomes a thinking ContentPart at index 0", %{model: m} do
      chunk = %{
        "choices" => [%{"delta" => %{"reasoning" => "hmm, thinking"}, "index" => 0}]
      }

      assert %MessageDelta{
               content: %ContentPart{type: :thinking, content: "hmm, thinking"},
               index: 0,
               status: :incomplete
             } = ChatAwsMantle.do_process_response(m, chunk)
    end

    test "combined reasoning + content in one chunk emits two deltas in order", %{model: m} do
      chunk = %{
        "choices" => [
          %{"delta" => %{"reasoning" => "r", "content" => "c"}, "index" => 0}
        ]
      }

      assert [
               %MessageDelta{
                 content: %ContentPart{type: :thinking, content: "r"},
                 index: 0
               },
               %MessageDelta{content: "c", index: 1}
             ] = ChatAwsMantle.do_process_response(m, chunk)
    end

    test "terminal delta with finish_reason: stop produces :complete status", %{model: m} do
      chunk = %{
        "choices" => [%{"delta" => %{}, "finish_reason" => "stop", "index" => 0}]
      }

      assert %MessageDelta{status: :complete} = ChatAwsMantle.do_process_response(m, chunk)
    end

    test "usage-only terminal event is :skip", %{model: m} do
      chunk = %{
        "choices" => [],
        "usage" => %{"prompt_tokens" => 10, "completion_tokens" => 5, "total_tokens" => 15}
      }

      assert :skip = ChatAwsMantle.do_process_response(m, chunk)
    end

    test "tool_call delta carries a ToolCall in the MessageDelta.tool_calls field", %{model: m} do
      chunk = %{
        "choices" => [
          %{
            "delta" => %{
              "tool_calls" => [
                %{
                  "index" => 0,
                  "id" => "call_abc",
                  "type" => "function",
                  "function" => %{"name" => "get_weather", "arguments" => ""}
                }
              ]
            },
            "index" => 0
          }
        ]
      }

      assert %MessageDelta{tool_calls: [%ToolCall{} = tc]} =
               ChatAwsMantle.do_process_response(m, chunk)

      assert tc.name == "get_weather"
      assert tc.call_id == "call_abc"
      assert tc.index == 0
    end
  end

  describe "streaming integration — merging a simulated stream" do
    setup do
      m = ChatAwsMantle.new!(%{model: @kimi_model, region: "us-east-1", api_key: "k"})
      %{model: m}
    end

    test "reasoning chunks followed by content chunks produce [thinking, text] message", %{
      model: m
    } do
      # Simulate Mantle's observed streaming shape:
      #   role → reasoning fragments → content fragments → terminal stop
      chunks = [
        %{"choices" => [%{"delta" => %{"role" => "assistant"}, "index" => 0}]},
        %{"choices" => [%{"delta" => %{"reasoning" => "100 mod 7 = 2. "}, "index" => 0}]},
        %{"choices" => [%{"delta" => %{"reasoning" => "So Wed + 2 = Fri."}, "index" => 0}]},
        %{"choices" => [%{"delta" => %{"content" => "The answer "}, "index" => 0}]},
        %{"choices" => [%{"delta" => %{"content" => "is Friday."}, "index" => 0}]},
        %{"choices" => [%{"delta" => %{}, "finish_reason" => "stop", "index" => 0}]}
      ]

      deltas =
        chunks
        |> Enum.map(&ChatAwsMantle.do_process_response(m, &1))
        |> List.flatten()

      merged = MessageDelta.merge_deltas(deltas)
      assert {:ok, %Message{} = msg} = MessageDelta.to_message(merged)

      assert [
               %ContentPart{type: :thinking, content: "100 mod 7 = 2. So Wed + 2 = Fri."},
               %ContentPart{type: :text, content: "The answer is Friday."}
             ] = msg.content

      assert msg.role == :assistant
      assert msg.status == :complete
    end

    test "content-only stream (no reasoning) still merges into a single text part", %{model: m} do
      chunks = [
        %{"choices" => [%{"delta" => %{"role" => "assistant"}, "index" => 0}]},
        %{"choices" => [%{"delta" => %{"content" => "pong"}, "index" => 0}]},
        %{"choices" => [%{"delta" => %{}, "finish_reason" => "stop", "index" => 0}]}
      ]

      deltas =
        chunks
        |> Enum.map(&ChatAwsMantle.do_process_response(m, &1))
        |> List.flatten()

      merged = MessageDelta.merge_deltas(deltas)

      assert {:ok, %Message{content: [%ContentPart{type: :text, content: "pong"}]}} =
               MessageDelta.to_message(merged)
    end

    test "streaming tool_calls accumulate across chunks into a complete ToolCall", %{model: m} do
      # Mirrors Mantle's tool-call stream: role → call init → fragmented args → stop.
      chunks = [
        %{"choices" => [%{"delta" => %{"role" => "assistant"}, "index" => 0}]},
        %{
          "choices" => [
            %{
              "delta" => %{
                "tool_calls" => [
                  %{
                    "index" => 0,
                    "id" => "functions.get_weather:0",
                    "type" => "function",
                    "function" => %{"name" => "get_weather", "arguments" => ""}
                  }
                ]
              },
              "index" => 0
            }
          ]
        },
        %{
          "choices" => [
            %{
              "delta" => %{
                "tool_calls" => [
                  %{"index" => 0, "function" => %{"arguments" => "{\"city\":"}}
                ]
              },
              "index" => 0
            }
          ]
        },
        %{
          "choices" => [
            %{
              "delta" => %{
                "tool_calls" => [
                  %{"index" => 0, "function" => %{"arguments" => "\"Moab\",\"state\":\"UT\"}"}}
                ]
              },
              "index" => 0
            }
          ]
        },
        %{"choices" => [%{"delta" => %{}, "finish_reason" => "tool_calls", "index" => 0}]}
      ]

      deltas =
        chunks
        |> Enum.map(&ChatAwsMantle.do_process_response(m, &1))
        |> List.flatten()

      merged = MessageDelta.merge_deltas(deltas)
      assert {:ok, %Message{} = msg} = MessageDelta.to_message(merged)
      assert [%ToolCall{} = call] = msg.tool_calls
      assert call.name == "get_weather"
      assert call.call_id == "functions.get_weather:0"
      assert call.arguments == %{"city" => "Moab", "state" => "UT"}
      assert call.status == :complete
    end
  end

  # ---------------------------------------------------------------------------
  # Live streaming tests — end-to-end through ChatAwsMantle against Mantle.
  # ---------------------------------------------------------------------------
  describe "live: streaming through ChatAwsMantle" do
    # Callbacks are not part of the cast fields (library convention — they're
    # runtime handlers, not config). Set via struct update after new!/1.
    defp with_callbacks(%ChatAwsMantle{} = m, handlers),
      do: %ChatAwsMantle{m | callbacks: handlers}

    defp delta_capture_handler(test_pid) do
      %{
        on_llm_new_delta: fn deltas when is_list(deltas) ->
          Enum.each(deltas, fn %MessageDelta{} = d -> send(test_pid, {:delta, d}) end)
        end
      }
    end

    @tag live_call: true, live_aws_mantle: true, timeout: 180_000
    test "basic streaming yields deltas that merge into a non-empty text message" do
      # Tests the streaming wire format end-to-end: deltas arrive via callback,
      # accumulate, and merge to a Message with text content. Intentionally
      # does not assert on Kimi's specific wording — the model is occasionally
      # flaky on exact outputs. The structural assertions are what matter here.
      model =
        ChatAwsMantle.new!(%{
          model: @kimi_model,
          region: "us-east-1",
          api_key: System.fetch_env!("AWS_BEARER_TOKEN_BEDROCK"),
          stream: true,
          temperature: 0.2,
          max_tokens: 128
        })
        |> with_callbacks([delta_capture_handler(self())])

      assert {:ok, result} =
               ChatAwsMantle.call(model, [
                 Message.new_user!("Say hello in three words or less.")
               ])

      deltas = List.flatten(result)
      assert length(deltas) >= 2
      assert Enum.all?(deltas, &match?(%MessageDelta{}, &1))
      assert_received {:delta, %MessageDelta{}}

      merged = MessageDelta.merge_deltas(deltas)
      {:ok, %Message{role: :assistant} = msg} = MessageDelta.to_message(merged)

      text = ContentPart.parts_to_string(msg.content)
      assert is_binary(text)
      assert String.length(text) > 0
    end

    @tag live_call: true, live_aws_mantle: true, timeout: 180_000
    test "streaming with reasoning_effort: high yields a merged message with thinking + text" do
      model =
        ChatAwsMantle.new!(%{
          model: @kimi_model,
          region: "us-east-1",
          api_key: System.fetch_env!("AWS_BEARER_TOKEN_BEDROCK"),
          stream: true,
          temperature: 1.0,
          # Reasoning can consume the full budget on hard prompts; allot enough
          # headroom for the model to finish thinking AND emit visible content.
          max_tokens: 2048,
          reasoning_effort: "high"
        })
        |> with_callbacks([delta_capture_handler(self())])

      assert {:ok, result} =
               ChatAwsMantle.call(model, [
                 Message.new_user!(
                   "If today is Wednesday, what day is it in 100 days? Answer briefly."
                 )
               ])

      deltas = List.flatten(result)
      assert length(deltas) >= 2

      thinking_deltas =
        Enum.filter(deltas, fn
          %MessageDelta{content: %ContentPart{type: :thinking}} -> true
          _ -> false
        end)

      assert length(thinking_deltas) >= 1,
             "expected at least one thinking ContentPart in streamed deltas"

      merged = MessageDelta.merge_deltas(deltas)
      {:ok, %Message{} = msg} = MessageDelta.to_message(merged)

      thinking_parts = Enum.filter(msg.content, &(&1.type == :thinking))
      text_parts = Enum.filter(msg.content, &(&1.type == :text))

      assert length(thinking_parts) >= 1
      assert length(text_parts) >= 1

      [thinking | _] = thinking_parts
      assert String.length(thinking.content) > 20

      text = ContentPart.parts_to_string(text_parts)
      assert text =~ ~r/friday/i
    end

    @tag live_call: true, live_aws_mantle: true, timeout: 180_000
    test "streaming with tool calls accumulates into a complete tool_call message" do
      weather =
        Function.new!(%{
          name: "get_weather",
          description: "Get the current weather in a given US location",
          parameters: [
            FunctionParam.new!(%{name: "city", type: "string", required: true}),
            FunctionParam.new!(%{name: "state", type: "string", required: true})
          ],
          function: fn _args, _ctx -> {:ok, "75 degrees and sunny"} end
        })

      model =
        ChatAwsMantle.new!(%{
          model: @kimi_model,
          region: "us-east-1",
          api_key: System.fetch_env!("AWS_BEARER_TOKEN_BEDROCK"),
          stream: true,
          temperature: 0.2,
          max_tokens: 256
        })
        |> with_callbacks([delta_capture_handler(self())])

      {:ok, result} =
        ChatAwsMantle.call(
          model,
          [Message.new_user!("What's the weather in Moab, Utah?")],
          [weather]
        )

      deltas = List.flatten(result)
      assert length(deltas) >= 2
      assert_received {:delta, %MessageDelta{}}

      tool_call_deltas =
        Enum.filter(deltas, fn d -> d.tool_calls not in [nil, []] end)

      assert length(tool_call_deltas) > 0

      merged = MessageDelta.merge_deltas(deltas)
      {:ok, %Message{} = msg} = MessageDelta.to_message(merged)

      assert [%ToolCall{} = call] = msg.tool_calls
      assert call.name == "get_weather"
      assert call.status == :complete
      # Kimi sometimes returns "UT" and sometimes "Utah" for state — accept either.
      assert %{"city" => "Moab", "state" => state} = call.arguments
      assert state =~ ~r/^(UT|Utah)$/i
    end
  end

  # ---------------------------------------------------------------------------
  # Live multimodal — K2.5 is a natively multimodal model. This test verifies
  # end-to-end image input through the library: Message with text + image
  # ContentParts → for_api serialization via ChatOpenAI helpers → Mantle →
  # vision-aware response.
  # ---------------------------------------------------------------------------
  describe "live: multimodal through ChatAwsMantle" do
    @tag live_call: true, live_aws_mantle: true, timeout: 180_000
    test "K2.5 recognizes the subject of a real JPG via base64 image ContentPart" do
      image_path = Path.expand("../support/images/barn_owl.jpg", __DIR__)
      {:ok, image_bytes} = File.read(image_path)
      image_b64 = Base.encode64(image_bytes)

      model =
        ChatAwsMantle.new!(%{
          model: @kimi_model,
          region: "us-east-1",
          api_key: System.fetch_env!("AWS_BEARER_TOKEN_BEDROCK"),
          temperature: 0.2,
          max_tokens: 128
        })

      user_message =
        Message.new_user!([
          ContentPart.text!("What kind of bird is in this image? Reply in one or two words."),
          ContentPart.image!(image_b64, media: :jpeg)
        ])

      assert {:ok, [%Message{role: :assistant, content: content} = msg]} =
               ChatAwsMantle.call(model, [user_message])

      text = ContentPart.parts_to_string(content)
      IO.inspect(text, label: "ChatAwsMantle MULTIMODAL RESPONSE")
      assert is_binary(text)
      assert String.length(text) > 0

      # Owl recognition is the functional proof that Kimi actually saw the image.
      # Accept either "owl" (most likely) or "barn owl" (more specific) so the
      # test isn't overly sensitive to Kimi's wording.
      assert text =~ ~r/owl/i

      assert %TokenUsage{} = msg.metadata.usage
    end
  end
end
