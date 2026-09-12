defmodule LangChain.ChatModels.ChatFireworksTest do
  use LangChain.BaseCase
  use Mimic

  alias LangChain.ChatModels.ChatFireworks
  alias LangChain.Chains.LLMChain
  alias LangChain.Function
  alias LangChain.FunctionParam
  alias LangChain.LangChainError
  alias LangChain.Message
  alias LangChain.Message.ContentPart
  alias LangChain.Message.ToolCall
  alias LangChain.MessageDelta
  alias LangChain.TokenUsage

  @model "accounts/fireworks/models/glm-5p3-flash"

  defp fireworks(attrs \\ %{}) do
    ChatFireworks.new!(Map.merge(%{model: @model, api_key: "test-fireworks-key"}, attrs))
  end

  defp tool_call_turn(content) do
    Message.new!(%{
      role: :assistant,
      content: content,
      tool_calls: [
        ToolCall.new!(%{
          status: :complete,
          type: :function,
          call_id: "call_1",
          name: "get_weather",
          arguments: %{"city" => "Paris"}
        })
      ]
    })
  end

  defp chunk(delta, finish_reason \\ nil) do
    %{"choices" => [%{"index" => 0, "delta" => delta, "finish_reason" => finish_reason}]}
  end

  defp assemble(chunks, model \\ nil) do
    model = model || fireworks()

    chunks
    |> Enum.flat_map(&List.wrap(ChatFireworks.do_process_response(model, &1)))
    |> MessageDelta.merge_deltas()
    |> MessageDelta.to_message()
  end

  describe "new/1" do
    test "applies Fireworks defaults" do
      assert {:ok, %ChatFireworks{} = model} = ChatFireworks.new(%{model: @model})
      assert model.endpoint == "https://api.fireworks.ai/inference/v1/chat/completions"
      assert model.max_tokens == 32_768
      assert model.receive_timeout == 300_000
      assert model.send_reasoning_content == false
      assert model.temperature == nil
    end

    test "requires a model" do
      assert {:error, changeset} = ChatFireworks.new(%{})
      assert {"can't be blank", _} = changeset.errors[:model]
    end

    test "accepts only the documented reasoning_history values" do
      for value <- ~w(disabled interleaved preserved) do
        assert {:ok, _model} = ChatFireworks.new(%{model: @model, reasoning_history: value})
      end

      assert {:error, changeset} = ChatFireworks.new(%{model: @model, reasoning_history: "all"})

      assert {"must be one of: disabled, interleaved, preserved", _} =
               changeset.errors[:reasoning_history]
    end
  end

  describe "for_api/3" do
    test "sends only the fields that are set" do
      body = ChatFireworks.for_api(fireworks(), [Message.new_user!("Hi")], [])

      assert %{model: @model, stream: false, max_tokens: 32_768, messages: [_user]} = body

      for key <- [
            :temperature,
            :top_p,
            :reasoning_effort,
            :reasoning_history,
            :prompt_cache_key,
            :user,
            :tools,
            :tool_choice,
            :parallel_tool_calls,
            :response_format,
            :n,
            :stream_options,
            :max_completion_tokens
          ] do
        refute Map.has_key?(body, key), "did not expect #{inspect(key)} in the request body"
      end
    end

    test "includes reasoning, caching and sampling fields when set" do
      model =
        fireworks(%{
          reasoning_effort: "high",
          reasoning_history: "interleaved",
          prompt_cache_key: "conversation-1",
          temperature: 1.0,
          top_p: 0.95
        })

      assert %{
               reasoning_effort: "high",
               reasoning_history: "interleaved",
               prompt_cache_key: "conversation-1",
               temperature: 1.0,
               top_p: 0.95
             } = ChatFireworks.for_api(model, [Message.new_user!("Hi")], [])
    end

    test "merges extra_body last, and a nil value removes a key" do
      model = fireworks(%{extra_body: %{"top_k" => 20, "max_tokens" => nil}})
      body = ChatFireworks.for_api(model, [Message.new_user!("Hi")], [])

      assert %{"top_k" => 20} = body
      refute Map.has_key?(body, :max_tokens)
    end

    test "keeps the system role when reasoning_effort is set" do
      body =
        ChatFireworks.for_api(
          fireworks(%{reasoning_effort: "high"}),
          [Message.new_system!("Be brief."), Message.new_user!("Hi")],
          []
        )

      assert [%{"role" => :system}, %{"role" => :user}] = body.messages
    end

    test "does not send a message name" do
      %Message{} = user_message = Message.new_user!("Hi")
      message = %{user_message | name: "bob"}

      assert %{messages: [user]} = ChatFireworks.for_api(fireworks(), [message], [])
      refute Map.has_key?(user, "name")
    end

    test "sends null content for a tool-call turn that held only thinking" do
      message = tool_call_turn([ContentPart.thinking!("I should look up the weather.")])

      assert %{messages: [assistant]} = ChatFireworks.for_api(fireworks(), [message], [])
      assert %{"role" => :assistant, "content" => nil, "tool_calls" => [_call]} = assistant
      refute Map.has_key?(assistant, "reasoning_content")
    end

    test "sends an empty string for an assistant turn that held only thinking" do
      message = Message.new!(%{role: :assistant, content: [ContentPart.thinking!("Hmm.")]})

      assert %{messages: [%{"role" => :assistant, "content" => ""}]} =
               ChatFireworks.for_api(fireworks(), [message], [])
    end

    test "sends earlier thinking back when send_reasoning_content is on" do
      message = tool_call_turn([ContentPart.thinking!("I should look up the weather.")])

      assert %{
               messages: [
                 %{"reasoning_content" => "I should look up the weather.", "content" => nil}
               ]
             } = ChatFireworks.for_api(fireworks(%{send_reasoning_content: true}), [message], [])
    end

    test "never sends redacted thinking" do
      message =
        Message.new!(%{
          role: :assistant,
          content: [
            ContentPart.new!(%{
              type: :unsupported,
              content: "<encrypted>",
              options: [type: "redacted_thinking"]
            }),
            ContentPart.text!("Done.")
          ]
        })

      assert %{messages: [%{"content" => [%{"type" => "text", "text" => "Done."}]} = assistant]} =
               ChatFireworks.for_api(fireworks(%{send_reasoning_content: true}), [message], [])

      refute Map.has_key?(assistant, "reasoning_content")
    end

    test "serializes tools and tool_choice" do
      weather =
        Function.new!(%{
          name: "get_weather",
          parameters: [FunctionParam.new!(%{name: "city", type: "string", required: true})],
          function: fn _args, _context -> {:ok, "18C"} end
        })

      auto =
        ChatFireworks.for_api(
          fireworks(%{tool_choice: %{"type" => "auto"}}),
          [Message.new_user!("Weather?")],
          [weather]
        )

      assert %{
               tools: [%{"type" => "function", "function" => %{"name" => "get_weather"}}],
               tool_choice: "auto"
             } = auto

      named_choice = %{"type" => "function", "function" => %{"name" => "get_weather"}}

      assert %{tool_choice: ^named_choice} =
               ChatFireworks.for_api(
                 fireworks(%{tool_choice: named_choice}),
                 [Message.new_user!("Weather?")],
                 [weather]
               )
    end

    test "sets a json_schema response format" do
      schema = %{"name" => "answer", "schema" => %{"type" => "object"}}

      assert %{response_format: %{"type" => "json_schema", "json_schema" => ^schema}} =
               ChatFireworks.for_api(
                 fireworks(%{json_response: true, json_schema: schema}),
                 [Message.new_user!("Hi")],
                 []
               )
    end
  end

  describe "do_process_response/2 with a complete response" do
    test "puts thinking ahead of the answer and attaches usage" do
      body = %{
        "choices" => [
          %{
            "index" => 0,
            "finish_reason" => "stop",
            "message" => %{
              "role" => "assistant",
              "content" => "9.9 is larger.",
              "reasoning_content" => "Compare the tenths."
            }
          }
        ],
        "usage" => %{"prompt_tokens" => 10, "completion_tokens" => 20}
      }

      assert [%Message{role: :assistant, status: :complete} = message] =
               ChatFireworks.do_process_response(fireworks(), body)

      assert [
               %ContentPart{type: :thinking, content: "Compare the tenths."},
               %ContentPart{type: :text, content: "9.9 is larger."}
             ] = message.content

      assert %TokenUsage{input: 10, output: 20} = message.metadata.usage
    end

    test "reads a reasoning field when reasoning_content is absent" do
      body = %{
        "choices" => [
          %{
            "index" => 0,
            "finish_reason" => "stop",
            "message" => %{"role" => "assistant", "content" => "Yes.", "reasoning" => "Think."}
          }
        ]
      }

      assert [%Message{content: [%ContentPart{type: :thinking, content: "Think."}, _text]}] =
               ChatFireworks.do_process_response(fireworks(), body)
    end

    test "keeps thinking alongside a tool call" do
      body = %{
        "choices" => [
          %{
            "index" => 0,
            "finish_reason" => "tool_calls",
            "message" => %{
              "role" => "assistant",
              "content" => nil,
              "reasoning_content" => "I need the weather tool.",
              "tool_calls" => [
                %{
                  "id" => "call_1",
                  "type" => "function",
                  "function" => %{"name" => "get_weather", "arguments" => ~s({"city":"Paris"})}
                }
              ]
            }
          }
        ]
      }

      assert [%Message{} = message] = ChatFireworks.do_process_response(fireworks(), body)

      assert [%ContentPart{type: :thinking, content: "I need the weather tool."}] =
               message.content

      assert [%ToolCall{call_id: "call_1", name: "get_weather", arguments: %{"city" => "Paris"}}] =
               message.tool_calls
    end

    test "marks a response cut off by the token limit" do
      body = %{
        "choices" => [
          %{
            "index" => 0,
            "finish_reason" => "length",
            "message" => %{"role" => "assistant", "content" => "Partial"}
          }
        ]
      }

      assert [%Message{status: :length}] = ChatFireworks.do_process_response(fireworks(), body)
    end
  end

  describe "do_process_response/2 with streamed chunks" do
    test "thinking goes to position 0 and answer text to position 1" do
      assert [%MessageDelta{index: 0, content: %ContentPart{type: :thinking, content: "Hmm"}}] =
               ChatFireworks.do_process_response(
                 fireworks(),
                 chunk(%{"role" => "assistant", "reasoning_content" => "Hmm"})
               )

      assert [%MessageDelta{index: 1, content: "Hello"}] =
               ChatFireworks.do_process_response(fireworks(), chunk(%{"content" => "Hello"}))
    end

    test "reads a streamed reasoning field" do
      assert [%MessageDelta{index: 0, content: %ContentPart{type: :thinking, content: "Hmm"}}] =
               ChatFireworks.do_process_response(fireworks(), chunk(%{"reasoning" => "Hmm"}))
    end

    test "answer chunks that omit the reasoning key assemble after the thinking" do
      assert {:ok, %Message{} = message} =
               assemble([
                 chunk(%{"role" => "assistant", "reasoning_content" => "Compare "}),
                 chunk(%{"reasoning_content" => "the tenths."}),
                 chunk(%{"content" => "9.9 "}),
                 chunk(%{"content" => "is larger."}, "stop")
               ])

      assert [
               %ContentPart{type: :thinking, content: "Compare the tenths."},
               %ContentPart{type: :text, content: "9.9 is larger."}
             ] = message.content
    end

    test "answer chunks with a null reasoning key assemble the same way" do
      assert {:ok, %Message{} = message} =
               assemble([
                 chunk(%{
                   "role" => "assistant",
                   "reasoning_content" => "Think.",
                   "content" => nil
                 }),
                 chunk(%{"reasoning_content" => nil, "content" => "Done."}, "stop")
               ])

      assert [
               %ContentPart{type: :thinking, content: "Think."},
               %ContentPart{type: :text, content: "Done."}
             ] = message.content
    end

    test "a stream without thinking assembles into a single text part" do
      assert {:ok, %Message{} = message} =
               assemble([
                 chunk(%{"role" => "assistant", "content" => "Hello "}),
                 chunk(%{"content" => "world"}, "stop")
               ])

      assert [%ContentPart{type: :text, content: "Hello world"}] = message.content
    end

    test "tool call fragments assemble into a complete tool call" do
      assert {:ok, %Message{} = message} =
               assemble([
                 chunk(%{"role" => "assistant", "reasoning_content" => "Use the tool."}),
                 chunk(%{
                   "tool_calls" => [
                     %{
                       "index" => 0,
                       "id" => "call_1",
                       "type" => "function",
                       "function" => %{"name" => "get_weather", "arguments" => ""}
                     }
                   ]
                 }),
                 chunk(%{
                   "tool_calls" => [%{"index" => 0, "function" => %{"arguments" => ~s({"city":)}}]
                 }),
                 chunk(
                   %{
                     "tool_calls" => [
                       %{"index" => 0, "function" => %{"arguments" => ~s("Paris"})}}
                     ]
                   },
                   "tool_calls"
                 )
               ])

      assert [%ContentPart{type: :thinking, content: "Use the tool."}] = message.content

      assert [%ToolCall{call_id: "call_1", name: "get_weather", arguments: %{"city" => "Paris"}}] =
               message.tool_calls
    end

    test "a usage-only chunk returns token usage" do
      assert %TokenUsage{input: 5, output: 7} =
               ChatFireworks.do_process_response(fireworks(), %{
                 "choices" => [],
                 "usage" => %{"prompt_tokens" => 5, "completion_tokens" => 7}
               })
    end

    test "usage on the final chunk is attached to one delta" do
      data =
        %{"content" => "Done."}
        |> chunk("stop")
        |> Map.put("usage", %{"prompt_tokens" => 5, "completion_tokens" => 7})

      assert [%MessageDelta{metadata: %{usage: %TokenUsage{input: 5, output: 7}}}] =
               ChatFireworks.do_process_response(fireworks(), data)
    end

    test "a role-only opening chunk becomes an empty assistant delta" do
      assert [%MessageDelta{role: :assistant, index: 0, content: nil}] =
               ChatFireworks.do_process_response(
                 fireworks(),
                 chunk(%{"role" => "assistant", "content" => ""})
               )
    end
  end

  describe "error_from_response/2" do
    test "types errors by HTTP status" do
      for {status, type} <- [
            {400, "invalid_request"},
            {401, "authentication_error"},
            {403, "authentication_error"},
            {404, "not_found"},
            {408, "timeout"},
            {413, "request_too_large"},
            {422, "invalid_request"},
            {429, "rate_limit_exceeded"},
            {500, "server_error"},
            {502, "server_error"},
            {503, "overloaded"},
            {504, "timeout"}
          ] do
        assert %LangChainError{type: ^type} =
                 ChatFireworks.error_from_response(status, %{"error" => %{"message" => "boom"}})
      end
    end

    test "reads the message from OpenAI-style and validation error bodies" do
      assert %LangChainError{message: "Fireworks returned HTTP 429: slow down"} =
               ChatFireworks.error_from_response(429, %{"error" => %{"message" => "slow down"}})

      validation = %{
        "detail" => [
          %{
            "loc" => ["body", "temperature"],
            "msg" => "Input should be less than or equal to 2",
            "type" => "less_than_equal"
          }
        ]
      }

      assert %LangChainError{
               type: "invalid_request",
               message:
                 "Fireworks returned HTTP 422: body.temperature: Input should be less than or equal to 2"
             } = ChatFireworks.error_from_response(422, validation)
    end
  end

  describe "retry_on_fallback?/1" do
    test "falls back on rate limits, overload, server errors, timeouts and dropped connections" do
      for type <- ~w(rate_limit_exceeded overloaded server_error timeout connection) do
        assert ChatFireworks.retry_on_fallback?(
                 LangChainError.exception(type: type, message: "x")
               )
      end
    end

    test "does not fall back on request problems" do
      for type <- ~w(invalid_request authentication_error not_found request_too_large api_error) do
        refute ChatFireworks.retry_on_fallback?(
                 LangChainError.exception(type: type, message: "x")
               )
      end
    end
  end

  describe "call/3" do
    test "returns the parsed message" do
      expect(Req, :post, fn _request ->
        {:ok,
         %Req.Response{
           status: 200,
           body: %{
             "choices" => [
               %{
                 "index" => 0,
                 "finish_reason" => "stop",
                 "message" => %{"role" => "assistant", "content" => "Hi!"}
               }
             ]
           }
         }}
      end)

      assert {:ok, [%Message{role: :assistant}]} =
               ChatFireworks.call(fireworks(), [Message.new_user!("Hi")], [])
    end

    test "a 429 becomes a rate limit error that allows a fallback" do
      expect(Req, :post, fn _request ->
        {:ok, %Req.Response{status: 429, body: %{"error" => %{"message" => "Too many requests"}}}}
      end)

      assert {:error, %LangChainError{type: "rate_limit_exceeded"} = error} =
               ChatFireworks.call(fireworks(), [Message.new_user!("Hi")], [])

      assert ChatFireworks.retry_on_fallback?(error)
    end

    test "a streamed 503 becomes an overloaded error" do
      expect(Req, :post, fn _request, opts ->
        collector = Keyword.fetch!(opts, :into)
        start = {Req.Request.new(), %Req.Response{status: 503, headers: %{}, body: ""}}

        {:halt, {_req, response}} =
          collector.({:data, ~s({"error":{"message":"Service overloaded"}})}, start)

        {:ok, response}
      end)

      assert {:error, %LangChainError{type: "overloaded"}} =
               ChatFireworks.call(fireworks(%{stream: true}), [Message.new_user!("Hi")], [])
    end

    test "retries a closed connection, then reports it" do
      expect(Req, :post, 3, fn _request -> {:error, %Req.TransportError{reason: :closed}} end)

      assert {:error, %LangChainError{type: "connection"}} =
               ChatFireworks.call(fireworks(%{retry_count: 2}), [Message.new_user!("Hi")], [])
    end
  end

  describe "serialize_config/1 and restore_from_map/1" do
    test "round-trips the configuration without the API key" do
      model =
        fireworks(%{
          reasoning_effort: "high",
          reasoning_history: "interleaved",
          send_reasoning_content: true,
          extra_body: %{"top_k" => 20}
        })

      serialized = ChatFireworks.serialize_config(model)
      refute Map.has_key?(serialized, "api_key")

      assert {:ok,
              %ChatFireworks{
                model: @model,
                reasoning_effort: "high",
                reasoning_history: "interleaved",
                send_reasoning_content: true,
                extra_body: %{"top_k" => 20}
              }} = ChatFireworks.restore_from_map(serialized)
    end
  end

  # Live tests make billable calls to Fireworks. They use the effort level we
  # run in production, with token limits kept small.
  #
  #     mix test test/chat_models/chat_fireworks_test.exs --include live_fireworks
  #
  # Set FIREWORKS_API_KEY. FIREWORKS_ENDPOINT and FIREWORKS_MODEL override the
  # defaults, for example to run against the US-only router.
  defp live_model(attrs \\ %{}) do
    ChatFireworks.new!(
      Map.merge(
        %{
          endpoint:
            System.get_env(
              "FIREWORKS_ENDPOINT",
              "https://api.fireworks.ai/inference/v1/chat/completions"
            ),
          model: System.get_env("FIREWORKS_MODEL", @model),
          api_key: System.fetch_env!("FIREWORKS_API_KEY"),
          reasoning_effort: "high",
          max_tokens: 8192
        },
        attrs
      )
    )
  end

  defp employee_tools(test_pid) do
    [
      Function.new!(%{
        name: "lookup_employee_id",
        description: "Look up an employee's ID number by their first name",
        parameters: [
          FunctionParam.new!(%{
            name: "name",
            type: "string",
            description: "The employee's first name",
            required: true
          })
        ],
        function: fn args, _context ->
          send(test_pid, {:tool_called, "lookup_employee_id", args})
          {:ok, "EMP-4471"}
        end
      }),
      Function.new!(%{
        name: "get_vacation_days",
        description:
          "Get remaining vacation days for an employee ID. Requires the ID, not a name.",
        parameters: [
          FunctionParam.new!(%{
            name: "employee_id",
            type: "string",
            description: "The employee ID, in the form EMP-0000",
            required: true
          })
        ],
        function: fn args, _context ->
          send(test_pid, {:tool_called, "get_vacation_days", args})
          {:ok, "12 days remaining"}
        end
      })
    ]
  end

  defp run_vacation_chain(llm) do
    {:ok, chain} =
      %{llm: llm}
      |> LLMChain.new!()
      |> LLMChain.add_tools(employee_tools(self()))
      |> LLMChain.add_message(
        Message.new_system!("You are an HR assistant. Use the tools to answer.")
      )
      |> LLMChain.add_message(
        Message.new_user!(
          "How many vacation days does Marcy have left? " <>
            "Look up her employee ID first, then use that ID to check her vacation days."
        )
      )
      |> LLMChain.run(mode: :while_needs_response)

    chain
  end

  describe "live: Fireworks" do
    @tag live_call: true, live_fireworks: true, timeout: 300_000
    test "a complete response carries thinking and text" do
      assert {:ok, [%Message{} = message]} =
               ChatFireworks.call(live_model(), [
                 Message.new_user!("Which is larger, 9.11 or 9.9?")
               ])

      assert [%ContentPart{type: :thinking} | _rest] = message.content
      assert ContentPart.parts_to_string(message.content) =~ "9.9"
    end

    @tag live_call: true, live_fireworks: true, timeout: 300_000
    test "a streamed response merges into thinking and text" do
      {:ok, chain} =
        %{llm: live_model(%{stream: true})}
        |> LLMChain.new!()
        |> LLMChain.add_message(Message.new_user!("Which is larger, 9.11 or 9.9?"))
        |> LLMChain.run()

      assert [%ContentPart{type: :thinking}, %ContentPart{type: :text}] =
               chain.last_message.content
    end

    @tag live_call: true, live_fireworks: true, timeout: 300_000
    test "follows the system prompt while reasoning_effort is set" do
      {:ok, chain} =
        %{llm: live_model()}
        |> LLMChain.new!()
        |> LLMChain.add_message(
          Message.new_system!("End every answer with the single word PINEAPPLE.")
        )
        |> LLMChain.add_message(Message.new_user!("Say hello."))
        |> LLMChain.run()

      assert ContentPart.parts_to_string(chain.last_message.content) =~ "PINEAPPLE"
    end

    @tag live_call: true, live_fireworks: true, timeout: 300_000
    test "runs a streamed chain where each tool call depends on the last" do
      chain = run_vacation_chain(live_model(%{stream: true}))

      assert_received {:tool_called, "lookup_employee_id", %{"name" => name}}
      assert String.downcase(name) =~ "marcy"
      assert_received {:tool_called, "get_vacation_days", %{"employee_id" => "EMP-4471"}}
      assert ContentPart.parts_to_string(chain.last_message.content) =~ "12"
    end

    @tag live_call: true, live_fireworks: true, timeout: 300_000
    test "accepts earlier thinking sent back as reasoning_content" do
      llm = live_model(%{send_reasoning_content: true, reasoning_history: "interleaved"})
      chain = run_vacation_chain(llm)

      assert_received {:tool_called, "get_vacation_days", %{"employee_id" => "EMP-4471"}}
      assert ContentPart.parts_to_string(chain.last_message.content) =~ "12"
    end

    @tag live_call: true, live_fireworks: true, timeout: 300_000
    test "an unknown model is a not_found error" do
      assert {:error, %LangChainError{type: "not_found"}} =
               ChatFireworks.call(
                 live_model(%{model: "accounts/fireworks/models/no-such-model"}),
                 [Message.new_user!("Hi")]
               )
    end
  end
end
