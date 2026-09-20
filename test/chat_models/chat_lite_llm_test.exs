defmodule LangChain.ChatModels.ChatLiteLLMTest do
  use LangChain.BaseCase

  doctest LangChain.ChatModels.ChatLiteLLM
  alias LangChain.ChatModels.ChatLiteLLM
  alias LangChain.Function
  alias LangChain.FunctionParam
  alias LangChain.TokenUsage
  alias LangChain.LangChainError
  alias LangChain.Message
  alias LangChain.Message.ContentPart
  alias LangChain.Message.ToolCall
  alias LangChain.Message.ToolResult
  alias LangChain.MessageDelta

  @test_model "gpt-4o-mini"
  @prefixed_model "anthropic/claude-sonnet-4-5"
  @default_endpoint "http://localhost:4000/v1/chat/completions"

  setup do
    {:ok, hello_world} =
      Function.new(%{
        name: "hello_world",
        description: "Give a hello world greeting",
        function: fn _args, _context -> {:ok, "Hello world!"} end
      })

    {:ok, weather} =
      Function.new(%{
        name: "get_weather",
        description: "Get the current weather in a given US location",
        parameters: [
          FunctionParam.new!(%{
            name: "city",
            type: "string",
            description: "The city name, e.g. San Francisco",
            required: true
          })
        ],
        function: fn _args, _context -> {:ok, "75 degrees"} end
      })

    %{hello_world: hello_world, weather: weather}
  end

  describe "new/1" do
    test "works with minimal attributes" do
      assert {:ok, %ChatLiteLLM{} = model} = ChatLiteLLM.new(%{"model" => @test_model})
      assert model.model == @test_model
      assert model.endpoint == @default_endpoint
      assert model.stream == false
      assert model.n == 1
    end

    test "requires a model" do
      assert {:error, changeset} = ChatLiteLLM.new(%{})
      assert {"can't be blank", _} = changeset.errors[:model]
    end

    test "accepts a provider-prefixed model name unchanged" do
      assert {:ok, %ChatLiteLLM{model: @prefixed_model}} =
               ChatLiteLLM.new(%{"model" => @prefixed_model})
    end

    test "supports overriding the endpoint for a remote gateway" do
      assert {:ok, %ChatLiteLLM{endpoint: "https://litellm.example.com/v1/chat/completions"}} =
               ChatLiteLLM.new(%{
                 "model" => @test_model,
                 "endpoint" => "https://litellm.example.com/v1/chat/completions"
               })
    end

    test "rejects an invalid endpoint URL" do
      assert {:error, changeset} =
               ChatLiteLLM.new(%{"model" => @test_model, "endpoint" => "not-a-url"})

      assert {"must be a valid http or https URL", _} = changeset.errors[:endpoint]
    end

    test "validates temperature range" do
      assert {:error, changeset} =
               ChatLiteLLM.new(%{"model" => @test_model, "temperature" => 3.0})

      assert {"must be less than or equal to %{number}", _} = changeset.errors[:temperature]
    end

    test "the API key is optional" do
      # A gateway started without a master key serves unauthenticated requests,
      # so a missing key must not be a validation error.
      assert {:ok, %ChatLiteLLM{api_key: nil}} = ChatLiteLLM.new(%{"model" => @test_model})
    end
  end

  describe "new!/1" do
    test "returns the struct" do
      assert %ChatLiteLLM{model: @test_model} = ChatLiteLLM.new!(%{"model" => @test_model})
    end

    test "raises on invalid attributes" do
      assert_raise LangChainError, fn -> ChatLiteLLM.new!(%{}) end
    end
  end

  describe "for_api/3" do
    test "generates a map for a basic call" do
      {:ok, model} = ChatLiteLLM.new(%{"model" => @test_model, "temperature" => 0.5})

      data = ChatLiteLLM.for_api(model, [Message.new_user!("Hi")], [])

      assert %{model: @test_model, temperature: 0.5, stream: false, n: 1} = data
      assert [%{"role" => :user, "content" => "Hi"}] = data.messages
    end

    test "omits optional params that are not set" do
      {:ok, model} = ChatLiteLLM.new(%{"model" => @test_model})

      data = ChatLiteLLM.for_api(model, [Message.new_user!("Hi")], [])

      refute Map.has_key?(data, :temperature)
      refute Map.has_key?(data, :max_tokens)
      refute Map.has_key?(data, :seed)
      refute Map.has_key?(data, :response_format)
    end

    test "includes tools when provided", %{weather: weather} do
      {:ok, model} = ChatLiteLLM.new(%{"model" => @test_model})

      data = ChatLiteLLM.for_api(model, [Message.new_user!("What's the weather?")], [weather])

      assert [%{"type" => "function", "function" => %{"name" => "get_weather"}}] = data.tools
    end

    test "supports a json_object response format" do
      {:ok, model} = ChatLiteLLM.new(%{"model" => @test_model, "json_response" => true})

      data = ChatLiteLLM.for_api(model, [Message.new_user!("Hi")], [])

      assert %{"type" => "json_object"} = data.response_format
    end

    test "supports a json_schema response format" do
      schema = %{"name" => "answer", "schema" => %{"type" => "object"}}

      {:ok, model} =
        ChatLiteLLM.new(%{
          "model" => @test_model,
          "json_response" => true,
          "json_schema" => schema
        })

      data = ChatLiteLLM.for_api(model, [Message.new_user!("Hi")], [])

      assert %{"type" => "json_schema", "json_schema" => ^schema} = data.response_format
    end

    test "supports tool_choice by name" do
      {:ok, model} =
        ChatLiteLLM.new(%{
          "model" => @test_model,
          "tool_choice" => %{"type" => "function", "function" => %{"name" => "get_weather"}}
        })

      data = ChatLiteLLM.for_api(model, [Message.new_user!("Hi")], [])

      assert %{"type" => "function", "function" => %{"name" => "get_weather"}} = data.tool_choice
    end

    test "supports a simple tool_choice type" do
      {:ok, model} =
        ChatLiteLLM.new(%{"model" => @test_model, "tool_choice" => %{"type" => "auto"}})

      data = ChatLiteLLM.for_api(model, [Message.new_user!("Hi")], [])

      assert data.tool_choice == "auto"
    end

    test "merges extra_body for gateway-specific routing controls" do
      {:ok, model} =
        ChatLiteLLM.new(%{
          "model" => @test_model,
          "extra_body" => %{"fallbacks" => ["gpt-4o"], "tags" => ["team-a"]}
        })

      data = ChatLiteLLM.for_api(model, [Message.new_user!("Hi")], [])

      assert %{"fallbacks" => ["gpt-4o"], "tags" => ["team-a"]} = data
      # the standard keys survive the merge
      assert %{model: @test_model} = data
    end

    test "includes stream_options when set" do
      {:ok, model} =
        ChatLiteLLM.new(%{
          "model" => @test_model,
          "stream" => true,
          "stream_options" => %{"include_usage" => true}
        })

      data = ChatLiteLLM.for_api(model, [Message.new_user!("Hi")], [])

      assert %{"include_usage" => true} = data.stream_options
    end

    test "converts a tool result into a tool message" do
      {:ok, model} = ChatLiteLLM.new(%{"model" => @test_model})

      tool_result = ToolResult.new!(%{tool_call_id: "call_123", content: "75 degrees"})
      message = Message.new_tool_result!(%{tool_results: [tool_result]})

      data = ChatLiteLLM.for_api(model, [message], [])

      assert [%{"role" => :tool, "tool_call_id" => "call_123", "content" => "75 degrees"}] =
               data.messages
    end

    test "converts an assistant message with tool calls" do
      {:ok, model} = ChatLiteLLM.new(%{"model" => @test_model})

      tool_call =
        ToolCall.new!(%{
          type: :function,
          call_id: "call_abc",
          name: "get_weather",
          arguments: %{"city" => "Moab"}
        })

      message = Message.new_assistant!(%{tool_calls: [tool_call]})

      data = ChatLiteLLM.for_api(model, [message], [])

      assert [%{"role" => :assistant, "tool_calls" => [tool_call_data]}] = data.messages
      assert %{"id" => "call_abc", "type" => "function"} = tool_call_data
      assert %{"name" => "get_weather", "arguments" => args} = tool_call_data["function"]
      assert {:ok, %{"city" => "Moab"}} = Jason.decode(args)
    end
  end

  describe "models_url/1" do
    test "derives the models endpoint from a chat completions endpoint" do
      assert ChatLiteLLM.models_url("http://localhost:4000/v1/chat/completions") ==
               "http://localhost:4000/v1/models"
    end

    test "handles a gateway mounted under a path prefix" do
      assert ChatLiteLLM.models_url("https://example.com/litellm/v1/chat/completions") ==
               "https://example.com/litellm/v1/models"
    end

    test "appends when the endpoint is a bare base URL" do
      assert ChatLiteLLM.models_url("http://localhost:4000/v1") ==
               "http://localhost:4000/v1/models"

      assert ChatLiteLLM.models_url("http://localhost:4000/v1/") ==
               "http://localhost:4000/v1/models"
    end
  end

  describe "do_process_response/2 - complete messages" do
    test "handles a standard assistant response" do
      model = ChatLiteLLM.new!(%{"model" => @test_model})

      response = %{
        "id" => "chatcmpl-abc",
        "model" => "gpt-4o-mini",
        "object" => "chat.completion",
        "choices" => [
          %{
            "finish_reason" => "stop",
            "index" => 0,
            "message" => %{"role" => "assistant", "content" => "Hello!"}
          }
        ]
      }

      assert [%Message{} = message] = ChatLiteLLM.do_process_response(model, response)
      assert message.role == :assistant
      assert message.status == :complete
      assert [%ContentPart{type: :text, content: "Hello!"}] = message.content
    end

    test "records the responding model, since a gateway may route elsewhere" do
      model = ChatLiteLLM.new!(%{"model" => "cheap-alias"})

      response = %{
        "id" => "chatcmpl-xyz",
        # the gateway resolved the alias to a concrete upstream model
        "model" => "gpt-4o-mini",
        "object" => "chat.completion",
        "choices" => [
          %{
            "finish_reason" => "stop",
            "index" => 0,
            "message" => %{"role" => "assistant", "content" => "Hi"}
          }
        ]
      }

      assert [%Message{metadata: metadata}] = ChatLiteLLM.do_process_response(model, response)
      assert %{model: "gpt-4o-mini", id: "chatcmpl-xyz"} = metadata
    end

    test "handles a tool call response" do
      model = ChatLiteLLM.new!(%{"model" => @test_model})

      response = %{
        "choices" => [
          %{
            "finish_reason" => "tool_calls",
            "index" => 0,
            "message" => %{
              "role" => "assistant",
              "content" => nil,
              "tool_calls" => [
                %{
                  "id" => "call_123",
                  "type" => "function",
                  "function" => %{"name" => "get_weather", "arguments" => "{\"city\":\"Moab\"}"}
                }
              ]
            }
          }
        ]
      }

      assert [%Message{} = message] = ChatLiteLLM.do_process_response(model, response)
      assert [%ToolCall{} = call] = message.tool_calls
      assert call.name == "get_weather"
      assert call.call_id == "call_123"
      assert call.arguments == %{"city" => "Moab"}
    end

    test "promotes reasoning_content to a thinking ContentPart" do
      model = ChatLiteLLM.new!(%{"model" => @test_model})

      response = %{
        "choices" => [
          %{
            "finish_reason" => "stop",
            "index" => 0,
            "message" => %{
              "role" => "assistant",
              "content" => "4",
              "reasoning_content" => "2 plus 2 is 4."
            }
          }
        ]
      }

      assert [%Message{content: content}] = ChatLiteLLM.do_process_response(model, response)
      assert [%ContentPart{type: :thinking}, %ContentPart{type: :text, content: "4"}] = content
    end

    test "attaches token usage" do
      model = ChatLiteLLM.new!(%{"model" => @test_model})

      response = %{
        "choices" => [
          %{
            "finish_reason" => "stop",
            "index" => 0,
            "message" => %{"role" => "assistant", "content" => "Hi"}
          }
        ],
        "usage" => %{"prompt_tokens" => 10, "completion_tokens" => 2}
      }

      assert [%Message{metadata: %{usage: %TokenUsage{input: 10, output: 2}}}] =
               ChatLiteLLM.do_process_response(model, response)
    end
  end

  describe "do_process_response/2 - streaming" do
    test "handles a delta with a missing finish_reason" do
      # The gateway frequently omits finish_reason on intermediate chunks. This
      # is the exact shape that required fixes #367 and #551 for other models.
      model = ChatLiteLLM.new!(%{"model" => @test_model})

      chunk = %{
        "choices" => [
          %{"delta" => %{"role" => "assistant", "content" => "Hel"}, "index" => 0}
        ]
      }

      assert [%MessageDelta{} = delta] = ChatLiteLLM.do_process_response(model, chunk)
      assert delta.status == :incomplete
      assert delta.role == :assistant
    end

    test "handles a delta for an arbitrary model name" do
      # Regression guard: the delta clause must not match on a literal model
      # name, because the gateway routes arbitrary model strings.
      model = ChatLiteLLM.new!(%{"model" => @prefixed_model})

      chunk = %{
        "choices" => [%{"delta" => %{"role" => "assistant", "content" => "x"}, "index" => 0}]
      }

      assert [%MessageDelta{role: :assistant}] = ChatLiteLLM.do_process_response(model, chunk)
    end

    test "handles a terminal delta with finish_reason" do
      model = ChatLiteLLM.new!(%{"model" => @test_model})

      chunk = %{
        "choices" => [%{"delta" => %{"content" => "!"}, "index" => 0, "finish_reason" => "stop"}]
      }

      assert [%MessageDelta{status: :complete}] = ChatLiteLLM.do_process_response(model, chunk)
    end

    test "handles a usage-only terminal chunk with empty choices" do
      model = ChatLiteLLM.new!(%{"model" => @test_model})

      chunk = %{
        "choices" => [],
        "usage" => %{"prompt_tokens" => 7, "completion_tokens" => 3}
      }

      assert %TokenUsage{input: 7, output: 3} = ChatLiteLLM.do_process_response(model, chunk)
    end

    test "skips a chunk with empty choices and no usage" do
      model = ChatLiteLLM.new!(%{"model" => @test_model})

      assert :skip = ChatLiteLLM.do_process_response(model, %{"choices" => []})
    end

    test "captures usage from the gateway's terminal chunk shape" do
      # The gateway does NOT emit OpenAI's usage-only terminal chunk. Where
      # OpenAI sends `"choices": []` alongside `usage`, the gateway sends a
      # populated choices array holding an EMPTY delta, with usage at the top
      # level:
      #
      #   {"choices":[{"index":0,"delta":{}}],"usage":{...}}
      #
      # That lands in the normal choices branch, so the usage has to ride back
      # on the delta's metadata to reach `ChatModel.token_usage_from_result/1`.
      # If it did not, every streamed call through the gateway would silently
      # report no token usage.
      model = ChatLiteLLM.new!(%{"model" => @test_model})

      chunk = %{
        "id" => "chatcmpl-abc",
        "model" => "gpt-4.1-mini",
        "object" => "chat.completion.chunk",
        "choices" => [%{"index" => 0, "delta" => %{}}],
        "usage" => %{"prompt_tokens" => 19, "completion_tokens" => 9, "total_tokens" => 28}
      }

      assert [%MessageDelta{metadata: %{usage: %TokenUsage{input: 19, output: 9}}}] =
               result = ChatLiteLLM.do_process_response(model, chunk)

      # and it survives the extraction the telemetry layer actually performs
      assert %{token_usage: %TokenUsage{input: 19, output: 9}} =
               LangChain.ChatModels.ChatModel.token_usage_from_result({:ok, result})
    end

    test "handles a streamed reasoning delta" do
      model = ChatLiteLLM.new!(%{"model" => @test_model})

      chunk = %{
        "choices" => [
          %{"delta" => %{"role" => "assistant", "reasoning_content" => "thinking"}, "index" => 0}
        ]
      }

      assert [%MessageDelta{content: %ContentPart{type: :thinking}}] =
               ChatLiteLLM.do_process_response(model, chunk)
    end

    test "handles a streamed tool call delta" do
      model = ChatLiteLLM.new!(%{"model" => @test_model})

      chunk = %{
        "choices" => [
          %{
            "delta" => %{
              "role" => "assistant",
              "tool_calls" => [
                %{
                  "index" => 0,
                  "id" => "call_1",
                  "function" => %{"name" => "get_weather", "arguments" => ""}
                }
              ]
            },
            "index" => 0
          }
        ]
      }

      assert [%MessageDelta{tool_calls: [%ToolCall{name: "get_weather"}]}] =
               ChatLiteLLM.do_process_response(model, chunk)
    end
  end

  describe "do_process_response/2 - errors" do
    test "handles an error payload" do
      model = ChatLiteLLM.new!(%{"model" => @test_model})

      assert {:error, %LangChainError{message: "Budget exceeded"}} =
               ChatLiteLLM.do_process_response(model, %{
                 "error" => %{"message" => "Budget exceeded"}
               })
    end

    test "maps a 429 error code to rate_limit_exceeded" do
      model = ChatLiteLLM.new!(%{"model" => @test_model})

      assert {:error, %LangChainError{type: "rate_limit_exceeded"}} =
               ChatLiteLLM.do_process_response(model, %{
                 "error" => %{"code" => "429", "message" => "Too many requests"}
               })
    end

    test "handles invalid JSON" do
      model = ChatLiteLLM.new!(%{"model" => @test_model})

      assert {:error, %LangChainError{type: "invalid_json"}} =
               ChatLiteLLM.do_process_response(
                 model,
                 {:error, %Jason.DecodeError{data: "not json"}}
               )
    end

    test "handles an unexpected response shape" do
      model = ChatLiteLLM.new!(%{"model" => @test_model})

      assert {:error, %LangChainError{message: "Unexpected response"}} =
               ChatLiteLLM.do_process_response(model, "nonsense")
    end
  end

  describe "decode_stream/1" do
    test "decodes a single SSE chunk" do
      data =
        "data: {\"id\":\"1\",\"choices\":[{\"delta\":{\"content\":\"Hi\"},\"index\":0}]}\n\n"

      assert {[%{"id" => "1"}], ""} = ChatLiteLLM.decode_stream({data, ""})
    end

    test "decodes multiple chunks in one payload" do
      data =
        "data: {\"id\":\"1\"}\n\ndata: {\"id\":\"2\"}\n\n"

      assert {[%{"id" => "1"}, %{"id" => "2"}], ""} = ChatLiteLLM.decode_stream({data, ""})
    end

    test "buffers an incomplete chunk" do
      assert {[], "{\"id\":\"1\"" <> _} = ChatLiteLLM.decode_stream({"data: {\"id\":\"1\",", ""})
    end

    test "ignores the [DONE] sentinel" do
      assert {[], ""} = ChatLiteLLM.decode_stream({"data: [DONE]\n\n", ""})
    end
  end

  describe "serialize_config/1 and restore_from_map/1" do
    test "round trips the configuration" do
      {:ok, model} =
        ChatLiteLLM.new(%{
          "model" => @prefixed_model,
          "endpoint" => "https://litellm.example.com/v1/chat/completions",
          "temperature" => 0.2,
          "max_tokens" => 100
        })

      serialized = ChatLiteLLM.serialize_config(model)

      assert %{"model" => @prefixed_model, "temperature" => 0.2, "version" => 1} = serialized
      # the API key must never be serialized
      refute Map.has_key?(serialized, "api_key")

      assert {:ok, %ChatLiteLLM{} = restored} = ChatLiteLLM.restore_from_map(serialized)
      assert restored.model == model.model
      assert restored.endpoint == model.endpoint
      assert restored.temperature == model.temperature
      assert restored.max_tokens == model.max_tokens
    end
  end

  describe "provider/0" do
    test "returns the canonical provider name" do
      assert ChatLiteLLM.provider() == "litellm"
    end

    test "is resolved through the ChatModel helper" do
      model = ChatLiteLLM.new!(%{"model" => @test_model})
      assert LangChain.ChatModels.ChatModel.provider(model) == "litellm"
    end
  end

  describe "retry_on_fallback?/1" do
    test "retries transient failures" do
      for type <-
            ~w(rate_limited rate_limit_exceeded timeout too_many_requests server_error overloaded) do
        assert ChatLiteLLM.retry_on_fallback?(%LangChainError{type: type})
      end
    end

    test "does not retry an authentication failure" do
      refute ChatLiteLLM.retry_on_fallback?(%LangChainError{type: "authentication_error"})
    end
  end

  describe "live tests" do
    # These run against a real gateway. The model and key vary by deployment,
    # so both are read from the environment with local-gateway defaults:
    #
    #   LITELLM_TEST_MODEL=gpt-4.1-mini LITELLM_API_KEY=sk-... \
    #     mix test test/chat_models/chat_lite_llm_test.exs --include live_call
    defp live_model, do: System.get_env("LITELLM_TEST_MODEL", "gpt-4o-mini")

    defp live_chat(attrs) do
      ChatLiteLLM.new!(
        Map.merge(
          %{model: live_model(), api_key: System.get_env("LITELLM_API_KEY")},
          attrs
        )
      )
    end

    @tag live_call: true, live_litellm: true
    test "runs a basic call against a local gateway" do
      chat = live_chat(%{temperature: 0, stream: false})

      {:ok, result} = ChatLiteLLM.call(chat, "Return the number 4 and nothing else.", [])

      assert [%Message{role: :assistant} = message] = result
      assert ContentPart.content_to_string(message.content) =~ "4"
    end

    @tag live_call: true, live_litellm: true
    test "streams a response from a local gateway" do
      chat = live_chat(%{temperature: 0, stream: true})

      {:ok, result} = ChatLiteLLM.call(chat, "Say the word streaming and nothing else.", [])

      text =
        result
        |> List.flatten()
        |> Enum.filter(&match?(%MessageDelta{}, &1))
        |> Enum.map(fn
          %MessageDelta{content: %ContentPart{content: content}} when is_binary(content) ->
            content

          %MessageDelta{content: content} when is_binary(content) ->
            content

          _ ->
            ""
        end)
        |> Enum.join()

      assert text =~ "streaming"
    end

    @tag live_call: true, live_litellm: true
    test "discovers the models the gateway serves" do
      chat = live_chat(%{})

      assert {:ok, models} = ChatLiteLLM.list_models(chat)
      assert is_list(models)
      assert Enum.all?(models, &is_binary/1)
    end
  end
end
