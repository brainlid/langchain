defmodule LangChain.ChatModels.ChatPerplexityTest do
  alias LangChain.FunctionParam
  use LangChain.BaseCase

  doctest LangChain.ChatModels.ChatPerplexity
  alias LangChain.ChatModels.ChatPerplexity
  alias LangChain.Message
  alias LangChain.MessageDelta
  alias LangChain.Message.ContentPart
  alias LangChain.TokenUsage
  alias LangChain.LangChainError
  alias LangChain.Function

  @test_model "sonar-pro"

  describe "new/1" do
    test "works with minimal attrs" do
      assert {:ok, %ChatPerplexity{} = perplexity} = ChatPerplexity.new(%{"model" => @test_model})
      assert perplexity.model == @test_model
    end

    test "returns error when invalid" do
      assert {:error, changeset} = ChatPerplexity.new(%{"model" => nil})
      refute changeset.valid?
      assert {"can't be blank", _} = changeset.errors[:model]
    end

    test "supports overriding the API endpoint" do
      override_url = "http://localhost:1234/chat/completions"

      model =
        ChatPerplexity.new!(%{
          endpoint: override_url,
          model: @test_model
        })

      assert model.endpoint == override_url
    end

    test "validates temperature range" do
      {:error, changeset} = ChatPerplexity.new(%{model: @test_model, temperature: 2.1})
      assert {"must be less than %{number}", _} = changeset.errors[:temperature]

      {:error, changeset} = ChatPerplexity.new(%{model: @test_model, temperature: -0.1})
      assert {"must be greater than or equal to %{number}", _} = changeset.errors[:temperature]

      {:ok, model} = ChatPerplexity.new(%{model: @test_model, temperature: 1.5})
      assert model.temperature == 1.5
    end

    test "validates top_p range" do
      {:error, changeset} = ChatPerplexity.new(%{model: @test_model, top_p: 1.1})
      assert {"must be less than or equal to %{number}", _} = changeset.errors[:top_p]

      {:error, changeset} = ChatPerplexity.new(%{model: @test_model, top_p: 0})
      assert {"must be greater than %{number}", _} = changeset.errors[:top_p]

      {:ok, model} = ChatPerplexity.new(%{model: @test_model, top_p: 0.8})
      assert model.top_p == 0.8
    end

    test "validates top_k range" do
      {:error, changeset} = ChatPerplexity.new(%{model: @test_model, top_k: 2049})
      assert {"must be less than or equal to %{number}", _} = changeset.errors[:top_k]

      {:error, changeset} = ChatPerplexity.new(%{model: @test_model, top_k: -1})
      assert {"must be greater than or equal to %{number}", _} = changeset.errors[:top_k]

      {:ok, model} = ChatPerplexity.new(%{model: @test_model, top_k: 100})
      assert model.top_k == 100
    end

    test "validates presence_penalty range" do
      {:error, changeset} = ChatPerplexity.new(%{model: @test_model, presence_penalty: 2.1})
      assert {"must be less than or equal to %{number}", _} = changeset.errors[:presence_penalty]

      {:error, changeset} = ChatPerplexity.new(%{model: @test_model, presence_penalty: -2.1})

      assert {"must be greater than or equal to %{number}", _} =
               changeset.errors[:presence_penalty]

      {:ok, model} = ChatPerplexity.new(%{model: @test_model, presence_penalty: 1.5})
      assert model.presence_penalty == 1.5
    end

    test "validates frequency_penalty range" do
      {:error, changeset} = ChatPerplexity.new(%{model: @test_model, frequency_penalty: 0})
      assert {"must be greater than %{number}", _} = changeset.errors[:frequency_penalty]

      {:ok, model} = ChatPerplexity.new(%{model: @test_model, frequency_penalty: 0.5})
      assert model.frequency_penalty == 0.5
    end
  end

  describe "for_api/3" do
    test "generates a map for an API call" do
      {:ok, perplexity} =
        ChatPerplexity.new(%{
          "model" => @test_model,
          "temperature" => 1,
          "frequency_penalty" => 1.5,
          "api_key" => "api_key"
        })

      data = ChatPerplexity.for_api(perplexity, [], [])
      assert data.model == @test_model
      assert data.temperature == 1
      assert data.frequency_penalty == 1.5
      assert data.top_p == 0.9
      assert data.top_k == 0
    end

    test "includes tool calls as JSON schema when tools are provided" do
      calculator =
        Function.new!(%{
          name: "calculator",
          description: "A basic calculator",
          parameters: [
            FunctionParam.new!(%{name: "operation", type: :string, enum: ["+", "-", "*", "/"]}),
            FunctionParam.new!(%{name: "x", type: :number}),
            FunctionParam.new!(%{name: "y", type: :number})
          ],
          function: fn %{"operation" => operation, "x" => x, "y" => y}, _ ->
            {:ok, "Operation: #{operation}, x: #{x}, y: #{y}"}
          end
        })

      {:ok, perplexity} = ChatPerplexity.new(%{model: @test_model})
      data = ChatPerplexity.for_api(perplexity, [], [calculator])

      assert data.response_format["type"] == "json_schema"

      # The schema is nested under json_schema.schema
      schema = data.response_format["json_schema"]["schema"]
      assert schema["type"] == "object"
      assert schema["required"] == ["tool_calls"]
      assert schema["properties"]["tool_calls"]["type"] == "array"

      tool_item = schema["properties"]["tool_calls"]["items"]
      assert tool_item["required"] == ["name", "arguments"]
      assert tool_item["properties"]["name"]["enum"] == ["calculator"]
      assert tool_item["properties"]["arguments"]["type"] == "object"
    end

    test "includes optional parameters when set" do
      {:ok, perplexity} =
        ChatPerplexity.new(%{
          model: @test_model,
          max_tokens: 100,
          search_domain_filter: ["domain1.com", "domain2.com"],
          return_images: true,
          return_related_questions: true,
          search_recency_filter: "1d"
        })

      data = ChatPerplexity.for_api(perplexity, [], [])
      assert data.max_tokens == 100
      assert data.search_domain_filter == ["domain1.com", "domain2.com"]
      assert data.return_images == true
      assert data.return_related_questions == true
      assert data.search_recency_filter == "1d"
    end
  end

  describe "for_api/2" do
    test "turns a basic user message into the expected JSON format" do
      perplexity = ChatPerplexity.new!(%{model: @test_model})

      expected = %{"role" => :user, "content" => "Hi."}
      result = ChatPerplexity.for_api(perplexity, Message.new_user!("Hi."))
      assert result == expected
    end

    test "turns an assistant message into expected JSON format" do
      perplexity = ChatPerplexity.new!(%{model: @test_model})

      expected = %{"role" => :assistant, "content" => "Hi."}
      result = ChatPerplexity.for_api(perplexity, Message.new_assistant!("Hi."))
      assert result == expected
    end
  end

  describe "call/2" do
    # Skip live API calls in CI
    @tag live_call: true, live_perplexity_ai: true
    test "call/2 basic content example and fires token usage callback" do
      test_pid = self()

      handlers = %{
        on_llm_token_usage: fn usage ->
          send(test_pid, {:fired_token_usage, usage})
        end
      }

      {:ok, chat} =
        ChatPerplexity.new(%{
          model: @test_model,
          temperature: 1,
          stream: false,
          callbacks: [handlers]
        })

      {:ok, [%Message{role: :assistant, content: response}]} =
        ChatPerplexity.call(chat, [
          Message.new_user!("Return the response 'Hello World'.")
        ])

      assert response =~ "Hello World"

      # Token usage might not always be available
      receive do
        {:fired_token_usage, usage} ->
          assert %TokenUsage{} = usage
          assert usage.input > 0
          assert usage.output > 0
      after
        1000 -> :ok
      end
    end

    # Skip live API calls in CI
    @tag live_call: true, live_perplexity_ai: true
    test "call/2 basic streamed content example" do
      handlers = %{
        on_llm_new_delta: fn %MessageDelta{} = delta ->
          send(self(), {:message_delta, delta})
        end
      }

      chat =
        ChatPerplexity.new!(%{
          model: @test_model,
          temperature: 1,
          stream: true,
          callbacks: [handlers]
        })

      {:ok, _result} =
        ChatPerplexity.call(chat, [
          Message.new_user!("Return the response 'Hello World'.")
        ])

      # we expect to receive the response over multiple delta messages
      assert_receive {:message_delta, delta_1}, 2000
      assert_receive {:message_delta, delta_2}, 2000

      merged =
        delta_1
        |> MessageDelta.merge_delta(delta_2)

      assert merged.role == :assistant
      assert merged.content =~ "Hello World"
      assert merged.status == :complete
    end

    # Skip live API calls in CI
    @tag live_call: true, live_perplexity_ai: true
    test "call/2 handles complex tool calling scenarios" do
      store_article =
        Function.new!(%{
          name: "store_article",
          description: "Store an article with metadata",
          parameters: [
            FunctionParam.new!(%{
              name: "title",
              type: :string,
              description: "The article title",
              required: true
            }),
            FunctionParam.new!(%{
              name: "keywords",
              type: :array,
              item_type: "string",
              description: "SEO keywords",
              required: true
            }),
            FunctionParam.new!(%{
              name: "meta_description",
              type: :string,
              description: "SEO meta description",
              required: true
            })
          ],
          function: fn %{"title" => title, "keywords" => keywords, "meta_description" => meta},
                       _ ->
            {:ok, "Stored article: #{title} with #{length(keywords)} keywords and meta: #{meta}"}
          end
        })

      chat =
        ChatPerplexity.new!(%{
          model: @test_model,
          temperature: 0.7,
          stream: false
        })

      prompt = """
      Rules:
      1. Provide only the final answer. It is important that you do not include any explanation on the steps below.
      2. Do not show the intermediate steps information.

      Steps:
      Generate an SEO-optimized article title and metadata about artificial intelligence.
      Use the store_article function to save the generated content.
      Make sure to include relevant keywords and a compelling meta description.

      Output a JSON object with the following fields:
      - title: The article title
      - keywords: An array of SEO keywords
      - meta_description: The SEO meta description
      """

      {:ok, [%Message{} = response]} =
        ChatPerplexity.call(
          chat,
          [
            Message.new_system!("You are an SEO expert."),
            Message.new_user!(prompt)
          ],
          [store_article]
        )

      assert response.role == :assistant
      assert length(response.tool_calls) == 1

      [tool_call] = response.tool_calls
      assert tool_call.name == "store_article"
      assert tool_call.type == :function

      assert is_binary(tool_call.arguments["title"])
      assert is_list(tool_call.arguments["keywords"])
      assert is_binary(tool_call.arguments["meta_description"])
    end

    @tag live_call: true, live_perplexity_ai: true
    test "call/2 handles errors in response_format schema" do
      calculator = %Function{
        name: "calculator",
        description: "Basic calculator",
        parameters: [
          FunctionParam.new!(%{
            name: "operation",
            type: :string,
            enum: ["+", "-", "*", "/"]
          }),
          FunctionParam.new!(%{
            name: "x",
            type: :number
          }),
          FunctionParam.new!(%{
            name: "y",
            type: :number
          })
        ]
      }

      chat =
        ChatPerplexity.new!(%{
          model: @test_model,
          temperature: 0.7,
          stream: false
        })

      {:error, error} =
        ChatPerplexity.call(
          chat,
          [Message.new_user!("What is 5 + 3?")],
          [calculator]
        )

      assert %LangChainError{} = error
      assert error.type == "bad_request"
      assert error.message =~ "response_format"
    end
  end

  describe "do_process_response/3" do
    setup do
      model = ChatPerplexity.new!(%{model: @test_model})
      %{model: model}
    end

    test "captures full response data in processed_content", %{model: model} do
      response_data = %{
        "id" => "cmpl-123",
        "model" => "sonar-pro",
        "created" => 1_234_567_890,
        "usage" => %{
          "prompt_tokens" => 10,
          "completion_tokens" => 15,
          "total_tokens" => 25
        },
        "object" => "chat.completion",
        "citations" => ["https://example.com/article1", "https://example.com/article2"],
        "search_results" => [
          %{
            "title" => "Article 1",
            "url" => "https://example.com/article1",
            "date" => "2023-12-25"
          },
          %{
            "title" => "Article 2",
            "url" => "https://example.com/article2",
            "date" => "2023-12-26"
          }
        ],
        "choices" => [
          %{
            "finish_reason" => "stop",
            "message" => %{
              "content" => "This is a test response.",
              "role" => "assistant"
            },
            "index" => 0
          }
        ]
      }

      result = ChatPerplexity.do_process_response(model, response_data, [])

      assert %Message{} = result
      assert result.role == :assistant
      assert result.content == [ContentPart.text!("This is a test response.")]
      assert result.status == :complete

      # Verify the full response data is captured in processed_content
      assert result.processed_content.id == "cmpl-123"
      assert result.processed_content.model == "sonar-pro"
      assert result.processed_content.created == 1_234_567_890

      assert result.processed_content.usage == %{
               "prompt_tokens" => 10,
               "completion_tokens" => 15,
               "total_tokens" => 25
             }

      assert result.processed_content.citations == [
               "https://example.com/article1",
               "https://example.com/article2"
             ]

      assert length(result.processed_content.search_results) == 2
      assert List.first(result.processed_content.search_results)["title"] == "Article 1"
    end

    test "handles tool call responses", %{model: model} do
      tools = [
        %LangChain.Function{
          name: "calculator",
          description: "A basic calculator"
        }
      ]

      response_data = %{
        "id" => "cmpl-tool-123",
        "model" => "sonar-pro",
        "created" => 1_234_567_890,
        "usage" => %{"prompt_tokens" => 10, "completion_tokens" => 15, "total_tokens" => 25},
        "object" => "chat.completion",
        "citations" => [],
        "search_results" => [],
        "choices" => [
          %{
            "finish_reason" => "stop",
            "message" => %{
              "content" =>
                Jason.encode!(%{
                  "tool_calls" => [
                    %{
                      "name" => "calculator",
                      "arguments" => %{
                        "operation" => "+",
                        "x" => 5,
                        "y" => 3
                      }
                    }
                  ]
                })
            }
          }
        ]
      }

      result = ChatPerplexity.do_process_response(model, response_data, tools)

      assert %Message{} = result
      assert result.role == :assistant
      assert result.status == :complete
      assert length(result.tool_calls) == 1

      [tool_call] = result.tool_calls
      assert tool_call.name == "calculator"
      assert tool_call.type == :function
      assert tool_call.status == :complete

      args = Jason.decode!(tool_call.arguments)
      assert args["operation"] == "+"
      assert args["x"] == 5
      assert args["y"] == 3

      # Verify full response data is captured
      assert result.processed_content.id == "cmpl-tool-123"
      assert result.processed_content.model == "sonar-pro"
      assert result.processed_content.citations == []
      assert result.processed_content.search_results == []
    end

    test "handles regular message responses", %{model: model} do
      response_data = %{
        "id" => "cmpl-regular-123",
        "model" => "sonar-pro",
        "created" => 1_234_567_890,
        "usage" => %{"prompt_tokens" => 5, "completion_tokens" => 3, "total_tokens" => 8},
        "object" => "chat.completion",
        "citations" => [],
        "search_results" => [],
        "choices" => [
          %{
            "message" => %{"content" => "Hello!", "role" => "assistant"},
            "finish_reason" => "stop",
            "index" => 1
          }
        ]
      }

      assert %Message{} = message = ChatPerplexity.do_process_response(model, response_data, [])
      assert message.role == :assistant
      assert message.content == [ContentPart.text!("Hello!")]
      assert message.index == 1
      assert message.status == :complete

      # Verify full response data is captured
      assert message.processed_content.id == "cmpl-regular-123"
      assert message.processed_content.citations == []
    end

    test "returns skip when given an empty choices list", %{model: model} do
      assert :skip == ChatPerplexity.do_process_response(model, %{"choices" => []}, [])
    end

    test "handles error from server that the max length has been reached", %{model: model} do
      response_data = %{
        "id" => "cmpl-length-123",
        "model" => "sonar-pro",
        "created" => 1_234_567_890,
        "usage" => %{"prompt_tokens" => 100, "completion_tokens" => 50, "total_tokens" => 150},
        "object" => "chat.completion",
        "citations" => [],
        "search_results" => [],
        "choices" => [
          %{
            "finish_reason" => "length",
            "index" => 0,
            "message" => %{
              "content" => "Some of the response that was abruptly",
              "role" => "assistant"
            }
          }
        ]
      }

      assert %Message{} = struct = ChatPerplexity.do_process_response(model, response_data, [])
      assert struct.role == :assistant
      assert struct.content == [ContentPart.text!("Some of the response that was abruptly")]
      assert struct.index == 0
      assert struct.status == :length

      # Verify full response data is captured
      assert struct.processed_content.id == "cmpl-length-123"
    end

    test "handles json parse error from server", %{model: model} do
      {:error, %LangChainError{} = error} =
        ChatPerplexity.do_process_response(model, {:error, %Jason.DecodeError{}}, [])

      assert error.type == "invalid_json"
      assert "Received invalid JSON: " <> _ = error.message
    end

    test "handles unexpected response", %{model: model} do
      {:error, %LangChainError{} = error} =
        ChatPerplexity.do_process_response(model, "unexpected", [])

      assert error.type == nil
      assert error.message == "Unexpected response"
    end
  end

  describe "decode_stream/1" do
    test "correctly handles fully formed chat completion chunks" do
      data =
        "data: {\"id\":\"msg_123\",\"object\":\"chat.completion.chunk\",\"created\":1234567890,\"model\":\"pplx-7b-chat\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":\"Hello\"},\"finish_reason\":null}]}\n\n" <>
          "data: {\"id\":\"msg_123\",\"object\":\"chat.completion.chunk\",\"created\":1234567890,\"model\":\"pplx-7b-chat\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\" World\"},\"finish_reason\":null}]}\n\n"

      {parsed, incomplete} = ChatPerplexity.decode_stream({data, ""})

      assert incomplete == ""
      assert length(parsed) == 2

      [msg1, msg2] = parsed
      assert msg1["choices"] |> hd() |> get_in(["delta", "content"]) == "Hello"
      assert msg2["choices"] |> hd() |> get_in(["delta", "content"]) == " World"
    end

    test "correctly parses when data split over received messages" do
      data =
        "data: {\"id\":\"msg_123\",\"object\":\"chat.comple" <>
          "data: tion.chunk\",\"created\":1234567890,\"model\":\"pplx-7b-chat\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":\"Hello\"},\"finish_reason\":null}]}\n\n"

      {parsed, incomplete} = ChatPerplexity.decode_stream({data, ""})

      assert incomplete == ""
      assert length(parsed) == 1

      [msg] = parsed
      assert msg["choices"] |> hd() |> get_in(["delta", "content"]) == "Hello"
    end
  end

  describe "serialize_config/1" do
    test "creates expected map" do
      model =
        ChatPerplexity.new!(%{
          model: @test_model,
          temperature: 0.5,
          top_p: 0.8,
          top_k: 100,
          max_tokens: 1000,
          presence_penalty: 1.0,
          frequency_penalty: 1.5,
          search_domain_filter: ["domain.com"],
          return_images: true,
          return_related_questions: true,
          search_recency_filter: "1d"
        })

      result = ChatPerplexity.serialize_config(model)

      assert result == %{
               "endpoint" => "https://api.perplexity.ai/chat/completions",
               "model" => @test_model,
               "temperature" => 0.5,
               "top_p" => 0.8,
               "top_k" => 100,
               "max_tokens" => 1000,
               "stream" => false,
               "presence_penalty" => 1.0,
               "frequency_penalty" => 1.5,
               "search_domain_filter" => ["domain.com"],
               "return_images" => true,
               "return_related_questions" => true,
               "search_recency_filter" => "1d",
               "response_format" => nil,
               "receive_timeout" => 60000,
               "version" => 1
             }
    end

    test "does not include the API key or callbacks" do
      model = ChatPerplexity.new!(%{model: @test_model})
      result = ChatPerplexity.serialize_config(model)
      assert result["version"] == 1
      refute Map.has_key?(result, "api_key")
      refute Map.has_key?(result, "callbacks")
    end
  end
end
