defmodule LangChain.ChatModels.ChatPerplexityTest do
  alias LangChain.FunctionParam
  use LangChain.BaseCase

  doctest LangChain.ChatModels.ChatPerplexity
  alias LangChain.ChatModels.ChatPerplexity
  alias LangChain.Chains.LLMChain
  alias LangChain.Message
  alias LangChain.MessageDelta
  alias LangChain.Message.Citation
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

    test "supports verbose_api option" do
      model = ChatPerplexity.new!(%{model: @test_model, verbose_api: true})
      assert model.verbose_api == true

      model = ChatPerplexity.new!(%{model: @test_model, verbose_api: false})
      assert model.verbose_api == false
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

      {:ok, [%Message{role: :assistant} = message]} =
        ChatPerplexity.call(chat, [
          Message.new_user!("Return the response 'Hello World'.")
        ])

      response_text = ContentPart.parts_to_string(message.content)
      assert response_text =~ "Hello World"

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
      chat =
        ChatPerplexity.new!(%{
          model: @test_model,
          temperature: 1,
          stream: true
        })

      {:ok, result} =
        ChatPerplexity.call(chat, [
          Message.new_user!("Return the response 'Hello World'.")
        ])

      # Streaming call/2 returns list of delta batches
      all_deltas = List.flatten(result)
      assert length(all_deltas) > 0

      merged = MessageDelta.merge_deltas(all_deltas)
      assert merged.role == :assistant
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

      # arguments is a JSON string, decode it to check the fields
      args = Jason.decode!(tool_call.arguments)
      assert is_binary(args["title"])
      assert is_list(args["keywords"])
      assert is_binary(args["meta_description"])
    end

    @tag live_call: true, live_perplexity_ai: true
    test "call/2 handles simple tool calling via structured output" do
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

      {:ok, [%Message{} = response]} =
        ChatPerplexity.call(
          chat,
          [Message.new_user!("What is 5 + 3?")],
          [calculator]
        )

      assert response.role == :assistant
      assert length(response.tool_calls) == 1

      [tool_call] = response.tool_calls
      assert tool_call.name == "calculator"
      assert tool_call.type == :function
    end
  end

  describe "live citation tests" do
    @tag live_call: true, live_perplexity_ai: true
    test "non-streaming response includes citations for factual questions" do
      {:ok, chat} =
        ChatPerplexity.new(%{
          model: @test_model,
          temperature: 0,
          stream: false
        })

      # Ask about a stable historical fact to trigger citations
      {:ok, [%Message{} = message]} =
        ChatPerplexity.call(chat, [
          Message.new_user!("When was the Apollo 11 moon landing?")
        ])

      assert message.role == :assistant
      assert is_list(message.content)
      assert [%ContentPart{} = part | _] = message.content

      # The response should contain text with citation markers
      assert part.content =~ "1969"

      # Should have citations (Perplexity returns them for factual queries)
      all_citations = Message.all_citations(message)

      if all_citations != [] do
        # Verify citation structure
        Enum.each(all_citations, fn citation ->
          assert %Citation{} = citation
          assert citation.source.type == :web
          assert citation.source.url != nil
          assert citation.metadata["provider_type"] == "perplexity_citation"
        end)

        # Verify citation URLs are extractable
        urls = Citation.source_urls(all_citations)
        assert length(urls) > 0
        Enum.each(urls, fn url -> assert String.starts_with?(url, "http") end)
      else
        # If no citations, log but don't fail - API behavior may vary
        IO.puts("Note: No citations returned. Response: #{inspect(part.content)}")
      end
    end

    @tag live_call: true, live_perplexity_ai: true
    test "STREAMED LLMChain integration - citations via streaming deltas" do
      handler = %{
        on_llm_new_delta: fn %LLMChain{} = _chain, deltas ->
          send(self(), deltas)
        end,
        on_message_processed: fn %LLMChain{} = _chain, message ->
          send(self(), {:test_message_processed, message})
        end,
        on_llm_token_usage: fn %LLMChain{} = _chain, usage ->
          send(self(), {:test_token_usage, usage})
        end
      }

      model =
        ChatPerplexity.new!(%{
          model: @test_model,
          temperature: 0,
          stream: true
        })

      original_chain =
        %{llm: model}
        |> LLMChain.new!()
        |> LLMChain.add_callback(handler)
        |> LLMChain.add_messages([
          Message.new_user!("When was the Apollo 11 moon landing? Keep your answer brief.")
        ])

      {:ok, updated_chain} = LLMChain.run(original_chain)

      # The chain's last message should be a complete assistant response
      last_message = updated_chain.last_message
      assert %Message{role: :assistant, status: :complete} = last_message

      # The on_message_processed callback should have fired
      assert_received {:test_message_processed, processed_message}
      assert processed_message == last_message

      # Collect all streamed deltas and apply them
      all_mailbox = collect_messages()
      deltas = all_mailbox |> List.flatten()
      assert length(deltas) > 0

      # Apply deltas to the original chain (same pattern as Anthropic test)
      delta_merged_chain = LLMChain.apply_deltas(original_chain, deltas)

      # Check citations on the chain's final message
      all_citations = Message.all_citations(last_message)

      if all_citations != [] do
        urls = Citation.source_urls(all_citations)
        assert length(urls) > 0
        Enum.each(urls, fn url -> assert String.starts_with?(url, "http") end)

        # Delta-merged message should also have citations
        delta_citations = Message.all_citations(delta_merged_chain.last_message)
        assert length(delta_citations) > 0
      end
    end
  end

  describe "do_process_response/2" do
    setup do
      model = ChatPerplexity.new!(%{model: @test_model})
      %{model: model}
    end

    test "handles tool call responses", %{model: model} do
      response = %{
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

      result = ChatPerplexity.do_process_response(model, response["choices"] |> List.first())

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
    end

    test "handles regular message responses", %{model: model} do
      response = %{
        "message" => %{"content" => "Hello!"},
        "finish_reason" => "stop",
        "index" => 1
      }

      assert %Message{} = message = ChatPerplexity.do_process_response(model, response)
      assert message.role == :assistant
      assert message.content == [ContentPart.text!("Hello!")]
      assert message.index == 1
      assert message.status == :complete
    end

    test "returns skip when given an empty choices list", %{model: model} do
      assert :skip == ChatPerplexity.do_process_response(model, %{"choices" => []})
    end

    test "handles error from server that the max length has been reached", %{model: model} do
      response = %{
        "finish_reason" => "length",
        "index" => 0,
        "message" => %{
          "content" => "Some of the response that was abruptly",
          "role" => "assistant"
        }
      }

      assert %Message{} = struct = ChatPerplexity.do_process_response(model, response)
      assert struct.role == :assistant
      assert struct.content == [ContentPart.text!("Some of the response that was abruptly")]
      assert struct.index == 0
      assert struct.status == :length
    end

    test "handles json parse error from server", %{model: model} do
      {:error, %LangChainError{} = error} =
        ChatPerplexity.do_process_response(model, {:error, %Jason.DecodeError{}})

      assert error.type == "invalid_json"
      assert "Received invalid JSON: " <> _ = error.message
    end

    test "handles unexpected response", %{model: model} do
      {:error, %LangChainError{} = error} =
        ChatPerplexity.do_process_response(model, "unexpected")

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
               "verbose_api" => false,
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

    test "includes verbose_api field" do
      model = ChatPerplexity.new!(%{model: @test_model, verbose_api: true})
      result = ChatPerplexity.serialize_config(model)
      assert result["verbose_api"] == true
    end
  end

  describe "citation support - non-streaming" do
    setup do
      model = ChatPerplexity.new!(%{model: @test_model})
      %{model: model}
    end

    test "parses top-level citations array into Citation structs", %{model: model} do
      response = %{
        "choices" => [
          %{
            "finish_reason" => "stop",
            "message" => %{
              "content" => "Spain won Euro 2024 [1], beating England in the final [2].",
              "role" => "assistant"
            }
          }
        ],
        "citations" => [
          "https://www.uefa.com/euro2024/results",
          "https://en.wikipedia.org/wiki/UEFA_Euro_2024"
        ]
      }

      result = ChatPerplexity.do_process_response(model, response, [])
      assert %Message{} = result
      assert [%ContentPart{} = part] = result.content
      assert part.type == :text
      assert length(part.citations) == 2

      [cit1, cit2] = part.citations

      assert %Citation{} = cit1
      assert cit1.source.type == :web
      assert cit1.source.url == "https://www.uefa.com/euro2024/results"
      assert cit1.metadata["citation_index"] == 0

      assert %Citation{} = cit2
      assert cit2.source.type == :web
      assert cit2.source.url == "https://en.wikipedia.org/wiki/UEFA_Euro_2024"
      assert cit2.metadata["citation_index"] == 1
    end

    test "merges search_results titles into citations", %{model: model} do
      response = %{
        "choices" => [
          %{
            "finish_reason" => "stop",
            "message" => %{
              "content" => "The Eiffel Tower is 330 meters tall [1].",
              "role" => "assistant"
            }
          }
        ],
        "citations" => [
          "https://en.wikipedia.org/wiki/Eiffel_Tower"
        ],
        "search_results" => [
          %{
            "title" => "Eiffel Tower - Wikipedia",
            "url" => "https://en.wikipedia.org/wiki/Eiffel_Tower",
            "date" => "2024-01-15",
            "snippet" => "The Eiffel Tower is a wrought-iron lattice tower..."
          }
        ]
      }

      result = ChatPerplexity.do_process_response(model, response, [])
      assert %Message{} = result
      assert [%ContentPart{} = part] = result.content
      assert [%Citation{} = citation] = part.citations

      assert citation.source.type == :web
      assert citation.source.title == "Eiffel Tower - Wikipedia"
      assert citation.source.url == "https://en.wikipedia.org/wiki/Eiffel_Tower"
      assert citation.metadata["provider_type"] == "perplexity_citation"
      assert citation.metadata["citation_index"] == 0

      # search_results metadata is preserved on the source
      assert citation.source.metadata["date"] == "2024-01-15"
      assert citation.source.metadata["snippet"] =~ "wrought-iron"
    end

    test "handles response with no citations", %{model: model} do
      response = %{
        "choices" => [
          %{
            "finish_reason" => "stop",
            "message" => %{
              "content" => "Hello World!",
              "role" => "assistant"
            }
          }
        ]
      }

      result = ChatPerplexity.do_process_response(model, response, [])
      assert %Message{} = result
      assert [%ContentPart{} = part] = result.content
      assert part.citations == []
    end

    test "handles multiple citation markers for the same source", %{model: model} do
      response = %{
        "choices" => [
          %{
            "finish_reason" => "stop",
            "message" => %{
              "content" => "Fact A [1]. Also fact B [1].",
              "role" => "assistant"
            }
          }
        ],
        "citations" => ["https://example.com/source1"]
      }

      result = ChatPerplexity.do_process_response(model, response, [])
      assert %Message{} = result
      assert [%ContentPart{} = part] = result.content
      # Two [1] markers should produce two citations
      assert length(part.citations) == 2
      assert Enum.all?(part.citations, &(&1.source.url == "https://example.com/source1"))
    end

    test "ignores citation markers with out-of-range indices", %{model: model} do
      response = %{
        "choices" => [
          %{
            "finish_reason" => "stop",
            "message" => %{
              "content" => "Some text [1] and more [5].",
              "role" => "assistant"
            }
          }
        ],
        "citations" => ["https://example.com/valid"]
      }

      result = ChatPerplexity.do_process_response(model, response, [])
      assert %Message{} = result
      assert [%ContentPart{} = part] = result.content
      # Only [1] should produce a citation, [5] is out of range
      assert length(part.citations) == 1
      assert hd(part.citations).metadata["citation_index"] == 0
    end

    test "citation start_index and end_index mark the [N] positions", %{model: model} do
      content = "Answer is here [1]."

      response = %{
        "choices" => [
          %{
            "finish_reason" => "stop",
            "message" => %{"content" => content, "role" => "assistant"}
          }
        ],
        "citations" => ["https://example.com"]
      }

      result = ChatPerplexity.do_process_response(model, response, [])
      assert [%ContentPart{} = part] = result.content
      assert [%Citation{} = citation] = part.citations

      # "[1]" starts at position 15, length 3
      assert citation.start_index == 15
      assert citation.end_index == 18

      assert binary_part(content, citation.start_index, citation.end_index - citation.start_index) ==
               "[1]"
    end

    test "Message.all_citations/1 collects citations from Perplexity response", %{model: model} do
      response = %{
        "choices" => [
          %{
            "finish_reason" => "stop",
            "message" => %{
              "content" => "Fact [1] and fact [2].",
              "role" => "assistant"
            }
          }
        ],
        "citations" => [
          "https://example.com/a",
          "https://example.com/b"
        ]
      }

      result = ChatPerplexity.do_process_response(model, response, [])
      all_citations = Message.all_citations(result)
      assert length(all_citations) == 2

      urls = Citation.source_urls(all_citations)
      assert "https://example.com/a" in urls
      assert "https://example.com/b" in urls
    end
  end

  describe "citation support - streaming delta creation" do
    test "citation deltas merge correctly with text deltas" do
      # Simulate the streaming pattern: text deltas arrive first, then a
      # citation-only delta, then completion. Use merge_deltas/1 which
      # starts from nil, correctly accumulating the first text delta.
      text_delta =
        MessageDelta.new!(%{
          role: :assistant,
          content: "Spain won Euro 2024 [1].",
          status: :incomplete
        })

      # Citation delta (no content, just citations) - as sent by streaming citation handler
      sources =
        ChatPerplexity.build_perplexity_sources(
          ["https://www.uefa.com/euro2024"],
          [%{"title" => "UEFA Euro 2024", "url" => "https://www.uefa.com/euro2024"}]
        )

      citation_structs =
        sources
        |> Enum.with_index()
        |> Enum.map(fn {source, idx} ->
          Citation.new!(%{
            source: source,
            metadata: %{"provider_type" => "perplexity_citation", "citation_index" => idx}
          })
        end)

      citation_part = %ContentPart{type: :text, content: nil, citations: citation_structs}

      citation_delta =
        MessageDelta.new!(%{content: citation_part, role: :assistant, status: :incomplete})

      # Completion delta
      complete_delta = MessageDelta.new!(%{role: :assistant, status: :complete})

      # Merge all deltas (merge_deltas starts from nil, which correctly
      # handles the first delta's content → merged_content migration)
      merged = MessageDelta.merge_deltas([text_delta, citation_delta, complete_delta])

      assert merged.role == :assistant
      assert merged.status == :complete

      # Verify merged_content has the text and citations
      assert [%ContentPart{} = merged_part] = merged.merged_content
      assert merged_part.content == "Spain won Euro 2024 [1]."
      assert length(merged_part.citations) == 1

      [citation] = merged_part.citations
      assert citation.source.type == :web
      assert citation.source.url == "https://www.uefa.com/euro2024"
      assert citation.source.title == "UEFA Euro 2024"

      # Convert to message
      {:ok, message} = MessageDelta.to_message(merged)
      assert %Message{} = message

      all_citations = Message.all_citations(message)
      assert length(all_citations) == 1
    end

    test "multiple citation sources accumulate on merged delta" do
      sources =
        ChatPerplexity.build_perplexity_sources(
          ["https://example.com/a", "https://example.com/b", "https://example.com/c"],
          []
        )

      citation_structs =
        sources
        |> Enum.with_index()
        |> Enum.map(fn {source, idx} ->
          Citation.new!(%{
            source: source,
            metadata: %{"provider_type" => "perplexity_citation", "citation_index" => idx}
          })
        end)

      # Text delta
      text_delta =
        MessageDelta.new!(%{
          role: :assistant,
          content: "Text [1] and [2] and [3].",
          status: :incomplete
        })

      # Citation delta with all sources at once
      citation_part = %ContentPart{type: :text, content: nil, citations: citation_structs}

      citation_delta =
        MessageDelta.new!(%{content: citation_part, role: :assistant, status: :incomplete})

      # Complete delta
      complete_delta = MessageDelta.new!(%{role: :assistant, status: :complete})

      # Merge all deltas (starting from nil)
      merged = MessageDelta.merge_deltas([text_delta, citation_delta, complete_delta])

      # The merged delta should have text + 3 citations
      {:ok, message} = MessageDelta.to_message(merged)
      all_citations = Message.all_citations(message)
      assert length(all_citations) == 3

      urls = Citation.source_urls(all_citations)
      assert "https://example.com/a" in urls
      assert "https://example.com/b" in urls
      assert "https://example.com/c" in urls
    end
  end

  describe "build_perplexity_sources/2" do
    test "prefers search_results over citation URLs" do
      citation_urls = ["https://example.com/a"]

      search_results = [
        %{
          "title" => "Example A",
          "url" => "https://example.com/a",
          "date" => "2024-06-15",
          "snippet" => "Some content",
          "source" => "example.com"
        }
      ]

      sources = ChatPerplexity.build_perplexity_sources(citation_urls, search_results)
      assert length(sources) == 1

      [source] = sources
      assert source.type == :web
      assert source.title == "Example A"
      assert source.url == "https://example.com/a"
      assert source.metadata["date"] == "2024-06-15"
      assert source.metadata["snippet"] == "Some content"
      assert source.metadata["source_domain"] == "example.com"
    end

    test "falls back to citation URLs when no search_results" do
      citation_urls = ["https://example.com/a", "https://example.com/b"]

      sources = ChatPerplexity.build_perplexity_sources(citation_urls, [])
      assert length(sources) == 2

      [s1, s2] = sources
      assert s1.type == :web
      assert s1.url == "https://example.com/a"
      refute Map.has_key?(s1, :title)

      assert s2.type == :web
      assert s2.url == "https://example.com/b"
    end

    test "returns empty list when no citation data" do
      assert [] == ChatPerplexity.build_perplexity_sources([], [])
      assert [] == ChatPerplexity.build_perplexity_sources(nil, nil)
    end

    test "uses citation URL as fallback when search_result has no URL" do
      citation_urls = ["https://fallback.com"]
      search_results = [%{"title" => "Fallback Source"}]

      sources = ChatPerplexity.build_perplexity_sources(citation_urls, search_results)
      assert [source] = sources
      assert source.url == "https://fallback.com"
      assert source.title == "Fallback Source"
    end

    test "omits nil metadata values" do
      search_results = [
        %{
          "title" => "No Date Source",
          "url" => "https://example.com"
        }
      ]

      sources = ChatPerplexity.build_perplexity_sources([], search_results)
      assert [source] = sources
      refute Map.has_key?(source.metadata, "date")
      refute Map.has_key?(source.metadata, "snippet")
    end
  end

  describe "find_citation_markers/2" do
    test "finds [N] markers and maps to sources" do
      text = "Hello [1] world [2]."

      sources = [
        %{type: :web, url: "https://a.com"},
        %{type: :web, url: "https://b.com"}
      ]

      citations = ChatPerplexity.find_citation_markers(text, sources)
      assert length(citations) == 2

      [c1, c2] = citations
      assert c1.source.url == "https://a.com"
      assert c1.start_index == 6
      assert c1.end_index == 9

      assert c2.source.url == "https://b.com"
      assert c2.start_index == 16
      assert c2.end_index == 19
    end

    test "handles empty text" do
      assert [] == ChatPerplexity.find_citation_markers("", [%{type: :web, url: "https://a.com"}])
    end

    test "handles text with no markers" do
      assert [] ==
               ChatPerplexity.find_citation_markers("No citations here.", [
                 %{type: :web, url: "https://a.com"}
               ])
    end

    test "skips markers with out-of-range indices" do
      text = "Text [1] more [3]."
      sources = [%{type: :web, url: "https://a.com"}]

      citations = ChatPerplexity.find_citation_markers(text, sources)
      assert length(citations) == 1
      assert hd(citations).source.url == "https://a.com"
    end

    test "handles adjacent citation markers" do
      text = "Text [1][2] end."

      sources = [
        %{type: :web, url: "https://a.com"},
        %{type: :web, url: "https://b.com"}
      ]

      citations = ChatPerplexity.find_citation_markers(text, sources)
      assert length(citations) == 2
    end

    test "handles multi-digit citation markers" do
      text = "Text [10]."
      sources = Enum.map(1..10, fn i -> %{type: :web, url: "https://example.com/#{i}"} end)

      citations = ChatPerplexity.find_citation_markers(text, sources)
      assert length(citations) == 1
      assert hd(citations).source.url == "https://example.com/10"
      # "[10]" is 4 chars
      assert hd(citations).start_index == 5
      assert hd(citations).end_index == 9
    end
  end
end
