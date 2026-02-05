defmodule LangChain.ChatModels.ChatOpenAIResponsesTest do
  use LangChain.BaseCase
  use Mimic

  doctest LangChain.ChatModels.ChatOpenAIResponses
  alias LangChain.ChatModels.ChatOpenAIResponses
  alias LangChain.Function
  alias LangChain.FunctionParam

  @test_model "gpt-4o-mini-2024-07-18"

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
          }),
          FunctionParam.new!(%{
            name: "state",
            type: "string",
            description: "The 2 letter US state abbreviation, e.g. CA, NY, UT",
            required: true
          })
        ],
        function: fn _args, _context -> {:ok, "75 degrees"} end
      })

    %{hello_world: hello_world, weather: weather}
  end

  describe "new/1" do
    test "works with minimal attr" do
      assert {:ok, %ChatOpenAIResponses{} = openai} =
               ChatOpenAIResponses.new(%{"model" => @test_model})

      assert openai.model == @test_model
    end

    test "returns error when invalid" do
      assert {:error, changeset} = ChatOpenAIResponses.new(%{"model" => nil})
      refute changeset.valid?
      assert {"can't be blank", _} = changeset.errors[:model]
    end

    test "supports overriding the API endpoint" do
      override_url = "http://localhost:1234/v1/chat/completions"

      model =
        ChatOpenAIResponses.new!(%{
          endpoint: override_url
        })

      assert model.endpoint == override_url
    end

    test "supports setting json_response and json_schema" do
      json_schema = %{
        "type" => "object",
        "properties" => %{
          "name" => %{"type" => "string"},
          "age" => %{"type" => "integer"}
        }
      }

      {:ok, openai} =
        ChatOpenAIResponses.new(%{
          "model" => @test_model,
          "json_response" => true,
          "json_schema" => json_schema
        })

      assert openai.json_response == true
      assert openai.json_schema == json_schema
    end

    test "supports overriding temperature" do
      {:ok, openai} = ChatOpenAIResponses.new(%{"model" => @test_model, "temperature" => 0.7})
      assert openai.temperature == 0.7

      # Verify it's included in API call when not default
      data = ChatOpenAIResponses.for_api(openai, [], [])
      assert data.temperature == 0.7
    end

    test "returns error for out-of-bounds temperature" do
      assert {:error, changeset} =
               ChatOpenAIResponses.new(%{"model" => @test_model, "temperature" => 2.5})

      refute changeset.valid?
      assert {"must be less than or equal to %{number}", _} = changeset.errors[:temperature]

      assert {:error, changeset} =
               ChatOpenAIResponses.new(%{"model" => @test_model, "temperature" => -0.1})

      refute changeset.valid?
      assert {"must be greater than or equal to %{number}", _} = changeset.errors[:temperature]
    end

    test "top_p defaults to 1.0 and is included for gpt-4 models" do
      {:ok, openai} = ChatOpenAIResponses.new(%{"model" => @test_model})
      assert openai.top_p == 1.0

      data = ChatOpenAIResponses.for_api(openai, [], [])
      assert data.top_p == 1.0
    end

    test "top_p is excluded for gpt-5.2 and newer models" do
      {:ok, openai} = ChatOpenAIResponses.new(%{"model" => "gpt-5.2", "top_p" => 0.9})
      assert openai.top_p == 0.9

      data = ChatOpenAIResponses.for_api(openai, [], [])
      refute Map.has_key?(data, :top_p)
    end

    test "top_p is included for gpt-5.1 and earlier models" do
      {:ok, openai} = ChatOpenAIResponses.new(%{"model" => "gpt-5.1", "top_p" => 0.8})
      data = ChatOpenAIResponses.for_api(openai, [], [])
      assert data.top_p == 0.8

      {:ok, openai} = ChatOpenAIResponses.new(%{"model" => "gpt-5.0", "top_p" => 0.7})
      data = ChatOpenAIResponses.for_api(openai, [], [])
      assert data.top_p == 0.7
    end

    test "supports_top_p?/1 returns correct values for various models" do
      assert ChatOpenAIResponses.supports_top_p?("gpt-4")
      assert ChatOpenAIResponses.supports_top_p?("gpt-4o")
      assert ChatOpenAIResponses.supports_top_p?("gpt-4o-mini")
      assert ChatOpenAIResponses.supports_top_p?("gpt-4o-mini-2024-07-18")
      assert ChatOpenAIResponses.supports_top_p?("gpt-5.0")
      assert ChatOpenAIResponses.supports_top_p?("gpt-5.1")
      refute ChatOpenAIResponses.supports_top_p?("gpt-5.2")
      refute ChatOpenAIResponses.supports_top_p?("gpt-5.3")
      refute ChatOpenAIResponses.supports_top_p?("gpt-6")
    end

    test "supports overriding top_p" do
      {:ok, openai} = ChatOpenAIResponses.new(%{"model" => @test_model, "top_p" => 0.9})
      assert openai.top_p == 0.9

      data = ChatOpenAIResponses.for_api(openai, [], [])
      assert data.top_p == 0.9
    end

    test "returns error for out-of-bounds top_p" do
      assert {:error, changeset} =
               ChatOpenAIResponses.new(%{"model" => @test_model, "top_p" => 1.5})

      refute changeset.valid?
      assert {"must be less than or equal to %{number}", _} = changeset.errors[:top_p]

      assert {:error, changeset} =
               ChatOpenAIResponses.new(%{"model" => @test_model, "top_p" => -0.1})

      refute changeset.valid?
      assert {"must be greater than or equal to %{number}", _} = changeset.errors[:top_p]
    end

    test "supports setting reasoning options" do
      {:ok, openai} =
        ChatOpenAIResponses.new(%{
          "model" => @test_model,
          "reasoning" => %{
            "effort" => "high"
          }
        })

      assert openai.reasoning.effort == :high
    end

    test "validates reasoning_effort values" do
      assert {:error, changeset} =
               ChatOpenAIResponses.new(%{
                 "model" => @test_model,
                 "reasoning" => %{"effort" => "invalid"}
               })

      refute changeset.valid?
      assert changeset.errors == []
      assert changeset.changes.reasoning.errors[:effort] != nil
    end

    test "supports setting reasoning_summary" do
      {:ok, openai} =
        ChatOpenAIResponses.new(%{
          "model" => @test_model,
          "reasoning" => %{
            "summary" => "detailed"
          }
        })

      assert openai.reasoning.summary == :detailed
    end

    test "validates reasoning_summary values" do
      assert {:error, changeset} =
               ChatOpenAIResponses.new(%{
                 "model" => @test_model,
                 "reasoning" => %{"summary" => "invalid"}
               })

      refute changeset.valid?
      assert changeset.errors == []
      assert changeset.changes.reasoning.errors[:summary] != nil
    end

    test "supports setting reasoning_generate_summary (deprecated)" do
      {:ok, openai} =
        ChatOpenAIResponses.new(%{
          "model" => @test_model,
          "reasoning" => %{
            "generate_summary" => "concise"
          }
        })

      assert openai.reasoning.generate_summary == :concise
    end

    test "validates reasoning_generate_summary values" do
      assert {:error, changeset} =
               ChatOpenAIResponses.new(%{
                 "model" => @test_model,
                 "reasoning" => %{"generate_summary" => "invalid"}
               })

      refute changeset.valid?
      assert changeset.errors == []
      assert changeset.changes.reasoning.errors[:generate_summary] != nil
    end

    test "accepts all valid reasoning_effort values" do
      valid_efforts = ["minimal", "low", "medium", "high"]

      for effort <- valid_efforts do
        assert {:ok, %ChatOpenAIResponses{reasoning: reasoning}} =
                 ChatOpenAIResponses.new(%{
                   "model" => @test_model,
                   "reasoning" => %{"effort" => effort}
                 })

        assert reasoning.effort == String.to_atom(effort)
      end
    end

    test "accepts all valid reasoning summary values" do
      valid_summaries = ["auto", "concise", "detailed"]

      for summary <- valid_summaries do
        assert {:ok, %ChatOpenAIResponses{reasoning: reasoning}} =
                 ChatOpenAIResponses.new(%{
                   "model" => @test_model,
                   "reasoning" => %{"summary" => summary}
                 })

        assert reasoning.summary == String.to_atom(summary)

        assert {:ok, %ChatOpenAIResponses{reasoning: reasoning}} =
                 ChatOpenAIResponses.new(%{
                   "model" => @test_model,
                   "reasoning" => %{"generate_summary" => summary}
                 })

        assert reasoning.generate_summary == String.to_atom(summary)
      end
    end

    # Support
  end

  describe "for_api/3 reasoning options" do
    test "includes reasoning options when set" do
      openai =
        ChatOpenAIResponses.new!(%{
          "model" => @test_model,
          "reasoning" => %{
            "effort" => "high",
            "summary" => "detailed"
          }
        })

      result = ChatOpenAIResponses.for_api(openai, [], [])

      assert result.reasoning == %{
               "effort" => "high",
               "summary" => "detailed"
             }
    end

    test "excludes reasoning when no options are set" do
      openai = ChatOpenAIResponses.new!(%{"model" => @test_model})

      result = ChatOpenAIResponses.for_api(openai, [], [])

      refute Map.has_key?(result, :reasoning)
    end

    test "includes only set reasoning options" do
      openai =
        ChatOpenAIResponses.new!(%{
          "model" => @test_model,
          "reasoning" => %{
            "effort" => "medium"
          }
        })

      result = ChatOpenAIResponses.for_api(openai, [], [])

      assert result.reasoning == %{"effort" => "medium"}
    end

    test "includes deprecated reasoning_generate_summary" do
      openai =
        ChatOpenAIResponses.new!(%{
          "model" => @test_model,
          "reasoning" => %{
            "generate_summary" => "auto"
          }
        })

      result = ChatOpenAIResponses.for_api(openai, [], [])

      assert result.reasoning == %{"generate_summary" => "auto"}
    end

    test "includes all reasoning options when all are set" do
      openai =
        ChatOpenAIResponses.new!(%{
          "model" => @test_model,
          "reasoning" => %{
            "effort" => "low",
            "summary" => "concise",
            "generate_summary" => "auto"
          }
        })

      result = ChatOpenAIResponses.for_api(openai, [], [])

      assert result.reasoning == %{
               "effort" => "low",
               "summary" => "concise",
               "generate_summary" => "auto"
             }
    end
  end

  describe "for_api/3 messages" do
    test "generates a map for an API call" do
      {:ok, openai} =
        ChatOpenAIResponses.new(%{
          "model" => @test_model,
          "temperature" => 1,
          "api_key" => "api_key"
        })

      data = ChatOpenAIResponses.for_api(openai, [], [])
      assert data.model == @test_model
      assert data[:temperature] == 1.0
      assert data.store == false
      assert data[:text] == nil
      assert data.input == []
      assert data[:tools] == nil
    end

    test "generates a map for an API call with JSON response set to true" do
      {:ok, openai} =
        ChatOpenAIResponses.new(%{
          "model" => @test_model,
          "json_response" => true
        })

      data = ChatOpenAIResponses.for_api(openai, [], [])
      assert data.model == @test_model
      assert data.text == %{"format" => %{"type" => "json_object"}}
      refute data[:temperature]
    end

    test "generates a map for an API call with JSON response and schema" do
      json_schema = %{
        "type" => "object",
        "properties" => %{
          "name" => %{"type" => "string"},
          "age" => %{"type" => "integer"}
        }
      }

      {:ok, openai} =
        ChatOpenAIResponses.new(%{
          "model" => @test_model,
          "temperature" => 1,
          "json_response" => true,
          "json_schema" => json_schema,
          "json_schema_name" => "person"
        })

      data = ChatOpenAIResponses.for_api(openai, [], [])
      assert data.model == @test_model

      assert data.text == %{
               "format" => %{
                 "type" => "json_schema",
                 "name" => "person",
                 "schema" => json_schema,
                 "strict" => true
               }
             }
    end

    test "includes tools when provided" do
      {:ok, openai} = ChatOpenAIResponses.new(%{"model" => @test_model})

      {:ok, weather} =
        Function.new(%{
          name: "get_weather",
          description: "Get weather",
          function: fn _, _ -> {:ok, "sunny"} end
        })

      data = ChatOpenAIResponses.for_api(openai, [], [weather])
      assert length(data.tools) == 1
      [tool] = data.tools
      assert tool["type"] == "function"
      assert tool["name"] == "get_weather"
    end

    test "sets tool_choice correctly for different values" do
      # Auto mode (default)
      {:ok, openai} = ChatOpenAIResponses.new(%{"model" => @test_model, "tool_choice" => "auto"})
      data = ChatOpenAIResponses.for_api(openai, [], [])
      assert data.tool_choice == "auto"

      # None mode
      {:ok, openai} = ChatOpenAIResponses.new(%{"model" => @test_model, "tool_choice" => "none"})
      data = ChatOpenAIResponses.for_api(openai, [], [])
      assert data.tool_choice == "none"

      # Required mode
      {:ok, openai} =
        ChatOpenAIResponses.new(%{"model" => @test_model, "tool_choice" => "required"})

      data = ChatOpenAIResponses.for_api(openai, [], [])
      assert data.tool_choice == "required"

      # Specific function
      {:ok, openai} =
        ChatOpenAIResponses.new(%{"model" => @test_model, "tool_choice" => "get_weather"})

      data = ChatOpenAIResponses.for_api(openai, [], [])
      assert data.tool_choice == %{"type" => "function", "name" => "get_weather"}

      # Native tool
      {:ok, openai} =
        ChatOpenAIResponses.new(%{"model" => @test_model, "tool_choice" => "web_search_preview"})

      data = ChatOpenAIResponses.for_api(openai, [], [])
      assert data.tool_choice == %{"type" => "web_search_preview"}
    end
  end

  describe "for_api/1 content parts" do
    test "converts text content part to input_text" do
      model = ChatOpenAIResponses.new!(%{"model" => @test_model})
      msg = LangChain.Message.new_user!([LangChain.Message.ContentPart.text!("Hello")])

      api = ChatOpenAIResponses.for_api(model, msg)
      assert api["role"] == "user"
      assert api["type"] == "message"
      [part] = api["content"]
      assert part == %{"type" => "input_text", "text" => "Hello"}
    end

    test "converts image content to input_image with detail and media" do
      model = ChatOpenAIResponses.new!(%{"model" => @test_model})
      img = LangChain.Message.ContentPart.image!("BASE64DATA", media: :png, detail: "low")
      msg = LangChain.Message.new_user!([img])

      api = ChatOpenAIResponses.for_api(model, msg)
      [part] = api["content"]
      assert part["type"] == "input_image"
      assert String.starts_with?(part["image_url"], "data:image/png;base64,")
      assert part["detail"] == "low"
    end

    test "converts file_id to input_image" do
      model = ChatOpenAIResponses.new!(%{"model" => @test_model})
      file = LangChain.Message.ContentPart.image!("file-123", type: :file_id)
      msg = LangChain.Message.new_user!([file])

      api = ChatOpenAIResponses.for_api(model, msg)
      [part] = api["content"]
      assert part["type"] == "input_image"
      assert part["file_id"] == "file-123"
    end

    test "converts file base64 to input_file with filename" do
      model = ChatOpenAIResponses.new!(%{"model" => @test_model})
      file = LangChain.Message.ContentPart.file!("PDF_BASE64", type: :base64, filename: "a.pdf")
      msg = LangChain.Message.new_user!([file])

      api = ChatOpenAIResponses.for_api(model, msg)
      [part] = api["content"]
      assert part["type"] == "input_file"
      assert part["filename"] == "a.pdf"
      assert String.starts_with?(part["file_data"], "data:application/pdf;base64,")
    end

    test "converts file_id to input_file" do
      model = ChatOpenAIResponses.new!(%{"model" => @test_model})
      file = LangChain.Message.ContentPart.file!("file-123", type: :file_id)
      msg = LangChain.Message.new_user!([file])

      api = ChatOpenAIResponses.for_api(model, msg)
      [part] = api["content"]
      assert part["type"] == "input_file"
      assert part["file_id"] == "file-123"
    end

    test "converts file_url to input_file" do
      model = ChatOpenAIResponses.new!(%{"model" => @test_model})

      file_url =
        LangChain.Message.ContentPart.new!(%{
          type: :file_url,
          content: "https://example.com/document.pdf"
        })

      msg = LangChain.Message.new_user!([file_url])

      api = ChatOpenAIResponses.for_api(model, msg)
      [part] = api["content"]
      assert part["type"] == "input_file"
      assert part["file_url"] == "https://example.com/document.pdf"
    end

    test "omits thinking content parts when converting to API format" do
      model = ChatOpenAIResponses.new!(%{"model" => @test_model})

      # Create a message with thinking content (e.g., from a previous assistant response)
      thinking_part =
        LangChain.Message.ContentPart.new!(%{type: :thinking, content: "Some reasoning"})

      text_part = LangChain.Message.ContentPart.text!("Here's my answer")

      msg = LangChain.Message.new_assistant!([thinking_part, text_part])

      api = ChatOpenAIResponses.for_api(model, msg)

      # The result should be a list with the message and no tool calls
      assert is_list(api)
      [message_api] = api
      assert message_api["type"] == "message"

      # Content should only include the text part, not the thinking part
      [content_part] = message_api["content"]
      assert content_part["type"] == "input_text"
      assert content_part["text"] == "Here's my answer"
    end
  end

  describe "for_api/1 tool calls and results" do
    test "turns a tool_call into expected JSON format" do
      tool_call =
        LangChain.Message.ToolCall.new!(%{
          call_id: "call_abc123",
          name: "hello_world",
          arguments: %{}
        })

      json = ChatOpenAIResponses.for_api(ChatOpenAIResponses.new!(), tool_call)

      assert json == %{
               "call_id" => "call_abc123",
               "type" => "function_call",
               "name" => "hello_world",
               "arguments" => "{}",
               "status" => "completed"
             }
    end

    test "turns an assistant tool_call into expected JSON format with arguments" do
      msg =
        LangChain.Message.new_assistant!(%{
          tool_calls: [
            LangChain.Message.ToolCall.new!(%{
              call_id: "call_abc123",
              name: "calculator",
              arguments: %{expression: "11 + 10"}
            })
          ]
        })

      result = ChatOpenAIResponses.for_api(ChatOpenAIResponses.new!(), msg)

      assert is_list(result)
      assert length(result) == 1
      [tool_call] = result
      assert tool_call["type"] == "function_call"
      assert tool_call["call_id"] == "call_abc123"
      assert tool_call["name"] == "calculator"
    end

    test "converts tool result to function_call_output" do
      tool_result =
        LangChain.Message.ToolResult.new!(%{
          tool_call_id: "call_123",
          content: [LangChain.Message.ContentPart.text!("Result: 42")]
        })

      msg = LangChain.Message.new_tool_result!(%{tool_results: [tool_result]})
      result = ChatOpenAIResponses.for_api(ChatOpenAIResponses.new!(), msg)

      assert is_list(result)
      [output] = result
      assert output["type"] == "function_call_output"
      assert output["call_id"] == "call_123"
      assert output["output"] == "Result: 42"
    end

    test "handles assistant message with both content and tool calls" do
      msg =
        LangChain.Message.new_assistant!(%{
          content: [LangChain.Message.ContentPart.text!("Let me calculate that")],
          tool_calls: [
            LangChain.Message.ToolCall.new!(%{
              call_id: "call_123",
              name: "calculator",
              arguments: %{expr: "2+2"}
            })
          ]
        })

      result = ChatOpenAIResponses.for_api(ChatOpenAIResponses.new!(), msg)

      # Should return a list with message and tool call
      assert is_list(result)
      assert length(result) == 2

      [message, tool_call] = result
      assert message["type"] == "message"
      # Assistant messages become user in Responses API
      assert message["role"] == "user"
      assert tool_call["type"] == "function_call"
    end

    test "handles multiple tool results" do
      tool_result1 =
        LangChain.Message.ToolResult.new!(%{
          tool_call_id: "call_1",
          content: [LangChain.Message.ContentPart.text!("Result 1")]
        })

      tool_result2 =
        LangChain.Message.ToolResult.new!(%{
          tool_call_id: "call_2",
          content: [LangChain.Message.ContentPart.text!("Result 2")]
        })

      msg = LangChain.Message.new_tool_result!(%{tool_results: [tool_result1, tool_result2]})
      result = ChatOpenAIResponses.for_api(ChatOpenAIResponses.new!(), msg)

      assert is_list(result)
      assert length(result) == 2

      [output1, output2] = result
      assert output1["type"] == "function_call_output"
      assert output1["call_id"] == "call_1"
      assert output2["call_id"] == "call_2"
    end
  end

  describe "do_process_response non-streaming" do
    setup do
      %{model: ChatOpenAIResponses.new!(%{"model" => @test_model})}
    end

    test "handles completed response with text output", %{model: model} do
      response = %{
        "status" => "completed",
        "output" => [
          %{
            "type" => "message",
            "content" => [
              %{"type" => "output_text", "text" => "Hello, world!"}
            ]
          }
        ]
      }

      result = ChatOpenAIResponses.do_process_response(model, response)
      assert %LangChain.Message{} = result
      assert result.role == :assistant
      assert result.status == :complete
      [content_part] = result.content
      assert content_part.type == :text
      assert content_part.content == "Hello, world!"
    end

    test "handles completed response with function call", %{model: model} do
      response = %{
        "status" => "completed",
        "output" => [
          %{
            "type" => "function_call",
            "call_id" => "call_123",
            "name" => "get_weather",
            "arguments" => ~s({"city":"NYC"})
          }
        ]
      }

      result = ChatOpenAIResponses.do_process_response(model, response)
      assert %LangChain.Message{} = result
      assert result.role == :assistant
      assert length(result.tool_calls) == 1
      [tool_call] = result.tool_calls
      assert tool_call.call_id == "call_123"
      assert tool_call.name == "get_weather"
      assert tool_call.arguments == %{"city" => "NYC"}
    end

    test "handles completed response with multiple outputs", %{model: model} do
      response = %{
        "status" => "completed",
        "output" => [
          %{
            "type" => "message",
            "content" => [%{"type" => "output_text", "text" => "I'll check the weather."}]
          },
          %{
            "type" => "function_call",
            "call_id" => "call_456",
            "name" => "get_weather",
            "arguments" => "{}"
          }
        ]
      }

      result = ChatOpenAIResponses.do_process_response(model, response)
      assert %LangChain.Message{} = result
      assert result.role == :assistant
      assert length(result.content) == 1
      assert length(result.tool_calls) == 1
    end

    test "handles completed response with usage metadata", %{model: model} do
      response = %{
        "status" => "completed",
        "output" => [
          %{
            "content" => [
              %{
                "annotations" => [],
                "logprobs" => [],
                "text" => "hello",
                "type" => "output_text"
              }
            ],
            "role" => "assistant",
            "status" => "completed",
            "type" => "message"
          }
        ],
        "usage" => %{
          "input_tokens" => 27,
          "input_tokens_details" => %{"cached_tokens" => 0},
          "output_tokens" => 115,
          "output_tokens_details" => %{"reasoning_tokens" => 0},
          "total_tokens" => 142
        }
      }

      result = ChatOpenAIResponses.do_process_response(model, response)
      assert %LangChain.Message{} = result
      assert %{usage: %LangChain.TokenUsage{} = usage} = result.metadata
      assert %LangChain.TokenUsage{input: 27, output: 115} = usage
    end

    test "handles completed response with file_search results", %{model: model} do
      response = %{
        "status" => "completed",
        "output" => [
          %{
            "id" => "fs_0d1b1549e16f51d20168e6af70c7b8819fae111b23577a202d",
            "queries" => ["What is the meaning of life?"],
            "results" => [
              %{
                "attributes" => %{},
                "file_id" => "file-yTrvU63VL1AgEyFqtUYzfVT3",
                "filename" => "Enreach_Contact.pdf",
                "score" => 0.0059,
                "text" => "text part 1",
                "vector_store_id" => "vs_1"
              },
              %{
                "attributes" => %{},
                "file_id" => "file-yTrvU63VL1AgEyFqtUYzfVT3",
                "filename" => "Enreach_Contact.pdf",
                "score" => 0.0037,
                "text" => "text part 2",
                "vector_store_id" => "vs_2"
              }
            ],
            "status" => "completed",
            "type" => "file_search_call"
          }
        ]
      }

      result = ChatOpenAIResponses.do_process_response(model, response)
      assert %LangChain.Message{} = result
      assert result.role == :assistant
      assert result.status == :complete
      [content_part] = result.content
      assert content_part.type == :unsupported
      assert %{results: [_, _], queries: [_], type: "file_search_call"} = content_part.options
    end

    test "handles error responses", %{model: model} do
      response = %{"error" => %{"message" => "API key invalid"}}

      assert {:error, %LangChain.LangChainError{} = error} =
               ChatOpenAIResponses.do_process_response(model, response)

      assert error.message == "API key invalid"
    end

    test "handles failed response status", %{model: model} do
      response = %{
        "response" => %{
          "status" => "failed",
          "error" => %{
            "code" => "server_error",
            "message" => "The server had an error processing your request"
          }
        }
      }

      assert {:error, %LangChain.LangChainError{} = error} =
               ChatOpenAIResponses.do_process_response(model, response)

      # Uses actual error code from response for better error categorization
      assert error.type == "server_error"
      assert error.message =~ "The server had an error processing your request"
    end

    test "handles failed response status without error details", %{model: model} do
      response = %{
        "response" => %{
          "status" => "failed"
        }
      }

      assert {:error, %LangChain.LangChainError{} = error} =
               ChatOpenAIResponses.do_process_response(model, response)

      # Falls back to "api_error" when no error code provided
      assert error.type == "api_error"
      assert error.message =~ "failed"
    end

    test "handles failed response status with string error", %{model: model} do
      response = %{
        "response" => %{
          "status" => "failed",
          "error" => "Something went wrong"
        }
      }

      assert {:error, %LangChain.LangChainError{} = error} =
               ChatOpenAIResponses.do_process_response(model, response)

      # Handles string error format defensively
      assert error.type == "api_error"
      assert error.message =~ "Something went wrong"
    end
  end

  describe "do_process_response streaming events" do
    setup do
      %{model: ChatOpenAIResponses.new!(%{"model" => @test_model})}
    end

    test "parses response.output_text.delta", %{model: model} do
      delta = %{"type" => "response.output_text.delta", "delta" => "Hi"}

      assert %LangChain.MessageDelta{content: "Hi", status: :incomplete, role: :assistant} =
               ChatOpenAIResponses.do_process_response(model, delta)
    end

    test "parses response.output_text.done", %{model: model} do
      done = %{"type" => "response.output_text.done"}

      assert %LangChain.MessageDelta{status: :complete, role: :assistant} =
               ChatOpenAIResponses.do_process_response(model, done)
    end

    test "parses function call added/delta/done sequence", %{model: model} do
      added = %{
        "type" => "response.output_item.added",
        "output_index" => 0,
        "item" => %{
          "type" => "function_call",
          "call_id" => "call_1",
          "name" => "calc",
          "arguments" => ""
        }
      }

      %LangChain.MessageDelta{tool_calls: [call1]} =
        ChatOpenAIResponses.do_process_response(model, added)

      assert call1.type == :function
      assert call1.name == "calc"
      assert call1.call_id == "call_1"

      arg_delta = %{
        "type" => "response.function_call_arguments.delta",
        "output_index" => 0,
        "delta" => "{\"expression\": \"1+1\"}"
      }

      %LangChain.MessageDelta{tool_calls: [call2]} =
        ChatOpenAIResponses.do_process_response(model, arg_delta)

      assert call2.arguments == "{\"expression\": \"1+1\"}"

      done = %{
        "type" => "response.output_item.done",
        "output_index" => 0,
        "item" => %{
          "type" => "function_call",
          "call_id" => "call_1",
          "name" => "calc",
          "arguments" => "{\"expression\":\"1+1\"}"
        }
      }

      %LangChain.MessageDelta{status: :complete, tool_calls: [call3]} =
        ChatOpenAIResponses.do_process_response(model, done)

      assert call3.status == :complete
      assert call3.name == "calc"
      assert call3.arguments == %{"expression" => "1+1"}
    end

    test "parses reasoning output_item.added event", %{model: model} do
      event = %{
        "type" => "response.output_item.added",
        "sequence_number" => 2,
        "output_index" => 0,
        "item" => %{
          "id" => "rs_077ecb7bd77f1554016940159a98d081909d82480668e57471",
          "type" => "reasoning",
          "summary" => []
        }
      }

      result = ChatOpenAIResponses.do_process_response(model, event)
      assert %LangChain.MessageDelta{} = result
      assert result.status == :incomplete
      assert result.role == :assistant
      assert result.index == 0
      assert %LangChain.Message.ContentPart{type: :thinking, content: ""} = result.content
    end

    test "parses response.reasoning.delta event", %{model: model} do
      event = %{
        "type" => "response.reasoning.delta",
        "output_index" => 0,
        "delta" => "Let me think about this problem..."
      }

      result = ChatOpenAIResponses.do_process_response(model, event)
      assert %LangChain.MessageDelta{} = result
      assert result.status == :incomplete
      assert result.role == :assistant
      assert result.index == 0

      assert %LangChain.Message.ContentPart{
               type: :thinking,
               content: "Let me think about this problem..."
             } = result.content
    end

    test "parses reasoning output_item.done event", %{model: model} do
      event = %{
        "type" => "response.output_item.done",
        "sequence_number" => 3,
        "output_index" => 0,
        "item" => %{
          "id" => "rs_063b9657f7c2e68601694016d7008881909a128744538cebec",
          "type" => "reasoning",
          "summary" => []
        }
      }

      result = ChatOpenAIResponses.do_process_response(model, event)
      assert %LangChain.MessageDelta{} = result
      assert result.status == :complete
      assert result.role == :assistant
      assert result.index == 0
      assert %LangChain.Message.ContentPart{type: :thinking, content: ""} = result.content
    end

    test "parses response.output_text.delta with output_index", %{model: model} do
      delta = %{
        "type" => "response.output_text.delta",
        "output_index" => 1,
        "delta" => "Hello"
      }

      result = ChatOpenAIResponses.do_process_response(model, delta)

      assert %LangChain.MessageDelta{
               content: "Hello",
               status: :incomplete,
               role: :assistant,
               index: 1
             } = result
    end

    test "parses response.completed with token usage", %{model: model} do
      completed = %{
        "type" => "response.completed",
        "response" => %{"usage" => %{"input_tokens" => 5, "output_tokens" => 2}}
      }

      %LangChain.MessageDelta{metadata: %{usage: %LangChain.TokenUsage{input: 5, output: 2}}} =
        ChatOpenAIResponses.do_process_response(model, completed)
    end

    test "handles response.failed streaming event", %{model: model} do
      failed_event = %{
        "type" => "response.failed",
        "response" => %{
          "status" => "failed",
          "error" => %{
            "code" => "timeout",
            "message" => "Request timed out"
          }
        }
      }

      assert {:error, %LangChain.LangChainError{} = error} =
               ChatOpenAIResponses.do_process_response(model, failed_event)

      # Uses actual error code from response for better error categorization
      assert error.type == "timeout"
      assert error.message =~ "Request timed out"
    end

    test "fires on_llm_reasoning_delta callback for reasoning_summary_text.delta", %{model: model} do
      test_pid = self()

      model_with_callback = %{
        model
        | callbacks: [
            %{on_llm_reasoning_delta: fn delta -> send(test_pid, {:reasoning_delta, delta}) end}
          ]
      }

      event = %{
        "type" => "response.reasoning_summary_text.delta",
        "delta" => "Let me think..."
      }

      assert :skip == ChatOpenAIResponses.do_process_response(model_with_callback, event)
      assert_received {:reasoning_delta, "Let me think..."}
    end

    test "fires on_llm_reasoning_delta callback for reasoning_summary.delta", %{model: model} do
      test_pid = self()

      model_with_callback = %{
        model
        | callbacks: [
            %{on_llm_reasoning_delta: fn delta -> send(test_pid, {:reasoning_delta, delta}) end}
          ]
      }

      event = %{
        "type" => "response.reasoning_summary.delta",
        "delta" => "Reasoning step..."
      }

      assert :skip == ChatOpenAIResponses.do_process_response(model_with_callback, event)
      assert_received {:reasoning_delta, "Reasoning step..."}
    end

    test "skips reasoning summary non-delta events without callback", %{model: model} do
      events = [
        %{"type" => "response.reasoning_summary_text.done"},
        %{"type" => "response.reasoning_summary_part.added"},
        %{"type" => "response.reasoning_summary_part.done"},
        %{"type" => "response.reasoning_summary.done"}
      ]

      for event <- events do
        assert :skip == ChatOpenAIResponses.do_process_response(model, event)
      end
    end

    test "skips expected streaming events", %{model: model} do
      events_to_skip = [
        %{"type" => "response.created"},
        %{"type" => "response.in_progress"},
        %{"type" => "response.content_part.added"},
        %{"type" => "response.content_part.done"},
        %{"type" => "response.function_call_arguments.done"},
        %{"type" => "response.queued"}
      ]

      for event <- events_to_skip do
        assert :skip == ChatOpenAIResponses.do_process_response(model, event)
      end
    end

    test "handles list of streaming events", %{model: model} do
      events = [
        %{"type" => "response.output_text.delta", "delta" => "Hello"},
        %{"type" => "response.output_text.delta", "delta" => " world"},
        %{"type" => "response.output_text.done"}
      ]

      results = ChatOpenAIResponses.do_process_response(model, events)
      assert is_list(results)
      assert length(results) == 3

      [d1, d2, done] = results
      assert %LangChain.MessageDelta{content: "Hello", status: :incomplete} = d1
      assert %LangChain.MessageDelta{content: " world", status: :incomplete} = d2
      assert %LangChain.MessageDelta{status: :complete} = done
    end
  end

  describe "decode_stream/1" do
    test "decodes event-based streaming format" do
      raw = "event: response.output_text.delta\ndata: {\"delta\": \"Hi\"}\n\n"
      {[parsed], buffer} = ChatOpenAIResponses.decode_stream({raw, ""})

      assert parsed == %{"delta" => "Hi"}
      assert buffer == ""
    end

    test "handles multiple events in one chunk" do
      raw =
        "event: response.output_text.delta\ndata: {\"delta\": \"A\"}\n\nevent: response.output_text.delta\ndata: {\"delta\": \"B\"}\n\n"

      {parsed, buffer} = ChatOpenAIResponses.decode_stream({raw, ""})

      assert length(parsed) == 2
      assert buffer == ""
    end

    test "handles incomplete JSON across chunks" do
      chunk1 = "event: response.output_text.delta\ndata: {\"del"
      chunk2 = "ta\": \"test\"}\n\n"

      {[], buffer1} = ChatOpenAIResponses.decode_stream({chunk1, ""})
      assert buffer1 != ""

      {[parsed], buffer2} = ChatOpenAIResponses.decode_stream({chunk2, buffer1})
      assert parsed == %{"delta" => "test"}
      assert buffer2 == ""
    end
  end

  describe "serialize and restore" do
    test "serializes and restores config" do
      original =
        ChatOpenAIResponses.new!(%{
          "model" => @test_model,
          "temperature" => 0.7,
          "endpoint" => "https://custom.api/v1/responses",
          "reasoning" => %{"effort" => "high"}
        })

      config = ChatOpenAIResponses.serialize_config(original)
      assert config["model"] == @test_model
      assert config["temperature"] == 0.7
      assert config["endpoint"] == "https://custom.api/v1/responses"
      assert config["reasoning"]["effort"] == "high"

      {:ok, restored} = ChatOpenAIResponses.restore_from_map(config)
      assert restored.model == original.model
      assert restored.temperature == original.temperature
      assert restored.endpoint == original.endpoint
      assert restored.reasoning.effort == original.reasoning.effort
    end
  end

  describe "previous_response_id" do
    test "accepts previous_response_id in new/1" do
      {:ok, model} =
        ChatOpenAIResponses.new(%{
          "model" => @test_model,
          "previous_response_id" => "resp_abc123"
        })

      assert model.previous_response_id == "resp_abc123"
    end

    test "includes previous_response_id in API request when set" do
      model =
        ChatOpenAIResponses.new!(%{
          "model" => @test_model,
          "previous_response_id" => "resp_previous_123"
        })

      api_data = ChatOpenAIResponses.for_api(model, [], [])
      assert api_data.previous_response_id == "resp_previous_123"
    end

    test "omits previous_response_id from API request when nil" do
      model = ChatOpenAIResponses.new!(%{"model" => @test_model})

      api_data = ChatOpenAIResponses.for_api(model, [], [])
      refute Map.has_key?(api_data, :previous_response_id)
    end

    test "extracts response_id from completed response and adds to metadata" do
      model = ChatOpenAIResponses.new!(%{"model" => @test_model})

      response = %{
        "id" => "resp_new_456",
        "status" => "completed",
        "output" => [
          %{
            "type" => "message",
            "content" => [%{"type" => "output_text", "text" => "Hello!"}]
          }
        ]
      }

      result = ChatOpenAIResponses.do_process_response(model, response)
      assert %LangChain.Message{} = result
      assert result.metadata.response_id == "resp_new_456"
    end

    test "extracts response_id from streaming response.completed event" do
      model = ChatOpenAIResponses.new!(%{"model" => @test_model})

      completed_event = %{
        "type" => "response.completed",
        "response" => %{
          "id" => "resp_stream_789",
          "usage" => %{"input_tokens" => 10, "output_tokens" => 5}
        }
      }

      result = ChatOpenAIResponses.do_process_response(model, completed_event)
      assert %LangChain.MessageDelta{} = result
      assert result.metadata.response_id == "resp_stream_789"
    end

    test "conversation continuity pattern: response_id becomes previous_response_id" do
      # First call - no previous_response_id
      model1 = ChatOpenAIResponses.new!(%{"model" => @test_model})
      api_data1 = ChatOpenAIResponses.for_api(model1, [], [])
      refute Map.has_key?(api_data1, :previous_response_id)

      # Simulate response with id
      response1 = %{
        "id" => "resp_first_call",
        "status" => "completed",
        "output" => [
          %{
            "type" => "message",
            "content" => [%{"type" => "output_text", "text" => "First response"}]
          }
        ]
      }

      message1 = ChatOpenAIResponses.do_process_response(model1, response1)
      assert message1.metadata.response_id == "resp_first_call"

      # Second call - use response_id from first call as previous_response_id
      model2 =
        ChatOpenAIResponses.new!(%{
          "model" => @test_model,
          "previous_response_id" => message1.metadata.response_id
        })

      api_data2 = ChatOpenAIResponses.for_api(model2, [], [])
      assert api_data2.previous_response_id == "resp_first_call"
    end
  end

  describe "req_config" do
    test "merges req_config into the request (non-streaming)" do
      expect(Req, :post, fn req_struct ->
        # assert retry value from req_config
        assert req_struct.options.retry == false

        {:error, RuntimeError.exception("Something went wrong")}
      end)

      model =
        ChatOpenAIResponses.new!(%{
          stream: false,
          model: @test_model,
          req_config: %{retry: false}
        })

      assert {:error, _} = ChatOpenAIResponses.call(model, "prompt", [])
      verify!()
    end

    test "merges req_config into the request (streaming)" do
      expect(Req, :post, fn req_struct, _opts ->
        # assert retry value from req_config
        assert req_struct.options.retry == false

        {:error, RuntimeError.exception("Something went wrong")}
      end)

      model =
        ChatOpenAIResponses.new!(%{stream: true, model: @test_model, req_config: %{retry: false}})

      assert {:error, _} = ChatOpenAIResponses.call(model, "prompt", [])
      verify!()
    end
  end
end
