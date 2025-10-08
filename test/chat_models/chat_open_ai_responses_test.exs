defmodule LangChain.ChatModels.ChatOpenAIResponsesTest do
  use LangChain.BaseCase

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

    test "handles error responses", %{model: model} do
      response = %{"error" => %{"message" => "API key invalid"}}

      assert {:error, %LangChain.LangChainError{} = error} =
               ChatOpenAIResponses.do_process_response(model, response)

      assert error.message == "API key invalid"
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

    test "parses response.completed with token usage", %{model: model} do
      completed = %{
        "type" => "response.completed",
        "response" => %{"usage" => %{"input_tokens" => 5, "output_tokens" => 2}}
      }

      %LangChain.MessageDelta{metadata: %{usage: %LangChain.TokenUsage{input: 5, output: 2}}} =
        ChatOpenAIResponses.do_process_response(model, completed)
    end

    test "skips expected streaming events", %{model: model} do
      events_to_skip = [
        %{"type" => "response.created"},
        %{"type" => "response.in_progress"},
        %{"type" => "response.content_part.added"},
        %{"type" => "response.content_part.done"},
        %{"type" => "response.function_call_arguments.done"},
        %{"type" => "response.reasoning.delta"},
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
end
