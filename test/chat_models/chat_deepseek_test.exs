defmodule LangChain.ChatModels.ChatDeepSeekTest do
  use LangChain.BaseCase

  doctest LangChain.ChatModels.ChatDeepSeek
  alias LangChain.ChatModels.ChatDeepSeek
  alias LangChain.Function
  alias LangChain.FunctionParam
  alias LangChain.TokenUsage
  alias LangChain.LangChainError
  alias LangChain.Message
  alias LangChain.Message.ContentPart
  alias LangChain.Message.ToolCall
  alias LangChain.Message.ToolResult

  @test_model "deepseek-chat"
  @test_model_2 "deepseek-reasoner"

  defp hello_world(_args, _context) do
    "Hello world!"
  end

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
      assert {:ok, %ChatDeepSeek{} = deepseek} = ChatDeepSeek.new(%{"model" => @test_model})
      assert deepseek.model == @test_model
    end

    test "returns error when invalid" do
      assert {:error, changeset} = ChatDeepSeek.new(%{"model" => nil})
      refute changeset.valid?
      assert {"can't be blank", _} = changeset.errors[:model]
    end

    test "supports overriding the API endpoint" do
      override_url = "http://localhost:1234/chat/completions"

      model =
        ChatDeepSeek.new!(%{
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

      {:ok, deepseek} =
        ChatDeepSeek.new(%{
          "model" => @test_model,
          "json_response" => true,
          "json_schema" => json_schema
        })

      assert deepseek.json_response == true
      assert deepseek.json_schema == json_schema
    end

    test "supports passing parallel_tool_calls" do
      # defaults to nil
      %ChatDeepSeek{} = deepseek = ChatDeepSeek.new!()
      assert deepseek.parallel_tool_calls == nil

      # can override the default to false
      %ChatDeepSeek{} = deepseek = ChatDeepSeek.new!(%{"parallel_tool_calls" => false})
      assert deepseek.parallel_tool_calls == false
    end

    test "supports logprobs configuration" do
      # defaults to false
      %ChatDeepSeek{} = deepseek = ChatDeepSeek.new!()
      assert deepseek.logprobs == false
      assert deepseek.top_logprobs == nil

      # can enable logprobs
      %ChatDeepSeek{} = deepseek = ChatDeepSeek.new!(%{"logprobs" => true})
      assert deepseek.logprobs == true
      assert deepseek.top_logprobs == nil

      # can enable logprobs with top_logprobs
      %ChatDeepSeek{} = deepseek = ChatDeepSeek.new!(%{"logprobs" => true, "top_logprobs" => 5})
      assert deepseek.logprobs == true
      assert deepseek.top_logprobs == 5
    end

    test "validates top_logprobs requires logprobs" do
      assert {:error, changeset} = ChatDeepSeek.new(%{"top_logprobs" => 5})
      assert {"requires logprobs to be enabled", _} = changeset.errors[:top_logprobs]
    end

    test "validates top_logprobs range" do
      assert {:error, changeset} = ChatDeepSeek.new(%{"logprobs" => true, "top_logprobs" => 25})
      assert {"must be less than or equal to %{number}", _} = changeset.errors[:top_logprobs]

      assert {:error, changeset} = ChatDeepSeek.new(%{"logprobs" => true, "top_logprobs" => -1})
      assert {"must be greater than or equal to %{number}", _} = changeset.errors[:top_logprobs]
    end
  end

  describe "for_api/3" do
    test "generates a map for an API call" do
      {:ok, deepseek} =
        ChatDeepSeek.new(%{
          "model" => @test_model,
          "temperature" => 1,
          "frequency_penalty" => 0.5,
          "api_key" => "api_key"
        })

      data = ChatDeepSeek.for_api(deepseek, [], [])
      assert data.model == @test_model
      assert data.temperature == 1
      assert data.frequency_penalty == 0.5
      assert data[:response_format] == nil
      assert data[:parallel_tool_calls] == nil
    end

    test "when frequency_penalty is not explicitly configured, it is not specified in the API call" do
      {:ok, deepseek} = ChatDeepSeek.new(%{"model" => @test_model})
      data = ChatDeepSeek.for_api(deepseek, [], [])
      assert data[:frequency_penalty] == nil
    end

    test "generates a map for an API call with JSON response set to true" do
      {:ok, deepseek} =
        ChatDeepSeek.new(%{
          "model" => @test_model,
          "temperature" => 1,
          "frequency_penalty" => 0.5,
          "json_response" => true
        })

      data = ChatDeepSeek.for_api(deepseek, [], [])
      assert data.model == @test_model
      assert data.temperature == 1
      assert data.frequency_penalty == 0.5
      assert data.response_format == %{"type" => "json_object"}
    end

    test "generates a map for an API call with JSON response and schema" do
      json_schema = %{
        "type" => "object",
        "properties" => %{
          "name" => %{"type" => "string"},
          "age" => %{"type" => "integer"}
        }
      }

      {:ok, deepseek} =
        ChatDeepSeek.new(%{
          "model" => @test_model,
          "temperature" => 1,
          "frequency_penalty" => 0.5,
          "json_response" => true,
          "json_schema" => json_schema
        })

      data = ChatDeepSeek.for_api(deepseek, [], [])
      assert data.model == @test_model
      assert data.temperature == 1
      assert data.frequency_penalty == 0.5

      assert data.response_format == %{
               "type" => "json_schema",
               "json_schema" => json_schema
             }
    end

    test "generates a map for an API call with max_tokens set" do
      {:ok, deepseek} =
        ChatDeepSeek.new(%{
          "model" => @test_model,
          "temperature" => 1,
          "frequency_penalty" => 0.5,
          "max_tokens" => 1234
        })

      data = ChatDeepSeek.for_api(deepseek, [], [])
      assert data.model == @test_model
      assert data.temperature == 1
      assert data.frequency_penalty == 0.5
      assert data.max_tokens == 1234
    end

    test "generates a map for an API call with stream_options set correctly" do
      {:ok, deepseek} =
        ChatDeepSeek.new(%{
          model: @test_model,
          stream_options: %{include_usage: true}
        })

      data = ChatDeepSeek.for_api(deepseek, [], [])
      assert data.model == @test_model
      assert data.stream_options == %{"include_usage" => true}
    end

    test "generated a map for an API call with tool_choice set correctly to auto" do
      {:ok, deepseek} =
        ChatDeepSeek.new(%{
          model: @test_model,
          tool_choice: %{"type" => "auto"}
        })

      data = ChatDeepSeek.for_api(deepseek, [], [])
      assert data.model == @test_model
      assert data.tool_choice == "auto"
    end

    test "generated a map for an API call with tool_choice set correctly to a specific function" do
      {:ok, deepseek} =
        ChatDeepSeek.new(%{
          model: @test_model,
          tool_choice: %{"type" => "function", "function" => %{"name" => "get_weather"}}
        })

      data = ChatDeepSeek.for_api(deepseek, [], [])
      assert data.model == @test_model
      assert data.tool_choice == %{"type" => "function", "function" => %{"name" => "get_weather"}}
    end

    test "generated a map for an API call with parallel_tool_calls set to false" do
      {:ok, deepseek} =
        ChatDeepSeek.new(%{
          model: @test_model,
          parallel_tool_calls: false
        })

      data = ChatDeepSeek.for_api(deepseek, [], [])
      assert data.model == @test_model
      assert data.parallel_tool_calls == false
    end

    test "generates a map for an API call with logprobs enabled" do
      {:ok, deepseek} =
        ChatDeepSeek.new(%{
          model: @test_model,
          logprobs: true
        })

      data = ChatDeepSeek.for_api(deepseek, [], [])
      assert data.model == @test_model
      assert data.logprobs == true
      assert data[:top_logprobs] == nil
    end

    test "generates a map for an API call with logprobs and top_logprobs" do
      {:ok, deepseek} =
        ChatDeepSeek.new(%{
          model: @test_model,
          logprobs: true,
          top_logprobs: 5
        })

      data = ChatDeepSeek.for_api(deepseek, [], [])
      assert data.model == @test_model
      assert data.logprobs == true
      assert data.top_logprobs == 5
    end
  end

  describe "for_api/1" do
    test "turns a tool_call into expected JSON format" do
      tool_call =
        ToolCall.new!(%{call_id: "call_abc123", name: "hello_world", arguments: "{}"})

      json = ChatDeepSeek.for_api(ChatDeepSeek.new!(), tool_call)

      assert json ==
               %{
                 "id" => "call_abc123",
                 "type" => "function",
                 "function" => %{
                   "name" => "hello_world",
                   "arguments" => "\"{}\""
                 }
               }
    end

    test "turns an assistant tool_call into expected JSON format with arguments" do
      msg =
        Message.new_assistant!(%{
          tool_calls: [
            ToolCall.new!(%{
              call_id: "call_abc123",
              name: "hello_world",
              arguments: %{expression: "11 + 10"}
            })
          ]
        })

      json = ChatDeepSeek.for_api(ChatDeepSeek.new!(), msg)

      assert json == %{
               "role" => :assistant,
               "content" => nil,
               "tool_calls" => [
                 %{
                   "function" => %{
                     "arguments" => "{\"expression\":\"11 + 10\"}",
                     "name" => "hello_world"
                   },
                   "id" => "call_abc123",
                   "type" => "function"
                 }
               ]
             }
    end

    test "turns a tool message into expected JSON format" do
      msg =
        Message.new_tool_result!(%{
          tool_results: [
            ToolResult.new!(%{tool_call_id: "tool_abc123", content: "Hello World!"})
          ]
        })

      [json] = ChatDeepSeek.for_api(ChatDeepSeek.new!(), msg)

      assert json == %{
               "content" => "Hello World!",
               "tool_call_id" => "tool_abc123",
               "role" => :tool
             }
    end

    test "turns multiple tool results into expected JSON format" do
      message =
        Message.new_tool_result!(%{
          tool_results: [
            ToolResult.new!(%{tool_call_id: "tool_abc123", content: "Hello World!"})
          ]
        })
        |> Message.append_tool_result(
          ToolResult.new!(%{tool_call_id: "tool_abc234", content: "Hello"})
        )
        |> Message.append_tool_result(
          ToolResult.new!(%{tool_call_id: "tool_abc345", content: "World!"})
        )

      list = ChatDeepSeek.for_api(ChatDeepSeek.new!(), message)

      assert is_list(list)

      [r1, r2, r3] = list

      assert r1 == %{
               "content" => "Hello World!",
               "tool_call_id" => "tool_abc123",
               "role" => :tool
             }

      assert r2 == %{
               "content" => "Hello",
               "tool_call_id" => "tool_abc234",
               "role" => :tool
             }

      assert r3 == %{
               "content" => "World!",
               "tool_call_id" => "tool_abc345",
               "role" => :tool
             }
    end

    test "tools work with minimal definition and no parameters", %{hello_world: hello_world} do
      result = ChatDeepSeek.for_api(ChatDeepSeek.new!(), hello_world)

      assert result == %{
               "name" => "hello_world",
               "description" => "Give a hello world greeting",
               #  NOTE: Sends the required empty parameter definition when none set
               "parameters" => %{"properties" => %{}, "type" => "object"}
             }
    end

    test "supports parameters" do
      params_def = %{
        "type" => "object",
        "properties" => %{
          "p1" => %{"type" => "string"},
          "p2" => %{"description" => "Param 2", "type" => "number"},
          "p3" => %{
            "enum" => ["yellow", "red", "green"],
            "type" => "string"
          }
        },
        "required" => ["p1"]
      }

      {:ok, fun} =
        Function.new(%{
          name: "say_hi",
          description: "Provide a friendly greeting.",
          parameters: [
            FunctionParam.new!(%{name: "p1", type: :string, required: true}),
            FunctionParam.new!(%{name: "p2", type: :number, description: "Param 2"}),
            FunctionParam.new!(%{name: "p3", type: :string, enum: ["yellow", "red", "green"]})
          ],
          function: fn _args, _context -> {:ok, "SUCCESS"} end
        })

      result = ChatDeepSeek.for_api(ChatDeepSeek.new!(), fun)

      assert result == %{
               "name" => "say_hi",
               "description" => "Provide a friendly greeting.",
               "parameters" => params_def
             }
    end

    test "supports parameters_schema" do
      params_def = %{
        "type" => "object",
        "properties" => %{
          "p1" => %{"description" => nil, "type" => "string"},
          "p2" => %{"description" => "Param 2", "type" => "number"},
          "p3" => %{
            "description" => nil,
            "enum" => ["yellow", "red", "green"],
            "type" => "string"
          }
        },
        "required" => ["p1"]
      }

      {:ok, fun} =
        Function.new(%{
          name: "say_hi",
          description: "Provide a friendly greeting.",
          parameters_schema: params_def,
          function: fn _args, _context -> {:ok, "SUCCESS"} end
        })

      result = ChatDeepSeek.for_api(ChatDeepSeek.new!(), fun)

      assert result == %{
               "name" => "say_hi",
               "description" => "Provide a friendly greeting.",
               "parameters" => params_def
             }
    end

    test "does not allow both parameters and parameters_schema" do
      {:error, changeset} =
        Function.new(%{
          name: "problem",
          parameters: [
            FunctionParam.new!(%{name: "p1", type: :string, required: true})
          ],
          parameters_schema: %{stuff: true}
        })

      assert {"Cannot use both parameters and parameters_schema", _} =
               changeset.errors[:parameters]
    end

    test "does not include the function to execute" do
      {:ok, fun} = Function.new(%{"name" => "hello_world", "function" => &hello_world/2})
      result = ChatDeepSeek.for_api(ChatDeepSeek.new!(), fun)
      refute Map.has_key?(result, "function")
    end
  end

  describe "for_api/2" do
    test "turns a basic user message into the expected JSON format" do
      deepseek = ChatDeepSeek.new!()

      expected = %{"role" => :user, "content" => "Hi."}
      result = ChatDeepSeek.for_api(deepseek, Message.new_user!("Hi."))
      assert result == expected
    end

    test "includes 'name' when set" do
      deepseek = ChatDeepSeek.new!()

      expected = %{
        "role" => :user,
        "content" => "Hi.",
        "name" => "Harold"
      }

      result =
        ChatDeepSeek.for_api(
          deepseek,
          Message.new!(%{role: :user, content: "Hi.", name: "Harold"})
        )

      assert result == expected
    end

    test "turns an assistant message into expected JSON format" do
      deepseek = ChatDeepSeek.new!()

      expected = %{"role" => :assistant, "content" => "Hi."}

      result =
        ChatDeepSeek.for_api(deepseek, Message.new_assistant!(%{content: "Hi.", tool_calls: []}))

      assert result == expected
    end

    test "turns a ToolResult into the expected JSON format" do
      deepseek = ChatDeepSeek.new!()
      result = ToolResult.new!(%{tool_call_id: "tool_abc123", content: "Hello World!"})

      json = ChatDeepSeek.for_api(deepseek, result)

      assert json == %{
               "content" => "Hello World!",
               "tool_call_id" => "tool_abc123",
               "role" => :tool
             }
    end
  end

  describe "do_process_response/2" do
    setup do
      model = ChatDeepSeek.new!(%{"model" => @test_model})
      %{model: model}
    end

    test "returns skip when given an empty choices list", %{model: model} do
      assert :skip == ChatDeepSeek.do_process_response(model, %{"choices" => []})
    end

    test "handles receiving a message", %{model: model} do
      response = %{
        "message" => %{"role" => "assistant", "content" => "Greetings!"},
        "finish_reason" => "stop",
        "index" => 1
      }

      assert %Message{} = struct = ChatDeepSeek.do_process_response(model, response)
      assert struct.role == :assistant
      assert struct.content == [ContentPart.text!("Greetings!")]
      assert struct.index == 1
      assert struct.metadata == %{logprobs: nil}
    end

    test "handles receiving a message with token usage information", %{model: model} do
      response = %{
        "choices" => [
          %{
            "finish_reason" => "stop",
            "index" => 0,
            "message" => %{
              "content" => "Hello DeepSeek",
              "role" => "assistant"
            }
          }
        ],
        "usage" => %{
          "completion_tokens" => 3,
          "prompt_tokens" => 10,
          "total_tokens" => 13
        }
      }

      assert [%Message{} = struct] = ChatDeepSeek.do_process_response(model, response)
      assert struct.role == :assistant
      assert struct.content == [ContentPart.text!("Hello DeepSeek")]
      assert struct.index == 0
      %TokenUsage{} = usage = struct.metadata.usage
      assert usage.input == 10
      assert usage.output == 3
    end

    test "handles receiving a message with DeepSeek-specific metadata", %{model: model} do
      response = %{
        "id" => "8c78fede-06e9-4a15-b546-cc30f470ab77",
        "object" => "chat.completion",
        "model" => "deepseek-chat",
        "system_fingerprint" => "fp_ffc7281d48_prod0820_fp8_kvcache",
        "choices" => [
          %{
            "index" => 0,
            "message" => %{
              "role" => "assistant",
              "content" => "Hello! How can I assist you today? 😊"
            },
            "logprobs" => nil,
            "finish_reason" => "stop"
          }
        ],
        "usage" => %{
          "prompt_tokens" => 12,
          "completion_tokens" => 11,
          "total_tokens" => 23,
          "prompt_tokens_details" => %{
            "cached_tokens" => 0
          },
          "prompt_cache_hit_tokens" => 0,
          "prompt_cache_miss_tokens" => 12
        }
      }

      assert [%Message{} = struct] = ChatDeepSeek.do_process_response(model, response)
      assert struct.role == :assistant
      assert struct.content == [ContentPart.text!("Hello! How can I assist you today? 😊")]
      assert struct.index == 0

      # Check metadata contains DeepSeek-specific fields
      assert struct.metadata.system_fingerprint == "fp_ffc7281d48_prod0820_fp8_kvcache"
      assert struct.metadata.object == "chat.completion"
      assert struct.metadata.logprobs == nil

      # Check token usage includes detailed information
      %TokenUsage{} = usage = struct.metadata.usage
      assert usage.input == 12
      assert usage.output == 11
      assert usage.raw["prompt_cache_hit_tokens"] == 0
      assert usage.raw["prompt_cache_miss_tokens"] == 12
    end

    test "handles receiving a single tool_calls message", %{model: model} do
      response = %{
        "finish_reason" => "tool_calls",
        "index" => 0,
        "message" => %{
          "content" => nil,
          "role" => "assistant",
          "tool_calls" => [
            %{
              "function" => %{
                "arguments" => "{\"city\":\"Moab\",\"state\":\"UT\"}",
                "name" => "get_weather"
              },
              "id" => "call_mMSPuyLd915TQ9bcrk4NvLDX",
              "type" => "function"
            }
          ]
        }
      }

      assert %Message{} = struct = ChatDeepSeek.do_process_response(model, response)

      assert struct.role == :assistant

      assert [%ToolCall{} = call] = struct.tool_calls
      assert call.call_id == "call_mMSPuyLd915TQ9bcrk4NvLDX"
      assert call.type == :function
      assert call.name == "get_weather"
      assert call.arguments == %{"city" => "Moab", "state" => "UT"}
      assert struct.index == 0
    end

    test "handles receiving multiple tool_calls messages", %{model: model} do
      response = %{
        "finish_reason" => "tool_calls",
        "index" => 0,
        "message" => %{
          "content" => nil,
          "role" => "assistant",
          "tool_calls" => [
            %{
              "function" => %{
                "arguments" => "{\"city\": \"Moab\", \"state\": \"UT\"}",
                "name" => "get_weather"
              },
              "id" => "call_4L8NfePhSW8PdoHUWkvhzguu",
              "type" => "function"
            },
            %{
              "function" => %{
                "arguments" => "{\"city\": \"Portland\", \"state\": \"OR\"}",
                "name" => "get_weather"
              },
              "id" => "call_ylRu5SPegST9tppLEj6IJ0Rs",
              "type" => "function"
            }
          ]
        }
      }

      assert %Message{} = struct = ChatDeepSeek.do_process_response(model, response)

      assert struct.role == :assistant

      assert struct.tool_calls == [
               ToolCall.new!(%{
                 type: :function,
                 call_id: "call_4L8NfePhSW8PdoHUWkvhzguu",
                 name: "get_weather",
                 arguments: %{"city" => "Moab", "state" => "UT"},
                 status: :complete
               }),
               ToolCall.new!(%{
                 type: :function,
                 call_id: "call_ylRu5SPegST9tppLEj6IJ0Rs",
                 name: "get_weather",
                 arguments: %{"city" => "Portland", "state" => "OR"},
                 status: :complete
               })
             ]
    end

    test "handles error from server", %{model: model} do
      response = %{
        "error" => %{"code" => "429", "message" => "Rate limit exceeded"}
      }

      assert {:error, %LangChainError{} = error} =
               ChatDeepSeek.do_process_response(model, response)

      assert error.type == "rate_limit_exceeded"
      assert error.message == "Rate limit exceeded"
    end

    test "handles json parse error from server", %{model: model} do
      {:error, %LangChainError{} = error} =
        ChatDeepSeek.do_process_response(model, Jason.decode("invalid json"))

      assert error.type == "invalid_json"
      assert "Received invalid JSON: " <> _ = error.message
    end

    test "handles unexpected response", %{model: model} do
      {:error, %LangChainError{} = error} =
        ChatDeepSeek.do_process_response(model, "unexpected")

      assert error.type == nil
      assert error.message == "Unexpected response"
    end

    test "return multiple responses when given multiple choices", %{model: model} do
      response = %{
        "choices" => [
          %{
            "message" => %{"role" => "assistant", "content" => "Greetings!"},
            "finish_reason" => "stop",
            "index" => 0
          },
          %{
            "message" => %{"role" => "assistant", "content" => "Howdy!"},
            "finish_reason" => "stop",
            "index" => 1
          }
        ]
      }

      [msg1, msg2] = ChatDeepSeek.do_process_response(model, response)
      assert %Message{role: :assistant, index: 0} = msg1
      assert %Message{role: :assistant, index: 1} = msg2
      assert msg1.content == [ContentPart.text!("Greetings!")]
      assert msg2.content == [ContentPart.text!("Howdy!")]
    end
  end

  describe "do_process_response/2 for deepseek-reasoner" do
    setup do
      model = ChatDeepSeek.new!(%{"model" => @test_model_2})
      %{model: model}
    end

    test "handles receiving message with reasoning content", %{model: model} do
      response = %{
        "finish_reason" => "stop",
        "message" => %{
          "role" => "assistant",
          "content" => "test",
          "reasoning_content" => "reasoning test"
        },
        "index" => 0
      }

      assert %Message{} = struct = ChatDeepSeek.do_process_response(model, response)

      assert struct.content == [
               ContentPart.thinking!("reasoning test"),
               ContentPart.text!("test")
             ]
    end

    test "handles receiving message delta with reasoning content", %{model: model} do
      reasoning_response1 = %{
        "index" => 0,
        "delta" => %{
          "content" => nil,
          "reasoning_content" => "reasoning content"
        },
        "finish_reason" => nil
      }

      reasoning_response2 = %{
        "index" => 0,
        "delta" => %{
          "content" => "content",
          "reasoning_content" => nil
        },
        "finish_reason" => nil
      }

      assert %MessageDelta{} =
               delta1 = ChatDeepSeek.do_process_response(model, reasoning_response1)

      assert %MessageDelta{} =
               delta2 = ChatDeepSeek.do_process_response(model, reasoning_response2)

      assert delta1.content == ContentPart.thinking!("reasoning content")
      assert delta2.content == ContentPart.text!("content")
    end
  end

  describe "decode_stream/2" do
    test "correctly handles fully formed chat completion chunks" do
      data =
        "data: {\"id\":\"chatcmpl-abc123\",\"object\":\"chat.completion.chunk\",\"created\":1689801995,\"model\":\"deepseek-chat\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":\"Hello\"},\"finish_reason\":null}]}\n\ndata: {\"id\":\"chatcmpl-abc123\",\"object\":\"chat.completion.chunk\",\"created\":1689801995,\"model\":\"deepseek-chat\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\" World\"},\"finish_reason\":null}]}\n\n"

      {parsed, incomplete} = ChatDeepSeek.decode_stream({data, ""})

      # nothing incomplete. Parsed 2 objects.
      assert incomplete == ""
      assert length(parsed) == 2

      assert Enum.at(parsed, 0)["choices"] |> Enum.at(0) |> get_in(["delta", "content"]) ==
               "Hello"

      assert Enum.at(parsed, 1)["choices"] |> Enum.at(0) |> get_in(["delta", "content"]) ==
               " World"
    end

    test "correctly parses when data split over received messages" do
      data =
        "data: {\"id\":\"chatcmpl-abc123\",\"object\":\"chat.completion.chunk\",\"created\":1689801995,\"model\":\"deepseek-chat\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":\"Hello\"},\"finish_reason\":null}]}\n\n"

      {parsed, incomplete} = ChatDeepSeek.decode_stream({data, ""})

      # nothing incomplete. Parsed 1 object.
      assert incomplete == ""
      assert length(parsed) == 1

      assert Enum.at(parsed, 0)["choices"] |> Enum.at(0) |> get_in(["delta", "content"]) ==
               "Hello"
    end

    test "correctly parses when data split over decode calls" do
      buffered = "{\"id\":\"chatcmpl-abc123\",\"object\":\"chat.comple"

      data =
        "data: tion.chunk\",\"created\":1689801995,\"model\":\"deepseek-chat\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":\"Hello\"},\"finish_reason\":null}]}\n\n"

      {parsed, incomplete} = ChatDeepSeek.decode_stream({data, buffered})

      # nothing incomplete. Parsed 1 object.
      assert incomplete == ""
      assert length(parsed) == 1

      assert Enum.at(parsed, 0)["choices"] |> Enum.at(0) |> get_in(["delta", "content"]) ==
               "Hello"
    end

    test "handles incomplete JSON data" do
      data = "data: {\"id\":\"test\",\"incomplete\":"
      {parsed, incomplete} = ChatDeepSeek.decode_stream({data, ""})

      # Should handle incomplete JSON gracefully by returning it as incomplete
      assert incomplete == "{\"id\":\"test\",\"incomplete\":"
      assert parsed == []
    end

    test "handles malformed JSON chunks" do
      data = "data: {invalid json}\n\n"
      {parsed, incomplete} = ChatDeepSeek.decode_stream({data, ""})

      # Malformed JSON is returned as incomplete, not parsed
      assert incomplete == "{invalid json}"
      assert parsed == []
    end

    test "handles empty data" do
      {parsed, incomplete} = ChatDeepSeek.decode_stream({"", ""})

      assert parsed == []
      assert incomplete == ""
    end

    test "handles [DONE] marker" do
      data = "data: [DONE]\n\n"
      {parsed, incomplete} = ChatDeepSeek.decode_stream({data, ""})

      assert parsed == []
      assert incomplete == ""
    end
  end

  describe "processing complex responses" do
    setup do
      model = ChatDeepSeek.new!(%{"model" => @test_model})
      %{model: model}
    end

    test "handles response with content_filter finish reason", %{model: model} do
      response = %{
        "choices" => [
          %{
            "message" => %{"role" => "assistant", "content" => "Filtered content"},
            "finish_reason" => "content_filter",
            "index" => 0
          }
        ]
      }

      assert [%Message{} = msg] = ChatDeepSeek.do_process_response(model, response)
      assert msg.role == :assistant
      assert msg.status == :complete
    end

    test "handles response with logprobs data", %{model: model} do
      response = %{
        "choices" => [
          %{
            "message" => %{"role" => "assistant", "content" => "Test response"},
            "finish_reason" => "stop",
            "logprobs" => %{"token_logprobs" => [0.1, 0.2]},
            "index" => 0
          }
        ]
      }

      assert [%Message{} = msg] = ChatDeepSeek.do_process_response(model, response)
      assert msg.metadata.logprobs == %{"token_logprobs" => [0.1, 0.2]}
    end
  end

  describe "serialize_config/2" do
    test "does not include the API key or callbacks" do
      model = ChatDeepSeek.new!(%{model: @test_model})
      result = ChatDeepSeek.serialize_config(model)
      assert result["version"] == 1
      refute Map.has_key?(result, "api_key")
      refute Map.has_key?(result, "callbacks")
    end

    test "creates expected map" do
      model =
        ChatDeepSeek.new!(%{
          model: @test_model,
          temperature: 0,
          frequency_penalty: 0.5,
          seed: 123,
          max_tokens: 1234,
          stream_options: %{include_usage: true}
        })

      result = ChatDeepSeek.serialize_config(model)

      assert result == %{
               "endpoint" => "https://api.deepseek.com/chat/completions",
               "frequency_penalty" => 0.5,
               "json_response" => false,
               "max_tokens" => 1234,
               "model" => @test_model,
               "n" => 1,
               "receive_timeout" => 60000,
               "seed" => 123,
               "stream" => false,
               "stream_options" => %{"include_usage" => true},
               "temperature" => 0.0,
               "version" => 1,
               "json_schema" => nil,
               "module" => "Elixir.LangChain.ChatModels.ChatDeepSeek"
             }
    end
  end

  describe "inspect" do
    test "redacts the API key" do
      chain = ChatDeepSeek.new!()

      changeset = Ecto.Changeset.cast(chain, %{api_key: "1234567890"}, [:api_key])

      refute inspect(changeset) =~ "1234567890"
      assert inspect(changeset) =~ "**redacted**"
    end
  end

  describe "retry_on_fallback?/1" do
    test "returns true for retryable errors" do
      assert ChatDeepSeek.retry_on_fallback?(%LangChainError{type: "rate_limited"})
      assert ChatDeepSeek.retry_on_fallback?(%LangChainError{type: "rate_limit_exceeded"})
      assert ChatDeepSeek.retry_on_fallback?(%LangChainError{type: "timeout"})
      assert ChatDeepSeek.retry_on_fallback?(%LangChainError{type: "too_many_requests"})
    end

    test "returns false for non-retryable errors" do
      refute ChatDeepSeek.retry_on_fallback?(%LangChainError{type: "invalid_request"})
      refute ChatDeepSeek.retry_on_fallback?(%LangChainError{type: "authentication_error"})
    end
  end

  describe "restore_from_map/1" do
    test "restores model from serialized config" do
      original =
        ChatDeepSeek.new!(%{
          model: @test_model,
          temperature: 0.7,
          max_tokens: 1000
        })

      serialized = ChatDeepSeek.serialize_config(original)
      restored = ChatDeepSeek.restore_from_map(serialized)

      assert {:ok, %ChatDeepSeek{}} = restored
      assert elem(restored, 1).model == original.model
      assert elem(restored, 1).temperature == original.temperature
      assert elem(restored, 1).max_tokens == original.max_tokens
    end
  end

  describe "endpoint validation" do
    test "accepts valid https URLs in new/1" do
      assert {:ok, %ChatDeepSeek{}} =
               ChatDeepSeek.new(%{endpoint: "https://api.deepseek.com/chat/completions"})
    end

    test "accepts valid http URLs in new/1" do
      assert {:ok, %ChatDeepSeek{}} =
               ChatDeepSeek.new(%{endpoint: "http://localhost:8080/chat/completions"})
    end

    test "rejects invalid URLs without http/https in new/1" do
      assert {:error, %Ecto.Changeset{}} =
               ChatDeepSeek.new(%{endpoint: "ftp://example.com"})

      assert {:error, %Ecto.Changeset{}} =
               ChatDeepSeek.new(%{endpoint: "api.deepseek.com/chat/completions"})
    end
  end

  describe "api_key configuration" do
    test "uses provided api_key" do
      model = ChatDeepSeek.new!(%{api_key: "test_key_123"})
      # Test that api_key is set correctly through the config process
      assert model.api_key == "test_key_123"
    end

    test "accepts nil api_key" do
      model = ChatDeepSeek.new!(%{api_key: nil})
      assert model.api_key == nil
    end
  end

  describe "do_process_response/2 error scenarios" do
    setup do
      model = ChatDeepSeek.new!(%{"model" => @test_model})
      %{model: model}
    end

    test "handles non-map response", %{model: model} do
      assert {:error, %LangChainError{} = error} =
               ChatDeepSeek.do_process_response(model, "invalid_response")

      assert error.message == "Unexpected response"
    end
  end

  describe "tool call processing edge cases" do
    test "processes response with valid tool calls and JSON arguments" do
      model = ChatDeepSeek.new!(%{"model" => @test_model})

      response = %{
        "choices" => [
          %{
            "message" => %{
              "role" => "assistant",
              "content" => nil,
              "tool_calls" => [
                %{
                  "function" => %{
                    "arguments" => "{\"city\": \"Moab\", \"state\": \"UT\"}",
                    "name" => "get_weather"
                  },
                  "id" => "call_123",
                  "type" => "function"
                }
              ]
            },
            "finish_reason" => "tool_calls",
            "index" => 0
          }
        ]
      }

      assert [%Message{} = msg] = ChatDeepSeek.do_process_response(model, response)
      assert length(msg.tool_calls) == 1

      tool_call = hd(msg.tool_calls)
      assert tool_call.call_id == "call_123"
      assert tool_call.name == "get_weather"
      assert tool_call.arguments == %{"city" => "Moab", "state" => "UT"}
    end
  end

  describe "response metadata extraction" do
    test "processes response with minimal metadata" do
      model = ChatDeepSeek.new!(%{"model" => @test_model})

      response = %{
        "choices" => [
          %{
            "message" => %{"role" => "assistant", "content" => "Simple response"},
            "finish_reason" => "stop",
            "index" => 0
          }
        ]
      }

      assert [%Message{} = msg] = ChatDeepSeek.do_process_response(model, response)
      assert msg.role == :assistant
      assert msg.content == [ContentPart.text!("Simple response")]
      assert msg.status == :complete
      assert msg.metadata.logprobs == nil
    end

    test "processes response with usage metadata" do
      model = ChatDeepSeek.new!(%{"model" => @test_model})

      response = %{
        "choices" => [
          %{
            "message" => %{"role" => "assistant", "content" => "Response with usage"},
            "finish_reason" => "stop",
            "index" => 0
          }
        ],
        "usage" => %{
          "prompt_tokens" => 5,
          "completion_tokens" => 3,
          "total_tokens" => 8
        }
      }

      assert [%Message{} = msg] = ChatDeepSeek.do_process_response(model, response)
      assert msg.metadata.usage.input == 5
      assert msg.metadata.usage.output == 3
    end
  end
end
