defmodule ChatModels.ChatGoogleAITest do
  use LangChain.BaseCase
  use Mimic

  doctest LangChain.ChatModels.ChatGoogleAI
  alias LangChain.ChatModels.ChatGoogleAI
  alias LangChain.Message
  alias LangChain.Message.Citation
  alias LangChain.Message.ContentPart
  alias LangChain.Message.ToolCall
  alias LangChain.Message.ToolResult
  alias LangChain.TokenUsage
  alias LangChain.MessageDelta
  alias LangChain.Function
  alias LangChain.FunctionParam
  alias LangChain.LangChainError
  alias LangChain.ChatModels.ChatGoogleAI

  @test_model "gemini-2.5-flash"

  setup do
    {:ok, hello_world} =
      Function.new(%{
        name: "hello_world",
        description: "Give a hello world greeting.",
        function: fn _args, _context -> {:ok, "Hello world!"} end
      })

    model = ChatGoogleAI.new!(%{})

    %{model: model, hello_world: hello_world}
  end

  describe "new/1" do
    test "works with minimal attr" do
      assert {:ok, %ChatGoogleAI{} = google_ai} = ChatGoogleAI.new(%{"model" => @test_model})
      assert google_ai.model == @test_model
    end

    test "returns error when invalid" do
      assert {:error, changeset} = ChatGoogleAI.new(%{"model" => nil})
      refute changeset.valid?
      assert {"can't be blank", _} = changeset.errors[:model]
    end

    test "supports overriding the API endpoint" do
      override_url = "http://localhost:1234/"

      model =
        ChatGoogleAI.new!(%{
          endpoint: override_url
        })

      assert model.endpoint == override_url
    end

    test "supports overriding the API version" do
      api_version = "v1"

      model =
        ChatGoogleAI.new!(%{
          api_version: api_version
        })

      assert model.api_version == api_version
    end

    test "supports setting json_response and json_schema" do
      json_schema = %{
        "type" => "object",
        "properties" => %{
          "name" => %{"type" => "string"},
          "age" => %{"type" => "integer"}
        }
      }

      {:ok, google_ai} =
        ChatGoogleAI.new(%{
          "model" => @test_model,
          "json_response" => true,
          "json_schema" => json_schema
        })

      assert google_ai.json_response == true
      assert google_ai.json_schema == json_schema
    end
  end

  describe "for_api/3" do
    setup do
      params = %{
        "model" => @test_model,
        "temperature" => 1.0,
        "top_p" => 1.0,
        "top_k" => 1.0
      }

      {:ok, google_ai} = ChatGoogleAI.new(params)

      %{google_ai: google_ai, params: params}
    end

    test "generates a map for an API call", %{google_ai: google_ai} do
      data = ChatGoogleAI.for_api(google_ai, [], [])
      assert %{"contents" => [], "generationConfig" => config} = data
      assert %{"temperature" => 1.0, "topK" => 1.0, "topP" => 1.0} = config
    end

    test "generates a map containing user and assistant messages", %{google_ai: google_ai} do
      user_message = "Hello Assistant!"
      assistant_message = "Hello User!"

      data =
        ChatGoogleAI.for_api(
          google_ai,
          [
            Message.new_user!(user_message),
            Message.new_assistant!(assistant_message)
          ],
          []
        )

      assert %{"contents" => [msg1, msg2]} = data
      assert %{"role" => "user", "parts" => [%{"text" => ^user_message}]} = msg1
      assert %{"role" => "model", "parts" => [%{"text" => ^assistant_message}]} = msg2
    end

    test "generated a map containing response_mime_type and response_schema", %{params: params} do
      google_ai =
        params
        |> Map.merge(%{"json_response" => true, "json_schema" => %{"type" => "object"}})
        |> ChatGoogleAI.new!()

      data = ChatGoogleAI.for_api(google_ai, [], [])

      assert %{
               "generationConfig" => %{
                 "response_mime_type" => "application/json",
                 "response_schema" => %{"type" => "object"}
               }
             } = data
    end

    test "generates a map containing function and function call messages", %{google_ai: google_ai} do
      message = "Can you do an action for me?"
      arguments = %{"args" => "data"}
      function_result = %{"result" => "data"}

      data =
        ChatGoogleAI.for_api(
          google_ai,
          [
            Message.new_user!(message),
            Message.new_assistant!(%{
              tool_calls: [
                ToolCall.new!(%{
                  call_id: "call_123",
                  name: "userland_action",
                  arguments: Jason.encode!(arguments)
                })
              ]
            }),
            Message.new_tool_result!(%{
              tool_results: [
                ToolResult.new!(%{
                  tool_call_id: "call_123",
                  name: "userland_action",
                  content: Jason.encode!(function_result)
                })
              ]
            })
          ],
          []
        )

      assert %{"contents" => [msg1, msg2, msg3]} = data
      assert %{"role" => "user", "parts" => [%{"text" => ^message}]} = msg1
      assert %{"role" => "model", "parts" => [tool_call]} = msg2
      assert %{"role" => "model", "parts" => [tool_result]} = msg3

      assert %{
               "functionCall" => %{
                 "args" => ^arguments,
                 "name" => "userland_action"
               }
             } = tool_call

      assert %{
               "functionResponse" => %{
                 "name" => "userland_action",
                 "response" => %{
                   "name" => "userland_action",
                   "content" => ^function_result
                 }
               }
             } = tool_result
    end

    test "for_api includes thoughtSignature when present in ToolCall metadata" do
      tool_call =
        ToolCall.new!(%{
          call_id: "call_123",
          name: "test_function",
          arguments: %{"arg" => "value"},
          metadata: %{thought_signature: "sig_abc123"}
        })

      result = ChatGoogleAI.for_api(tool_call)

      assert result["thoughtSignature"] == "sig_abc123"
      assert result["functionCall"]["name"] == "test_function"
    end

    test "for_api excludes thoughtSignature when not in ToolCall metadata" do
      tool_call =
        ToolCall.new!(%{
          call_id: "call_123",
          name: "test_function",
          arguments: %{"arg" => "value"}
        })

      result = ChatGoogleAI.for_api(tool_call)

      refute Map.has_key?(result, "thoughtSignature")
      assert Map.has_key?(result, "functionCall")
    end

    test "generate a map containing text and inline image parts", %{google_ai: google_ai} do
      messages = [
        %LangChain.Message{
          content:
            "You are an expert at providing an image description for assistive technology and SEO benefits.",
          role: :system
        },
        %LangChain.Message{
          content: [
            %LangChain.Message.ContentPart{
              type: :text,
              content: "This is the text."
            },
            %LangChain.Message.ContentPart{
              type: :image,
              content: "/9j/4AAQSkz",
              options: [media: :jpg, detail: "low"]
            }
          ],
          role: :user
        }
      ]

      data = ChatGoogleAI.for_api(google_ai, messages, [])
      assert %{"contents" => [msg1]} = data

      assert %{
               "parts" => [
                 %{
                   "text" => "This is the text."
                 },
                 %{
                   "inline_data" => %{
                     "mime_type" => "image/jpeg",
                     "data" => "/9j/4AAQSkz"
                   }
                 }
               ]
             } = msg1
    end

    test "translates a Message with function results to the expected structure" do
      expected =
        %{
          "role" => "model",
          "parts" => [
            %{
              "functionResponse" => %{
                "name" => "find_theaters",
                "response" => %{
                  "name" => "find_theaters",
                  "content" => %{
                    "movie" => "Barbie",
                    "theaters" => [
                      %{
                        "name" => "AMC",
                        "address" => "2000 W El Camino Real"
                      }
                    ]
                  }
                }
              }
            }
          ]
        }

      message =
        Message.new_tool_result!(%{
          tool_results: [
            ToolResult.new!(%{
              name: "find_theaters",
              tool_call_id: "call-find_theaters",
              content:
                Jason.encode!(%{
                  "movie" => "Barbie",
                  "theaters" => [
                    %{
                      "name" => "AMC",
                      "address" => "2000 W El Camino Real"
                    }
                  ]
                })
            })
          ]
        })

      assert expected == ChatGoogleAI.for_api(message)
    end

    test "tool result creates expected map" do
      expected = %{
        "functionResponse" => %{
          "name" => "find_theaters",
          "response" => %{
            "name" => "find_theaters",
            "content" => %{"result" => "I don't know where the theaters are."}
          }
        }
      }

      tool_result =
        ToolResult.new!(%{
          name: "find_theaters",
          tool_call_id: "call-find_theaters",
          content: "I don't know where the theaters are."
        })

      assert expected == ChatGoogleAI.for_api(tool_result)
    end

    test "adds safety settings to the request if present" do
      settings = [
        %{"category" => "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold" => "BLOCK_ONLY_HIGH"}
      ]

      google_ai = ChatGoogleAI.new!(%{safety_settings: settings})
      data = ChatGoogleAI.for_api(google_ai, [], [])

      assert %{"safetySettings" => ^settings} = data
    end

    test "does not add safety settings to the request if list of settings is empty" do
      google_ai = ChatGoogleAI.new!(%{safety_settings: []})
      data = ChatGoogleAI.for_api(google_ai, [], [])
      refute Map.has_key?(data, "safetySettings")
    end

    test "adds system instruction to the request if present", %{google_ai: google_ai} do
      message = "You are a helpful assistant."
      data = ChatGoogleAI.for_api(google_ai, [Message.new_system!(message)], [])

      assert %{"system_instruction" => %{"parts" => [%{"text" => ^message}]}} = data
    end

    test "does not add system instruction if not present", %{google_ai: google_ai} do
      data = ChatGoogleAI.for_api(google_ai, [Message.new_user!("Hello!")], [])
      refute Map.has_key?(data, "system_instruction")
    end

    test "support file_url", %{google_ai: google_ai} do
      message =
        Message.new_user!([
          ContentPart.text!("User prompt"),
          ContentPart.file_url!("example.com/test.pdf", media: "application/pdf")
        ])

      data = ChatGoogleAI.for_api(google_ai, [message], [])

      assert %{
               "contents" => [
                 %{
                   "parts" => [
                     %{"text" => "User prompt"},
                     %{
                       "file_data" => %{
                         "file_uri" => "example.com/test.pdf",
                         "mime_type" => "application/pdf"
                       }
                     }
                   ],
                   "role" => "user"
                 }
               ]
             } = data
    end

    test "raises an error if more than one system message is present", %{google_ai: google_ai} do
      assert_raise LangChainError, "Google AI only supports a single System message", fn ->
        ChatGoogleAI.for_api(
          google_ai,
          [Message.new_system!("First instruction."), Message.new_system!("Second instruction.")],
          []
        )
      end
    end

    test "generates a map containing function declarations", %{
      google_ai: google_ai,
      hello_world: hello_world
    } do
      data = ChatGoogleAI.for_api(google_ai, [], [hello_world])

      assert %{"contents" => []} = data
      assert %{"tools" => [tool_call]} = data

      assert %{
               "functionDeclarations" => [
                 %{
                   "name" => "hello_world",
                   "description" => "Give a hello world greeting."
                 }
               ]
             } = tool_call
    end

    test "handles converting functions with parameters" do
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

      assert %{
               "description" => "Get the current weather in a given US location",
               "name" => "get_weather",
               "parameters" => %{
                 "properties" => %{
                   "city" => %{
                     "description" => "The city name, e.g. San Francisco",
                     "type" => "string"
                   },
                   "state" => %{
                     "description" => "The 2 letter US state abbreviation, e.g. CA, NY, UT",
                     "type" => "string"
                   }
                 },
                 "required" => ["city", "state"],
                 "type" => "object"
               }
             } == ChatGoogleAI.for_api(weather)
    end

    test "handles functions without parameters" do
      {:ok, function} =
        Function.new(%{
          name: "hello_world",
          description: "Give a hello world greeting.",
          parameters: [],
          function: fn _args, _context -> {:ok, "Hello User!"} end
        })

      assert %{
               "description" => "Give a hello world greeting.",
               "name" => "hello_world"
             } == ChatGoogleAI.for_api(function)
    end
  end

  describe "do_process_response/2" do
    test "handles receiving a message", %{model: model} do
      response = %{
        "candidates" => [
          %{
            "content" => %{"role" => "model", "parts" => [%{"text" => "Hello User!"}]},
            "finishReason" => "STOP",
            "index" => 0
          }
        ]
      }

      assert [%Message{} = struct] = ChatGoogleAI.do_process_response(model, response)
      assert struct.role == :assistant
      [%ContentPart{type: :text, content: "Hello User!"}] = struct.content
      assert struct.index == 0
      assert struct.status == :complete
    end

    test "handles receiving a message with an empty text part", %{model: model} do
      response = %{
        "candidates" => [
          %{
            "content" => %{"role" => "model", "parts" => [%{"text" => ""}]},
            "finishReason" => "STOP",
            "index" => 0
          }
        ]
      }

      assert [%Message{} = struct] = ChatGoogleAI.do_process_response(model, response)
      assert struct.content == []
    end

    test "error if receiving non-text content", %{model: model} do
      response = %{
        "candidates" => [
          %{
            "content" => %{"role" => "bad_role", "parts" => [%{"text" => "Hello user"}]},
            "finishReason" => "STOP",
            "index" => 0
          }
        ]
      }

      assert [{:error, %LangChainError{} = error}] =
               ChatGoogleAI.do_process_response(model, response)

      assert error.type == "changeset"
      assert error.message == "role: is invalid"
    end

    test "handles receiving function calls", %{model: model} do
      data = Jason.encode!(%{"value" => 123})

      response = %{
        "candidates" => [
          %{
            "content" => %{
              "role" => "model",
              "parts" => [%{"functionCall" => %{"args" => data, "name" => "hello_world"}}]
            },
            "finishReason" => "STOP",
            "index" => 0
          }
        ]
      }

      assert [%Message{} = struct] = ChatGoogleAI.do_process_response(model, response)
      assert struct.role == :assistant
      assert struct.index == 0
      [call] = struct.tool_calls
      assert call.name == "hello_world"
      assert call.arguments == %{"value" => 123}
    end

    test "handles function calls with thoughtSignature (Gemini 3)", %{model: model} do
      response = %{
        "candidates" => [
          %{
            "content" => %{
              "role" => "model",
              "parts" => [
                %{
                  "functionCall" => %{"args" => %{"key" => "value"}, "name" => "my_func"},
                  "thoughtSignature" => "gemini3_thought_sig_xyz"
                }
              ]
            },
            "finishReason" => "STOP",
            "index" => 0
          }
        ]
      }

      assert [%Message{} = msg] = ChatGoogleAI.do_process_response(model, response)
      assert [%ToolCall{} = call] = msg.tool_calls
      assert call.metadata.thought_signature == "gemini3_thought_sig_xyz"
      assert call.name == "my_func"
    end

    test "handles function calls without thoughtSignature", %{model: model} do
      response = %{
        "candidates" => [
          %{
            "content" => %{
              "role" => "model",
              "parts" => [%{"functionCall" => %{"args" => %{}, "name" => "my_func"}}]
            },
            "finishReason" => "STOP",
            "index" => 0
          }
        ]
      }

      assert [%Message{} = msg] = ChatGoogleAI.do_process_response(model, response)
      assert [%ToolCall{} = call] = msg.tool_calls
      assert call.metadata == nil
    end

    test "handles no parts in content", %{model: model} do
      response = %{
        "candidates" => [
          %{
            "content" => %{
              "role" => "model"
            },
            "finishReason" => "STOP",
            "index" => 0
          }
        ]
      }

      assert [%Message{} = struct] = ChatGoogleAI.do_process_response(model, response)
      assert struct.role == :assistant
      assert struct.content == []
      assert struct.status == :complete
    end

    test "handles receiving MessageDeltas as well", %{model: model} do
      response = %{
        "candidates" => [
          %{
            "content" => %{
              "role" => "model",
              "parts" => [%{"text" => "This is the first part of a mes"}]
            },
            "finishReason" => "STOP",
            "index" => 0
          }
        ]
      }

      assert [%MessageDelta{} = struct] =
               ChatGoogleAI.do_process_response(model, response, MessageDelta)

      assert struct.role == :assistant
      assert struct.content == ContentPart.text!("This is the first part of a mes")
      assert struct.index == 0
      assert struct.status == :complete
    end

    test "handles receiving a MessageDelta with an empty text part", %{model: model} do
      response = %{
        "candidates" => [
          %{
            "content" => %{
              "role" => "model",
              "parts" => [%{"text" => ""}]
            },
            "finishReason" => "STOP",
            "index" => 0
          }
        ]
      }

      assert [%MessageDelta{} = struct] =
               ChatGoogleAI.do_process_response(model, response, MessageDelta)

      assert struct.content == ContentPart.text!("")
    end

    test "handles receiving a MessageDelta with no parts in content", %{model: model} do
      response = %{
        "candidates" => [
          %{
            "content" => %{
              "role" => "model"
            },
            "finishReason" => "STOP",
            "index" => 0
          }
        ]
      }

      assert [%MessageDelta{} = struct] =
               ChatGoogleAI.do_process_response(model, response, MessageDelta)

      assert struct.role == :assistant
      assert struct.content == nil
      assert struct.status == :complete
    end

    test "handles API error messages", %{model: model} do
      response = %{
        "error" => %{
          "code" => 400,
          "message" => "Invalid request",
          "status" => "INVALID_ARGUMENT"
        }
      }

      assert {:error, %LangChainError{} = error} =
               ChatGoogleAI.do_process_response(model, response)

      assert error.type == "invalid_argument"
      assert error.message == "Invalid request"
    end

    test "handles Jason.DecodeError", %{model: model} do
      response = {:error, %Jason.DecodeError{}}

      assert {:error, %LangChainError{} = error} =
               ChatGoogleAI.do_process_response(model, response)

      assert error.type == "invalid_json"
      assert "Received invalid JSON:" <> _ = error.message
    end

    test "handles unexpected response with error", %{model: model} do
      response = %{}

      assert {:error, %LangChainError{} = error} =
               ChatGoogleAI.do_process_response(model, response)

      assert error.type == "unexpected_response"
      assert error.message == "Unexpected response"
    end

    test "handles receiving a message with token usage", %{model: model} do
      response = %{
        "candidates" => [
          %{
            "content" => %{"role" => "model", "parts" => [%{"text" => "Hello User!"}]},
            "finishReason" => "STOP",
            "index" => 0
          }
        ],
        "usageMetadata" => %{
          "promptTokenCount" => 10,
          "candidatesTokenCount" => 5,
          "totalTokenCount" => 15
        }
      }

      assert [%Message{} = struct] = ChatGoogleAI.do_process_response(model, response)
      assert struct.role == :assistant
      [%ContentPart{type: :text, content: "Hello User!"}] = struct.content
      assert struct.index == 0
      assert struct.status == :complete

      # Verify that token usage is properly included in metadata
      assert %TokenUsage{} = struct.metadata.usage
      assert struct.metadata.usage.input == 10
      assert struct.metadata.usage.output == 5

      assert struct.metadata.usage.raw == %{
               "promptTokenCount" => 10,
               "candidatesTokenCount" => 5,
               "totalTokenCount" => 15
             }
    end

    test "handles receiving MessageDelta with token usage", %{model: model} do
      response = %{
        "candidates" => [
          %{
            "content" => %{
              "role" => "model",
              "parts" => [%{"text" => "This is a partial message"}]
            },
            "finishReason" => "STOP",
            "index" => 0
          }
        ],
        "usageMetadata" => %{
          "promptTokenCount" => 8,
          "candidatesTokenCount" => 3,
          "totalTokenCount" => 11
        }
      }

      assert [%MessageDelta{} = struct] =
               ChatGoogleAI.do_process_response(model, response, MessageDelta)

      assert struct.role == :assistant
      assert struct.content == ContentPart.text!("This is a partial message")
      assert struct.index == 0
      assert struct.status == :complete

      # Verify that token usage is properly included in metadata
      assert %TokenUsage{} = struct.metadata.usage
      assert struct.metadata.usage.input == 8
      assert struct.metadata.usage.output == 3

      assert struct.metadata.usage.raw == %{
               "promptTokenCount" => 8,
               "candidatesTokenCount" => 3,
               "totalTokenCount" => 11
             }
    end
  end

  describe "filter_parts_for_types/2" do
    test "returns a single functionCall type" do
      parts = [
        %{"text" => "I think I'll call this function."},
        %{
          "functionCall" => %{
            "args" => %{"args" => "data"},
            "name" => "userland_action"
          }
        }
      ]

      assert [%{"text" => _}] = ChatGoogleAI.filter_parts_for_types(parts, ["text"])

      assert [%{"functionCall" => _}] =
               ChatGoogleAI.filter_parts_for_types(parts, ["functionCall"])
    end

    test "returns a set of types" do
      parts = [
        %{"text" => "I think I'll call this function."},
        %{
          "functionCall" => %{
            "args" => %{"args" => "data"},
            "name" => "userland_action"
          }
        }
      ]

      assert parts == ChatGoogleAI.filter_parts_for_types(parts, ["text", "functionCall"])
    end
  end

  describe "filter_text_parts/1" do
    test "returns only text parts that are not nil or empty" do
      parts = [
        %{"text" => "I have text"},
        %{"text" => nil},
        %{"text" => ""},
        %{"text" => "I have more text"}
      ]

      assert ChatGoogleAI.filter_text_parts(parts) == [
               %{"text" => "I have text"},
               %{"text" => "I have more text"}
             ]
    end
  end

  describe "get_message_contents/1" do
    test "returns basic text as a ContentPart" do
      message = Message.new_user!("Howdy!")

      result = ChatGoogleAI.get_message_contents(message)

      assert result == [%{"text" => "Howdy!"}]
    end

    test "supports a list of ContentParts" do
      message =
        Message.new_user!([
          ContentPart.new!(%{type: :text, content: "Hello!"}),
          ContentPart.new!(%{type: :text, content: "What's up?"})
        ])

      result = ChatGoogleAI.get_message_contents(message)

      assert result == [
               %{"text" => "Hello!"},
               %{"text" => "What's up?"}
             ]
    end
  end

  describe "serialize_config/2" do
    test "does not include the API key or callbacks" do
      model = ChatGoogleAI.new!(%{model: @test_model})
      result = ChatGoogleAI.serialize_config(model)
      assert result["version"] == 1
      refute Map.has_key?(result, "api_key")
      refute Map.has_key?(result, "callbacks")
    end

    test "creates expected map" do
      model =
        ChatGoogleAI.new!(%{
          model: "gemini-1.5-flash",
          temperature: 0,
          frequency_penalty: 0.5,
          seed: 123,
          max_tokens: 1234,
          stream_options: %{include_usage: true}
        })

      result = ChatGoogleAI.serialize_config(model)

      assert result == %{
               "endpoint" => "https://generativelanguage.googleapis.com",
               "model" => "gemini-1.5-flash",
               "module" => "Elixir.LangChain.ChatModels.ChatGoogleAI",
               "receive_timeout" => 60000,
               "thinking_config" => nil,
               "stream" => false,
               "temperature" => 0.0,
               "version" => 1,
               "api_version" => "v1beta",
               "top_k" => 1.0,
               "top_p" => 1.0,
               "safety_settings" => [],
               "json_response" => false,
               "json_schema" => nil
             }
    end
  end

  describe "build_url/1" do
    test "builds the correct URL for the request" do
      llm = ChatGoogleAI.new!(%{model: "gemini-1.5-flash", stream: false})
      result = ChatGoogleAI.build_url(llm)

      assert result =~
               "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key="
    end
  end

  describe "live tests and token usage information" do
    @tag live_call: true, live_google_ai: true
    test "basic non-streamed response works and fires token usage callback" do
      handlers = %{
        on_llm_token_usage: fn usage ->
          send(self(), {:fired_token_usage, usage})
        end
      }

      %ChatGoogleAI{} =
        chat =
        ChatGoogleAI.new!(%{
          temperature: 0,
          stream: false
        })

      chat = %ChatGoogleAI{chat | callbacks: [handlers]}

      {:ok, result} =
        ChatGoogleAI.call(chat, [
          Message.new_user!("Return the response 'Colorful Threads'.")
        ])

      # returns a list of MessageDeltas. A list of a list because it's "n" choices.
      assert [
               %Message{
                 content: [
                   %Message.ContentPart{
                     type: :text,
                     content: "Colorful Threads",
                     options: []
                   }
                 ],
                 status: :complete,
                 role: :assistant,
                 index: 0,
                 tool_calls: [],
                 metadata: %{
                   usage: %TokenUsage{
                     input: 8,
                     output: 2
                   }
                 }
               }
             ] = result

      assert_received {:fired_token_usage, usage}
      assert %TokenUsage{input: 8, output: 2} = usage
    end

    @tag live_call: true, live_google_ai: true
    test "streamed response works and fires token usage callback" do
      handlers = %{
        on_llm_token_usage: fn usage ->
          # NOTE: The token usage fires for every received delta. That's an
          # oddity with Google.
          #
          # IO.inspect usage, label: "USAGE DATA
          send(self(), {:fired_token_usage, usage})
        end
      }

      %ChatGoogleAI{} =
        chat =
        ChatGoogleAI.new!(%{
          temperature: 0,
          stream: true
        })

      chat = %ChatGoogleAI{chat | callbacks: [handlers]}

      {:ok, result} =
        ChatGoogleAI.call(chat, [
          Message.new_user!("Return the response 'Colorful Threads'.")
        ])

      assert [
               [
                 %MessageDelta{
                   content: "Colorful Threads",
                   status: :complete,
                   index: 0,
                   role: :assistant,
                   tool_calls: nil,
                   metadata: %{
                     usage: %TokenUsage{
                       input: 8,
                       output: 2
                     }
                   }
                 }
               ]
             ] = result

      assert_received {:fired_token_usage, usage}
      assert %TokenUsage{input: 8, output: 2} = usage
    end
  end

  describe "grounding citations" do
    test "parses groundingMetadata into citations on ContentParts", %{model: model} do
      response = %{
        "candidates" => [
          %{
            "content" => %{
              "role" => "model",
              "parts" => [%{"text" => "Spain won Euro 2024, defeating England 2-1 in the final."}]
            },
            "finishReason" => "STOP",
            "index" => 0,
            "groundingMetadata" => %{
              "webSearchQueries" => ["UEFA Euro 2024 winner"],
              "groundingChunks" => [
                %{
                  "web" => %{
                    "uri" => "https://example.com/euro2024",
                    "title" => "UEFA Euro 2024 Results"
                  }
                }
              ],
              "groundingSupports" => [
                %{
                  "segment" => %{
                    "startIndex" => 0,
                    "endIndex" => 56,
                    "text" => "Spain won Euro 2024, defeating England 2-1 in the final."
                  },
                  "groundingChunkIndices" => [0],
                  "confidenceScores" => [0.95]
                }
              ]
            }
          }
        ]
      }

      assert [%Message{} = msg] = ChatGoogleAI.do_process_response(model, response)
      assert msg.role == :assistant
      assert [%ContentPart{} = part] = msg.content
      assert part.content == "Spain won Euro 2024, defeating England 2-1 in the final."

      assert [%Citation{} = citation] = part.citations
      assert citation.cited_text == "Spain won Euro 2024, defeating England 2-1 in the final."
      assert citation.start_index == 0
      assert citation.end_index == 56
      assert citation.confidence == 0.95
      assert citation.source.type == :web
      assert citation.source.title == "UEFA Euro 2024 Results"
      assert citation.source.url == "https://example.com/euro2024"
      assert citation.metadata["provider_type"] == "grounding_support"
      assert citation.metadata["chunk_index"] == 0
    end

    test "handles multiple groundingSupports mapping to different partIndex values", %{
      model: model
    } do
      response = %{
        "candidates" => [
          %{
            "content" => %{
              "role" => "model",
              "parts" => [
                %{"text" => "First part about Spain."},
                %{"text" => "Second part about England."}
              ]
            },
            "finishReason" => "STOP",
            "index" => 0,
            "groundingMetadata" => %{
              "groundingChunks" => [
                %{"web" => %{"uri" => "https://example.com/spain", "title" => "Spain Info"}},
                %{"web" => %{"uri" => "https://example.com/england", "title" => "England Info"}}
              ],
              "groundingSupports" => [
                %{
                  "segment" => %{
                    "partIndex" => 0,
                    "startIndex" => 0,
                    "endIndex" => 22,
                    "text" => "First part about Spain."
                  },
                  "groundingChunkIndices" => [0]
                },
                %{
                  "segment" => %{
                    "partIndex" => 1,
                    "startIndex" => 0,
                    "endIndex" => 26,
                    "text" => "Second part about England."
                  },
                  "groundingChunkIndices" => [1]
                }
              ]
            }
          }
        ]
      }

      assert [%Message{} = msg] = ChatGoogleAI.do_process_response(model, response)
      assert [part0, part1] = msg.content

      assert [%Citation{} = c0] = part0.citations
      assert c0.source.title == "Spain Info"
      assert c0.source.url == "https://example.com/spain"

      assert [%Citation{} = c1] = part1.citations
      assert c1.source.title == "England Info"
      assert c1.source.url == "https://example.com/england"
    end

    test "denormalizes support x chunk pairs into individual citations", %{model: model} do
      response = %{
        "candidates" => [
          %{
            "content" => %{
              "role" => "model",
              "parts" => [%{"text" => "Spain won Euro 2024."}]
            },
            "finishReason" => "STOP",
            "index" => 0,
            "groundingMetadata" => %{
              "groundingChunks" => [
                %{"web" => %{"uri" => "https://example.com/a", "title" => "Source A"}},
                %{"web" => %{"uri" => "https://example.com/b", "title" => "Source B"}}
              ],
              "groundingSupports" => [
                %{
                  "segment" => %{
                    "startIndex" => 0,
                    "endIndex" => 20,
                    "text" => "Spain won Euro 2024."
                  },
                  "groundingChunkIndices" => [0, 1],
                  "confidenceScores" => [0.9, 0.8]
                }
              ]
            }
          }
        ]
      }

      assert [%Message{} = msg] = ChatGoogleAI.do_process_response(model, response)
      assert [part] = msg.content

      assert [%Citation{} = c0, %Citation{} = c1] = part.citations
      assert c0.source.title == "Source A"
      assert c0.confidence == 0.9
      assert c0.metadata["chunk_index"] == 0

      assert c1.source.title == "Source B"
      assert c1.confidence == 0.8
      assert c1.metadata["chunk_index"] == 1

      # Both share the same segment text
      assert c0.cited_text == "Spain won Euro 2024."
      assert c1.cited_text == "Spain won Euro 2024."
      assert c0.start_index == 0
      assert c1.start_index == 0
    end

    test "preserves confidence scores", %{model: model} do
      response = %{
        "candidates" => [
          %{
            "content" => %{
              "role" => "model",
              "parts" => [%{"text" => "Test content."}]
            },
            "finishReason" => "STOP",
            "index" => 0,
            "groundingMetadata" => %{
              "groundingChunks" => [
                %{"web" => %{"uri" => "https://example.com", "title" => "Example"}}
              ],
              "groundingSupports" => [
                %{
                  "segment" => %{"startIndex" => 0, "endIndex" => 13, "text" => "Test content."},
                  "groundingChunkIndices" => [0],
                  "confidenceScores" => [0.42]
                }
              ]
            }
          }
        ]
      }

      assert [%Message{} = msg] = ChatGoogleAI.do_process_response(model, response)
      assert [%Citation{confidence: 0.42}] = hd(msg.content).citations
    end

    test "handles missing groundingMetadata gracefully", %{model: model} do
      response = %{
        "candidates" => [
          %{
            "content" => %{
              "role" => "model",
              "parts" => [%{"text" => "Simple response."}]
            },
            "finishReason" => "STOP",
            "index" => 0
          }
        ]
      }

      assert [%Message{} = msg] = ChatGoogleAI.do_process_response(model, response)
      assert [part] = msg.content
      assert part.citations == []
      assert msg.metadata == nil
    end

    test "handles groundingMetadata with no supports", %{model: model} do
      response = %{
        "candidates" => [
          %{
            "content" => %{
              "role" => "model",
              "parts" => [%{"text" => "Response text."}]
            },
            "finishReason" => "STOP",
            "index" => 0,
            "groundingMetadata" => %{
              "webSearchQueries" => ["test query"],
              "groundingChunks" => [
                %{"web" => %{"uri" => "https://example.com", "title" => "Example"}}
              ],
              "groundingSupports" => []
            }
          }
        ]
      }

      assert [%Message{} = msg] = ChatGoogleAI.do_process_response(model, response)
      assert [part] = msg.content
      assert part.citations == []
    end

    test "preserves raw grounding_metadata in message metadata", %{model: model} do
      grounding_metadata = %{
        "webSearchQueries" => ["test query"],
        "groundingChunks" => [
          %{"web" => %{"uri" => "https://example.com", "title" => "Example"}}
        ],
        "groundingSupports" => [
          %{
            "segment" => %{"startIndex" => 0, "endIndex" => 5, "text" => "Test."},
            "groundingChunkIndices" => [0]
          }
        ]
      }

      response = %{
        "candidates" => [
          %{
            "content" => %{
              "role" => "model",
              "parts" => [%{"text" => "Test."}]
            },
            "finishReason" => "STOP",
            "index" => 0,
            "groundingMetadata" => grounding_metadata
          }
        ]
      }

      assert [%Message{} = msg] = ChatGoogleAI.do_process_response(model, response)
      assert msg.metadata["grounding_metadata"] == grounding_metadata
    end

    test "preserves searchEntryPoint in message metadata", %{model: model} do
      response = %{
        "candidates" => [
          %{
            "content" => %{
              "role" => "model",
              "parts" => [%{"text" => "Result."}]
            },
            "finishReason" => "STOP",
            "index" => 0,
            "groundingMetadata" => %{
              "searchEntryPoint" => %{
                "renderedContent" => "<div>Search Widget</div>"
              },
              "groundingChunks" => [],
              "groundingSupports" => []
            }
          }
        ]
      }

      assert [%Message{} = msg] = ChatGoogleAI.do_process_response(model, response)

      assert msg.metadata["search_entry_point"] == %{
               "renderedContent" => "<div>Search Widget</div>"
             }
    end

    test "handles retrievedContext chunks", %{model: model} do
      response = %{
        "candidates" => [
          %{
            "content" => %{
              "role" => "model",
              "parts" => [%{"text" => "Retrieved info."}]
            },
            "finishReason" => "STOP",
            "index" => 0,
            "groundingMetadata" => %{
              "groundingChunks" => [
                %{
                  "retrievedContext" => %{
                    "uri" => "https://storage.example.com/doc.pdf",
                    "title" => "Internal Doc"
                  }
                }
              ],
              "groundingSupports" => [
                %{
                  "segment" => %{"startIndex" => 0, "endIndex" => 15, "text" => "Retrieved info."},
                  "groundingChunkIndices" => [0]
                }
              ]
            }
          }
        ]
      }

      assert [%Message{} = msg] = ChatGoogleAI.do_process_response(model, response)
      assert [%Citation{} = citation] = hd(msg.content).citations
      assert citation.source.type == :document
      assert citation.source.title == "Internal Doc"
      assert citation.source.url == "https://storage.example.com/doc.pdf"
      assert citation.source.metadata["retrieved_context"] == true
    end

    test "handles maps chunks", %{model: model} do
      response = %{
        "candidates" => [
          %{
            "content" => %{
              "role" => "model",
              "parts" => [%{"text" => "The Eiffel Tower is in Paris."}]
            },
            "finishReason" => "STOP",
            "index" => 0,
            "groundingMetadata" => %{
              "groundingChunks" => [
                %{
                  "maps" => %{
                    "uri" => "https://maps.google.com/eiffel-tower",
                    "title" => "Eiffel Tower"
                  }
                }
              ],
              "groundingSupports" => [
                %{
                  "segment" => %{
                    "startIndex" => 0,
                    "endIndex" => 28,
                    "text" => "The Eiffel Tower is in Paris."
                  },
                  "groundingChunkIndices" => [0]
                }
              ]
            }
          }
        ]
      }

      assert [%Message{} = msg] = ChatGoogleAI.do_process_response(model, response)
      assert [%Citation{} = citation] = hd(msg.content).citations
      assert citation.source.type == :place
      assert citation.source.title == "Eiffel Tower"
      assert citation.source.url == "https://maps.google.com/eiffel-tower"
      assert citation.source.metadata["maps"] == true
    end

    test "handles streaming delta with grounding metadata", %{model: model} do
      response = %{
        "candidates" => [
          %{
            "content" => %{
              "role" => "model",
              "parts" => [%{"text" => "Grounded response."}]
            },
            "finishReason" => "STOP",
            "index" => 0,
            "groundingMetadata" => %{
              "groundingChunks" => [
                %{"web" => %{"uri" => "https://example.com", "title" => "Example"}}
              ],
              "groundingSupports" => [
                %{
                  "segment" => %{
                    "startIndex" => 0,
                    "endIndex" => 19,
                    "text" => "Grounded response."
                  },
                  "groundingChunkIndices" => [0],
                  "confidenceScores" => [0.88]
                }
              ]
            }
          }
        ]
      }

      assert [%MessageDelta{} = delta] =
               ChatGoogleAI.do_process_response(model, response, MessageDelta)

      assert %ContentPart{} = part = delta.content
      assert part.content == "Grounded response."
      assert [%Citation{} = citation] = part.citations
      assert citation.source.type == :web
      assert citation.source.url == "https://example.com"
      assert citation.confidence == 0.88
    end

    test "streaming delta without grounding metadata has no citations", %{model: model} do
      response = %{
        "candidates" => [
          %{
            "content" => %{
              "role" => "model",
              "parts" => [%{"text" => "Simple response."}]
            },
            "finishReason" => "STOP",
            "index" => 0
          }
        ]
      }

      assert [%MessageDelta{} = delta] =
               ChatGoogleAI.do_process_response(model, response, MessageDelta)

      assert delta.content.citations == []
    end

    test "uses Message.all_citations/1 to collect citations across parts", %{model: model} do
      response = %{
        "candidates" => [
          %{
            "content" => %{
              "role" => "model",
              "parts" => [
                %{"text" => "Part one."},
                %{"text" => "Part two."}
              ]
            },
            "finishReason" => "STOP",
            "index" => 0,
            "groundingMetadata" => %{
              "groundingChunks" => [
                %{"web" => %{"uri" => "https://a.com", "title" => "A"}},
                %{"web" => %{"uri" => "https://b.com", "title" => "B"}}
              ],
              "groundingSupports" => [
                %{
                  "segment" => %{
                    "partIndex" => 0,
                    "startIndex" => 0,
                    "endIndex" => 9,
                    "text" => "Part one."
                  },
                  "groundingChunkIndices" => [0]
                },
                %{
                  "segment" => %{
                    "partIndex" => 1,
                    "startIndex" => 0,
                    "endIndex" => 9,
                    "text" => "Part two."
                  },
                  "groundingChunkIndices" => [1]
                }
              ]
            }
          }
        ]
      }

      assert [%Message{} = msg] = ChatGoogleAI.do_process_response(model, response)

      all_citations = Message.all_citations(msg)
      assert length(all_citations) == 2
      assert Enum.any?(all_citations, &(&1.source.title == "A"))
      assert Enum.any?(all_citations, &(&1.source.title == "B"))

      urls = Citation.source_urls(all_citations)
      assert "https://a.com" in urls
      assert "https://b.com" in urls
    end

    test "handles confidence scores absent from response", %{model: model} do
      response = %{
        "candidates" => [
          %{
            "content" => %{
              "role" => "model",
              "parts" => [%{"text" => "No confidence."}]
            },
            "finishReason" => "STOP",
            "index" => 0,
            "groundingMetadata" => %{
              "groundingChunks" => [
                %{"web" => %{"uri" => "https://example.com", "title" => "Example"}}
              ],
              "groundingSupports" => [
                %{
                  "segment" => %{
                    "startIndex" => 0,
                    "endIndex" => 14,
                    "text" => "No confidence."
                  },
                  "groundingChunkIndices" => [0]
                }
              ]
            }
          }
        ]
      }

      assert [%Message{} = msg] = ChatGoogleAI.do_process_response(model, response)
      assert [%Citation{confidence: nil}] = hd(msg.content).citations
    end
  end

  describe "google_search native tool" do
    @tag live_call: true, live_google_ai: true
    test "should include grounding metadata and citations in response" do
      alias LangChain.Chains.LLMChain
      alias LangChain.Message
      alias LangChain.NativeTool

      model = ChatGoogleAI.new!(%{temperature: 0, stream: false, model: "gemini-2.0-flash"})

      {:ok, updated_chain} =
        %{llm: model, verbose: false, stream: false}
        |> LLMChain.new!()
        |> LLMChain.add_message(Message.new_user!("What is the current Google stock price?"))
        |> LLMChain.add_tools(NativeTool.new!(%{name: "google_search", configuration: %{}}))
        |> LLMChain.run()

      assert %Message{} = msg = updated_chain.last_message
      assert msg.role == :assistant
      # Raw grounding metadata is preserved under the new key
      assert Map.has_key?(msg.metadata, "grounding_metadata")

      # Citations should be parsed onto ContentParts
      all_citations = Message.all_citations(msg)
      assert length(all_citations) > 0

      # All citations should be web type from Google Search
      Enum.each(all_citations, fn citation ->
        assert citation.source.type == :web
        assert citation.source.url != nil
      end)
    end
  end

  describe "calculator with GoogleAI model" do
    @tag live_call: true, live_google_ai: true
    test "should work" do
      alias LangChain.Chains.LLMChain
      alias LangChain.Tools.Calculator

      test_pid = self()

      handlers = %{
        on_llm_new_message: fn %LLMChain{} = _chain, %Message{} = message ->
          send(test_pid, {:callback_msg, message})
        end,
        on_tool_response_created: fn _chain, %Message{} = tool_message ->
          send(test_pid, {:callback_tool_msg, tool_message})
        end
      }

      model = ChatGoogleAI.new!(%{temperature: 0, stream: false})

      {:ok, updated_chain} =
        LLMChain.new!(%{
          llm: model,
          verbose: false,
          stream: false
        })
        |> LLMChain.add_message(
          Message.new_user!("Answer the following math question: What is 100 + 300 - 200?")
        )
        |> LLMChain.add_tools(Calculator.new!())
        |> LLMChain.add_callback(handlers)
        |> LLMChain.run(mode: :while_needs_response)

      assert %Message{} = updated_chain.last_message
      assert updated_chain.last_message.role == :assistant

      answer = LangChain.Utils.ChainResult.to_string!(updated_chain)
      assert answer =~ "is 200"

      # assert received multiple messages as callbacks
      assert_received {:callback_msg, message}
      assert message.role == :assistant

      assert [%ToolCall{name: "calculator", arguments: %{"expression" => _}}] =
               message.tool_calls

      # the function result message
      assert_received {:callback_tool_msg, message}
      assert message.role == :tool
      assert [%ToolResult{content: answer}] = message.tool_results
      assert ContentPart.content_to_string(answer) == "200"

      assert_received {:callback_msg, message}
      assert message.role == :assistant
    end
  end

  @tag live_call: true, live_google_ai: true
  test "image classification with Google AI model" do
    alias LangChain.Chains.LLMChain
    alias LangChain.Message
    alias LangChain.Message.ContentPart
    alias LangChain.Utils.ChainResult

    model = ChatGoogleAI.new!(%{temperature: 0, stream: false, model: "gemini-1.5-flash"})

    image_data =
      File.read!("test/support/images/barn_owl.jpg")
      |> Base.encode64()

    {:ok, updated_chain} =
      %{llm: model, verbose: false, stream: false}
      |> LLMChain.new!()
      |> LLMChain.add_message(
        Message.new_user!([
          ContentPart.text!("Please describe the image."),
          ContentPart.image!(image_data, media: :jpg)
        ])
      )
      |> LLMChain.run()

    {:ok, string} = ChainResult.to_string(updated_chain)
    assert string =~ "owl"
  end

  describe "inspect" do
    test "redacts the API key" do
      chain = ChatGoogleAI.new!()

      changeset = Ecto.Changeset.cast(chain, %{api_key: "1234567890"}, [:api_key])

      refute inspect(changeset) =~ "1234567890"
      assert inspect(changeset) =~ "**redacted**"
    end
  end

  describe "API error handling" do
    test "non-streaming 404 returns structured error with type" do
      error_body = %{
        "error" => %{
          "code" => 404,
          "message" =>
            "models/gemini-1.5-flash is not found for API version v1beta, or is not supported for generateContent. Call ListModels to see the list of available models and their supported methods.",
          "status" => "NOT_FOUND"
        }
      }

      expect(Req, :post, fn _req ->
        {:ok,
         %Req.Response{
           status: 404,
           headers: %{},
           body: error_body
         }}
      end)

      model = ChatGoogleAI.new!(%{stream: false, model: "gemini-1.5-flash"})

      assert {:error, %LangChainError{} = error} =
               ChatGoogleAI.call(model, [Message.new_user!("Hello")])

      assert error.type == "not_found"
      assert error.message =~ "models/gemini-1.5-flash is not found"
      assert %Req.Response{status: 404} = error.original

      # Should NOT be retried
      refute ChatGoogleAI.retry_on_fallback?(error)
    end

    test "non-streaming 429 quota exceeded returns structured error with type" do
      error_body = %{
        "error" => %{
          "code" => 429,
          "message" =>
            "You exceeded your current quota, please check your plan and billing details.",
          "status" => "RESOURCE_EXHAUSTED"
        }
      }

      expect(Req, :post, fn _req ->
        {:ok,
         %Req.Response{
           status: 429,
           headers: %{},
           body: error_body
         }}
      end)

      model = ChatGoogleAI.new!(%{stream: false, model: "gemini-2.5-pro"})

      assert {:error, %LangChainError{} = error} =
               ChatGoogleAI.call(model, [Message.new_user!("Hello")])

      assert error.type == "resource_exhausted"
      assert error.message =~ "You exceeded your current quota"

      # Should NOT be retried - quota exceeded is not a transient error
      refute ChatGoogleAI.retry_on_fallback?(error)
    end

    test "streaming 404 returns structured error" do
      # When streaming, the handle_stream_fn processes error chunks and puts
      # the result of do_process_response into the body. For a 404, the body
      # would be {:error, %LangChainError{}} from do_process_response.
      expect(Req, :post, fn _req, _opts ->
        {:ok,
         %Req.Response{
           status: 404,
           headers: %{},
           body:
             {:error,
              LangChainError.exception(
                type: "not_found",
                message:
                  "models/gemini-1.5-flash is not found for API version v1beta.",
                original: %{
                  "error" => %{
                    "code" => 404,
                    "message" =>
                      "models/gemini-1.5-flash is not found for API version v1beta.",
                    "status" => "NOT_FOUND"
                  }
                }
              )}
         }}
      end)

      model = ChatGoogleAI.new!(%{stream: true, model: "gemini-1.5-flash"})

      assert {:error, %LangChainError{} = error} =
               ChatGoogleAI.call(model, [Message.new_user!("Hello")])

      assert error.type == "not_found"
      assert error.message =~ "not found"
    end

    test "streaming 429 quota exceeded returns structured error" do
      # Simulates what happens after handle_stream_fn processes the 429 error
      expect(Req, :post, fn _req, _opts ->
        {:ok,
         %Req.Response{
           status: 429,
           headers: %{},
           body:
             {:error,
              LangChainError.exception(
                type: "resource_exhausted",
                message:
                  "You exceeded your current quota, please check your plan and billing details.",
                original: %{
                  "error" => %{
                    "code" => 429,
                    "message" =>
                      "You exceeded your current quota, please check your plan and billing details.",
                    "status" => "RESOURCE_EXHAUSTED"
                  }
                }
              )}
         }}
      end)

      model = ChatGoogleAI.new!(%{stream: true, model: "gemini-2.5-pro"})

      assert {:error, %LangChainError{} = error} =
               ChatGoogleAI.call(model, [Message.new_user!("Hello")])

      assert error.type == "resource_exhausted"
      assert error.message =~ "You exceeded your current quota"

      # Should NOT be retried
      refute ChatGoogleAI.retry_on_fallback?(error)
    end

    test "streaming error buffer handles chunked error JSON" do
      # Test the utils.ex buffering directly - when error JSON spans multiple chunks,
      # the handler should buffer and decode on the second chunk
      model_struct = %{verbose_api: false}

      transform_fn = fn data ->
        ChatGoogleAI.do_process_response(
          ChatGoogleAI.new!(%{stream: true}),
          data,
          MessageDelta
        )
      end

      handler_fn =
        LangChain.Utils.handle_stream_fn(
          model_struct,
          &LangChain.ChatModels.ChatOpenAI.decode_stream/1,
          transform_fn
        )

      full_json =
        Jason.encode!(%{
          "error" => %{
            "code" => 429,
            "message" => "You exceeded your current quota",
            "status" => "RESOURCE_EXHAUSTED"
          }
        })

      chunk1 = String.slice(full_json, 0, 30)
      chunk2 = String.slice(full_json, 30..-1//1)

      response = %Req.Response{status: 429, headers: %{}, body: ""}

      # First chunk - incomplete JSON, should buffer and continue
      assert {:cont, {req1, response1}} =
               handler_fn.({:data, chunk1}, {Req.Request.new(), response})

      # Verify data was buffered
      assert Req.Response.get_private(response1, :error_buffer) == chunk1

      # Second chunk - completes the JSON, should halt with parsed error
      assert {:halt, {_req2, response2}} =
               handler_fn.({:data, chunk2}, {req1, response1})

      assert {:error, %LangChainError{} = error} = response2.body
      assert error.type == "resource_exhausted"
      assert error.message =~ "exceeded"
    end

    test "streaming error with unbuffered body falls back to error buffer extraction" do
      # When the stream ends but body wasn't set (e.g. error buffer wasn't decoded
      # by the handler), the do_api_request should try to extract from the buffer
      error_json =
        Jason.encode!(%{
          "error" => %{
            "code" => 429,
            "message" => "Quota exceeded",
            "status" => "RESOURCE_EXHAUSTED"
          }
        })

      expect(Req, :post, fn _req, _opts ->
        response = %Req.Response{
          status: 429,
          headers: %{},
          body: ""
        }

        # Simulate a response where the buffer has data but it wasn't decoded
        response = Req.Response.put_private(response, :error_buffer, error_json)
        {:ok, response}
      end)

      model = ChatGoogleAI.new!(%{stream: true, model: "gemini-2.5-pro"})

      assert {:error, %LangChainError{} = error} =
               ChatGoogleAI.call(model, [Message.new_user!("Hello")])

      assert error.type == "resource_exhausted"
      assert error.message == "Quota exceeded"
    end

    test "do_process_response extracts error type from Google NOT_FOUND error", %{model: model} do
      response = %{
        "error" => %{
          "code" => 404,
          "message" => "models/gemini-1.5-flash is not found",
          "status" => "NOT_FOUND"
        }
      }

      assert {:error, %LangChainError{} = error} =
               ChatGoogleAI.do_process_response(model, response)

      assert error.type == "not_found"
      assert error.message =~ "not found"
    end

    test "do_process_response extracts error type from Google RESOURCE_EXHAUSTED error", %{
      model: model
    } do
      response = %{
        "error" => %{
          "code" => 429,
          "message" => "You exceeded your current quota",
          "status" => "RESOURCE_EXHAUSTED"
        }
      }

      assert {:error, %LangChainError{} = error} =
               ChatGoogleAI.do_process_response(model, response)

      assert error.type == "resource_exhausted"
      assert error.message =~ "exceeded"
    end

    test "retry_on_fallback? returns false for not_found errors" do
      error = LangChainError.exception(type: "not_found", message: "Not found")
      refute ChatGoogleAI.retry_on_fallback?(error)
    end

    test "retry_on_fallback? returns false for resource_exhausted errors" do
      error = LangChainError.exception(type: "resource_exhausted", message: "Quota exceeded")
      refute ChatGoogleAI.retry_on_fallback?(error)
    end
  end
end
