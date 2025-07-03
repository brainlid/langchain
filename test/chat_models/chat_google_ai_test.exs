defmodule ChatModels.ChatGoogleAITest do
  use LangChain.BaseCase

  doctest LangChain.ChatModels.ChatGoogleAI
  alias LangChain.ChatModels.ChatGoogleAI
  alias LangChain.Message
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
      assert %{"role" => :user, "parts" => [%{"text" => ^user_message}]} = msg1
      assert %{"role" => :model, "parts" => [%{"text" => ^assistant_message}]} = msg2
    end

    test "generated a map containing response_mime_type and response_schema", %{params: params} do
      google_ai =
        ChatGoogleAI.new!(
          params
          |> Map.merge(%{"json_response" => true, "json_schema" => %{"type" => "object"}})
        )

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
      assert %{"role" => :user, "parts" => [%{"text" => ^message}]} = msg1
      assert %{"role" => :model, "parts" => [tool_call]} = msg2
      assert %{"role" => :function, "parts" => [tool_result]} = msg3

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
          "role" => :function,
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

      tool_result =
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
                   "role" => :user
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
      assert struct.content == "This is the first part of a mes"
      assert struct.index == 0
      assert struct.status == :incomplete
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

      assert struct.content == ""
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

      assert error.type == nil
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
      assert struct.content == "This is a partial message"
      assert struct.index == 0
      assert struct.status == :incomplete

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
      model = ChatGoogleAI.new!(%{model: "gpt-4o"})
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

  describe "google_search native tool" do
    @tag live_call: true, live_google_ai: true
    test "should include grounding metadata in response" do
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

      assert %Message{} = updated_chain.last_message
      assert updated_chain.last_message.role == :assistant
      assert Map.has_key?(updated_chain.last_message.metadata, "groundingChunks")
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
end
