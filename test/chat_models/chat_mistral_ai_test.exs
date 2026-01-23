defmodule LangChain.ChatModels.ChatMistralAITest do
  use LangChain.BaseCase

  alias LangChain.ChatModels.ChatMistralAI
  alias LangChain.Message
  alias LangChain.MessageDelta
  alias LangChain.Message.ContentPart
  alias LangChain.Message.ToolCall
  alias LangChain.Message.ToolResult
  alias LangChain.LangChainError
  alias LangChain.TokenUsage
  alias LangChain.Function

  setup do
    model = ChatMistralAI.new!(%{"model" => "mistral-tiny"})
    %{model: model}
  end

  describe "new/1" do
    test "works with minimal attr" do
      assert {:ok, %ChatMistralAI{} = mistral_ai} =
               ChatMistralAI.new(%{"model" => "mistral-tiny"})

      assert mistral_ai.model == "mistral-tiny"
    end

    test "returns error when invalid" do
      assert {:error, changeset} = ChatMistralAI.new(%{"model" => nil})
      refute changeset.valid?
      assert {"can't be blank", _} = changeset.errors[:model]
    end

    test "supports overriding the API endpoint" do
      override_url = "http://localhost:1234/"

      model =
        ChatMistralAI.new!(%{
          "model" => "mistral-tiny",
          "endpoint" => override_url
        })

      assert model.endpoint == override_url
    end

    test "supports verbose_api option" do
      model = ChatMistralAI.new!(%{model: "mistral-tiny", verbose_api: true})
      assert model.verbose_api == true

      model = ChatMistralAI.new!(%{model: "mistral-tiny", verbose_api: false})
      assert model.verbose_api == false
    end

    test "supports passing parallel_tool_calls" do
      # defaults to true (Mistral API default)
      %ChatMistralAI{} = mistral_ai = ChatMistralAI.new!(%{"model" => "mistral-tiny"})
      assert mistral_ai.parallel_tool_calls == true

      # can override the default to false
      %ChatMistralAI{} =
        mistral_ai =
        ChatMistralAI.new!(%{"model" => "mistral-tiny", "parallel_tool_calls" => false})

      assert mistral_ai.parallel_tool_calls == false
    end
  end

  describe "for_api/3" do
    setup do
      {:ok, mistral_ai} =
        ChatMistralAI.new(%{
          model: "mistral-tiny",
          temperature: 1.0,
          top_p: 1.0,
          max_tokens: 100,
          safe_prompt: true,
          random_seed: 42
        })

      %{mistral_ai: mistral_ai}
    end

    test "generates a map for an API call", %{mistral_ai: mistral_ai} do
      data = ChatMistralAI.for_api(mistral_ai, [], [])

      assert data ==
               %{
                 model: "mistral-tiny",
                 temperature: 1.0,
                 top_p: 1.0,
                 messages: [],
                 stream: false,
                 max_tokens: 100,
                 safe_prompt: true,
                 random_seed: 42,
                 parallel_tool_calls: true
               }
    end

    test "generates a map for an API call with parallel_tool_calls set to false" do
      {:ok, mistral_ai} =
        ChatMistralAI.new(%{
          model: "mistral-tiny",
          parallel_tool_calls: false
        })

      data = ChatMistralAI.for_api(mistral_ai, [], [])
      assert data.model == "mistral-tiny"
      assert data.parallel_tool_calls == false
    end

    test "generates a map containing user and assistant messages", %{mistral_ai: mistral_ai} do
      user_message = "Hello Assistant!"
      assistant_message = "Hello User!"

      data =
        ChatMistralAI.for_api(
          mistral_ai,
          [Message.new_user!(user_message), Message.new_assistant!(assistant_message)],
          []
        )

      assert get_in(data, [:messages, Access.at(0), "role"]) == :user
      assert get_in(data, [:messages, Access.at(0), "content"]) == user_message
      assert get_in(data, [:messages, Access.at(1), "role"]) == :assistant
      assert get_in(data, [:messages, Access.at(1), "content"]) == assistant_message
    end

    test "converts ToolResult with string content correctly", %{mistral_ai: mistral_ai} do
      tool_result = ToolResult.new!(%{tool_call_id: "call_123", content: "Hello World!"})

      result = ChatMistralAI.for_api(mistral_ai, tool_result)

      assert result == %{
               "role" => :tool,
               "tool_call_id" => "call_123",
               "content" => "Hello World!"
             }
    end

    test "converts ToolResult with ContentParts to string", %{mistral_ai: mistral_ai} do
      tool_result =
        ToolResult.new!(%{
          tool_call_id: "call_456",
          content: [ContentPart.text!("Result: 42"), ContentPart.text!(" and more")]
        })

      result = ChatMistralAI.for_api(mistral_ai, tool_result)

      # ContentPart.parts_to_string/1 joins parts with "\n\n"
      assert result == %{
               "role" => :tool,
               "tool_call_id" => "call_456",
               "content" => "Result: 42\n\n and more"
             }
    end

    test "converts Message with tool role and ContentParts to list of tool messages", %{
      mistral_ai: mistral_ai
    } do
      tool_result =
        ToolResult.new!(%{
          tool_call_id: "call_789",
          content: [ContentPart.text!("Tool executed successfully")]
        })

      message = Message.new_tool_result!(%{tool_results: [tool_result]})

      result = ChatMistralAI.for_api(mistral_ai, message)

      # Returns a list of tool messages, one per tool result
      assert result == [
               %{
                 "role" => :tool,
                 "tool_call_id" => "call_789",
                 "content" => "Tool executed successfully"
               }
             ]
    end

    test "converts Message with multiple tool results to list of tool messages", %{
      mistral_ai: mistral_ai
    } do
      tool_result_1 =
        ToolResult.new!(%{
          tool_call_id: "call_abc",
          content: "Result 1"
        })

      tool_result_2 =
        ToolResult.new!(%{
          tool_call_id: "call_def",
          content: "Result 2"
        })

      message = Message.new_tool_result!(%{tool_results: [tool_result_1, tool_result_2]})

      result = ChatMistralAI.for_api(mistral_ai, message)

      # Each tool result becomes a separate tool message
      assert result == [
               %{
                 "role" => :tool,
                 "tool_call_id" => "call_abc",
                 "content" => "Result 1"
               },
               %{
                 "role" => :tool,
                 "tool_call_id" => "call_def",
                 "content" => "Result 2"
               }
             ]
    end
  end

  describe "do_process_response/2" do
    test "handles receiving a message", %{model: model} do
      response = %{
        "choices" => [
          %{
            "message" => %{
              "role" => "assistant",
              "content" => "Hello User!",
              "tool_calls" => []
            },
            "finish_reason" => "stop",
            "index" => 0
          }
        ],
        "usage" => %{
          "prompt_tokens" => 7,
          "completion_tokens" => 10,
          "total_tokens" => 17
        }
      }

      assert [%Message{} = msg] = ChatMistralAI.do_process_response(model, response)
      assert msg.role == :assistant
      assert msg.content == [ContentPart.text!("Hello User!")]
      assert msg.index == 0
      assert msg.status == :complete
    end

    test "errors with invalid role", %{model: model} do
      response = %{
        "choices" => [
          %{
            "delta" => %{
              "role" => "unknown role",
              "content" => "Hello User!"
            },
            "finish_reason" => "stop",
            "index" => 0
          }
        ]
      }

      assert [{:error, %LangChainError{} = error}] =
               ChatMistralAI.do_process_response(model, response)

      assert error.message =~ "role" and error.message =~ "invalid"
    end

    test "handles receiving MessageDeltas as well", %{model: model} do
      response = %{
        "choices" => [
          %{
            "delta" => %{
              "role" => "assistant",
              "content" => "This is the first part of a mes"
            },
            "finish_reason" => nil,
            "index" => 0
          }
        ]
      }

      assert [%MessageDelta{} = delta] =
               ChatMistralAI.do_process_response(model, response)

      assert delta.role == :assistant
      assert delta.content == "This is the first part of a mes"
      assert delta.index == 1
      assert delta.status == :incomplete
    end

    test "handles receiving MessageDeltas with thinking content", %{model: model} do
      response = %{
        "choices" => [
          %{
            "delta" => %{
              "role" => "assistant",
              "content" => [
                %{
                  "type" => "thinking",
                  "thinking" => [
                    %{"type" => "text", "text" => "Let me think about this"}
                  ]
                }
              ]
            },
            "finish_reason" => nil,
            "index" => 0
          }
        ]
      }

      assert [%MessageDelta{} = delta] =
               ChatMistralAI.do_process_response(model, response)

      assert delta.role == :assistant
      assert %ContentPart{type: :thinking, content: "Let me think about this"} = delta.content
      assert delta.index == 0
      assert delta.status == :incomplete
    end

    test "handles receiving MessageDeltas with multiple thinking text parts", %{model: model} do
      response = %{
        "choices" => [
          %{
            "delta" => %{
              "role" => "assistant",
              "content" => [
                %{
                  "type" => "thinking",
                  "thinking" => [
                    %{"type" => "text", "text" => "First part "},
                    %{"type" => "text", "text" => "second part"}
                  ]
                }
              ]
            },
            "finish_reason" => nil,
            "index" => 0
          }
        ]
      }

      assert [%MessageDelta{} = delta] =
               ChatMistralAI.do_process_response(model, response)

      assert delta.role == :assistant
      assert %ContentPart{type: :thinking, content: "First part second part"} = delta.content
      assert delta.index == 0
      assert delta.status == :incomplete
    end

    test "handles receiving MessageDeltas with text content in list format", %{model: model} do
      response = %{
        "choices" => [
          %{
            "delta" => %{
              "role" => "assistant",
              "content" => [
                %{
                  "type" => "text",
                  "text" => "This is regular text"
                }
              ]
            },
            "finish_reason" => nil,
            "index" => 0
          }
        ]
      }

      assert [%MessageDelta{} = delta] =
               ChatMistralAI.do_process_response(model, response)

      assert delta.role == :assistant
      assert %ContentPart{type: :text, content: "This is regular text"} = delta.content
      # Text content at index 0 is shifted to index 1 to avoid merging with thinking at index 0
      assert delta.index == 1
      assert delta.status == :incomplete
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
               ChatMistralAI.do_process_response(model, response)

      assert error.message == "Invalid request"
    end

    test "handles Jason.DecodeError", %{model: model} do
      response = {:error, %Jason.DecodeError{}}

      assert {:error, %LangChainError{} = error} =
               ChatMistralAI.do_process_response(model, response)

      assert error.type == "invalid_json"
      assert error.message =~ "Received invalid JSON:"
    end

    test "handles unexpected response with error", %{model: model} do
      response = %{}

      assert {:error, %LangChainError{} = error} =
               ChatMistralAI.do_process_response(model, response)

      assert error.type == "unexpected_response"
      assert error.message == "Unexpected response"
    end
  end

  describe "do_process_response/2 with tool calls" do
    test "handles receiving multiple tool_calls messages", %{model: model} do
      response = %{
        "choices" => [
          %{
            "message" => %{
              "role" => "assistant",
              "content" => nil,
              "tool_calls" => [
                %{
                  "type" => "function",
                  "id" => "call_abc123",
                  "function" => %{
                    "name" => "get_weather",
                    "arguments" => "{\"city\":\"Moab\",\"state\":\"UT\"}"
                  }
                },
                %{
                  "type" => "function",
                  "id" => "call_def456",
                  "function" => %{
                    "name" => "get_weather",
                    "arguments" => "{\"city\":\"Portland\",\"state\":\"OR\"}"
                  }
                }
              ]
            },
            "finish_reason" => "tool_calls",
            "index" => 0
          }
        ],
        "usage" => %{
          "prompt_tokens" => 10,
          "completion_tokens" => 5,
          "total_tokens" => 15
        }
      }

      assert [%Message{} = msg] =
               ChatMistralAI.do_process_response(model, response)

      assert msg.role == :assistant
      assert msg.status == :complete
      assert msg.content == nil
      assert length(msg.tool_calls) == 2

      [call1, call2] = msg.tool_calls

      assert %ToolCall{
               type: :function,
               call_id: "call_abc123",
               name: "get_weather",
               arguments: %{"city" => "Moab", "state" => "UT"}
             } = call1

      assert %ToolCall{
               type: :function,
               call_id: "call_def456",
               name: "get_weather",
               arguments: %{"city" => "Portland", "state" => "OR"}
             } = call2
    end

    test "handles invalid JSON in a tool_call", %{model: model} do
      response = %{
        "choices" => [
          %{
            "message" => %{
              "role" => "assistant",
              "content" => nil,
              "tool_calls" => [
                %{
                  "type" => "function",
                  "id" => "call_abc123",
                  "function" => %{
                    "name" => "get_weather",
                    "arguments" => "{\"invalid\"}"
                  }
                }
              ]
            },
            "finish_reason" => "tool_calls",
            "index" => 0
          }
        ]
      }

      assert [{:error, %LangChainError{} = error}] =
               ChatMistralAI.do_process_response(model, response)

      assert error.type == "changeset"
      assert error.message =~ "invalid json"
    end
  end

  describe "do_process_response/2 with token usage" do
    test "handles a normal message and usage info", %{model: model} do
      response = %{
        "choices" => [
          %{
            "message" => %{
              "role" => "assistant",
              "content" => "Hello from Mistral!",
              "tool_calls" => []
            },
            "finish_reason" => "stop",
            "index" => 0
          }
        ],
        "usage" => %{
          "prompt_tokens" => 7,
          "completion_tokens" => 10,
          "total_tokens" => 17
        }
      }

      result = ChatMistralAI.do_process_response(model, response)

      assert [%Message{role: :assistant, status: :complete} = message] = result
      assert message.content == [ContentPart.text!("Hello from Mistral!")]
    end
  end

  describe "serialize_config/2" do
    test "does not include the API key or callbacks" do
      model = ChatMistralAI.new!(%{model: "mistral-tiny"})
      result = ChatMistralAI.serialize_config(model)
      assert result["version"] == 1
      refute Map.has_key?(result, "api_key")
      refute Map.has_key?(result, "callbacks")
    end

    test "includes verbose_api field" do
      model = ChatMistralAI.new!(%{model: "mistral-tiny", verbose_api: true})
      result = ChatMistralAI.serialize_config(model)
      assert result["verbose_api"] == true
    end

    test "creates expected map" do
      model =
        ChatMistralAI.new!(%{
          model: "mistral-tiny",
          temperature: 1.0,
          top_p: 1.0,
          max_tokens: 100,
          safe_prompt: true,
          random_seed: 42
        })

      result = ChatMistralAI.serialize_config(model)

      assert result == %{
               "endpoint" => "https://api.mistral.ai/v1/chat/completions",
               "model" => "mistral-tiny",
               "max_tokens" => 100,
               "module" => "Elixir.LangChain.ChatModels.ChatMistralAI",
               "receive_timeout" => 60000,
               "stream" => false,
               "temperature" => 1.0,
               "random_seed" => 42,
               "safe_prompt" => true,
               "top_p" => 1.0,
               "version" => 1,
               "json_response" => false,
               "json_schema" => nil,
               "verbose_api" => false
             }
    end
  end

  describe "inspect" do
    test "redacts the API key" do
      chain = ChatMistralAI.new!(%{"model" => "mistral-tiny"})

      changeset = Ecto.Changeset.cast(chain, %{api_key: "1234567890"}, [:api_key])

      refute inspect(changeset) =~ "1234567890"
      assert inspect(changeset) =~ "**redacted**"
    end
  end

  describe "structured output support" do
    test "new/1 accepts json_response and json_schema fields" do
      schema = %{
        "type" => "json_schema",
        "json_schema" => %{
          "schema" => %{
            "properties" => %{
              "name" => %{"title" => "Name", "type" => "string"},
              "authors" => %{
                "items" => %{"type" => "string"},
                "title" => "Authors",
                "type" => "array"
              }
            },
            "required" => ["name", "authors"],
            "title" => "Book",
            "type" => "object",
            "additionalProperties" => false
          },
          "name" => "book",
          "strict" => true
        }
      }

      assert {:ok, %ChatMistralAI{} = mistral_ai} =
               ChatMistralAI.new(%{
                 "model" => "ministral-8b-latest",
                 "json_response" => true,
                 "json_schema" => schema
               })

      assert mistral_ai.json_response == true
      assert mistral_ai.json_schema == schema
    end

    test "for_api/3 includes response_format when json_response is true with schema" do
      schema = %{
        "type" => "json_schema",
        "json_schema" => %{
          "schema" => %{
            "properties" => %{
              "name" => %{"title" => "Name", "type" => "string"},
              "authors" => %{
                "items" => %{"type" => "string"},
                "title" => "Authors",
                "type" => "array"
              }
            },
            "required" => ["name", "authors"],
            "title" => "Book",
            "type" => "object",
            "additionalProperties" => false
          },
          "name" => "book",
          "strict" => true
        }
      }

      mistral_ai =
        ChatMistralAI.new!(%{
          "model" => "ministral-8b-latest",
          "json_response" => true,
          "json_schema" => schema
        })

      data = ChatMistralAI.for_api(mistral_ai, [], [])

      assert data.response_format == schema
    end

    test "for_api/3 includes response_format when json_response is true without schema" do
      mistral_ai =
        ChatMistralAI.new!(%{
          "model" => "ministral-8b-latest",
          "json_response" => true
        })

      data = ChatMistralAI.for_api(mistral_ai, [], [])

      assert data.response_format == %{"type" => "json_object"}
    end

    test "for_api/3 does not include response_format when json_response is false" do
      mistral_ai =
        ChatMistralAI.new!(%{
          "model" => "ministral-8b-latest",
          "json_response" => false
        })

      data = ChatMistralAI.for_api(mistral_ai, [], [])

      refute Map.has_key?(data, :response_format)
    end

    test "serialize_config/1 includes json_response and json_schema fields" do
      schema = %{
        "type" => "json_schema",
        "json_schema" => %{
          "schema" => %{
            "properties" => %{
              "name" => %{"title" => "Name", "type" => "string"}
            },
            "required" => ["name"],
            "title" => "Book",
            "type" => "object"
          },
          "name" => "book",
          "strict" => true
        }
      }

      model =
        ChatMistralAI.new!(%{
          "model" => "ministral-8b-latest",
          "json_response" => true,
          "json_schema" => schema
        })

      result = ChatMistralAI.serialize_config(model)

      assert result["json_response"] == true
      assert result["json_schema"] == schema
    end

    test "restore_from_map/1 restores json_response and json_schema fields" do
      schema = %{
        "type" => "json_schema",
        "json_schema" => %{
          "schema" => %{
            "properties" => %{
              "name" => %{"title" => "Name", "type" => "string"}
            },
            "required" => ["name"],
            "title" => "Book",
            "type" => "object"
          },
          "name" => "book",
          "strict" => true
        }
      }

      config = %{
        "version" => 1,
        "model" => "ministral-8b-latest",
        "json_response" => true,
        "json_schema" => schema
      }

      assert {:ok, %ChatMistralAI{} = restored} = ChatMistralAI.restore_from_map(config)
      assert restored.json_response == true
      assert restored.json_schema == schema
    end
  end

  describe "live tests and token usage information" do
    @tag live_call: true, live_mistral_ai: true
    test "basic non-streamed response works and fires token usage callback" do
      handlers = %{
        on_llm_token_usage: fn usage ->
          send(self(), {:fired_token_usage, usage})
        end
      }

      %ChatMistralAI{} =
        chat =
        ChatMistralAI.new!(%{
          temperature: 0,
          model: "mistral-small-2503",
          stream: false
        })

      chat = %ChatMistralAI{chat | callbacks: [handlers]}

      {:ok, result} =
        ChatMistralAI.call(
          chat,
          [
            Message.new_user!(
              "Return the response 'Colorful Threads'. Don't return anything else."
            )
          ],
          []
        )

      # returns a list of MessageDeltas. A list of a list because it's "n" choices.
      assert result == [
               %Message{
                 content: [ContentPart.text!("Colorful Threads")],
                 status: :complete,
                 role: :assistant,
                 index: 0,
                 tool_calls: []
               }
             ]

      assert_received {:fired_token_usage, usage}
      assert %TokenUsage{input: 18} = usage
      # Allow for slight variation in token count
      assert usage.output in [4, 5]
    end

    @tag live_call: true, live_mistral_ai: true
    test "streamed response works and fires token usage callback" do
      handlers = %{
        on_llm_token_usage: fn usage ->
          send(self(), {:fired_token_usage, usage})
        end
      }

      %ChatMistralAI{} =
        chat =
        ChatMistralAI.new!(%{
          temperature: 0,
          model: "mistral-small-2503",
          stream: true
        })

      chat = %ChatMistralAI{chat | callbacks: [handlers]}

      {:ok, result} =
        ChatMistralAI.call(
          chat,
          [
            Message.new_user!(
              "Return the response 'Colorful Threads'. Don't return anything else."
            )
          ],
          []
        )

      result_string =
        Enum.map_join(result, fn msg ->
          assert [%MessageDelta{role: :assistant, content: content, tool_calls: nil}] = msg
          content
        end)

      [last_delta] = List.last(result)
      assert last_delta.status == :complete
      assert result_string == "Colorful Threads"

      assert_received {:fired_token_usage, usage}
      assert %TokenUsage{input: 18} = usage
      # Allow for slight variation in token count
      assert usage.output in [4, 5]
    end

    @tag live_call: true, live_mistral_ai: true
    test "streamed response with tool calls work" do
      chat =
        ChatMistralAI.new!(%{
          temperature: 0,
          model: "mistral-small-2503",
          stream: true
        })

      function =
        Function.new!(%{
          name: "current_time",
          description: "Get the current time",
          function: fn _args, _context -> {:ok, dbg("It's late")} end
        })

      {:ok, result} =
        ChatMistralAI.call(
          chat,
          [
            Message.new_user!("Call the current_time function and return the response.")
          ],
          [function]
        )

      tool_call_msg = Enum.find(result, fn [msg] -> msg.tool_calls != nil end)
      assert [%MessageDelta{tool_calls: [%ToolCall{name: "current_time"}]}] = tool_call_msg
    end

    @tag live_call: true, live_mistral_ai: true
    test "structured output with JSON schema works" do
      schema = %{
        "type" => "json_schema",
        "json_schema" => %{
          "schema" => %{
            "properties" => %{
              "name" => %{"title" => "Name", "type" => "string"},
              "authors" => %{
                "items" => %{"type" => "string"},
                "title" => "Authors",
                "type" => "array"
              }
            },
            "required" => ["name", "authors"],
            "title" => "Book",
            "type" => "object",
            "additionalProperties" => false
          },
          "name" => "book",
          "strict" => true
        }
      }

      chat =
        ChatMistralAI.new!(%{
          temperature: 0,
          model: "ministral-8b-latest",
          stream: false,
          json_response: true,
          json_schema: schema
        })

      {:ok, result} =
        ChatMistralAI.call(
          chat,
          [
            Message.new_system!("Extract the books information."),
            Message.new_user!("I recently read To Kill a Mockingbird by Harper Lee.")
          ],
          []
        )

      assert [%Message{content: content, status: :complete, role: :assistant}] = result

      # The content should be a valid JSON string that matches our schema
      assert is_list(content)
      assert [%ContentPart{type: :text, content: json_content}] = content
      assert is_binary(json_content)
      {:ok, parsed_json} = Jason.decode(json_content)

      # Verify the structure matches our schema
      assert %{"name" => name, "authors" => authors} = parsed_json
      assert is_binary(name)
      assert is_list(authors)
      assert Enum.all?(authors, &is_binary/1)
    end

    @tag live_call: true, live_mistral_ai: true
    test "structured output with json_object type works" do
      chat =
        ChatMistralAI.new!(%{
          temperature: 0,
          model: "ministral-8b-latest",
          stream: false,
          json_response: true
        })

      {:ok, result} =
        ChatMistralAI.call(
          chat,
          [
            Message.new_system!("Extract the books information and return as JSON."),
            Message.new_user!("I recently read To Kill a Mockingbird by Harper Lee.")
          ],
          []
        )

      assert [%Message{content: content, status: :complete, role: :assistant}] = result

      # The content should be a valid JSON string
      assert is_list(content)
      assert [%ContentPart{type: :text, content: json_content}] = content
      assert is_binary(json_content)
      assert {:ok, _parsed_json} = Jason.decode(json_content)
    end
  end
end
