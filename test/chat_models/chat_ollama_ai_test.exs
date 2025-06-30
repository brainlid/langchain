defmodule ChatModels.ChatOllamaAITest do
  use LangChain.BaseCase

  doctest LangChain.ChatModels.ChatOllamaAI

  alias LangChain.ChatModels.ChatOllamaAI
  alias LangChain.Function
  alias LangChain.FunctionParam
  alias LangChain.Message.ContentPart

  use Mimic

  setup do
    model = ChatOllamaAI.new!(%{"model" => "llama2:latest"})

    %{model: model}
  end

  describe "new/1" do
    test "works with minimal attributes" do
      assert {:ok, %ChatOllamaAI{} = ollama_ai} = ChatOllamaAI.new(%{"model" => "llama2:latest"})
      assert ollama_ai.model == "llama2:latest"
      assert ollama_ai.endpoint == "http://localhost:11434/api/chat"
    end

    test "returns errors given invalid attributes" do
      assert {:error, changeset} =
               ChatOllamaAI.new(%{"model" => nil, "temperature" => 4.4, "mirostat_eta" => 4.4})

      refute changeset.valid?
      assert {"can't be blank", _} = changeset.errors[:model]
      assert {"must be less than or equal to %{number}", _} = changeset.errors[:temperature]
      assert {"must be less than or equal to %{number}", _} = changeset.errors[:mirostat_eta]
    end

    test "supports overriding the API endpoint" do
      override_url = "http://localhost:99999/api/chat"

      model =
        ChatOllamaAI.new!(%{
          endpoint: override_url
        })

      assert model.endpoint == override_url
    end
  end

  describe "for_api/3" do
    setup do
      {:ok, ollama_ai} =
        ChatOllamaAI.new(%{
          "model" => "llama2:latest",
          "temperature" => 0.4,
          "stream" => false,
          "seed" => 0,
          "num_ctx" => 2048,
          "num_predict" => 128,
          "repeat_last_n" => 64,
          "repeat_penalty" => 1.1,
          "mirostat" => 0,
          "mirostat_eta" => 0.1,
          "mirostat_tau" => 5.0,
          "num_gqa" => 8,
          "num_gpu" => 1,
          "num_thread" => 0,
          "receive_timeout" => 300_000,
          "stop" => "",
          "tfs_z" => 0.0,
          "top_k" => 0,
          "top_p" => 0.0
        })

      %{ollama_ai: ollama_ai}
    end

    test "generates a map for an API call with no messages", %{ollama_ai: ollama_ai} do
      data = ChatOllamaAI.for_api(ollama_ai, [], [])
      assert data.model == "llama2:latest"
      assert data.stream == false
      assert data.messages == []
      assert data.receive_timeout == 300_000

      assert data.options.temperature == 0.4
      assert data.options.seed == 0
      assert data.options.num_ctx == 2048
      assert data.options.num_predict == 128
      assert data.options.repeat_last_n == 64
      assert data.options.repeat_penalty == 1.1
      assert data.options.mirostat == 0
      assert data.options.mirostat_eta == 0.1
      assert data.options.mirostat_tau == 5.0
      assert data.options.num_gqa == 8
      assert data.options.num_gpu == 1
      assert data.options.num_thread == 0
      # TODO: figure out why this is field is is being cast to nil instead of empty string
      assert data.options.stop == nil
      assert data.options.tfs_z == 0.0
      assert data.options.top_k == 0
      assert data.options.top_p == 0.0
    end

    test "generates a map for an API call with a single message", %{ollama_ai: ollama_ai} do
      user_message = "What color is the sky?"

      data = ChatOllamaAI.for_api(ollama_ai, [Message.new_user!(user_message)], [])
      assert data.model == "llama2:latest"
      assert data.options.temperature == 0.4

      assert [%{"content" => "What color is the sky?", "role" => :user}] = data.messages
    end

    test "generates a map for an API call with user and system messages", %{ollama_ai: ollama_ai} do
      user_message = "What color is the sky?"
      system_message = "You are a weather man"

      data =
        ChatOllamaAI.for_api(
          ollama_ai,
          [Message.new_system!(system_message), Message.new_user!(user_message)],
          []
        )

      assert data.model == "llama2:latest"
      assert data.options.temperature == 0.4

      assert [
               %{"role" => :system} = system_msg,
               %{"role" => :user} = user_msg
             ] = data.messages

      assert system_msg["content"] == "You are a weather man"
      assert user_msg["content"] == "What color is the sky?"
    end

    test "generates a map for an API call with a tool", %{ollama_ai: ollama_ai} do
      fun =
        Function.new!(%{
          name: "give_greeting",
          description: "Gives a friendly greeting for the given subject",
          parameters_schema: %{
            type: "object",
            properties: %{
              name: %{
                type: "string",
                description: "The subject to greet"
              }
            },
            required: ["name"]
          },
          function: fn %{"name" => name} = _arguments, _context -> {:ok, "Hello, #{name}!"} end
        })

      expected = [
        %{
          "function" => %{
            "description" => "Gives a friendly greeting for the given subject",
            "name" => "give_greeting",
            "parameters" => %{
              type: "object",
              required: ["name"],
              properties: %{
                name: %{type: "string", description: "The subject to greet"}
              }
            }
          },
          "type" => "function"
        }
      ]

      data = ChatOllamaAI.for_api(ollama_ai, [], [fun])

      assert expected == data.tools
    end

    test "generates a map for an API call with a tool using FunctionParams", %{
      ollama_ai: ollama_ai
    } do
      fun =
        Function.new!(%{
          name: "give_greeting",
          description: "Gives a friendly greeting for the given subject",
          parameters: [
            FunctionParam.new!(%{name: "name", type: :string, required: true})
          ],
          function: fn %{"name" => name} = _arguments, _context -> {:ok, "Hello, #{name}!"} end
        })

      expected = [
        %{
          "function" => %{
            "description" => "Gives a friendly greeting for the given subject",
            "name" => "give_greeting",
            "parameters" => %{
              "properties" => %{"name" => %{"type" => "string"}},
              "required" => ["name"],
              "type" => "object"
            }
          },
          "type" => "function"
        }
      ]

      data = ChatOllamaAI.for_api(ollama_ai, [], [fun])

      assert expected == data.tools
    end

    test "generates a map for an API call with a tool without parameters", %{ollama_ai: ollama_ai} do
      fun =
        Function.new!(%{
          name: "greet_the_world",
          description: "Be friendly to the world",
          function: fn _arguments, _context -> {:ok, "Hello, world!"} end
        })

      expected = [
        %{
          "function" => %{
            "description" => "Be friendly to the world",
            "name" => "greet_the_world",
            "parameters" => %{"properties" => %{}, "type" => "object"}
          },
          "type" => "function"
        }
      ]

      data = ChatOllamaAI.for_api(ollama_ai, [], [fun])

      assert expected == data.tools
    end

    test "generates a map for an API call without tools", %{ollama_ai: ollama_ai} do
      data = ChatOllamaAI.for_api(ollama_ai, [], nil)

      assert data[:tools] == nil
    end

    test "for assistant message with non-empty tool_calls, generates an assistant message with a list of ToolCall messages",
         %{ollama_ai: ollama_ai} do
      tool_call =
        Message.ToolCall.new!(%{
          call_id: "call_123",
          name: "give_greeting",
          arguments: %{"name" => "world"}
        })

      expected = [
        %{
          "content" => nil,
          "role" => :assistant,
          "tool_calls" => [
            %{
              "function" => %{
                "arguments" => %{"name" => "world"},
                "name" => "give_greeting"
              },
              "id" => "call_123",
              "type" => "function"
            }
          ]
        }
      ]

      data =
        ChatOllamaAI.for_api(ollama_ai, [Message.new_assistant!(%{tool_calls: [tool_call]})], [])

      assert expected == data.messages
    end

    test "for assistant message empty tool-calls, generates an assistant message", %{
      ollama_ai: ollama_ai
    } do
      data = ChatOllamaAI.for_api(ollama_ai, [Message.new_assistant!("Hello, world!")], [])
      expected = [%{"content" => "Hello, world!", "role" => :assistant}]

      assert expected == data.messages
    end

    test "for tool call, generate expected structure" do
      tool_call =
        Message.ToolCall.new!(%{
          call_id: "call_123",
          name: "give_greeting",
          arguments: %{"name" => "world"}
        })

      expected = %{
        "function" => %{"arguments" => %{"name" => "world"}, "name" => "give_greeting"},
        "id" => "call_123",
        "type" => "function"
      }

      assert expected == ChatOllamaAI.for_api(tool_call)
    end

    test "for function, return expected structure" do
      function =
        Function.new!(%{
          name: "give_greeting",
          description: "Gives a friendly greeting to the given recipient",
          parameters: [
            FunctionParam.new!(%{name: "name", type: :string, required: true})
          ],
          function: fn %{"name" => name} = _args, _context ->
            {:ok, "Hello, #{name}!"}
          end
        })

      expected = %{
        "description" => "Gives a friendly greeting to the given recipient",
        "name" => "give_greeting",
        "parameters" => %{
          "properties" => %{
            "name" => %{"type" => "string"}
          },
          "required" => ["name"],
          "type" => "object"
        }
      }

      assert expected == ChatOllamaAI.for_api(function)
    end

    test "for message with a list of tool results, generate expected structure" do
      tool_result =
        Message.ToolResult.new!(%{
          type: :function,
          tool_call_id: "call_123",
          name: "give_greeting",
          content: [ContentPart.text!("Hello, world!")],
          display_text: nil,
          is_error: false
        })

      message =
        Message.new_tool_result!(%{
          tool_results: [tool_result]
        })

      expected = [%{"role" => :tool, "content" => "Hello, world!"}]
      assert expected == ChatOllamaAI.for_api(message)
    end

    test "for user message, generate expected structure" do
      message = Message.new_user!("Hello!")
      expected = %{"role" => :user, "content" => "Hello!"}

      assert expected == ChatOllamaAI.for_api(message)
    end

    test "for nested messages, handle them all", %{ollama_ai: ollama_ai} do
      messages = [
        %LangChain.Message{
          content: [ContentPart.text!("Where is the hairbrush located?")],
          processed_content: nil,
          index: nil,
          status: :complete,
          role: :user,
          name: nil,
          tool_calls: [],
          tool_results: nil
        },
        %LangChain.Message{
          content: nil,
          processed_content: nil,
          index: nil,
          status: :complete,
          role: :assistant,
          name: nil,
          tool_calls: [
            %LangChain.Message.ToolCall{
              status: :complete,
              type: :function,
              call_id: "54836033-8394-4a97-abc5-34c2d4b9fdbf",
              name: "custom",
              arguments: %{"thing" => "hairbrush"},
              index: nil
            }
          ],
          tool_results: nil
        },
        %LangChain.Message{
          content: nil,
          processed_content: nil,
          index: nil,
          status: :complete,
          role: :tool,
          name: nil,
          tool_calls: [],
          tool_results: [
            %LangChain.Message.ToolResult{
              type: :function,
              tool_call_id: "54836033-8394-4a97-abc5-34c2d4b9fdbf",
              name: "custom",
              content: [ContentPart.text!("drawer")],
              display_text: nil,
              is_error: false
            }
          ]
        },
        %LangChain.Message{
          content: [ContentPart.text!("The hairbrush is located in the drawer.")],
          processed_content: nil,
          index: nil,
          status: :complete,
          role: :assistant,
          name: nil,
          tool_calls: [],
          tool_results: nil
        }
      ]

      expected = [
        %{"content" => "Where is the hairbrush located?", "role" => :user},
        %{
          "content" => nil,
          "role" => :assistant,
          "tool_calls" => [
            %{
              "function" => %{"arguments" => %{"thing" => "hairbrush"}, "name" => "custom"},
              "id" => "54836033-8394-4a97-abc5-34c2d4b9fdbf",
              "type" => "function"
            }
          ]
        },
        %{"content" => "drawer", "role" => :tool},
        %{"content" => "The hairbrush is located in the drawer.", "role" => :assistant}
      ]

      assert %{messages: ^expected} = ChatOllamaAI.for_api(ollama_ai, messages, nil)
    end
  end

  describe "call/2" do
    @tag live_call: true, live_ollama_ai: true
    test "basic content example with no streaming" do
      {:ok, chat} =
        ChatOllamaAI.new(%{
          model: "llama2:latest",
          temperature: 1,
          seed: 0,
          stream: false
        })

      {:ok, %Message{role: :assistant, content: response}} =
        ChatOllamaAI.call(chat, [
          Message.new_user!("Return the response 'Colorful Threads'.")
        ])

      assert response =~ "Colorful Threads"
    end

    @tag live_call: true, live_ollama_ai: true
    test "basic content example with streaming" do
      {:ok, chat} =
        ChatOllamaAI.new(%{
          model: "llama2:latest",
          temperature: 1,
          seed: 0,
          stream: true
        })

      result =
        ChatOllamaAI.call(chat, [
          Message.new_user!("Return the response 'Colorful Threads'.")
        ])

      assert {:ok, deltas} = result
      assert length(deltas) > 0

      deltas_except_last = Enum.slice(deltas, 0..-2//-1)

      for delta <- deltas_except_last do
        assert delta.__struct__ == LangChain.MessageDelta
        assert is_binary(delta.content)
        assert delta.status == :incomplete
        assert delta.role == :assistant
      end

      last_delta = Enum.at(deltas, -1)
      assert last_delta.__struct__ == LangChain.Message
      assert is_nil(last_delta.content)
      assert last_delta.status == :complete
      assert last_delta.role == :assistant
    end

    @tag live_call: true, live_ollama_ai: true
    test "returns an error when given an invalid payload" do
      invalid_model = "invalid"

      {:error, reason} =
        ChatOllamaAI.call(%ChatOllamaAI{model: invalid_model}, [
          Message.new_user!("Return the response 'Colorful Threads'.")
        ])

      assert reason == "model '#{invalid_model}' not found, try pulling it first"
    end

    @tag live_call: true, live_ollama_ai: true
    test "provided tool is not necessarily used", %{
      models: %{llama31: model},
      tools: %{locator: locator}
    } do
      {:ok, chat} = ChatOllamaAI.new(model)
      {:ok, msg} = Message.new_user("Good morning")
      {:ok, %{tool_calls: calls} = _message} = ChatOllamaAI.call(chat, [msg], [locator])

      assert [] == calls
    end

    @tag live_call: true, live_ollama_ai: true
    test "provided tool is called (online)", %{
      models: %{llama31: model},
      tools: %{locator: locator}
    } do
      {:ok, chat} = ChatOllamaAI.new(model)
      {:ok, msg} = Message.new_user("Where is the hairbrush located?")
      {:ok, %{tool_calls: calls} = _message} = ChatOllamaAI.call(chat, [msg], [locator])

      assert [%Message.ToolCall{name: "locator", arguments: %{"thing" => "hairbrush"}}] = calls
    end

    @tag live_ollama_ai: true
    test "provided tool is called", %{models: %{llama31: model}, tools: %{locator: locator}} do
      {:ok, chat} = ChatOllamaAI.new(model)
      {:ok, msg} = Message.new_user("Where is the hairbrush located?")

      expect(ChatOllamaAI, :do_api_request, fn _model, _msgs, _tools ->
        %LangChain.Message{
          content: nil,
          processed_content: nil,
          index: nil,
          status: :complete,
          role: :assistant,
          name: nil,
          tool_calls: [
            %LangChain.Message.ToolCall{
              status: :complete,
              type: :function,
              call_id: "4806e4e4-b1fd-48a4-b969-b6ae0045bb90",
              name: "locator",
              arguments: %{"thing" => "hairbrush"},
              index: nil
            }
          ],
          tool_results: nil
        }
      end)

      {:ok, %{tool_calls: calls} = _message} = ChatOllamaAI.call(chat, [msg], [locator])

      assert [%Message.ToolCall{name: "locator", arguments: %{"thing" => "hairbrush"}}] = calls
    end

    setup do
      locator =
        Function.new!(%{
          name: "locator",
          description: "Returns the location of the requested element or item.",
          parameters: [
            FunctionParam.new!(%{
              name: "thing",
              type: :string,
              description: "the thing whose location is being request"
            })
          ],
          function: fn %{"thing" => thing} = _arguments, context ->
            # our context is a pretend item/location location map
            {:ok, context[thing]}
          end
        })

      llama31 = %{
        model: "llama3.1:latest",
        temperature: 1,
        seed: 0,
        stream: false
      }

      {:ok, %{models: %{llama31: llama31}, tools: %{locator: locator}}}
    end
  end

  describe "do_process_response/1" do
    test "handles receiving a non streamed message result", %{model: model} do
      response = %{
        "model" => "llama2",
        "created_at" => "2024-01-15T23:02:24.087444Z",
        "message" => %{
          "role" => "assistant",
          "content" => "Greetings!"
        },
        "done" => true,
        "total_duration" => 12_323_379_834,
        "load_duration" => 6_889_264_834,
        "prompt_eval_count" => 26,
        "prompt_eval_duration" => 91_493_000,
        "eval_count" => 362,
        "eval_duration" => 5_336_241_000
      }

      assert %Message{} = struct = ChatOllamaAI.do_process_response(model, response)
      assert struct.role == :assistant
      assert struct.content == [ContentPart.text!("Greetings!")]
      assert struct.index == nil
    end

    test "handles receiving a streamed message result", %{model: model} do
      response = %{
        "model" => "llama2",
        "created_at" => "2024-01-15T23:02:24.087444Z",
        "message" => %{
          "role" => "assistant",
          "content" => "Gre"
        },
        "done" => false
      }

      assert %MessageDelta{} = struct = ChatOllamaAI.do_process_response(model, response)
      assert struct.role == :assistant
      assert struct.content == "Gre"
      assert struct.status == :incomplete
    end

    test "handles receiving a tool call request response", %{model: model} do
      response = %{
        "created_at" => "2024-08-05T09:13:24.222066Z",
        "done" => true,
        "done_reason" => "stop",
        "eval_count" => 17,
        "eval_duration" => 303_049_000,
        "load_duration" => 12_754_875,
        "message" => %{
          "content" => "",
          "role" => "assistant",
          "tool_calls" => [
            %{
              "function" => %{
                "arguments" => %{"thing" => "hairbrush"},
                "name" => "custom"
              }
            }
          ]
        },
        "model" => "llama3.1",
        "prompt_eval_count" => 160,
        "prompt_eval_duration" => 441_402_000,
        "total_duration" => 757_930_875
      }

      assert %Message{} = msg = ChatOllamaAI.do_process_response(model, response)
      assert msg.role == :assistant
      assert msg.content == nil
      assert msg.index == nil

      assert [
               %LangChain.Message.ToolCall{
                 status: :complete,
                 type: :function,
                 name: "custom",
                 arguments: %{"thing" => "hairbrush"},
                 index: nil
               }
             ] = msg.tool_calls
    end
  end

  describe "serialize_config/2" do
    test "does not include the API key or callbacks" do
      model = ChatOllamaAI.new!(%{model: "llama2"})
      result = ChatOllamaAI.serialize_config(model)
      assert result["version"] == 1
      refute Map.has_key?(result, "callbacks")
    end

    test "creates expected map" do
      model =
        ChatOllamaAI.new!(%{
          model: "llama2",
          temperature: 0,
          frequency_penalty: 0.5,
          seed: 123,
          num_gpu: 2,
          stream: true
        })

      result = ChatOllamaAI.serialize_config(model)

      assert result == %{
               "endpoint" => "http://localhost:11434/api/chat",
               "keep_alive" => "5m",
               "mirostat" => 0,
               "mirostat_eta" => 0.1,
               "mirostat_tau" => 5.0,
               "model" => "llama2",
               "module" => "Elixir.LangChain.ChatModels.ChatOllamaAI",
               "num_ctx" => 2048,
               "num_gpu" => 2,
               "num_gqa" => nil,
               "num_predict" => 128,
               "num_thread" => nil,
               "receive_timeout" => 300_000,
               "repeat_last_n" => 64,
               "repeat_penalty" => 1.1,
               "seed" => 123,
               "stop" => nil,
               "stream" => true,
               "temperature" => 0.0,
               "tfs_z" => 1.0,
               "top_k" => 40,
               "top_p" => 0.9,
               "version" => 1
             }
    end
  end
end
