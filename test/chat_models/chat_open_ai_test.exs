defmodule LangChain.ChatModels.ChatOpenAITest do
  use LangChain.BaseCase
  import LangChain.Fixtures
  import LangChain.TestingHelpers

  doctest LangChain.ChatModels.ChatOpenAI
  alias LangChain.ChatModels.ChatOpenAI
  alias LangChain.Function
  alias LangChain.FunctionParam
  alias LangChain.TokenUsage
  alias LangChain.LangChainError
  alias LangChain.Message
  alias LangChain.Message.ContentPart
  alias LangChain.Message.ToolCall
  alias LangChain.Message.ToolResult
  alias LangChain.Chains.LLMChain

  @test_model "gpt-4o-mini-2024-07-18"
  @gpt4 "gpt-4-1106-preview"

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
      assert {:ok, %ChatOpenAI{} = openai} = ChatOpenAI.new(%{"model" => @test_model})
      assert openai.model == @test_model
    end

    test "returns error when invalid" do
      assert {:error, changeset} = ChatOpenAI.new(%{"model" => nil})
      refute changeset.valid?
      assert {"can't be blank", _} = changeset.errors[:model]
    end

    test "supports overriding the API endpoint" do
      override_url = "http://localhost:1234/v1/chat/completions"

      model =
        ChatOpenAI.new!(%{
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
        ChatOpenAI.new(%{
          "model" => @test_model,
          "json_response" => true,
          "json_schema" => json_schema
        })

      assert openai.json_response == true
      assert openai.json_schema == json_schema
    end

    test "supports overriding reasoning_effort" do
      # defaults to "medium"
      %ChatOpenAI{} = openai = ChatOpenAI.new!()
      assert openai.reasoning_effort == "medium"

      # can override the default to "high"
      %ChatOpenAI{} = openai = ChatOpenAI.new!(%{"reasoning_effort" => "high"})
      assert openai.reasoning_effort == "high"
    end
  end

  describe "for_api/3" do
    test "generates a map for an API call" do
      {:ok, openai} =
        ChatOpenAI.new(%{
          "model" => @test_model,
          "temperature" => 1,
          "frequency_penalty" => 0.5,
          "api_key" => "api_key"
        })

      data = ChatOpenAI.for_api(openai, [], [])
      assert data.model == @test_model
      assert data.temperature == 1
      assert data.frequency_penalty == 0.5
      # NOTE: %{"type" => "text"} is the default when not specified
      assert data[:response_format] == nil
    end

    test "generates a map for an API call with JSON response set to true" do
      {:ok, openai} =
        ChatOpenAI.new(%{
          "model" => @test_model,
          "temperature" => 1,
          "frequency_penalty" => 0.5,
          "json_response" => true
        })

      data = ChatOpenAI.for_api(openai, [], [])
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

      {:ok, openai} =
        ChatOpenAI.new(%{
          "model" => @test_model,
          "temperature" => 1,
          "frequency_penalty" => 0.5,
          "json_response" => true,
          "json_schema" => json_schema
        })

      data = ChatOpenAI.for_api(openai, [], [])
      assert data.model == @test_model
      assert data.temperature == 1
      assert data.frequency_penalty == 0.5

      assert data.response_format == %{
               "type" => "json_schema",
               "json_schema" => json_schema
             }
    end

    test "generates a map for an API call with max_tokens set" do
      {:ok, openai} =
        ChatOpenAI.new(%{
          "model" => @test_model,
          "temperature" => 1,
          "frequency_penalty" => 0.5,
          "max_tokens" => 1234
        })

      data = ChatOpenAI.for_api(openai, [], [])
      assert data.model == @test_model
      assert data.temperature == 1
      assert data.frequency_penalty == 0.5
      assert data.max_tokens == 1234
    end

    test "generates a map for an API call with stream_options set correctly" do
      {:ok, openai} =
        ChatOpenAI.new(%{
          model: @test_model,
          stream_options: %{include_usage: true}
        })

      data = ChatOpenAI.for_api(openai, [], [])
      assert data.model == @test_model
      assert data.stream_options == %{"include_usage" => true}
    end

    test "generated a map for an API call with tool_choice set correctly to auto" do
      {:ok, openai} =
        ChatOpenAI.new(%{
          model: @test_model,
          tool_choice: %{"type" => "auto"}
        })

      data = ChatOpenAI.for_api(openai, [], [])
      assert data.model == @test_model
      assert data.tool_choice == "auto"
    end

    test "generated a map for an API call with tool_choice set correctly to a specific function" do
      {:ok, openai} =
        ChatOpenAI.new(%{
          model: @test_model,
          tool_choice: %{"type" => "function", "function" => %{"name" => "set_weather"}}
        })

      data = ChatOpenAI.for_api(openai, [], [])
      assert data.model == @test_model
      assert data.tool_choice == %{"type" => "function", "function" => %{"name" => "set_weather"}}
    end
  end

  describe "for_api/1" do
    test "turns a tool_call into expected JSON format" do
      tool_call =
        ToolCall.new!(%{call_id: "call_abc123", name: "hello_world", arguments: "{}"})

      json = ChatOpenAI.for_api(ChatOpenAI.new!(), tool_call)

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
      # Needed when restoring a conversation from structs for history.
      # args = %{"expression" => "11 + 10"}
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

      json = ChatOpenAI.for_api(ChatOpenAI.new!(), msg)

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

      [json] = ChatOpenAI.for_api(ChatOpenAI.new!(), msg)

      assert json == %{
               "content" => [%{"text" => "Hello World!", "type" => "text"}],
               "tool_call_id" => "tool_abc123",
               "role" => :tool
             }
    end

    test "turns multiple tool results into expected JSON format" do
      # Should generate multiple tool entries.
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

      # ChatGPT expects each tool response to stand alone. This splits them out
      # and returns them individually.
      list = ChatOpenAI.for_api(ChatOpenAI.new!(), message)

      assert is_list(list)

      [r1, r2, r3] = list

      assert r1 == %{
               "content" => [%{"text" => "Hello World!", "type" => "text"}],
               "tool_call_id" => "tool_abc123",
               "role" => :tool
             }

      assert r2 == %{
               "content" => [%{"text" => "Hello", "type" => "text"}],
               "tool_call_id" => "tool_abc234",
               "role" => :tool
             }

      assert r3 == %{
               "content" => [%{"text" => "World!", "type" => "text"}],
               "tool_call_id" => "tool_abc345",
               "role" => :tool
             }
    end

    test "tools work with minimal definition and no parameters", %{hello_world: hello_world} do
      result = ChatOpenAI.for_api(ChatOpenAI.new!(), hello_world)

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

      # result = Function.for_api(fun)
      result = ChatOpenAI.for_api(ChatOpenAI.new!(), fun)

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

      # result = Function.for_api(fun)
      result = ChatOpenAI.for_api(ChatOpenAI.new!(), fun)

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
      # don't try and send an Elixir function ref through to the API
      {:ok, fun} = Function.new(%{"name" => "hello_world", "function" => &hello_world/2})
      # result = Function.for_api(fun)
      result = ChatOpenAI.for_api(ChatOpenAI.new!(), fun)
      refute Map.has_key?(result, "function")
    end
  end

  describe "for_api/2" do
    test "turns a basic user message into the expected JSON format" do
      openai = ChatOpenAI.new!()

      expected = %{"role" => :user, "content" => [%{"type" => "text", "text" => "Hi."}]}
      result = ChatOpenAI.for_api(openai, Message.new_user!("Hi."))
      assert result == expected
    end

    test "includes 'name' when set" do
      openai = ChatOpenAI.new!()

      expected = %{
        "role" => :user,
        "content" => [%{"type" => "text", "text" => "Hi."}],
        "name" => "Harold"
      }

      result =
        ChatOpenAI.for_api(openai, Message.new!(%{role: :user, content: "Hi.", name: "Harold"}))

      assert result == expected
    end

    test "turns an assistant message into expected JSON format" do
      openai = ChatOpenAI.new!()

      # NOTE: Does not include tool_calls if empty
      expected = %{"role" => :assistant, "content" => [%{"type" => "text", "text" => "Hi."}]}

      result =
        ChatOpenAI.for_api(openai, Message.new_assistant!(%{content: "Hi.", tool_calls: []}))

      # TODO: The for_api call is not correctly handling ContentParts and the idea that a message no longer has a string content.
      # TODO: Need a content_parts_for_api. Call that in multiple places.
      assert result == expected
    end

    test "turns an assistant message with text and tool calls into expected JSON format" do
      openai = ChatOpenAI.new!()

      # NOTE: Does not include tool_calls if empty
      expected = %{
        "role" => :assistant,
        "content" =>
          "It seems there was an error, as the response indicates that `a.txt` is present rather than showing the current directory path. Let me try that again.",
        "tool_calls" => [
          %{
            "function" => %{
              "arguments" => "{\"command\":\"pwd\"}",
              "name" => "execute_command"
            },
            "id" => "call_123",
            "type" => "function"
          }
        ]
      }

      result =
        ChatOpenAI.for_api(openai, %Message{
          content:
            "It seems there was an error, as the response indicates that `a.txt` is present rather than showing the current directory path. Let me try that again.",
          status: :complete,
          role: :assistant,
          tool_calls: [
            %ToolCall{
              status: :complete,
              type: :function,
              call_id: "call_123",
              name: "execute_command",
              arguments: %{"command" => "pwd"},
              index: nil
            }
          ],
          tool_results: nil
        })

      assert result == expected
    end

    test "turns a ToolResult into the expected JSON format" do
      openai = ChatOpenAI.new!()
      result = ToolResult.new!(%{tool_call_id: "tool_abc123", content: "Hello World!"})

      json = ChatOpenAI.for_api(openai, result)

      assert json == %{
               "content" => [%{"text" => "Hello World!", "type" => "text"}],
               "tool_call_id" => "tool_abc123",
               "role" => :tool
             }
    end

    test "turns a multi-modal user message into the expected JSON format" do
      openai = ChatOpenAI.new!()

      expected = %{
        "role" => :user,
        "content" => [
          %{"type" => "text", "text" => "Tell me about this image:"},
          %{"type" => "image_url", "image_url" => %{"url" => "url-to-image"}}
        ]
      }

      result =
        ChatOpenAI.for_api(
          openai,
          Message.new_user!([
            ContentPart.text!("Tell me about this image:"),
            ContentPart.image_url!("url-to-image")
          ])
        )

      assert result == expected
    end

    test "turns system role in to developer role based on flag" do
      openai = ChatOpenAI.new!()
      openai_dev = ChatOpenAI.new!(%{reasoning_mode: true})

      assert %{"role" => :system} =
               ChatOpenAI.for_api(openai, Message.new_system!("System prompt!"))

      assert %{"role" => :developer} =
               ChatOpenAI.for_api(openai_dev, Message.new_system!("System prompt!"))
    end
  end

  describe "content_part_for_api/2" do
    test "turns a text ContentPart into the expected JSON format" do
      expected = %{"type" => "text", "text" => "Tell me about this image:"}

      result =
        ChatOpenAI.content_part_for_api(
          ChatOpenAI.new!(),
          ContentPart.text!("Tell me about this image:")
        )

      assert result == expected
    end

    test "turns an image ContentPart into the expected JSON format" do
      expected = %{"type" => "image_url", "image_url" => %{"url" => "image_base64_data"}}

      result =
        ChatOpenAI.content_part_for_api(
          ChatOpenAI.new!(),
          ContentPart.image!("image_base64_data")
        )

      assert result == expected
    end

    test "turns an image ContentPart into the expected JSON format with detail option" do
      expected = %{
        "type" => "image_url",
        "image_url" => %{"url" => "image_base64_data", "detail" => "low"}
      }

      result =
        ChatOpenAI.content_part_for_api(
          ChatOpenAI.new!(),
          ContentPart.image!("image_base64_data", detail: "low")
        )

      assert result == expected
    end

    test "turns ContentPart's media type the expected JSON values" do
      expected = "data:image/jpg;base64,image_base64_data"

      result =
        ChatOpenAI.content_part_for_api(
          ChatOpenAI.new!(),
          ContentPart.image!("image_base64_data", media: :jpg)
        )

      assert %{"image_url" => %{"url" => ^expected}} = result

      expected = "data:image/jpg;base64,image_base64_data"

      result =
        ChatOpenAI.content_part_for_api(
          ChatOpenAI.new!(),
          ContentPart.image!("image_base64_data", media: :jpeg)
        )

      assert %{"image_url" => %{"url" => ^expected}} = result

      expected = "data:image/gif;base64,image_base64_data"

      result =
        ChatOpenAI.content_part_for_api(
          ChatOpenAI.new!(),
          ContentPart.image!("image_base64_data", media: :gif)
        )

      assert %{"image_url" => %{"url" => ^expected}} = result

      expected = "data:image/webp;base64,image_base64_data"

      result =
        ChatOpenAI.content_part_for_api(
          ChatOpenAI.new!(),
          ContentPart.image!("image_base64_data", media: :webp)
        )

      assert %{"image_url" => %{"url" => ^expected}} = result

      expected = "data:image/png;base64,image_base64_data"

      result =
        ChatOpenAI.content_part_for_api(
          ChatOpenAI.new!(),
          ContentPart.image!("image_base64_data", media: :png)
        )

      assert %{"image_url" => %{"url" => ^expected}} = result

      # an string value is passed through
      expected = "data:file/pdf;base64,image_base64_data"

      result =
        ChatOpenAI.content_part_for_api(
          ChatOpenAI.new!(),
          ContentPart.image!("image_base64_data", media: "file/pdf")
        )

      assert %{"image_url" => %{"url" => ^expected}} = result
    end

    test "turns an image_url ContentPart into the expected JSON format" do
      expected = %{"type" => "image_url", "image_url" => %{"url" => "url-to-image"}}

      result =
        ChatOpenAI.content_part_for_api(ChatOpenAI.new!(), ContentPart.image_url!("url-to-image"))

      assert result == expected
    end

    test "turns an image_url ContentPart into the expected JSON format with detail option" do
      expected = %{
        "type" => "image_url",
        "image_url" => %{"url" => "url-to-image", "detail" => "low"}
      }

      result =
        ChatOpenAI.content_part_for_api(
          ChatOpenAI.new!(),
          ContentPart.image_url!("url-to-image", detail: "low")
        )

      assert result == expected
    end

    test "turns a base64 file ContentPart into the expected JSON format" do
      file_base64_data = "some_file_base64_data"
      filename = "my_file.pdf"

      expected = %{
        "type" => "file",
        "file" => %{
          "filename" => filename,
          "file_data" => "data:application/pdf;base64," <> file_base64_data
        }
      }

      result =
        ChatOpenAI.content_part_for_api(
          ChatOpenAI.new!(),
          ContentPart.file!(file_base64_data, media: :pdf, type: :base64, filename: filename)
        )

      assert result == expected
    end

    test "turns a file_id file ContentPart into the expected JSON format" do
      file_id = "file-1234"

      expected = %{
        "type" => "file",
        "file" => %{
          "file_id" => file_id
        }
      }

      result =
        ChatOpenAI.content_part_for_api(
          ChatOpenAI.new!(),
          ContentPart.file!(file_id, media: :pdf, type: :file_id)
        )

      assert result == expected
    end
  end

  describe "call/2" do
    @tag live_call: true, live_open_ai: true
    test "basic content example and fires ratelimit callback" do
      handlers = %{
        on_llm_ratelimit_info: fn headers ->
          send(self(), {:fired_ratelimit_info, headers})
        end
      }

      # https://js.langchain.com/docs/modules/models/chat/
      {:ok, chat} =
        ChatOpenAI.new(%{
          temperature: 1,
          seed: 0,
          stream: false,
          verbose_api: true
        })

      chat = %ChatOpenAI{chat | callbacks: [handlers]}

      {:ok, [%Message{role: :assistant, content: response} = message]} =
        ChatOpenAI.call(chat, [
          Message.new_user!("Return the response 'Colorful Threads'.")
        ])

      IO.inspect(message, label: "MESSAGE")

      assert [%ContentPart{}] = response
      assert ContentPart.parts_to_string(response) =~ "Colorful Threads"

      assert_received {:fired_ratelimit_info, info}

      assert %{
               "x-ratelimit-limit-requests" => _,
               "x-ratelimit-limit-tokens" => _,
               "x-ratelimit-remaining-requests" => _,
               "x-ratelimit-remaining-tokens" => _,
               "x-ratelimit-reset-requests" => _,
               "x-ratelimit-reset-tokens" => _,
               "x-request-id" => _
             } = info
    end

    @tag live_call: true, live_open_ai: true
    test "basic streamed content example's final result and fires ratelimit callback" do
      handlers = %{
        on_llm_ratelimit_info: fn headers ->
          send(self(), {:fired_ratelimit_info, headers})
        end
      }

      # https://js.langchain.com/docs/modules/models/chat/
      {:ok, chat} =
        ChatOpenAI.new(%{temperature: 1, seed: 0, stream: true})

      chat = %ChatOpenAI{chat | callbacks: [handlers]}

      {:ok, result} =
        ChatOpenAI.call(chat, [
          Message.new_user!("Return the response 'Colorful Threads'.")
        ])

      # returns a list of MessageDeltas. A list of a list because it's "n" choices.
      assert result == [
               [
                 %LangChain.MessageDelta{
                   content: "",
                   status: :incomplete,
                   index: 0,
                   role: :assistant
                 }
               ],
               [
                 %LangChain.MessageDelta{
                   content: "Color",
                   status: :incomplete,
                   index: 0,
                   role: :unknown
                 }
               ],
               [
                 %LangChain.MessageDelta{
                   content: "ful",
                   status: :incomplete,
                   index: 0,
                   role: :unknown
                 }
               ],
               [
                 %LangChain.MessageDelta{
                   content: " Threads",
                   status: :incomplete,
                   index: 0,
                   role: :unknown
                 }
               ],
               [
                 %LangChain.MessageDelta{
                   content: nil,
                   status: :complete,
                   index: 0,
                   role: :unknown
                 }
               ]
             ]

      assert_received {:fired_ratelimit_info, info}

      assert %{
               "x-ratelimit-limit-requests" => _,
               "x-ratelimit-limit-tokens" => _,
               "x-ratelimit-remaining-requests" => _,
               "x-ratelimit-remaining-tokens" => _,
               "x-ratelimit-reset-requests" => _,
               "x-ratelimit-reset-tokens" => _,
               "x-request-id" => _
             } = info
    end

    @tag live_call: true, live_open_ai: true
    test "non-streamed response returns token usage" do
      # https://js.langchain.com/docs/modules/models/chat/
      {:ok, chat} =
        ChatOpenAI.new(%{
          temperature: 1,
          seed: 0,
          stream: false
        })

      {:ok, [result]} =
        ChatOpenAI.call(chat, [
          Message.new_user!("Return the response 'Colorful Threads'.")
        ])

      assert result.content == [ContentPart.text!("Colorful Threads")]

      assert %TokenUsage{} = usage = result.metadata.usage
      assert usage.input == 15
      assert usage.output == 3
    end

    @tag live_call: true, live_open_ai: true
    test "executing a function with arguments", %{weather: weather} do
      {:ok, chat} = ChatOpenAI.new(%{seed: 0, stream: false, model: @gpt4})

      {:ok, message} =
        Message.new_user("What is the weather like in Moab Utah?")

      {:ok, [message]} = ChatOpenAI.call(chat, [message], [weather])

      assert %Message{role: :assistant} = message
      assert message.status == :complete
      assert message.role == :assistant
      assert message.content == nil
      [call] = message.tool_calls
      assert call.status == :complete
      assert call.type == :function
      assert call.call_id != nil
      assert call.arguments == %{"city" => "Moab", "state" => "UT"}
    end

    @tag live_call: true, live_open_ai: true
    test "executing a call with tool_choice set as none", %{
      weather: weather,
      hello_world: hello_world
    } do
      {:ok, chat} =
        ChatOpenAI.new(%{seed: 0, stream: false, model: @gpt4, tool_choice: %{"type" => "none"}})

      {:ok, message} =
        Message.new_user("What is the weather like in Moab Utah?")

      {:ok, [message]} = ChatOpenAI.call(chat, [message], [weather, hello_world])

      assert %Message{role: :assistant} = message
      assert message.status == :complete
      assert message.role == :assistant
      assert message.content != nil
      assert message.tool_calls == []
    end

    @tag live_call: true, live_open_ai: true
    test "executing a call with required tool_choice", %{
      weather: weather,
      hello_world: hello_world
    } do
      {:ok, chat} =
        ChatOpenAI.new(%{
          seed: 0,
          stream: false,
          model: @gpt4,
          tool_choice: %{"type" => "function", "function" => %{"name" => "get_weather"}}
        })

      {:ok, message} =
        Message.new_user("What is the weather like in Moab Utah?")

      {:ok, [message]} = ChatOpenAI.call(chat, [message], [weather, hello_world])

      assert %Message{role: :assistant} = message
      assert message.status == :complete
      assert message.role == :assistant
      assert [%LangChain.Message.ToolCall{} = tool_call] = message.tool_calls
      assert tool_call.name == "get_weather"
      assert tool_call.type == :function
      assert tool_call.status == :complete
      assert is_map(tool_call.arguments)
    end

    @tag live_call: true, live_open_ai: true
    test "LIVE: supports receiving multiple tool calls in a single response", %{weather: weather} do
      {:ok, chat} =
        ChatOpenAI.new(%{
          seed: 0,
          stream: false,
          model: @gpt4
        })

      {:ok, message} =
        Message.new_user(
          "What is the weather like in Moab Utah, Portland Oregon, and Baltimore MD? Explain your thought process."
        )

      {:ok, [message]} = ChatOpenAI.call(chat, [message], [weather])

      assert %Message{role: :assistant} = message
      assert message.status == :complete
      assert message.content == nil
      [call1, call2, call3] = message.tool_calls

      assert call1.status == :complete
      assert call1.type == :function
      assert call1.name == "get_weather"
      assert call1.arguments == %{"city" => "Moab", "state" => "UT"}

      assert call2.name == "get_weather"
      assert call2.arguments == %{"city" => "Portland", "state" => "OR"}

      assert call3.name == "get_weather"
      assert call3.arguments == %{"city" => "Baltimore", "state" => "MD"}
    end

    @tag live_call: true, live_open_ai: true
    test "executes callback function when data is NOT streamed" do
      handler = %{
        on_llm_new_message: fn %Message{} = new_message ->
          send(self(), {:message_received, new_message})
        end
      }

      # https://js.langchain.com/docs/modules/models/chat/
      # NOTE streamed. Should receive complete message.
      {:ok, chat} =
        ChatOpenAI.new(%{seed: 0, temperature: 1, stream: false})

      chat = %ChatOpenAI{chat | callbacks: [handler]}

      {:ok, [message]} =
        ChatOpenAI.call(
          chat,
          [
            Message.new_user!("Return the response 'Hi'.")
          ],
          []
        )

      assert [%ContentPart{}] = message.content
      assert ContentPart.parts_to_string(message.content) =~ "Hi"
      assert message.index == 0
      assert_receive {:message_received, received_item}, 500
      assert %Message{} = received_item
      assert received_item.role == :assistant
      assert [%ContentPart{}] = received_item.content
      assert ContentPart.parts_to_string(received_item.content) =~ "Hi"
      assert received_item.index == 0
    end

    @tag live_call: true, live_open_ai: true
    test "handles when request is too large" do
      {:ok, chat} =
        ChatOpenAI.new(%{model: "gpt-4-0613", seed: 0, stream: false, temperature: 1})

      {:error, %LangChainError{} = reason} = ChatOpenAI.call(chat, [too_large_user_request()])
      assert reason.type == nil
      assert reason.message =~ "maximum context length"
    end

    @tag live_call: true, live_azure: true
    test "supports Azure hosted OpenAI models" do
      # https://learn.microsoft.com/en-us/azure/ai-services/openai/chatgpt-quickstart?tabs=command-line%2Cjavascript-keyless%2Ctypescript-keyless%2Cpython-new&pivots=rest-api

      endpoint = System.fetch_env!("AZURE_OPENAI_ENDPOINT")
      api_key = System.fetch_env!("AZURE_OPENAI_KEY")

      {:ok, chat} =
        ChatOpenAI.new(%{
          endpoint: endpoint,
          api_key: api_key,
          seed: 0,
          temperature: 1,
          stream: false
        })

      {:ok, [message]} =
        ChatOpenAI.call(
          chat,
          [
            Message.new_user!("Return the response 'Hi'.")
          ],
          []
        )

      assert [%ContentPart{}] = message.content
      assert ContentPart.parts_to_string(message.content) =~ "Hi"
      assert message.role == :assistant
      assert message.index == 0
    end
  end

  describe "use in LLMChain" do
    @tag live_call: true, live_open_ai: true
    test "NOT STREAMED with callbacks and token usage" do
      handler = %{
        on_llm_new_delta: fn %LLMChain{} = _chain, deltas ->
          send(self(), {:test_stream_deltas, deltas})
        end,
        on_message_processed: fn _chain, message ->
          send(self(), {:test_message_processed, message})
        end
      }

      # We can construct an LLMChain from a PromptTemplate and an LLM.
      model = ChatOpenAI.new!(%{temperature: 1, seed: 0, stream: false})

      {:ok, updated_chain} =
        %{llm: model}
        |> LLMChain.new!()
        |> LLMChain.add_callback(handler)
        |> LLMChain.add_messages([
          Message.new_user!("Suggest one good name for a company that makes colorful socks?")
        ])
        |> LLMChain.run()

      assert %Message{role: :assistant, status: :complete} = updated_chain.last_message
      assert %TokenUsage{input: 20} = updated_chain.last_message.metadata.usage

      assert_received {:test_message_processed, message}
      assert %Message{role: :assistant} = message
      # the final returned message should match the callback message
      assert message == updated_chain.last_message
      # we should have received the final combined message
      refute_received {:test_stream_deltas, _delta}
    end

    @tag live_call: true, live_open_ai: true
    test "STREAMED with callbacks and token usage" do
      handler = %{
        on_llm_new_delta: fn %LLMChain{} = _chain, deltas ->
          send(self(), deltas)
        end,
        on_message_processed: fn _chain, message ->
          send(self(), {:test_message_processed, message})
        end
      }

      # We can construct an LLMChain from a PromptTemplate and an LLM.
      model =
        ChatOpenAI.new!(%{
          temperature: 1,
          seed: 0,
          stream: true,
          stream_options: %{include_usage: true}
        })

      original_chain =
        %{llm: model}
        |> LLMChain.new!()
        |> LLMChain.add_callback(handler)
        |> LLMChain.add_messages([
          Message.new_user!("Suggest one good name for a company that makes colorful socks?")
        ])

      {:ok, updated_chain} = original_chain |> LLMChain.run()

      assert %Message{role: :assistant} = updated_chain.last_message
      assert %TokenUsage{input: 20} = updated_chain.last_message.metadata.usage

      assert_received {:test_message_processed, message}
      assert %Message{role: :assistant} = message
      # the final returned message should match the callback message
      assert message == updated_chain.last_message

      # get all the deltas sent to the test process
      deltas = collect_messages() |> List.flatten()

      # apply the deltas to the original chain
      delta_merged_chain = LLMChain.apply_deltas(original_chain, deltas)

      # the received merged deltas should match the ones assembled by the chain.
      # This is also verifying that we're receiving the token usage via sent
      # deltas.
      assert delta_merged_chain.last_message == updated_chain.last_message
    end
  end

  describe "do_process_response/2" do
    setup do
      model = ChatOpenAI.new!(%{"model" => @test_model})
      %{model: model}
    end

    test "returns skip when given an empty choices list", %{model: model} do
      assert :skip == ChatOpenAI.do_process_response(model, %{"choices" => []})
    end

    test "handles receiving a message", %{model: model} do
      response = %{
        "message" => %{"role" => "assistant", "content" => "Greetings!"},
        "finish_reason" => "stop",
        "index" => 1
      }

      assert %Message{} = struct = ChatOpenAI.do_process_response(model, response)
      assert struct.role == :assistant
      assert struct.content == [ContentPart.text!("Greetings!")]
      assert struct.index == 1
      assert struct.metadata == nil
    end

    test "handles receiving a message with token usage information", %{model: model} do
      response = %{
        "choices" => [
          %{
            "finish_reason" => "stop",
            "index" => 0,
            "logprobs" => nil,
            "message" => %{
              "annotations" => [],
              "content" => "Colorful Threads",
              "refusal" => nil,
              "role" => "assistant"
            }
          }
        ],
        "created" => 1_745_192_205,
        "id" => "chatcmpl-BOYVJArISYBhZWbEoLVBNu0DOamHi",
        "model" => "gpt-3.5-turbo-0125",
        "object" => "chat.completion",
        "service_tier" => "default",
        "system_fingerprint" => nil,
        "usage" => %{
          "completion_tokens" => 4,
          "completion_tokens_details" => %{
            "accepted_prediction_tokens" => 0,
            "audio_tokens" => 0,
            "reasoning_tokens" => 0,
            "rejected_prediction_tokens" => 0
          },
          "prompt_tokens" => 15,
          "prompt_tokens_details" => %{"audio_tokens" => 0, "cached_tokens" => 0},
          "total_tokens" => 19
        }
      }

      assert [%Message{} = struct] = ChatOpenAI.do_process_response(model, response)
      assert struct.role == :assistant
      assert struct.content == [ContentPart.text!("Colorful Threads")]
      assert struct.index == 0
      # token usage attached to metadata
      %TokenUsage{} = usage = struct.metadata.usage
      assert usage.input == 15
      assert usage.output == 4

      assert usage.raw == %{
               "completion_tokens" => 4,
               "completion_tokens_details" => %{
                 "accepted_prediction_tokens" => 0,
                 "audio_tokens" => 0,
                 "reasoning_tokens" => 0,
                 "rejected_prediction_tokens" => 0
               },
               "prompt_tokens" => 15,
               "prompt_tokens_details" => %{"audio_tokens" => 0, "cached_tokens" => 0},
               "total_tokens" => 19
             }
    end

    test "handles receiving the final empty streamed delta with token usage information", %{
      model: model
    } do
      response = %{
        "choices" => [],
        "created" => 1_750_622_279,
        "id" => "chatcmpl-BlL79wvCX5gewO44UgZN1ul0wwU5j",
        "model" => "gpt-3.5-turbo-0125",
        "object" => "chat.completion.chunk",
        "service_tier" => "default",
        "system_fingerprint" => nil,
        "usage" => %{
          "completion_tokens" => 3,
          "completion_tokens_details" => %{
            "accepted_prediction_tokens" => 0,
            "audio_tokens" => 0,
            "reasoning_tokens" => 0,
            "rejected_prediction_tokens" => 0
          },
          "prompt_tokens" => 15,
          "prompt_tokens_details" => %{"audio_tokens" => 0, "cached_tokens" => 0},
          "total_tokens" => 18
        }
      }

      assert result = ChatOpenAI.do_process_response(model, response)

      assert %TokenUsage{
               input: 15,
               output: 3,
               raw: %{
                 "completion_tokens" => 3,
                 "completion_tokens_details" => %{
                   "accepted_prediction_tokens" => 0,
                   "audio_tokens" => 0,
                   "reasoning_tokens" => 0,
                   "rejected_prediction_tokens" => 0
                 },
                 "prompt_tokens" => 15,
                 "prompt_tokens_details" => %{"audio_tokens" => 0, "cached_tokens" => 0},
                 "total_tokens" => 18
               }
             } == result
    end

    test "handles receiving a single tool_calls message", %{model: model} do
      response = %{
        "finish_reason" => "tool_calls",
        "index" => 0,
        "logprobs" => nil,
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

      assert %Message{} = struct = ChatOpenAI.do_process_response(model, response)

      assert struct.role == :assistant

      assert [%ToolCall{} = call] = struct.tool_calls
      assert call.call_id == "call_mMSPuyLd915TQ9bcrk4NvLDX"
      assert call.type == :function
      assert call.name == "get_weather"
      assert call.arguments == %{"city" => "Moab", "state" => "UT"}
      assert struct.index == 0
    end

    test "handles receiving a nil tool_calls message", %{model: model} do
      response = %{
        "finish_reason" => "tool_calls",
        "index" => 0,
        "logprobs" => nil,
        "message" => %{
          "content" => nil,
          "role" => "assistant",
          "tool_calls" => nil
        }
      }

      assert %Message{} = struct = ChatOpenAI.do_process_response(model, response)

      assert struct.role == :assistant

      assert [] = struct.tool_calls
    end

    test "handles receiving multiple tool_calls messages", %{model: model} do
      response = %{
        "finish_reason" => "tool_calls",
        "index" => 0,
        "logprobs" => nil,
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
            },
            %{
              "function" => %{
                "arguments" => "{\"city\": \"Baltimore\", \"state\": \"MD\"}",
                "name" => "get_weather"
              },
              "id" => "call_G17PCZZBTyK0gwpzIzD4OBep",
              "type" => "function"
            }
          ]
        }
      }

      assert %Message{} = struct = ChatOpenAI.do_process_response(model, response)

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
               }),
               ToolCall.new!(%{
                 type: :function,
                 call_id: "call_G17PCZZBTyK0gwpzIzD4OBep",
                 name: "get_weather",
                 arguments: %{"city" => "Baltimore", "state" => "MD"},
                 status: :complete
               })
             ]
    end

    test "handles receiving multiple tool_calls and one has invalid JSON", %{model: model} do
      response = %{
        "finish_reason" => "tool_calls",
        "index" => 0,
        "logprobs" => nil,
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
                "arguments" => "{\"invalid\"}",
                "name" => "get_weather"
              },
              "id" => "call_ylRu5SPegST9tppLEj6IJ0Rs",
              "type" => "function"
            }
          ]
        }
      }

      assert {:error, %LangChainError{} = reason} =
               ChatOpenAI.do_process_response(model, response)

      assert reason.type == "changeset"
      assert reason.message == "tool_calls: arguments: invalid json"
    end

    test "handles a single tool_call from list", %{model: model} do
      call = %{
        "function" => %{
          "arguments" => "{\"city\": \"Moab\", \"state\": \"UT\"}",
          "name" => "get_weather"
        },
        "id" => "call_4L8NfePhSW8PdoHUWkvhzguu",
        "type" => "function"
      }

      assert %ToolCall{} = call = ChatOpenAI.do_process_response(model, call)
      assert call.type == :function
      assert call.status == :complete
      assert call.call_id == "call_4L8NfePhSW8PdoHUWkvhzguu"
      assert call.name == "get_weather"
      assert call.arguments == %{"city" => "Moab", "state" => "UT"}
    end

    test "handles receiving a tool_call with invalid JSON", %{model: model} do
      call = %{
        "function" => %{
          "arguments" => "{\"invalid\"}",
          "name" => "get_weather"
        },
        "id" => "call_4L8NfePhSW8PdoHUWkvhzguu",
        "type" => "function"
      }

      assert {:error, %LangChainError{} = error} = ChatOpenAI.do_process_response(model, call)

      assert error.type == "changeset"
      assert error.message == "arguments: invalid json"
    end

    test "handles streamed deltas for multiple tool calls", %{model: model} do
      deltas =
        Enum.map(
          get_streamed_deltas_multiple_tool_calls(),
          &ChatOpenAI.do_process_response(model, &1)
        )

      combined =
        deltas
        |> List.flatten()
        |> Enum.reduce(nil, &MessageDelta.merge_delta(&2, &1))

      expected = %MessageDelta{
        content: nil,
        status: :complete,
        index: 0,
        role: :assistant,
        tool_calls: [
          %ToolCall{
            status: :incomplete,
            type: :function,
            call_id: "call_fFRRtPwaroz9wbs2eWR7dpcW",
            name: "get_weather",
            arguments: "{\"city\": \"Moab\", \"state\": \"UT\"}",
            index: 0
          },
          %ToolCall{
            status: :incomplete,
            type: :function,
            call_id: "call_sEmznyM1sGqYQ4dbNGdubmxa",
            name: "get_weather",
            arguments: "{\"city\": \"Portland\", \"state\": \"OR\"}",
            index: 1
          },
          %ToolCall{
            status: :incomplete,
            type: :function,
            call_id: "call_cPufqMGm4TOFtqiqFPfz7pcp",
            name: "get_weather",
            arguments: "{\"city\": \"Baltimore\", \"state\": \"MD\"}",
            index: 2
          }
        ]
      }

      assert combined == expected
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

      assert %Message{} = struct = ChatOpenAI.do_process_response(model, response)

      assert struct.role == :assistant
      assert struct.content == [ContentPart.text!("Some of the response that was abruptly")]
      assert struct.index == 0
      assert struct.status == :length
    end

    test "handles receiving a delta message for a content message at different parts", %{
      model: model
    } do
      delta_content = LangChain.Fixtures.raw_deltas_for_content()

      msg_1 = Enum.at(delta_content, 0)
      msg_2 = Enum.at(delta_content, 1)
      msg_10 = Enum.at(delta_content, 10)

      expected_1 = %MessageDelta{
        content: "",
        index: 0,
        role: :assistant,
        status: :incomplete
      }

      [%MessageDelta{} = delta_1] = ChatOpenAI.do_process_response(model, msg_1)
      assert delta_1 == expected_1

      expected_2 = %MessageDelta{
        content: "Hello",
        index: 0,
        role: :unknown,
        status: :incomplete
      }

      [%MessageDelta{} = delta_2] = ChatOpenAI.do_process_response(model, msg_2)
      assert delta_2 == expected_2

      expected_10 = %MessageDelta{
        content: nil,
        index: 0,
        role: :unknown,
        status: :complete
      }

      [%MessageDelta{} = delta_10] = ChatOpenAI.do_process_response(model, msg_10)
      assert delta_10 == expected_10
    end

    test "handles json parse error from server", %{model: model} do
      {:error, %LangChainError{} = error} =
        ChatOpenAI.do_process_response(model, Jason.decode("invalid json"))

      assert error.type == "invalid_json"
      assert "Received invalid JSON: " <> _ = error.message
    end

    test "handles unexpected response", %{model: model} do
      {:error, %LangChainError{} = error} =
        ChatOpenAI.do_process_response(model, "unexpected")

      assert error.type == nil
      assert error.message == "Unexpected response"
    end

    test "return multiple responses when given multiple choices", %{model: model} do
      # received multiple responses because multiples were requested.
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

      [msg1, msg2] = ChatOpenAI.do_process_response(model, response)
      assert %Message{role: :assistant, index: 0} = msg1
      assert %Message{role: :assistant, index: 1} = msg2
      assert msg1.content == [ContentPart.text!("Greetings!")]
      assert msg2.content == [ContentPart.text!("Howdy!")]
    end
  end

  describe "streaming examples" do
    @tag live_call: true, live_open_ai: true
    test "supports streaming response calling function with args" do
      handler = %{
        on_llm_new_delta: fn %MessageDelta{} = data ->
          # IO.inspect(data, label: "DATA")
          send(self(), {:streamed_fn, data})
        end
      }

      {:ok, chat} = ChatOpenAI.new(%{seed: 0, stream: true})

      chat = %ChatOpenAI{chat | callbacks: [handler]}

      {:ok, message} =
        Message.new_user("Answer the following math question: What is 100 + 300 - 200?")

      _response =
        ChatOpenAI.do_api_request(chat, [message], [LangChain.Tools.Calculator.new!()])

      # IO.inspect(response, label: "OPEN AI POST RESPONSE")

      assert_receive {:streamed_fn, received_data}, 300
      assert %MessageDelta{} = received_data
      assert received_data.role == :assistant
      assert received_data.index == 0
    end

    @tag live_call: true, live_open_ai: true
    test "STREAMING handles receiving an error when no messages sent" do
      chat = ChatOpenAI.new!(%{seed: 0, stream: true})

      {:error, %LangChainError{} = reason} = ChatOpenAI.call(chat, [], [])

      assert reason.type == nil

      assert reason.message ==
               "Invalid 'messages': empty array. Expected an array with minimum length 1, but got an empty array instead."
    end

    @tag live_call: true, live_open_ai: true
    test "STREAMING handles receiving a timeout error" do
      handler = %{
        on_llm_new_delta: fn %MessageDelta{} = data ->
          send(self(), {:streamed_fn, data})
        end
      }

      chat =
        ChatOpenAI.new!(%{seed: 0, stream: true, receive_timeout: 50})

      chat = %ChatOpenAI{chat | callbacks: [handler]}

      {:error, %LangChainError{} = reason} =
        ChatOpenAI.call(chat, [Message.new_user!("Why is the sky blue?")], [])

      assert reason.type == "timeout"
      assert reason.message == "Request timed out"
    end
  end

  def setup_expected_json(_) do
    json_1 = %{
      "choices" => [
        %{
          "delta" => %{
            "content" => nil,
            "function_call" => %{"arguments" => "", "name" => "calculator"},
            "role" => "assistant"
          },
          "finish_reason" => nil,
          "index" => 0
        }
      ],
      "created" => 1_689_801_995,
      "id" => "chatcmpl-7e8yp1xBhriNXiqqZ0xJkgNrmMuGS",
      "model" => "gpt-4-0613",
      "object" => "chat.completion.chunk"
    }

    json_2 = %{
      "choices" => [
        %{
          "delta" => %{"function_call" => %{"arguments" => "{\n"}},
          "finish_reason" => nil,
          "index" => 0
        }
      ],
      "created" => 1_689_801_995,
      "id" => "chatcmpl-7e8yp1xBhriNXiqqZ0xJkgNrmMuGS",
      "model" => "gpt-4-0613",
      "object" => "chat.completion.chunk"
    }

    %{json_1: json_1, json_2: json_2}
  end

  describe "decode_stream/2" do
    setup :setup_expected_json

    test "correctly handles fully formed chat completion chunks", %{
      json_1: json_1,
      json_2: json_2
    } do
      data =
        "data: {\"id\":\"chatcmpl-7e8yp1xBhriNXiqqZ0xJkgNrmMuGS\",\"object\":\"chat.completion.chunk\",\"created\":1689801995,\"model\":\"gpt-4-0613\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":null,\"function_call\":{\"name\":\"calculator\",\"arguments\":\"\"}},\"finish_reason\":null}]}\n\n
         data: {\"id\":\"chatcmpl-7e8yp1xBhriNXiqqZ0xJkgNrmMuGS\",\"object\":\"chat.completion.chunk\",\"created\":1689801995,\"model\":\"gpt-4-0613\",\"choices\":[{\"index\":0,\"delta\":{\"function_call\":{\"arguments\":\"{\\n\"}},\"finish_reason\":null}]}\n\n"

      {parsed, incomplete} = ChatOpenAI.decode_stream({data, ""})

      # nothing incomplete. Parsed 2 objects.
      assert incomplete == ""
      assert parsed == [json_1, json_2]
    end

    test "correctly parses when data content contains spaces such as python code with indentation" do
      data =
        "data: {\"id\":\"chatcmpl-7e8yp1xBhriNXiqqZ0xJkgNrmMuGS\",\"object\":\"chat.completion.chunk\",\"created\":1689801995,\"model\":\"gpt-4-0613\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"def my_function(x):\\n    return x + 1\"},\"finish_reason\":null}]}\n\n"

      {parsed, incomplete} = ChatOpenAI.decode_stream({data, ""})

      assert incomplete == ""

      assert parsed == [
               %{
                 "id" => "chatcmpl-7e8yp1xBhriNXiqqZ0xJkgNrmMuGS",
                 "object" => "chat.completion.chunk",
                 "created" => 1_689_801_995,
                 "model" => "gpt-4-0613",
                 "choices" => [
                   %{
                     "index" => 0,
                     "delta" => %{"content" => "def my_function(x):\n    return x + 1"},
                     "finish_reason" => nil
                   }
                 ]
               }
             ]
    end

    test "correctly parses when data split over received messages", %{json_1: json_1} do
      # split the data over multiple messages
      data =
        "data: {\"id\":\"chatcmpl-7e8yp1xBhriNXiqqZ0xJkgNrmMuGS\",\"object\":\"chat.comple
         data: tion.chunk\",\"created\":1689801995,\"model\":\"gpt-4-0613\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":null,\"function_call\":{\"name\":\"calculator\",\"arguments\":\"\"}},\"finish_reason\":null}]}\n\n"

      {parsed, incomplete} = ChatOpenAI.decode_stream({data, ""})

      # nothing incomplete. Parsed 1 object.
      assert incomplete == ""
      assert parsed == [json_1]
    end

    test "correctly parses when data split over decode calls", %{json_1: json_1} do
      buffered = "{\"id\":\"chatcmpl-7e8yp1xBhriNXiqqZ0xJkgNrmMuGS\",\"object\":\"chat.comple"

      # incomplete message chunk processed in next call
      data =
        "data: tion.chunk\",\"created\":1689801995,\"model\":\"gpt-4-0613\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":null,\"function_call\":{\"name\":\"calculator\",\"arguments\":\"\"}},\"finish_reason\":null}]}\n\n"

      {parsed, incomplete} = ChatOpenAI.decode_stream({data, buffered})

      # nothing incomplete. Parsed 1 object.
      assert incomplete == ""
      assert parsed == [json_1]
    end

    test "correctly parses when messages are split in the middle of \"data:\"" do
      # message that ends in the middle of "data: "
      buffered =
        "{\"id\":\"chatcmpl-9mB4I7Cec88xzrOc7wEoxtKWyYczS\",\"object\":\"chat.completion.chunk\",\"created\":1721269318,\"model\":\"gpt-4o-2024-05-13\",\"system_fingerprint\":\"fp_c4e5b6fa31\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\" jointly\"},\"logprobs\":null,\"finish_reason\":null}]}\n\nd"

      # incomplete message chunk starting with the remaining chars of "data: "
      data =
        "ata: {\"id\":\"chatcmpl-9mB4I7Cec88xzrOc7wEoxtKWyYczS\",\"object\":\"chat.completion.chunk\",\"created\":1721269318,\"model\":\"gpt-4o-2024-05-13\",\"system_fingerprint\":\"fp_c4e5b6fa31\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\" with\"},\"logprobs\":null,\"finish_reason\":null}]}"

      {parsed, incomplete} = ChatOpenAI.decode_stream({data, buffered})

      [message_1, message_2] =
        (buffered <> data) |> String.split("data: ") |> Enum.map(&Jason.decode!/1)

      # nothing incomplete. Parsed 2 objects.
      assert incomplete == ""
      assert parsed == [message_1, message_2]
    end

    test "correctly parses when data previously buffered and responses split and has leftovers",
         %{json_1: json_1, json_2: json_2} do
      buffered = "{\"id\":\"chatcmpl-7e8yp1xBhriNXiqqZ0xJkgNrmMuGS\",\"object\":\"chat.comple"

      # incomplete message chunk processed in next call
      data =
        "data: tion.chunk\",\"created\":1689801995,\"model\":\"gpt-4-0613\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":null,\"function_call\":{\"name\":\"calculator\",\"arguments\":\"\"}},\"finish_reason\":null}]}\n\n
         data: {\"id\":\"chatcmpl-7e8yp1xBhriNXiqqZ0xJkgNrmMuGS\",\"object\":\"chat.completion.chunk\",\"crea
         data: ted\":1689801995,\"model\":\"gpt-4-0613\",\"choices\":[{\"index\":0,\"delta\":{\"function_call\":{\"argu
         data: ments\":\"{\\n\"}},\"finish_reason\":null}]}\n\n
         data: {\"id\":\"chatcmpl-7e8yp1xBhriNXiqqZ0xJkgNrmMuGS\",\"object\":\"chat.comp"

      {parsed, incomplete} = ChatOpenAI.decode_stream({data, buffered})

      # nothing incomplete. Parsed 1 object.
      assert incomplete ==
               "{\"id\":\"chatcmpl-7e8yp1xBhriNXiqqZ0xJkgNrmMuGS\",\"object\":\"chat.comp"

      assert parsed == [json_1, json_2]
    end
  end

  describe "image vision using message parts" do
    @tag live_call: true, live_open_ai: true
    test "supports multi-modal user message with image prompt" do
      # https://platform.openai.com/docs/guides/vision
      {:ok, chat} = ChatOpenAI.new(%{model: "gpt-4o-2024-08-06", seed: 0})

      url =
        "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg"

      message =
        Message.new_user!([
          ContentPart.text!("Identify what this is a picture of:"),
          ContentPart.image_url!(url)
        ])

      {:ok, [response]} = ChatOpenAI.call(chat, [message], [])

      assert %Message{role: :assistant} = response
      assert String.contains?(ContentPart.parts_to_string(response.content), "boardwalk")
      assert String.contains?(ContentPart.parts_to_string(response.content), "grass")
    end
  end

  describe "do_process_response - MessageDeltas" do
    setup do
      model = ChatOpenAI.new(%{"model" => @test_model})
      %{model: model}
    end

    test "parses basic text delta", %{model: model} do
      [d1, d2, d3, d4] = get_streamed_deltas_basic_text()

      [delta1] = ChatOpenAI.do_process_response(model, d1)

      assert %MessageDelta{
               role: :assistant,
               content: "",
               status: :incomplete,
               index: 0
             } = delta1

      [delta2] = ChatOpenAI.do_process_response(model, d2)

      assert %MessageDelta{
               role: :unknown,
               content: "Colorful",
               status: :incomplete,
               index: 0
             } = delta2

      [delta3] = ChatOpenAI.do_process_response(model, d3)

      assert %MessageDelta{
               role: :unknown,
               content: " Threads",
               status: :incomplete,
               index: 0
             } = delta3

      [delta4] = ChatOpenAI.do_process_response(model, d4)

      assert %MessageDelta{
               role: :unknown,
               content: nil,
               status: :complete,
               index: 0
             } = delta4
    end

    test "parses initial tool call delta message correctly", %{model: model} do
      raw_delta = %{
        "delta" => %{
          "content" => nil,
          "role" => "assistant",
          "tool_calls" => [
            %{
              "function" => %{"arguments" => "", "name" => "find_by_code"},
              "id" => "call_567",
              "index" => 0,
              "type" => "function"
            }
          ]
        },
        "finish_reason" => nil,
        "index" => 0
      }

      %MessageDelta{} = delta = ChatOpenAI.do_process_response(model, raw_delta)
      assert delta.content == nil
      assert delta.role == :assistant
      assert [%ToolCall{} = call] = delta.tool_calls
      assert call.call_id == "call_567"
      assert call.index == 0
      assert call.type == :function
      assert call.arguments == nil
    end

    test "parses individual tool_calls in a delta message", %{model: model} do
      # chunk 1
      tool_call_response = %{
        "function" => %{"arguments" => "", "name" => "get_weather"},
        "id" => "call_1234",
        "index" => 0,
        "type" => "function"
      }

      assert %ToolCall{} = call = ChatOpenAI.do_process_response(model, tool_call_response)
      assert call.status == :incomplete
      assert call.type == :function
      assert call.name == "get_weather"
      assert call.arguments == nil
      assert call.index == 0

      # chunk 2
      tool_call_response = %{
        "function" => %{"arguments" => "{\"city\": \"Moab\", "},
        "index" => 0
      }

      assert %ToolCall{} = call = ChatOpenAI.do_process_response(model, tool_call_response)
      assert call.status == :incomplete
      assert call.type == :function
      assert call.name == nil
      assert call.arguments == "{\"city\": \"Moab\", "
      assert call.index == 0
    end

    test "parses a MessageDelta with tool_calls", %{model: model} do
      response = get_streamed_deltas_multiple_tool_calls()
      [d1, d2, d3 | _rest] = response
      last = List.last(response)

      assert [%MessageDelta{} = delta1] = ChatOpenAI.do_process_response(model, d1)
      assert delta1.role == :assistant
      assert delta1.status == :incomplete
      assert delta1.content == nil
      assert delta1.index == 0
      assert delta1.tool_calls == nil

      assert [%MessageDelta{} = delta2] = ChatOpenAI.do_process_response(model, d2)
      assert delta2.role == :unknown
      assert delta2.status == :incomplete
      assert delta2.content == nil
      assert delta2.index == 0

      expected_call =
        ToolCall.new!(%{
          call_id: "call_fFRRtPwaroz9wbs2eWR7dpcW",
          index: 0,
          type: :function,
          status: :incomplete,
          name: "get_weather",
          arguments: nil
        })

      assert [expected_call] == delta2.tool_calls

      assert [%MessageDelta{} = delta3] = ChatOpenAI.do_process_response(model, d3)
      assert delta3.role == :unknown
      assert delta3.status == :incomplete
      assert delta3.content == nil
      assert delta3.index == 0

      expected_call =
        ToolCall.new!(%{
          id: nil,
          index: 0,
          type: :function,
          status: :incomplete,
          name: nil,
          arguments: "{\"ci"
        })

      assert [expected_call] == delta3.tool_calls

      assert [%MessageDelta{} = delta4] = ChatOpenAI.do_process_response(model, last)
      assert delta4.role == :unknown
      assert delta4.status == :complete
      assert delta4.content == nil
      assert delta4.index == 0
      assert delta4.tool_calls == nil
    end
  end

  def get_streamed_deltas_basic_text do
    [
      %{
        "choices" => [
          %{
            "delta" => %{"content" => "", "role" => "assistant"},
            "finish_reason" => nil,
            "index" => 0,
            "logprobs" => nil
          }
        ],
        "created" => 1_712_610_022,
        "id" => "chatcmpl-9BqOELc5ktxtKeK1BtTBG2t0aaDty",
        "model" => "gpt-3.5-turbo-0125",
        "object" => "chat.completion.chunk",
        "system_fingerprint" => "fp_b28b39ffa8"
      },
      %{
        "choices" => [
          %{
            "delta" => %{"content" => "Colorful"},
            "finish_reason" => nil,
            "index" => 0,
            "logprobs" => nil
          }
        ],
        "created" => 1_712_610_022,
        "id" => "chatcmpl-9BqOELc5ktxtKeK1BtTBG2t0aaDty",
        "model" => "gpt-3.5-turbo-0125",
        "object" => "chat.completion.chunk",
        "system_fingerprint" => "fp_b28b39ffa8"
      },
      %{
        "choices" => [
          %{
            "delta" => %{"content" => " Threads"},
            "finish_reason" => nil,
            "index" => 0,
            "logprobs" => nil
          }
        ],
        "created" => 1_712_610_022,
        "id" => "chatcmpl-9BqOELc5ktxtKeK1BtTBG2t0aaDty",
        "model" => "gpt-3.5-turbo-0125",
        "object" => "chat.completion.chunk",
        "system_fingerprint" => "fp_b28b39ffa8"
      },
      %{
        "choices" => [
          %{
            "delta" => %{},
            "finish_reason" => "stop",
            "index" => 0,
            "logprobs" => nil
          }
        ],
        "created" => 1_712_610_022,
        "id" => "chatcmpl-9BqOELc5ktxtKeK1BtTBG2t0aaDty",
        "model" => "gpt-3.5-turbo-0125",
        "object" => "chat.completion.chunk",
        "system_fingerprint" => "fp_b28b39ffa8"
      }
    ]
  end

  def get_streamed_deltas_multiple_tool_calls() do
    # NOTE: these are artificially condensed for brevity.

    [
      %{
        "choices" => [
          %{
            "delta" => %{"content" => nil, "role" => "assistant"},
            "finish_reason" => nil,
            "index" => 0,
            "logprobs" => nil
          }
        ],
        "created" => 1_712_606_513,
        "id" => "chatcmpl-9BpTdyN883PR9yKpIK2wYYypgWw1Q",
        "model" => "gpt-4-1106-preview",
        "object" => "chat.completion.chunk",
        "system_fingerprint" => "fp_d6526cacfe"
      },
      %{
        "choices" => [
          %{
            "delta" => %{
              "tool_calls" => [
                %{
                  "function" => %{"arguments" => "", "name" => "get_weather"},
                  "id" => "call_fFRRtPwaroz9wbs2eWR7dpcW",
                  "index" => 0,
                  "type" => "function"
                }
              ]
            },
            "finish_reason" => nil,
            "index" => 0,
            "logprobs" => nil
          }
        ],
        "created" => 1_712_606_513,
        "id" => "chatcmpl-9BpTdyN883PR9yKpIK2wYYypgWw1Q",
        "model" => "gpt-4-1106-preview",
        "object" => "chat.completion.chunk",
        "system_fingerprint" => "fp_d6526cacfe"
      },
      %{
        "choices" => [
          %{
            "delta" => %{
              "tool_calls" => [
                %{"function" => %{"arguments" => "{\"ci"}, "index" => 0}
              ]
            },
            "finish_reason" => nil,
            "index" => 0,
            "logprobs" => nil
          }
        ],
        "created" => 1_712_606_513,
        "id" => "chatcmpl-9BpTdyN883PR9yKpIK2wYYypgWw1Q",
        "model" => "gpt-4-1106-preview",
        "object" => "chat.completion.chunk",
        "system_fingerprint" => "fp_d6526cacfe"
      },
      %{
        "choices" => [
          %{
            "delta" => %{
              "tool_calls" => [
                %{
                  "function" => %{"arguments" => "ty\": \"Moab\", \"state\": \"UT\"}"},
                  "index" => 0
                }
              ]
            },
            "finish_reason" => nil,
            "index" => 0,
            "logprobs" => nil
          }
        ],
        "created" => 1_712_606_513,
        "id" => "chatcmpl-9BpTdyN883PR9yKpIK2wYYypgWw1Q",
        "model" => "gpt-4-1106-preview",
        "object" => "chat.completion.chunk",
        "system_fingerprint" => "fp_d6526cacfe"
      },
      %{
        "choices" => [
          %{
            "delta" => %{
              "tool_calls" => [
                %{
                  "function" => %{"arguments" => "", "name" => "get_weather"},
                  "id" => "call_sEmznyM1sGqYQ4dbNGdubmxa",
                  "index" => 1,
                  "type" => "function"
                }
              ]
            },
            "finish_reason" => nil,
            "index" => 0,
            "logprobs" => nil
          }
        ],
        "created" => 1_712_606_513,
        "id" => "chatcmpl-9BpTdyN883PR9yKpIK2wYYypgWw1Q",
        "model" => "gpt-4-1106-preview",
        "object" => "chat.completion.chunk",
        "system_fingerprint" => "fp_d6526cacfe"
      },
      %{
        "choices" => [
          %{
            "delta" => %{
              "tool_calls" => [
                %{"function" => %{"arguments" => "{\"ci"}, "index" => 1}
              ]
            },
            "finish_reason" => nil,
            "index" => 0,
            "logprobs" => nil
          }
        ],
        "created" => 1_712_606_513,
        "id" => "chatcmpl-9BpTdyN883PR9yKpIK2wYYypgWw1Q",
        "model" => "gpt-4-1106-preview",
        "object" => "chat.completion.chunk",
        "system_fingerprint" => "fp_d6526cacfe"
      },
      %{
        "choices" => [
          %{
            "delta" => %{
              "tool_calls" => [
                %{
                  "function" => %{"arguments" => "ty\": \"Portland\", \"state\": \"OR\"}"},
                  "index" => 1
                }
              ]
            },
            "finish_reason" => nil,
            "index" => 0,
            "logprobs" => nil
          }
        ],
        "created" => 1_712_606_513,
        "id" => "chatcmpl-9BpTdyN883PR9yKpIK2wYYypgWw1Q",
        "model" => "gpt-4-1106-preview",
        "object" => "chat.completion.chunk",
        "system_fingerprint" => "fp_d6526cacfe"
      },
      %{
        "choices" => [
          %{
            "delta" => %{
              "tool_calls" => [
                %{
                  "function" => %{"arguments" => "", "name" => "get_weather"},
                  "id" => "call_cPufqMGm4TOFtqiqFPfz7pcp",
                  "index" => 2,
                  "type" => "function"
                }
              ]
            },
            "finish_reason" => nil,
            "index" => 0,
            "logprobs" => nil
          }
        ],
        "created" => 1_712_606_513,
        "id" => "chatcmpl-9BpTdyN883PR9yKpIK2wYYypgWw1Q",
        "model" => "gpt-4-1106-preview",
        "object" => "chat.completion.chunk",
        "system_fingerprint" => "fp_d6526cacfe"
      },
      %{
        "choices" => [
          %{
            "delta" => %{
              "tool_calls" => [
                %{"function" => %{"arguments" => "{\"ci"}, "index" => 2}
              ]
            },
            "finish_reason" => nil,
            "index" => 0,
            "logprobs" => nil
          }
        ],
        "created" => 1_712_606_513,
        "id" => "chatcmpl-9BpTdyN883PR9yKpIK2wYYypgWw1Q",
        "model" => "gpt-4-1106-preview",
        "object" => "chat.completion.chunk",
        "system_fingerprint" => "fp_d6526cacfe"
      },
      %{
        "choices" => [
          %{
            "delta" => %{
              "tool_calls" => [
                %{
                  "function" => %{"arguments" => "ty\": \"Baltimore\", \"state\": \"MD\"}"},
                  "index" => 2
                }
              ]
            },
            "finish_reason" => nil,
            "index" => 0,
            "logprobs" => nil
          }
        ],
        "created" => 1_712_606_513,
        "id" => "chatcmpl-9BpTdyN883PR9yKpIK2wYYypgWw1Q",
        "model" => "gpt-4-1106-preview",
        "object" => "chat.completion.chunk",
        "system_fingerprint" => "fp_d6526cacfe"
      },
      %{
        "choices" => [
          %{
            "delta" => %{},
            "finish_reason" => "tool_calls",
            "index" => 0,
            "logprobs" => nil
          }
        ],
        "created" => 1_712_606_513,
        "id" => "chatcmpl-9BpTdyN883PR9yKpIK2wYYypgWw1Q",
        "model" => "gpt-4-1106-preview",
        "object" => "chat.completion.chunk",
        "system_fingerprint" => "fp_d6526cacfe"
      }
    ]
  end

  describe "serialize_config/2" do
    test "does not include the API key or callbacks" do
      model = ChatOpenAI.new!(%{model: "gpt-4o"})
      result = ChatOpenAI.serialize_config(model)
      assert result["version"] == 1
      refute Map.has_key?(result, "api_key")
      refute Map.has_key?(result, "callbacks")
    end

    test "creates expected map" do
      model =
        ChatOpenAI.new!(%{
          model: "gpt-4o",
          temperature: 0,
          frequency_penalty: 0.5,
          seed: 123,
          max_tokens: 1234,
          stream_options: %{include_usage: true}
        })

      result = ChatOpenAI.serialize_config(model)

      assert result == %{
               "endpoint" => "https://api.openai.com/v1/chat/completions",
               "frequency_penalty" => 0.5,
               "json_response" => false,
               "max_tokens" => 1234,
               "model" => "gpt-4o",
               "n" => 1,
               "reasoning_mode" => false,
               "reasoning_effort" => "medium",
               "receive_timeout" => 60000,
               "seed" => 123,
               "stream" => false,
               "stream_options" => %{"include_usage" => true},
               "temperature" => 0.0,
               "version" => 1,
               "json_schema" => nil,
               "module" => "Elixir.LangChain.ChatModels.ChatOpenAI"
             }
    end
  end

  describe "set_response_format/1" do
    test "generates a map for an API call with text format when json_response is false" do
      {:ok, openai} =
        ChatOpenAI.new(%{
          model: @test_model,
          json_response: false
        })

      data = ChatOpenAI.for_api(openai, [], [])

      # NOTE: %{"type" => "text"} is the default
      assert data[:response_format] == nil
    end

    test "generates a map for an API call with json_object format when json_response is true and no schema" do
      {:ok, openai} =
        ChatOpenAI.new(%{
          model: @test_model,
          json_response: true
        })

      data = ChatOpenAI.for_api(openai, [], [])

      assert data.response_format == %{"type" => "json_object"}
    end

    test "generates a map for an API call with json_schema format when json_response is true and schema is provided" do
      json_schema = %{
        "type" => "object",
        "properties" => %{
          "name" => %{"type" => "string"},
          "age" => %{"type" => "integer"}
        }
      }

      {:ok, openai} =
        ChatOpenAI.new(%{
          model: @test_model,
          json_response: true,
          json_schema: json_schema
        })

      data = ChatOpenAI.for_api(openai, [], [])

      assert data.response_format == %{
               "type" => "json_schema",
               "json_schema" => json_schema
             }
    end
  end

  describe "inspect" do
    test "redacts the API key" do
      chain = ChatOpenAI.new!()

      changeset = Ecto.Changeset.cast(chain, %{api_key: "1234567890"}, [:api_key])

      refute inspect(changeset) =~ "1234567890"
      assert inspect(changeset) =~ "**redacted**"
    end
  end
end
