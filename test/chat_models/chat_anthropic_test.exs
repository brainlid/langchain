defmodule LangChain.ChatModels.ChatAnthropicTest do
  use LangChain.BaseCase

  doctest LangChain.ChatModels.ChatAnthropic
  alias LangChain.ChatModels.ChatAnthropic
  alias LangChain.Chains.LLMChain
  alias LangChain.Message
  alias LangChain.Message.ContentPart
  alias LangChain.Message.ToolCall
  alias LangChain.Message.ToolResult
  alias LangChain.Function
  alias LangChain.FunctionParam

  @test_model "claude-3-opus-20240229"

  defp hello_world(_args, _context) do
    "Hello world!"
  end

  setup do
    {:ok, hello_world} =
      Function.new(%{
        name: "hello_world",
        description: "Give a hello world greeting.",
        function: fn _args, _context -> "Hello world!" end
      })

    %{hello_world: hello_world}
  end

  describe "new/1" do
    test "works with minimal attr" do
      assert {:ok, %ChatAnthropic{} = anthropic} =
               ChatAnthropic.new(%{"model" => @test_model})

      assert anthropic.model == @test_model
    end

    test "returns error when invalid" do
      assert {:error, changeset} = ChatAnthropic.new(%{"model" => nil})
      refute changeset.valid?
      assert {"can't be blank", _} = changeset.errors[:model]
    end

    test "supports overriding the API endpoint" do
      override_url = "http://localhost:1234/v1/messages"

      model =
        ChatAnthropic.new!(%{
          endpoint: override_url
        })

      assert model.endpoint == override_url
    end
  end

  describe "for_api/3" do
    test "generates a map for an API call" do
      {:ok, anthropic} =
        ChatAnthropic.new(%{
          "model" => @test_model,
          "temperature" => 1,
          "top_p" => 0.5,
          "api_key" => "api_key"
        })

      data = ChatAnthropic.for_api(anthropic, [], [])
      assert data.model == @test_model
      assert data.temperature == 1
      assert data.top_p == 0.5
    end

    test "correctly applies the system message" do
      {:ok, anthropic} = ChatAnthropic.new()

      data =
        ChatAnthropic.for_api(
          anthropic,
          [
            Message.new_system!("You are my helpful hero.")
          ],
          []
        )

      assert "You are my helpful hero." == data[:system]
    end

    test "generates a map for an API call with max_tokens set" do
      {:ok, anthropic} =
        ChatAnthropic.new(%{
          "model" => @test_model,
          "temperature" => 1,
          "top_p" => 0.5,
          "max_tokens" => 1234
        })

      data = ChatAnthropic.for_api(anthropic, [], [])
      assert data.model == @test_model
      assert data.temperature == 1
      assert data.top_p == 0.5
      assert data.max_tokens == 1234
    end

    test "adds tool definitions to map" do
      tool =
        Function.new!(%{
          name: "greet",
          description: "Give a greeting using a specific name",
          parameters: [
            FunctionParam.new!(%{
              type: :object,
              name: "person",
              required: true,
              object_properties: [
                FunctionParam.new!(%{name: "name", type: :string, required: true})
              ]
            })
          ],
          function: fn _args, _context -> :ok end
        })

      output = ChatAnthropic.for_api(ChatAnthropic.new!(), [], [tool])

      assert output[:tools] ==
               [
                 %{
                   "name" => "greet",
                   "description" => "Give a greeting using a specific name",
                   "input_schema" => %{
                     "properties" => %{
                       "person" => %{
                         "properties" => %{"name" => %{"type" => "string"}},
                         "required" => ["name"],
                         "type" => "object"
                       }
                     },
                     "required" => ["person"],
                     "type" => "object"
                   }
                 }
               ]
    end

    test "includes multiple tool responses into a single user message" do
      # ability to restore a conversation and continue it
      messages =
        [
          Message.new_user!("Hi."),
          Message.new_assistant!(%{
            tool_calls: [
              ToolCall.new!(%{call_id: "call_123", name: "greet1", arguments: nil}),
              ToolCall.new!(%{call_id: "call_234", name: "greet2", arguments: nil}),
              ToolCall.new!(%{call_id: "call_345", name: "greet3", arguments: nil})
            ]
          }),
          Message.new_user!("That was a lot of stuff."),
          Message.new_tool_result!(%{
            tool_results: [
              ToolResult.new!(%{tool_call_id: "call_123", content: "sudo hi 1"}),
              ToolResult.new!(%{tool_call_id: "call_234", content: "sudo hi 2"}),
              ToolResult.new!(%{tool_call_id: "call_345", content: "sudo hi 3"})
            ]
          }),
          Message.new_assistant!(%{content: "No, \"sudo hi\""})
        ]

      output = ChatAnthropic.for_api(ChatAnthropic.new!(), messages, [])

      assert output[:messages] ==
               [
                 %{"content" => "Hi.", "role" => "user"},
                 #  tool calls
                 %{
                   "role" => "assistant",
                   "content" => [
                     %{
                       "id" => "call_123",
                       "input" => %{},
                       "name" => "greet1",
                       "type" => "tool_use"
                     },
                     %{
                       "id" => "call_234",
                       "input" => %{},
                       "name" => "greet2",
                       "type" => "tool_use"
                     },
                     %{
                       "id" => "call_345",
                       "input" => %{},
                       "name" => "greet3",
                       "type" => "tool_use"
                     }
                   ]
                 },
                 #  tool result responses
                 %{
                   "role" => "user",
                   "content" => [
                     %{"text" => "That was a lot of stuff.", "type" => "text"},
                     %{
                       "content" => "sudo hi 1",
                       "is_error" => false,
                       "tool_use_id" => "call_123",
                       "type" => "tool_result"
                     },
                     %{
                       "content" => "sudo hi 2",
                       "is_error" => false,
                       "tool_use_id" => "call_234",
                       "type" => "tool_result"
                     },
                     %{
                       "content" => "sudo hi 3",
                       "is_error" => false,
                       "tool_use_id" => "call_345",
                       "type" => "tool_result"
                     }
                   ]
                 },
                 %{"content" => "No, \"sudo hi\"", "role" => "assistant"}
               ]
    end
  end

  describe "do_process_response/1" do
    test "handles receiving a message" do
      response = %{
        "id" => "id-123",
        "type" => "message",
        "role" => "assistant",
        "content" => [%{"type" => "text", "text" => "Greetings!"}],
        "model" => "claude-3-haiku-20240307",
        "stop_reason" => "end_turn"
      }

      assert %Message{} = struct = ChatAnthropic.do_process_response(response)
      assert struct.role == :assistant
      assert struct.content == "Greetings!"
      assert is_nil(struct.index)
    end

    test "handles receiving a content_block_start event" do
      response = %{
        "type" => "content_block_start",
        "index" => 0,
        "content_block" => %{"type" => "text", "text" => ""}
      }

      assert %MessageDelta{} = struct = ChatAnthropic.do_process_response(response)
      assert struct.role == :assistant
      assert struct.content == ""
      assert is_nil(struct.index)
    end

    test "handles receiving a content_block_delta event" do
      response = %{
        "type" => "content_block_delta",
        "index" => 0,
        "delta" => %{"type" => "text_delta", "text" => "Hello"}
      }

      assert %MessageDelta{} = struct = ChatAnthropic.do_process_response(response)
      assert struct.role == :assistant
      assert struct.content == "Hello"
      assert is_nil(struct.index)
    end

    test "handles receiving a message_delta event" do
      response = %{
        "type" => "message_delta",
        "delta" => %{"stop_reason" => "end_turn", "stop_sequence" => nil},
        "usage" => %{"output_tokens" => 47}
      }

      assert %MessageDelta{} = struct = ChatAnthropic.do_process_response(response)
      assert struct.role == :assistant
      assert struct.content == ""
      assert struct.status == :complete
      assert is_nil(struct.index)
    end

    test "handles receiving a tool call with no parameters" do
      response = %{
        "content" => [
          %{"id" => "toolu_0123", "input" => %{}, "name" => "hello_world", "type" => "tool_use"}
        ],
        "id" => "msg_0123",
        "model" => "claude-3-haiku-20240307",
        "role" => "assistant",
        "stop_reason" => "tool_use",
        "stop_sequence" => nil,
        "type" => "message",
        "usage" => %{"input_tokens" => 324, "output_tokens" => 36}
      }

      assert %Message{} = struct = ChatAnthropic.do_process_response(response)

      assert struct.role == :assistant
      [%ToolCall{} = call] = struct.tool_calls
      assert call.type == :function
      assert call.status == :complete
      assert call.call_id == "toolu_0123"
      assert call.name == "hello_world"
      # detects empty and returns nil
      assert call.arguments == nil
    end

    test "handles receiving a tool call with nested empty properties supplied" do
      response = %{
        "content" => [
          %{
            "id" => "toolu_0123",
            "input" => %{"properties" => %{}},
            "name" => "hello_world",
            "type" => "tool_use"
          }
        ],
        "role" => "assistant",
        "stop_reason" => "tool_use",
        "stop_sequence" => nil,
        "type" => "message"
      }

      assert %Message{} = struct = ChatAnthropic.do_process_response(response)

      assert struct.role == :assistant
      [%ToolCall{} = call] = struct.tool_calls
      assert call.type == :function
      assert call.status == :complete
      assert call.call_id == "toolu_0123"
      assert call.name == "hello_world"
      # detects empty and returns as nil
      assert call.arguments == nil
    end

    test "handles receiving text and a tool_use in same message" do
      response = %{
        "id" => "msg_01Aq9w938a90dw8q",
        "model" => @test_model,
        "stop_reason" => "tool_use",
        "role" => "assistant",
        "content" => [
          %{
            "type" => "text",
            "text" =>
              "<thinking>I need to use the get_weather, and the user wants SF, which is likely San Francisco, CA.</thinking>"
          },
          %{
            "type" => "tool_use",
            "id" => "toolu_0123",
            "name" => "get_weather",
            "input" => %{"location" => "San Francisco, CA", "unit" => "celsius"}
          }
        ]
      }

      assert %Message{} = struct = ChatAnthropic.do_process_response(response)

      assert struct.role == :assistant
      assert struct.status == :complete

      assert struct.content ==
               "<thinking>I need to use the get_weather, and the user wants SF, which is likely San Francisco, CA.</thinking>"

      [%ToolCall{} = call] = struct.tool_calls
      assert call.type == :function
      assert call.status == :complete
      assert call.call_id == "toolu_0123"
      assert call.name == "get_weather"
      assert call.arguments == %{"location" => "San Francisco, CA", "unit" => "celsius"}
    end
  end

  describe "call/2" do
    @tag live_call: true, live_anthropic: true
    test "handles when invalid API key given" do
      {:ok, chat} = ChatAnthropic.new(%{stream: true, api_key: "invalid"})

      {:error, reason} =
        ChatAnthropic.call(chat, [
          Message.new_user!("Return the response 'Colorful Threads'.")
        ])

      assert reason == "Authentication failure with request"
    end

    @tag live_call: true, live_anthropic: true
    test "basic streamed content example" do
      {:ok, chat} = ChatAnthropic.new(%{stream: true})

      {:ok, result} =
        ChatAnthropic.call(chat, [
          Message.new_user!("Return the response 'Colorful Threads'.")
        ])

      # returns a list of MessageDeltas.
      assert result == [
               %LangChain.MessageDelta{
                 content: "",
                 status: :incomplete,
                 index: nil,
                 function_name: nil,
                 role: :assistant,
                 arguments: nil
               },
               %LangChain.MessageDelta{
                 content: "Color",
                 status: :incomplete,
                 index: nil,
                 function_name: nil,
                 role: :assistant,
                 arguments: nil
               },
               %LangChain.MessageDelta{
                 content: "ful",
                 status: :incomplete,
                 index: nil,
                 function_name: nil,
                 role: :assistant,
                 arguments: nil
               },
               %LangChain.MessageDelta{
                 content: " Threads",
                 status: :incomplete,
                 index: nil,
                 function_name: nil,
                 role: :assistant,
                 arguments: nil
               },
               %LangChain.MessageDelta{
                 content: "",
                 status: :complete,
                 index: nil,
                 function_name: nil,
                 role: :assistant,
                 arguments: nil
               }
             ]
    end
  end

  describe "decode_stream/1" do
    test "when data is broken" do
      data1 =
        ~s|event: content_block_delta\ndata: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"hr"}       }\n\n
event: content_block_delta\ndata: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"asing"}      }\n\n
event: content_block_delta\ndata: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":" back"}           }\n\n
event: content_block_delta\ndata: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":" what"}             }\n\nevent: content_block_delta\ndata: {"type":"content_block_delta","index":0|

      {processed1, incomplete} = ChatAnthropic.decode_stream({data1, ""})

      assert incomplete ==
               ~s|event: content_block_delta\ndata: {"type":"content_block_delta","index":0|

      assert processed1 == [
               %{
                 "delta" => %{"text" => "hr", "type" => "text_delta"},
                 "index" => 0,
                 "type" => "content_block_delta"
               },
               %{
                 "delta" => %{"text" => "asing", "type" => "text_delta"},
                 "index" => 0,
                 "type" => "content_block_delta"
               },
               %{
                 "delta" => %{"text" => " back", "type" => "text_delta"},
                 "index" => 0,
                 "type" => "content_block_delta"
               },
               %{
                 "delta" => %{"text" => " what", "type" => "text_delta"},
                 "index" => 0,
                 "type" => "content_block_delta"
               }
             ]

      data2 =
        ~s|,"delta":{"type":"text_delta","text":" your"}    }\n\n
event: content_block_delta\ndata: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":" friend"}               }\n\n
event: content_block_delta\ndata: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":" said"}   }\n\n|

      {processed2, incomplete} = ChatAnthropic.decode_stream({data2, incomplete})

      assert incomplete == ""

      assert processed2 == [
               %{
                 "delta" => %{"text" => " your", "type" => "text_delta"},
                 "index" => 0,
                 "type" => "content_block_delta"
               },
               %{
                 "delta" => %{"text" => " friend", "type" => "text_delta"},
                 "index" => 0,
                 "type" => "content_block_delta"
               },
               %{
                 "delta" => %{"text" => " said", "type" => "text_delta"},
                 "index" => 0,
                 "type" => "content_block_delta"
               }
             ]
    end

    test "can parse streaming events" do
      chunk = """
      event: message_start
      data: {"type":"message_start","message":{"id":"msg_01CsrHBjq3eHRQjYG5ayuo5o","type":"message","role":"assistant","content":[],"model":"claude-3-sonnet-20240229","stop_reason":null,"stop_sequence":null,"usage":{"input_tokens":14,"output_tokens":1}}}

      event: content_block_start
      data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}

      event: ping
      data: {"type": "ping"}

      event: content_block_delta
      data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hello"}}

      """

      {parsed, buffer} = ChatAnthropic.decode_stream({chunk, ""})

      assert [
               %{
                 "type" => "content_block_start",
                 "content_block" => %{"text" => "", "type" => "text"},
                 "index" => 0
               },
               %{
                 "type" => "content_block_delta",
                 "delta" => %{"text" => "Hello", "type" => "text_delta"},
                 "index" => 0
               }
             ] = parsed

      assert buffer == ""

      chunk = """
      event: content_block_delta
      data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"!"}}

      """

      {parsed, buffer} = ChatAnthropic.decode_stream({chunk, ""})

      assert [
               %{
                 "type" => "content_block_delta",
                 "delta" => %{"text" => "!", "type" => "text_delta"},
                 "index" => 0
               }
             ] = parsed

      assert buffer == ""

      chunk = """
      event: content_block_stop
      data: {"type":"content_block_stop","index":0}

      event: message_delta
      data: {"type":"message_delta","delta":{"stop_reason":"end_turn","stop_sequence":null},"usage":{"output_tokens": 3}}

      event: message_stop
      data: {"type":"message_stop"}

      """

      {parsed, buffer} = ChatAnthropic.decode_stream({chunk, ""})

      assert [
               %{
                 "type" => "message_delta",
                 "delta" => %{"stop_reason" => "end_turn", "stop_sequence" => nil},
                 "usage" => %{"output_tokens" => 3}
               }
             ] = parsed

      assert buffer == ""
    end

    test "non-ascii unicode character (en dash U+2013)" do
      chunk = """
      event: content_block_delta
      data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":" –"}}

      event: content_block_delta
      data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":" Anthrop"}}

      """

      {parsed, buffer} = ChatAnthropic.decode_stream({chunk, ""})

      assert [
               %{
                 "type" => "content_block_delta",
                 "index" => 0,
                 "delta" => %{"type" => "text_delta", "text" => " –"}
               },
               %{
                 "type" => "content_block_delta",
                 "index" => 0,
                 "delta" => %{"type" => "text_delta", "text" => " Anthrop"}
               }
             ] = parsed

      assert buffer == ""
    end

    test "handles incomplete chunks" do
      chunk_1 =
        "event: content_blo"

      {parsed, buffer} = ChatAnthropic.decode_stream({chunk_1, ""})

      assert [] = parsed
      assert buffer == chunk_1

      chunk_2 =
        "ck_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"de"

      {parsed, buffer} = ChatAnthropic.decode_stream({chunk_2, buffer})

      assert [] = parsed
      assert buffer == chunk_1 <> chunk_2

      chunk_3 = ~s|lta":{"type":"text_delta","text":"!"}}

event: content_block_delta
data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":" Anthrop"}}

|

      {parsed, buffer} = ChatAnthropic.decode_stream({chunk_3, buffer})

      assert [
               %{
                 "type" => "content_block_delta",
                 "delta" => %{"text" => "!", "type" => "text_delta"},
                 "index" => 0
               },
               %{
                 "type" => "content_block_delta",
                 "delta" => %{"text" => " Anthrop", "type" => "text_delta"},
                 "index" => 0
               }
             ] = parsed

      assert buffer == ""
    end
  end

  describe "split_system_message/1" do
    test "returns system message and rest separately" do
      system = Message.new_system!()
      user_msg = Message.new_user!("Hi")
      assert {system, [user_msg]} == ChatAnthropic.split_system_message([system, user_msg])
    end

    test "return nil when no system message set" do
      user_msg = Message.new_user!("Hi")
      assert {nil, [user_msg]} == ChatAnthropic.split_system_message([user_msg])
    end

    test "raises exception with multiple system messages" do
      assert_raise LangChain.LangChainError,
                   "Anthropic only supports a single System message",
                   fn ->
                     system = Message.new_system!()
                     user_msg = Message.new_user!("Hi")
                     ChatAnthropic.split_system_message([system, user_msg, system])
                   end
    end
  end

  describe "for_api/1" do
    test "turns a basic user message into the expected JSON format" do
      expected = %{"role" => "user", "content" => "Hi."}
      result = ChatAnthropic.for_api(Message.new_user!("Hi."))
      assert result == expected
    end

    test "turns a multi-modal user message into the expected JSON format" do
      expected = %{
        "role" => "user",
        "content" => [
          %{"type" => "text", "text" => "Tell me about this image:"},
          %{
            "type" => "image",
            "source" => %{
              "data" => "base64-text-data",
              "type" => "base64",
              "media_type" => "image/jpeg"
            }
          }
        ]
      }

      result =
        ChatAnthropic.for_api(
          Message.new_user!([
            ContentPart.text!("Tell me about this image:"),
            ContentPart.image!("base64-text-data", media: "image/jpeg")
          ])
        )

      assert result == expected
    end

    test "turns a text ContentPart into the expected JSON format" do
      expected = %{"type" => "text", "text" => "Tell me about this image:"}
      result = ChatAnthropic.for_api(ContentPart.text!("Tell me about this image:"))
      assert result == expected
    end

    test "turns an image ContentPart into the expected JSON format" do
      expected = %{
        "type" => "image",
        "source" => %{
          "data" => "image_base64_data",
          "type" => "base64",
          "media_type" => "image/png"
        }
      }

      result =
        ChatAnthropic.for_api(ContentPart.image!("image_base64_data", media: "image/png"))

      assert result == expected
    end

    test "errors on ContentPart type image_url" do
      assert_raise LangChain.LangChainError, "Anthropic does not support image_url", fn ->
        ChatAnthropic.for_api(ContentPart.image_url!("url-to-image"))
      end
    end

    test "turns a function definition into the expected JSON format" do
      # with no description
      tool =
        Function.new!(%{
          name: "do_something",
          parameters: [],
          function: fn _args, _context -> :ok end
        })

      output = ChatAnthropic.for_api(tool)

      assert output == %{
               "name" => "do_something",
               "input_schema" => %{"properties" => %{}, "type" => "object"}
             }

      # with no parameters but has description
      tool =
        Function.new!(%{
          name: "do_something",
          parameters: [],
          description: "Does something",
          function: fn _args, _context -> :ok end
        })

      output = ChatAnthropic.for_api(tool)

      assert output == %{
               "name" => "do_something",
               "description" => "Does something",
               "input_schema" => %{"properties" => %{}, "type" => "object"}
             }

      # with parameters
      tool =
        Function.new!(%{
          name: "do_something",
          parameters: [
            FunctionParam.new!(%{
              name: "person",
              type: :object,
              required: true,
              object_properties: [
                FunctionParam.new!(%{type: :string, name: "name", required: true}),
                FunctionParam.new!(%{type: :number, name: "age"}),
                FunctionParam.new!(%{type: :string, name: "occupation"})
              ]
            })
          ],
          function: fn _args, _context -> :ok end
        })

      output = ChatAnthropic.for_api(tool)

      assert output == %{
               "name" => "do_something",
               "input_schema" => %{
                 "type" => "object",
                 "properties" => %{
                   "person" => %{
                     "type" => "object",
                     "properties" => %{
                       "age" => %{"type" => "number"},
                       "name" => %{"type" => "string"},
                       "occupation" => %{"type" => "string"}
                     },
                     "required" => ["name"]
                   }
                 },
                 "required" => ["person"]
               }
             }
    end

    test "turns a tool_call into expected JSON format" do
      call = ToolCall.new!(%{call_id: "toolu_123", name: "greet", arguments: %{"name" => "John"}})

      json = ChatAnthropic.for_api(call)

      expected = %{
        "type" => "tool_use",
        "id" => "toolu_123",
        "name" => "greet",
        "input" => %{"name" => "John"}
      }

      assert json == expected
    end

    test "turns an assistant message with basic content and tool calls into expected JSON format" do
      msg =
        Message.new_assistant!(%{
          content: "Hi! I think I'll call a tool.",
          tool_calls: [
            ToolCall.new!(%{call_id: "toolu_123", name: "greet", arguments: %{"name" => "John"}})
          ]
        })

      json = ChatAnthropic.for_api(msg)

      expected = %{
        "role" => "assistant",
        "content" => [
          %{
            "type" => "text",
            "text" => "Hi! I think I'll call a tool."
          },
          %{
            "type" => "tool_use",
            "id" => "toolu_123",
            "name" => "greet",
            "input" => %{"name" => "John"}
          }
        ]
      }

      assert json == expected
    end

    test "turns a tool message into expected JSON format" do
      tool_success_result =
        Message.new_tool_result!(%{
          tool_results: [
            ToolResult.new!(%{tool_call_id: "toolu_123", content: "tool answer"})
          ]
        })

      json = ChatAnthropic.for_api(tool_success_result)

      expected = %{
        "role" => "user",
        "content" => [
          %{
            "type" => "tool_result",
            "tool_use_id" => "toolu_123",
            "content" => "tool answer",
            "is_error" => false
          }
        ]
      }

      assert json == expected

      tool_error_result =
        Message.new_tool_result!(%{
          tool_results: [
            ToolResult.new!(%{tool_call_id: "toolu_234", content: "stuff failed", is_error: true})
          ]
        })

      json = ChatAnthropic.for_api(tool_error_result)

      expected = %{
        "role" => "user",
        "content" => [
          %{
            "type" => "tool_result",
            "tool_use_id" => "toolu_234",
            "content" => "stuff failed",
            "is_error" => true
          }
        ]
      }

      assert json == expected
    end

    test "turns a tool result into expected JSON format" do
      tool_success_result = ToolResult.new!(%{tool_call_id: "toolu_123", content: "tool answer"})

      json = ChatAnthropic.for_api(tool_success_result)

      expected = %{
        "type" => "tool_result",
        "tool_use_id" => "toolu_123",
        "content" => "tool answer",
        "is_error" => false
      }

      assert json == expected

      tool_error_result =
        ToolResult.new!(%{tool_call_id: "toolu_234", content: "stuff failed", is_error: true})

      json = ChatAnthropic.for_api(tool_error_result)

      expected = %{
        "type" => "tool_result",
        "tool_use_id" => "toolu_234",
        "content" => "stuff failed",
        "is_error" => true
      }

      assert json == expected
    end

    test "tools work with minimal definition and no parameters" do
      {:ok, fun} =
        Function.new(%{name: "hello_world", function: &hello_world/2})

      result = ChatAnthropic.for_api(fun)

      assert result == %{
               "name" => "hello_world",
               #  NOTE: Sends the required empty parameter definition when none set
               "input_schema" => %{"properties" => %{}, "type" => "object"}
             }
    end

    test "supports function parameters_schema" do
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

      fun =
        Function.new!(%{
          name: "say_hi",
          description: "Provide a friendly greeting.",
          parameters_schema: params_def,
          function: fn _args, _context -> "Hi" end
        })

      result = ChatAnthropic.for_api(fun)

      assert result == %{
               "name" => "say_hi",
               "description" => "Provide a friendly greeting.",
               "input_schema" => params_def
             }
    end
  end

  describe "post_process_and_combines_messages/1" do
    test "returns role alternating messages unchanged" do
      messages =
        [
          Message.new_user!("Hi."),
          Message.new_assistant!(%{content: "Well, hi to you too."}),
          Message.new_user!([
            ContentPart.new!(%{type: :text, content: "No, I said 'hi' first."})
          ]),
          Message.new_assistant!(%{
            tool_calls: [ToolCall.new!(%{call_id: "call_123", name: "greet", arguments: %{}})]
          }),
          Message.new_tool_result!(%{
            tool_results: [
              ToolResult.new!(%{
                tool_call_id: "call_123",
                content: "sudo hi",
                name: "greet",
                arguments: %{}
              })
            ]
          }),
          Message.new_assistant!(%{content: "No, \"sudo hi\""}),
          Message.new_user!("Ah, yes, you win."),
          Message.new_assistant!(%{content: "Thank you for playing."})
        ]
        |> Enum.map(&ChatAnthropic.for_api(&1))

      assert messages == ChatAnthropic.post_process_and_combine_messages(messages)
    end
  end


  describe "image vision using message parts" do
    @tag live_call: true, live_anthropic: true
    test "supports multi-modal user message with image prompt" do
      image_data = load_image_base64("barn_owl.jpg")

      # https://docs.anthropic.com/claude/reference/messages-examples#vision
      {:ok, chat} = ChatAnthropic.new(%{model: @test_model})

      message =
        Message.new_user!([
          ContentPart.text!("Identify what this is a picture of:"),
          ContentPart.image!(image_data, media: "image/jpeg")
        ])

      {:ok, response} = ChatAnthropic.call(chat, [message], [])

      assert %Message{role: :assistant} = response
      assert String.contains?(response.content, "barn owl")
    end
  end

  describe "a tool use" do
    @tag live_call: true, live_anthropic: true
    test "uses a tool with no parameters" do
      # https://docs.anthropic.com/claude/reference/messages-examples#vision
      {:ok, chat} = ChatAnthropic.new(%{model: @test_model})

      message = Message.new_user!("Use the 'do_something' tool.")

      tool =
        Function.new!(%{
          name: "do_something",
          parameters: [],
          function: fn _args, _context -> :ok end
        })

      {:ok, response} = ChatAnthropic.call(chat, [message], [tool])

      assert %Message{role: :assistant} = response
      assert [%ToolCall{} = call] = response.tool_calls
      assert call.status == :complete
      assert call.type == :function
      assert call.name == "do_something"
      # detects empty and returns nil
      assert call.arguments == nil

      # %LangChain.Message{
      #   content: "<thinking>\nThe user has requested to use the 'do_something' tool. Let's look at the parameters for this tool:\n<function>{\"name\": \"do_something\", \"parameters\": {\"properties\": {}, \"type\": \"object\"}}</function>\n\nThis tool does not require any parameters. Since there are no required parameters missing, we can proceed with invoking the 'do_something' tool.\n</thinking>",
      #   index: nil,
      #   status: :complete,
      #   role: :assistant,
      #   name: nil,
      #   tool_calls: [
      #     %LangChain.Message.ToolCall{
      #       status: :complete,
      #       type: :function,
      #       call_id: "toolu_01Pch8mywrRttVZNK3zvntuF",
      #       name: "do_something",
      #       arguments: %{},
      #       index: nil
      #     }
      #   ],
      #   tool_call_id: nil,
      #   is_error: false,
      #   function_name: nil,
      #   arguments: nil
      # }
    end

    @tag live_call: true, live_anthropic: true
    test "uses a tool with parameters" do
      # https://docs.anthropic.com/claude/reference/messages-examples#vision
      {:ok, chat} = ChatAnthropic.new(%{model: @test_model})

      message = Message.new_user!("Use the 'do_something' tool with the value 'cat'.")

      tool =
        Function.new!(%{
          name: "do_something",
          parameters: [FunctionParam.new!(%{type: :string, name: "value", required: true})],
          function: fn _args, _context -> :ok end
        })

      {:ok, response} = ChatAnthropic.call(chat, [message], [tool])

      assert %Message{role: :assistant} = response
      assert [%ToolCall{} = call] = response.tool_calls
      assert call.status == :complete
      assert call.type == :function
      assert call.name == "do_something"
      assert call.arguments == %{"value" => "cat"}

      # %LangChain.Message{
      #   content: "<thinking>\nThe user is requesting to call the do_something tool, which has one required parameter 'value'. They have provided the value \"cat\" which is a valid string for this parameter. Since all required parameters are provided, we can proceed with calling the function.\n</thinking>",
      #   index: nil,
      #   status: :complete,
      #   role: :assistant,
      #   name: nil,
      #   tool_calls: [
      #     %LangChain.Message.ToolCall{
      #       status: :complete,
      #       type: :function,
      #       call_id: "toolu_01U7B3rKa12PxbSunG49SRHD",
      #       name: "do_something",
      #       arguments: %{"value" => "cat"},
      #       index: nil
      #     }
      #   ],
      #   tool_call_id: nil,
      #   is_error: false,
      #   function_name: nil,
      #   arguments: nil
      # }
    end
  end

  describe "works within a chain" do
    @tag live_call: true, live_anthropic: true
    test "works with a streaming response" do
      test_pid = self()

      callback_fn = fn data ->
        send(test_pid, {:streamed_fn, data})
      end

      {:ok, _result_chain, last_message} =
        LLMChain.new!(%{llm: %ChatAnthropic{stream: true}})
        |> LLMChain.add_message(Message.new_user!("Say, 'Hi!'!"))
        |> LLMChain.run(callback_fn: callback_fn)

      assert last_message.content == "Hi!"
      assert last_message.status == :complete
      assert last_message.role == :assistant

      assert_received {:streamed_fn, data}
      assert %MessageDelta{role: :assistant} = data
    end

    @tag live_call: true, live_anthropic: true
    test "works with NON streaming response" do
      test_pid = self()

      callback_fn = fn data ->
        # IO.inspect(data, label: "DATA")
        send(test_pid, {:streamed_fn, data})
      end

      {:ok, _result_chain, last_message} =
        LLMChain.new!(%{llm: %ChatAnthropic{stream: false}})
        |> LLMChain.add_message(Message.new_user!("Say, 'Hi!'!"))
        |> LLMChain.run(callback_fn: callback_fn)

      assert last_message.content == "Hi!"
      assert last_message.status == :complete
      assert last_message.role == :assistant

      assert_received {:streamed_fn, data}
      assert %Message{role: :assistant} = data
    end

    @tag live_call: true, live_anthropic: true
    test "supports continuing a conversation with streaming" do
      test_pid = self()

      callback_fn = fn data ->
        # IO.inspect(data, label: "DATA")
        send(test_pid, {:streamed_fn, data})
      end

      {:ok, _result_chain, last_message} =
        LLMChain.new!(%{llm: %ChatAnthropic{model: @test_model, stream: true}})
        |> LLMChain.add_message(Message.new_system!("You are a helpful and concise assistant."))
        |> LLMChain.add_message(Message.new_user!("Say, 'Hi!'!"))
        |> LLMChain.add_message(Message.new_assistant!("Hi!"))
        |> LLMChain.add_message(Message.new_user!("What's the capitol of Norway?"))
        |> LLMChain.run(callback_fn: callback_fn)

      assert last_message.content =~ "Oslo"
      assert last_message.status == :complete
      assert last_message.role == :assistant

      assert_received {:streamed_fn, data}
      assert %MessageDelta{role: :assistant} = data
    end
  end
end
