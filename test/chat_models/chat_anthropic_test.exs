defmodule LangChain.ChatModels.ChatAnthropicTest do
  use LangChain.BaseCase
  use Mimic

  doctest LangChain.ChatModels.ChatAnthropic

  import LangChain.TestingHelpers
  alias LangChain.ChatModels.ChatAnthropic
  alias LangChain.Chains.LLMChain
  alias LangChain.Message
  alias LangChain.Message.ContentPart
  alias LangChain.Message.ToolCall
  alias LangChain.Message.ToolResult
  alias LangChain.TokenUsage
  alias LangChain.Function
  alias LangChain.FunctionParam
  alias LangChain.BedrockHelpers
  alias LangChain.LangChainError
  alias LangChain.Utils.BedrockStreamDecoder

  @test_model "claude-3-5-sonnet-20241022"
  @bedrock_test_model "anthropic.claude-3-5-sonnet-20241022-v2:0"
  @claude_3_7 "claude-3-7-sonnet-20250219"
  @apis [:anthropic, :anthropic_bedrock]

  defp hello_world(_args, _context) do
    "Hello world!"
  end

  defp api_config_for(:anthropic_bedrock) do
    %{bedrock: BedrockHelpers.bedrock_config(), model: @bedrock_test_model}
  end

  defp api_config_for(:anthropic) do
    %{model: @test_model}
  end

  defp api_config_for(_), do: %{}

  setup context do
    api_config = api_config_for(context[:live_api])

    {:ok, hello_world} =
      Function.new(%{
        name: "hello_world",
        description: "Give a hello world greeting.",
        function: fn _args, _context -> "Hello world!" end
      })

    %{hello_world: hello_world, api_config: api_config}
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

  describe "get_system_text/1" do
    test "returns default system message for nil" do
      result = ChatAnthropic.get_system_text(nil)
      assert result == [%{"type" => "text", "text" => "You are a helpful assistant."}]
    end

    test "returns system message for a system message" do
      result =
        ChatAnthropic.get_system_text(Message.new_system!("You are a custom helpful assistant."))

      assert result == [%{"type" => "text", "text" => "You are a custom helpful assistant."}]
    end

    test "returns system message for a system message with multiple content parts" do
      result =
        ChatAnthropic.get_system_text(
          Message.new_system!([
            ContentPart.text!("You are helpful 1."),
            ContentPart.text!("You are helpful 2.")
          ])
        )

      assert result == [
               %{"type" => "text", "text" => "You are helpful 1."},
               %{"type" => "text", "text" => "You are helpful 2."}
             ]
    end

    test "includes prompt caching" do
      msg =
        Message.new_system!([
          ContentPart.text!(
            "You are an AI assistant tasked with analyzing literary works. Your goal is to provide insightful commentary on themes, characters, and writing style.\n"
          ),
          ContentPart.text!("<the entire contents of Pride and Prejudice>",
            cache_control: true
          )
        ])

      result = ChatAnthropic.get_system_text(msg)

      assert result == [
               %{
                 "text" =>
                   "You are an AI assistant tasked with analyzing literary works. Your goal is to provide insightful commentary on themes, characters, and writing style.\n",
                 "type" => "text"
               },
               %{
                 "cache_control" => %{"type" => "ephemeral"},
                 "text" => "<the entire contents of Pride and Prejudice>",
                 "type" => "text"
               }
             ]
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

      assert [%{"text" => "You are my helpful hero.", "type" => "text"}] == data[:system]
    end

    test "supports prompt caching in the system message" do
      {:ok, anthropic} = ChatAnthropic.new()

      # this example is from https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching.
      data =
        ChatAnthropic.for_api(
          anthropic,
          [
            Message.new_system!([
              ContentPart.text!(
                "You are an AI assistant tasked with analyzing literary works. Your goal is to provide insightful commentary on themes, characters, and writing style.\n"
              ),
              ContentPart.text!("<the entire contents of Pride and Prejudice>",
                cache_control: true
              )
            ])
          ],
          []
        )

      assert data.system ==
               [
                 %{
                   "text" =>
                     "You are an AI assistant tasked with analyzing literary works. Your goal is to provide insightful commentary on themes, characters, and writing style.\n",
                   "type" => "text"
                 },
                 %{
                   "cache_control" => %{"type" => "ephemeral"},
                   "text" => "<the entire contents of Pride and Prejudice>",
                   "type" => "text"
                 }
               ]
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

    test "generated a map for an API call with tool_choice set correctly to auto" do
      {:ok, anthropic} =
        ChatAnthropic.new(%{
          model: @test_model,
          tool_choice: %{"type" => "auto"}
        })

      data = ChatAnthropic.for_api(anthropic, [], [])
      assert data.model == @test_model
      assert data.tool_choice == %{"type" => "auto"}
    end

    test "generated a map for an API call with tool_choice set correctly to a specific function" do
      {:ok, anthropic} =
        ChatAnthropic.new(%{
          model: @test_model,
          tool_choice: %{"type" => "tool", "name" => "get_weather"}
        })

      data = ChatAnthropic.for_api(anthropic, [], [])
      assert data.model == @test_model
      assert data.tool_choice == %{"type" => "tool", "name" => "get_weather"}
    end

    test "includes disable_parallel_tool_use when set in tool_choice" do
      {:ok, anthropic} =
        ChatAnthropic.new(%{
          model: @test_model,
          tool_choice: %{"type" => "auto", "disable_parallel_tool_use" => true}
        })

      data = ChatAnthropic.for_api(anthropic, [], [])
      assert data.model == @test_model
      assert data.tool_choice == %{"type" => "auto", "disable_parallel_tool_use" => true}
    end

    test "includes disable_parallel_tool_use with specific tool" do
      {:ok, anthropic} =
        ChatAnthropic.new(%{
          model: @test_model,
          tool_choice: %{
            "type" => "tool",
            "name" => "get_weather",
            "disable_parallel_tool_use" => true
          }
        })

      data = ChatAnthropic.for_api(anthropic, [], [])
      assert data.model == @test_model

      assert data.tool_choice == %{
               "type" => "tool",
               "name" => "get_weather",
               "disable_parallel_tool_use" => true
             }
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
                 %{"content" => [%{"text" => "Hi.", "type" => "text"}], "role" => "user"},
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
                       "content" => [%{"text" => "sudo hi 1", "type" => "text"}],
                       "is_error" => false,
                       "tool_use_id" => "call_123",
                       "type" => "tool_result"
                     },
                     %{
                       "content" => [%{"text" => "sudo hi 2", "type" => "text"}],
                       "is_error" => false,
                       "tool_use_id" => "call_234",
                       "type" => "tool_result"
                     },
                     %{
                       "content" => [%{"text" => "sudo hi 3", "type" => "text"}],
                       "is_error" => false,
                       "tool_use_id" => "call_345",
                       "type" => "tool_result"
                     }
                   ]
                 },
                 %{
                   "content" => [%{"text" => "No, \"sudo hi\"", "type" => "text"}],
                   "role" => "assistant"
                 }
               ]
    end
  end

  describe "do_process_response/2 with Bedrock" do
    setup do
      model =
        ChatAnthropic.new!(%{stream: false} |> Map.merge(api_config_for(:anthropic_bedrock)))

      %{model: model}
    end

    test "handles messages the same as Anthropics API", %{model: model} do
      response = %{
        "id" => "id-123",
        "type" => "message",
        "role" => "assistant",
        "content" => [%{"type" => "text", "text" => "Greetings!"}],
        "model" => "claude-3-haiku-20240307",
        "stop_reason" => "end_turn",
        "usage" => %{
          "cache_creation_input_tokens" => 0,
          "cache_read_input_tokens" => 0,
          "input_tokens" => 17,
          "output_tokens" => 11
        }
      }

      assert %Message{} = struct = ChatAnthropic.do_process_response(model, response)
      assert struct.role == :assistant
      assert struct.content == [ContentPart.text!("Greetings!")]
      assert is_nil(struct.index)
    end

    test "handles error messages", %{model: model} do
      error = "Invalid API key"
      message = "Received error from API: #{error}"

      assert {:error, exception} =
               ChatAnthropic.do_process_response(model, %{"message" => error})

      assert exception.type == nil
      assert exception.message == message
    end

    test "handles stream error messages", %{model: model} do
      error = "Internal error"
      message = "Stream exception received: #{inspect(error)}"

      assert {:error, exception} =
               ChatAnthropic.do_process_response(model, %{bedrock_exception: error})

      assert exception.type == nil
      assert exception.message == message
    end
  end

  describe "do_process_response/2" do
    setup do
      model = ChatAnthropic.new!(%{stream: false})
      %{model: model}
    end

    test "handles receiving a message", %{model: model} do
      response = %{
        "id" => "id-123",
        "type" => "message",
        "role" => "assistant",
        "content" => [%{"type" => "text", "text" => "Greetings!"}],
        "model" => "claude-3-haiku-20240307",
        "stop_reason" => "end_turn",
        "usage" => %{
          "cache_creation_input_tokens" => 0,
          "cache_read_input_tokens" => 0,
          "input_tokens" => 17,
          "output_tokens" => 11
        }
      }

      assert %Message{} = struct = ChatAnthropic.do_process_response(model, response)
      assert struct.role == :assistant
      assert struct.content == [ContentPart.text!("Greetings!")]
      assert is_nil(struct.index)
    end

    test "handles receiving a complete message with thinking", %{model: model} do
      response = %{
        "content" => [
          %{
            "signature" =>
              "ErUBCkYIAhgCIkCjvrY94GV6fW82e0vtSzgOMjj50Xi8cgtzBCrd40+oqA/JPL0zXEzrAfm5wYXouROoJ7pdFGghKXip41uqFGT7EgykmZx29tPiUdB2z+0aDMccABirA75MuvB0miIwNHtxSOApjBf0ugl+Td/9cqDgC6sMQqXwgoQDx4OLgFBQeFi82nAidRi0+QDlcF3/Kh0nUGyGp1bFenHB7KazXjcNg2XOs9Lfz2UBG3nJVQ==",
            "thinking" =>
              "This is a basic addition problem. I need to add all the numbers together.\n\n400 + 50 = 450\n450 + 3 = 453\n\nSo the answer is 453.",
            "type" => "thinking"
          },
          %{"text" => "The sum of 400 + 50 + 3 is 453.", "type" => "text"}
        ],
        "id" => "msg_01D877HzPUSSWs881P3dSm8c",
        "model" => "claude-3-7-sonnet-20250219",
        "role" => "assistant",
        "stop_reason" => "end_turn",
        "stop_sequence" => nil,
        "type" => "message",
        "usage" => %{
          "cache_creation_input_tokens" => 0,
          "cache_read_input_tokens" => 0,
          "input_tokens" => 55,
          "output_tokens" => 74
        }
      }

      assert %Message{} = struct = ChatAnthropic.do_process_response(model, response)

      assert struct == %LangChain.Message{
               content: [
                 %LangChain.Message.ContentPart{
                   type: :thinking,
                   content:
                     "This is a basic addition problem. I need to add all the numbers together.\n\n400 + 50 = 450\n450 + 3 = 453\n\nSo the answer is 453.",
                   options: [
                     signature:
                       "ErUBCkYIAhgCIkCjvrY94GV6fW82e0vtSzgOMjj50Xi8cgtzBCrd40+oqA/JPL0zXEzrAfm5wYXouROoJ7pdFGghKXip41uqFGT7EgykmZx29tPiUdB2z+0aDMccABirA75MuvB0miIwNHtxSOApjBf0ugl+Td/9cqDgC6sMQqXwgoQDx4OLgFBQeFi82nAidRi0+QDlcF3/Kh0nUGyGp1bFenHB7KazXjcNg2XOs9Lfz2UBG3nJVQ=="
                   ]
                 },
                 %LangChain.Message.ContentPart{
                   type: :text,
                   content: "The sum of 400 + 50 + 3 is 453.",
                   options: []
                 }
               ],
               processed_content: nil,
               index: nil,
               status: :complete,
               role: :assistant,
               name: nil,
               tool_calls: [],
               tool_results: nil,
               metadata: %{
                 usage: %LangChain.TokenUsage{
                   input: 55,
                   output: 74,
                   raw: %{
                     "cache_creation_input_tokens" => 0,
                     "cache_read_input_tokens" => 0,
                     "input_tokens" => 55,
                     "output_tokens" => 74
                   }
                 }
               }
             }
    end

    test "handles receiving a complete message with redacted thinking", %{model: model} do
      # Message prompt:
      # "ANTHROPIC_MAGIC_STRING_TRIGGER_REDACTED_THINKING_46C9A13E193C177646C7398A98432ECCCE4C1253D5E2D82641AC0E52CC2876CB"

      response = %{
        "content" => [
          %{
            "data" =>
              "EtEFCkYIAhgCKkB5MF4IWF6ZUk73fPrddkcENccAG4Ctacz8IeXEagiWsu5YTopMPuNdb/xuvw2T3oX3VuWyVAg5coEqmP3HPMnkEgzSBdTdNApcEh9KxQQaDHB/cJWBjUb32VKHnSIwOck0q4wJ+lKr5dOBFnqKLORXvh1E6L4o5azEkdcn3m/VsF/z9ukn/zIVfCHl6VkoKrgEXZ9Nx0seFKT2yqdAxtihL5guHVathGDCW32fEDS/ANVtKdIm6AoDNzwpN8nIqbUWOHpJjL+vhQpJ/8NxaP4hOKM+Qp9gt50DZCOf+576mpn0BOjB52QQJ/BxzjyuJjbs8WvsvvEKpcj9/oTWoC/XPAlywlRWvXBOynlDrB8zNOBnvYqQArpGvTTNGka7SsrohBHxPymholns0nazFO8TjrzYJDrrsshi4GvfVIIfvRxQp76q244C1nCoWDeV4lOOWpb+VTwZBqkcyVfA27jyAIHhhyVfxpxi1OKNpO3+j0k3pZKsrVa54ddPoImLeVrOjPRFsgH/kK+o5n1qkDsZaEb/+vqRiyAxgg9Yz3JBG/eB2yhEtconwm70Uu+FsmRhMq0XxrifnJO3XA1NFUzM7SVZDB32xNwocYS8XQMGe8An5Wz6NfY5QpMpAtbpCD5dcyI1N84gU97dLjwbURq2HdERAc6KU0eKPImCRhqJzPps49hVdJinTSsYb/RJW10LpqKbysPb5Via7H/XD/zbeTIv3Sbtnmyw/5vSP6Mm60+XJem3iDN78kuJKNsWVqON6HCdbN/MB5wG8AqIbzhSlwR9ZzFjyIfbrnUa934xwzK0vY7da09IInfqZsKPqiTtEfps0Uv810p4kZCldJfwER9jgJRUNsohhXp6p2rWZ9xlAR3QqHlAyEO1qB+ctm1CszS2IDNltCi5xnXTXFa7D11i99UrCO1pgouaalRKedxnC1Qc9cZnUw==",
            "type" => "redacted_thinking"
          },
          %{
            "text" =>
              "I notice you've sent what appears to be some kind of test string or prompt. I don't have any special \"magic strings\" or hidden commands that would change my behavior. I'm Claude, an AI assistant made by Anthropic to be helpful, harmless, and honest.\n\nIs there something specific I can help you with today? I'm happy to answer questions, provide information, or assist with various tasks within my capabilities.",
            "type" => "text"
          }
        ],
        "id" => "msg_01NAJgmpwCPiHMLjyUkEq5Vp",
        "model" => "claude-3-7-sonnet-20250219",
        "role" => "assistant",
        "stop_reason" => "end_turn",
        "stop_sequence" => nil,
        "type" => "message",
        "usage" => %{
          "cache_creation_input_tokens" => 0,
          "cache_read_input_tokens" => 0,
          "input_tokens" => 99,
          "output_tokens" => 248
        }
      }

      assert %Message{} = struct = ChatAnthropic.do_process_response(model, response)

      assert struct == %LangChain.Message{
               content: [
                 %LangChain.Message.ContentPart{
                   type: :unsupported,
                   content:
                     "EtEFCkYIAhgCKkB5MF4IWF6ZUk73fPrddkcENccAG4Ctacz8IeXEagiWsu5YTopMPuNdb/xuvw2T3oX3VuWyVAg5coEqmP3HPMnkEgzSBdTdNApcEh9KxQQaDHB/cJWBjUb32VKHnSIwOck0q4wJ+lKr5dOBFnqKLORXvh1E6L4o5azEkdcn3m/VsF/z9ukn/zIVfCHl6VkoKrgEXZ9Nx0seFKT2yqdAxtihL5guHVathGDCW32fEDS/ANVtKdIm6AoDNzwpN8nIqbUWOHpJjL+vhQpJ/8NxaP4hOKM+Qp9gt50DZCOf+576mpn0BOjB52QQJ/BxzjyuJjbs8WvsvvEKpcj9/oTWoC/XPAlywlRWvXBOynlDrB8zNOBnvYqQArpGvTTNGka7SsrohBHxPymholns0nazFO8TjrzYJDrrsshi4GvfVIIfvRxQp76q244C1nCoWDeV4lOOWpb+VTwZBqkcyVfA27jyAIHhhyVfxpxi1OKNpO3+j0k3pZKsrVa54ddPoImLeVrOjPRFsgH/kK+o5n1qkDsZaEb/+vqRiyAxgg9Yz3JBG/eB2yhEtconwm70Uu+FsmRhMq0XxrifnJO3XA1NFUzM7SVZDB32xNwocYS8XQMGe8An5Wz6NfY5QpMpAtbpCD5dcyI1N84gU97dLjwbURq2HdERAc6KU0eKPImCRhqJzPps49hVdJinTSsYb/RJW10LpqKbysPb5Via7H/XD/zbeTIv3Sbtnmyw/5vSP6Mm60+XJem3iDN78kuJKNsWVqON6HCdbN/MB5wG8AqIbzhSlwR9ZzFjyIfbrnUa934xwzK0vY7da09IInfqZsKPqiTtEfps0Uv810p4kZCldJfwER9jgJRUNsohhXp6p2rWZ9xlAR3QqHlAyEO1qB+ctm1CszS2IDNltCi5xnXTXFa7D11i99UrCO1pgouaalRKedxnC1Qc9cZnUw==",
                   options: [type: "redacted_thinking"]
                 },
                 %LangChain.Message.ContentPart{
                   type: :text,
                   content:
                     "I notice you've sent what appears to be some kind of test string or prompt. I don't have any special \"magic strings\" or hidden commands that would change my behavior. I'm Claude, an AI assistant made by Anthropic to be helpful, harmless, and honest.\n\nIs there something specific I can help you with today? I'm happy to answer questions, provide information, or assist with various tasks within my capabilities.",
                   options: []
                 }
               ],
               processed_content: nil,
               index: nil,
               status: :complete,
               role: :assistant,
               name: nil,
               tool_calls: [],
               tool_results: nil,
               metadata: %{
                 usage: %LangChain.TokenUsage{
                   input: 99,
                   output: 248,
                   raw: %{
                     "cache_creation_input_tokens" => 0,
                     "cache_read_input_tokens" => 0,
                     "input_tokens" => 99,
                     "output_tokens" => 248
                   }
                 }
               }
             }
    end

    test "handles receiving a message_start event and parses usage to metadata", %{model: model} do
      response = %{
        "message" => %{
          "content" => [],
          "id" => "msg_017vYxGobHipWyoZT5uDbGnJ",
          "model" => @claude_3_7,
          "role" => "assistant",
          "stop_reason" => nil,
          "stop_sequence" => nil,
          "type" => "message",
          "usage" => %{
            "cache_creation_input_tokens" => 0,
            "cache_read_input_tokens" => 0,
            "input_tokens" => 55,
            "output_tokens" => 4
          }
        },
        "type" => "message_start"
      }

      assert %MessageDelta{} = struct = ChatAnthropic.do_process_response(model, response)
      assert struct.role == :assistant
      assert struct.content == []

      assert struct.metadata[:usage] ==
               TokenUsage.new!(%{
                 input: 55,
                 output: 4,
                 raw: %{
                   "cache_creation_input_tokens" => 0,
                   "cache_read_input_tokens" => 0,
                   "input_tokens" => 55,
                   "output_tokens" => 4
                 }
               })

      assert is_nil(struct.index)
    end

    test "handles receiving a delta message done with usage", %{model: model} do
      response = %{
        "type" => "message_delta",
        "delta" => %{"stop_reason" => "end_turn", "stop_sequence" => nil},
        "usage" => %{"output_tokens" => 80}
      }

      assert %MessageDelta{} = struct = ChatAnthropic.do_process_response(model, response)
      assert struct.role == :assistant
      assert struct.content == nil
      assert struct.status == :complete
      assert struct.index == nil

      assert struct.metadata[:usage] ==
               TokenUsage.new!(%{
                 input: nil,
                 output: 80,
                 raw: %{"output_tokens" => 80}
               })
    end

    test "handles receiving a content_block_start event for text", %{model: model} do
      response = %{
        "type" => "content_block_start",
        "index" => 0,
        "content_block" => %{"type" => "text", "text" => ""}
      }

      assert %MessageDelta{} = struct = ChatAnthropic.do_process_response(model, response)
      assert struct.role == :assistant
      assert struct.content == %ContentPart{type: :text, options: [], content: ""}
      assert struct.index == 0
    end

    test "handles receiving a content_block_delta event for text", %{model: model} do
      response = %{
        "type" => "content_block_delta",
        "index" => 0,
        "delta" => %{"type" => "text_delta", "text" => "Hello"}
      }

      assert %MessageDelta{} = struct = ChatAnthropic.do_process_response(model, response)
      assert struct.role == :assistant
      assert struct.content == ContentPart.text!("Hello")
      assert struct.index == 0
    end

    test "handles receiving a content_block_start event for tool call", %{model: model} do
      response = %{
        "type" => "content_block_start",
        "index" => 0,
        "content_block" => %{
          "type" => "tool_use",
          "id" => "toolu_01T1x1fJ34qAmk2tNTrN7Up6",
          "name" => "do_something"
        }
      }

      assert %MessageDelta{} = struct = ChatAnthropic.do_process_response(model, response)
      assert struct.role == :assistant
      assert struct.content == nil
      assert is_nil(struct.index)

      assert [
               %LangChain.Message.ToolCall{
                 status: :incomplete,
                 type: :function,
                 call_id: "toolu_01T1x1fJ34qAmk2tNTrN7Up6",
                 name: "do_something",
                 arguments: nil,
                 index: 0
               }
             ] == struct.tool_calls
    end

    test "handles receiving a content_block_delta event for tool call", %{model: model} do
      response = %{
        "type" => "content_block_delta",
        "index" => 0,
        "delta" => %{"type" => "input_json_delta", "partial_json" => "{\"pr"}
      }

      assert %MessageDelta{} = struct = ChatAnthropic.do_process_response(model, response)
      assert struct.role == :assistant
      [tool_call] = struct.tool_calls
      assert tool_call.type == :function
      assert tool_call.arguments == "{\"pr"
      assert tool_call.index == 0
    end

    test "handles receiving a message_delta event", %{model: model} do
      response = %{
        "type" => "message_delta",
        "delta" => %{"stop_reason" => "end_turn", "stop_sequence" => nil},
        "usage" => %{"output_tokens" => 47}
      }

      assert %MessageDelta{} = struct = ChatAnthropic.do_process_response(model, response)
      assert struct.role == :assistant
      assert struct.content == nil
      assert struct.status == :complete
      assert struct.index == nil
    end

    test "handles receiving a tool call with no parameters", %{model: model} do
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

      assert %Message{} = struct = ChatAnthropic.do_process_response(model, response)

      assert struct.role == :assistant
      [%ToolCall{} = call] = struct.tool_calls
      assert call.type == :function
      assert call.status == :complete
      assert call.call_id == "toolu_0123"
      assert call.name == "hello_world"
      # detects empty and returns nil
      assert call.arguments == nil
    end

    test "handles receiving a tool call with nested empty properties supplied", %{model: model} do
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
        "type" => "message",
        "usage" => %{"input_tokens" => 324, "output_tokens" => 36}
      }

      assert %Message{} = struct = ChatAnthropic.do_process_response(model, response)

      assert struct.role == :assistant
      [%ToolCall{} = call] = struct.tool_calls
      assert call.type == :function
      assert call.status == :complete
      assert call.call_id == "toolu_0123"
      assert call.name == "hello_world"
      # detects empty and returns as nil
      assert call.arguments == nil
    end

    test "handles receiving text and a tool_use in same message", %{model: model} do
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
        ],
        "type" => "message",
        "usage" => %{"input_tokens" => 324, "output_tokens" => 36}
      }

      assert %Message{} = struct = ChatAnthropic.do_process_response(model, response)

      assert struct.role == :assistant
      assert struct.status == :complete

      assert struct.content == [
               ContentPart.text!(
                 "<thinking>I need to use the get_weather, and the user wants SF, which is likely San Francisco, CA.</thinking>"
               )
             ]

      [%ToolCall{} = call] = struct.tool_calls
      assert call.type == :function
      assert call.status == :complete
      assert call.call_id == "toolu_0123"
      assert call.name == "get_weather"
      assert call.arguments == %{"location" => "San Francisco, CA", "unit" => "celsius"}
    end

    test "handles receiving overloaded error", %{model: model} do
      response = %{
        "type" => "error",
        "error" => %{
          "details" => nil,
          "type" => "overloaded_error",
          "message" => "Overloaded"
        }
      }

      assert {:error, exception} = ChatAnthropic.do_process_response(model, response)

      assert exception.type == "overloaded_error"
      assert exception.message == "Overloaded"
    end

    test "handles received thinking content blocks", %{model: model} do
      response = %{
        "delta" => %{"thinking" => "Let's ad", "type" => "thinking_delta"},
        "index" => 0,
        "type" => "content_block_delta"
      }

      assert %MessageDelta{} = struct = ChatAnthropic.do_process_response(model, response)

      assert struct.role == :assistant
      assert struct.content == ContentPart.new!(%{type: :thinking, content: "Let's ad"})
      assert struct.index == 0
    end

    test "handles received thinking signature", %{model: model} do
      response = %{
        "delta" => %{
          "signature" =>
            "ErUBCkYIARgCIkCspHHl1+BPuvAExtRMzy6e6DGYV4vI7D8dgqnzLm7RbQ5e4j+aAopCyq29fZqUNNdZbOLleuq/DYIyXjX4HIyIEgwE4N3Vb+9hzkFk/NwaDOy3fw0f0zqRZhAk4CIwp18hR9UsOWYC+pkvt1SnIOGCXBcLdwUxIoUeG3z6WfNwWJV7fulSvz7EVCN5ypzwKh2m/EY9LS1DK1EdUc770O8XdI/j4i0ibc8zRNIjvA==",
          "type" => "signature_delta"
        },
        "index" => 0,
        "type" => "content_block_delta"
      }

      assert %MessageDelta{} = struct = ChatAnthropic.do_process_response(model, response)

      assert struct.role == :assistant

      assert struct.content ==
               ContentPart.new!(%{
                 type: :thinking,
                 options: [
                   signature:
                     "ErUBCkYIARgCIkCspHHl1+BPuvAExtRMzy6e6DGYV4vI7D8dgqnzLm7RbQ5e4j+aAopCyq29fZqUNNdZbOLleuq/DYIyXjX4HIyIEgwE4N3Vb+9hzkFk/NwaDOy3fw0f0zqRZhAk4CIwp18hR9UsOWYC+pkvt1SnIOGCXBcLdwUxIoUeG3z6WfNwWJV7fulSvz7EVCN5ypzwKh2m/EY9LS1DK1EdUc770O8XdI/j4i0ibc8zRNIjvA=="
                 ]
               })

      assert struct.index == 0
    end

    test "handles a streamed thinking response", %{model: model} do
      processed =
        [
          %{
            "message" => %{
              "content" => [],
              "id" => "msg_017vYxGobHipWyoZT5uDbGnJ",
              "model" => @claude_3_7,
              "role" => "assistant",
              "stop_reason" => nil,
              "stop_sequence" => nil,
              "type" => "message",
              "usage" => %{
                "cache_creation_input_tokens" => 0,
                "cache_read_input_tokens" => 0,
                "input_tokens" => 55,
                "output_tokens" => 4
              }
            },
            "type" => "message_start"
          },
          %{
            "content_block" => %{"signature" => "", "thinking" => "", "type" => "thinking"},
            "index" => 0,
            "type" => "content_block_start"
          },
          %{"type" => "ping"},
          %{
            "delta" => %{"thinking" => "Let's ad", "type" => "thinking_delta"},
            "index" => 0,
            "type" => "content_block_delta"
          },
          %{
            "delta" => %{
              "thinking" => "d these numbers.\n400 + 50 = 450\n450 ",
              "type" => "thinking_delta"
            },
            "index" => 0,
            "type" => "content_block_delta"
          },
          %{
            "delta" => %{"thinking" => "+ 3 = 453\n\nSo 400 + 50", "type" => "thinking_delta"},
            "index" => 0,
            "type" => "content_block_delta"
          },
          %{
            "delta" => %{"thinking" => " + 3 = 453", "type" => "thinking_delta"},
            "index" => 0,
            "type" => "content_block_delta"
          },
          %{
            "delta" => %{
              "signature" =>
                "ErUBCkYIARgCIkCspHHl1+BPuvAExtRMzy6e6DGYV4vI7D8dgqnzLm7RbQ5e4j+aAopCyq29fZqUNNdZbOLleuq/DYIyXjX4HIyIEgwE4N3Vb+9hzkFk/NwaDOy3fw0f0zqRZhAk4CIwp18hR9UsOWYC+pkvt1SnIOGCXBcLdwUxIoUeG3z6WfNwWJV7fulSvz7EVCN5ypzwKh2m/EY9LS1DK1EdUc770O8XdI/j4i0ibc8zRNIjvA==",
              "type" => "signature_delta"
            },
            "index" => 0,
            "type" => "content_block_delta"
          },
          %{"index" => 0, "type" => "content_block_stop"},
          %{
            "content_block" => %{"text" => "", "type" => "text"},
            "index" => 1,
            "type" => "content_block_start"
          },
          %{
            "delta" => %{
              "text" => "The answer is 453.\n\n400 + 50 = 450\n450 + 3 =",
              "type" => "text_delta"
            },
            "index" => 1,
            "type" => "content_block_delta"
          },
          %{
            "delta" => %{"text" => " 453", "type" => "text_delta"},
            "index" => 1,
            "type" => "content_block_delta"
          },
          %{"index" => 1, "type" => "content_block_stop"},
          %{
            "delta" => %{"stop_reason" => "end_turn", "stop_sequence" => nil},
            "type" => "message_delta",
            "usage" => %{"output_tokens" => 80}
          },
          %{"type" => "message_stop"}
        ]
        |> Enum.filter(&ChatAnthropic.relevant_event?/1)
        |> Enum.map(&ChatAnthropic.do_process_response(model, &1))

      expected = [
        %LangChain.MessageDelta{
          content: [],
          status: :incomplete,
          index: nil,
          role: :assistant,
          tool_calls: nil,
          metadata: %{
            usage: %LangChain.TokenUsage{
              input: 55,
              output: 4,
              raw: %{
                "cache_creation_input_tokens" => 0,
                "cache_read_input_tokens" => 0,
                "input_tokens" => 55,
                "output_tokens" => 4
              }
            }
          }
        },
        %LangChain.MessageDelta{
          content:
            ContentPart.new!(%{
              type: :thinking,
              content: "",
              options: [signature: ""]
            }),
          status: :incomplete,
          index: 0,
          role: :assistant,
          tool_calls: nil,
          metadata: nil
        },
        %LangChain.MessageDelta{
          content:
            ContentPart.new!(%{
              type: :thinking,
              content: "Let's ad"
            }),
          status: :incomplete,
          index: 0,
          role: :assistant,
          tool_calls: nil,
          metadata: nil
        },
        %LangChain.MessageDelta{
          content:
            ContentPart.new!(%{
              type: :thinking,
              content: "d these numbers.\n400 + 50 = 450\n450 "
            }),
          status: :incomplete,
          index: 0,
          role: :assistant,
          tool_calls: nil,
          metadata: nil
        },
        %LangChain.MessageDelta{
          content:
            ContentPart.new!(%{
              type: :thinking,
              content: "+ 3 = 453\n\nSo 400 + 50"
            }),
          status: :incomplete,
          index: 0,
          role: :assistant,
          tool_calls: nil,
          metadata: nil
        },
        %LangChain.MessageDelta{
          content:
            ContentPart.new!(%{
              type: :thinking,
              content: " + 3 = 453"
            }),
          status: :incomplete,
          index: 0,
          role: :assistant,
          tool_calls: nil,
          metadata: nil
        },
        %LangChain.MessageDelta{
          content:
            ContentPart.new!(%{
              type: :thinking,
              content: nil,
              options: [
                signature:
                  "ErUBCkYIARgCIkCspHHl1+BPuvAExtRMzy6e6DGYV4vI7D8dgqnzLm7RbQ5e4j+aAopCyq29fZqUNNdZbOLleuq/DYIyXjX4HIyIEgwE4N3Vb+9hzkFk/NwaDOy3fw0f0zqRZhAk4CIwp18hR9UsOWYC+pkvt1SnIOGCXBcLdwUxIoUeG3z6WfNwWJV7fulSvz7EVCN5ypzwKh2m/EY9LS1DK1EdUc770O8XdI/j4i0ibc8zRNIjvA=="
              ]
            }),
          status: :incomplete,
          index: 0,
          role: :assistant,
          tool_calls: nil,
          metadata: nil
        },
        %LangChain.MessageDelta{
          content:
            ContentPart.new!(%{
              type: :text,
              content: ""
            }),
          status: :incomplete,
          index: 1,
          role: :assistant,
          tool_calls: nil,
          metadata: nil
        },
        %LangChain.MessageDelta{
          content:
            ContentPart.new!(%{
              type: :text,
              content: "The answer is 453.\n\n400 + 50 = 450\n450 + 3 ="
            }),
          status: :incomplete,
          index: 1,
          role: :assistant,
          tool_calls: nil,
          metadata: nil
        },
        %LangChain.MessageDelta{
          content:
            ContentPart.new!(%{
              type: :text,
              content: " 453"
            }),
          status: :incomplete,
          index: 1,
          role: :assistant,
          tool_calls: nil,
          metadata: nil
        },
        %LangChain.MessageDelta{
          content: nil,
          status: :complete,
          index: nil,
          role: :assistant,
          tool_calls: nil,
          metadata: %{
            usage: %LangChain.TokenUsage{
              input: nil,
              output: 80,
              raw: %{"output_tokens" => 80}
            }
          }
        }
      ]

      assert processed == expected
    end

    test "handles a streamed thinking response with redacted_thinking data", %{model: model} do
      # NOTE: Modified to compress some of the deltas for length
      processed =
        [
          %{
            "message" => %{
              "content" => [],
              "id" => "msg_01RsGSfsGLQb6yhuzWMBQSg1",
              "model" => "claude-3-7-sonnet-20250219",
              "role" => "assistant",
              "stop_reason" => nil,
              "stop_sequence" => nil,
              "type" => "message",
              "usage" => %{
                "cache_creation_input_tokens" => 0,
                "cache_read_input_tokens" => 0,
                "input_tokens" => 99,
                "output_tokens" => 128
              }
            },
            "type" => "message_start"
          },
          %{
            "content_block" => %{
              "data" =>
                "EvQECkYIAhgCKkDvEa/WZMw1F/dtvM39wAkA/15hBdsvURBke1dUppZjaXioL+jyNrIyDrh3dWAo89tXAxfh0Q/dUW47okn3GKF3EgwaAHR6sf2mjOXvnKQaDFn6KLce6N0KixDL7yIwW0Sc4Niv2Smpa2ORkILXaxruojR4GbyljCyre9xxvciM35PnDNvT5jVAYS1peE4oKtsDKogDdVs4tobPY1LPCfduO/EZM16fvK2rmPV9BNB+S0/kvjMBwnk7pNfosLkqB/ZFyqyJIkqAHKH+AW+Xoa+P17AkINS41qIIAM0U784Xee6hCSHQ2Cb9s8oGvbksLLONSwMi7IWJmXyTslMRSBX4VfOIDzkoWVtXH1/Vq1UQCAFgKc5TZM0FtycCsFycZtTDhgcc/1+jwgWndNBB8LDQfTdfvgdxHbbj8/7pyqYWUsGPT/7HYksbkcVBoKLBYvGB0pFrSZiK6ypgXuJyNLJ7jjA4DXyvHe4EWxnZgyFwECmWDan5gGlH+LkG0NfZuEOEWw8XJGKjDIp1EgIw8jGqZ2uzPNpQlIOoJ3XiV9qPxHjN2wKu9u1UDUpk58p5LxkDPV+nY/lAiP0usJyekBEoTZTuQvXmJRt7i9KRrT9xitA3qAl1vAMwSnRQu4LhLeMZWcNKdsVPaWmQcAVrhxvXHBy4cEzb2C3wkCFPjN/98vAHmhXrTbuI2Jx3/VLD23T9XYai56aABEYDeFvstaRYAos1Aa0qknoOmHKPdeNuE+SRGw59BDdTgYQKlWEjU/EHchyGLbjAu3KOCbWK2l5LbSv9upDsp701Xyf9oQByMwTVyytgZwCFexq/bBgB",
              "type" => "redacted_thinking"
            },
            "index" => 0,
            "type" => "content_block_start"
          },
          %{"type" => "ping"},
          %{"index" => 0, "type" => "content_block_stop"},
          %{
            "content_block" => %{
              "data" =>
                "EuYCCkYIAhgCKkBSkyxhK6POgXTXPAclNtjp78AAASV2HS2ZAzZR2lv0Okejxi0rgTIgr9ClXtxQdar10138sgdmjM7wbCgkb3dSEgw2scaN2MhglBVrItgaDASQ2DnWN3hUfTcgoSIwgWHGG0VsLZ5LMZRO0FLhjCTnUE3SZBhNwsf0jDdsTgsyhyHbR1Pv7DFuJ4ZgKZ6GKs0B8cML342Jo3TOR4ZH1VeYuXPjeLzdEIH5y/oDje6YRoNc0ms9haqifiVYwvQaVFdFwl3H/Cca1/Bg57zdtyrUl6PyYRTM7XIwRkbGWc7nSnAF8Jr0YslZRMThjM1Fw/Ygrp/X6o3Wth9YNU9dSNirIois0fAdgxUaik0tNVqXOeoLGiKk8UfNrY8y5lGhqah6OCk7dzhU0azns6OmflRIpvqMqXGlPZIxxxTHwZCVD02Sri+LHNl87dIr2k6rOQq7KOwG0ViNIGxt3KCiIRgB",
              "type" => "redacted_thinking"
            },
            "index" => 1,
            "type" => "content_block_start"
          },
          %{"index" => 1, "type" => "content_block_stop"},
          %{
            "content_block" => %{"text" => "", "type" => "text"},
            "index" => 2,
            "type" => "content_block_start"
          },
          %{
            "delta" => %{
              "text" => "I notice you've sent what appears to be some",
              "type" => "text_delta"
            },
            "index" => 2,
            "type" => "content_block_delta"
          },
          %{
            "delta" => %{
              "text" => " kind of test string or trigger phrase.",
              "type" => "text_delta"
            },
            "index" => 2,
            "type" => "content_block_delta"
          },
          %{
            "delta" => %{"stop_reason" => "end_turn", "stop_sequence" => nil},
            "type" => "message_delta",
            "usage" => %{"output_tokens" => 234}
          },
          %{"type" => "message_stop"}
        ]
        |> Enum.filter(&ChatAnthropic.relevant_event?/1)
        |> Enum.map(&ChatAnthropic.do_process_response(model, &1))

      # merge the deltas and convert to a message
      {:ok, %Message{} = merged} =
        MessageDelta.merge_deltas(processed) |> MessageDelta.to_message()

      # IO.inspect(merged, label: "PROCESSED")
      assert merged == %LangChain.Message{
               content: [
                 %LangChain.Message.ContentPart{
                   type: :unsupported,
                   content:
                     "EvQECkYIAhgCKkDvEa/WZMw1F/dtvM39wAkA/15hBdsvURBke1dUppZjaXioL+jyNrIyDrh3dWAo89tXAxfh0Q/dUW47okn3GKF3EgwaAHR6sf2mjOXvnKQaDFn6KLce6N0KixDL7yIwW0Sc4Niv2Smpa2ORkILXaxruojR4GbyljCyre9xxvciM35PnDNvT5jVAYS1peE4oKtsDKogDdVs4tobPY1LPCfduO/EZM16fvK2rmPV9BNB+S0/kvjMBwnk7pNfosLkqB/ZFyqyJIkqAHKH+AW+Xoa+P17AkINS41qIIAM0U784Xee6hCSHQ2Cb9s8oGvbksLLONSwMi7IWJmXyTslMRSBX4VfOIDzkoWVtXH1/Vq1UQCAFgKc5TZM0FtycCsFycZtTDhgcc/1+jwgWndNBB8LDQfTdfvgdxHbbj8/7pyqYWUsGPT/7HYksbkcVBoKLBYvGB0pFrSZiK6ypgXuJyNLJ7jjA4DXyvHe4EWxnZgyFwECmWDan5gGlH+LkG0NfZuEOEWw8XJGKjDIp1EgIw8jGqZ2uzPNpQlIOoJ3XiV9qPxHjN2wKu9u1UDUpk58p5LxkDPV+nY/lAiP0usJyekBEoTZTuQvXmJRt7i9KRrT9xitA3qAl1vAMwSnRQu4LhLeMZWcNKdsVPaWmQcAVrhxvXHBy4cEzb2C3wkCFPjN/98vAHmhXrTbuI2Jx3/VLD23T9XYai56aABEYDeFvstaRYAos1Aa0qknoOmHKPdeNuE+SRGw59BDdTgYQKlWEjU/EHchyGLbjAu3KOCbWK2l5LbSv9upDsp701Xyf9oQByMwTVyytgZwCFexq/bBgB",
                   options: [type: "redacted_thinking"]
                 },
                 %LangChain.Message.ContentPart{
                   type: :unsupported,
                   content:
                     "EuYCCkYIAhgCKkBSkyxhK6POgXTXPAclNtjp78AAASV2HS2ZAzZR2lv0Okejxi0rgTIgr9ClXtxQdar10138sgdmjM7wbCgkb3dSEgw2scaN2MhglBVrItgaDASQ2DnWN3hUfTcgoSIwgWHGG0VsLZ5LMZRO0FLhjCTnUE3SZBhNwsf0jDdsTgsyhyHbR1Pv7DFuJ4ZgKZ6GKs0B8cML342Jo3TOR4ZH1VeYuXPjeLzdEIH5y/oDje6YRoNc0ms9haqifiVYwvQaVFdFwl3H/Cca1/Bg57zdtyrUl6PyYRTM7XIwRkbGWc7nSnAF8Jr0YslZRMThjM1Fw/Ygrp/X6o3Wth9YNU9dSNirIois0fAdgxUaik0tNVqXOeoLGiKk8UfNrY8y5lGhqah6OCk7dzhU0azns6OmflRIpvqMqXGlPZIxxxTHwZCVD02Sri+LHNl87dIr2k6rOQq7KOwG0ViNIGxt3KCiIRgB",
                   options: [type: "redacted_thinking"]
                 },
                 %LangChain.Message.ContentPart{
                   type: :text,
                   content:
                     "I notice you've sent what appears to be some kind of test string or trigger phrase.",
                   options: []
                 }
               ],
               processed_content: nil,
               index: 2,
               status: :complete,
               role: :assistant,
               name: nil,
               tool_calls: [],
               tool_results: nil,
               metadata: %{
                 usage: %LangChain.TokenUsage{
                   input: 99,
                   output: 362,
                   raw: %{
                     "cache_creation_input_tokens" => 0,
                     "cache_read_input_tokens" => 0,
                     "input_tokens" => 99,
                     "output_tokens" => 362
                   }
                 }
               }
             }

      # going back to the server should keep the redacted_thinking data
      data = ChatAnthropic.message_for_api(merged)
      # IO.inspect(data, label: "DATA")
      assert data == %{
               "content" => [
                 %{
                   "data" =>
                     "EvQECkYIAhgCKkDvEa/WZMw1F/dtvM39wAkA/15hBdsvURBke1dUppZjaXioL+jyNrIyDrh3dWAo89tXAxfh0Q/dUW47okn3GKF3EgwaAHR6sf2mjOXvnKQaDFn6KLce6N0KixDL7yIwW0Sc4Niv2Smpa2ORkILXaxruojR4GbyljCyre9xxvciM35PnDNvT5jVAYS1peE4oKtsDKogDdVs4tobPY1LPCfduO/EZM16fvK2rmPV9BNB+S0/kvjMBwnk7pNfosLkqB/ZFyqyJIkqAHKH+AW+Xoa+P17AkINS41qIIAM0U784Xee6hCSHQ2Cb9s8oGvbksLLONSwMi7IWJmXyTslMRSBX4VfOIDzkoWVtXH1/Vq1UQCAFgKc5TZM0FtycCsFycZtTDhgcc/1+jwgWndNBB8LDQfTdfvgdxHbbj8/7pyqYWUsGPT/7HYksbkcVBoKLBYvGB0pFrSZiK6ypgXuJyNLJ7jjA4DXyvHe4EWxnZgyFwECmWDan5gGlH+LkG0NfZuEOEWw8XJGKjDIp1EgIw8jGqZ2uzPNpQlIOoJ3XiV9qPxHjN2wKu9u1UDUpk58p5LxkDPV+nY/lAiP0usJyekBEoTZTuQvXmJRt7i9KRrT9xitA3qAl1vAMwSnRQu4LhLeMZWcNKdsVPaWmQcAVrhxvXHBy4cEzb2C3wkCFPjN/98vAHmhXrTbuI2Jx3/VLD23T9XYai56aABEYDeFvstaRYAos1Aa0qknoOmHKPdeNuE+SRGw59BDdTgYQKlWEjU/EHchyGLbjAu3KOCbWK2l5LbSv9upDsp701Xyf9oQByMwTVyytgZwCFexq/bBgB",
                   "type" => "redacted_thinking"
                 },
                 %{
                   "data" =>
                     "EuYCCkYIAhgCKkBSkyxhK6POgXTXPAclNtjp78AAASV2HS2ZAzZR2lv0Okejxi0rgTIgr9ClXtxQdar10138sgdmjM7wbCgkb3dSEgw2scaN2MhglBVrItgaDASQ2DnWN3hUfTcgoSIwgWHGG0VsLZ5LMZRO0FLhjCTnUE3SZBhNwsf0jDdsTgsyhyHbR1Pv7DFuJ4ZgKZ6GKs0B8cML342Jo3TOR4ZH1VeYuXPjeLzdEIH5y/oDje6YRoNc0ms9haqifiVYwvQaVFdFwl3H/Cca1/Bg57zdtyrUl6PyYRTM7XIwRkbGWc7nSnAF8Jr0YslZRMThjM1Fw/Ygrp/X6o3Wth9YNU9dSNirIois0fAdgxUaik0tNVqXOeoLGiKk8UfNrY8y5lGhqah6OCk7dzhU0azns6OmflRIpvqMqXGlPZIxxxTHwZCVD02Sri+LHNl87dIr2k6rOQq7KOwG0ViNIGxt3KCiIRgB",
                   "type" => "redacted_thinking"
                 },
                 %{
                   "text" =>
                     "I notice you've sent what appears to be some kind of test string or trigger phrase.",
                   "type" => "text"
                 }
               ],
               "role" => "assistant"
             }
    end
  end

  describe "call/2" do
    @tag live_call: true, live_anthropic: true
    test "handles when invalid API key given" do
      {:ok, chat} = ChatAnthropic.new(%{stream: true, api_key: "invalid"})

      {:error, %LangChainError{} = exception} =
        ChatAnthropic.call(chat, [
          Message.new_user!("Return the response 'Colorful Threads'.")
        ])

      assert exception.message == "Authentication failure with request"
    end

    @tag live_call: true, live_anthropic_bedrock: true
    test "Bedrock: handles when invalid credentials given" do
      {:ok, chat} =
        ChatAnthropic.new(%{
          stream: true,
          bedrock: %{
            credentials: fn -> [access_key_id: "invalid", secret_access_key: "invalid"] end,
            region: "us-east-1"
          }
        })

      {:error, %LangChainError{} = exception} =
        ChatAnthropic.call(chat, [
          Message.new_user!("Return the response 'Colorful Threads'.")
        ])

      assert exception.message ==
               "Received error from API: The security token included in the request is invalid."
    end

    @tag live_call: true, live_anthropic: true
    test "handles tool_call with text content response when NOT streaming" do
      chat =
        ChatAnthropic.new!(%{
          stream: false,
          model: @claude_3_7,
          verbose_api: false
        })

      {:ok, message} =
        ChatAnthropic.call(
          chat,
          [
            Message.new_user!(
              "Use the do_thing tool passing 'test value' and tell me you are using it at the same time."
            )
          ],
          [
            Function.new!(%{
              name: "do_thing",
              parameters: [FunctionParam.new!(%{type: :string, name: "value", required: true})],
              function: fn _args, _context -> :ok end
            })
          ]
        )

      # should be saying something about the tool call
      assert ContentPart.parts_to_string(message.content) =~ "do_thing"
      assert message.status == :complete
      assert message.role == :assistant
      [%ToolCall{} = tool_call] = message.tool_calls
      assert tool_call.status == :complete
      assert tool_call.type == :function
      assert tool_call.name == "do_thing"
      assert tool_call.arguments == %{"value" => "test value"}
    end

    test "returns error tuple when receiving overloaded_error" do
      # Made NOT LIVE here
      expect(Req, :post, fn _req_struct, _opts ->
        # IO.puts "REQ OVERLOAD USED!!!!"
        {:ok,
         {:error,
          LangChainError.exception(type: "overloaded_error", message: "Overloaded (from test)")}}
      end)

      model = ChatAnthropic.new!(%{stream: true, model: @test_model})
      assert {:error, reason} = ChatAnthropic.call(model, "prompt", [])

      assert reason.type == "overloaded_error"
      assert reason.message == "Overloaded (from test)"
    end

    for api <- @apis do
      Module.put_attribute(__MODULE__, :tag, {:"live_#{api}", true})
      @tag live_call: true, live_api: api
      test "#{BedrockHelpers.prefix_for(api)}basic streamed content example and fires ratelimit callback",
           %{live_api: api, api_config: api_config} do
        # NOTE: These callback handlers are not wrapped because they are not
        # being run through the LLMChain. They are being run directly from the
        # Chat model.
        handlers = %{
          on_llm_ratelimit_info: fn headers ->
            send(self(), {:fired_ratelimit_info, headers})
          end,
          on_llm_response_headers: fn response_headers ->
            send(self(), {:fired_response_headers, response_headers})
          end
        }

        {:ok, chat} =
          ChatAnthropic.new(%{stream: true} |> Map.merge(api_config))

        chat = %ChatAnthropic{chat | callbacks: [handlers]}

        {:ok, result} =
          ChatAnthropic.call(chat, [
            Message.new_user!("Return the response 'Keep up the good work!'.")
          ])

        # NOTE: The results differ pretty significantly between the Bedrock and
        # Anthropic APIs. To avoid test failures, we combine the results into a
        # merged delta for final comparison.
        merged = result |> List.flatten() |> MessageDelta.merge_deltas()

        assert merged.merged_content == [ContentPart.text!("Keep up the good work!")]
        assert %TokenUsage{input: input, output: 10} = merged.metadata.usage
        # NOTE: Each API computes tokens differently. Anthropic is 25 while Bedrock is 50.
        assert input >= 25
        assert merged.status == :complete
        assert merged.role == :assistant

        assert_received {:fired_ratelimit_info, info}

        if api != :anthropic_bedrock do
          assert %{
                   "anthropic-ratelimit-requests-limit" => _,
                   "anthropic-ratelimit-requests-remaining" => _,
                   "anthropic-ratelimit-requests-reset" => _,
                   "anthropic-ratelimit-tokens-limit" => _,
                   "anthropic-ratelimit-tokens-remaining" => _,
                   "anthropic-ratelimit-tokens-reset" => _,
                   #  Not always included
                   #  "retry-after" => _,
                   "request-id" => _
                 } = info
        end

        assert_received {:fired_response_headers, response_headers}

        assert %{
                 "connection" => ["keep-alive"],
                 "content-type" => ["text/event-stream; charset=utf-8"]
               } = response_headers
      end
    end
  end

  describe "decode_stream/2 with Bedrock" do
    setup do
      {:ok, model} =
        ChatAnthropic.new(
          %{}
          |> Map.merge(api_config_for(:anthropic_bedrock))
        )

      %{model: model}
    end

    test "filters irrelevant events", %{model: model} do
      relevant_events = [
        %{"type" => "message_start"},
        %{"type" => "content_block_start"},
        %{"type" => "content_block_delta"},
        %{"type" => "message_delta"}
      ]

      BedrockStreamDecoder
      |> stub(:decode_stream, fn _, _ ->
        {[
           %{"type" => "message_stop"}
         ] ++ relevant_events, ""}
      end)

      {chunks, remaining} = ChatAnthropic.decode_stream(model, {"", ""})

      assert chunks == relevant_events
      assert remaining == ""
    end

    test "it passes through exception_message", %{model: model} do
      BedrockStreamDecoder
      |> stub(:decode_stream, fn _, _ -> {[%{bedrock_exception: "internalServerError"}], ""} end)

      {chunks, remaining} = ChatAnthropic.decode_stream(model, {"", ""})
      assert chunks == [%{bedrock_exception: "internalServerError"}]
      assert remaining == ""
    end
  end

  describe "decode_stream/2" do
    test "when data is broken" do
      data1 =
        ~s|event: content_block_delta\ndata: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"hr"}       }\n\n
event: content_block_delta\ndata: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"asing"}      }\n\n
event: content_block_delta\ndata: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":" back"}           }\n\n
event: content_block_delta\ndata: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":" what"}             }\n\nevent: content_block_delta\ndata: {"type":"content_block_delta","index":0|

      {processed1, incomplete} = ChatAnthropic.decode_stream(%ChatAnthropic{}, {data1, ""})

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

      {processed2, incomplete} =
        ChatAnthropic.decode_stream(%ChatAnthropic{}, {data2, incomplete})

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

      {parsed, buffer} = ChatAnthropic.decode_stream(%ChatAnthropic{}, {chunk, ""})

      assert [
               %{
                 "message" => %{
                   "content" => [],
                   "id" => "msg_01CsrHBjq3eHRQjYG5ayuo5o",
                   "model" => "claude-3-sonnet-20240229",
                   "role" => "assistant",
                   "stop_reason" => nil,
                   "stop_sequence" => nil,
                   "type" => "message",
                   "usage" => %{"input_tokens" => 14, "output_tokens" => 1}
                 },
                 "type" => "message_start"
               },
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

      {parsed, buffer} = ChatAnthropic.decode_stream(%ChatAnthropic{}, {chunk, ""})

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

      {parsed, buffer} = ChatAnthropic.decode_stream(%ChatAnthropic{}, {chunk, ""})

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

      {parsed, buffer} = ChatAnthropic.decode_stream(%ChatAnthropic{}, {chunk, ""})

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

      {parsed, buffer} = ChatAnthropic.decode_stream(%ChatAnthropic{}, {chunk_1, ""})

      assert [] = parsed
      assert buffer == chunk_1

      chunk_2 =
        "ck_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"de"

      {parsed, buffer} = ChatAnthropic.decode_stream(%ChatAnthropic{}, {chunk_2, buffer})

      assert [] = parsed
      assert buffer == chunk_1 <> chunk_2

      chunk_3 = ~s|lta":{"type":"text_delta","text":"!"}}

event: content_block_delta
data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":" Anthrop"}}

|

      {parsed, buffer} = ChatAnthropic.decode_stream(%ChatAnthropic{}, {chunk_3, buffer})

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

    test "handles error overloaded message" do
      chunk = """
      event: error\ndata: {\"type\":\"error\",\"error\":{\"details\":null,\"type\":\"overloaded_error\",\"message\":\"Overloaded\"}}

      """

      {parsed, buffer} = ChatAnthropic.decode_stream(%ChatAnthropic{}, {chunk, ""})

      assert [
               %{
                 "type" => "error",
                 "error" => %{
                   "details" => nil,
                   "type" => "overloaded_error",
                   "message" => "Overloaded"
                 }
               }
             ] = parsed

      assert buffer == ""
    end
  end

  describe "parse_stream_events/2" do
    setup _ do
      model = ChatAnthropic.new!(%{stream: true, model: @claude_3_7})

      chunks = [
        "event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_017vYxGobHipWyoZT5uDbGnJ\",\"type\":\"message\",\"role\":\"assistant\",\"model\":\"claude-3-7-sonnet-20250219\",\"content\":[],\"stop_reason\":null,\"stop_sequence\":null,\"usage\":{\"input_tokens\":55,\"cache_creation_input_tokens\":0,\"cache_read_input_tokens\":0,\"output_tokens\":4}} }\n\nevent: content_block_start\ndata: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"thinking\",\"thinking\":\"\",\"signature\":\"\"}             }\n\n",
        "event: ping\ndata: {\"type\": \"ping\"}\n\nevent: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"thinking_delta\",\"thinking\":\"Let's ad\"}               }\n\n",
        "event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"thinking_delta\",\"thinking\":\"d these numbers.\\n400 + 50 = 450\\n450 \"}               }\n\n",
        "event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"thinking_delta\",\"thinking\":\"+ 3 = 453\\n\\nSo 400 + 50\"}           }\n\n",
        "event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"thinking_delta\",\"thinking\":\" + 3 = 453\"}        }\n\n",
        "event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"signature_delta\",\"signature\":\"ErUBCkYIARgCIkCspHHl1+BPuvAExtRMzy6e6DGYV4vI7D8dgqnzLm7RbQ5e4j+aAopCyq29fZqUNNdZbOLleuq/DYIyXjX4HIyIEgwE4N3Vb+9hzkFk/NwaDOy3fw0f0zqRZhAk4CIwp18hR9UsOWYC+pkvt1SnIOGCXBcLdwUxIoUeG3z6WfNwWJV7fulSvz7EVCN5ypzwKh2m/EY9LS1DK1EdUc770O8XdI/j4i0ibc8zRNIjvA==\"}             }\n\nevent: content_block_stop\ndata: {\"type\":\"content_block_stop\",\"index\":0    }\n\nevent: content_block_start\ndata: {\"type\":\"content_block_start\",\"index\":1,\"content_block\":{\"type\":\"text\",\"text\":\"\"}           }\n\nevent: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":1,\"delta\":{\"type\":\"text_delta\",\"text\":\"The answer is 453.\\n\\n400 + 50 = 450\\n450 + 3 =\"}   }\n\n",
        "event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":1,\"delta\":{\"type\":\"text_delta\",\"text\":\" 453\"}       }\n\nevent: content_block_stop\ndata: {\"type\":\"content_block_stop\",\"index\":1  }\n\n",
        "event: message_delta\ndata: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\",\"stop_sequence\":null},\"usage\":{\"output_tokens\":80}   }\n\nevent: message_stop\ndata: {\"type\":\"message_stop\"             }\n\n"
      ]

      %{model: model, chunks: chunks}
    end

    test "decodes single chunk lines as received from a real server call", %{
      model: model,
      chunks: chunks
    } do
      first_chunk = List.first(chunks)
      {parsed, buffer} = ChatAnthropic.parse_stream_events(model, {first_chunk, ""})

      assert parsed == [
               %{
                 "type" => "message_start",
                 "message" => %{
                   "id" => "msg_017vYxGobHipWyoZT5uDbGnJ",
                   "type" => "message",
                   "role" => "assistant",
                   "model" => @claude_3_7,
                   "content" => [],
                   "stop_reason" => nil,
                   "stop_sequence" => nil,
                   "usage" => %{
                     "input_tokens" => 55,
                     "cache_creation_input_tokens" => 0,
                     "cache_read_input_tokens" => 0,
                     "output_tokens" => 4
                   }
                 }
               },
               %{
                 "type" => "content_block_start",
                 "index" => 0,
                 "content_block" => %{"type" => "thinking", "thinking" => "", "signature" => ""}
               }
             ]

      assert buffer == ""

      # Line 2
      next_chunk = Enum.at(chunks, 1)

      {parsed, buffer} = ChatAnthropic.parse_stream_events(model, {next_chunk, ""})

      assert parsed == [
               %{"type" => "ping"},
               %{
                 "type" => "content_block_delta",
                 "index" => 0,
                 "delta" => %{"type" => "thinking_delta", "thinking" => "Let's ad"}
               }
             ]

      assert buffer == ""

      # Line 3
      next_chunk = Enum.at(chunks, 2)

      {parsed, buffer} = ChatAnthropic.parse_stream_events(model, {next_chunk, ""})

      assert parsed == [
               %{
                 "type" => "content_block_delta",
                 "delta" => %{
                   "thinking" => "d these numbers.\n400 + 50 = 450\n450 ",
                   "type" => "thinking_delta"
                 },
                 "index" => 0
               }
             ]

      assert buffer == ""

      # Line 6
      next_chunk = Enum.at(chunks, 5)

      {parsed, buffer} = ChatAnthropic.parse_stream_events(model, {next_chunk, ""})

      assert parsed == [
               %{
                 "delta" => %{
                   "type" => "signature_delta",
                   "signature" =>
                     "ErUBCkYIARgCIkCspHHl1+BPuvAExtRMzy6e6DGYV4vI7D8dgqnzLm7RbQ5e4j+aAopCyq29fZqUNNdZbOLleuq/DYIyXjX4HIyIEgwE4N3Vb+9hzkFk/NwaDOy3fw0f0zqRZhAk4CIwp18hR9UsOWYC+pkvt1SnIOGCXBcLdwUxIoUeG3z6WfNwWJV7fulSvz7EVCN5ypzwKh2m/EY9LS1DK1EdUc770O8XdI/j4i0ibc8zRNIjvA=="
                 },
                 "index" => 0,
                 "type" => "content_block_delta"
               },
               %{"index" => 0, "type" => "content_block_stop"},
               %{
                 "content_block" => %{"text" => "", "type" => "text"},
                 "index" => 1,
                 "type" => "content_block_start"
               },
               %{
                 "delta" => %{
                   "text" => "The answer is 453.\n\n400 + 50 = 450\n450 + 3 =",
                   "type" => "text_delta"
                 },
                 "index" => 1,
                 "type" => "content_block_delta"
               }
             ]

      assert buffer == ""
    end

    test "decodes a complete stream of events", %{model: model, chunks: chunks} do
      parsed =
        Enum.map(chunks, fn chunk ->
          {parsed, buffer} = ChatAnthropic.parse_stream_events(model, {chunk, ""})
          assert buffer == ""
          parsed
        end)
        |> List.flatten()

      # complete and unfiltered
      expected =
        [
          %{
            "message" => %{
              "content" => [],
              "id" => "msg_017vYxGobHipWyoZT5uDbGnJ",
              "model" => @claude_3_7,
              "role" => "assistant",
              "stop_reason" => nil,
              "stop_sequence" => nil,
              "type" => "message",
              "usage" => %{
                "cache_creation_input_tokens" => 0,
                "cache_read_input_tokens" => 0,
                "input_tokens" => 55,
                "output_tokens" => 4
              }
            },
            "type" => "message_start"
          },
          %{
            "content_block" => %{"signature" => "", "thinking" => "", "type" => "thinking"},
            "index" => 0,
            "type" => "content_block_start"
          },
          %{"type" => "ping"},
          %{
            "delta" => %{"thinking" => "Let's ad", "type" => "thinking_delta"},
            "index" => 0,
            "type" => "content_block_delta"
          },
          %{
            "delta" => %{
              "thinking" => "d these numbers.\n400 + 50 = 450\n450 ",
              "type" => "thinking_delta"
            },
            "index" => 0,
            "type" => "content_block_delta"
          },
          %{
            "delta" => %{"thinking" => "+ 3 = 453\n\nSo 400 + 50", "type" => "thinking_delta"},
            "index" => 0,
            "type" => "content_block_delta"
          },
          %{
            "delta" => %{"thinking" => " + 3 = 453", "type" => "thinking_delta"},
            "index" => 0,
            "type" => "content_block_delta"
          },
          %{
            "delta" => %{
              "signature" =>
                "ErUBCkYIARgCIkCspHHl1+BPuvAExtRMzy6e6DGYV4vI7D8dgqnzLm7RbQ5e4j+aAopCyq29fZqUNNdZbOLleuq/DYIyXjX4HIyIEgwE4N3Vb+9hzkFk/NwaDOy3fw0f0zqRZhAk4CIwp18hR9UsOWYC+pkvt1SnIOGCXBcLdwUxIoUeG3z6WfNwWJV7fulSvz7EVCN5ypzwKh2m/EY9LS1DK1EdUc770O8XdI/j4i0ibc8zRNIjvA==",
              "type" => "signature_delta"
            },
            "index" => 0,
            "type" => "content_block_delta"
          },
          %{"index" => 0, "type" => "content_block_stop"},
          %{
            "content_block" => %{"text" => "", "type" => "text"},
            "index" => 1,
            "type" => "content_block_start"
          },
          %{
            "delta" => %{
              "text" => "The answer is 453.\n\n400 + 50 = 450\n450 + 3 =",
              "type" => "text_delta"
            },
            "index" => 1,
            "type" => "content_block_delta"
          },
          %{
            "delta" => %{"text" => " 453", "type" => "text_delta"},
            "index" => 1,
            "type" => "content_block_delta"
          },
          %{"index" => 1, "type" => "content_block_stop"},
          %{
            "delta" => %{"stop_reason" => "end_turn", "stop_sequence" => nil},
            "type" => "message_delta",
            "usage" => %{"output_tokens" => 80}
          },
          %{"type" => "message_stop"}
        ]

      assert parsed == expected
    end
  end

  describe "live thinking test" do
    @tag live_call: true, live_anthropic: true
    test "decodes a live streamed thinking call" do
      llm =
        ChatAnthropic.new!(%{
          stream: true,
          model: @claude_3_7,
          thinking: %{type: "enabled", budget_tokens: 1024},
          verbose_api: false
        })

      {:ok, deltas} = ChatAnthropic.call(llm, "What is 400 + 50 + 3?")
      # IO.inspect(deltas, label: "RESULT DELTAS")

      {:ok, %Message{} = merged} =
        deltas |> List.flatten() |> MessageDelta.merge_deltas() |> MessageDelta.to_message()

      # IO.inspect(merged, label: "MERGED")

      answer = ContentPart.parts_to_string(merged.content)
      # IO.inspect(answer, label: "ANSWER")
      assert answer =~ "453"
    end

    # @tag live_call: true, live_anthropic: true
    # test "decodes a live NON-streamed thinking call with redacted thinking content" do
    #   llm =
    #     ChatAnthropic.new!(%{
    #       stream: true,
    #       model: @claude_3_7,
    #       thinking: %{type: "enabled", budget_tokens: 1024},
    #       verbose_api: true
    #     })

    #   {:ok, result} =
    #     ChatAnthropic.call(
    #       llm,
    #       "ANTHROPIC_MAGIC_STRING_TRIGGER_REDACTED_THINKING_46C9A13E193C177646C7398A98432ECCCE4C1253D5E2D82641AC0E52CC2876CB"
    #     )

    #   IO.inspect(result, label: "RESULT")

    #   assert false
    # end
  end

  describe "message_for_api/1" do
    test "turns a basic user message into the expected JSON format" do
      expected = %{"role" => "user", "content" => [%{"text" => "Hi.", "type" => "text"}]}
      result = ChatAnthropic.message_for_api(Message.new_user!("Hi."))
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
        ChatAnthropic.message_for_api(
          Message.new_user!([
            ContentPart.text!("Tell me about this image:"),
            ContentPart.image!("base64-text-data", media: :jpeg)
          ])
        )

      assert result == expected
    end

    test "turns an assistant message with basic content and tool calls into expected JSON format" do
      msg =
        Message.new_assistant!(%{
          content: "Hi! I think I'll call a tool.",
          tool_calls: [
            ToolCall.new!(%{call_id: "toolu_123", name: "greet", arguments: %{"name" => "John"}})
          ]
        })

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

      assert expected == ChatAnthropic.message_for_api(msg)
    end

    test "turns a tool message into expected JSON format" do
      tool_success_result =
        Message.new_tool_result!(%{
          tool_results: [
            ToolResult.new!(%{tool_call_id: "toolu_123", content: "tool answer"})
          ]
        })

      expected = %{
        "role" => "user",
        "content" => [
          %{
            "type" => "tool_result",
            "tool_use_id" => "toolu_123",
            "content" => [%{"text" => "tool answer", "type" => "text"}],
            "is_error" => false
          }
        ]
      }

      assert expected == ChatAnthropic.message_for_api(tool_success_result)

      tool_error_result =
        Message.new_tool_result!(%{
          tool_results: [
            ToolResult.new!(%{tool_call_id: "toolu_234", content: "stuff failed", is_error: true})
          ]
        })

      expected = %{
        "role" => "user",
        "content" => [
          %{
            "type" => "tool_result",
            "tool_use_id" => "toolu_234",
            "content" => [%{"text" => "stuff failed", "type" => "text"}],
            "is_error" => true
          }
        ]
      }

      assert expected == ChatAnthropic.message_for_api(tool_error_result)
    end

    test "tool result supports prompt caching in the result options" do
      # NOTE: This is legacy support isn't listed in the current API docs.
      tool_success_result =
        Message.new_tool_result!(%{
          tool_results: [
            ToolResult.new!(%{
              tool_call_id: "toolu_123",
              # content is a plain string
              content: "tool answer",
              options: [cache_control: true]
            })
          ]
        })

      expected = %{
        "role" => "user",
        "content" => [
          %{
            "type" => "tool_result",
            "tool_use_id" => "toolu_123",
            "content" => [%{"text" => "tool answer", "type" => "text"}],
            "is_error" => false,
            "cache_control" => %{"type" => "ephemeral"}
          }
        ]
      }

      assert expected == ChatAnthropic.message_for_api(tool_success_result)
    end

    test "tool result supports prompt caching in content parts" do
      tool_success_result =
        Message.new_tool_result!(%{
          tool_results: [
            ToolResult.new!(%{
              tool_call_id: "toolu_123",
              content: [ContentPart.text!("tool answer", cache_control: true)]
            })
          ]
        })

      expected = %{
        "role" => "user",
        "content" => [
          %{
            "type" => "tool_result",
            "tool_use_id" => "toolu_123",
            "content" => [
              %{
                "text" => "tool answer",
                "type" => "text",
                "cache_control" => %{"type" => "ephemeral"}
              }
            ],
            "is_error" => false
          }
        ]
      }

      assert expected == ChatAnthropic.message_for_api(tool_success_result)
    end
  end

  describe "content_part_for_api/1" do
    test "turns a text ContentPart into the expected JSON format" do
      expected = %{"type" => "text", "text" => "Tell me about this image:"}
      result = ChatAnthropic.content_part_for_api(ContentPart.text!("Tell me about this image:"))
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
        ChatAnthropic.content_part_for_api(ContentPart.image!("image_base64_data", media: :png))

      assert result == expected
    end

    test "turns image ContentPart's media_type into the expected value" do
      assert %{"source" => %{"media_type" => "image/png"}} =
               ChatAnthropic.content_part_for_api(
                 ContentPart.image!("image_base64_data", media: :png)
               )

      assert %{"source" => %{"media_type" => "image/jpeg"}} =
               ChatAnthropic.content_part_for_api(
                 ContentPart.image!("image_base64_data", media: :jpg)
               )

      assert %{"source" => %{"media_type" => "image/jpeg"}} =
               ChatAnthropic.content_part_for_api(
                 ContentPart.image!("image_base64_data", media: :jpeg)
               )

      assert %{"source" => %{"media_type" => "image/webp"}} =
               ChatAnthropic.content_part_for_api(
                 ContentPart.image!("image_base64_data", media: "image/webp")
               )
    end

    test "cache_control: true uses default settings" do
      part = ContentPart.text!("content", cache_control: true)

      result = ChatAnthropic.content_part_for_api(part)

      assert result == %{
               "type" => "text",
               "text" => "content",
               "cache_control" => %{"type" => "ephemeral"}
             }
    end

    test "cache_control supports explicit settings" do
      part = ContentPart.text!("content", cache_control: %{"type" => "ephemeral", "ttl" => "1h"})

      result = ChatAnthropic.content_part_for_api(part)

      assert result == %{
               "type" => "text",
               "text" => "content",
               "cache_control" => %{"type" => "ephemeral", "ttl" => "1h"}
             }
    end

    test "errors on ContentPart type image_url" do
      assert_raise LangChain.LangChainError, "Anthropic does not support image_url", fn ->
        ChatAnthropic.content_part_for_api(ContentPart.image_url!("url-to-image"))
      end
    end
  end

  describe "function_for_api/1" do
    test "turns a function definition into the expected JSON format" do
      # with no description
      tool =
        Function.new!(%{
          name: "do_something",
          parameters: [],
          function: fn _args, _context -> :ok end
        })

      output = ChatAnthropic.function_for_api(tool)

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

      output = ChatAnthropic.function_for_api(tool)

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

      output = ChatAnthropic.function_for_api(tool)

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

    test "supports cache_control on the function options" do
      tool =
        Function.new!(%{
          name: "do_something",
          parameters: [],
          function: fn _args, _context -> :ok end,
          options: [cache_control: true]
        })

      output = ChatAnthropic.function_for_api(tool)

      assert output == %{
               "name" => "do_something",
               "input_schema" => %{"properties" => %{}, "type" => "object"},
               "cache_control" => %{"type" => "ephemeral"}
             }
    end

    test "tools work with minimal definition and no parameters" do
      {:ok, fun} =
        Function.new(%{name: "hello_world", function: &hello_world/2})

      result = ChatAnthropic.function_for_api(fun)

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

      result = ChatAnthropic.function_for_api(fun)

      assert result == %{
               "name" => "say_hi",
               "description" => "Provide a friendly greeting.",
               "input_schema" => params_def
             }
    end
  end

  describe "for_api/1" do
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

    test "turns a tool result into expected JSON format" do
      tool_success_result = ToolResult.new!(%{tool_call_id: "toolu_123", content: "tool answer"})

      json = ChatAnthropic.for_api(tool_success_result)

      expected = %{
        "type" => "tool_result",
        "tool_use_id" => "toolu_123",
        "content" => [%{"text" => "tool answer", "type" => "text"}],
        "is_error" => false
      }

      assert json == expected

      tool_error_result =
        ToolResult.new!(%{tool_call_id: "toolu_234", content: "stuff failed", is_error: true})

      json = ChatAnthropic.for_api(tool_error_result)

      expected = %{
        "type" => "tool_result",
        "tool_use_id" => "toolu_234",
        "content" => [%{"text" => "stuff failed", "type" => "text"}],
        "is_error" => true
      }

      assert json == expected
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
        |> Enum.map(&ChatAnthropic.message_for_api(&1))

      assert messages == ChatAnthropic.post_process_and_combine_messages(messages)
    end
  end

  describe "image vision using message parts" do
    for api <- @apis do
      Module.put_attribute(__MODULE__, :tag, {:"live_#{api}", true})
      @tag live_call: true, live_api: api
      test "#{BedrockHelpers.prefix_for(api)} supports multi-modal user message with image prompt",
           %{api_config: api_config} do
        image_data = load_image_base64("barn_owl.jpg")

        # https://docs.anthropic.com/claude/reference/messages-examples#vision
        {:ok, chat} = ChatAnthropic.new(%{model: @test_model} |> Map.merge(api_config))

        message =
          Message.new_user!([
            ContentPart.text!("Identify what this is a picture of:"),
            ContentPart.image!(image_data, media: :jpg)
          ])

        {:ok, response} = ChatAnthropic.call(chat, [message], [])

        assert %Message{role: :assistant} = response
        text_content = ContentPart.parts_to_string(response.content)
        assert String.contains?(text_content |> String.downcase(), "barn owl")
      end
    end
  end

  describe "a tool use" do
    @tag live_call: true, live_anthropic: true
    test "executes a call with tool_choice set as a specific name" do
      # https://docs.anthropic.com/claude/reference/messages-examples#vision
      {:ok, chat} =
        ChatAnthropic.new(%{
          model: @test_model,
          tool_choice: %{"type" => "tool", "name" => "do_another_thing"}
        })

      message =
        Message.new_user!("Call the tool with the name 'foo'")

      tool_1 =
        Function.new!(%{
          name: "do_something",
          parameters: [FunctionParam.new!(%{type: :string, name: "value", required: true})],
          function: fn _args, _context -> :ok end
        })

      tool_2 =
        Function.new!(%{
          name: "do_another_thing",
          parameters: [FunctionParam.new!(%{type: :string, name: "name", required: true})],
          function: fn _args, _context -> :ok end
        })

      {:ok, response} = ChatAnthropic.call(chat, [message], [tool_1, tool_2])

      assert %Message{role: :assistant} = response
      assert [%ToolCall{} = call] = response.tool_calls
      assert call.status == :complete
      assert call.type == :function
      assert call.name == "do_another_thing"
      assert call.arguments == %{"name" => "foo"}
    end

    for api <- @apis do
      Module.put_attribute(__MODULE__, :tag, {:"live_#{api}", true})
      @tag live_call: true, live_api: api
      test "#{BedrockHelpers.prefix_for(api)} uses a tool with no parameters", %{
        api_config: api_config
      } do
        # https://docs.anthropic.com/en/docs/tool-use
        {:ok, chat} = ChatAnthropic.new(%{model: api_config.model} |> Map.merge(api_config))

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
        # }
      end
    end
  end

  for api <- @apis do
    Module.put_attribute(__MODULE__, :tag, {:"live_#{api}", true})
    @tag live_call: true, live_api: api
    test "#{BedrockHelpers.prefix_for(api)} uses a tool with parameters", %{
      api_config: api_config
    } do
      # https://docs.anthropic.com/claude/reference/messages-examples#vision
      {:ok, chat} = ChatAnthropic.new(%{model: api_config.model} |> Map.merge(api_config))

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
      # }
    end

    @tag live_call: true, live_api: api
    test "#{BedrockHelpers.prefix_for(api)} streams a tool call with parameters", %{
      api_config: api_config
    } do
      handler = %{
        on_llm_new_delta: fn %LLMChain{} = _chain, delta ->
          # IO.inspect(delta, label: "DELTA")
          send(self(), {:streamed_fn, delta})
        end
      }

      {:ok, chat} =
        ChatAnthropic.new(%{model: api_config.model, stream: true} |> Map.merge(api_config))

      text =
        "People tell me I should be more patient, but I can't just sit around waiting for that to happen"

      user_message = Message.new_user!("Use the 'do_something' tool with the value '#{text}'.")

      tool =
        Function.new!(%{
          name: "do_something",
          parameters: [FunctionParam.new!(%{type: :string, name: "value", required: true})],
          function: fn _args, _context ->
            # IO.inspect(args, label: "FUNCTION EXECUTED")
            {:ok, "SUCCESS"}
          end
        })

      # verbose: true
      {:ok, updated_chain} =
        LLMChain.new!(%{llm: chat, verbose: false})
        |> LLMChain.add_message(user_message)
        |> LLMChain.add_tools(tool)
        |> LLMChain.add_callback(handler)
        |> LLMChain.run(mode: :until_success)

      # has the result from the function execution
      [tool_result] = updated_chain.last_message.tool_results
      assert tool_result.content == [ContentPart.text!("SUCCESS")]
    end

    describe "#{BedrockHelpers.prefix_for(api)} works within a chain" do
      Module.put_attribute(__MODULE__, :tag, {:"live_#{api}", true})
      @tag live_call: true, live_api: api
      test "works with a streaming response", %{api_config: api_config} do
        test_pid = self()

        handler = %{
          on_llm_new_delta: fn %LLMChain{} = _chain, delta ->
            send(test_pid, {:streamed_fn, delta})
          end
        }

        {:ok, chat} =
          ChatAnthropic.new(
            %{stream: true}
            |> Map.merge(api_config)
          )

        {:ok, updated_chain} =
          %{llm: chat}
          |> LLMChain.new!()
          |> LLMChain.add_message(Message.new_user!("Say, 'Hi!'!"))
          |> LLMChain.add_callback(handler)
          |> LLMChain.run()

        assert updated_chain.last_message.content == [ContentPart.text!("Hi!")]
        assert updated_chain.last_message.status == :complete
        assert updated_chain.last_message.role == :assistant
        # the final message includes the token usage
        assert %TokenUsage{input: usage} = updated_chain.last_message.metadata.usage
        # Anthropic and Bedrock compute token usage differently for the same inputs.
        assert usage in [20, 40]

        assert_received {:streamed_fn, data}
        assert [%MessageDelta{role: :assistant} | _] = data
      end

      Module.put_attribute(__MODULE__, :tag, {:"live_#{api}", true})
      @tag live_call: true, live_api: api
      test "works with NON streaming response and fires ratelimit callback and token usage", %{
        api_config: api_config,
        live_api: api
      } do
        test_pid = self()

        handler = %{
          on_llm_new_message: fn %LLMChain{} = _chain, message ->
            send(test_pid, {:received_msg, message})
          end,
          on_llm_ratelimit_info: fn %LLMChain{} = _chain, headers ->
            send(test_pid, {:fired_ratelimit_info, headers})
          end,
          on_llm_token_usage: fn %LLMChain{} = _chain, usage ->
            send(self(), {:fired_token_usage, usage})
          end,
          on_llm_response_headers: fn %LLMChain{} = _chain, response_headers ->
            send(self(), {:fired_response_headers, response_headers})
          end
        }

        {:ok, updated_chain} =
          LLMChain.new!(%{
            llm: ChatAnthropic.new!(%{stream: false} |> Map.merge(api_config))
          })
          |> LLMChain.add_message(Message.new_user!("Say, 'Hi!'!"))
          |> LLMChain.add_callback(handler)
          |> LLMChain.run()

        assert updated_chain.last_message.content == [ContentPart.text!("Hi!")]
        assert updated_chain.last_message.status == :complete
        assert updated_chain.last_message.role == :assistant
        # the last message includes the token usage
        assert %TokenUsage{input: 20} = updated_chain.last_message.metadata.usage

        assert_received {:received_msg, data}
        assert %Message{role: :assistant} = data

        assert_received {:fired_ratelimit_info, info}

        if api != :anthropic_bedrock do
          assert %{
                   "anthropic-ratelimit-requests-limit" => _,
                   "anthropic-ratelimit-requests-remaining" => _,
                   "anthropic-ratelimit-requests-reset" => _,
                   "anthropic-ratelimit-tokens-limit" => _,
                   "anthropic-ratelimit-tokens-remaining" => _,
                   "anthropic-ratelimit-tokens-reset" => _,
                   #  Not always included
                   #  "retry-after" => _,
                   "request-id" => _
                 } = info
        end

        # should have fired the token usage callback
        assert_received {:fired_token_usage, usage}
        assert %TokenUsage{input: 20} = usage

        # should have fired the response headers callback
        assert_received {:fired_response_headers, response_headers}

        assert %{
                 "connection" => ["keep-alive"],
                 "content-type" => ["application/json"]
               } = response_headers
      end

      Module.put_attribute(__MODULE__, :tag, {:"live_#{api}", true})
      @tag live_call: true, live_api: api
      test "supports continuing a conversation with streaming", %{api_config: api_config} do
        test_pid = self()

        handler = %{
          on_llm_new_delta: fn %LLMChain{} = _chain, delta ->
            # IO.inspect(data, label: "DATA")
            send(test_pid, {:streamed_fn, delta})
          end
        }

        chat =
          ChatAnthropic.new!(
            %{model: api_config.model, stream: true}
            |> Map.merge(api_config)
          )

        {:ok, updated_chain} =
          %{llm: chat}
          |> LLMChain.new!()
          |> LLMChain.add_message(Message.new_system!("You are a helpful and concise assistant."))
          |> LLMChain.add_message(Message.new_user!("Say, 'Hi!'!"))
          |> LLMChain.add_message(Message.new_assistant!("Hi!"))
          |> LLMChain.add_message(Message.new_user!("What's the capitol of Norway?"))
          |> LLMChain.add_callback(handler)
          |> LLMChain.run()

        assert ContentPart.parts_to_string(updated_chain.last_message.content) =~ "Oslo"
        assert updated_chain.last_message.status == :complete
        assert updated_chain.last_message.role == :assistant

        assert_received {:streamed_fn, data}
        assert [%MessageDelta{role: :assistant} | _] = data
      end
    end

    # @tag live_call: true, live_anthropic: true
    # test "supports starting the assistant's response message and continuing it" do
    #   test_pid = self()

    #   handler = %{
    #     on_llm_new_delta: fn _model, delta ->
    #       # IO.inspect(data, label: "DATA")
    #       send(test_pid, {:streamed_fn, data})
    #     end
    #   }

    #   {:ok, result_chain, last_message} =
    #     LLMChain.new!(%{llm: %ChatAnthropic{model: api_config.model, stream: true, callbacks: [handler]}})
    #     |> LLMChain.add_message(Message.new_system!("You are a helpful and concise assistant."))
    #     |> LLMChain.add_message(
    #       Message.new_user!(
    #         "What's the capitol of Norway? Please respond with the answer <answer>{{ANSWER}}</answer>."
    #       )
    #     )
    #     |> LLMChain.add_message(Message.new_assistant!("<answer>"))
    #     |> LLMChain.run()

    #   assert last_message.content =~ "Oslo"
    #   assert last_message.status == :complete
    #   assert last_message.role == :assistant

    #   # TODO: MERGE A CONTINUED Assistant message with the one we provided.

    #   IO.inspect(result_chain, label: "FINAL CHAIN")
    #   IO.inspect(last_message)

    #   assert_received {:streamed_fn, data}
    #   assert %MessageDelta{role: :assistant} = data

    #   assert false
    # end
  end

  describe "use in LLMChain" do
    @tag live_call: true, live_anthropic: true
    test "NOT STREAMED with callbacks and token usage" do
      handler = %{
        on_llm_new_delta: fn %LLMChain{} = _chain, deltas ->
          send(self(), {:test_stream_deltas, deltas})
        end,
        on_message_processed: fn %LLMChain{} = _chain, message ->
          send(self(), {:test_message_processed, message})
        end,
        on_llm_token_usage: fn %LLMChain{} = _chain, usage ->
          send(self(), {:test_token_usage, usage})
        end
      }

      # We can construct an LLMChain from a PromptTemplate and an LLM.
      model = ChatAnthropic.new!(%{temperature: 1, seed: 0, stream: false})

      {:ok, updated_chain} =
        %{llm: model}
        |> LLMChain.new!()
        |> LLMChain.add_callback(handler)
        |> LLMChain.add_messages([
          Message.new_user!("Suggest one good name for a company that makes colorful socks?")
        ])
        |> LLMChain.run()

      assert %Message{role: :assistant} = updated_chain.last_message
      assert %TokenUsage{input: 28} = updated_chain.last_message.metadata.usage

      assert_received {:test_message_processed, message}
      assert %Message{role: :assistant} = message
      # the final returned message should match the callback message
      assert message == updated_chain.last_message
      # we should have received the final combined message
      refute_received {:test_stream_deltas, _delta}

      assert_received {:test_token_usage, usage}
      assert %TokenUsage{input: 28} = usage
    end

    @tag live_call: true, live_anthropic: true
    test "STREAMED with callbacks and token usage" do
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

      # We can construct an LLMChain from a PromptTemplate and an LLM.
      model =
        ChatAnthropic.new!(%{
          temperature: 1,
          seed: 0,
          stream: true
        })

      original_chain =
        %{llm: model}
        |> LLMChain.new!()
        |> LLMChain.add_callback(handler)
        |> LLMChain.add_messages([
          Message.new_user!("Suggest one good name for a company that makes colorful socks?")
        ])

      {:ok, updated_chain} = original_chain |> LLMChain.run()

      assert %Message{role: :assistant, status: :complete} = updated_chain.last_message
      assert %TokenUsage{input: 56} = updated_chain.last_message.metadata.usage

      assert_received {:test_message_processed, message}
      assert %Message{role: :assistant} = message
      # the final returned message should match the callback message
      assert message == updated_chain.last_message

      assert_received {:test_token_usage, usage}
      assert %TokenUsage{input: 56} = usage

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

  describe "serialize_config/2" do
    test "does not include the API key or callbacks" do
      model = ChatAnthropic.new!(%{model: "claude-3-haiku-20240307"})
      result = ChatAnthropic.serialize_config(model)
      assert result["version"] == 1
      refute Map.has_key?(result, "api_key")
      refute Map.has_key?(result, "callbacks")
    end

    test "creates expected map" do
      model =
        ChatAnthropic.new!(%{
          model: "claude-3-haiku-20240307",
          temperature: 0,
          max_tokens: 1234
        })

      result = ChatAnthropic.serialize_config(model)

      assert result == %{
               "endpoint" => "https://api.anthropic.com/v1/messages",
               "model" => "claude-3-haiku-20240307",
               "max_tokens" => 1234,
               "receive_timeout" => 60000,
               "stream" => false,
               "temperature" => 0.0,
               "api_version" => "2023-06-01",
               "top_k" => nil,
               "top_p" => nil,
               "beta_headers" => ["tools-2024-04-04"],
               "module" => "Elixir.LangChain.ChatModels.ChatAnthropic",
               "version" => 1
             }
    end

    test "includes beta_headers in the serialized config" do
      custom_beta_headers = ["custom-beta-feature-1", "custom-beta-feature-2"]

      model =
        ChatAnthropic.new!(%{
          model: "claude-3-haiku-20240307",
          beta_headers: custom_beta_headers
        })

      result = ChatAnthropic.serialize_config(model)

      assert result["beta_headers"] == custom_beta_headers
    end
  end

  describe "beta_headers" do
    test "adds beta headers to request headers when provided" do
      expect(Req, :post, fn req_struct, _opts ->
        assert req_struct.headers["anthropic-beta"] == ["beta1"]
      end)

      model = ChatAnthropic.new!(%{stream: true, model: @test_model, beta_headers: ["beta1"]})
      ChatAnthropic.call(model, "prompt", [])
    end

    test "joins multiple beta headers with commas" do
      expect(Req, :post, fn req_struct, _opts ->
        assert req_struct.headers["anthropic-beta"] == ["beta1,beta2"]
      end)

      model =
        ChatAnthropic.new!(%{stream: true, model: @test_model, beta_headers: ["beta1", "beta2"]})

      ChatAnthropic.call(model, "prompt", [])
    end

    test "does not add anthropic-beta header when beta_headers is empty" do
      expect(Req, :post, fn req_struct, _opts ->
        refute Map.has_key?(req_struct.headers, "anthropic-beta")
      end)

      model =
        ChatAnthropic.new!(%{stream: true, model: @test_model, beta_headers: []})

      ChatAnthropic.call(model, "prompt", [])
    end

    test "defaults to tools-2024-04-04 when beta_headers is not provided" do
      expect(Req, :post, fn req_struct, _opts ->
        assert req_struct.headers["anthropic-beta"] == ["tools-2024-04-04"]
      end)

      model = ChatAnthropic.new!(%{stream: true, model: @test_model})

      ChatAnthropic.call(model, "prompt", [])
    end
  end

  describe "cache_messages for multi-turn conversations" do
    test "when cache_messages is enabled, adds cache_control to last user message's last ContentPart" do
      anthropic = ChatAnthropic.new!(%{cache_messages: %{enabled: true}})

      messages = [
        Message.new_user!("First message"),
        Message.new_assistant!("First response"),
        Message.new_user!("Second message")
      ]

      data = ChatAnthropic.for_api(anthropic, messages, [])

      # Find the last user message in the API data
      last_user_message = List.last(data.messages)

      assert last_user_message["role"] == "user"
      # Get the last content part
      last_content = List.last(last_user_message["content"])
      # Should have cache_control set
      assert last_content["cache_control"] == %{"type" => "ephemeral"}
    end

    test "when cache_messages is disabled, does not add cache_control automatically" do
      anthropic = ChatAnthropic.new!(%{cache_messages: %{enabled: false}})

      messages = [
        Message.new_user!("First message"),
        Message.new_assistant!("First response"),
        Message.new_user!("Second message")
      ]

      data = ChatAnthropic.for_api(anthropic, messages, [])

      # Find the last user message in the API data
      last_user_message = List.last(data.messages)

      assert last_user_message["role"] == "user"
      # Get the last content part
      last_content = List.last(last_user_message["content"])
      # Should NOT have cache_control
      refute Map.has_key?(last_content, "cache_control")
    end

    test "when cache_messages is not set, does not add cache_control automatically" do
      anthropic = ChatAnthropic.new!(%{})

      messages = [
        Message.new_user!("First message"),
        Message.new_assistant!("First response"),
        Message.new_user!("Second message")
      ]

      data = ChatAnthropic.for_api(anthropic, messages, [])

      # Find the last user message in the API data
      last_user_message = List.last(data.messages)

      assert last_user_message["role"] == "user"
      # Get the last content part
      last_content = List.last(last_user_message["content"])
      # Should NOT have cache_control
      refute Map.has_key?(last_content, "cache_control")
    end

    test "cache_messages works with multi-part messages" do
      anthropic = ChatAnthropic.new!(%{cache_messages: %{enabled: true}})

      messages = [
        Message.new_user!([
          ContentPart.text!("First part"),
          ContentPart.text!("Second part")
        ]),
        Message.new_assistant!("Response"),
        Message.new_user!([
          ContentPart.text!("Question about document"),
          ContentPart.text!("Additional context")
        ])
      ]

      data = ChatAnthropic.for_api(anthropic, messages, [])

      # Find the last user message
      last_user_message = List.last(data.messages)

      assert last_user_message["role"] == "user"
      content_parts = last_user_message["content"]
      assert length(content_parts) == 2

      # Only the LAST content part should have cache_control
      first_part = Enum.at(content_parts, 0)
      refute Map.has_key?(first_part, "cache_control")

      last_part = List.last(content_parts)
      assert last_part["cache_control"] == %{"type" => "ephemeral"}
    end

    test "cache_messages supports TTL configuration with map value" do
      anthropic = ChatAnthropic.new!(%{cache_messages: %{enabled: true, ttl: "1h"}})

      messages = [
        Message.new_user!("First message"),
        Message.new_assistant!("Response"),
        Message.new_user!("Second message")
      ]

      data = ChatAnthropic.for_api(anthropic, messages, [])

      # Find the last user message
      last_user_message = List.last(data.messages)

      assert last_user_message["role"] == "user"
      last_content = List.last(last_user_message["content"])
      assert last_content["cache_control"] == %{"type" => "ephemeral", "ttl" => "1h"}
    end

    test "cache_messages with tool results in conversation" do
      anthropic = ChatAnthropic.new!(%{cache_messages: %{enabled: true}})

      messages = [
        Message.new_user!("Use a tool"),
        Message.new_assistant!(%{
          tool_calls: [ToolCall.new!(%{call_id: "call_123", name: "greet", arguments: nil})]
        }),
        Message.new_tool_result!(%{
          tool_results: [ToolResult.new!(%{tool_call_id: "call_123", content: "result"})]
        }),
        Message.new_assistant!("Tool executed"),
        Message.new_user!("What was the result?")
      ]

      data = ChatAnthropic.for_api(anthropic, messages, [])

      # The last message in the API should be the last user message
      # Note: tool results get combined with the previous user message by post_process
      last_message = List.last(data.messages)

      assert last_message["role"] == "user"
      # Find the last text content part (not tool_result)
      text_parts =
        Enum.filter(last_message["content"], fn part -> part["type"] == "text" end)

      last_text_part = List.last(text_parts)
      assert last_text_part["cache_control"] == %{"type" => "ephemeral"}
    end

    test "cache_messages does not override explicit cache_control in ContentParts" do
      anthropic = ChatAnthropic.new!(%{cache_messages: %{enabled: true}})

      messages = [
        Message.new_user!([
          ContentPart.text!("First part", cache_control: %{"type" => "ephemeral", "ttl" => "5m"}),
          ContentPart.text!("Second part")
        ])
      ]

      data = ChatAnthropic.for_api(anthropic, messages, [])

      last_user_message = List.last(data.messages)
      content_parts = last_user_message["content"]

      # First part should keep its explicit cache_control
      first_part = Enum.at(content_parts, 0)
      assert first_part["cache_control"] == %{"type" => "ephemeral", "ttl" => "5m"}

      # Last part should get the cache_messages setting
      last_part = List.last(content_parts)
      assert last_part["cache_control"] == %{"type" => "ephemeral"}
    end

    test "simulates moving cache breakpoint in multi-turn conversation" do
      anthropic = ChatAnthropic.new!(%{cache_messages: %{enabled: true}})

      # First turn
      messages_turn1 = [
        Message.new_user!("First question")
      ]

      data1 = ChatAnthropic.for_api(anthropic, messages_turn1, [])
      last_msg1 = List.last(data1.messages)
      assert last_msg1["role"] == "user"
      last_content1 = List.last(last_msg1["content"])
      assert last_content1["cache_control"] == %{"type" => "ephemeral"}

      # Second turn - cache should move to the new last user message
      messages_turn2 = [
        Message.new_user!("First question"),
        Message.new_assistant!("First answer"),
        Message.new_user!("Second question")
      ]

      data2 = ChatAnthropic.for_api(anthropic, messages_turn2, [])

      # The last user message should have cache_control
      last_msg2 = List.last(data2.messages)
      assert last_msg2["role"] == "user"
      last_content2 = List.last(last_msg2["content"])
      assert last_content2["cache_control"] == %{"type" => "ephemeral"}

      # The first user message should NOT have cache_control
      first_msg2 = Enum.at(data2.messages, 0)
      assert first_msg2["role"] == "user"
      first_content2 = List.last(first_msg2["content"])
      refute Map.has_key?(first_content2, "cache_control")
    end
  end

  describe "inspect" do
    test "redacts the API key" do
      chain = ChatAnthropic.new!()

      changeset = Ecto.Changeset.cast(chain, %{api_key: "1234567890"}, [:api_key])

      refute inspect(changeset) =~ "1234567890"
      assert inspect(changeset) =~ "**redacted**"
    end
  end
end
