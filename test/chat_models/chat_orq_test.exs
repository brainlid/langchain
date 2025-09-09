defmodule LangChain.ChatModels.ChatOrqTest do
  use LangChain.BaseCase

  alias LangChain.ChatModels.ChatOrq
  alias LangChain.Function
  alias LangChain.FunctionParam
  alias LangChain.Message
  alias LangChain.Message.ContentPart
  alias LangChain.MessageDelta
  alias LangChain.TokenUsage

  describe "do_process_response/2 - non streaming" do
    test "handles receiving a message and attaches token usage" do
      model = ChatOrq.new!(%{key: "deployment_key"})

      response = %{
        "choices" => [
          %{
            "message" => %{
              "role" => "assistant",
              "content" => "Hello from orq!"
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

      assert [%Message{} = msg] = ChatOrq.do_process_response(model, response)
      assert msg.role == :assistant
      assert msg.status == :complete
      assert msg.index == 0
      assert msg.content == [ContentPart.text!("Hello from orq!")]

      # token usage attached to metadata
      assert %TokenUsage{} = usage = msg.metadata.usage
      assert usage.input == 7
      assert usage.output == 10

      assert usage.raw == %{
               "prompt_tokens" => 7,
               "completion_tokens" => 10,
               "total_tokens" => 17
             }
    end
  end

  describe "do_process_response/2 - streaming" do
    test "parses basic text deltas and final completion" do
      model = ChatOrq.new!(%{key: "deployment_key", stream: true})

      deltas = [
        %{
          "choices" => [
            %{
              "delta" => %{"content" => "", "role" => "assistant"},
              "finish_reason" => nil,
              "index" => 0
            }
          ]
        },
        %{
          "choices" => [
            %{
              "delta" => %{"content" => "Colorful"},
              "finish_reason" => nil,
              "index" => 0
            }
          ]
        },
        %{
          "choices" => [
            %{
              "delta" => %{"content" => " Threads"},
              "finish_reason" => nil,
              "index" => 0
            }
          ]
        },
        %{
          "choices" => [
            %{
              "delta" => %{},
              "finish_reason" => "stop",
              "index" => 0
            }
          ]
        }
      ]

      [d1, d2, d3, d4] =
        Enum.map(deltas, fn d ->
          # ChatOrq.do_process_response returns list for "choices"
          [delta] = ChatOrq.do_process_response(model, d)
          delta
        end)

      assert %MessageDelta{
               role: :assistant,
               content: nil,
               status: :incomplete,
               index: 0
             } = d1

      assert %MessageDelta{
               role: :unknown,
               content: %ContentPart{type: :text, content: "Colorful"},
               status: :incomplete,
               index: 0
             } = d2

      assert %MessageDelta{
               role: :unknown,
               content: %ContentPart{type: :text, content: " Threads"},
               status: :incomplete,
               index: 0
             } = d3

      # The final delta has empty content, which should be nil after processing
      assert %MessageDelta{
               role: :unknown,
               status: :complete,
               index: 0
             } = d4

      # Content should be empty list for empty delta body
      assert d4.content == []
    end

    test "handles streaming tool call deltas correctly" do
      model = ChatOrq.new!(%{key: "deployment_key", stream: true})

      # Simulate the exact streaming tool call data that was causing the error
      streaming_deltas = [
        # First chunk with tool call initialization
        %{
          "choices" => [
            %{
              "delta" => %{
                "role" => "assistant",
                "tool_calls" => [
                  %{
                    "function" => %{"arguments" => "", "name" => "add"},
                    "id" => "call_Q2wAHnrzlZCLQLKAWyxK9PHh",
                    "index" => 0,
                    "type" => "function"
                  }
                ]
              },
              "finish_reason" => nil,
              "index" => 0
            }
          ]
        },
        # Second chunk with partial arguments
        %{
          "choices" => [
            %{
              "delta" => %{
                "tool_calls" => [
                  %{
                    "function" => %{"arguments" => "{\""},
                    "index" => 0
                  }
                ]
              },
              "finish_reason" => nil,
              "index" => 0
            }
          ]
        },
        # Third chunk with more arguments
        %{
          "choices" => [
            %{
              "delta" => %{
                "tool_calls" => [
                  %{
                    "function" => %{"arguments" => "a"},
                    "index" => 0
                  }
                ]
              },
              "finish_reason" => nil,
              "index" => 0
            }
          ]
        },
        # Fourth chunk completing the arguments
        %{
          "choices" => [
            %{
              "delta" => %{
                "tool_calls" => [
                  %{
                    "function" => %{"arguments" => "\":31490,\"b\":112722}"},
                    "index" => 0
                  }
                ]
              },
              "finish_reason" => nil,
              "index" => 0
            }
          ]
        },
        # Final chunk with completion
        %{
          "choices" => [
            %{
              "delta" => %{},
              "finish_reason" => "tool_calls",
              "index" => 0
            }
          ]
        }
      ]

      # Process each delta and verify they don't cause errors
      processed_deltas =
        Enum.map(streaming_deltas, fn delta ->
          # ChatOrq.do_process_response returns list for "choices"
          [processed] = ChatOrq.do_process_response(model, delta)
          processed
        end)

      # Verify the first delta has the tool call structure
      [d1, d2, d3, d4, d5] = processed_deltas

      # First delta should have role and tool calls
      assert %MessageDelta{
               role: :assistant,
               status: :incomplete,
               index: 0,
               tool_calls: [tool_call]
             } = d1

      assert tool_call.name == "add"
      assert tool_call.call_id == "call_Q2wAHnrzlZCLQLKAWyxK9PHh"
      # Empty string gets converted to nil
      assert tool_call.arguments == nil
      assert tool_call.index == 0
      assert tool_call.status == :incomplete

      # Second delta should have partial arguments
      assert %MessageDelta{
               role: :unknown,
               status: :incomplete,
               index: 0,
               tool_calls: [tool_call2]
             } = d2

      assert tool_call2.arguments == "{\""
      assert tool_call2.index == 0

      # Third delta should have more arguments
      assert %MessageDelta{
               role: :unknown,
               status: :incomplete,
               index: 0,
               tool_calls: [tool_call3]
             } = d3

      assert tool_call3.arguments == "a"
      assert tool_call3.index == 0

      # Fourth delta should have completed arguments
      assert %MessageDelta{
               role: :unknown,
               status: :incomplete,
               index: 0,
               tool_calls: [tool_call4]
             } = d4

      assert tool_call4.arguments == "\":31490,\"b\":112722}"
      assert tool_call4.index == 0

      # Final delta should be complete
      assert %MessageDelta{
               role: :unknown,
               status: :complete,
               index: 0,
               tool_calls: nil
             } = d5

      # Test merging the deltas to ensure they combine correctly
      merged = MessageDelta.merge_deltas(processed_deltas)
      assert merged.status == :complete
      assert length(merged.tool_calls) == 1

      final_tool_call = List.first(merged.tool_calls)
      assert final_tool_call.name == "add"
      assert final_tool_call.call_id == "call_Q2wAHnrzlZCLQLKAWyxK9PHh"
      # The arguments should be merged from all the chunks
      assert final_tool_call.arguments == "{\"a\":31490,\"b\":112722}"
    end
  end

  describe "for_api/2 - tool result handling" do
    test "converts ToolResult content to string for API" do
      model = ChatOrq.new!(%{key: "deployment_key"})

      # Create a ToolResult with ContentPart list (the problematic case)
      tool_result = %LangChain.Message.ToolResult{
        type: :function,
        tool_call_id: "call_test123",
        content: [
          %ContentPart{type: :text, content: "230521"}
        ]
      }

      # Convert to API format
      api_data = ChatOrq.for_api(model, tool_result)

      # Verify the content is a string, not an array
      assert api_data["role"] == :tool
      assert api_data["tool_call_id"] == "call_test123"
      assert api_data["content"] == "230521"
      assert is_binary(api_data["content"])
    end

    test "converts Message with tool_results to string content for API" do
      model = ChatOrq.new!(%{key: "deployment_key"})

      # Create a Message with tool_results (another problematic case)
      tool_result = %LangChain.Message.ToolResult{
        type: :function,
        tool_call_id: "call_test456",
        content: [
          %ContentPart{type: :text, content: "Hello"},
          %ContentPart{type: :text, content: " World"}
        ]
      }

      message = %Message{
        role: :tool,
        tool_results: [tool_result]
      }

      # Convert to API format
      api_data = ChatOrq.for_api(model, message)

      # Should return a list of tool messages
      assert is_list(api_data)
      assert length(api_data) == 1

      [tool_msg] = api_data
      assert tool_msg["role"] == :tool
      assert tool_msg["tool_call_id"] == "call_test456"
      assert tool_msg["content"] == "Hello World"
      assert is_binary(tool_msg["content"])
    end

    test "handles binary content directly" do
      model = ChatOrq.new!(%{key: "deployment_key"})

      # Create a ToolResult with direct string content
      tool_result = %LangChain.Message.ToolResult{
        type: :function,
        tool_call_id: "call_test789",
        content: "Direct string content"
      }

      # Convert to API format
      api_data = ChatOrq.for_api(model, tool_result)

      # Verify the content remains a string
      assert api_data["role"] == :tool
      assert api_data["tool_call_id"] == "call_test789"
      assert api_data["content"] == "Direct string content"
      assert is_binary(api_data["content"])
    end
  end

  describe "call/2 - LIVE" do
    # Skip live API calls in CI
    @tag live_call: true, live_orq_ai: true
    test "LIVE non-streamed basic content example" do
      deployment_key = System.fetch_env!("ORQ_DEPLOYMENT_KEY")
      api_key = System.get_env("ORQ_API_KEY")

      {:ok, chat} =
        ChatOrq.new(%{
          key: deployment_key,
          api_key: api_key,
          stream: false
        })

      {:ok, [%Message{role: :assistant, content: content}]} =
        ChatOrq.call(chat, [
          Message.new_user!("Return the response 'Hello World'.")
        ])

      assert is_list(content)
      assert ContentPart.parts_to_string(content) =~ "Hello World"
    end

    # Skip live API calls in CI
    @tag live_call: true, live_orq_ai: true
    test "LIVE streamed basic content example" do
      test_pid = self()

      handlers = %{
        on_llm_new_delta: fn deltas ->
          send(test_pid, deltas)
        end
      }

      deployment_key = System.fetch_env!("ORQ_DEPLOYMENT_KEY")
      api_key = System.get_env("ORQ_API_KEY")

      chat =
        ChatOrq.new!(%{
          key: deployment_key,
          api_key: api_key,
          stream: true,
          callbacks: [handlers]
        })

      {:ok, _result} =
        ChatOrq.call(chat, [
          Message.new_user!(
            "Write a detailed 200-word explanation about the benefits of streaming responses in AI applications. Include technical details and use cases."
          )
        ])

      # we expect to receive the response over multiple delta messages
      assert_receive deltas_1, 2000
      assert_receive deltas_2, 2000

      # Extract first delta from each list
      delta_1 = List.first(deltas_1)
      delta_2 = List.first(deltas_2)

      # Verify the deltas are properly formed
      assert delta_1.role == :assistant
      assert delta_2.role == :assistant
      assert delta_1.status == :incomplete
      assert delta_2.status == :incomplete

      # Merge the deltas
      merged = MessageDelta.merge_delta(delta_1, delta_2)

      assert merged.role == :assistant
      # The merged content should contain text from both deltas
      merged_text = merged.content || ""
      assert is_binary(merged_text)
      # Most deltas will be incomplete until the final one
      assert merged.status == :incomplete
    end

    @tag live_call: true, live_orq: true
    test "supports tools/functions with non-streaming" do
      deployment_key = System.fetch_env!("ORQ_DEPLOYMENT_KEY")
      api_key = System.get_env("ORQ_API_KEY")

      {:ok, chat} =
        ChatOrq.new(%{
          model: "meta-llama/llama-3.1-8b-instruct",
          key: deployment_key,
          api_key: api_key,
          stream: false
        })

      # Create a weather function for testing
      weather_function =
        Function.new!(%{
          name: "get_weather",
          description: "Get the current weather for a given location",
          parameters: [
            FunctionParam.new!(%{
              name: "location",
              type: :string,
              description: "The city and state, e.g. San Francisco, CA",
              required: true
            }),
            FunctionParam.new!(%{
              name: "unit",
              type: :string,
              description: "Temperature unit (celsius or fahrenheit)",
              required: false
            })
          ],
          function: fn args, _context ->
            location = Map.get(args, "location", "unknown")
            unit = Map.get(args, "unit", "fahrenheit")

            weather_data = %{
              location: location,
              temperature: if(unit == "celsius", do: "22°C", else: "72°F"),
              condition: "sunny",
              humidity: "45%"
            }

            {:ok, Jason.encode!(weather_data)}
          end
        })

      {:ok, [message]} =
        ChatOrq.call(
          chat,
          [
            Message.new_user!("What's the weather like in San Francisco, CA? Use celsius.")
          ],
          [weather_function]
        )

      assert %Message{role: :assistant} = message
      assert message.tool_calls != nil
      assert length(message.tool_calls) >= 1

      # Find the weather tool call
      weather_call = Enum.find(message.tool_calls, fn call -> call.name == "get_weather" end)
      assert weather_call != nil
      assert weather_call.arguments["location"] =~ "San Francisco"
    end

    @tag live_call: true, live_orq: true
    test "supports tools/functions with streaming" do
      deployment_key = System.fetch_env!("ORQ_DEPLOYMENT_KEY")
      api_key = System.get_env("ORQ_API_KEY")
      test_pid = self()

      handler = %{
        on_llm_new_delta: fn _model, delta ->
          send(test_pid, {:delta_received, delta})
          :ok
        end
      }

      {:ok, chat} =
        ChatOrq.new(%{
          model: "meta-llama/llama-3.1-8b-instruct",
          key: deployment_key,
          api_key: api_key,
          stream: true,
          callbacks: [handler]
        })

      # Create a weather function for testing
      weather_function =
        Function.new!(%{
          name: "get_weather",
          description: "Get the current weather for a given location",
          parameters: [
            FunctionParam.new!(%{
              name: "location",
              type: :string,
              description: "The city and state, e.g. San Francisco, CA",
              required: true
            }),
            FunctionParam.new!(%{
              name: "unit",
              type: :string,
              description: "Temperature unit (celsius or fahrenheit)",
              required: false
            })
          ],
          function: fn args, _context ->
            location = Map.get(args, "location", "unknown")
            unit = Map.get(args, "unit", "fahrenheit")

            weather_data = %{
              location: location,
              temperature: if(unit == "celsius", do: "22°C", else: "72°F"),
              condition: "sunny",
              humidity: "45%"
            }

            {:ok, Jason.encode!(weather_data)}
          end
        })

      {:ok, [message]} =
        ChatOrq.call(
          chat,
          [
            Message.new_user!("What's the weather like in New York, NY? Use fahrenheit.")
          ],
          [weather_function]
        )

      # Should receive streaming deltas
      assert_receive {:delta_received, deltas}, 5000
      assert is_list(deltas)

      # Final message should have tool calls
      assert %Message{role: :assistant} = message
      assert message.tool_calls != nil
      assert length(message.tool_calls) >= 1

      # Find the weather tool call
      weather_call = Enum.find(message.tool_calls, fn call -> call.name == "get_weather" end)
      assert weather_call != nil
      assert weather_call.arguments["location"] =~ "New York"
    end
  end
end
