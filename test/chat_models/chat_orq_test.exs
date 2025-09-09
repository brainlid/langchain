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
               content: "",
               status: :incomplete,
               index: 0
             } = d1

      assert %MessageDelta{
               role: :unknown,
               content: "Colorful",
               status: :incomplete,
               index: 0
             } = d2

      assert %MessageDelta{
               role: :unknown,
               content: " Threads",
               status: :incomplete,
               index: 0
             } = d3

      assert %MessageDelta{
               role: :unknown,
               content: nil,
               status: :complete,
               index: 0
             } = d4
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
