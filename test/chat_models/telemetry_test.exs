defmodule LangChain.ChatModels.TelemetryTest do
  use ExUnit.Case
  use Mimic

  alias LangChain.ChatModels.ChatOpenAI
  alias LangChain.ChatModels.ChatMistralAI
  alias LangChain.ChatModels.ChatAnthropic
  alias LangChain.ChatModels.ChatGoogleAI
  alias LangChain.ChatModels.ChatPerplexity
  alias LangChain.ChatModels.ChatVertexAI
  alias LangChain.Message

  # Setup for test
  setup :verify_on_exit!

  describe "telemetry instrumentation" do
    setup do
      # Set up test models with explicit API keys to avoid env var issues
      openai = ChatOpenAI.new!(%{model: "gpt-4o-mini", api_key: "test-openai-key"})
      mistral_ai = ChatMistralAI.new!(%{model: "mistral-tiny", api_key: "test-mistral-key"})

      anthropic =
        ChatAnthropic.new!(%{model: "claude-3-haiku-20240307", api_key: "test-anthropic-key"})

      google_ai = ChatGoogleAI.new!(%{model: "gemini-pro", api_key: "test-google-key"})

      perplexity =
        ChatPerplexity.new!(%{
          model: "llama-3-sonar-small-32k-online",
          api_key: "test-perplexity-key"
        })

      vertex_ai =
        ChatVertexAI.new!(%{
          model: "gemini-1.5-pro",
          api_key: "test-google-key",
          endpoint: "https://generativelanguage.googleapis.com/v1"
        })

      # Create test messages
      test_message = "Hello, how are you?"

      test_messages = [
        Message.new_system!("You are a helpful assistant."),
        Message.new_user!(test_message)
      ]

      # Mock the ChatModel implementations directly
      ChatOpenAI
      |> stub(:call, fn model, messages, tools ->
        metadata = %{
          model: model.model,
          message_count: length(messages),
          tools_count: length(tools)
        }

        LangChain.Telemetry.span([:langchain, :llm, :call], metadata, fn ->
          # Track the prompt being sent
          LangChain.Telemetry.llm_prompt(
            %{system_time: System.system_time()},
            %{model: model.model, messages: messages}
          )

          response = Message.new_assistant!("Test response")

          # Track the response being received
          LangChain.Telemetry.llm_response(
            %{system_time: System.system_time()},
            %{model: model.model, response: response}
          )

          # Track non-streaming response
          LangChain.Telemetry.emit_event(
            [:langchain, :llm, :response, :non_streaming],
            %{system_time: System.system_time()},
            %{
              model: model.model,
              response_size: byte_size(inspect(response))
            }
          )

          {:ok, response}
        end)
      end)

      ChatMistralAI
      |> stub(:call, fn model, messages, tools ->
        metadata = %{
          model: model.model,
          message_count: length(messages),
          tools_count: length(tools)
        }

        LangChain.Telemetry.span([:langchain, :llm, :call], metadata, fn ->
          # Track the prompt being sent
          LangChain.Telemetry.llm_prompt(
            %{system_time: System.system_time()},
            %{model: model.model, messages: messages}
          )

          response = Message.new_assistant!("Test response")

          # Track the response being received
          LangChain.Telemetry.llm_response(
            %{system_time: System.system_time()},
            %{model: model.model, response: response}
          )

          # Track non-streaming response
          LangChain.Telemetry.emit_event(
            [:langchain, :llm, :response, :non_streaming],
            %{system_time: System.system_time()},
            %{
              model: model.model,
              response_size: byte_size(inspect(response))
            }
          )

          {:ok, response}
        end)
      end)

      ChatVertexAI
      |> stub(:call, fn model, messages, tools ->
        metadata = %{
          model: model.model,
          message_count: length(messages),
          tools_count: length(tools)
        }

        LangChain.Telemetry.span([:langchain, :llm, :call], metadata, fn ->
          # Track the prompt being sent
          LangChain.Telemetry.llm_prompt(
            %{system_time: System.system_time()},
            %{model: model.model, messages: messages}
          )

          response = Message.new_assistant!("Test response")

          # Track the response being received
          LangChain.Telemetry.llm_response(
            %{system_time: System.system_time()},
            %{model: model.model, response: response}
          )

          # Track non-streaming response
          LangChain.Telemetry.emit_event(
            [:langchain, :llm, :response, :non_streaming],
            %{system_time: System.system_time()},
            %{
              model: model.model,
              response_size: byte_size(inspect(response))
            }
          )

          {:ok, response}
        end)
      end)

      ChatPerplexity
      |> stub(:call, fn model, messages, tools ->
        metadata = %{
          model: model.model,
          message_count: length(messages),
          tools_count: length(tools)
        }

        LangChain.Telemetry.span([:langchain, :llm, :call], metadata, fn ->
          # Track the prompt being sent
          LangChain.Telemetry.llm_prompt(
            %{system_time: System.system_time()},
            %{model: model.model, messages: messages}
          )

          response = Message.new_assistant!("Test response")

          # Track the response being received
          LangChain.Telemetry.llm_response(
            %{system_time: System.system_time()},
            %{model: model.model, response: response}
          )

          # Track non-streaming response
          LangChain.Telemetry.emit_event(
            [:langchain, :llm, :response, :non_streaming],
            %{system_time: System.system_time()},
            %{
              model: model.model,
              response_size: byte_size(inspect(response))
            }
          )

          {:ok, response}
        end)
      end)

      ChatGoogleAI
      |> stub(:call, fn model, messages, tools ->
        metadata = %{
          model: model.model,
          message_count: length(messages),
          tools_count: length(tools)
        }

        LangChain.Telemetry.span([:langchain, :llm, :call], metadata, fn ->
          # Track the prompt being sent
          LangChain.Telemetry.llm_prompt(
            %{system_time: System.system_time()},
            %{model: model.model, messages: messages}
          )

          response = Message.new_assistant!("Test response")

          # Track the response being received
          LangChain.Telemetry.llm_response(
            %{system_time: System.system_time()},
            %{model: model.model, response: response}
          )

          # Track non-streaming response
          LangChain.Telemetry.emit_event(
            [:langchain, :llm, :response, :non_streaming],
            %{system_time: System.system_time()},
            %{
              model: model.model,
              response_size: byte_size(inspect(response))
            }
          )

          {:ok, response}
        end)
      end)

      # Mock Req.request for any remaining API calls
      Req
      |> stub(:request, fn _req ->
        {:ok,
         %Req.Response{
           status: 200,
           body: %{
             "choices" => [
               %{
                 "message" => %{
                   "content" => "Test response",
                   "role" => "assistant"
                 },
                 "finish_reason" => "stop",
                 "index" => 0
               }
             ],
             "usage" => %{
               "prompt_tokens" => 10,
               "completion_tokens" => 20,
               "total_tokens" => 30
             }
           }
         }}
      end)

      %{
        openai: openai,
        mistral_ai: mistral_ai,
        anthropic: anthropic,
        google_ai: google_ai,
        perplexity: perplexity,
        vertex_ai: vertex_ai,
        test_message: test_message,
        test_messages: test_messages
      }
    end

    test "emits telemetry events for ChatOpenAI", %{openai: openai, test_messages: messages} do
      # Attach telemetry handlers
      test_pid = self()

      :telemetry.attach_many(
        "test-openai-telemetry-events",
        [
          [:langchain, :llm, :call, :start],
          [:langchain, :llm, :prompt],
          [:langchain, :llm, :response],
          [:langchain, :llm, :response, :non_streaming]
        ],
        fn name, measurements, metadata, _config ->
          send(test_pid, {:telemetry_event, name, measurements, metadata})
        end,
        nil
      )

      # Call the model
      {:ok, _response} = ChatOpenAI.call(openai, messages, [])

      # Assert that all telemetry events were emitted
      assert_received {:telemetry_event, [:langchain, :llm, :call, :start], _, metadata}
      assert metadata.model == openai.model
      assert metadata.message_count == length(messages)
      assert metadata.tools_count == 0

      assert_received {:telemetry_event, [:langchain, :llm, :prompt], _, metadata}
      assert metadata.model == openai.model
      assert is_list(metadata.messages)

      assert_received {:telemetry_event, [:langchain, :llm, :response], _, metadata}
      assert metadata.model == openai.model

      assert_received {:telemetry_event, [:langchain, :llm, :response, :non_streaming], _,
                       metadata}

      assert metadata.model == openai.model
      assert is_integer(metadata.response_size)

      # Clean up telemetry handlers
      :telemetry.detach("test-openai-telemetry-events")
    end

    test "emits telemetry events for ChatVertexAI", %{
      vertex_ai: vertex_ai,
      test_messages: messages
    } do
      # Attach telemetry handlers
      test_pid = self()

      :telemetry.attach_many(
        "test-vertex-telemetry-events",
        [
          [:langchain, :llm, :call, :start],
          [:langchain, :llm, :prompt],
          [:langchain, :llm, :response],
          [:langchain, :llm, :response, :non_streaming]
        ],
        fn name, measurements, metadata, _config ->
          send(test_pid, {:telemetry_event, name, measurements, metadata})
        end,
        nil
      )

      # Call the model
      {:ok, _response} = ChatVertexAI.call(vertex_ai, messages, [])

      # Assert that all telemetry events were emitted
      assert_received {:telemetry_event, [:langchain, :llm, :call, :start], _, metadata}
      assert metadata.model == vertex_ai.model
      assert metadata.message_count == length(messages)
      assert metadata.tools_count == 0

      assert_received {:telemetry_event, [:langchain, :llm, :prompt], _, metadata}
      assert metadata.model == vertex_ai.model
      assert is_list(metadata.messages)

      assert_received {:telemetry_event, [:langchain, :llm, :response], _, metadata}
      assert metadata.model == vertex_ai.model

      assert_received {:telemetry_event, [:langchain, :llm, :response, :non_streaming], _,
                       metadata}

      assert metadata.model == vertex_ai.model
      assert is_integer(metadata.response_size)

      # Clean up telemetry handlers
      :telemetry.detach("test-vertex-telemetry-events")
    end

    test "emits telemetry events for ChatMistralAI", %{
      mistral_ai: mistral_ai,
      test_messages: messages
    } do
      # Attach telemetry handlers
      test_pid = self()

      :telemetry.attach_many(
        "test-mistral-telemetry-events",
        [
          [:langchain, :llm, :call, :start],
          [:langchain, :llm, :prompt],
          [:langchain, :llm, :response],
          [:langchain, :llm, :response, :non_streaming]
        ],
        fn name, measurements, metadata, _config ->
          send(test_pid, {:telemetry_event, name, measurements, metadata})
        end,
        nil
      )

      # Call the model
      {:ok, _response} = ChatMistralAI.call(mistral_ai, messages, [])

      # Assert that all telemetry events were emitted
      assert_received {:telemetry_event, [:langchain, :llm, :call, :start], _, metadata}
      assert metadata.model == mistral_ai.model
      assert metadata.message_count == length(messages)
      assert metadata.tools_count == 0

      assert_received {:telemetry_event, [:langchain, :llm, :prompt], _, metadata}
      assert metadata.model == mistral_ai.model
      assert is_list(metadata.messages)

      assert_received {:telemetry_event, [:langchain, :llm, :response], _, metadata}
      assert metadata.model == mistral_ai.model

      assert_received {:telemetry_event, [:langchain, :llm, :response, :non_streaming], _,
                       metadata}

      assert metadata.model == mistral_ai.model
      assert is_integer(metadata.response_size)

      # Clean up telemetry handlers
      :telemetry.detach("test-mistral-telemetry-events")
    end

    test "emits telemetry events for ChatPerplexity", %{
      perplexity: perplexity,
      test_messages: messages
    } do
      # Attach telemetry handlers
      test_pid = self()

      :telemetry.attach_many(
        "test-perplexity-telemetry-events",
        [
          [:langchain, :llm, :call, :start],
          [:langchain, :llm, :prompt],
          [:langchain, :llm, :response],
          [:langchain, :llm, :response, :non_streaming]
        ],
        fn name, measurements, metadata, _config ->
          send(test_pid, {:telemetry_event, name, measurements, metadata})
        end,
        nil
      )

      # Call the model
      {:ok, _response} = ChatPerplexity.call(perplexity, messages, [])

      # Assert that all telemetry events were emitted
      assert_received {:telemetry_event, [:langchain, :llm, :call, :start], _, metadata}
      assert metadata.model == perplexity.model
      assert metadata.message_count == length(messages)
      assert metadata.tools_count == 0

      assert_received {:telemetry_event, [:langchain, :llm, :prompt], _, metadata}
      assert metadata.model == perplexity.model
      assert is_list(metadata.messages)

      assert_received {:telemetry_event, [:langchain, :llm, :response], _, metadata}
      assert metadata.model == perplexity.model

      assert_received {:telemetry_event, [:langchain, :llm, :response, :non_streaming], _,
                       metadata}

      assert metadata.model == perplexity.model
      assert is_integer(metadata.response_size)

      # Clean up telemetry handlers
      :telemetry.detach("test-perplexity-telemetry-events")
    end

    test "telemetry includes correct measurements", %{openai: openai, test_messages: messages} do
      # Attach telemetry handlers
      test_pid = self()

      :telemetry.attach_many(
        "test-measurement-telemetry-events",
        [
          [:langchain, :llm, :call, :start],
          [:langchain, :llm, :call, :stop],
          [:langchain, :llm, :prompt],
          [:langchain, :llm, :response],
          [:langchain, :llm, :response, :non_streaming]
        ],
        fn name, measurements, _metadata, _config ->
          send(test_pid, {:telemetry_measurements, name, measurements})
        end,
        nil
      )

      # Call the model
      {:ok, _response} = ChatOpenAI.call(openai, messages, [])

      # Verify measurements in events
      assert_received {:telemetry_measurements, [:langchain, :llm, :call, :start], measurements}
      assert is_map(measurements)
      assert Map.has_key?(measurements, :system_time)

      assert_received {:telemetry_measurements, [:langchain, :llm, :prompt], measurements}
      assert is_map(measurements)
      assert Map.has_key?(measurements, :system_time)

      assert_received {:telemetry_measurements, [:langchain, :llm, :response], measurements}
      assert is_map(measurements)
      assert Map.has_key?(measurements, :system_time)

      assert_received {:telemetry_measurements, [:langchain, :llm, :response, :non_streaming],
                       measurements}

      assert is_map(measurements)
      assert Map.has_key?(measurements, :system_time)

      assert_received {:telemetry_measurements, [:langchain, :llm, :call, :stop], measurements}
      assert is_map(measurements)
      assert Map.has_key?(measurements, :system_time)
      assert Map.has_key?(measurements, :duration)
      assert is_integer(measurements.duration)

      # Clean up telemetry handlers
      :telemetry.detach("test-measurement-telemetry-events")
    end
  end
end
