defmodule LangChain.ChatModels.TelemetryTest do
  use ExUnit.Case
  use Mimic

  alias LangChain.ChatModels.ChatOpenAI
  alias LangChain.ChatModels.ChatVertexAI
  alias LangChain.ChatModels.ChatMistralAI
  alias LangChain.ChatModels.ChatPerplexity
  alias LangChain.Message

  # Setup for test
  setup :verify_on_exit!

  describe "telemetry instrumentation" do
    setup do
      # Set up test models with explicit API keys to avoid env var issues
      openai = ChatOpenAI.new!(%{model: "gpt-4o-mini", api_key: "test-openai-key"})
      vertex_ai = ChatVertexAI.new!(%{
        model: "gemini-1.5-pro",
        api_key: "test-google-key",
        endpoint: "https://generativelanguage.googleapis.com/v1"
      })
      mistral_ai = ChatMistralAI.new!(%{model: "mistral-large-latest", api_key: "test-mistral-key"})
      perplexity = ChatPerplexity.new!(%{model: "sonar-small-online", api_key: "test-perplexity-key"})

      # Create a test message
      test_message = Message.new_user!("Hello, world!")
      test_messages = [Message.new_system!(), test_message]

      # Return the models and messages for use in tests
      %{
        openai: openai,
        vertex_ai: vertex_ai,
        mistral_ai: mistral_ai,
        perplexity: perplexity,
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

      # Mock the API request to avoid actual API calls
      mock_response = %{
        body: %{
          "choices" => [
            %{
              "message" => %{
                "content" => "Test response",
                "role" => "assistant"
              },
              "finish_reason" => "stop"
            }
          ]
        }
      }

      Req
      |> stub(:request, fn _req ->
        {:ok, mock_response}
      end)

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

      assert_received {:telemetry_event, [:langchain, :llm, :response, :non_streaming], _, metadata}
      assert metadata.model == openai.model
      assert is_integer(metadata.response_size)

      # Clean up telemetry handlers
      :telemetry.detach("test-openai-telemetry-events")
    end

    test "emits telemetry events for ChatVertexAI", %{vertex_ai: vertex_ai, test_messages: messages} do
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

      # Mock the API request to avoid actual API calls
      mock_response = %{
        body: %{
          "candidates" => [
            %{
              "content" => %{
                "parts" => [
                  %{"text" => "Test response"}
                ]
              }
            }
          ]
        }
      }

      Req
      |> stub(:request, fn _req ->
        {:ok, mock_response}
      end)

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

      assert_received {:telemetry_event, [:langchain, :llm, :response, :non_streaming], _, metadata}
      assert metadata.model == vertex_ai.model
      assert is_integer(metadata.response_size)

      # Clean up telemetry handlers
      :telemetry.detach("test-vertex-telemetry-events")
    end

    test "emits telemetry events for ChatMistralAI", %{mistral_ai: mistral_ai, test_messages: messages} do
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

      # Mock the API request to avoid actual API calls
      mock_response = %{
        body: %{
          "choices" => [
            %{
              "message" => %{
                "content" => "Test response",
                "role" => "assistant"
              },
              "finish_reason" => "stop"
            }
          ]
        }
      }

      Req
      |> stub(:request, fn _req ->
        {:ok, mock_response}
      end)

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

      assert_received {:telemetry_event, [:langchain, :llm, :response, :non_streaming], _, metadata}
      assert metadata.model == mistral_ai.model
      assert is_integer(metadata.response_size)

      # Clean up telemetry handlers
      :telemetry.detach("test-mistral-telemetry-events")
    end

    test "emits telemetry events for ChatPerplexity", %{perplexity: perplexity, test_messages: messages} do
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

      # Mock the API request to avoid actual API calls
      mock_response = %{
        body: %{
          "choices" => [
            %{
              "message" => %{
                "content" => "Test response",
                "role" => "assistant"
              },
              "finish_reason" => "stop"
            }
          ]
        }
      }

      Req
      |> stub(:request, fn _req ->
        {:ok, mock_response}
      end)

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

      assert_received {:telemetry_event, [:langchain, :llm, :response, :non_streaming], _, metadata}
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

      # Mock the API request to avoid actual API calls
      mock_response = %{
        body: %{
          "choices" => [
            %{
              "message" => %{
                "content" => "Test response",
                "role" => "assistant"
              },
              "finish_reason" => "stop"
            }
          ]
        }
      }

      Req
      |> stub(:request, fn _req ->
        {:ok, mock_response}
      end)

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

      assert_received {:telemetry_measurements, [:langchain, :llm, :response, :non_streaming], measurements}
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
