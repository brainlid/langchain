defmodule LangChain.ChatModels.TelemetryTest do
  use ExUnit.Case
  use Mimic

  alias LangChain.ChatModels.ChatModel
  alias LangChain.ChatModels.ChatOpenAI
  alias LangChain.ChatModels.ChatMistralAI
  alias LangChain.ChatModels.ChatAnthropic
  alias LangChain.ChatModels.ChatGoogleAI
  alias LangChain.ChatModels.ChatPerplexity
  alias LangChain.ChatModels.ChatVertexAI
  alias LangChain.ChatModels.ChatGrok
  alias LangChain.Chains.LLMChain
  alias LangChain.Function
  alias LangChain.Message
  alias LangChain.Message.ToolCall
  alias LangChain.TokenUsage

  # Setup for test
  setup :verify_on_exit!

  # Helper to create a standard LLM call stub with provider in metadata
  defp make_llm_stub(module, provider) do
    module
    |> stub(:call, fn model, messages, tools ->
      model_name =
        case Map.get(model, :model) do
          nil -> inspect(Map.get(model, :serving, "unknown"))
          name -> name
        end

      metadata = %{
        model: model_name,
        provider: provider,
        message_count: length(messages),
        tool_count: length(tools)
      }

      usage = TokenUsage.new!(%{input: 10, output: 20})

      LangChain.Telemetry.span(
        [:langchain, :llm, :call],
        metadata,
        fn ->
          LangChain.Telemetry.llm_prompt(
            %{system_time: System.system_time()},
            %{model: model_name, messages: messages}
          )

          response =
            Message.new_assistant!(%{content: "Test response", metadata: %{usage: usage}})

          LangChain.Telemetry.llm_response(
            %{system_time: System.system_time()},
            %{model: model_name, response: response}
          )

          LangChain.Telemetry.emit_event(
            [:langchain, :llm, :response, :non_streaming],
            %{system_time: System.system_time()},
            %{model: model_name, response_size: byte_size(inspect(response))}
          )

          {:ok, response}
        end,
        enrich_stop: &ChatModel.token_usage_from_result/1
      )
    end)
  end

  describe "telemetry instrumentation" do
    setup do
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

      grok = ChatGrok.new!(%{model: "grok-3-mini", api_key: "test-xai-key"})

      test_messages = [
        Message.new_system!("You are a helpful assistant."),
        Message.new_user!("Hello, how are you?")
      ]

      # Create stubs for all models
      make_llm_stub(ChatOpenAI, "openai")
      make_llm_stub(ChatMistralAI, "mistralai")
      make_llm_stub(ChatVertexAI, "vertex_ai")
      make_llm_stub(ChatPerplexity, "perplexity")
      make_llm_stub(ChatGoogleAI, "google")
      make_llm_stub(ChatAnthropic, "anthropic")
      make_llm_stub(ChatGrok, "xai")

      # Mock Req for any remaining API calls
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
        grok: grok,
        test_messages: test_messages
      }
    end

    test "emits telemetry events for ChatOpenAI with provider", %{
      openai: openai,
      test_messages: messages
    } do
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

      {:ok, _response} = ChatOpenAI.call(openai, messages, [])

      assert_received {:telemetry_event, [:langchain, :llm, :call, :start], _, metadata}
      assert metadata.model == openai.model
      assert metadata.provider == "openai"
      assert metadata.message_count == length(messages)
      assert metadata.tool_count == 0

      assert_received {:telemetry_event, [:langchain, :llm, :prompt], _, metadata}
      assert metadata.model == openai.model

      assert_received {:telemetry_event, [:langchain, :llm, :response], _, metadata}
      assert metadata.model == openai.model

      assert_received {:telemetry_event, [:langchain, :llm, :response, :non_streaming], _,
                       metadata}

      assert metadata.model == openai.model

      :telemetry.detach("test-openai-telemetry-events")
    end

    test "emits telemetry events for ChatVertexAI with provider", %{
      vertex_ai: vertex_ai,
      test_messages: messages
    } do
      test_pid = self()

      :telemetry.attach_many(
        "test-vertex-telemetry-events",
        [[:langchain, :llm, :call, :start]],
        fn name, measurements, metadata, _config ->
          send(test_pid, {:telemetry_event, name, measurements, metadata})
        end,
        nil
      )

      {:ok, _response} = ChatVertexAI.call(vertex_ai, messages, [])

      assert_received {:telemetry_event, [:langchain, :llm, :call, :start], _, metadata}
      assert metadata.model == vertex_ai.model
      assert metadata.provider == "vertex_ai"

      :telemetry.detach("test-vertex-telemetry-events")
    end

    test "emits telemetry events for ChatMistralAI with provider", %{
      mistral_ai: mistral_ai,
      test_messages: messages
    } do
      test_pid = self()

      :telemetry.attach_many(
        "test-mistral-telemetry-events",
        [[:langchain, :llm, :call, :start]],
        fn name, measurements, metadata, _config ->
          send(test_pid, {:telemetry_event, name, measurements, metadata})
        end,
        nil
      )

      {:ok, _response} = ChatMistralAI.call(mistral_ai, messages, [])

      assert_received {:telemetry_event, [:langchain, :llm, :call, :start], _, metadata}
      assert metadata.model == mistral_ai.model
      assert metadata.provider == "mistralai"

      :telemetry.detach("test-mistral-telemetry-events")
    end

    test "emits telemetry events for ChatPerplexity with provider", %{
      perplexity: perplexity,
      test_messages: messages
    } do
      test_pid = self()

      :telemetry.attach_many(
        "test-perplexity-telemetry-events",
        [[:langchain, :llm, :call, :start]],
        fn name, measurements, metadata, _config ->
          send(test_pid, {:telemetry_event, name, measurements, metadata})
        end,
        nil
      )

      {:ok, _response} = ChatPerplexity.call(perplexity, messages, [])

      assert_received {:telemetry_event, [:langchain, :llm, :call, :start], _, metadata}
      assert metadata.model == perplexity.model
      assert metadata.provider == "perplexity"

      :telemetry.detach("test-perplexity-telemetry-events")
    end

    test "emits telemetry events for ChatAnthropic with provider", %{
      anthropic: anthropic,
      test_messages: messages
    } do
      test_pid = self()

      :telemetry.attach_many(
        "test-anthropic-telemetry-events",
        [[:langchain, :llm, :call, :start]],
        fn name, measurements, metadata, _config ->
          send(test_pid, {:telemetry_event, name, measurements, metadata})
        end,
        nil
      )

      {:ok, _response} = ChatAnthropic.call(anthropic, messages, [])

      assert_received {:telemetry_event, [:langchain, :llm, :call, :start], _, metadata}
      assert metadata.model == anthropic.model
      assert metadata.provider == "anthropic"

      :telemetry.detach("test-anthropic-telemetry-events")
    end

    test "emits telemetry events for ChatGrok with provider", %{
      grok: grok,
      test_messages: messages
    } do
      test_pid = self()

      :telemetry.attach_many(
        "test-grok-telemetry-events",
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

      {:ok, _response} = ChatGrok.call(grok, messages, [])

      assert_received {:telemetry_event, [:langchain, :llm, :call, :start], _, metadata}
      assert metadata.model == grok.model
      assert metadata.provider == "xai"
      assert metadata.message_count == length(messages)
      assert metadata.tool_count == 0

      assert_received {:telemetry_event, [:langchain, :llm, :prompt], _, _metadata}
      assert_received {:telemetry_event, [:langchain, :llm, :response], _, _metadata}

      assert_received {:telemetry_event, [:langchain, :llm, :response, :non_streaming], _,
                       _metadata}

      :telemetry.detach("test-grok-telemetry-events")
    end

    test "call_id is present in start and stop events and is the same UUID", %{
      openai: openai,
      test_messages: messages
    } do
      test_pid = self()

      :telemetry.attach_many(
        "test-call-id-events",
        [
          [:langchain, :llm, :call, :start],
          [:langchain, :llm, :call, :stop]
        ],
        fn name, _measurements, metadata, _config ->
          send(test_pid, {:telemetry_event, name, metadata})
        end,
        nil
      )

      {:ok, _response} = ChatOpenAI.call(openai, messages, [])

      assert_received {:telemetry_event, [:langchain, :llm, :call, :start], start_metadata}
      assert_received {:telemetry_event, [:langchain, :llm, :call, :stop], stop_metadata}

      # call_id is a valid UUID present in both events
      assert is_binary(start_metadata.call_id)
      assert byte_size(start_metadata.call_id) == 36
      assert start_metadata.call_id == stop_metadata.call_id

      :telemetry.detach("test-call-id-events")
    end

    test "LLM call stop event includes token_usage from enrich_stop", %{
      openai: openai,
      test_messages: messages
    } do
      test_pid = self()

      :telemetry.attach_many(
        "test-token-usage-stop",
        [[:langchain, :llm, :call, :stop]],
        fn name, _measurements, metadata, _config ->
          send(test_pid, {:telemetry_event, name, metadata})
        end,
        nil
      )

      {:ok, _response} = ChatOpenAI.call(openai, messages, [])

      assert_received {:telemetry_event, [:langchain, :llm, :call, :stop], stop_metadata}
      assert %TokenUsage{input: 10, output: 20} = stop_metadata.token_usage

      :telemetry.detach("test-token-usage-stop")
    end

    test "telemetry includes correct measurements", %{openai: openai, test_messages: messages} do
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

      {:ok, _response} = ChatOpenAI.call(openai, messages, [])

      assert_received {:telemetry_measurements, [:langchain, :llm, :call, :start], measurements}
      assert Map.has_key?(measurements, :system_time)

      assert_received {:telemetry_measurements, [:langchain, :llm, :prompt], measurements}
      assert Map.has_key?(measurements, :system_time)

      assert_received {:telemetry_measurements, [:langchain, :llm, :response], measurements}
      assert Map.has_key?(measurements, :system_time)

      assert_received {:telemetry_measurements, [:langchain, :llm, :response, :non_streaming],
                       measurements}

      assert Map.has_key?(measurements, :system_time)

      assert_received {:telemetry_measurements, [:langchain, :llm, :call, :stop], measurements}
      assert Map.has_key?(measurements, :system_time)
      assert Map.has_key?(measurements, :duration)
      assert is_integer(measurements.duration)

      :telemetry.detach("test-measurement-telemetry-events")
    end
  end

  describe "chain telemetry with custom_context" do
    setup :verify_on_exit!

    test "chain execution telemetry includes custom_context" do
      test_pid = self()

      ChatOpenAI
      |> stub(:call, fn _model, _messages, _tools ->
        {:ok, Message.new_assistant!("Test response")}
      end)

      Req
      |> stub(:request, fn _req ->
        {:ok, %Req.Response{status: 200, body: %{}}}
      end)

      :telemetry.attach_many(
        "test-chain-custom-context",
        [
          [:langchain, :chain, :execute, :start],
          [:langchain, :chain, :execute, :stop]
        ],
        fn name, _measurements, metadata, _config ->
          send(test_pid, {:telemetry_event, name, metadata})
        end,
        nil
      )

      custom_ctx = %{user_id: "user-123", session_id: "sess-456"}

      {:ok, _chain} =
        LLMChain.new!(%{
          llm: ChatOpenAI.new!(%{model: "gpt-4o-mini", api_key: "test-key"}),
          custom_context: custom_ctx
        })
        |> LLMChain.add_message(Message.new_system!("You are helpful."))
        |> LLMChain.add_message(Message.new_user!("Hello"))
        |> LLMChain.run()

      assert_received {:telemetry_event, [:langchain, :chain, :execute, :start], start_metadata}
      assert start_metadata.custom_context == custom_ctx
      assert start_metadata.chain_type == "llm_chain"

      # call_id is shared between start and stop
      assert_received {:telemetry_event, [:langchain, :chain, :execute, :stop], stop_metadata}
      assert is_binary(start_metadata.call_id)
      assert start_metadata.call_id == stop_metadata.call_id

      :telemetry.detach("test-chain-custom-context")
    end

    test "chain stop event includes last_message and token_usage" do
      test_pid = self()
      usage = TokenUsage.new!(%{input: 15, output: 25})

      ChatOpenAI
      |> stub(:call, fn _model, _messages, _tools ->
        {:ok, Message.new_assistant!(%{content: "Hello!", metadata: %{usage: usage}})}
      end)

      :telemetry.attach(
        "test-chain-stop-enrichment",
        [:langchain, :chain, :execute, :stop],
        fn _name, _measurements, metadata, _config ->
          send(test_pid, {:chain_stop, metadata})
        end,
        nil
      )

      {:ok, _chain} =
        LLMChain.new!(%{
          llm: ChatOpenAI.new!(%{model: "gpt-4o-mini", api_key: "test-key"})
        })
        |> LLMChain.add_message(Message.new_user!("Hi"))
        |> LLMChain.run()

      assert_received {:chain_stop, stop_metadata}
      assert %Message{role: :assistant} = stop_metadata.last_message
      assert %TokenUsage{input: 15, output: 25} = stop_metadata.token_usage

      :telemetry.detach("test-chain-stop-enrichment")
    end
  end

  describe "tool call telemetry with custom_context" do
    setup :verify_on_exit!

    test "tool call telemetry includes custom_context" do
      test_pid = self()

      :telemetry.attach_many(
        "test-tool-custom-context",
        [
          [:langchain, :tool, :call, :start],
          [:langchain, :tool, :call, :stop]
        ],
        fn name, _measurements, metadata, _config ->
          send(test_pid, {:telemetry_event, name, metadata})
        end,
        nil
      )

      custom_ctx = %{user_id: "user-123", trace_id: "trace-789"}

      {:ok, fun} =
        Function.new(%{name: "hello", function: fn _args, _ctx -> "world" end})

      call = ToolCall.new!(%{call_id: "call-1", name: "hello", arguments: %{}})

      LLMChain.execute_tool_call(call, fun, context: custom_ctx)

      assert_received {:telemetry_event, [:langchain, :tool, :call, :start], start_metadata}
      assert start_metadata.tool_name == "hello"
      assert start_metadata.custom_context == custom_ctx

      assert_received {:telemetry_event, [:langchain, :tool, :call, :stop], stop_metadata}
      assert start_metadata.call_id == stop_metadata.call_id

      :telemetry.detach("test-tool-custom-context")
    end
  end

  describe "ChatModel.provider/1 fallback" do
    defmodule ChatFakeProvider do
      @moduledoc false
      defstruct [:model]
    end

    test "derives provider from module name when provider/0 is not implemented" do
      assert ChatModel.provider(%ChatFakeProvider{}) == "fake_provider"
    end

    test "dispatches to provider/0 when implemented" do
      openai = ChatOpenAI.new!(%{model: "gpt-4o-mini", api_key: "test-key"})
      assert ChatModel.provider(openai) == "openai"
    end
  end

  describe "ChatModel.token_usage_from_result/1" do
    test "extracts token_usage from a single message result" do
      usage = TokenUsage.new!(%{input: 5, output: 10})
      msg = Message.new_assistant!(%{content: "hi", metadata: %{usage: usage}})

      assert %{token_usage: ^usage} = ChatModel.token_usage_from_result({:ok, msg})
    end

    test "extracts token_usage from a list of messages" do
      usage = TokenUsage.new!(%{input: 5, output: 10})
      msg1 = Message.new_assistant!("no usage here")
      msg2 = Message.new_assistant!(%{content: "hi", metadata: %{usage: usage}})

      assert %{token_usage: ^usage} = ChatModel.token_usage_from_result({:ok, [msg1, msg2]})
    end

    test "returns nil token_usage for error results" do
      assert %{token_usage: nil} = ChatModel.token_usage_from_result({:error, "something"})
    end

    test "returns nil token_usage when message has no usage metadata" do
      msg = Message.new_assistant!("plain response")
      assert %{token_usage: nil} = ChatModel.token_usage_from_result({:ok, msg})
    end
  end
end
