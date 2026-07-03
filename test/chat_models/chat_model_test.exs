defmodule LangChain.ChatModels.ChatModelTest do
  use ExUnit.Case
  doctest LangChain.ChatModels.ChatModel
  alias LangChain.ChatModels.ChatModel
  alias LangChain.ChatModels.ChatOpenAI
  alias LangChain.ChatModels.ChatAnthropic
  alias LangChain.Message
  alias LangChain.MessageDelta
  alias LangChain.TokenUsage

  describe "token_usage_from_result/1" do
    setup do
      %{usage: TokenUsage.new!(%{input: 11, output: 47})}
    end

    test "extracts usage from a single message", %{usage: usage} do
      msg = %Message{role: :assistant, content: "hi", metadata: %{usage: usage}}
      assert %{token_usage: ^usage} = ChatModel.token_usage_from_result({:ok, msg})
    end

    test "extracts usage from a single message delta", %{usage: usage} do
      delta = %MessageDelta{role: :assistant, content: "hi", metadata: %{usage: usage}}
      assert %{token_usage: ^usage} = ChatModel.token_usage_from_result({:ok, delta})
    end

    test "extracts usage from a streaming list of message deltas", %{usage: usage} do
      # Streaming responses return a list of deltas; usage rides on the final one.
      deltas = [
        %MessageDelta{role: :assistant, content: "hi", metadata: nil},
        %MessageDelta{role: :assistant, content: " there", metadata: nil},
        %MessageDelta{role: :assistant, content: nil, metadata: %{usage: usage}}
      ]

      assert %{token_usage: ^usage} = ChatModel.token_usage_from_result({:ok, deltas})
    end

    test "extracts usage from a list of messages", %{usage: usage} do
      messages = [%Message{role: :assistant, content: "hi", metadata: %{usage: usage}}]
      assert %{token_usage: ^usage} = ChatModel.token_usage_from_result({:ok, messages})
    end

    test "returns nil usage when none is present" do
      deltas = [%MessageDelta{role: :assistant, content: "hi", metadata: nil}]
      assert %{token_usage: nil} = ChatModel.token_usage_from_result({:ok, deltas})
      assert %{token_usage: nil} = ChatModel.token_usage_from_result({:error, :boom})
    end
  end

  describe "llm_telemetry_span/3" do
    # This is the single wiring point every chat model must route its LLM span
    # through. It guarantees `:enrich_stop` is attached so token usage reaches the
    # `[:langchain, :llm, :call, :stop]` event. If a provider bypasses it (as
    # ChatAwsMantle/ChatReqLLM historically did), token usage silently vanishes.
    setup do
      test_pid = self()
      handler_id = "test-llm-telemetry-span-#{System.unique_integer([:positive])}"

      :telemetry.attach_many(
        handler_id,
        [[:langchain, :llm, :call, :start], [:langchain, :llm, :call, :stop]],
        fn [_, _, _, stage], _measurements, metadata, _config ->
          send(test_pid, {stage, metadata})
        end,
        nil
      )

      on_exit(fn -> :telemetry.detach(handler_id) end)
      :ok
    end

    test "surfaces token usage from the returned message onto the :stop event" do
      usage = TokenUsage.new!(%{input: 9, output: 13})
      metadata = %{model: "test-model", provider: "test"}

      result =
        ChatModel.llm_telemetry_span(nil, metadata, fn ->
          {:ok, %Message{role: :assistant, content: "hi", metadata: %{usage: usage}}}
        end)

      assert {:ok, %Message{}} = result
      assert_received {:stop, stop_metadata}
      assert %TokenUsage{input: 9, output: 13} = stop_metadata.token_usage
    end

    test "surfaces token usage from a streaming delta list onto the :stop event" do
      usage = TokenUsage.new!(%{input: 4, output: 6})
      metadata = %{model: "test-model", provider: "test"}

      ChatModel.llm_telemetry_span(nil, metadata, fn ->
        {:ok,
         [
           %MessageDelta{role: :assistant, content: "hi", metadata: nil},
           %MessageDelta{role: :assistant, content: nil, metadata: %{usage: usage}}
         ]}
      end)

      assert_received {:stop, stop_metadata}
      assert %TokenUsage{input: 4, output: 6} = stop_metadata.token_usage
    end

    test "sets token_usage to nil (never omits the key) when the result has no usage" do
      metadata = %{model: "test-model", provider: "test"}

      ChatModel.llm_telemetry_span(nil, metadata, fn -> {:error, :boom} end)

      assert_received {:stop, stop_metadata}
      # The key must be present even without usage — its presence is what proves
      # enrich_stop ran. A model that forgot enrich_stop would omit it entirely.
      assert Map.has_key?(stop_metadata, :token_usage)
      assert stop_metadata.token_usage == nil
    end

    test "injects the model's :request_options into the :start metadata" do
      model = ChatOpenAI.new!(%{model: "gpt-4o", temperature: 0.7, seed: 42})
      metadata = %{model: "gpt-4o", provider: "openai"}

      ChatModel.llm_telemetry_span(model, metadata, fn -> {:error, :boom} end)

      assert_received {:start, start_metadata}
      assert %{temperature: 0.7, seed: 42} = start_metadata.request_options
    end

    test "a nil model yields empty :request_options rather than omitting the key" do
      ChatModel.llm_telemetry_span(nil, %{model: "m", provider: "p"}, fn -> :ok end)

      assert_received {:start, start_metadata}
      assert start_metadata.request_options == %{}
    end

    test "a caller-provided :request_options is not overwritten" do
      model = ChatOpenAI.new!(%{model: "gpt-4o", temperature: 0.7})
      metadata = %{model: "gpt-4o", provider: "openai", request_options: %{temperature: 0.1}}

      ChatModel.llm_telemetry_span(model, metadata, fn -> :ok end)

      assert_received {:start, start_metadata}
      assert start_metadata.request_options == %{temperature: 0.1}
    end
  end

  describe "request_options/1" do
    test "extracts the standard request parameters a model sets, dropping nils" do
      model =
        ChatOpenAI.new!(%{
          model: "gpt-4o",
          temperature: 0.7,
          frequency_penalty: 0.2,
          seed: 7,
          stream: true
        })

      opts = ChatModel.request_options(model)

      assert %{temperature: 0.7, frequency_penalty: 0.2, seed: 7, stream: true} = opts
      # `n` defaults to 1 on ChatOpenAI, so choice_count is present.
      assert opts.choice_count == 1
      # Fields the struct doesn't define are absent, not nil.
      refute Map.has_key?(opts, :top_k)
      refute Map.has_key?(opts, :presence_penalty)
    end

    test "maps :reasoning_effort to :reasoning_level" do
      model = ChatOpenAI.new!(%{model: "gpt-4o", reasoning_effort: "medium"})
      assert %{reasoning_level: "medium"} = ChatModel.request_options(model)
    end

    test "extracts max_tokens, top_p, and top_k from a model that sets them" do
      # ChatOpenAI has none of these three; ChatAnthropic exposes all of them, so
      # it exercises the field readers the ChatOpenAI-based test above cannot.
      model =
        ChatAnthropic.new!(%{
          model: "claude-3-5-sonnet-latest",
          max_tokens: 1024,
          top_p: 0.9,
          top_k: 40
        })

      assert %{max_tokens: 1024, top_p: 0.9, top_k: 40} = ChatModel.request_options(model)
    end

    test "retains falsy-but-set values (stream: false, seed: 0, temperature: 0.0)" do
      # Only `nil` should be dropped. A regression to a truthiness check would
      # silently omit a disabled stream or a zero seed/temperature.
      model = ChatOpenAI.new!(%{model: "gpt-4o", temperature: 0.0, seed: 0, stream: false})

      opts = ChatModel.request_options(model)

      assert Map.fetch(opts, :temperature) == {:ok, 0.0}
      assert Map.fetch(opts, :seed) == {:ok, 0}
      assert Map.fetch(opts, :stream) == {:ok, false}
    end

    test "returns an empty map for nil" do
      assert ChatModel.request_options(nil) == %{}
    end

    test "returns an empty map for a non-struct map argument" do
      # A plain map has no `__struct__`, so it hits the catch-all clause rather
      # than the `%_module{}` reader.
      assert ChatModel.request_options(%{temperature: 0.7, stream: true}) == %{}
    end
  end

  describe "serialize_config/1" do
    test "creates a map from a chat model" do
      model = ChatOpenAI.new!(%{model: "gpt-4o"})
      result = ChatModel.serialize_config(model)
      assert Map.get(result, "module") == "Elixir.LangChain.ChatModels.ChatOpenAI"
      assert Map.get(result, "model") == "gpt-4o"
      assert Map.get(result, "version") == 1
    end
  end

  describe "restore_from_map/1" do
    test "return error when nil data given" do
      assert {:error, reason} = ChatModel.restore_from_map(nil)
      assert reason == "No data to restore"
    end

    test "return error when module not found" do
      assert {:error, reason} =
               ChatModel.restore_from_map(%{
                 "module" => "Elixir.InvalidModule",
                 "version" => 1,
                 "model" => "howdy"
               })

      assert reason == "ChatModel module \"Elixir.InvalidModule\" not found"
    end

    test "restores using the module" do
      model = ChatOpenAI.new!(%{model: "gpt-4o"})
      serialized = ChatModel.serialize_config(model)
      {:ok, restored} = ChatModel.restore_from_map(serialized)
      assert restored == model
    end
  end
end
