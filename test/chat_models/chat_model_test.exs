defmodule LangChain.ChatModels.ChatModelTest do
  use ExUnit.Case
  doctest LangChain.ChatModels.ChatModel
  alias LangChain.ChatModels.ChatModel
  alias LangChain.ChatModels.ChatOpenAI
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

  describe "llm_telemetry_span/2" do
    # This is the single wiring point every chat model must route its LLM span
    # through. It guarantees `:enrich_stop` is attached so token usage reaches the
    # `[:langchain, :llm, :call, :stop]` event. If a provider bypasses it (as
    # ChatAwsMantle/ChatReqLLM historically did), token usage silently vanishes.
    setup do
      test_pid = self()
      handler_id = "test-llm-telemetry-span-#{System.unique_integer([:positive])}"

      :telemetry.attach(
        handler_id,
        [:langchain, :llm, :call, :stop],
        fn _event, _measurements, metadata, _config ->
          send(test_pid, {:stop, metadata})
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
        ChatModel.llm_telemetry_span(metadata, fn ->
          {:ok, %Message{role: :assistant, content: "hi", metadata: %{usage: usage}}}
        end)

      assert {:ok, %Message{}} = result
      assert_received {:stop, stop_metadata}
      assert %TokenUsage{input: 9, output: 13} = stop_metadata.token_usage
    end

    test "surfaces token usage from a streaming delta list onto the :stop event" do
      usage = TokenUsage.new!(%{input: 4, output: 6})
      metadata = %{model: "test-model", provider: "test"}

      ChatModel.llm_telemetry_span(metadata, fn ->
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

      ChatModel.llm_telemetry_span(metadata, fn -> {:error, :boom} end)

      assert_received {:stop, stop_metadata}
      # The key must be present even without usage — its presence is what proves
      # enrich_stop ran. A model that forgot enrich_stop would omit it entirely.
      assert Map.has_key?(stop_metadata, :token_usage)
      assert stop_metadata.token_usage == nil
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
