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
