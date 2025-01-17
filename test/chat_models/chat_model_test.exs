defmodule LangChain.ChatModels.ChatModelTest do
  use ExUnit.Case
  doctest LangChain.ChatModels.ChatModel
  alias LangChain.ChatModels.ChatModel
  alias LangChain.ChatModels.ChatOpenAI

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
