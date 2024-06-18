defmodule LangChain.ChatModels.ChatModelTest do
  use ExUnit.Case
  doctest LangChain.ChatModels.ChatModel
  alias LangChain.ChatModels.ChatModel
  alias LangChain.ChatModels.ChatOpenAI

  describe "add_callback/2" do
    test "appends the callback to the model" do
      model = %ChatOpenAI{}
      assert model.callbacks == []
      handler = %{on_llm_new_message: fn _model, _msg -> :ok end}
      %ChatOpenAI{} = updated = ChatModel.add_callback(model, handler)
      assert updated.callbacks == [handler]
    end

    test "does nothing on a model that doesn't support callbacks" do
      handler = %{on_llm_new_message: fn _model, _msg -> :ok end}
      non_model = %{something: "else"}
      updated = ChatModel.add_callback(non_model, handler)
      assert updated == non_model
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
