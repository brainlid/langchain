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
end
