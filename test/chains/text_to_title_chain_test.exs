defmodule LangChain.Chains.TextToTitleChainTest do
  use LangChain.BaseCase

  doctest LangChain.Chains.TextToTitleChain

  alias LangChain.Chains.TextToTitleChain
  alias LangChain.Chains.LLMChain
  alias LangChain.Message
  alias LangChain.ChatModels.ChatOpenAI
  alias LangChain.LangChainError

  setup do
    llm = ChatOpenAI.new!(%{model: "gpt-3.5-turbo", stream: false, seed: 0})
    input_text = "Let's start a new blog post about the magical properties of pineapple cookies."

    data = %{
      llm: llm,
      input_text: input_text,
      fallback_title: "Default new title"
    }

    title_chain = TextToTitleChain.new!(data)
    Map.put(data, :title_chain, title_chain)
  end

  describe "new/1" do
    test "defines a text to title chain", data do
      assert {:ok, router} = TextToTitleChain.new(data)

      assert %TextToTitleChain{} = router
      assert router.input_text == data[:input_text]
      assert router == data[:title_chain]
    end

    test "requires llm, input_text" do
      assert {:error, changeset} = TextToTitleChain.new(%{})
      refute changeset.valid?
      assert {"can't be blank", _} = changeset.errors[:llm]
      assert {"can't be blank", _} = changeset.errors[:input_text]
    end
  end

  describe "new!/1" do
    test "returns the configured text to title chain", data do
      assert %TextToTitleChain{} = router = TextToTitleChain.new!(data)

      assert %TextToTitleChain{} = router
      assert router == data[:title_chain]
    end

    test "raises exception when invalid", data do
      use_data = Map.delete(data, :llm)

      assert_raise LangChainError, "llm: can't be blank", fn ->
        TextToTitleChain.new!(use_data)
      end
    end
  end

  describe "run/2" do
    test "runs and returns updated chain and last message", %{title_chain: title_chain} do
      fake_message = Message.new_assistant!("Summarized Title")
      fake_response = {:ok, [fake_message], nil}
      set_api_override(fake_response)

      assert {:ok, updated_chain, last_msg} = TextToTitleChain.run(title_chain)
      assert %LLMChain{} = updated_chain
      assert last_msg == fake_message
    end
  end

  describe "evaluate/2" do
    test "returns the summarized title", %{title_chain: title_chain} do
      set_api_override({:ok, [Message.new_assistant!("Special Title")], nil})
      assert "Special Title" == TextToTitleChain.evaluate(title_chain)
    end

    test "returns fallback title something goes wrong", %{
      title_chain: title_chain,
      fallback_title: fallback_title
    } do
      set_api_override({:error, "FAKE API call failure"})
      assert fallback_title == TextToTitleChain.evaluate(title_chain)
    end
  end
end
