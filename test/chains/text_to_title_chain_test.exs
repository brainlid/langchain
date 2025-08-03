defmodule LangChain.Chains.TextToTitleChainTest do
  use LangChain.BaseCase
  use Mimic

  doctest LangChain.Chains.TextToTitleChain

  alias LangChain.Chains.TextToTitleChain
  alias LangChain.Chains.LLMChain
  alias LangChain.Message
  alias LangChain.Message.ContentPart
  alias LangChain.ChatModels.ChatOpenAI
  alias LangChain.LangChainError
  alias LangChain.Utils

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

      # Made NOT LIVE here
      expect(ChatOpenAI, :call, fn _model, _messages, _tools, _opts ->
        {:ok, [fake_message]}
      end)

      assert {:ok, updated_chain} = TextToTitleChain.run(title_chain)
      assert %LLMChain{} = updated_chain
      assert updated_chain.last_message == fake_message
    end

    test "uses override_system_prompt", %{llm: llm} do
      fake_message = Message.new_assistant!("Summarized Title")

      # Made NOT LIVE here
      expect(ChatOpenAI, :call, fn _model, _messages, _tools, _opts ->
        {:ok, [fake_message]}
      end)

      {:ok, updated_chain} =
        %{
          llm: llm,
          input_text: "Initial user text.",
          override_system_prompt: "Custom system prompt"
        }
        |> TextToTitleChain.new!()
        |> TextToTitleChain.run()

      assert %LLMChain{} = updated_chain
      {system, _rest} = Utils.split_system_message(updated_chain.messages)
      assert system.content == [ContentPart.text!("Custom system prompt")]
    end
  end

  describe "evaluate/2" do
    test "returns the summarized title", %{title_chain: title_chain} do
      # Made NOT LIVE here
      expect(ChatOpenAI, :call, fn _model, _messages, _tools, _opts ->
        {:ok, [Message.new_assistant!("Special Title")]}
      end)

      assert "Special Title" == TextToTitleChain.evaluate(title_chain)
    end

    test "returns fallback title something goes wrong", %{
      title_chain: title_chain,
      fallback_title: fallback_title
    } do
      # Made NOT LIVE here
      expect(ChatOpenAI, :call, fn _model, _messages, _tools, _opts ->
        {:error, "FAKE API call failure"}
      end)

      assert fallback_title == TextToTitleChain.evaluate(title_chain)
    end

    @tag live_call: true, live_open_ai: true
    test "supports using examples", %{llm: llm, input_text: input_text} do
      data = %{
        llm: llm,
        input_text: input_text,
        fallback_title: "Default new title",
        examples: [
          "Blog Post: Making Delicious and Healthy Smoothies",
          "System Email: Notifying Users of Planned Downtime"
        ],
        verbose: true
      }

      result_title =
        data
        |> TextToTitleChain.new!()
        |> TextToTitleChain.evaluate()

      assert String.starts_with?(result_title, "Blog Post:")
      assert String.contains?(result_title, "Pineapple")
    end
  end
end
