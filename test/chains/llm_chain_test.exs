defmodule Langchain.Chains.LLMChainTest do
  use Langchain.BaseCase

  doctest Langchain.Chains.LLMChain
  alias Langchain.ChatModels.ChatOpenAI
  alias Langchain.Chains.LLMChain
  alias Langchain.PromptTemplate

  describe "new/1" do
    test "works with minimal setup" do
      {:ok, chat} = ChatOpenAI.new(%{temperature: 0})

      assert {:ok, %LLMChain{} = chain} =
               LLMChain.new(%{
                 prompt: "What year is it?",
                 llm: chat
               })

      assert chain.llm == chat
      assert chain.prompt == "What year is it?"
    end
  end

  describe "JS inspired test" do
    # test "usage with LLMs" do
      # # https://js.langchain.com/docs/modules/chains/llm_chain

      # # We can construct an LLMChain from a PromptTemplate and an LLM.
      # {:ok, model} = ChatOpenAI.new(%{temperature: 0})

      # {:ok, prompt} =
      #   PromptTemplate.from_template(
      #     "What is a good name for a company that makes <%= @product %>?"
      #   )

      # {:ok, chain_A} = LLMChain.new(%{llm: model, prompt: prompt})

      # # The result is an LLMChain with a `text` property that's been set.
      # {:ok, res_A} = chain_A.call(%{product: "colorful socks"})
      # # console.log({ resA });
      # # // { resA: { text: '\n\nSocktastic!' } }

      # # Since the LLMChain is a single-input, single-output chain, we can also `run` it.
      # # This takes in a string and returns the `text` property.
      # {:ok, res_A2} = chain_A.run("colorful socks")
      # # console.log({ resA2 });
      # # // { resA2: '\n\nSocktastic!' }

      # assert false
    # end

    # @tag :live_call
    test "usage with chat models" do
      # https://js.langchain.com/docs/modules/chains/llm_chain#usage-with-chat-models
      {:ok, chat} = ChatOpenAI.new(%{temperature: 0})

      chat_prompt = [
        PromptTemplate.new!(%{
          role: :system,
          text:
            "You are a helpful assistant that translates <%= @input_language %> to <%= @output_language %>."
        }),
        PromptTemplate.new!(%{role: :user, text: "<%= @text %>"})
      ]

      set_api_override({:ok, Message.new_assistant!("Amo programar")})

      {:ok, chain} =
        LLMChain.new(%{
          prompt: chat_prompt,
          llm: chat,
          verbose: true
        })

      {:ok, result} =
        LLMChain.call_chat(chain, %{
          input_language: "English",
          output_language: "Spanish",
          text: "I love programming."
        })

      assert %{text: "Amo programar"} = result
    end
  end

  # TODO: Sequential chains
  # https://js.langchain.com/docs/modules/chains/sequential_chain

  # TODO: Index related chains
  # https://js.langchain.com/docs/modules/chains/index_related_chains/

  # TODO: OpenAI Function Chains
  # https://js.langchain.com/docs/modules/chains/openai_functions/

  # TODO: Extraction - TODO: Need to dig deeper to see what functions are declare and the prompt given.
  # https://js.langchain.com/docs/modules/chains/openai_functions/extraction

  # TODO: Other Chains
  # https://js.langchain.com/docs/modules/chains/other_chains/
end
