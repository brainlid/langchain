defmodule Langchain.ChatModels.ChatOpenAITest do
  use Langchain.BaseCase

  doctest Langchain.ChatModels.ChatOpenAI
  alias Langchain.ChatModels.ChatOpenAI
  alias Langchain.Chains.LlmChain
  alias Langchain.PromptTemplate

  describe "new/1" do
    test "works with minimal attr" do
      assert {:ok, %ChatOpenAI{} = openai} = ChatOpenAI.new(%{"model" => "gpt-3.5-turbo-0613"})
      assert openai.model == "gpt-3.5-turbo-0613"
    end

    test "returns error when invalid" do
      assert {:error, changeset} = ChatOpenAI.new(%{"model" => nil})
      refute changeset.valid?
      assert {"can't be blank", _} = changeset.errors[:model]
    end
  end

  describe "for_api/3" do
    test "generates a map for an API call" do
      {:ok, openai} =
        ChatOpenAI.new(%{
          "model" => "gpt-3.5-turbo-0613",
          "temperature" => 1,
          "frequency_penalty" => 0.5
        })

      data = ChatOpenAI.for_api(openai, [], [])
      assert data.model == "gpt-3.5-turbo-0613"
      assert data.temperature == 1
      assert data.frequency_penalty == 0.5
    end
  end

  describe "call/2" do
    @tag :live_call
    test "basic example" do
      # set_fake_llm_response({:ok, Message.new_assistant("\n\nRainbow Sox Co.")})

      # https://js.langchain.com/docs/modules/models/chat/
      {:ok, chat} = ChatOpenAI.new(%{temperature: 1})

      {:ok, %Message{role: :assistant, content: response}} =
        ChatOpenAI.call(chat, [
          Message.new_user!("Return the response 'Colorful Threads'.")
        ])

      assert response == "Colorful Threads"
    end
  end

  # TODO: prompt template work/tests. Doesn't include API calls.

  # import {
  #   ChatPromptTemplate,
  #   HumanMessagePromptTemplate,
  #   PromptTemplate,
  #   SystemMessagePromptTemplate,
  # } from "langchain/prompts";

  # export const run = async () => {
  #   // A `PromptTemplate` consists of a template string and a list of input variables.
  #   const template = "What is a good name for a company that makes {product}?";
  #   const promptA = new PromptTemplate({ template, inputVariables: ["product"] });

  #   // We can use the `format` method to format the template with the given input values.
  #   const responseA = await promptA.format({ product: "colorful socks" });
  #   console.log({ responseA });
  #   /*
  #   {
  #     responseA: 'What is a good name for a company that makes colorful socks?'
  #   }
  #   */

  #   // We can also use the `fromTemplate` method to create a `PromptTemplate` object.
  #   const promptB = PromptTemplate.fromTemplate(
  #     "What is a good name for a company that makes {product}?"
  #   );
  #   const responseB = await promptB.format({ product: "colorful socks" });
  #   console.log({ responseB });
  #   /*
  #   {
  #     responseB: 'What is a good name for a company that makes colorful socks?'
  #   }
  #   */

  #   // For chat models, we provide a `ChatPromptTemplate` class that can be used to format chat prompts.
  #   const chatPrompt = ChatPromptTemplate.fromPromptMessages([
  #     SystemMessagePromptTemplate.fromTemplate(
  #       "You are a helpful assistant that translates {input_language} to {output_language}."
  #     ),
  #     HumanMessagePromptTemplate.fromTemplate("{text}"),
  #   ]);

  #   // The result can be formatted as a string using the `format` method.
  #   const responseC = await chatPrompt.format({
  #     input_language: "English",
  #     output_language: "French",
  #     text: "I love programming.",
  #   });
  #   console.log({ responseC });
  #   /*
  #   {
  #     responseC: '[{"text":"You are a helpful assistant that translates English to French."},{"text":"I love programming."}]'
  #   }
  #   */

  #   // The result can also be formatted as a list of `ChatMessage` objects by returning a `PromptValue` object and calling the `toChatMessages` method.
  #   // More on this below.
  #   const responseD = await chatPrompt.formatPromptValue({
  #     input_language: "English",
  #     output_language: "French",
  #     text: "I love programming.",
  #   });
  #   const messages = responseD.toChatMessages();
  #   console.log({ messages });
  #   /*
  #   {
  #     messages: [
  #         SystemMessage {
  #           text: 'You are a helpful assistant that translates English to French.'
  #         },
  #         HumanMessage { text: 'I love programming.' }
  #       ]
  #   }
  #   */
  # };
end
