defmodule Langchain.ChatModels.ChatOpenAITest do
  use Langchain.BaseCase

  doctest Langchain.ChatModels.ChatOpenAI
  alias Langchain.ChatModels.ChatOpenAI
  alias Langchain.Chains.LlmChain
  alias Langchain.PromptTemplate
  alias Langchain.Functions.Function

  setup do
    {:ok, hello_world} =
      Function.new(%{
        name: "hello_world",
        description: "Give a hello world greeting.",
        function: fn -> IO.puts("Hello world!") end
      })

    %{hello_world: hello_world}
  end

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

    @tag :live_call
    test "executing a function", %{hello_world: hello_world} do
      {:ok, chat} = ChatOpenAI.new(%{verbose: true})

      {:ok, message} =
        Message.new_user(
          "Only using the functions you have been provided with, give a greeting."
        )

      {:ok, [message]} = ChatOpenAI.call(chat, [message], [hello_world])

      assert %Message{role: :function_call} = message
      assert message.arguments == %{}
      assert message.content == nil
    end
  end

  describe "do_process_response/1" do
    test "handles receiving a complete message" do
      response = %{
        "message" => %{"role" => "assistant", "content" => "Greetings!", "index" => 1},
        "finish_reason" => "stop"
      }

      assert %Message{} = struct = ChatOpenAI.do_process_response(response)
      assert struct.role == :assistant
      assert struct.content == "Greetings!"
      assert struct.index == 1
      assert struct.complete
    end

    test "handles receiving a function_call message" do
      response = %{
        "finish_reason" => "function_call",
        "index" => 0,
        "message" => %{
          "content" => nil,
          "function_call" => %{"arguments" => "{}", "name" => "hello_world"},
          "role" => "assistant"
        }
      }

      assert %Message{} = struct = ChatOpenAI.do_process_response(response)
      assert struct.role == :function_call
      assert struct.content == nil
      assert struct.function_name == "hello_world"
      assert struct.arguments == %{}
      assert struct.complete
    end

    test "handles error from server that the max length has been reached"
    test "handles unsupported response from server"
    test "handles receiving a delta message with different portions"
    test "handles receiving a delta message when complete and incomplete"
    test "handles receiving error message from server"

    test "return multiple responses when given multiple choices" do
      # received multiple responses because multiples were requested.
      response = %{
        "choices" => [
          %{
            "message" => %{"role" => "assistant", "content" => "Greetings!", "index" => 1},
            "finish_reason" => "stop"
          },
          %{
            "message" => %{"role" => "assistant", "content" => "Howdy!", "index" => 1},
            "finish_reason" => "stop"
          }
        ]
      }

      [msg1, msg2] = ChatOpenAI.do_process_response(response)
      assert %Message{role: :assistant, index: 1, complete: true} = msg1
      assert %Message{role: :assistant, index: 1, complete: true} = msg2
      assert msg1.content == "Greetings!"
      assert msg2.content == "Howdy!"
    end
  end

  #TODO: TEST streaming in a function_call? How can I tell? Need ability to flag as complete or not.

  #TODO: TEST that a non-streaming result could return content with "finish_reason" => "length". If so,
  #      I would need to store content on a message AND flag the length error.

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
