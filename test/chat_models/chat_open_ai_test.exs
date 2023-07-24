defmodule Langchain.ChatModels.ChatOpenAITest do
  use Langchain.BaseCase

  doctest Langchain.ChatModels.ChatOpenAI
  alias Langchain.ChatModels.ChatOpenAI
  alias Langchain.Chains.LlmChain
  alias Langchain.PromptTemplate
  alias Langchain.Functions.Function
  alias Langchain.MessageDelta

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
        Message.new_user("Only using the functions you have been provided with, give a greeting.")

      {:ok, [message]} = ChatOpenAI.call(chat, [message], [hello_world])

      assert %Message{role: :function_call} = message
      assert message.arguments == %{}
      assert message.content == nil
    end
  end

  describe "do_process_response/1" do
    setup do
      delta_content = [
        %{
          "choices" => [
            %{
              "delta" => %{"content" => "", "role" => "assistant"},
              "finish_reason" => nil,
              "index" => 0
            }
          ],
          "created" => 1_689_774_181,
          "id" => "chatcmpl-7e1kD0YMC3AmCycOK4oLGfFMBcCdv",
          "model" => "gpt-4-0613",
          "object" => "chat.completion.chunk"
        },
        %{
          "choices" => [
            %{"delta" => %{"content" => "Hello"}, "finish_reason" => nil, "index" => 0}
          ],
          "created" => 1_689_774_181,
          "id" => "chatcmpl-7e1kD0YMC3AmCycOK4oLGfFMBcCdv",
          "model" => "gpt-4-0613",
          "object" => "chat.completion.chunk"
        },
        %{
          "choices" => [
            %{"delta" => %{"content" => "!"}, "finish_reason" => nil, "index" => 0}
          ],
          "created" => 1_689_774_181,
          "id" => "chatcmpl-7e1kD0YMC3AmCycOK4oLGfFMBcCdv",
          "model" => "gpt-4-0613",
          "object" => "chat.completion.chunk"
        },
        %{
          "choices" => [
            %{"delta" => %{"content" => " How"}, "finish_reason" => nil, "index" => 0}
          ],
          "created" => 1_689_774_181,
          "id" => "chatcmpl-7e1kD0YMC3AmCycOK4oLGfFMBcCdv",
          "model" => "gpt-4-0613",
          "object" => "chat.completion.chunk"
        },
        %{
          "choices" => [
            %{"delta" => %{"content" => " can"}, "finish_reason" => nil, "index" => 0}
          ],
          "created" => 1_689_774_181,
          "id" => "chatcmpl-7e1kD0YMC3AmCycOK4oLGfFMBcCdv",
          "model" => "gpt-4-0613",
          "object" => "chat.completion.chunk"
        },
        %{
          "choices" => [
            %{"delta" => %{"content" => " I"}, "finish_reason" => nil, "index" => 0}
          ],
          "created" => 1_689_774_181,
          "id" => "chatcmpl-7e1kD0YMC3AmCycOK4oLGfFMBcCdv",
          "model" => "gpt-4-0613",
          "object" => "chat.completion.chunk"
        },
        %{
          "choices" => [
            %{
              "delta" => %{"content" => " assist"},
              "finish_reason" => nil,
              "index" => 0
            }
          ],
          "created" => 1_689_774_181,
          "id" => "chatcmpl-7e1kD0YMC3AmCycOK4oLGfFMBcCdv",
          "model" => "gpt-4-0613",
          "object" => "chat.completion.chunk"
        },
        %{
          "choices" => [
            %{"delta" => %{"content" => " you"}, "finish_reason" => nil, "index" => 0}
          ],
          "created" => 1_689_774_181,
          "id" => "chatcmpl-7e1kD0YMC3AmCycOK4oLGfFMBcCdv",
          "model" => "gpt-4-0613",
          "object" => "chat.completion.chunk"
        },
        %{
          "choices" => [
            %{"delta" => %{"content" => " today"}, "finish_reason" => nil, "index" => 0}
          ],
          "created" => 1_689_774_181,
          "id" => "chatcmpl-7e1kD0YMC3AmCycOK4oLGfFMBcCdv",
          "model" => "gpt-4-0613",
          "object" => "chat.completion.chunk"
        },
        %{
          "choices" => [
            %{"delta" => %{"content" => "?"}, "finish_reason" => nil, "index" => 0}
          ],
          "created" => 1_689_774_181,
          "id" => "chatcmpl-7e1kD0YMC3AmCycOK4oLGfFMBcCdv",
          "model" => "gpt-4-0613",
          "object" => "chat.completion.chunk"
        },
        %{
          "choices" => [%{"delta" => %{}, "finish_reason" => "stop", "index" => 0}],
          "created" => 1_689_774_181,
          "id" => "chatcmpl-7e1kD0YMC3AmCycOK4oLGfFMBcCdv",
          "model" => "gpt-4-0613",
          "object" => "chat.completion.chunk"
        }
      ]

      delta_function = [
        %{
          "choices" => [
            %{
              "delta" => %{
                "content" => nil,
                "function_call" => %{"arguments" => "", "name" => "hello_world"},
                "role" => "assistant"
              },
              "finish_reason" => nil,
              "index" => 0
            }
          ],
          "created" => 1_689_775_878,
          "id" => "chatcmpl-7e2BalHi6Ly6AAfZJRLkqLZ4cD4C7",
          "model" => "gpt-4-0613",
          "object" => "chat.completion.chunk"
        },
        %{
          "choices" => [
            %{
              "delta" => %{"function_call" => %{"arguments" => "{}"}},
              "finish_reason" => nil,
              "index" => 0
            }
          ],
          "created" => 1_689_775_878,
          "id" => "chatcmpl-7e2BalHi6Ly6AAfZJRLkqLZ4cD4C7",
          "model" => "gpt-4-0613",
          "object" => "chat.completion.chunk"
        },
        %{
          "choices" => [
            %{"delta" => %{}, "finish_reason" => "function_call", "index" => 0}
          ],
          "created" => 1_689_775_878,
          "id" => "chatcmpl-7e2BalHi6Ly6AAfZJRLkqLZ4cD4C7",
          "model" => "gpt-4-0613",
          "object" => "chat.completion.chunk"
        }
      ]

      %{delta_content: delta_content, delta_function: delta_function}
    end

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

      IO.inspect struct

      assert struct.role == :function_call
      assert struct.content == nil
      assert struct.function_name == "hello_world"
      assert struct.arguments == %{}
      assert struct.complete
    end

    test "handles error from server that the max length has been reached"
    test "handles unsupported response from server"

    test "handles receiving a delta message for a content message at different parts", %{
      delta_content: delta_content
    } do
      msg_1 = Enum.at(delta_content, 0)
      msg_2 = Enum.at(delta_content, 1)
      msg_10 = Enum.at(delta_content, 10)

      expected_1 = %MessageDelta{
        content: nil,
        index: 0,
        function_name: nil,
        role: :assistant,
        arguments: nil,
        complete: false
      }

      [%MessageDelta{} = delta_1] = ChatOpenAI.do_process_response(msg_1)
      assert delta_1 == expected_1

      expected_2 = %MessageDelta{
        content: "Hello",
        index: 0,
        function_name: nil,
        role: :unknown,
        arguments: nil,
        complete: false
      }

      [%MessageDelta{} = delta_2] = ChatOpenAI.do_process_response(msg_2)
      assert delta_2 == expected_2

      expected_10 = %MessageDelta{
        content: nil,
        index: 0,
        function_name: nil,
        role: :unknown,
        arguments: nil,
        complete: true
      }

      [%MessageDelta{} = delta_10] = ChatOpenAI.do_process_response(msg_10)
      assert delta_10 == expected_10

      # results = Enum.flat_map(delta_content, &ChatOpenAI.do_process_response(&1))
      # IO.inspect results

      # results = Enum.flat_map(delta_function, &ChatOpenAI.do_process_response(&1))
      # IO.inspect results

      # results = Enum.flat_map(streamed_function_with_arguments(), &ChatOpenAI.do_process_response(&1))
      # IO.inspect results

      # TODO: Store in-progress message on ChatOpenAI? Could be a list of choices as current message.
      # otherwise write the delta packet message and

      # - can't mutate the chat struct in the callback function.
      # - create the delta message and fire it off.
      # - in a separate process, receive the messages and apply them to a message?
      # - flag when complete

      # use functions but make data aggregation separate from from the OpenAI struct
    end

    test "handles receiving a delta message for a function_call", %{
      delta_function: delta_function
    } do
      msg_1 = Enum.at(delta_function, 0)
      msg_2 = Enum.at(delta_function, 1)
      msg_3 = Enum.at(delta_function, 2)

      expected_1 = %MessageDelta{
        content: nil,
        index: 0,
        function_name: "hello_world",
        role: :function_call,
        arguments: "",
        complete: false
      }

      [%MessageDelta{} = delta_1] = ChatOpenAI.do_process_response(msg_1)
      assert delta_1 == expected_1

      expected_2 = %MessageDelta{
        content: nil,
        index: 0,
        function_name: nil,
        role: :function_call,
        arguments: "{}",
        complete: false
      }

      [%MessageDelta{} = delta_2] = ChatOpenAI.do_process_response(msg_2)
      assert delta_2 == expected_2

      expected_3 = %MessageDelta{
        content: nil,
        index: 0,
        function_name: nil,
        role: :unknown,
        arguments: nil,
        complete: true
      }

      # it should not trim the arguments text
      [%MessageDelta{} = delta_3] = ChatOpenAI.do_process_response(msg_3)
      assert delta_3 == expected_3
    end

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

  describe "streaming examples" do
    @tag :live_call
    test "supports streaming response", %{hello_world: _hello_world} do
      {:ok, chat} = ChatOpenAI.new(%{stream: true, verbose: true})

      {:ok, message} = Message.new_user("Hello!")
      # Message.new_user(
      #   "Only using the functions you have been provided with, give a greeting."
      # )

      callback = fn data ->
        IO.inspect(data, label: "DATA")
        ChatOpenAI.do_process_response(data)
        :ok
      end

      # response = ChatOpenAI.do_api_stream(chat, [message], [hello_world], callback)
      response = ChatOpenAI.do_api_stream(chat, [message], [], callback)
      IO.inspect(response, label: "OPEN AI POST RESPONSE")

      Process.sleep(1_000)
    end

    @tag :live_call
    test "supports streaming response calling function with args" do
      {:ok, chat} = ChatOpenAI.new(%{stream: true, verbose: true})

      {:ok, message} =
        Message.new_user("Answer the following math question: What is 100 + 300 - 200?")

      callback = fn data ->
        IO.inspect(data, label: "DATA")
        ChatOpenAI.do_process_response(data)
        :ok
      end

      response =
        ChatOpenAI.do_api_stream(chat, [message], [Langchain.Tools.Calculator.new!()], callback)

      IO.inspect(response, label: "OPEN AI POST RESPONSE")

      Process.sleep(1_000)
    end
  end


  # TODO: TEST streaming in a function_call? How can I tell? Need ability to flag as complete or not.

  # TODO: TEST that a non-streaming result could return content with "finish_reason" => "length". If so,
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

  defp streamed_function_with_arguments() do
    [
      %{
        "choices" => [
          %{
            "delta" => %{
              "content" => nil,
              "function_call" => %{"arguments" => "", "name" => "calculator"},
              "role" => "assistant"
            },
            "finish_reason" => nil,
            "index" => 0
          }
        ]
      },
      %{
        "choices" => [
          %{
            "delta" => %{"function_call" => %{"arguments" => "{\n"}},
            "finish_reason" => nil,
            "index" => 0
          }
        ]
      },
      %{
        "choices" => [
          %{
            "delta" => %{"function_call" => %{"arguments" => " "}},
            "finish_reason" => nil,
            "index" => 0
          }
        ]
      },
      %{
        "choices" => [
          %{
            "delta" => %{"function_call" => %{"arguments" => " \""}},
            "finish_reason" => nil,
            "index" => 0
          }
        ]
      },
      %{
        "choices" => [
          %{
            "delta" => %{"function_call" => %{"arguments" => "expression"}},
            "finish_reason" => nil,
            "index" => 0
          }
        ]
      },
      %{
        "choices" => [
          %{
            "delta" => %{"function_call" => %{"arguments" => "\":"}},
            "finish_reason" => nil,
            "index" => 0
          }
        ]
      },
      %{
        "choices" => [
          %{
            "delta" => %{"function_call" => %{"arguments" => " \""}},
            "finish_reason" => nil,
            "index" => 0
          }
        ]
      },
      %{
        "choices" => [
          %{
            "delta" => %{"function_call" => %{"arguments" => "100"}},
            "finish_reason" => nil,
            "index" => 0
          }
        ]
      },
      %{
        "choices" => [
          %{
            "delta" => %{"function_call" => %{"arguments" => " +"}},
            "finish_reason" => nil,
            "index" => 0
          }
        ]
      },
      %{
        "choices" => [
          %{
            "delta" => %{"function_call" => %{"arguments" => " "}},
            "finish_reason" => nil,
            "index" => 0
          }
        ]
      },
      %{
        "choices" => [
          %{
            "delta" => %{"function_call" => %{"arguments" => "300"}},
            "finish_reason" => nil,
            "index" => 0
          }
        ]
      },
      %{
        "choices" => [
          %{
            "delta" => %{"function_call" => %{"arguments" => " -"}},
            "finish_reason" => nil,
            "index" => 0
          }
        ]
      },
      %{
        "choices" => [
          %{
            "delta" => %{"function_call" => %{"arguments" => " "}},
            "finish_reason" => nil,
            "index" => 0
          }
        ]
      },
      %{
        "choices" => [
          %{
            "delta" => %{"function_call" => %{"arguments" => "200"}},
            "finish_reason" => nil,
            "index" => 0
          }
        ]
      },
      %{
        "choices" => [
          %{
            "delta" => %{"function_call" => %{"arguments" => "\"\n"}},
            "finish_reason" => nil,
            "index" => 0
          }
        ]
      },
      %{
        "choices" => [
          %{
            "delta" => %{"function_call" => %{"arguments" => "}"}},
            "finish_reason" => nil,
            "index" => 0
          }
        ]
      },
      %{
        "choices" => [
          %{
            "delta" => %{},
            "finish_reason" => "function_call",
            "index" => 0
          }
        ]
      }
    ]
  end
end
