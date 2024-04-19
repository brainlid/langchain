defmodule LangChain.Fixtures do
  @moduledoc """
  This module defines test helpers for creating
  entities.
  """
  alias LangChain.ChatModels.ChatOpenAI
  alias LangChain.Message
  alias LangChain.Message.ToolCall

  def raw_deltas_for_tool_call(tool_name \\ "hello_world")

  def raw_deltas_for_tool_call("hello_world") do
    [
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
  end

  @doc """
  Return a list of MessageDelta structs for requesting a tool_call.
  """
  def deltas_for_tool_call(tool_name \\ "hello_world")

  # TODO: NEEDS call_id
  def deltas_for_tool_call("calculator") do
    [
      %LangChain.MessageDelta{
        content: nil,
        status: :incomplete,
        index: 0,
        role: :assistant,
        tool_calls: nil
      },
      %LangChain.MessageDelta{
        content: nil,
        status: :incomplete,
        index: 0,
        role: :unknown,
        tool_calls: [
          %LangChain.Message.ToolCall{
            status: :incomplete,
            type: :function,
            call_id: "call_IBDsG5rtgR9rt1CNrWkPMvXG",
            name: "calculator",
            arguments: nil,
            index: 0
          }
        ]
      },
      %LangChain.MessageDelta{
        content: nil,
        status: :incomplete,
        index: 0,
        role: :unknown,
        tool_calls: [
          %LangChain.Message.ToolCall{
            status: :incomplete,
            type: :function,
            call_id: nil,
            name: nil,
            arguments: "{\n \"",
            index: 0
          }
        ]
      },
      %LangChain.MessageDelta{
        content: nil,
        status: :incomplete,
        index: 0,
        role: :unknown,
        tool_calls: [
          %LangChain.Message.ToolCall{
            status: :incomplete,
            type: :function,
            call_id: nil,
            name: nil,
            arguments: "expression\": \"100 + 300 - 200\"}",
            index: 0
          }
        ]
      },
      %LangChain.MessageDelta{
        content: nil,
        status: :complete,
        index: 0,
        role: :unknown,
        tool_calls: nil
      }
    ]
  end

  def deltas_for_tool_call(tool_name) do
    tool_name
    |> raw_deltas_for_tool_call()
    |> Enum.flat_map(&ChatOpenAI.do_process_response(&1))
  end

  @doc """
  Return a list of MessageDelta structs that includes multiple tool calls.
  """
  def deltas_for_multiple_tool_calls() do
    [
      %LangChain.MessageDelta{
        content: nil,
        status: :incomplete,
        index: 0,
        role: :assistant,
        tool_calls: nil
      },
      %LangChain.MessageDelta{
        content: nil,
        status: :incomplete,
        index: 0,
        role: :unknown,
        tool_calls: [
          %LangChain.Message.ToolCall{
            status: :incomplete,
            type: :function,
            call_id: "call_123",
            name: "get_weather",
            arguments: nil,
            index: 0
          }
        ]
      },
      %LangChain.MessageDelta{
        content: nil,
        status: :incomplete,
        index: 0,
        role: :unknown,
        tool_calls: [
          %LangChain.Message.ToolCall{
            status: :incomplete,
            type: :function,
            call_id: nil,
            name: nil,
            arguments: "{\"ci",
            index: 0
          }
        ]
      },
      %LangChain.MessageDelta{
        content: nil,
        status: :incomplete,
        index: 0,
        role: :unknown,
        tool_calls: [
          %LangChain.Message.ToolCall{
            status: :incomplete,
            type: :function,
            call_id: nil,
            name: nil,
            arguments: "ty\": \"Moab\", \"state\": \"UT\"}",
            index: 0
          }
        ]
      },
      %LangChain.MessageDelta{
        content: nil,
        status: :incomplete,
        index: 0,
        role: :unknown,
        tool_calls: [
          %LangChain.Message.ToolCall{
            status: :incomplete,
            type: :function,
            call_id: "call_234",
            name: "get_weather",
            arguments: nil,
            index: 1
          }
        ]
      },
      %LangChain.MessageDelta{
        content: nil,
        status: :incomplete,
        index: 0,
        role: :unknown,
        tool_calls: [
          %LangChain.Message.ToolCall{
            status: :incomplete,
            type: :function,
            call_id: nil,
            name: nil,
            arguments: "{\"ci",
            index: 1
          }
        ]
      },
      %LangChain.MessageDelta{
        content: nil,
        status: :incomplete,
        index: 0,
        role: :unknown,
        tool_calls: [
          %LangChain.Message.ToolCall{
            status: :incomplete,
            type: :function,
            call_id: nil,
            name: nil,
            arguments: "ty\": \"Portland\", \"state\": \"OR\"}",
            index: 1
          }
        ]
      },
      %LangChain.MessageDelta{
        content: nil,
        status: :incomplete,
        index: 0,
        role: :unknown,
        tool_calls: [
          %LangChain.Message.ToolCall{
            status: :incomplete,
            type: :function,
            call_id: "call_345",
            name: "get_weather",
            arguments: nil,
            index: 2
          }
        ]
      },
      %LangChain.MessageDelta{
        content: nil,
        status: :incomplete,
        index: 0,
        role: :unknown,
        tool_calls: [
          %LangChain.Message.ToolCall{
            status: :incomplete,
            type: :function,
            call_id: nil,
            name: nil,
            arguments: "{\"ci",
            index: 2
          }
        ]
      },
      %LangChain.MessageDelta{
        content: nil,
        status: :incomplete,
        index: 0,
        role: :unknown,
        tool_calls: [
          %LangChain.Message.ToolCall{
            status: :incomplete,
            type: :function,
            call_id: nil,
            name: nil,
            arguments: "ty\": \"Baltimore\", \"state\": \"MD\"}",
            index: 2
          }
        ]
      },
      %LangChain.MessageDelta{
        content: nil,
        status: :complete,
        index: 0,
        role: :unknown,
        tool_calls: nil
      }
    ]
  end

  # TODO: These are specific to OpenAI. Move into that test module?
  def raw_deltas_for_content() do
    [
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
  end

  def too_large_user_request() do
    Message.new_user!("Analyze the following text: \n\n" <> text_chunks(8))
  end

  def results_in_too_long_response() do
    Message.new_user!(
      "Analyze the following text and give a detailed and in-depth analysis: \n\n" <>
        text_chunks(6)
    )
  end

  defp text_chunks(num) do
    # taken from https://www.grimmstories.com/en/grimm_fairy-tales/snow-white_and_rose-red
    # - expired copyright
    Enum.reduce(1..num//1, "", fn _n, text ->
      text <>
        """
        There was once a poor widow who lived in a lonely cottage. In front of the cottage was a garden wherein stood two rose-trees, one of which bore white and the other red roses. She had two children who were like the two rose-trees, and one was called Snow-white, and the other Rose-red. They were as good and happy, as busy and cheerful as ever two children in the world were, only Snow-white was more quiet and gentle than Rose- red. Rose-red liked better to run about in the meadows and fields seeking flowers and catching butterflies; but Snow-white sat at home with her mother, and helped her with her house-work, or read to her when there was nothing to do.

        The two children were so fond of each another that they always held each other by the hand when they went out together, and when Snow-white said, "We will not leave each other," Rose-red answered, "Never so long as we live," and their mother would add, "What one has she must share with the other."

        They often ran about the forest alone and gathered red berries, and no beasts did them any harm, but came close to them trustfully. The little hare would eat a cabbage-leaf out of their hands, the roe grazed by their side, the stag leapt merrily by them, and the birds sat still upon the boughs, and sang whatever they knew.

        No mishap overtook them; if they had stayed too late in the forest, and night came on, they laid themselves down near one another upon the moss, and slept until morning came, and their mother knew this and had no distress on their account.

        Once when they had spent the night in the wood and the dawn had roused them, they saw a beautiful child in a shining white dress sitting near their bed. He got up and looked quite kindly at them, but said nothing and went away into the forest. And when they looked round they found that they had been sleeping quite close to a precipice, and would certainly have fallen into it in the darkness if they had gone only a few paces further. And their mother told them that it must have been the angel who watches over good children.

        Snow-white and Rose-red kept their mother's little cottage so neat that it was a pleasure to look inside it. In the summer Rose-red took care of the house, and every morning laid a wreath of flowers by her mother's bed before she awoke, in which was a rose from each tree. In the winter Snow-white lit the fire and hung the kettle on the wrekin. The kettle was of copper and shone like gold, so brightly was it polished. In the evening, when the snowflakes fell, the mother said, "Go, Snow-white, and bolt the door," and then they sat round the hearth, and the mother took her spectacles and read aloud out of a large book, and the two girls listened as they sat and span. And close by them lay a lamb upon the floor, and behind them upon a perch sat a white dove with its head hidden beneath its wings.
        """
    end)
  end

  def delta_content_sample do
    # built from actual responses parsed to MessageDeltas
    # results = Enum.flat_map(delta_content, &ChatOpenAI.do_process_response(&1))
    # IO.inspect results
    [
      %LangChain.MessageDelta{
        content: nil,
        index: 0,
        role: :assistant,
        status: :incomplete
      },
      %LangChain.MessageDelta{
        content: "Hello",
        index: 0,
        role: :unknown,
        status: :incomplete
      },
      %LangChain.MessageDelta{
        content: "!",
        index: 0,
        role: :unknown,
        status: :incomplete
      },
      %LangChain.MessageDelta{
        content: " How",
        index: 0,
        role: :unknown,
        status: :incomplete
      },
      %LangChain.MessageDelta{
        content: " can",
        index: 0,
        role: :unknown,
        status: :incomplete
      },
      %LangChain.MessageDelta{
        content: " I",
        index: 0,
        role: :unknown,
        status: :incomplete
      },
      %LangChain.MessageDelta{
        content: " assist",
        index: 0,
        role: :unknown,
        status: :incomplete
      },
      %LangChain.MessageDelta{
        content: " you",
        index: 0,
        role: :unknown,
        status: :incomplete
      },
      %LangChain.MessageDelta{
        content: " today",
        index: 0,
        role: :unknown,
        status: :incomplete
      },
      %LangChain.MessageDelta{
        content: "?",
        index: 0,
        role: :unknown,
        status: :incomplete
      },
      %LangChain.MessageDelta{
        content: nil,
        index: 0,
        role: :unknown,
        status: :complete
      }
    ]
  end

  def delta_function_no_args() do
    [
      %LangChain.MessageDelta{
        content: nil,
        index: 0,
        function_name: "hello_world",
        role: :assistant,
        status: :incomplete
      },
      %LangChain.MessageDelta{
        content: nil,
        index: 0,
        role: :assistant,
        arguments: "{}",
        status: :incomplete
      },
      %LangChain.MessageDelta{
        content: nil,
        index: 0,
        role: :unknown,
        status: :complete
      }
    ]
  end

  def delta_function_streamed_args() do
    [
      %LangChain.MessageDelta{
        content: nil,
        index: 0,
        function_name: "calculator",
        role: :assistant,
        arguments: "",
        status: :incomplete
      },
      %LangChain.MessageDelta{
        content: nil,
        index: 0,
        role: :assistant,
        arguments: "{\n",
        status: :incomplete
      },
      %LangChain.MessageDelta{
        content: nil,
        index: 0,
        role: :assistant,
        arguments: " ",
        status: :incomplete
      },
      %LangChain.MessageDelta{
        content: nil,
        index: 0,
        role: :assistant,
        arguments: " \"",
        status: :incomplete
      },
      %LangChain.MessageDelta{
        content: nil,
        index: 0,
        role: :assistant,
        arguments: "expression",
        status: :incomplete
      },
      %LangChain.MessageDelta{
        content: nil,
        index: 0,
        role: :assistant,
        arguments: "\":",
        status: :incomplete
      },
      %LangChain.MessageDelta{
        content: nil,
        index: 0,
        role: :assistant,
        arguments: " \"",
        status: :incomplete
      },
      %LangChain.MessageDelta{
        content: nil,
        index: 0,
        role: :assistant,
        arguments: "100",
        status: :incomplete
      },
      %LangChain.MessageDelta{
        content: nil,
        index: 0,
        role: :assistant,
        arguments: " +",
        status: :incomplete
      },
      %LangChain.MessageDelta{
        content: nil,
        index: 0,
        role: :assistant,
        arguments: " ",
        status: :incomplete
      },
      %LangChain.MessageDelta{
        content: nil,
        index: 0,
        role: :assistant,
        arguments: "300",
        status: :incomplete
      },
      %LangChain.MessageDelta{
        content: nil,
        index: 0,
        role: :assistant,
        arguments: " -",
        status: :incomplete
      },
      %LangChain.MessageDelta{
        content: nil,
        index: 0,
        role: :assistant,
        arguments: " ",
        status: :incomplete
      },
      %LangChain.MessageDelta{
        content: nil,
        index: 0,
        role: :assistant,
        arguments: "200",
        status: :incomplete
      },
      %LangChain.MessageDelta{
        content: nil,
        index: 0,
        role: :assistant,
        arguments: "\"\n",
        status: :incomplete
      },
      %LangChain.MessageDelta{
        content: nil,
        index: 0,
        role: :assistant,
        arguments: "}",
        status: :incomplete
      },
      %LangChain.MessageDelta{
        content: nil,
        index: 0,
        role: :unknown,
        status: :complete
      }
    ]
  end

  # TODO: DELETE THIS?
  def delta_content_with_function_call() do
    # OpenAI treats a function_call as extra information with an `:assistant`
    # response. We treat a function_call as it's own thing.
    #
    # This replicates the data returned in that type of response.
    [
      [
        %LangChain.MessageDelta{
          content: "",
          status: :incomplete,
          index: 0,
          role: :assistant
        }
      ],
      [
        %LangChain.MessageDelta{
          content: "Sure",
          status: :incomplete,
          index: 0,
          role: :unknown
        }
      ],
      [
        %LangChain.MessageDelta{
          content: ",",
          status: :incomplete,
          index: 0,
          role: :unknown
        }
      ],
      [
        %LangChain.MessageDelta{
          content: " I",
          status: :incomplete,
          index: 0,
          role: :unknown
        }
      ],
      [
        %LangChain.MessageDelta{
          content: " can",
          status: :incomplete,
          index: 0,
          role: :unknown
        }
      ],
      [
        %LangChain.MessageDelta{
          content: " help",
          status: :incomplete,
          index: 0,
          role: :unknown
        }
      ],
      [
        %LangChain.MessageDelta{
          content: " with",
          status: :incomplete,
          index: 0,
          role: :unknown
        }
      ],
      [
        %LangChain.MessageDelta{
          content: " that",
          status: :incomplete,
          index: 0,
          role: :unknown
        }
      ],
      [
        %LangChain.MessageDelta{
          content: ".",
          status: :incomplete,
          index: 0,
          role: :unknown
        }
      ],
      [
        %LangChain.MessageDelta{
          content: " First",
          status: :incomplete,
          index: 0,
          role: :unknown
        }
      ],
      [
        %LangChain.MessageDelta{
          content: ",",
          status: :incomplete,
          index: 0,
          role: :unknown
        }
      ],
      [
        %LangChain.MessageDelta{
          content: " let",
          status: :incomplete,
          index: 0,
          role: :unknown
        }
      ],
      [
        %LangChain.MessageDelta{
          content: "'s",
          status: :incomplete,
          index: 0,
          role: :unknown
        }
      ],
      [
        %LangChain.MessageDelta{
          content: " check",
          status: :incomplete,
          index: 0,
          role: :unknown
        }
      ],
      [
        %LangChain.MessageDelta{
          content: " which",
          status: :incomplete,
          index: 0,
          role: :unknown
        }
      ],
      [
        %LangChain.MessageDelta{
          content: " regions",
          status: :incomplete,
          index: 0,
          role: :unknown
        }
      ],
      [
        %LangChain.MessageDelta{
          content: " are",
          status: :incomplete,
          index: 0,
          role: :unknown
        }
      ],
      [
        %LangChain.MessageDelta{
          content: " currently",
          status: :incomplete,
          index: 0,
          role: :unknown
        }
      ],
      [
        %LangChain.MessageDelta{
          content: " available",
          status: :incomplete,
          index: 0,
          role: :unknown
        }
      ],
      [
        %LangChain.MessageDelta{
          content: " for",
          status: :incomplete,
          index: 0,
          role: :unknown
        }
      ],
      [
        %LangChain.MessageDelta{
          content: " deployment",
          status: :incomplete,
          index: 0,
          role: :unknown
        }
      ],
      [
        %LangChain.MessageDelta{
          content: " on",
          status: :incomplete,
          index: 0,
          role: :unknown
        }
      ],
      [
        %LangChain.MessageDelta{
          content: " Fly",
          status: :incomplete,
          index: 0,
          role: :unknown
        }
      ],
      [
        %LangChain.MessageDelta{
          content: ".io",
          status: :incomplete,
          index: 0,
          role: :unknown
        }
      ],
      [
        %LangChain.MessageDelta{
          content: ".",
          status: :incomplete,
          index: 0,
          role: :unknown
        }
      ],
      [
        %LangChain.MessageDelta{
          content: " Please",
          status: :incomplete,
          index: 0,
          role: :unknown
        }
      ],
      [
        %LangChain.MessageDelta{
          content: " wait",
          status: :incomplete,
          index: 0,
          role: :unknown
        }
      ],
      [
        %LangChain.MessageDelta{
          content: " a",
          status: :incomplete,
          index: 0,
          role: :unknown
        }
      ],
      [
        %LangChain.MessageDelta{
          content: " moment",
          status: :incomplete,
          index: 0,
          role: :unknown
        }
      ],
      [
        %LangChain.MessageDelta{
          content: " while",
          status: :incomplete,
          index: 0,
          role: :unknown
        }
      ],
      [
        %LangChain.MessageDelta{
          content: " I",
          status: :incomplete,
          index: 0,
          role: :unknown
        }
      ],
      [
        %LangChain.MessageDelta{
          content: " fetch",
          status: :incomplete,
          index: 0,
          role: :unknown
        }
      ],
      [
        %LangChain.MessageDelta{
          content: " this",
          status: :incomplete,
          index: 0,
          role: :unknown
        }
      ],
      [
        %LangChain.MessageDelta{
          content: " information",
          status: :incomplete,
          index: 0,
          role: :unknown
        }
      ],
      [
        %LangChain.MessageDelta{
          content: " for",
          status: :incomplete,
          index: 0,
          role: :unknown
        }
      ],
      [
        %LangChain.MessageDelta{
          content: " you",
          status: :incomplete,
          index: 0,
          role: :unknown
        }
      ],
      [
        %LangChain.MessageDelta{
          content: ".",
          status: :incomplete,
          index: 0,
          role: :unknown
        }
      ],
      [
        %LangChain.MessageDelta{
          content: nil,
          status: :incomplete,
          index: 0,
          tool_calls: [%ToolCall{call_id: "call_123", name: "regions_list", index: 0}],
          role: :tool_call,
        }
      ],
      [
        %LangChain.MessageDelta{
          content: nil,
          status: :incomplete,
          index: 0,
          role: :tool_call,
          tool_calls: [%ToolCall{arguments: "{}", index: 0}],
        }
      ],
      [
        %LangChain.MessageDelta{
          content: nil,
          status: :complete,
          index: 0,
          role: :unknown
        }
      ]
    ]
  end

  def new_function_call!(call_id, name, arguments) do
    Message.new_assistant!(%{
      tool_calls: [
        ToolCall.new!(%{
          type: :function,
          status: :complete,
          call_id: call_id,
          name: name,
          arguments: arguments
        })
      ]
    })
  end

  def new_function_call!(%ToolCall{} = call) do
    Message.new_assistant!(%{tool_calls: [call]})
  end

  def new_function_calls!(calls) do
    Message.new_assistant!(%{tool_calls: calls})
  end
end
