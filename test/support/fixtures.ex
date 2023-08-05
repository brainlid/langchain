defmodule Langchain.Fixtures do
  @moduledoc """
  This module defines test helpers for creating
  entities.
  """
  alias Langchain.ChatModels.ChatOpenAI
  alias Langchain.Message

  def raw_deltas_for_function_call(function_name \\ "hello_world")

  def raw_deltas_for_function_call("hello_world") do
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

  def raw_deltas_for_function_call("calculator") do
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

  @doc """
  Return a list of MessageDelta structs for requesting a function_call.
  """
  def deltas_for_function_call(function_name \\ "hello_world")

  def deltas_for_function_call(function_name) do
    function_name
    |> raw_deltas_for_function_call()
    |> Enum.flat_map(&ChatOpenAI.do_process_response(&1))
  end

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
    Message.new_user!("Analyze the following text and give a detailed and in-depth analysis: \n\n" <> text_chunks(6))
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
end
