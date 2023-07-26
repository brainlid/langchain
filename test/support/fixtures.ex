defmodule Langchain.Fixtures do
  @moduledoc """
  This module defines test helpers for creating
  entities.
  """
  alias Langchain.ChatModels.ChatOpenAI

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

end
