defmodule LangChain.TrajectoryTest do
  use ExUnit.Case, async: true
  use LangChain.TrajectoryAssertions

  alias LangChain.Trajectory
  alias LangChain.Chains.LLMChain
  alias LangChain.ChatModels.ChatOpenAI
  alias LangChain.Message
  alias LangChain.Message.ToolCall
  alias LangChain.Message.ToolResult
  alias LangChain.TokenUsage

  # Helper to build a chain with pre-populated exchanged_messages
  defp chain_with_messages(messages) do
    {:ok, chat} = ChatOpenAI.new(%{temperature: 0})
    %LLMChain{} = chain = LLMChain.new!(%{llm: chat})
    %LLMChain{chain | exchanged_messages: messages}
  end

  defp assistant_msg(content, opts \\ []) do
    tool_calls = Keyword.get(opts, :tool_calls, nil)
    usage = Keyword.get(opts, :usage, nil)

    msg =
      Message.new!(%{
        role: :assistant,
        content: content,
        status: :complete,
        tool_calls: tool_calls
      })

    if usage, do: TokenUsage.set(msg, usage), else: msg
  end

  defp user_msg(content) do
    Message.new_user!(content)
  end

  defp tool_msg(results) do
    Message.new!(%{
      role: :tool,
      tool_results: results
    })
  end

  defp make_tool_call(name, arguments, call_id \\ nil) do
    ToolCall.new!(%{
      status: :complete,
      type: :function,
      call_id: call_id || "call_#{name}",
      name: name,
      arguments: arguments
    })
  end

  defp make_tool_result(name, content, call_id \\ nil) do
    ToolResult.new!(%{
      tool_call_id: call_id || "call_#{name}",
      name: name,
      content: content
    })
  end

  defp make_usage(input, output) do
    TokenUsage.new!(%{input: input, output: output})
  end

  describe "from_chain/1" do
    test "returns empty trajectory for chain with no messages" do
      chain = chain_with_messages([])
      trajectory = Trajectory.from_chain(chain)

      assert trajectory.messages == []
      assert trajectory.tool_calls == []
      assert trajectory.token_usage == nil
    end

    test "returns trajectory with no tool calls for text-only messages" do
      messages = [
        user_msg("Hello"),
        assistant_msg("Hi there!")
      ]

      trajectory = Trajectory.from_chain(chain_with_messages(messages))

      assert trajectory.messages == messages
      assert trajectory.tool_calls == []
    end

    test "extracts single tool call" do
      tc = make_tool_call("search", %{"query" => "weather"})

      messages = [
        user_msg("What's the weather?"),
        assistant_msg(nil, tool_calls: [tc])
      ]

      trajectory = Trajectory.from_chain(chain_with_messages(messages))

      assert [%{name: "search", arguments: %{"query" => "weather"}}] = trajectory.tool_calls
    end

    test "extracts multiple tool calls across messages" do
      tc1 = make_tool_call("search", %{"query" => "weather"})
      tc2 = make_tool_call("get_forecast", %{"city" => "Paris"})
      tr1 = make_tool_result("search", "Sunny")

      messages = [
        user_msg("What's the weather in Paris?"),
        assistant_msg(nil, tool_calls: [tc1]),
        tool_msg([tr1]),
        assistant_msg(nil, tool_calls: [tc2])
      ]

      trajectory = Trajectory.from_chain(chain_with_messages(messages))

      assert [
               %{name: "search", arguments: %{"query" => "weather"}},
               %{name: "get_forecast", arguments: %{"city" => "Paris"}}
             ] = trajectory.tool_calls
    end

    test "extracts multiple tool calls from a single message" do
      tc1 = make_tool_call("search", %{"query" => "weather"}, "call_1")
      tc2 = make_tool_call("search", %{"query" => "news"}, "call_2")

      messages = [
        user_msg("Search for weather and news"),
        assistant_msg(nil, tool_calls: [tc1, tc2])
      ]

      trajectory = Trajectory.from_chain(chain_with_messages(messages))

      assert [
               %{name: "search", arguments: %{"query" => "weather"}},
               %{name: "search", arguments: %{"query" => "news"}}
             ] = trajectory.tool_calls
    end

    test "aggregates token usage across assistant messages" do
      usage1 = make_usage(10, 20)
      usage2 = make_usage(5, 15)

      messages = [
        user_msg("Hello"),
        assistant_msg("Hi", usage: usage1),
        user_msg("More"),
        assistant_msg("Sure", usage: usage2)
      ]

      trajectory = Trajectory.from_chain(chain_with_messages(messages))

      assert trajectory.token_usage.input == 15
      assert trajectory.token_usage.output == 35
    end

    test "returns nil token usage when no messages have usage" do
      messages = [
        user_msg("Hello"),
        assistant_msg("Hi")
      ]

      trajectory = Trajectory.from_chain(chain_with_messages(messages))
      assert trajectory.token_usage == nil
    end

    test "populates metadata with model and llm_module" do
      messages = [user_msg("Hello"), assistant_msg("Hi")]
      trajectory = Trajectory.from_chain(chain_with_messages(messages))

      assert trajectory.metadata.model == "gpt-3.5-turbo"
      assert trajectory.metadata.llm_module == ChatOpenAI
    end
  end

  describe "to_map/1" do
    test "serializes empty trajectory" do
      trajectory = %Trajectory{messages: [], tool_calls: [], token_usage: nil}
      result = Trajectory.to_map(trajectory)

      assert result == %{messages: [], tool_calls: [], token_usage: nil, metadata: %{}}
    end

    test "serializes messages with content" do
      trajectory = %Trajectory{
        messages: [user_msg("Hello"), assistant_msg("Hi")],
        tool_calls: [],
        token_usage: nil
      }

      result = Trajectory.to_map(trajectory)

      assert [
               %{role: :user, content: "Hello"},
               %{role: :assistant, content: "Hi"}
             ] = result.messages
    end

    test "serializes tool calls in messages" do
      tc = make_tool_call("search", %{"q" => "test"})

      trajectory = %Trajectory{
        messages: [assistant_msg(nil, tool_calls: [tc])],
        tool_calls: [%{name: "search", arguments: %{"q" => "test"}}],
        token_usage: nil
      }

      result = Trajectory.to_map(trajectory)

      assert [%{tool_calls: [%{name: "search", arguments: %{"q" => "test"}}]}] = result.messages
    end

    test "serializes tool results in messages" do
      tr = make_tool_result("search", "Result text")

      trajectory = %Trajectory{
        messages: [tool_msg([tr])],
        tool_calls: [],
        token_usage: nil
      }

      result = Trajectory.to_map(trajectory)

      assert [%{tool_results: [%{name: "search", content: "Result text", is_error: false}]}] =
               result.messages
    end

    test "serializes token usage" do
      trajectory = %Trajectory{
        messages: [],
        tool_calls: [],
        token_usage: make_usage(10, 20)
      }

      assert %{token_usage: %{input: 10, output: 20}} = Trajectory.to_map(trajectory)
    end

    test "handles nil content" do
      trajectory = %Trajectory{
        messages: [assistant_msg(nil)],
        tool_calls: [],
        token_usage: nil
      }

      result = Trajectory.to_map(trajectory)
      assert [%{content: nil}] = result.messages
    end

    test "serializes metadata" do
      trajectory = %Trajectory{
        messages: [],
        tool_calls: [],
        token_usage: nil,
        metadata: %{model: "gpt-4", llm_module: ChatOpenAI}
      }

      result = Trajectory.to_map(trajectory)
      assert result.metadata == %{model: "gpt-4", llm_module: ChatOpenAI}
    end
  end

  describe "from_map/1" do
    test "roundtrips tool_calls through to_map/from_map" do
      original = %Trajectory{
        messages: [],
        tool_calls: [
          %{name: "search", arguments: %{"query" => "weather"}},
          %{name: "get_forecast", arguments: %{"city" => "Paris"}}
        ],
        token_usage: make_usage(10, 20)
      }

      restored = original |> Trajectory.to_map() |> Trajectory.from_map()

      assert restored.tool_calls == original.tool_calls
      assert restored.token_usage.input == 10
      assert restored.token_usage.output == 20
    end

    test "handles string keys from JSON decoding" do
      json_map = %{
        "messages" => [],
        "tool_calls" => [
          %{"name" => "search", "arguments" => %{"q" => "test"}}
        ],
        "token_usage" => %{"input" => 5, "output" => 10}
      }

      trajectory = Trajectory.from_map(json_map)

      assert [%{name: "search", arguments: %{"q" => "test"}}] = trajectory.tool_calls
      assert trajectory.token_usage.input == 5
      assert trajectory.token_usage.output == 10
    end

    test "handles nil token_usage" do
      map = %{messages: [], tool_calls: [], token_usage: nil}
      trajectory = Trajectory.from_map(map)

      assert trajectory.token_usage == nil
    end

    test "handles missing keys with defaults" do
      trajectory = Trajectory.from_map(%{})

      assert trajectory.messages == []
      assert trajectory.tool_calls == []
      assert trajectory.token_usage == nil
    end

    test "preserves messages as raw maps" do
      map = %{
        messages: [%{role: :user, content: "Hello"}],
        tool_calls: [],
        token_usage: nil
      }

      trajectory = Trajectory.from_map(map)
      assert [%{role: :user, content: "Hello"}] = trajectory.messages
    end

    test "roundtrips metadata through to_map/from_map" do
      original = %Trajectory{
        messages: [],
        tool_calls: [],
        token_usage: nil,
        metadata: %{model: "gpt-4", llm_module: ChatOpenAI}
      }

      restored = original |> Trajectory.to_map() |> Trajectory.from_map()

      assert restored.metadata == original.metadata
    end

    test "handles string-keyed metadata from JSON" do
      json_map = %{
        "messages" => [],
        "tool_calls" => [],
        "token_usage" => nil,
        "metadata" => %{"model" => "gpt-4", "custom" => "value"}
      }

      trajectory = Trajectory.from_map(json_map)

      assert trajectory.metadata == %{"model" => "gpt-4", "custom" => "value"}
    end

    test "defaults metadata to empty map when missing" do
      trajectory = Trajectory.from_map(%{})
      assert trajectory.metadata == %{}
    end
  end

  describe "matches?/3" do
    setup do
      actual_calls = [
        %{name: "search", arguments: %{"query" => "weather"}},
        %{name: "get_forecast", arguments: %{"city" => "Paris", "units" => "celsius"}}
      ]

      trajectory = %Trajectory{
        messages: [],
        tool_calls: actual_calls,
        token_usage: nil
      }

      %{trajectory: trajectory, actual_calls: actual_calls}
    end

    # Strict mode
    test "strict mode matches identical sequences", %{trajectory: trajectory, actual_calls: calls} do
      assert Trajectory.matches?(trajectory, calls)
    end

    test "strict mode rejects different order", %{trajectory: trajectory} do
      expected = [
        %{name: "get_forecast", arguments: %{"city" => "Paris", "units" => "celsius"}},
        %{name: "search", arguments: %{"query" => "weather"}}
      ]

      refute Trajectory.matches?(trajectory, expected)
    end

    test "strict mode rejects different count", %{trajectory: trajectory} do
      expected = [%{name: "search", arguments: %{"query" => "weather"}}]
      refute Trajectory.matches?(trajectory, expected)
    end

    test "strict mode rejects different arguments", %{trajectory: trajectory} do
      expected = [
        %{name: "search", arguments: %{"query" => "news"}},
        %{name: "get_forecast", arguments: %{"city" => "Paris", "units" => "celsius"}}
      ]

      refute Trajectory.matches?(trajectory, expected)
    end

    # Unordered mode
    test "unordered mode matches regardless of order", %{trajectory: trajectory} do
      expected = [
        %{name: "get_forecast", arguments: %{"city" => "Paris", "units" => "celsius"}},
        %{name: "search", arguments: %{"query" => "weather"}}
      ]

      assert Trajectory.matches?(trajectory, expected, mode: :unordered)
    end

    test "unordered mode rejects different count", %{trajectory: trajectory} do
      expected = [%{name: "search", arguments: %{"query" => "weather"}}]
      refute Trajectory.matches?(trajectory, expected, mode: :unordered)
    end

    # Superset mode
    test "superset mode matches when actual contains all expected", %{trajectory: trajectory} do
      expected = [%{name: "search", arguments: %{"query" => "weather"}}]
      assert Trajectory.matches?(trajectory, expected, mode: :superset)
    end

    test "superset mode rejects when expected has calls not in actual", %{trajectory: trajectory} do
      expected = [%{name: "missing_tool", arguments: nil}]
      refute Trajectory.matches?(trajectory, expected, mode: :superset)
    end

    # nil arguments = wildcard
    test "nil arguments matches any arguments", %{trajectory: trajectory} do
      expected = [
        %{name: "search", arguments: nil},
        %{name: "get_forecast", arguments: nil}
      ]

      assert Trajectory.matches?(trajectory, expected)
    end

    # Subset args mode
    test "subset args matches when expected args are subset of actual", %{trajectory: trajectory} do
      expected = [
        %{name: "search", arguments: %{"query" => "weather"}},
        %{name: "get_forecast", arguments: %{"city" => "Paris"}}
      ]

      assert Trajectory.matches?(trajectory, expected, args: :subset)
    end

    test "subset args rejects when expected has extra keys" do
      trajectory = %Trajectory{
        messages: [],
        tool_calls: [%{name: "search", arguments: %{"query" => "weather"}}],
        token_usage: nil
      }

      expected = [%{name: "search", arguments: %{"query" => "weather", "limit" => 10}}]
      refute Trajectory.matches?(trajectory, expected, args: :subset)
    end

    # Trajectory vs Trajectory
    test "matches two trajectories" do
      t1 = %Trajectory{
        messages: [],
        tool_calls: [%{name: "search", arguments: %{"q" => "test"}}],
        token_usage: nil
      }

      t2 = %Trajectory{
        messages: [],
        tool_calls: [%{name: "search", arguments: %{"q" => "test"}}],
        token_usage: nil
      }

      assert Trajectory.matches?(t1, t2)
    end

    # Empty cases
    test "empty actual matches empty expected" do
      trajectory = %Trajectory{messages: [], tool_calls: [], token_usage: nil}
      assert Trajectory.matches?(trajectory, [])
    end

    test "empty actual does not match non-empty expected" do
      trajectory = %Trajectory{messages: [], tool_calls: [], token_usage: nil}
      refute Trajectory.matches?(trajectory, [%{name: "search", arguments: nil}])
    end
  end

  describe "calls_by_name/2" do
    test "returns matching tool calls" do
      trajectory = %Trajectory{
        messages: [],
        tool_calls: [
          %{name: "search", arguments: %{"query" => "weather"}},
          %{name: "get_forecast", arguments: %{"city" => "Paris"}},
          %{name: "search", arguments: %{"query" => "news"}}
        ],
        token_usage: nil
      }

      result = Trajectory.calls_by_name(trajectory, "search")

      assert [
               %{name: "search", arguments: %{"query" => "weather"}},
               %{name: "search", arguments: %{"query" => "news"}}
             ] = result
    end

    test "returns empty list when no calls match" do
      trajectory = %Trajectory{
        messages: [],
        tool_calls: [%{name: "search", arguments: %{"q" => "test"}}],
        token_usage: nil
      }

      assert [] = Trajectory.calls_by_name(trajectory, "missing")
    end

    test "returns empty list for empty trajectory" do
      trajectory = %Trajectory{messages: [], tool_calls: [], token_usage: nil}
      assert [] = Trajectory.calls_by_name(trajectory, "search")
    end
  end

  describe "calls_by_turn/1" do
    test "groups tool calls by assistant message turn" do
      tc1 = make_tool_call("search", %{"query" => "weather"})
      tc2 = make_tool_call("get_forecast", %{"city" => "Paris"})
      tr1 = make_tool_result("search", "Sunny")

      messages = [
        user_msg("What's the weather in Paris?"),
        assistant_msg(nil, tool_calls: [tc1]),
        tool_msg([tr1]),
        assistant_msg(nil, tool_calls: [tc2])
      ]

      trajectory = Trajectory.from_chain(chain_with_messages(messages))
      turns = Trajectory.calls_by_turn(trajectory)

      assert [
               {0, [%{name: "search", arguments: %{"query" => "weather"}}]},
               {1, [%{name: "get_forecast", arguments: %{"city" => "Paris"}}]}
             ] = turns
    end

    test "groups parallel tool calls in the same turn" do
      tc1 = make_tool_call("search", %{"query" => "weather"}, "call_1")
      tc2 = make_tool_call("search", %{"query" => "news"}, "call_2")

      messages = [
        user_msg("Search for weather and news"),
        assistant_msg(nil, tool_calls: [tc1, tc2])
      ]

      trajectory = Trajectory.from_chain(chain_with_messages(messages))
      turns = Trajectory.calls_by_turn(trajectory)

      assert [
               {0,
                [
                  %{name: "search", arguments: %{"query" => "weather"}},
                  %{name: "search", arguments: %{"query" => "news"}}
                ]}
             ] = turns
    end

    test "returns empty list when no tool calls exist" do
      messages = [user_msg("Hello"), assistant_msg("Hi")]
      trajectory = Trajectory.from_chain(chain_with_messages(messages))

      assert [] = Trajectory.calls_by_turn(trajectory)
    end
  end

  describe "assert_trajectory/2,3" do
    test "passes when trajectory matches" do
      trajectory = %Trajectory{
        messages: [],
        tool_calls: [%{name: "search", arguments: %{"q" => "test"}}],
        token_usage: nil
      }

      assert_trajectory(trajectory, [%{name: "search", arguments: %{"q" => "test"}}])
    end

    test "passes with options" do
      trajectory = %Trajectory{
        messages: [],
        tool_calls: [
          %{name: "search", arguments: %{"q" => "test"}},
          %{name: "fetch", arguments: %{"url" => "example.com"}}
        ],
        token_usage: nil
      }

      assert_trajectory(trajectory, [%{name: "search", arguments: nil}], mode: :superset)
    end

    test "raises ExUnit.AssertionError on mismatch" do
      trajectory = %Trajectory{
        messages: [],
        tool_calls: [%{name: "search", arguments: %{"q" => "test"}}],
        token_usage: nil
      }

      assert_raise ExUnit.AssertionError, ~r/Trajectory mismatch/, fn ->
        assert_trajectory(trajectory, [%{name: "other_tool", arguments: nil}])
      end
    end

    test "accepts an LLMChain directly" do
      tc = make_tool_call("search", %{"q" => "test"})
      messages = [user_msg("Search"), assistant_msg(nil, tool_calls: [tc])]
      chain = chain_with_messages(messages)

      assert_trajectory(chain, [%{name: "search", arguments: %{"q" => "test"}}])
    end
  end

  describe "refute_trajectory/2,3" do
    test "passes when trajectory does not match" do
      trajectory = %Trajectory{
        messages: [],
        tool_calls: [%{name: "search", arguments: %{"q" => "test"}}],
        token_usage: nil
      }

      refute_trajectory(trajectory, [%{name: "delete_all", arguments: nil}])
    end

    test "passes with superset mode when tool not present" do
      trajectory = %Trajectory{
        messages: [],
        tool_calls: [
          %{name: "search", arguments: %{"q" => "test"}},
          %{name: "fetch", arguments: %{"url" => "example.com"}}
        ],
        token_usage: nil
      }

      refute_trajectory(trajectory, [%{name: "delete_all", arguments: nil}], mode: :superset)
    end

    test "raises ExUnit.AssertionError on unexpected match" do
      trajectory = %Trajectory{
        messages: [],
        tool_calls: [%{name: "search", arguments: %{"q" => "test"}}],
        token_usage: nil
      }

      assert_raise ExUnit.AssertionError, ~r/Unexpected trajectory match/, fn ->
        refute_trajectory(trajectory, [%{name: "search", arguments: %{"q" => "test"}}])
      end
    end

    test "raises on superset match" do
      trajectory = %Trajectory{
        messages: [],
        tool_calls: [
          %{name: "search", arguments: %{"q" => "test"}},
          %{name: "fetch", arguments: %{"url" => "example.com"}}
        ],
        token_usage: nil
      }

      assert_raise ExUnit.AssertionError, ~r/Unexpected trajectory match/, fn ->
        refute_trajectory(trajectory, [%{name: "search", arguments: nil}], mode: :superset)
      end
    end
  end
end
