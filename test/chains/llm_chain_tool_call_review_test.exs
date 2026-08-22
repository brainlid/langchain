defmodule LangChain.Chains.LLMChainToolCallReviewTest do
  use ExUnit.Case

  alias LangChain.Chains.LLMChain
  alias LangChain.{Function, LangChainError, Message}
  alias LangChain.Message.{ContentPart, ToolCall}
  alias LangChain.ChatModels.ChatAnthropic

  defp tool(test_pid, opts \\ []) do
    Function.new!(%{
      name: Keyword.get(opts, :name, "ship_order"),
      description: "Ships an order",
      display_text: "Shipping the order",
      async: Keyword.get(opts, :async, false),
      parameters_schema: %{type: "object", properties: %{}},
      function: fn args, _ctx ->
        send(test_pid, {:tool_ran, args})
        {:ok, "shipped"}
      end
    })
  end

  defp chain_with(tools, callbacks) do
    LLMChain.new!(%{
      llm: ChatAnthropic.new!(%{model: "claude-sonnet-4-5-20250929"}),
      tools: List.wrap(tools),
      callbacks: callbacks
    })
  end

  defp with_call(chain, arguments \\ %{"amount" => 5000}, name \\ "ship_order") do
    LLMChain.add_message(
      chain,
      Message.new_assistant!(%{
        content: "calling",
        tool_calls: [ToolCall.new!(%{call_id: "call_1", name: name, arguments: arguments})]
      })
    )
  end

  defp only_result(%LLMChain{} = chain) do
    [result] = chain.last_message.tool_results
    result
  end

  defp text(%{content: [%ContentPart{content: text}]}), do: text

  describe "review decisions" do
    test ":ok lets the call through untouched" do
      chain =
        [tool(self())]
        |> chain_with([%{on_tool_call_review: fn _chain, _call, _func -> :ok end}])
        |> with_call()

      result = LLMChain.execute_tool_calls(chain) |> only_result()

      assert_received {:tool_ran, %{"amount" => 5000}}
      assert text(result) == "shipped"
      refute result.is_error
    end

    test "a chain with no review handler behaves as before" do
      chain = [tool(self())] |> chain_with([]) |> with_call()

      result = LLMChain.execute_tool_calls(chain) |> only_result()

      assert_received {:tool_ran, %{"amount" => 5000}}
      assert text(result) == "shipped"
    end

    test "{:deny, reason} keeps the tool from running and returns the reason" do
      handler = %{
        on_tool_call_review: fn _chain, _call, _func -> {:deny, "over the store limit"} end
      }

      chain = [tool(self())] |> chain_with([handler]) |> with_call()

      result = LLMChain.execute_tool_calls(chain) |> only_result()

      refute_received {:tool_ran, _}
      assert text(result) == "over the store limit"
      assert result.tool_call_id == "call_1"
      assert result.name == "ship_order"
      assert result.display_text == "Shipping the order"
    end

    test "a denial is not an error and leaves the failure counter alone" do
      handler = %{on_tool_call_review: fn _chain, _call, _func -> {:deny, "no"} end}

      chain =
        [tool(self())]
        |> chain_with([handler])
        |> LLMChain.increment_current_failure_count()
        |> with_call()

      assert chain.current_failure_count == 1

      updated = LLMChain.execute_tool_calls(chain)

      refute only_result(updated).is_error
      assert updated.current_failure_count == 0
    end

    test "{:update_arguments, args} rewrites what the tool receives" do
      handler = %{
        on_tool_call_review: fn _chain, call, _func ->
          {:update_arguments, Map.put(call.arguments, "warehouse", "EU-1")}
        end
      }

      chain = [tool(self())] |> chain_with([handler]) |> with_call()

      LLMChain.execute_tool_calls(chain)

      assert_received {:tool_ran, %{"amount" => 5000, "warehouse" => "EU-1"}}
    end

    test "{:interrupt, message, data} settles the call as an interrupt" do
      handler = %{
        on_tool_call_review: fn _chain, _call, _func ->
          {:interrupt, "needs a manager", %{approver: "manager"}}
        end
      }

      chain = [tool(self())] |> chain_with([handler]) |> with_call()

      result = LLMChain.execute_tool_calls(chain) |> only_result()

      refute_received {:tool_ran, _}
      assert result.is_interrupt
      assert result.interrupt_data == %{approver: "manager"}
      assert text(result) == "needs a manager"
    end

    test "an unrecognized return value raises with the tool named" do
      handler = %{on_tool_call_review: fn _chain, _call, _func -> :maybe end}

      chain = [tool(self())] |> chain_with([handler]) |> with_call()

      assert_raise LangChainError,
                   ~r/Unexpected :on_tool_call_review result for tool "ship_order"/,
                   fn ->
                     LLMChain.execute_tool_calls(chain)
                   end
    end
  end

  describe "review runs before anything is announced" do
    test "a denied call never fires on_tool_execution_started" do
      test_pid = self()

      handlers = [
        %{
          on_tool_call_review: fn _chain, _call, _func -> {:deny, "no"} end,
          on_tool_execution_started: fn _chain, call, _func ->
            send(test_pid, {:started, call.name})
          end,
          on_tool_execution_failed: fn _chain, call, reason ->
            send(test_pid, {:failed, call.name, reason})
          end
        }
      ]

      chain = [tool(self())] |> chain_with(handlers) |> with_call()

      LLMChain.execute_tool_calls(chain)

      refute_received {:started, _}
      assert_received {:failed, "ship_order", "no"}
    end

    test "an async tool is denied without spawning its Task" do
      handler = %{on_tool_call_review: fn _chain, _call, _func -> {:deny, "no"} end}

      chain =
        [tool(self(), async: true)]
        |> chain_with([handler])
        |> with_call()

      result = LLMChain.execute_tool_calls(chain) |> only_result()

      refute_received {:tool_ran, _}
      assert text(result) == "no"
    end

    test "on_tool_execution_started still fires for a call review allows" do
      test_pid = self()

      handlers = [
        %{
          on_tool_call_review: fn _chain, _call, _func -> :ok end,
          on_tool_execution_started: fn _chain, call, _func ->
            send(test_pid, {:started, call.name})
          end
        }
      ]

      chain = [tool(self())] |> chain_with(handlers) |> with_call()

      LLMChain.execute_tool_calls(chain)

      assert_received {:started, "ship_order"}
    end
  end

  describe "several handlers" do
    test "argument rewrites thread forward into the next handler" do
      test_pid = self()

      first = %{
        on_tool_call_review: fn _chain, call, _func ->
          {:update_arguments, Map.put(call.arguments, "step", ["first"])}
        end
      }

      second = %{
        on_tool_call_review: fn _chain, call, _func ->
          send(test_pid, {:second_saw, call.arguments})
          {:update_arguments, Map.update!(call.arguments, "step", &(&1 ++ ["second"]))}
        end
      }

      chain = [tool(self())] |> chain_with([first, second]) |> with_call()

      LLMChain.execute_tool_calls(chain)

      assert_received {:second_saw, %{"step" => ["first"]}}
      assert_received {:tool_ran, %{"step" => ["first", "second"]}}
    end

    test "the first denial settles the call and later handlers are skipped" do
      test_pid = self()

      first = %{on_tool_call_review: fn _chain, _call, _func -> {:deny, "first says no"} end}

      second = %{
        on_tool_call_review: fn _chain, _call, _func ->
          send(test_pid, :second_consulted)
          :ok
        end
      }

      chain = [tool(self())] |> chain_with([first, second]) |> with_call()

      result = LLMChain.execute_tool_calls(chain) |> only_result()

      refute_received :second_consulted
      assert text(result) == "first says no"
    end
  end

  describe "a batch of calls" do
    test "denying one call leaves the others running, one result per call" do
      test_pid = self()

      keep =
        Function.new!(%{
          name: "keep",
          description: "runs",
          parameters_schema: %{type: "object", properties: %{}},
          function: fn _args, _ctx ->
            send(test_pid, :keep_ran)
            {:ok, "kept"}
          end
        })

      handler = %{
        on_tool_call_review: fn _chain, call, _func ->
          if call.name == "ship_order", do: {:deny, "no shipping"}, else: :ok
        end
      }

      message =
        Message.new_assistant!(%{
          content: "calling",
          tool_calls: [
            ToolCall.new!(%{call_id: "call_1", name: "ship_order", arguments: %{}}),
            ToolCall.new!(%{call_id: "call_2", name: "keep", arguments: %{}})
          ]
        })

      chain =
        [tool(self()), keep]
        |> chain_with([handler])
        |> LLMChain.add_message(message)

      results = LLMChain.execute_tool_calls(chain).last_message.tool_results

      assert_received :keep_ran
      refute_received {:tool_ran, _}

      assert length(results) == 2
      assert MapSet.new(results, & &1.tool_call_id) == MapSet.new(["call_1", "call_2"])

      denied = Enum.find(results, &(&1.tool_call_id == "call_1"))
      assert text(denied) == "no shipping"
    end
  end

  describe "the human-decision path" do
    defp decided(chain, decisions) do
      tool_calls = chain.last_message.tool_calls
      LLMChain.execute_tool_calls_with_decisions(chain, tool_calls, decisions)
    end

    test "an approved call is still reviewed and can be denied" do
      handler = %{
        on_tool_call_review: fn _chain, _call, _func -> {:deny, "store policy still says no"} end
      }

      chain = [tool(self())] |> chain_with([handler]) |> with_call()

      result = decided(chain, [%{type: :approve}]) |> only_result()

      refute_received {:tool_ran, _}
      assert text(result) == "store policy still says no"
    end

    test "an edited call is reviewed with the edited arguments" do
      test_pid = self()

      handler = %{
        on_tool_call_review: fn _chain, call, _func ->
          send(test_pid, {:reviewed, call.arguments})
          :ok
        end
      }

      chain = [tool(self())] |> chain_with([handler]) |> with_call()

      decided(chain, [%{type: :edit, arguments: %{"amount" => 10}}])

      assert_received {:reviewed, %{"amount" => 10}}
      assert_received {:tool_ran, %{"amount" => 10}}
    end

    test "review can rewrite arguments on an approved call" do
      handler = %{
        on_tool_call_review: fn _chain, call, _func ->
          {:update_arguments, Map.put(call.arguments, "warehouse", "EU-1")}
        end
      }

      chain = [tool(self())] |> chain_with([handler]) |> with_call()

      decided(chain, [%{type: :approve}])

      assert_received {:tool_ran, %{"amount" => 5000, "warehouse" => "EU-1"}}
    end

    test "an unknown tool still reports not found" do
      chain = [tool(self())] |> chain_with([]) |> with_call(%{}, "ship_order")

      unknown = %{(chain.last_message.tool_calls |> hd()) | name: "nope"}
      updated = LLMChain.execute_tool_calls_with_decisions(chain, [unknown], [%{type: :approve}])

      result = only_result(updated)
      assert result.is_error
      assert text(result) =~ "not found"
    end

    test "a rejected call is unchanged by review" do
      chain = [tool(self())] |> chain_with([]) |> with_call()

      result = decided(chain, [%{type: :reject}]) |> only_result()

      refute_received {:tool_ran, _}
      assert text(result) =~ "rejected by a human reviewer"
    end
  end
end
