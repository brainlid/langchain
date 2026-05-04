defmodule LangChain.Chains.LLMChain.Mode.StepsTest do
  use LangChain.BaseCase
  use Mimic

  alias LangChain.Chains.LLMChain
  alias LangChain.Chains.LLMChain.Mode.Steps
  alias LangChain.ChatModels.ChatOpenAI
  alias LangChain.Message
  alias LangChain.Message.ToolCall
  alias LangChain.Message.ToolResult
  alias LangChain.Function
  alias LangChain.LangChainError

  setup :verify_on_exit!

  setup do
    {:ok, chat} = ChatOpenAI.new(%{temperature: 0})
    chain = LLMChain.new!(%{llm: chat})
    %{chat: chat, chain: chain}
  end

  describe "call_llm/1" do
    test "calls LLM and returns {:continue, updated_chain}", %{chain: chain} do
      expect(ChatOpenAI, :call, fn _model, _messages, _tools ->
        {:ok, [Message.new_assistant!("Hello!")]}
      end)

      chain =
        chain
        |> LLMChain.add_message(Message.new_user!("Hi"))
        |> Steps.ensure_mode_state()

      assert {:continue, updated_chain} = Steps.call_llm({:continue, chain})
      assert updated_chain.last_message.role == :assistant
      assert Steps.get_run_count(updated_chain) == 1
    end

    test "increments run_count on each call", %{chain: chain} do
      expect(ChatOpenAI, :call, fn _model, _messages, _tools ->
        {:ok, [Message.new_assistant!("Hello!")]}
      end)

      chain =
        chain
        |> LLMChain.add_message(Message.new_user!("Hi"))
        |> Steps.ensure_mode_state()
        |> LLMChain.update_custom_context(%{mode_state: %{run_count: 3}})

      assert {:continue, updated_chain} = Steps.call_llm({:continue, chain})
      assert Steps.get_run_count(updated_chain) == 4
    end

    test "returns error on LLM failure", %{chain: chain} do
      chain =
        chain
        |> LLMChain.add_message(Message.new_user!("Hi"))
        |> Steps.ensure_mode_state()
        |> Map.put(:current_failure_count, 3)
        |> Map.put(:max_retry_count, 3)

      assert {:error, _chain, %LangChainError{type: "exceeded_failure_count"}} =
               Steps.call_llm({:continue, chain})
    end

    test "passes through terminal results", %{chain: chain} do
      error = {:error, chain, LangChainError.exception(message: "fail")}
      assert ^error = Steps.call_llm(error)

      ok = {:ok, chain}
      assert ^ok = Steps.call_llm(ok)
    end
  end

  describe "execute_tools/1" do
    test "executes pending tool calls", %{chain: chain} do
      hello_world =
        Function.new!(%{
          name: "hello_world",
          description: "Says hello",
          function: fn _args, _context -> "Hello world!" end
        })

      tool_call = ToolCall.new!(%{call_id: "call_1", name: "hello_world", arguments: %{}})

      expect(ChatOpenAI, :call, fn _model, _messages, _tools ->
        {:ok, [Message.new_assistant!(%{tool_calls: [tool_call]})]}
      end)

      chain =
        chain
        |> LLMChain.add_tools(hello_world)
        |> LLMChain.add_message(Message.new_user!("Hi"))

      {:ok, chain_with_tool_calls} = LLMChain.execute_step(chain)

      assert {:continue, updated_chain} = Steps.execute_tools({:continue, chain_with_tool_calls})
      assert updated_chain.last_message.role == :tool
    end

    test "no-op when no pending tool calls", %{chain: chain} do
      expect(ChatOpenAI, :call, fn _model, _messages, _tools ->
        {:ok, [Message.new_assistant!("Hello!")]}
      end)

      chain = LLMChain.add_message(chain, Message.new_user!("Hi"))
      {:ok, chain_with_response} = LLMChain.execute_step(chain)

      assert {:continue, ^chain_with_response} =
               Steps.execute_tools({:continue, chain_with_response})
    end

    test "passes through terminal results", %{chain: chain} do
      ok = {:ok, chain}
      assert ^ok = Steps.execute_tools(ok)
    end
  end

  describe "check_max_runs/2" do
    test "returns error when run_count exceeds max_runs", %{chain: chain} do
      chain =
        chain
        |> Steps.ensure_mode_state()
        |> LLMChain.update_custom_context(%{mode_state: %{run_count: 26}})

      assert {:error, _chain, %LangChainError{type: "exceeded_max_runs"}} =
               Steps.check_max_runs({:continue, chain}, max_runs: 25)
    end

    test "returns continue when under limit", %{chain: chain} do
      chain =
        chain
        |> Steps.ensure_mode_state()
        |> LLMChain.update_custom_context(%{mode_state: %{run_count: 5}})

      assert {:continue, ^chain} = Steps.check_max_runs({:continue, chain}, max_runs: 25)
    end

    test "uses default max_runs of 25", %{chain: chain} do
      chain =
        chain
        |> Steps.ensure_mode_state()
        |> LLMChain.update_custom_context(%{mode_state: %{run_count: 26}})

      assert {:error, _chain, %LangChainError{type: "exceeded_max_runs"}} =
               Steps.check_max_runs({:continue, chain}, [])
    end

    test "passes through terminal results", %{chain: chain} do
      ok = {:ok, chain}
      assert ^ok = Steps.check_max_runs(ok, max_runs: 25)
    end

    test "includes count and limit in error message", %{chain: chain} do
      chain =
        chain
        |> Steps.ensure_mode_state()
        |> LLMChain.update_custom_context(%{mode_state: %{run_count: 50}})

      assert {:error, _chain, %LangChainError{message: message}} =
               Steps.check_max_runs({:continue, chain}, max_runs: 50)

      assert message == "Exceeded maximum number of runs (50/50)"
    end
  end

  describe "check_pause/2" do
    test "returns pause when should_pause? returns true", %{chain: chain} do
      assert {:pause, ^chain} =
               Steps.check_pause({:continue, chain}, should_pause?: fn -> true end)
    end

    test "returns continue when should_pause? returns false", %{chain: chain} do
      assert {:continue, ^chain} =
               Steps.check_pause({:continue, chain}, should_pause?: fn -> false end)
    end

    test "returns continue when no should_pause? function", %{chain: chain} do
      assert {:continue, ^chain} = Steps.check_pause({:continue, chain}, [])
    end

    test "passes through terminal results", %{chain: chain} do
      ok = {:ok, chain}
      assert ^ok = Steps.check_pause(ok, should_pause?: fn -> true end)
    end
  end

  describe "check_until_tool/2" do
    test "returns ok with tool_result when matching tool found", %{chain: chain} do
      tool_result =
        ToolResult.new!(%{
          tool_call_id: "call_1",
          name: "submit",
          content: "submitted"
        })

      tool_message = %Message{role: :tool, tool_results: [tool_result]}

      chain = %{chain | last_message: tool_message}

      assert {:ok, ^chain, ^tool_result} =
               Steps.check_until_tool({:continue, chain}, tool_names: ["submit"])
    end

    test "returns continue when no matching tool found", %{chain: chain} do
      tool_result =
        ToolResult.new!(%{
          tool_call_id: "call_1",
          name: "other_tool",
          content: "result"
        })

      tool_message = %Message{role: :tool, tool_results: [tool_result]}

      chain = %{chain | last_message: tool_message}

      assert {:continue, ^chain} =
               Steps.check_until_tool({:continue, chain}, tool_names: ["submit"])
    end

    test "returns continue when no tool_names in opts", %{chain: chain} do
      assert {:continue, ^chain} = Steps.check_until_tool({:continue, chain}, [])
    end

    test "passes through terminal results", %{chain: chain} do
      ok = {:ok, chain}
      assert ^ok = Steps.check_until_tool(ok, tool_names: ["submit"])
    end
  end

  describe "continue_or_done/3" do
    test "loops when needs_response is true", %{chain: chain} do
      chain = %{chain | needs_response: true}
      run_fn = fn c, _opts -> {:ok, c} end

      assert {:ok, ^chain} = Steps.continue_or_done({:continue, chain}, run_fn, [])
    end

    test "returns ok when needs_response is false", %{chain: chain} do
      chain = %{chain | needs_response: false}
      run_fn = fn _c, _opts -> raise "should not be called" end

      assert {:ok, ^chain} = Steps.continue_or_done({:continue, chain}, run_fn, [])
    end

    test "passes through terminal results", %{chain: chain} do
      run_fn = fn _c, _opts -> raise "should not be called" end

      ok = {:ok, chain}
      assert ^ok = Steps.continue_or_done(ok, run_fn, [])

      pause = {:pause, chain}
      assert ^pause = Steps.continue_or_done(pause, run_fn, [])

      error = {:error, chain, LangChainError.exception(message: "fail")}
      assert ^error = Steps.continue_or_done(error, run_fn, [])

      ok_extra = {:ok, chain, :extra}
      assert ^ok_extra = Steps.continue_or_done(ok_extra, run_fn, [])
    end
  end

  describe "ensure_mode_state/1" do
    test "creates mode_state on first call", %{chain: chain} do
      updated = Steps.ensure_mode_state(chain)
      assert updated.custom_context.mode_state == %{run_count: 0}
    end

    test "preserves existing mode_state", %{chain: chain} do
      chain =
        chain
        |> Steps.ensure_mode_state()
        |> LLMChain.update_custom_context(%{mode_state: %{run_count: 5}})

      updated = Steps.ensure_mode_state(chain)
      assert updated.custom_context.mode_state == %{run_count: 5}
    end
  end

  describe "check_tool_interrupts/2" do
    test "returns continue when no tool messages", %{chain: chain} do
      assert {:continue, ^chain} = Steps.check_tool_interrupts({:continue, chain}, [])
    end

    test "returns continue when tool message has no interrupts", %{chain: chain} do
      tool_result =
        ToolResult.new!(%{tool_call_id: "call_1", name: "search", content: "found it"})

      tool_message = Message.new_tool_result!(%{content: nil, tool_results: [tool_result]})
      chain = LLMChain.add_message(chain, tool_message)

      assert {:continue, ^chain} = Steps.check_tool_interrupts({:continue, chain}, [])
    end

    test "returns interrupt when single tool result is interrupted", %{chain: chain} do
      interrupt_data = %{type: :subagent_hitl, sub_agent_id: "agent-1"}

      tool_result =
        ToolResult.new!(%{
          tool_call_id: "call_1",
          name: "task",
          content: "SubAgent requires approval.",
          is_interrupt: true,
          interrupt_data: interrupt_data
        })

      tool_message = Message.new_tool_result!(%{content: nil, tool_results: [tool_result]})
      chain = LLMChain.add_message(chain, tool_message)

      assert {:interrupt, ^chain, returned_data} =
               Steps.check_tool_interrupts({:continue, chain}, [])

      # For single interrupt, data is the interrupt_data with tool_call_id merged in
      assert returned_data.type == :subagent_hitl
      assert returned_data.sub_agent_id == "agent-1"
      assert returned_data.tool_call_id == "call_1"
    end

    test "returns multiple_interrupts when multiple results interrupted", %{chain: chain} do
      result1 =
        ToolResult.new!(%{
          tool_call_id: "call_1",
          name: "task",
          content: "Agent 1 interrupted",
          is_interrupt: true,
          interrupt_data: %{type: :subagent_hitl, sub_agent_id: "agent-1"}
        })

      result2 =
        ToolResult.new!(%{
          tool_call_id: "call_2",
          name: "task",
          content: "Agent 2 interrupted",
          is_interrupt: true,
          interrupt_data: %{type: :subagent_hitl, sub_agent_id: "agent-2"}
        })

      tool_message =
        Message.new_tool_result!(%{content: nil, tool_results: [result1, result2]})

      chain = LLMChain.add_message(chain, tool_message)

      assert {:interrupt, ^chain, data} =
               Steps.check_tool_interrupts({:continue, chain}, [])

      assert data.type == :multiple_interrupts
      assert length(data.interrupts) == 2
      assert Enum.at(data.interrupts, 0).tool_call_id == "call_1"
      assert Enum.at(data.interrupts, 1).tool_call_id == "call_2"
    end

    test "passes through terminal results", %{chain: chain} do
      ok = {:ok, chain}
      assert ^ok = Steps.check_tool_interrupts(ok, [])

      pause = {:pause, chain}
      assert ^pause = Steps.check_tool_interrupts(pause, [])
    end

    test "does not crash when single interrupted result has nil interrupt_data", %{chain: chain} do
      # Simulates a ToolResult restored from persistence: `interrupt_data` is a
      # virtual field, so it always comes back as nil. Without the guard,
      # extract_interrupt_data/1 would raise BadMapError on Map.put(nil, ...).
      tool_result =
        ToolResult.new!(%{
          tool_call_id: "call_1",
          name: "ask_user",
          content: "Waiting for user response...",
          is_interrupt: true,
          interrupt_data: nil
        })

      tool_message = Message.new_tool_result!(%{content: nil, tool_results: [tool_result]})
      chain = LLMChain.add_message(chain, tool_message)

      assert {:interrupt, ^chain, returned_data} =
               Steps.check_tool_interrupts({:continue, chain}, [])

      assert returned_data == %{tool_call_id: "call_1"}
    end

    test "does not crash when multiple interrupted results have nil interrupt_data", %{
      chain: chain
    } do
      result1 =
        ToolResult.new!(%{
          tool_call_id: "call_1",
          name: "ask_user",
          content: "Interrupted",
          is_interrupt: true,
          interrupt_data: nil
        })

      result2 =
        ToolResult.new!(%{
          tool_call_id: "call_2",
          name: "ask_user",
          content: "Interrupted",
          is_interrupt: true,
          interrupt_data: nil
        })

      tool_message =
        Message.new_tool_result!(%{content: nil, tool_results: [result1, result2]})

      chain = LLMChain.add_message(chain, tool_message)

      assert {:interrupt, ^chain, data} =
               Steps.check_tool_interrupts({:continue, chain}, [])

      assert data.type == :multiple_interrupts
      assert Enum.at(data.interrupts, 0) == %{tool_call_id: "call_1"}
      assert Enum.at(data.interrupts, 1) == %{tool_call_id: "call_2"}
    end
  end

  describe "end-to-end pipeline" do
    test "compose steps into a mini-mode", %{chain: chain} do
      # First call: LLM returns tool call
      tool_call = ToolCall.new!(%{call_id: "call_1", name: "hello_world", arguments: %{}})

      expect(ChatOpenAI, :call, fn _model, _messages, _tools ->
        {:ok, [Message.new_assistant!(%{tool_calls: [tool_call]})]}
      end)

      # Second call: LLM returns final response
      expect(ChatOpenAI, :call, fn _model, _messages, _tools ->
        {:ok, [Message.new_assistant!("Done!")]}
      end)

      hello_world =
        Function.new!(%{
          name: "hello_world",
          description: "Says hello",
          function: fn _args, _context -> "Hello world!" end
        })

      chain =
        chain
        |> LLMChain.add_tools(hello_world)
        |> LLMChain.add_message(Message.new_user!("Hi"))

      chain = Steps.ensure_mode_state(chain)

      # First iteration: call LLM (returns tool call), execute tools, loop
      assert {:ok, final_chain} =
               {:continue, chain}
               |> Steps.call_llm()
               |> Steps.execute_tools()
               |> Steps.check_max_runs(max_runs: 25)
               |> Steps.continue_or_done(
                 fn c, o ->
                   # Second iteration: call LLM (returns assistant msg), done
                   {:continue, c}
                   |> Steps.call_llm()
                   |> Steps.execute_tools()
                   |> Steps.check_max_runs(o)
                   |> Steps.continue_or_done(fn _, _ -> raise "too many loops" end, o)
                 end,
                 max_runs: 25
               )

      assert final_chain.last_message.role == :assistant
      assert Steps.get_run_count(final_chain) == 2
    end
  end
end
