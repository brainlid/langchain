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
