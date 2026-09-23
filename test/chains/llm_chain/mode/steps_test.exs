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
  alias LangChain.Message.ContentPart
  alias LangChain.MessageExpansion

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

  describe "reset_run_count/1" do
    test "creates mode_state when absent", %{chain: chain} do
      assert Steps.reset_run_count(chain).custom_context.mode_state == %{run_count: 0}
    end

    test "zeroes the count and keeps other mode_state keys", %{chain: chain} do
      chain = LLMChain.update_custom_context(chain, %{mode_state: %{run_count: 7, other: :kept}})

      assert Steps.reset_run_count(chain).custom_context.mode_state == %{
               run_count: 0,
               other: :kept
             }
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

  describe "expand_tool_results/2" do
    setup %{chain: chain} do
      tool_call = ToolCall.new!(%{call_id: "call_1", name: "load_reference", arguments: %{}})

      staged =
        chain
        |> LLMChain.add_message(Message.new_user!("What does the policy say?"))
        |> LLMChain.add_message(Message.new_assistant!(%{tool_calls: [tool_call]}))

      %{staged: staged}
    end

    test "inserts the expansion's messages before the model is called", %{staged: staged} do
      chain = add_tool_results(staged, [expanding_result("call_1", "THE MATERIAL")])

      assert {:continue, expanded} = Steps.expand_tool_results({:continue, chain})

      assert [
               %Message{role: :user},
               %Message{role: :assistant, tool_calls: [_]},
               %Message{role: :tool},
               %Message{role: :assistant} = material,
               %Message{role: :user} = anchor
             ] = expanded.messages

      assert [%ContentPart{content: "THE MATERIAL"}] = material.content
      assert [%ContentPart{content: anchor_text}] = anchor.content
      assert is_binary(anchor_text)
    end

    test "inserts exactly the messages the tool chose", %{staged: staged} do
      chain =
        add_tool_results(staged, [
          expanding_result("call_1", "THE MATERIAL",
            messages: [Message.new_user!("THE MATERIAL")]
          )
        ])

      assert {:continue, expanded} = Steps.expand_tool_results({:continue, chain})

      assert [_user, _assistant, _tool, %Message{role: :user} = material] = expanded.messages
      assert [%ContentPart{content: "THE MATERIAL"}] = material.content
    end

    test "inserts a longer sequence unchanged", %{staged: staged} do
      messages = [
        Message.new_assistant!("one"),
        Message.new_user!("two"),
        Message.new_assistant!("three"),
        Message.new_user!("four")
      ]

      chain =
        add_tool_results(staged, [
          expanding_result("call_1", "THE MATERIAL", messages: messages)
        ])

      assert {:continue, expanded} = Steps.expand_tool_results({:continue, chain})

      assert Enum.drop(expanded.messages, 3) == messages
    end

    test "replaces the result content with what the expansion keeps", %{staged: staged} do
      result = expanding_result("call_1", "THE MATERIAL", result_content: "Loaded 1 document.")
      chain = add_tool_results(staged, [result])

      # Fail-open: before the step runs the result carries the whole payload.
      assert [%ContentPart{content: before_text}] = hd(chain.last_message.tool_results).content
      assert before_text =~ "THE MATERIAL"

      assert {:continue, expanded} = Steps.expand_tool_results({:continue, chain})

      assert [_user, _assistant, tool_message | _inserted] = expanded.messages

      assert [%ToolResult{content: [%ContentPart{content: "Loaded 1 document."}]}] =
               tool_message.tool_results
    end

    test "keeps messages, exchanged_messages and last_message in agreement", %{staged: staged} do
      chain = add_tool_results(staged, [expanding_result("call_1", "THE MATERIAL")])

      assert {:continue, expanded} = Steps.expand_tool_results({:continue, chain})

      assert expanded.last_message == List.last(expanded.messages)
      assert expanded.last_message.role == :user
      assert expanded.needs_response

      trimmed = Enum.find(expanded.messages, &(&1.role == :tool))
      assert trimmed in expanded.exchanged_messages
      refute Enum.any?(expanded.exchanged_messages, &(&1.role == :tool and &1 != trimmed))
    end

    test "applies when the tool message is absent from exchanged_messages", %{staged: staged} do
      # The shape a resume after human approval produces: the chain is rebuilt
      # from stored messages, so exchanged_messages starts empty.
      staged = add_tool_results(staged, [expanding_result("call_1", "THE MATERIAL")])
      chain = %LLMChain{staged | exchanged_messages: []}

      assert {:continue, expanded} = Steps.expand_tool_results({:continue, chain})

      assert %Message{role: :user} = expanded.last_message

      assert Enum.any?(
               expanded.messages,
               &(&1.role == :assistant and &1.tool_calls in [nil, []])
             )

      # Only the inserted messages were exchanged.
      assert [%Message{role: :assistant}, %Message{role: :user}] = expanded.exchanged_messages
    end

    test "is idempotent", %{staged: staged} do
      chain = add_tool_results(staged, [expanding_result("call_1", "THE MATERIAL")])

      assert {:continue, once} = Steps.expand_tool_results({:continue, chain})
      assert {:continue, twice} = Steps.expand_tool_results({:continue, once})

      assert once.messages == twice.messages
      assert once == twice
    end

    test "inserts in tool-call order when several tools deliver", %{staged: staged} do
      results = [
        expanding_result("call_1", "FIRST", result_content: "1"),
        expanding_result("call_2", "SECOND", result_content: "2")
      ]

      chain = add_tool_results(staged, results)

      assert {:continue, expanded} = Steps.expand_tool_results({:continue, chain})

      inserted =
        expanded.messages
        |> Enum.drop(3)
        |> Enum.map(fn %Message{content: [%ContentPart{content: text}]} -> text end)

      assert ["FIRST", _anchor, "SECOND", _anchor2] = inserted
    end

    test "ignores an interrupted result", %{staged: staged} do
      {:ok, result} =
        MessageExpansion.expand("THE MATERIAL", [Message.new_assistant!("THE MATERIAL")])

      interrupted = %ToolResult{
        result
        | tool_call_id: "call_1",
          name: "load_reference",
          is_interrupt: true,
          interrupt_data: %{type: :halt}
      }

      chain = add_tool_results(staged, [interrupted])

      assert {:continue, ^chain} = Steps.expand_tool_results({:continue, chain})
    end

    test "ignores a result with no expansion", %{staged: staged} do
      result = ToolResult.new!(%{tool_call_id: "call_1", name: "search", content: "found it"})
      chain = add_tool_results(staged, [result])

      assert {:continue, ^chain} = Steps.expand_tool_results({:continue, chain})
    end

    test "passes through when the last message is not a tool message", %{staged: staged} do
      chain = LLMChain.add_message(staged, Message.new_assistant!("done"))

      assert {:continue, ^chain} = Steps.expand_tool_results({:continue, chain})
    end

    test "passes terminals through untouched", %{staged: staged} do
      chain = add_tool_results(staged, [expanding_result("call_1", "THE MATERIAL")])

      assert {:ok, ^chain} = Steps.expand_tool_results({:ok, chain})
      assert {:ok, ^chain, :extra} = Steps.expand_tool_results({:ok, chain, :extra})
      assert {:pause, ^chain} = Steps.expand_tool_results({:pause, chain})
      assert {:interrupt, ^chain, :data} = Steps.expand_tool_results({:interrupt, chain, :data})
      assert {:error, ^chain, :why} = Steps.expand_tool_results({:error, chain, :why})
    end

    test "a terminal check still sees the tool message as last_message", %{staged: staged} do
      # The invariant the placement exists to preserve: until-tool termination
      # reads last_message, and must find what the tools returned.
      chain =
        add_tool_results(staged, [
          expanding_result("call_1", "THE MATERIAL", result_content: "Loaded.")
        ])

      assert {:ok, ^chain, %ToolResult{name: "load_reference"}} =
               Steps.check_until_tool({:continue, chain}, tool_names: ["load_reference"])
    end

    defp add_tool_results(chain, results) do
      LLMChain.add_message(
        chain,
        Message.new_tool_result!(%{content: nil, tool_results: results})
      )
    end

    # An expansion in the canonical shape: the material established as something
    # the model said, then a short user turn for it to answer.
    defp expanding_result(call_id, material, opts \\ []) do
      messages =
        Keyword.get(opts, :messages, [
          Message.new_assistant!(material),
          Message.new_user!("Use the content above to continue.")
        ])

      {:ok, result} =
        MessageExpansion.expand(material, messages, Keyword.take(opts, [:result_content]))

      %ToolResult{result | tool_call_id: call_id, name: "load_reference"}
    end
  end
end
