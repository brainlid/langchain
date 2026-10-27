defmodule LangChain.Chains.LLMChainRefusalTest do
  use ExUnit.Case, async: true
  use Mimic

  alias LangChain.Chains.LLMChain
  alias LangChain.{Function, FunctionParam, LangChainError, Message}
  alias LangChain.Message.{ToolCall, ToolResult}
  alias LangChain.ChatModels.{ChatAnthropic, ChatOpenAI}

  defp tool(attrs) do
    %{name: "edit", function: fn _args, _ctx -> {:ok, "edited"} end}
    |> Map.merge(attrs)
    |> Function.new!()
  end

  defp new_chain(tools) do
    LLMChain.new!(%{
      llm: ChatAnthropic.new!(%{model: "claude-sonnet-4-5-20250929"}),
      tools: List.wrap(tools)
    })
  end

  defp with_call(chain, arguments \\ %{}, name \\ "edit") do
    LLMChain.add_message(
      chain,
      Message.new_assistant!(%{
        tool_calls: [ToolCall.new!(%{call_id: "c1", name: name, arguments: arguments})]
      })
    )
  end

  defp run(chain), do: LLMChain.execute_tool_calls(chain)

  defp decide(chain, decision) do
    LLMChain.execute_tool_calls_with_decisions(chain, chain.last_message.tool_calls, [decision])
  end

  defp only_result(%LLMChain{last_message: %Message{tool_results: [result]}}), do: result

  describe "marking refusals" do
    test "a missing required parameter is a refusal" do
      result =
        tool(%{parameters: [FunctionParam.new!(%{name: "path", type: :string, required: true})]})
        |> new_chain()
        |> with_call()
        |> run()
        |> only_result()

      assert %ToolResult{is_error: true, is_refusal: true, is_exception: false} = result
    end

    test "a :parse_args error is a refusal" do
      result =
        tool(%{parse_args: fn _args -> {:error, "document_path is required."} end})
        |> new_chain()
        |> with_call()
        |> run()
        |> only_result()

      assert %ToolResult{is_error: true, is_refusal: true} = result
    end

    test "a :parse_args parser that raises is not a refusal" do
      result =
        tool(%{parse_args: fn _args -> raise "parser bug" end})
        |> new_chain()
        |> with_call()
        |> run()
        |> only_result()

      assert %ToolResult{is_error: true, is_exception: true, is_refusal: false} = result
    end

    test "an error returned by the tool body is not a refusal" do
      result =
        tool(%{function: fn _args, _ctx -> {:error, "disk full"} end})
        |> new_chain()
        |> with_call()
        |> run()
        |> only_result()

      assert %ToolResult{is_error: true, is_refusal: false} = result
    end

    test "a refusal the tool body returns is kept, and is always an error" do
      body = fn _args, _ctx ->
        {:ok, %ToolResult{is_refusal: true, is_error: false, content: "nothing changed"}}
      end

      result =
        tool(%{function: body})
        |> new_chain()
        |> with_call()
        |> run()
        |> only_result()

      assert %ToolResult{is_error: true, is_refusal: true} = result
    end

    test "a call to an unknown tool is a refusal" do
      result =
        tool(%{})
        |> new_chain()
        |> with_call(%{}, "missing")
        |> run()
        |> only_result()

      assert %ToolResult{is_error: true, is_refusal: true} = result
    end

    test "an approved call to an unknown tool is a refusal" do
      result =
        tool(%{})
        |> new_chain()
        |> with_call(%{}, "missing")
        |> decide(%{type: :approve})
        |> only_result()

      assert %ToolResult{is_error: true, is_refusal: true} = result
    end

    test "a rejection reported as an error is a refusal" do
      result =
        tool(%{})
        |> new_chain()
        |> with_call()
        |> decide(%{type: :reject, message: "bad arguments", is_error: true})
        |> only_result()

      assert %ToolResult{is_error: true, is_refusal: true} = result
    end

    test "a plain rejection is neither an error nor a refusal" do
      result =
        tool(%{})
        |> new_chain()
        |> with_call()
        |> decide(%{type: :reject})
        |> only_result()

      assert %ToolResult{is_error: false, is_refusal: false} = result
    end

    test "a refusal round still counts as a failure" do
      chain =
        tool(%{parse_args: fn _args -> {:error, "no"} end})
        |> new_chain()
        |> with_call()
        |> run()

      assert chain.current_failure_count == 1
    end
  end

  describe "the refusal budget" do
    defp refusing_tool, do: tool(%{parse_args: fn _args -> {:error, "path is required."} end})

    defp failing_tool,
      do: tool(%{name: "write", function: fn _args, _ctx -> {:error, "disk full"} end})

    defp ok_tool, do: tool(%{name: "read", function: fn _args, _ctx -> {:ok, "fine"} end})

    defp calls(chain, names) do
      LLMChain.add_message(
        chain,
        Message.new_assistant!(%{
          tool_calls:
            names
            |> Enum.with_index()
            |> Enum.map(fn {name, i} ->
              ToolCall.new!(%{call_id: "c#{i}", name: name, arguments: %{}})
            end)
        })
      )
    end

    defp round(chain, names), do: chain |> calls(names) |> run()

    defp budget_chain(attrs) do
      %{
        llm: ChatOpenAI.new!(%{stream: false}),
        tools: [refusing_tool(), failing_tool(), ok_tool()]
      }
      |> Map.merge(attrs)
      |> LLMChain.new!()
    end

    # The model asks for the refused call every time it is called.
    defp stub_model_calling(name) do
      stub(ChatOpenAI, :call, fn _model, _messages, _tools ->
        {:ok,
         [
           Message.new_assistant!(%{
             tool_calls: [ToolCall.new!(%{call_id: "call", name: name, arguments: %{}})]
           })
         ]}
      end)
    end

    defp notify_retries_exceeded do
      test_pid = self()
      %{on_retries_exceeded: fn chain -> send(test_pid, {:retries_exceeded, chain}) end}
    end

    test "max_refusal_count defaults to nil and must be positive when set" do
      assert %LLMChain{max_refusal_count: nil, current_refusal_count: 0} = budget_chain(%{})
      assert {:error, changeset} = LLMChain.new(%{llm: ChatOpenAI.new!(), max_refusal_count: 0})
      assert changeset.errors[:max_refusal_count]
    end

    test "without a refusal limit, refusal rounds stop the chain at max_retry_count" do
      stub_model_calling("edit")

      assert {:error, chain, %LangChainError{} = error} =
               budget_chain(%{max_retry_count: 3})
               |> LLMChain.add_message(Message.new_user!("go"))
               |> LLMChain.run(mode: :while_needs_response)

      assert error.type == "exceeded_failure_count"
      assert error.message == "Exceeded max failure count"
      assert chain.current_failure_count == 3
      assert chain.current_refusal_count == 0
    end

    test "with a refusal limit, refusal rounds spend it instead of max_retry_count" do
      stub_model_calling("edit")

      assert {:error, chain, %LangChainError{} = error} =
               budget_chain(%{
                 max_retry_count: 3,
                 max_refusal_count: 6,
                 callbacks: [notify_retries_exceeded()]
               })
               |> LLMChain.add_message(Message.new_user!("go"))
               |> LLMChain.run(mode: :while_needs_response)

      assert error.type == "exceeded_failure_count"
      assert error.message == "Exceeded max refusal count"
      assert chain.current_refusal_count == 6
      assert chain.current_failure_count == 0
      assert_received {:retries_exceeded, %LLMChain{}}
    end

    test "a round with a refusal and a body error counts as a failure" do
      chain = budget_chain(%{max_refusal_count: 6}) |> round(["edit", "write"])

      assert chain.current_failure_count == 1
      assert chain.current_refusal_count == 0
    end

    test "each count leaves the other alone" do
      chain =
        budget_chain(%{max_refusal_count: 6})
        |> round(["write"])
        |> round(["edit"])
        |> round(["write"])

      assert chain.current_failure_count == 2
      assert chain.current_refusal_count == 1
    end

    test "a clean round resets both counts" do
      chain =
        budget_chain(%{max_refusal_count: 6})
        |> round(["write"])
        |> round(["edit"])
        |> round(["read"])

      assert chain.current_failure_count == 0
      assert chain.current_refusal_count == 0
    end

    test "an assistant message that is not tool related resets both counts" do
      chain =
        budget_chain(%{max_refusal_count: 6})
        |> round(["write"])
        |> round(["edit"])
        |> LLMChain.process_message(Message.new_assistant!(%{content: "Done."}))

      assert chain.current_failure_count == 0
      assert chain.current_refusal_count == 0
    end

    test "until_success stops on the refusal limit and fires :on_retries_exceeded" do
      stub_model_calling("edit")

      assert {:error, _chain, %LangChainError{message: "Exceeded max refusal count"}} =
               budget_chain(%{max_refusal_count: 2, callbacks: [notify_retries_exceeded()]})
               |> LLMChain.add_message(Message.new_user!("go"))
               |> LLMChain.run(mode: :until_success)

      assert_received {:retries_exceeded, %LLMChain{current_refusal_count: 2}}
    end

    test "until_success fires :on_retries_exceeded on the failure limit" do
      stub_model_calling("write")

      assert {:error, _chain, %LangChainError{message: "Exceeded max failure count"}} =
               budget_chain(%{max_retry_count: 2, callbacks: [notify_retries_exceeded()]})
               |> LLMChain.add_message(Message.new_user!("go"))
               |> LLMChain.run(mode: :until_success)

      assert_received {:retries_exceeded, %LLMChain{current_failure_count: 2}}
    end
  end
end
