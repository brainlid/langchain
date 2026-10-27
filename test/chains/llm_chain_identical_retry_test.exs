defmodule LangChain.Chains.LLMChainIdenticalRetryTest do
  use ExUnit.Case, async: true

  alias LangChain.Chains.LLMChain
  alias LangChain.{Function, Message}
  alias LangChain.Message.{ContentPart, ToolCall, ToolResult}
  alias LangChain.ChatModels.ChatAnthropic

  @notice "This is the same call that just failed, with the same arguments. " <>
            "Change it before trying again."

  # Fails for path "bad", succeeds otherwise.
  defp edit_tool(opts \\ []) do
    Function.new!(%{
      name: "edit",
      async: Keyword.get(opts, :async, false),
      parameters_schema: %{type: "object", properties: %{}},
      function: fn
        %{"path" => "bad"}, _ctx -> {:error, "path not found"}
        _args, _ctx -> {:ok, "edited"}
      end
    })
  end

  # Returns the next outcome from `outcomes` on each call.
  defp scripted_tool(outcomes) do
    {:ok, agent} = Agent.start_link(fn -> outcomes end)

    Function.new!(%{
      name: "edit",
      parameters_schema: %{type: "object", properties: %{}},
      function: fn _args, _ctx ->
        Agent.get_and_update(agent, fn [next | rest] -> {next, rest} end)
      end
    })
  end

  defp new_chain(tools, callbacks \\ []) do
    LLMChain.new!(%{
      llm: ChatAnthropic.new!(%{model: "claude-sonnet-4-5-20250929"}),
      tools: List.wrap(tools),
      callbacks: callbacks
    })
  end

  defp call(chain, call_id, arguments, name \\ "edit") do
    LLMChain.add_message(
      chain,
      Message.new_assistant!(%{
        tool_calls: [ToolCall.new!(%{call_id: call_id, name: name, arguments: arguments})]
      })
    )
  end

  defp round(chain, call_id, arguments, name \\ "edit") do
    chain
    |> call(call_id, arguments, name)
    |> LLMChain.execute_tool_calls()
  end

  defp only_result(%LLMChain{last_message: %Message{tool_results: [result]}}), do: result

  defp text(%ToolResult{content: [%ContentPart{type: :text, content: text}]}), do: text

  describe "execute_tool_calls/2" do
    test "an identical failing call is prefixed with the notice" do
      first = new_chain(edit_tool()) |> round("c1", %{"path" => "bad"})
      assert text(only_result(first)) == "path not found"

      second = round(first, "c2", %{"path" => "bad"})
      result = only_result(second)

      assert result.is_error
      assert text(result) == @notice <> "\n\npath not found"
    end

    test "the same name with different arguments is not prefixed" do
      chain =
        new_chain(edit_tool())
        |> round("c1", %{"path" => "bad"})
        |> round("c2", %{"path" => "bad", "line" => 3})

      assert text(only_result(chain)) == "path not found"
    end

    test "argument key order does not matter" do
      chain =
        new_chain(scripted_tool([{:error, "nope"}, {:error, "nope"}]))
        |> round("c1", %{"a" => 1, "b" => 2})
        |> round("c2", %{"b" => 2, "a" => 1})

      assert text(only_result(chain)) == @notice <> "\n\nnope"
    end

    test "a failing call that succeeded in the previous round is not prefixed" do
      chain =
        new_chain(scripted_tool([{:ok, "fine"}, {:error, "timeout"}]))
        |> round("c1", %{"path" => "a"})
        |> round("c2", %{"path" => "a"})

      assert text(only_result(chain)) == "timeout"
    end

    test "an identical call that now succeeds is not prefixed" do
      chain =
        new_chain(scripted_tool([{:error, "timeout"}, {:ok, "fine"}]))
        |> round("c1", %{"path" => "a"})
        |> round("c2", %{"path" => "a"})

      result = only_result(chain)
      refute result.is_error
      assert text(result) == "fine"
    end

    test "each identical round is prefixed exactly once" do
      chain =
        new_chain(edit_tool())
        |> round("c1", %{"path" => "bad"})
        |> round("c2", %{"path" => "bad"})
        |> round("c3", %{"path" => "bad"})

      assert text(only_result(chain)) == @notice <> "\n\npath not found"
    end

    test "nil and empty arguments are the same call" do
      chain =
        new_chain(scripted_tool([{:error, "nope"}, {:error, "nope"}]))
        |> round("c1", nil)
        |> round("c2", %{})

      assert text(only_result(chain)) == @notice <> "\n\nnope"
    end

    test "a message between the rounds means no previous round" do
      chain =
        new_chain(edit_tool())
        |> round("c1", %{"path" => "bad"})
        |> LLMChain.add_message(Message.new_user!("try again"))
        |> round("c2", %{"path" => "bad"})

      assert text(only_result(chain)) == "path not found"
    end

    test "an unknown tool called twice is prefixed" do
      chain =
        new_chain(edit_tool())
        |> round("c1", %{}, "missing")
        |> round("c2", %{}, "missing")

      assert text(only_result(chain)) ==
               @notice <> "\n\nTool call made to missing but tool not found"
    end

    test "matching uses the arguments the model sent, not a review rewrite" do
      rewrite = %{
        on_tool_call_review: fn _chain, _call, _func, _review ->
          {:update_arguments, %{"path" => "bad"}}
        end
      }

      chain =
        new_chain(edit_tool())
        |> round("c1", %{"path" => "bad"})
        |> Map.update!(:callbacks, &[rewrite | &1])
        |> round("c2", %{"path" => "other"})

      assert text(only_result(chain)) == "path not found"
    end

    test ":on_tool_execution_failed receives the prefixed content" do
      test_pid = self()

      handler = %{
        on_tool_execution_failed: fn _chain, call, content ->
          send(test_pid, {:failed, call.call_id, content})
        end
      }

      new_chain(edit_tool(), [handler])
      |> round("c1", %{"path" => "bad"})
      |> round("c2", %{"path" => "bad"})

      assert_received {:failed, "c1", [%ContentPart{content: "path not found"}]}
      assert_received {:failed, "c2", [%ContentPart{content: content}]}
      assert content == @notice <> "\n\npath not found"
    end

    test "an async tool's result is prefixed" do
      chain =
        new_chain(edit_tool(async: true))
        |> round("c1", %{"path" => "bad"})
        |> round("c2", %{"path" => "bad"})

      assert text(only_result(chain)) == @notice <> "\n\npath not found"
    end

    test "string content returned in a ToolResult gets the prefix" do
      chain =
        new_chain(
          scripted_tool([
            {:ok, %ToolResult{is_error: true, content: "refused"}},
            {:ok, %ToolResult{is_error: true, content: "refused"}}
          ])
        )
        |> round("c1", %{})
        |> round("c2", %{})

      assert only_result(chain).content == @notice <> "\n\nrefused"
    end

    test "content that does not start with text gets a leading text part" do
      image = ContentPart.image!("aW1hZ2U=", media: :png)
      result = %ToolResult{is_error: true, content: [image]}

      chain =
        new_chain(scripted_tool([{:ok, result}, {:ok, result}]))
        |> round("c1", %{})
        |> round("c2", %{})

      assert [%ContentPart{type: :text, content: @notice}, ^image] =
               only_result(chain).content
    end
  end

  describe "execute_tool_calls_with_decisions/3" do
    test "an identical rejected call reported as an error is prefixed" do
      test_pid = self()

      handler = %{
        on_tool_execution_failed: fn _chain, call, content ->
          send(test_pid, {:failed, call.call_id, content})
        end
      }

      reject = %{type: :reject, message: "document_path is required.", is_error: true}

      decide = fn chain, call_id ->
        chain = call(chain, call_id, %{"path" => "x"})
        LLMChain.execute_tool_calls_with_decisions(chain, chain.last_message.tool_calls, [reject])
      end

      chain =
        new_chain(edit_tool(), [handler])
        |> decide.("c1")
        |> decide.("c2")

      assert text(only_result(chain)) == @notice <> "\n\ndocument_path is required."
      assert_received {:failed, "c1", "document_path is required."}
      assert_received {:failed, "c2", message}
      assert message == @notice <> "\n\ndocument_path is required."
    end

    test "a rejection that is not an error is not prefixed" do
      decide = fn chain, call_id, decision ->
        chain = call(chain, call_id, %{"path" => "bad"})

        LLMChain.execute_tool_calls_with_decisions(chain, chain.last_message.tool_calls, [
          decision
        ])
      end

      chain =
        new_chain(edit_tool())
        |> decide.("c1", %{type: :approve})
        |> decide.("c2", %{type: :reject, message: "Not now."})

      assert text(only_result(chain)) == "Not now."
    end

    test "an approved identical call that fails again is prefixed" do
      approve = fn chain, call_id ->
        chain = call(chain, call_id, %{"path" => "bad"})

        LLMChain.execute_tool_calls_with_decisions(chain, chain.last_message.tool_calls, [
          %{type: :approve}
        ])
      end

      chain =
        new_chain(edit_tool())
        |> approve.("c1")
        |> approve.("c2")

      assert text(only_result(chain)) == @notice <> "\n\npath not found"
    end
  end
end
