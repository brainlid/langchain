defmodule LangChain.Chains.LLMChain.Modes.WhileNeedsResponse do
  @moduledoc """
  Execution mode that loops while the chain needs a response.

  After each LLM call, if the response contains tool calls, this mode:
  1. Executes the pending tool calls
  2. Applies any `LangChain.MessageExpansion` those tools asked for
  3. Calls the LLM again with the tool results
  4. Repeats until `needs_response` is false (no more tool calls)

  The LLM always gets the last word after tool execution.

  An assistant message that is narration only (see
  `LangChain.Message.narration?/1`) also leaves `needs_response` true, so the
  LLM is called again to finish its turn.

  Step 2 costs nothing for a tool that asks for no expansion, which is every
  tool that has not opted in. It is included so that a tool carrying one is
  honoured under the standard mode rather than silently ignored.

  ## Options

  - `:max_runs` - Maximum LLM calls in one run before returning
    `%LangChainError{type: "exceeded_max_runs"}`. Default: 25. The count starts
    at 0 on every `LLMChain.run/2`, so a chain that is run again after a new
    message gets a fresh budget.

  ## Usage

      LLMChain.run(chain, mode: :while_needs_response)
      # or
      LLMChain.run(chain, mode: LangChain.Chains.LLMChain.Modes.WhileNeedsResponse)
  """

  @behaviour LangChain.Chains.LLMChain.Mode

  import LangChain.Chains.LLMChain.Mode.Steps

  alias LangChain.Chains.LLMChain

  @impl true
  def run(%LLMChain{needs_response: false} = chain, _opts) do
    {:ok, chain}
  end

  def run(%LLMChain{} = chain, opts) do
    chain
    |> reset_run_count()
    |> do_run(Keyword.put_new(opts, :max_runs, 25))
  end

  defp do_run(%LLMChain{needs_response: false} = chain, _opts) do
    {:ok, chain}
  end

  defp do_run(%LLMChain{} = chain, opts) do
    {:continue, chain}
    |> check_max_runs(opts)
    |> execute_tools()
    |> expand_tool_results(opts)
    |> call_llm()
    |> continue_or_done(&do_run/2, opts)
  end
end
