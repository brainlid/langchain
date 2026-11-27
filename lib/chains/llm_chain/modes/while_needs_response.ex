defmodule LangChain.Chains.LLMChain.Modes.WhileNeedsResponse do
  @moduledoc """
  Execution mode that loops while the chain needs a response.

  After each LLM call, if the response contains tool calls, this mode:
  1. Executes the pending tool calls
  2. Stops with `{:interrupt, chain, interrupt_data}` if any of those tools
     returned an interrupt
  3. Applies any `LangChain.MessageExpansion` those tools asked for
  4. Calls the LLM again with the tool results
  5. Repeats until `needs_response` is false (no more tool calls)

  The LLM gets the last word after tool execution unless a tool interrupted.

  An assistant message that leaves the model's turn open (see
  `LangChain.Message.continues_turn?/1`) also leaves `needs_response` true, so
  the LLM is called again to finish its turn.

  Steps 2 and 3 cost nothing for a tool that neither interrupts nor asks for
  an expansion, which is every tool that has not opted in. They are included
  so that a tool returning either is honoured under the standard mode rather
  than silently ignored.

  ## Tool Interrupts

  A tool interrupts by returning `{:interrupt, display_message, interrupt_data}`,
  for example to ask the user a question and wait for the answer. The run
  returns `{:interrupt, chain, interrupt_data}` with the tool's result recorded
  as the chain's last message, and the LLM is not called. `interrupt_data` is
  the tool's data with its `:tool_call_id` added; when several tools interrupt
  at once it is `%{type: :multiple_interrupts, interrupts: [...]}`.

  To continue, add a message that answers the interrupt (typically the user's
  reply) and run the chain again. An interrupted tool message that is no longer
  the last message does not interrupt again.

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
    |> check_tool_interrupts(opts)
    |> expand_tool_results(opts)
    |> call_llm()
    |> continue_or_done(&do_run/2, opts)
  end
end
