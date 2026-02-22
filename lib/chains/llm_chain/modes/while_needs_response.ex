defmodule LangChain.Chains.LLMChain.Modes.WhileNeedsResponse do
  @moduledoc """
  Execution mode that loops while the chain needs a response.

  After each LLM call, if the response contains tool calls, this mode:
  1. Executes the pending tool calls
  2. Calls the LLM again with the tool results
  3. Repeats until `needs_response` is false (no more tool calls)

  The LLM always gets the last word after tool execution.

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
    chain = ensure_mode_state(chain)

    {:continue, chain}
    |> execute_tools()
    |> call_llm()
    |> continue_or_done(&run/2, opts)
  end
end
