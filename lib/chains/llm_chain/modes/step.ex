defmodule LangChain.Chains.LLMChain.Modes.Step do
  @moduledoc """
  Execution mode that runs a single step at a time.

  A "step" is: execute any pending tool calls, then call the LLM.
  If no tools were pending, just call the LLM.

  ## Options

  - `:should_continue?` — Function `(LLMChain.t() -> boolean())` called after
    each step. If it returns `true`, another step runs. If `false` or not
    provided, the mode returns after one step.

  ## Usage

      # Single step
      LLMChain.run(chain, mode: :step)

      # Auto-loop with condition
      LLMChain.run(chain, mode: :step,
        should_continue?: fn chain ->
          chain.needs_response && length(chain.exchanged_messages) < 10
        end)
  """

  @behaviour LangChain.Chains.LLMChain.Mode

  alias LangChain.Chains.LLMChain

  @impl true
  def run(%LLMChain{} = chain, opts) do
    should_continue_fn = Keyword.get(opts, :should_continue?)

    case run_single_step(chain) do
      {:ok, updated_chain} ->
        if is_function(should_continue_fn, 1) && should_continue_fn.(updated_chain) do
          run(updated_chain, opts)
        else
          {:ok, updated_chain}
        end

      {:error, _chain, _reason} = error ->
        error
    end
  end

  defp run_single_step(%LLMChain{} = chain) do
    chain_after_tools = LLMChain.execute_tool_calls(chain)

    # If no tools were executed, automatically run LLM
    if chain_after_tools == chain do
      LLMChain.execute_step(chain_after_tools)
    else
      {:ok, chain_after_tools}
    end
  end
end
