defmodule LangChain.Chains.LLMChain.Modes.StepMode do
  @moduledoc """
  Execution mode for step-by-step control of the chain.

  This mode executes one step at a time, giving you control over the execution
  flow. Each step:

  1. Executes any pending tool calls
  2. If no tools were executed, automatically calls the LLM
  3. Returns the updated chain

  The mode supports two usage patterns:

  ## Manual Stepping

  Execute one step at a time without a continuation function. This allows you to
  inspect the chain, modify it, and decide whether to continue:

      {:ok, chain} = LLMChain.run(chain, mode: :step)
      # Inspect chain.last_message, check tool calls, etc.
      if should_continue?(chain) do
        # Optionally modify the chain before continuing
        {:ok, chain} = LLMChain.run(chain, mode: :step)
      end

  ## Automated Stepping with Continuation Function

  Provide a `should_continue?` function that inspects the chain state and returns
  a boolean. The mode will automatically loop based on this function:

      should_continue_fn = fn chain ->
        chain.needs_response && length(chain.exchanged_messages) < 10
      end

      {:ok, final_chain} = LLMChain.run(chain,
        mode: :step,
        should_continue?: should_continue_fn
      )

  ## Use Cases

  - Debugging workflows where you need to inspect each step
  - Conditional stopping based on chain state
  - Building custom control flows
  - Implementing max iteration limits
  - Stopping when specific conditions are met

  ## Example

      chain
      |> LLMChain.run(mode: :step)

  Or using the module directly:

      chain
      |> LLMChain.run(mode: LangChain.Chains.LLMChain.Modes.StepMode)

  """

  @behaviour LangChain.Chains.LLMChain.Mode

  alias LangChain.Chains.LLMChain

  @impl true
  def run(%LLMChain{} = chain, opts) do
    case Keyword.get(opts, :should_continue?) do
      should_continue_fn when is_function(should_continue_fn, 1) ->
        run_with_continuation(chain, should_continue_fn)

      _ ->
        LLMChain.execute_step_with_tools(chain)
    end
  end

  # Run step with automatic continuation based on the provided function
  defp run_with_continuation(%LLMChain{} = chain, should_continue_fn) do
    case LLMChain.execute_step_with_tools(chain) do
      {:ok, updated_chain} ->
        if should_continue_fn.(updated_chain) do
          run_with_continuation(updated_chain, should_continue_fn)
        else
          {:ok, updated_chain}
        end

      {:error, updated_chain, reason} ->
        {:error, updated_chain, reason}
    end
  end
end
