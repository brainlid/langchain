defmodule LangChain.Chains.LLMChain.Mode do
  @moduledoc """
  Behaviour for LLMChain execution modes.

  A mode controls the execution loop of an LLMChain — how many times the LLM
  is called, when tools are executed, and under what conditions the loop
  terminates.

  ## Built-in Modes

  - `LangChain.Chains.LLMChain.Modes.WhileNeedsResponse` - loop while `needs_response` is true
  - `LangChain.Chains.LLMChain.Modes.UntilSuccess` - loop until successful tool execution or assistant response
  - `LangChain.Chains.LLMChain.Modes.Step` - single step with optional continuation
  - `LangChain.Chains.LLMChain.Modes.UntilToolUsed` - loop until a specific tool is called

  ## Custom Modes

  Implement this behaviour to define custom execution loops:

      defmodule MyApp.Modes.Custom do
        @behaviour LangChain.Chains.LLMChain.Mode

        @impl true
        def run(chain, opts) do
          case LLMChain.execute_step(chain) do
            {:ok, updated} ->
              updated = LLMChain.execute_tool_calls(updated)
              if updated.needs_response, do: run(updated, opts), else: {:ok, updated}
            error -> error
          end
        end
      end

  ## Return Types

  - `{:ok, chain}` - normal completion
  - `{:ok, chain, extra}` - completion with additional data
  - `{:pause, chain}` - execution paused at a clean checkpoint; resumable
  - `{:error, chain, reason}` - execution failed
  """

  alias LangChain.Chains.LLMChain
  alias LangChain.LangChainError

  @type run_result ::
          {:ok, LLMChain.t()}
          | {:ok, LLMChain.t(), term()}
          | {:pause, LLMChain.t()}
          | {:error, LLMChain.t(), LangChainError.t()}

  @doc """
  Execute the chain according to this mode's logic.

  The `opts` keyword list contains all options passed to `LLMChain.run/2`,
  allowing modes to define their own options.
  """
  @callback run(chain :: LLMChain.t(), opts :: Keyword.t()) :: run_result()
end
