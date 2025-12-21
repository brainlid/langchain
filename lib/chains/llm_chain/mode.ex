defmodule LangChain.Chains.LLMChain.Mode do
  @moduledoc """
  Behavior for LLMChain execution modes.

  Execution modes control how an LLMChain processes messages and handles tool
  execution. By implementing this behavior, you can create custom execution
  strategies tailored to your specific use case.

  ## Built-in Modes

  LangChain provides several built-in modes:

  - `LangChain.Chains.LLMChain.Modes.WhileNeedsResponseMode` - Loops while the chain needs a response
  - `LangChain.Chains.LLMChain.Modes.UntilSuccessMode` - Retries until successful tool execution
  - `LangChain.Chains.LLMChain.Modes.StepMode` - Step-by-step execution with optional continuation

  ## Implementing a Custom Mode

  To create a custom mode, implement the `c:run/2` callback:

      defmodule MyApp.CustomMode do
        @behaviour LangChain.Chains.LLMChain.Mode

        alias LangChain.Chains.LLMChain

        @impl true
        def run(%LLMChain{} = chain, opts) do
          # Your custom execution logic here
          # Use LLMChain.execute_step/1 to run a single step
          # Use LLMChain.execute_tools/1 to execute pending tool calls
          # Return {:ok, updated_chain} or {:error, chain, reason}
        end
      end

  Then use it with `LLMChain.run/2`:

      LLMChain.run(chain, mode: MyApp.CustomMode)
  """

  alias LangChain.Chains.LLMChain

  @doc """
  Execute the chain according to the mode's strategy.

  This callback receives the chain and options, and returns either:
  - `{:ok, updated_chain}` on success
  - `{:error, chain, reason}` on failure

  The mode is responsible for:
  - Determining when to call the LLM
  - Executing tool calls when needed
  - Handling retry logic
  - Deciding when to stop execution

  ## Parameters

  - `chain` - The LLMChain to execute
  - `opts` - Keyword list of options that may include mode-specific configuration

  ## Returns

  - `{:ok, LLMChain.t()}` - Successfully executed chain
  - `{:error, LLMChain.t(), LangChainError.t()}` - Execution failed
  """
  @callback run(chain :: LLMChain.t(), opts :: Keyword.t()) ::
              {:ok, LLMChain.t()} | {:error, LLMChain.t(), LangChain.LangChainError.t()}
end
