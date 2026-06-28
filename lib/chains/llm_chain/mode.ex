defmodule LangChain.Chains.LLMChain.Mode do
  @moduledoc """
  Behaviour for LLMChain execution modes.

  A mode controls the execution loop of an LLMChain - how many times the LLM
  is called, when tools are executed, and under what conditions the loop
  terminates.

  ## Built-in Modes

  - `LangChain.Chains.LLMChain.Modes.WhileNeedsResponse` - loop while `needs_response` is true
  - `LangChain.Chains.LLMChain.Modes.UntilSuccess` - loop until successful tool execution or assistant response
  - `LangChain.Chains.LLMChain.Modes.Step` - single step with optional continuation
  - `LangChain.Chains.LLMChain.Modes.UntilToolUsed` - loop until a specific tool is called

  ## Custom Modes

  Implement this behaviour to define custom execution loops. The simplest
  approach is to import `LangChain.Chains.LLMChain.Mode.Steps` and compose
  the provided step functions into a pipeline. Each step follows a consistent
  contract:

  - `{:continue, chain}` - keep processing, pipe into the next step
  - Any other tuple (`:ok`, `:pause`, `:error`) — terminal result, passed through unchanged

  This means a pipeline short-circuits automatically when any step returns a
  terminal value.

  ### Minimal Example

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

  ### Composable Example Using Steps

  The built-in modes are all composed from `Steps` functions. You can do the
  same to build a mode that adds custom logic at any point in the pipeline.

  For example, a mode that loops like `WhileNeedsResponse` but adds a custom
  check after each tool execution round:

      defmodule MyApp.Modes.WithAudit do
        @behaviour LangChain.Chains.LLMChain.Mode
        import LangChain.Chains.LLMChain.Mode.Steps

        @impl true
        def run(chain, opts) do
          chain = ensure_mode_state(chain)

          {:continue, chain}
          |> execute_tools()
          |> call_llm()
          |> check_max_runs(opts)
          |> audit_step(opts)
          |> continue_or_done(&run/2, opts)
        end

        # A custom step - same contract as the built-in steps:
        # accept {:continue, chain} and return {:continue, chain} or a terminal.
        defp audit_step({:continue, chain}, _opts) do
          run_count = get_run_count(chain)
          Logger.info("LLM call #\#{run_count} completed")
          {:continue, chain}
        end

        defp audit_step(terminal, _opts), do: terminal
      end

  The pattern is always the same: start with `{:continue, chain}`, pipe
  through as many steps as needed, and end with `continue_or_done/3` to
  decide whether to loop.

  See `LangChain.Chains.LLMChain.Mode.Steps` for the full list of available
  step functions. The built-in modes (`WhileNeedsResponse`, `UntilSuccess`,
  `Step`, `UntilToolUsed`) also serve as practical examples to reference.

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
          | {:interrupt, LLMChain.t(), term()}
          | {:error, LLMChain.t(), LangChainError.t()}

  @doc """
  Execute the chain according to this mode's logic.

  The `opts` keyword list contains all options passed to `LLMChain.run/2`,
  allowing modes to define their own options.
  """
  @callback run(chain :: LLMChain.t(), opts :: Keyword.t()) :: run_result()
end
