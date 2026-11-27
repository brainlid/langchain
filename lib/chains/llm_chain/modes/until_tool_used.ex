defmodule LangChain.Chains.LLMChain.Modes.UntilToolUsed do
  @moduledoc """
  Execution mode that loops until a specific tool is called.

  Repeatedly calls the LLM and executes tools until a successful tool result
  matching one of the target tool names is found.

  A failed call to a target tool does not end the run. That includes a call
  whose arguments the tool's `:parse_args` parser rejected, which makes this
  mode suitable for a tool whose job is to receive structured arguments: the
  parser validates them, and invalid arguments go back to the model as an error
  to correct. The failure is counted like any other tool error, so a model
  that keeps failing stops at the chain's `max_retry_count` with
  `%LangChainError{type: "exceeded_failure_count"}`.

  A tool that returns `{:interrupt, display_message, interrupt_data}` stops the
  run with `{:interrupt, chain, interrupt_data}` before the LLM is called again.
  See `LangChain.Chains.LLMChain.Modes.WhileNeedsResponse` for the shape of the
  interrupt data and how to continue.

  ## Options

  - `:tool_names` — (required) List of tool name strings to watch for.
  - `:max_runs` — Maximum LLM calls before error. Default: 25.
  - `:require_tool_success` — When `false`, any call to a target tool ends the
    run, including a failed one, and its error result is returned. Default:
    `true`.

  ## Usage

      LLMChain.run(chain,
        mode: LangChain.Chains.LLMChain.Modes.UntilToolUsed,
        tool_names: ["submit"],
        max_runs: 25
      )
      # => {:ok, chain, %ToolResult{name: "submit", ...}}
  """

  @behaviour LangChain.Chains.LLMChain.Mode
  import LangChain.Chains.LLMChain.Mode.Steps

  alias LangChain.Chains.LLMChain
  alias LangChain.LangChainError

  @impl true
  def run(%LLMChain{} = chain, opts) do
    tool_names = Keyword.fetch!(opts, :tool_names)

    # Validate tools exist
    missing = Enum.filter(tool_names, fn name -> !Map.has_key?(chain._tool_map, name) end)

    if missing != [] do
      message =
        if length(missing) > 1 do
          "Tool names '#{Enum.join(missing, ", ")}' not found in available tools"
        else
          "Tool name '#{List.first(missing)}' not found in available tools"
        end

      {:error, chain,
       LangChainError.exception(
         type: "invalid_tool_name",
         message: message
       )}
    else
      chain = ensure_mode_state(chain)
      do_run(chain, Keyword.put_new(opts, :require_tool_success, true))
    end
  end

  defp do_run(chain, opts) do
    # Order matters: `check_max_runs` must come AFTER `check_until_tool`,
    # otherwise the very LLM call that successfully invokes the target
    # tool can have its result discarded — `call_llm` increments the run
    # counter, the check fires and short-circuits the pipeline before
    # `execute_tools`/`check_until_tool` get to see the result. With this
    # order, a successful run terminates first; `check_max_runs` is a
    # no-op on terminal pipeline states.
    {:continue, chain}
    |> call_llm()
    |> execute_tools()
    |> check_tool_interrupts(opts)
    |> check_until_tool(opts)
    |> check_max_runs(opts)
    |> continue_or_done(&do_run/2, opts)
  end
end
