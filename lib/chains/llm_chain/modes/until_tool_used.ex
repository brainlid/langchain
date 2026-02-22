defmodule LangChain.Chains.LLMChain.Modes.UntilToolUsed do
  @moduledoc """
  Execution mode that loops until a specific tool is called.

  Repeatedly calls the LLM and executes tools until a tool result
  matching one of the target tool names is found.

  ## Options

  - `:tool_names` — (required) List of tool name strings to watch for.
  - `:max_runs` — Maximum LLM calls before error. Default: 25.

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
      do_run(chain, opts)
    end
  end

  defp do_run(chain, opts) do
    {:continue, chain}
    |> call_llm()
    |> check_max_runs(opts)
    |> execute_tools()
    |> check_until_tool(opts)
    |> continue_or_done(&do_run/2, opts)
  end
end
