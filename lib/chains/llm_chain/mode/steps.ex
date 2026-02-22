defmodule LangChain.Chains.LLMChain.Mode.Steps do
  @moduledoc """
  Pipe-friendly building blocks for composing custom execution modes.

  Every step function follows the pipeline convention:

  - Input/output: `{:continue, chain}` means keep processing
  - Any other tuple (`:ok`, `:pause`, `:error`) is a terminal — passed through unchanged
  - Step functions take the pipeline result as first arg, optional config as second

  ## Example: Custom Mode

      defmodule MyApp.Modes.Simple do
        @behaviour LangChain.Chains.LLMChain.Mode
        import LangChain.Chains.LLMChain.Mode.Steps

        @impl true
        def run(chain, opts) do
          chain = ensure_mode_state(chain)

          {:continue, chain}
          |> call_llm()
          |> execute_tools()
          |> check_max_runs(opts)
          |> continue_or_done(&run/2, opts)
        end
      end
  """

  alias LangChain.Chains.LLMChain

  @type pipeline_result ::
          {:continue, LLMChain.t()}
          | {:ok, LLMChain.t()}
          | {:ok, LLMChain.t(), term()}
          | {:pause, LLMChain.t()}
          | {:error, LLMChain.t(), term()}
          | {:interrupt, LLMChain.t(), term()}

  # ── Core Execution ──────────────────────────────────────────────

  @doc """
  Call the LLM (single step). Wraps `LLMChain.execute_step/1`.

  On success, increments `mode_state.run_count` in `custom_context`.
  """
  def call_llm({:continue, chain}) do
    case LLMChain.execute_step(chain) do
      {:ok, updated_chain} ->
        {:continue, increment_run_count(updated_chain)}

      {:error, chain, reason} ->
        {:error, chain, reason}
    end
  end

  def call_llm(terminal), do: terminal

  @doc """
  Execute pending tool calls. Wraps `LLMChain.execute_tool_calls/1`.
  """
  def execute_tools({:continue, chain}) do
    {:continue, LLMChain.execute_tool_calls(chain)}
  end

  def execute_tools(terminal), do: terminal

  # ── Safety Checks ───────────────────────────────────────────────

  @doc """
  Check if max runs have been exceeded.

  Reads `run_count` from `custom_context.mode_state` and compares against
  `:max_runs` in opts (default: 25).
  """
  def check_max_runs({:continue, chain}, opts) do
    max = Keyword.get(opts, :max_runs, 25)
    count = get_run_count(chain)

    if count >= max do
      {:error, chain,
       LangChain.LangChainError.exception(
         type: "exceeded_max_runs",
         message: "Exceeded maximum number of runs"
       )}
    else
      {:continue, chain}
    end
  end

  def check_max_runs(terminal, _opts), do: terminal

  # ── Pause (Infrastructure Drain) ────────────────────────────────

  @doc """
  Check if execution should pause (e.g., node draining).

  Reads `:should_pause?` from opts — a zero-arity function that returns boolean.
  """
  def check_pause({:continue, chain}, opts) do
    case Keyword.get(opts, :should_pause?) do
      fun when is_function(fun, 0) ->
        if fun.(), do: {:pause, chain}, else: {:continue, chain}

      _ ->
        {:continue, chain}
    end
  end

  def check_pause(terminal, _opts), do: terminal

  # ── Until-Tool Termination ──────────────────────────────────────

  @doc """
  Check if a target tool was called in the most recent tool results.

  Reads `:tool_names` from opts — a list of tool name strings.
  If a matching tool result is found, returns `{:ok, chain, tool_result}`.
  """
  def check_until_tool({:continue, chain}, opts) do
    case Keyword.get(opts, :tool_names) do
      nil ->
        {:continue, chain}

      names when is_list(names) ->
        case find_matching_tool_result(chain, names) do
          {:found, tool_result} -> {:ok, chain, tool_result}
          :not_found -> {:continue, chain}
        end
    end
  end

  def check_until_tool(terminal, _opts), do: terminal

  # ── Loop Boundary ───────────────────────────────────────────────

  @doc """
  Decide whether to loop or return.

  - `{:continue, chain}` with `needs_response: true` → call `run_fn.(chain, opts)` (loop)
  - `{:continue, chain}` with `needs_response: false` → `{:ok, chain}` (done)
  - Any terminal result → pass through as-is
  """
  def continue_or_done({:continue, %LLMChain{needs_response: true} = chain}, run_fn, opts) do
    run_fn.(chain, opts)
  end

  def continue_or_done({:continue, chain}, _run_fn, _opts) do
    {:ok, chain}
  end

  def continue_or_done(terminal, _run_fn, _opts) do
    terminal
  end

  # ── Mode State Helpers ──────────────────────────────────────────

  @doc """
  Initialize mode_state in custom_context if not already present.

  Call this at the top of your mode's `run/2` to set up run_count tracking.
  On the first call, creates `mode_state: %{run_count: 0}`.
  On recursive calls (mode_state already exists), returns chain unchanged.
  """
  def ensure_mode_state(%LLMChain{} = chain) do
    case get_in_custom_context(chain, [:mode_state]) do
      nil ->
        LLMChain.update_custom_context(chain, %{mode_state: %{run_count: 0}})

      _existing ->
        chain
    end
  end

  @doc """
  Get the current run count from mode_state.
  """
  def get_run_count(%LLMChain{} = chain) do
    get_in_custom_context(chain, [:mode_state, :run_count]) || 0
  end

  # ── Private Helpers ─────────────────────────────────────────────

  defp increment_run_count(%LLMChain{} = chain) do
    count = get_run_count(chain)
    mode_state = get_in_custom_context(chain, [:mode_state]) || %{}
    updated_mode_state = Map.put(mode_state, :run_count, count + 1)
    LLMChain.update_custom_context(chain, %{mode_state: updated_mode_state})
  end

  defp find_matching_tool_result(%LLMChain{last_message: last_message}, tool_names) do
    case last_message do
      %{role: :tool, tool_results: tool_results} when is_list(tool_results) ->
        case Enum.find(tool_results, &(&1.name in tool_names)) do
          nil -> :not_found
          tool_result -> {:found, tool_result}
        end

      _ ->
        :not_found
    end
  end

  defp get_in_custom_context(%LLMChain{custom_context: ctx}, keys) when is_map(ctx) do
    get_in(ctx, keys)
  end

  defp get_in_custom_context(_chain, _keys), do: nil
end
