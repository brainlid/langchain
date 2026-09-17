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
  alias LangChain.Message
  alias LangChain.Message.ContentPart
  alias LangChain.Message.ToolResult
  alias LangChain.MessageExpansion

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

  @doc """
  Expand any tool results that asked to, before the model is called.

  A tool result can carry a `LangChain.MessageExpansion` asking for messages to
  be inserted into the conversation and for the result's own content to be
  trimmed once they are. This step is what honours that request. Placing it
  immediately before `call_llm/1` is what makes the guarantee the tool is
  relying on true: the messages are in the conversation for the very next model
  call, in the same run.

      {:continue, chain}
      |> expand_tool_results(opts)
      |> call_llm()

  ## What it does

  Reads `chain.last_message`. If it is a `:tool` message, every result carrying
  an expansion is applied in tool-call order:

  1. The result's content becomes the expansion's `result_content`, or is left
     alone when that is nil.
  2. The expansion is cleared from the result, so applying the step twice
     inserts once.
  3. The expansion's messages are appended with `LLMChain.add_messages/2`, which
     keeps `messages`, `exchanged_messages`, `last_message` and `needs_response`
     in agreement.

  Any other pipeline result passes through untouched, including a terminal. A
  turn that interrupted or satisfied an until-tool contract is over, so nothing
  is inserted into it.

  ## Where it must not go

  Not after the steps that decide whether the run is over. `check_until_tool/2`,
  `Sagents.Mode.Steps.check_until_tool_success/2` and the chain's telemetry all
  read `chain.last_message` to answer that question, and every one of them is
  entitled to find a message the model produced or the tools returned there.
  Expanding before the model call keeps that true; expanding after the terminal
  checks would put a synthetic message under code that cannot tell the
  difference.

  Inserted messages fire no callbacks, so they produce no transcript rows. A
  host that mirrors a conversation from `:on_message_processed` sees them when
  it next reads the whole chain, not as they are inserted.
  """
  def expand_tool_results(pipeline_result, opts \\ [])

  def expand_tool_results(
        {:continue, %LLMChain{last_message: %Message{role: :tool} = tool_message} = chain},
        _opts
      ) do
    case Enum.filter(tool_message.tool_results || [], &MessageExpansion.expandable?/1) do
      [] -> {:continue, chain}
      expanding -> {:continue, apply_expansions(chain, tool_message, expanding)}
    end
  end

  def expand_tool_results({:continue, chain}, _opts), do: {:continue, chain}

  def expand_tool_results(terminal, _opts), do: terminal

  defp apply_expansions(%LLMChain{} = chain, %Message{} = tool_message, expanding) do
    inserted = Enum.flat_map(expanding, & &1.message_expansion.messages)

    chain
    |> replace_message(tool_message, consume_expansions(tool_message))
    |> LLMChain.add_messages(inserted)
  end

  defp consume_expansions(%Message{tool_results: results} = tool_message) do
    %Message{tool_message | tool_results: Enum.map(results, &consume_expansion/1)}
  end

  defp consume_expansion(%ToolResult{message_expansion: %MessageExpansion{} = expansion} = result) do
    if MessageExpansion.expandable?(result) do
      %ToolResult{result | content: trimmed_content(expansion, result), message_expansion: nil}
    else
      result
    end
  end

  defp consume_expansion(%ToolResult{} = result), do: result

  # Mirrors what `ToolResult.new/1` does to a binary content, so a trimmed
  # result is shaped the same as one the tool built itself.
  defp trimmed_content(%MessageExpansion{result_content: nil}, %ToolResult{content: content}),
    do: content

  defp trimmed_content(%MessageExpansion{result_content: content}, _result)
       when is_binary(content),
       do: [ContentPart.text!(content)]

  defp trimmed_content(%MessageExpansion{result_content: %ContentPart{} = part}, _result),
    do: [part]

  defp trimmed_content(%MessageExpansion{result_content: content}, _result), do: content

  # The tool message is rewritten in place wherever the chain holds it.
  # `exchanged_messages` legitimately may not: it is reset per run, so a chain
  # built fresh from stored messages (a resume after human approval) carries the
  # tool message in `messages` alone.
  defp replace_message(%LLMChain{} = chain, old, new) do
    %LLMChain{
      chain
      | messages: swap_message(chain.messages, old, new),
        exchanged_messages: swap_message(chain.exchanged_messages, old, new),
        last_message: if(chain.last_message == old, do: new, else: chain.last_message)
    }
  end

  defp swap_message(messages, old, new) when is_list(messages) do
    Enum.map(messages, fn
      ^old -> new
      other -> other
    end)
  end

  defp swap_message(other, _old, _new), do: other

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
         message: "Exceeded maximum number of runs (#{count}/#{max})"
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

  # ── Tool Interrupts ────────────────────────────────────────────

  @doc """
  Check if any tool results in the most recent tool message are interrupts.

  Returns `{:interrupt, chain, interrupt_data}` if any tool results have
  `is_interrupt: true`. The `interrupt_data` is extracted from the first
  interrupted result (for single interrupts) or aggregated (for multiple).

  This step is generic — it doesn't know *why* a tool interrupted. The
  consumer (e.g., Sagents) interprets the interrupt data.
  """
  def check_tool_interrupts({:continue, chain}, _opts) do
    case find_interrupted_results(chain) do
      [] ->
        {:continue, chain}

      interrupted_results ->
        interrupt_data = extract_interrupt_data(interrupted_results)
        {:interrupt, chain, interrupt_data}
    end
  end

  def check_tool_interrupts(terminal, _opts), do: terminal

  defp find_interrupted_results(chain) do
    case Enum.find(Enum.reverse(chain.messages), &(&1.role == :tool)) do
      nil ->
        []

      %{tool_results: tool_results} when is_list(tool_results) ->
        Enum.filter(tool_results, & &1.is_interrupt)

      _ ->
        []
    end
  end

  defp extract_interrupt_data([single]) do
    data = single.interrupt_data || %{}
    Map.put(data, :tool_call_id, single.tool_call_id)
  end

  defp extract_interrupt_data(multiple) do
    %{
      type: :multiple_interrupts,
      interrupts:
        Enum.map(multiple, fn result ->
          data = result.interrupt_data || %{}
          Map.merge(data, %{tool_call_id: result.tool_call_id})
        end)
    }
  end

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
