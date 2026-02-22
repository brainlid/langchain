defmodule LangChain.Chains.LLMChain.Modes.UntilSuccess do
  @moduledoc """
  Execution mode that loops until a successful result.

  Calls the LLM, executes tool calls, and repeats until:
  - The last message is an assistant response (success)
  - The last message is a tool result with no errors (success)
  - Max retry count is exceeded (error)

  ## Options

  - `:force_recurse` — When `true`, forces recursion even after a successful
    result. Used internally by `UntilToolUsed` mode. Default: `false`.

  ## Usage

      LLMChain.run(chain, mode: :until_success)
  """

  @behaviour LangChain.Chains.LLMChain.Mode

  alias LangChain.Chains.LLMChain
  alias LangChain.Message
  alias LangChain.LangChainError

  @impl true
  def run(%LLMChain{last_message: %Message{} = last_message} = chain, opts) do
    force_recurse = Keyword.get(opts, :force_recurse, false)

    stop_or_recurse =
      cond do
        force_recurse ->
          :recurse

        chain.current_failure_count >= chain.max_retry_count ->
          {:error, chain,
           LangChainError.exception(
             type: "exceeded_failure_count",
             message: "Exceeded max failure count"
           )}

        last_message.role == :tool && !Message.tool_had_errors?(last_message) ->
          {:ok, chain}

        last_message.role == :assistant ->
          {:ok, chain}

        true ->
          :recurse
      end

    case stop_or_recurse do
      :recurse ->
        case LLMChain.execute_step(chain) do
          {:ok, updated_chain} ->
            updated_chain
            |> LLMChain.execute_tool_calls()
            |> run(opts)

          {:error, _chain, _reason} = error ->
            error
        end

      other ->
        other
    end
  end

  # Initial call when no messages have been processed yet
  def run(%LLMChain{} = chain, opts) do
    case LLMChain.execute_step(chain) do
      {:ok, updated_chain} ->
        updated_chain
        |> LLMChain.execute_tool_calls()
        |> run(opts)

      {:error, _chain, _reason} = error ->
        error
    end
  end
end
