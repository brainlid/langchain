defmodule LangChain.Chains.LLMChain.Modes.UntilSuccess do
  @moduledoc """
  Execution mode that retries until a successful result is obtained.

  This mode is designed for non-interactive processing where the LLM may need to
  retry failed operations. It:

  1. Calls the LLM to get a response
  2. Executes any tool calls
  3. Checks if the result is successful:
     - Successful tool result (no errors) → stops
     - Assistant message → stops
     - Failed tool result → retries
  4. Repeats until success or max retry count is exceeded

  The key characteristic is that execution stops as soon as a successful result
  is obtained. Unlike `:while_needs_response`, the LLM does not get the last word.

  ## Use Cases

  - Non-interactive workflows where you want to stop once a tool succeeds
  - Processing tasks where errors should trigger retries
  - Data extraction where you want to stop once valid data is obtained

  ## Example

      chain
      |> LLMChain.run(mode: :until_success)

  Or using the module directly:

      chain
      |> LLMChain.run(mode: LangChain.Chains.LLMChain.Modes.UntilSuccessMode)

  ## How it works

  The mode checks the last message and failure count:
  - If max retries exceeded → returns error
  - If last message is a successful tool result → returns success
  - If last message is an assistant message → returns success
  - Otherwise → retries by calling the LLM again
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
          # a successful tool result has no errors
          {:ok, chain}

        last_message.role == :assistant ->
          # it was successful if we didn't generate a user message in response to
          # an error.
          {:ok, chain}

        true ->
          :recurse
      end

    case stop_or_recurse do
      :recurse ->
        chain
        |> LLMChain.execute_step()
        |> case do
          {:ok, updated_chain} ->
            updated_chain
            |> LLMChain.execute_tool_calls()
            |> then(&run(&1, opts))

          {:error, updated_chain, reason} ->
            {:error, updated_chain, reason}
        end

      other ->
        # return the error or success result
        other
    end
  end
end
