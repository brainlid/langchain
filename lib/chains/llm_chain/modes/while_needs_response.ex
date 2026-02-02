defmodule LangChain.Chains.LLMChain.Modes.WhileNeedsResponse do
  @moduledoc """
  Execution mode that loops while the chain needs a response.

  This mode is designed for interactive chats that make tool calls. It:

  1. Executes any pending tool calls
  2. Calls the LLM to get a response
  3. Repeats steps 1-2 until the chain no longer needs a response

  The key characteristic is that the LLM always gets the "last word" - execution
  continues until the assistant provides a final response without tool calls.

  ## Use Cases

  - Conversational chatbots with tool/function calling
  - Interactive agents that need to use tools and respond to results
  - Any workflow where the assistant should respond to tool execution results

  ## Example

      chain
      |> LLMChain.run(mode: :while_needs_response)

  Or using the module directly:

      chain
      |> LLMChain.run(mode: LangChain.Chains.LLMChain.Modes.WhileNeedsResponseMode)

  ## How it works

  The mode checks the `needs_response` field on the chain:
  - If `false`, execution stops and returns the chain
  - If `true`, it executes tools, calls the LLM, and recursively continues
  """

  @behaviour LangChain.Chains.LLMChain.Mode

  alias LangChain.Chains.LLMChain

  @impl true
  def run(chain, opts \\ [])

  def run(%LLMChain{needs_response: false} = chain, _opts) do
    {:ok, chain}
  end

  def run(%LLMChain{needs_response: true} = chain, opts) do
    chain
    |> LLMChain.execute_tool_calls()
    |> LLMChain.execute_step()
    |> case do
      {:ok, updated_chain} ->
        run(updated_chain, opts)

      {:error, updated_chain, reason} ->
        {:error, updated_chain, reason}
    end
  end
end
