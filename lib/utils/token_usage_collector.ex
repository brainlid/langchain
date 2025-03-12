defmodule LangChain.Utils.TokenUsageCollector do
  @moduledoc """
  A utility module providing a token usage collector using an Agent.

  This module offers a convenient and best-practice approach to collecting
  token usage information within the LangChain library, specifically designed
  for use with the `on_llm_token_usage` callback.

  In scenarios where a single task involves multiple interactions with AI models,
  this Agent-backed collector allows for easy aggregation of total token usage.

  ## Example

    iex> collector = LangChain.Utils.TokenUsageCollector.new()
    iex> LangChain.Utils.TokenUsageCollector.acc(collector, %{
    ...>   "gemini-2.0-flash" => %LangChain.TokenUsage{
    ...>     input: 1,
    ...>     output: 2,
    ...>     raw: %{"promptTokenCount" => 1, "candidatesTokenCount" => 2, "totalTokenCount" => 3}
    ...>   }
    ...> })
    iex> LangChain.Utils.TokenUsageCollector.acc(collector, %{
    ...>   "gemini-2.0-flash" => %LangChain.TokenUsage{
    ...>     input: 3,
    ...>     output: 7,
    ...>     raw: %{"promptTokenCount" => 3, "candidatesTokenCount" => 7, "totalTokenCount" => 10}
    ...>   }
    ...> })
    iex> LangChain.Utils.TokenUsageCollector.acc(collector, %{
    ...>   "gemini-1.5-pro" => %LangChain.TokenUsage{
    ...>     input: 2,
    ...>     output: 5,
    ...>     raw: %{"promptTokenCount" => 2, "candidatesTokenCount" => 5, "totalTokenCount" => 7}
    ...>   }
    ...> })
    iex> LangChain.Utils.TokenUsageCollector.get(collector)
    %{
      "gemini-2.0-flash" => %LangChain.TokenUsage{
        input: 4,
        output: 9,
        raw: %{"promptTokenCount" => 4, "candidatesTokenCount" => 9, "totalTokenCount" => 13}
      },
      "gemini-1.5-pro" => %LangChain.TokenUsage{
        input: 2,
        output: 5,
        raw: %{"promptTokenCount" => 2, "candidatesTokenCount" => 5, "totalTokenCount" => 7}
      }
    }

  In this example, the collector returns a map where keys are model names(e.g., "gemini-2.0-flash") and
  the values are %LangChain.Schema.TokenUsage{} structs, which contains aggregated input, output, and raw information.
  """

  use Agent

  @doc """
  Starts a new token usage collector Agent.

  Returns the PID of the Agent.
  """
  @spec new(map()) :: pid()
  def new(init_input \\ %{}) do
    {:ok, agent} = Agent.start_link(fn -> init_input end)

    agent
  end

  @doc """
  Accumulates token usage data into the collector.

  Takes the Agent PID and a map containing token usage information. The map
  should follow the structure described in the TokenUsageCollector moduledoc.

  Returns the updated token usage map.
  """
  @spec acc(pid(), map()) :: :ok
  def acc(agent, input) do
    Agent.update(agent, &(&1 |> do_acc(input)))
  end

  @doc """
  Retrieves the current token usage data from the collector.

  Returns the accumulated token usage map.
  """
  @spec get(pid()) :: map()
  def get(agent) do
    Agent.get(agent, & &1)
  end

  defp do_acc(map0, map1) do
    Map.merge(map0, map1, fn
      _key, map0, map1 when is_map(map0) and is_map(map1) ->
        do_acc(map0, map1)

      _key, value0, value1 when is_number(value0) and is_number(value1) ->
        value0 + value1

      _key, _value0, value1 ->
        # If the value is not a map or number, it just gets replaced
        value1
    end)
  end
end
