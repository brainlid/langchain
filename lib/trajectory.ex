defmodule LangChain.Trajectory do
  @moduledoc """
  Captures the structured sequence of messages and tool calls produced during
  an `LLMChain` run for inspection, serialization, and comparison.

  A trajectory provides a first-class API for extracting the decision-making
  path from a chain run — which tools were called, in what order, with what
  arguments — enabling golden-file testing, logging, and correctness assertions
  for agent workflows.

  ## Usage

      trajectory = Trajectory.from_chain(chain)

      # Serialize for logging or golden-file comparison
      map = Trajectory.to_map(trajectory)

      # Deserialize back from stored map
      trajectory = Trajectory.from_map(map)

      # Compare against expected tool call sequence
      Trajectory.matches?(trajectory, [
        %{name: "search", arguments: %{"query" => "weather"}},
        %{name: "get_forecast", arguments: nil}
      ])

      # Filter tool calls by name
      Trajectory.calls_by_name(trajectory, "search")

      # Group tool calls by conversation turn
      Trajectory.calls_by_turn(trajectory)

  ## Arguments use string keys

  Tool call arguments come from JSON decoding and use string keys
  (e.g. `%{"city" => "Paris"}` not `%{city: "Paris"}`). Expected arguments
  in `matches?/3` should use string keys as well.
  """

  alias LangChain.Chains.LLMChain
  alias LangChain.Message
  alias LangChain.Message.ContentPart
  alias LangChain.TokenUsage

  defstruct messages: [],
            tool_calls: [],
            token_usage: nil

  @type tool_call_map :: %{name: String.t(), arguments: map() | nil}

  @type t :: %__MODULE__{
          messages: [Message.t()],
          tool_calls: [tool_call_map()],
          token_usage: TokenUsage.t() | nil
        }

  @doc """
  Build a `Trajectory` from a chain's `exchanged_messages`.

  Extracts tool calls into a flat list and aggregates token usage across all
  assistant messages.

  ## Example

      trajectory = Trajectory.from_chain(chain)
  """
  @spec from_chain(LLMChain.t()) :: t()
  def from_chain(%LLMChain{exchanged_messages: messages}) do
    tool_calls = extract_tool_calls(messages)
    token_usage = aggregate_token_usage(messages)

    %__MODULE__{
      messages: messages,
      tool_calls: tool_calls,
      token_usage: token_usage
    }
  end

  @doc """
  Serialize a trajectory to plain maps for logging, storage, or golden-file
  comparison.

  Messages are converted to maps with `:role`, `:content`, `:tool_calls`, and
  `:tool_results` keys. Content is normalized to strings via
  `ContentPart.content_to_string/1`.
  """
  @spec to_map(t()) :: map()
  def to_map(%__MODULE__{} = trajectory) do
    %{
      messages: Enum.map(trajectory.messages, &message_to_map/1),
      tool_calls: trajectory.tool_calls,
      token_usage: token_usage_to_map(trajectory.token_usage)
    }
  end

  @doc """
  Deserialize a trajectory from a plain map previously produced by `to_map/1`.

  Restores `tool_calls` and `token_usage` but stores messages as raw maps
  since full `Message` struct reconstruction requires schema context that
  plain maps don't carry.

  ## Example

      map = Trajectory.to_map(trajectory)
      restored = Trajectory.from_map(map)
      restored.tool_calls == trajectory.tool_calls
  """
  @spec from_map(map()) :: t()
  def from_map(%{} = map) do
    %__MODULE__{
      messages: Map.get(map, :messages, Map.get(map, "messages", [])),
      tool_calls: normalize_tool_calls(Map.get(map, :tool_calls, Map.get(map, "tool_calls", []))),
      token_usage: normalize_token_usage(Map.get(map, :token_usage, Map.get(map, "token_usage")))
    }
  end

  @doc """
  Compare a trajectory's tool calls against an expected sequence.

  `expected` can be a `Trajectory` struct or a bare list of
  `%{name: ..., arguments: ...}` maps for inline test expectations.

  When `arguments` is `nil` in an expected entry, it matches any arguments for
  that tool name.

  ## Options

    * `:mode` — comparison mode (default `:strict`)
      * `:strict` — same tool calls in the same order and count
      * `:unordered` — same tool calls in any order
      * `:superset` — actual contains at least all expected calls

    * `:args` — argument comparison (default `:exact`)
      * `:exact` — arguments must match exactly
      * `:subset` — expected arguments are a subset of actual arguments

  ## Examples

      # Strict order and exact arguments
      Trajectory.matches?(trajectory, [
        %{name: "search", arguments: %{"query" => "weather"}}
      ])

      # Any order, ignore extra calls
      Trajectory.matches?(trajectory, expected, mode: :superset, args: :subset)
  """
  @spec matches?(t() | [tool_call_map()], t() | [tool_call_map()], keyword()) :: boolean()
  def matches?(actual, expected, opts \\ [])

  def matches?(%__MODULE__{} = actual, %__MODULE__{} = expected, opts) do
    matches?(actual.tool_calls, expected.tool_calls, opts)
  end

  def matches?(%__MODULE__{} = actual, expected, opts) when is_list(expected) do
    matches?(actual.tool_calls, expected, opts)
  end

  def matches?(actual, expected, opts) when is_list(actual) and is_list(expected) do
    mode = Keyword.get(opts, :mode, :strict)
    args_mode = Keyword.get(opts, :args, :exact)

    case mode do
      :strict -> match_strict(actual, expected, args_mode)
      :unordered -> match_unordered(actual, expected, args_mode)
      :superset -> match_superset(actual, expected, args_mode)
    end
  end

  @doc """
  Return all tool calls matching the given tool `name`.

  ## Example

      Trajectory.calls_by_name(trajectory, "search")
      #=> [%{name: "search", arguments: %{"query" => "weather"}}]
  """
  @spec calls_by_name(t(), String.t()) :: [tool_call_map()]
  def calls_by_name(%__MODULE__{tool_calls: calls}, name) do
    Enum.filter(calls, &(&1.name == name))
  end

  @doc """
  Group tool calls by conversation turn (assistant message index).

  Returns a list of `{turn_index, [tool_call_map]}` tuples where `turn_index`
  is the 0-based position of the assistant message among all assistant messages
  that contained tool calls.

  ## Example

      Trajectory.calls_by_turn(trajectory)
      #=> [{0, [%{name: "search", arguments: %{"query" => "weather"}}]},
      #    {1, [%{name: "get_forecast", arguments: %{"city" => "Paris"}}]}]
  """
  @spec calls_by_turn(t()) :: [{non_neg_integer(), [tool_call_map()]}]
  def calls_by_turn(%__MODULE__{messages: messages}) do
    messages
    |> Enum.filter(&Message.is_tool_call?/1)
    |> Enum.with_index()
    |> Enum.map(fn {msg, idx} ->
      calls =
        Enum.map(msg.tool_calls, fn tc ->
          %{name: tc.name, arguments: tc.arguments}
        end)

      {idx, calls}
    end)
  end

  # --- Private helpers ---

  defp extract_tool_calls(messages) do
    messages
    |> Enum.filter(&Message.is_tool_call?/1)
    |> Enum.flat_map(fn msg ->
      Enum.map(msg.tool_calls, fn tc ->
        %{name: tc.name, arguments: tc.arguments}
      end)
    end)
  end

  defp aggregate_token_usage(messages) do
    Enum.reduce(messages, nil, fn msg, acc ->
      case TokenUsage.get(msg) do
        nil -> acc
        usage -> TokenUsage.add(acc, usage)
      end
    end)
  end

  defp message_to_map(%Message{} = msg) do
    base = %{
      role: msg.role,
      content: ContentPart.content_to_string(msg.content)
    }

    base
    |> maybe_put_tool_calls(msg)
    |> maybe_put_tool_results(msg)
  end

  defp maybe_put_tool_calls(map, %Message{tool_calls: tool_calls})
       when is_list(tool_calls) and tool_calls != [] do
    Map.put(
      map,
      :tool_calls,
      Enum.map(tool_calls, fn tc ->
        %{name: tc.name, arguments: tc.arguments}
      end)
    )
  end

  defp maybe_put_tool_calls(map, _msg), do: map

  defp maybe_put_tool_results(map, %Message{tool_results: tool_results})
       when is_list(tool_results) and tool_results != [] do
    Map.put(
      map,
      :tool_results,
      Enum.map(tool_results, fn tr ->
        %{
          name: tr.name,
          content: ContentPart.content_to_string(tr.content),
          is_error: tr.is_error
        }
      end)
    )
  end

  defp maybe_put_tool_results(map, _msg), do: map

  defp token_usage_to_map(nil), do: nil

  defp token_usage_to_map(%TokenUsage{} = usage) do
    %{input: usage.input, output: usage.output}
  end

  defp normalize_tool_calls(calls) when is_list(calls) do
    Enum.map(calls, fn call ->
      %{
        name: Map.get(call, :name) || Map.get(call, "name"),
        arguments: Map.get(call, :arguments, Map.get(call, "arguments"))
      }
    end)
  end

  defp normalize_tool_calls(_), do: []

  defp normalize_token_usage(nil), do: nil

  defp normalize_token_usage(%TokenUsage{} = usage), do: usage

  defp normalize_token_usage(%{} = map) do
    input = Map.get(map, :input) || Map.get(map, "input")
    output = Map.get(map, :output) || Map.get(map, "output")

    if input || output do
      TokenUsage.new!(%{input: input, output: output})
    end
  end

  # Strict: same order, same count
  defp match_strict(actual, expected, args_mode) do
    length(actual) == length(expected) &&
      Enum.zip(actual, expected)
      |> Enum.all?(fn {a, e} -> call_matches?(a, e, args_mode) end)
  end

  # Unordered: same calls in any order, same count
  defp match_unordered(actual, expected, args_mode) do
    length(actual) == length(expected) &&
      all_expected_found?(actual, expected, args_mode)
  end

  # Superset: actual contains at least all expected calls
  defp match_superset(actual, expected, args_mode) do
    all_expected_found?(actual, expected, args_mode)
  end

  defp all_expected_found?(actual, expected, args_mode) do
    Enum.reduce_while(expected, actual, fn exp, remaining ->
      case find_and_remove(remaining, exp, args_mode) do
        {:ok, rest} -> {:cont, rest}
        :not_found -> {:halt, :not_found}
      end
    end) != :not_found
  end

  defp find_and_remove(list, expected, args_mode) do
    case Enum.split_while(list, fn a -> not call_matches?(a, expected, args_mode) end) do
      {_before, []} -> :not_found
      {before, [_match | rest]} -> {:ok, before ++ rest}
    end
  end

  defp call_matches?(actual, expected, args_mode) do
    actual.name == expected.name && args_match?(actual.arguments, expected.arguments, args_mode)
  end

  # nil in expected = wildcard
  defp args_match?(_actual, nil, _mode), do: true
  defp args_match?(actual, expected, :exact), do: actual == expected

  defp args_match?(actual, expected, :subset) when is_map(actual) and is_map(expected) do
    Enum.all?(expected, fn {k, v} -> Map.get(actual, k) == v end)
  end

  defp args_match?(_actual, _expected, :subset), do: false
end
