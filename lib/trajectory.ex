defmodule LangChain.Trajectory do
  @moduledoc """
  Captures the structured sequence of messages and tool calls produced during
  an `LLMChain` run for inspection, serialization, and comparison.

  A trajectory provides a first-class API for extracting the decision-making
  path from a chain run — which tools were called, in what order, with what
  arguments — enabling golden-file testing, logging, and correctness assertions
  for agent workflows.

  ## Why trajectories matter

  When building agent systems, the final answer is only part of the story.
  Two agents can produce the same answer through very different reasoning
  paths — one might make a single efficient tool call while another makes
  five redundant ones. Trajectories let you evaluate the *process*, not just
  the outcome.

  This is especially important for:

  - **Regression testing** — catch when a prompt change causes the agent to
    take a different (possibly worse) path even if the final answer is correct
  - **Cost control** — detect unnecessary tool calls that waste tokens and time
  - **Safety** — verify that dangerous tools were NOT called
  - **Debugging** — understand exactly what the agent did and why

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

  ## Metadata

  Each trajectory captures metadata about the chain run including the model
  name and LLM module. You can also add custom metadata:

      trajectory = Trajectory.from_chain(chain)
      trajectory.metadata
      #=> %{model: "gpt-4", llm_module: LangChain.ChatModels.ChatOpenAI}

  ## Evaluation patterns

  ### Golden-file testing

  Save a known-good trajectory and compare future runs against it:

      # Save the golden file
      golden = chain |> Trajectory.from_chain() |> Trajectory.to_map()
      File.write!("test/fixtures/weather_agent.json", Jason.encode!(golden))

      # In your test
      golden = "test/fixtures/weather_agent.json" |> File.read!() |> Jason.decode!()
      expected = Trajectory.from_map(golden)
      actual = Trajectory.from_chain(chain)
      assert Trajectory.matches?(actual, expected)

  ### Verifying tools were NOT called

  Use `refute` with superset mode to ensure dangerous tools weren't invoked:

      # Using Trajectory.Assertions
      use LangChain.Trajectory.Assertions

      refute_trajectory trajectory, [
        %{name: "delete_all", arguments: nil}
      ], mode: :superset

  ### Flexible matching

  When you care about *which* tools were called but not exact arguments:

      Trajectory.matches?(trajectory, [
        %{name: "search", arguments: nil},
        %{name: "summarize", arguments: nil}
      ])

  When you care that certain tools were called but allow extra calls:

      Trajectory.matches?(trajectory, [
        %{name: "search", arguments: nil}
      ], mode: :superset)

  ## Comparison modes

  The `matches?/3` function supports three modes via the `:mode` option:

  - `:strict` (default) — same tool calls in the same order and count
  - `:unordered` — same tool calls in any order, same count
  - `:superset` — actual contains at least all expected calls

  And two argument comparison modes via the `:args` option:

  - `:exact` (default) — arguments must match exactly
  - `:subset` — expected arguments must be a subset of actual arguments

  ## External references

  For more on trajectory-based evaluation of agent systems, see:

  - [LangSmith Trajectory Evaluation](https://docs.smith.langchain.com/) —
    trajectory-level evaluators for scoring agent behavior
  - [AgentEvals](https://github.com/langchain-ai/agentevals) — reference
    implementations of trajectory matching algorithms

  ## Arguments use string keys

  Tool call arguments come from JSON decoding and use string keys
  (e.g. `%{"city" => "Paris"}` not `%{city: "Paris"}`). Expected arguments
  in `matches?/3` should use string keys as well.
  """

  alias __MODULE__
  alias LangChain.Chains.LLMChain
  alias LangChain.Message
  alias LangChain.Message.ContentPart
  alias LangChain.TokenUsage

  defstruct messages: [],
            tool_calls: [],
            token_usage: nil,
            metadata: %{}

  @type tool_call_map :: %{name: String.t(), arguments: map() | nil}

  @type t :: %Trajectory{
          messages: [Message.t()],
          tool_calls: [tool_call_map()],
          token_usage: TokenUsage.t() | nil,
          metadata: map()
        }

  @doc """
  Build a `Trajectory` from a chain's `exchanged_messages`.

  Uses `exchanged_messages` — the messages added during the chain run — rather
  than `messages` which includes pre-loaded system and user messages. This
  focuses the trajectory on the agent's actual decision-making path.

  Extracts tool calls into a flat list and aggregates token usage across all
  assistant messages.

  ## Important: call immediately after `run/2`

  `LLMChain.run/2` clears `exchanged_messages` at the start of each
  invocation. This means `from_chain/1` captures only the messages from the
  **most recent** `run` call. If you need to capture a trajectory, call
  `from_chain/1` immediately after `run/2` returns — before any subsequent
  `run` call on the same chain.

  ## Example

      trajectory = Trajectory.from_chain(chain)
  """
  @spec from_chain(LLMChain.t()) :: t()
  def from_chain(%LLMChain{exchanged_messages: messages, llm: llm}) do
    tool_calls = extract_tool_calls(messages)
    token_usage = aggregate_token_usage(messages)
    metadata = extract_metadata(llm)

    %Trajectory{
      messages: messages,
      tool_calls: tool_calls,
      token_usage: token_usage,
      metadata: metadata
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
  def to_map(%Trajectory{} = trajectory) do
    %{
      messages: Enum.map(trajectory.messages, &message_to_map/1),
      tool_calls: trajectory.tool_calls,
      token_usage: token_usage_to_map(trajectory.token_usage),
      metadata: trajectory.metadata
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
    %Trajectory{
      messages: Map.get(map, :messages, Map.get(map, "messages", [])),
      tool_calls: normalize_tool_calls(Map.get(map, :tool_calls, Map.get(map, "tool_calls", []))),
      token_usage: normalize_token_usage(Map.get(map, :token_usage, Map.get(map, "token_usage"))),
      metadata: Map.get(map, :metadata, Map.get(map, "metadata", %{}))
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
  @spec matches?(t() | LLMChain.t() | [tool_call_map()], t() | [tool_call_map()], keyword()) ::
          boolean()
  def matches?(actual, expected, opts \\ [])

  def matches?(%LLMChain{} = chain, expected, opts) do
    matches?(from_chain(chain), expected, opts)
  end

  def matches?(%Trajectory{} = actual, %Trajectory{} = expected, opts) do
    matches?(actual.tool_calls, expected.tool_calls, opts)
  end

  def matches?(%Trajectory{} = actual, expected, opts) when is_list(expected) do
    matches?(actual.tool_calls, expected, opts)
  end

  def matches?(actual, %Trajectory{} = expected, opts) when is_list(actual) do
    matches?(actual, expected.tool_calls, opts)
  end

  def matches?(actual, expected, opts) when is_list(actual) and is_list(expected) do
    mode = Keyword.get(opts, :mode, :strict)
    args_mode = Keyword.get(opts, :args, :exact)

    unless args_mode in [:exact, :subset] do
      raise ArgumentError,
            "unknown args mode: #{inspect(args_mode)}, expected :exact or :subset"
    end

    case mode do
      :strict ->
        match_strict(actual, expected, args_mode)

      :unordered ->
        match_unordered(actual, expected, args_mode)

      :superset ->
        match_superset(actual, expected, args_mode)

      other ->
        raise ArgumentError,
              "unknown mode: #{inspect(other)}, expected :strict, :unordered, or :superset"
    end
  end

  @doc """
  Return all tool calls matching the given tool `name`.

  ## Example

      Trajectory.calls_by_name(trajectory, "search")
      #=> [%{name: "search", arguments: %{"query" => "weather"}}]
  """
  @spec calls_by_name(t(), String.t()) :: [tool_call_map()]
  def calls_by_name(%Trajectory{tool_calls: calls}, name) do
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
  def calls_by_turn(%Trajectory{messages: messages}) do
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

  defp extract_metadata(llm) when is_struct(llm) do
    %{
      model: Map.get(llm, :model),
      llm_module: llm.__struct__
    }
  end

  defp extract_metadata(_llm), do: %{}

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

  # Passthrough for raw maps (e.g. from from_map/1 deserialization)
  defp message_to_map(%{} = raw_map) do
    raw_map
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
    input = Map.get(map, :input, Map.get(map, "input"))
    output = Map.get(map, :output, Map.get(map, "output"))

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
