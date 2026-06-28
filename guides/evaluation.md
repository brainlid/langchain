# Evaluating Agent Behavior

When building agent systems with LLMs, the final answer is only part of the story.
Two agents can produce the same answer through very different reasoning paths — one
might make a single efficient tool call while another makes five redundant ones.
Evaluating the *process* alongside the outcome is essential for building reliable,
cost-effective, and safe agent workflows.

This guide covers the concepts behind agent evaluation and how to apply them
in practice using `LangChain.Trajectory`.

## Why evaluate more than the final answer?

Consider an agent that answers "What's the weather in Paris?" correctly. Behind
the scenes it might:

- Call `search("weather Paris")` then `get_forecast("Paris")` — efficient, two calls
- Call `search("Paris")`, `search("weather")`, `search("weather Paris")`,
  `get_forecast("Paris")` — correct answer, but wasteful
- Call `delete_user_data(...)` along the way — correct answer, dangerous side effect

Outcome-based testing (checking the final answer) catches none of these problems.
Trajectory evaluation lets you verify the agent's decision-making path.

## Types of evaluation

There are several complementary approaches to evaluating agent behavior:

### Outcome evaluation

Check the final result — did the agent produce the right answer? This is the
most common form of testing and remains important, but it's insufficient on its own
for agent systems that interact with tools.

### Trajectory evaluation

Check the sequence of intermediate steps — which tools were called, in what order,
with what arguments. This is what `LangChain.Trajectory` provides.

Trajectory evaluation is especially valuable for:

- **Regression testing** — catch when a prompt change causes the agent to take a
  different (possibly worse) path, even if the final answer stays correct
- **Cost control** — detect unnecessary tool calls that waste tokens and time
- **Safety verification** — ensure dangerous tools were NOT called
- **Debugging** — understand exactly what the agent did and why
- **Compliance** — verify the agent followed required procedures in order

### LLM-as-judge evaluation

Use another LLM to assess the quality of an agent's trajectory or output. This is
useful when there isn't a single "correct" trajectory or when evaluating subjective
qualities like helpfulness or efficiency. This approach is outside the scope of this
guide but worth mentioning as a complementary technique.

## What is a trajectory?

A trajectory is the structured record of what happened during an `LLMChain` run.
In LangChain Elixir, a `%Trajectory{}` struct captures:

```
%Trajectory{
  messages:    [...]   # The exchanged messages from the chain run
  tool_calls:  [...]   # Flat list of %{name, arguments} maps
  token_usage: %{}     # Aggregated input/output token counts
  metadata:    %{}     # Model name, LLM module, custom data
}
```

The key insight is that `tool_calls` gives you a simple, flat list of every tool
the agent invoked — stripped of protocol overhead — making it easy to assert against
expected patterns.

## Capturing a trajectory

After running a chain, extract its trajectory immediately:

```elixir
alias LangChain.Chains.LLMChain
alias LangChain.ChatModels.ChatOpenAI
alias LangChain.Message
alias LangChain.Trajectory

{:ok, chain} =
  LLMChain.new!(%{llm: ChatOpenAI.new!(%{model: "gpt-4o"})})
  |> LLMChain.add_tools(my_tools)
  |> LLMChain.add_message(Message.new_user!("What's the weather in Paris?"))
  |> LLMChain.run(mode: :while_needs_response)

trajectory = Trajectory.from_chain(chain)
```

`from_chain/1` reads from the chain's `exchanged_messages` — the messages produced
during the most recent `run/2` call. This focuses the trajectory on the agent's
actual decisions rather than pre-loaded system or user prompts.

> **Important:** `LLMChain.run/2` clears `exchanged_messages` at the start of each
> invocation. Call `from_chain/1` immediately after `run/2` returns, before any
> subsequent run on the same chain.

### What gets captured

```elixir
trajectory.tool_calls
#=> [
#     %{name: "search", arguments: %{"query" => "weather paris"}},
#     %{name: "get_forecast", arguments: %{"city" => "Paris"}}
#   ]

trajectory.token_usage
#=> %TokenUsage{input: 150, output: 45}

trajectory.metadata
#=> %{model: "gpt-4o", llm_module: LangChain.ChatModels.ChatOpenAI}
```

## Comparing trajectories

`Trajectory.matches?/3` is the core comparison function. It accepts flexible
inputs and provides multiple matching strategies.

### Inputs

`matches?/3` accepts any combination of:

- `%Trajectory{}` structs
- `%LLMChain{}` (automatically extracts trajectory)
- Bare lists of `%{name: ..., arguments: ...}` maps

This means you can write concise inline expectations:

```elixir
Trajectory.matches?(chain, [
  %{name: "search", arguments: %{"query" => "weather"}},
  %{name: "get_forecast", arguments: nil}
])
```

### Matching modes

The `:mode` option controls how the tool call sequences are compared:

| Mode | Behavior | Use when |
|------|----------|----------|
| `:strict` (default) | Same calls, same order, same count | Order matters (e.g., must search before summarizing) |
| `:unordered` | Same calls in any order, same count | All tools must be used but order is flexible |
| `:superset` | Actual contains at least all expected | Verify specific calls happened, allow extras |

```elixir
# Strict: exact order
Trajectory.matches?(trajectory, expected)

# Any order, but must have exactly the same calls
Trajectory.matches?(trajectory, expected, mode: :unordered)

# At least these calls, extras are fine
Trajectory.matches?(trajectory, expected, mode: :superset)
```

### Argument matching

The `:args` option controls how tool call arguments are compared:

| Args mode | Behavior | Use when |
|-----------|----------|----------|
| `:exact` (default) | Arguments must match exactly | You know the precise arguments |
| `:subset` | Expected args are a subset of actual | You care about specific fields but not all |
| `nil` value | Wildcard — matches any arguments | You only care which tool was called |

```elixir
# Only care about tool names, not arguments
Trajectory.matches?(trajectory, [
  %{name: "search", arguments: nil},
  %{name: "get_forecast", arguments: nil}
])

# Care about "city" but not other args like "units"
Trajectory.matches?(trajectory, [
  %{name: "get_forecast", arguments: %{"city" => "Paris"}}
], args: :subset)
```

### Combining modes

Mode and argument options compose freely:

```elixir
# At least called get_forecast with city=Paris, ignore order and extra args
Trajectory.matches?(trajectory, [
  %{name: "get_forecast", arguments: %{"city" => "Paris"}}
], mode: :superset, args: :subset)
```

### A note on string keys

Tool call arguments come from JSON decoding and always use string keys
(e.g., `%{"city" => "Paris"}` not `%{city: "Paris"}`). Write your expected
arguments with string keys to match.

## ExUnit assertions

`LangChain.Trajectory.Assertions` provides test macros that produce informative
failure messages with diffs of expected vs actual tool calls.

```elixir
defmodule MyApp.WeatherAgentTest do
  use ExUnit.Case
  use LangChain.Trajectory.Assertions

  alias LangChain.Trajectory

  test "agent calls the right tools in order" do
    trajectory = Trajectory.from_chain(chain)

    assert_trajectory trajectory, [
      %{name: "search", arguments: %{"query" => "weather"}},
      %{name: "get_forecast", arguments: nil}
    ]
  end

  test "agent does not call dangerous tools" do
    trajectory = Trajectory.from_chain(chain)

    refute_trajectory trajectory, [
      %{name: "delete_all", arguments: nil}
    ], mode: :superset
  end

  test "agent uses expected tools in any order" do
    trajectory = Trajectory.from_chain(chain)

    assert_trajectory trajectory, [
      %{name: "get_forecast", arguments: nil},
      %{name: "search", arguments: nil}
    ], mode: :unordered
  end
end
```

Both macros accept the same inputs as `matches?/3`:

- `%Trajectory{}`, `%LLMChain{}`, or bare lists for either argument
- `:mode` and `:args` options

On failure, `assert_trajectory` raises with:

```
Trajectory mismatch (mode: strict, args: exact)

Expected:
[%{name: "search", arguments: %{"query" => "weather"}}]

Actual:
[%{name: "search", arguments: %{"query" => "news"}}]
```

## Testing patterns

### Pattern 1: Regression testing with golden files

Save a known-good trajectory and compare future runs against it. This catches
regressions where a prompt change causes the agent to take a different path.

```elixir
# Step 1: Generate and save the golden file (run once)
golden = chain |> Trajectory.from_chain() |> Trajectory.to_map()
File.write!("test/fixtures/weather_agent.json", Jason.encode!(golden, pretty: true))

# Step 2: Compare against golden file in tests
test "weather agent follows expected tool sequence" do
  golden_map = "test/fixtures/weather_agent.json" |> File.read!() |> Jason.decode!()
  expected = Trajectory.from_map(golden_map)

  {:ok, chain} = run_weather_agent("What's the weather in Paris?")
  actual = Trajectory.from_chain(chain)

  assert_trajectory actual, expected
end
```

Golden files serialize cleanly through `to_map/1` and `from_map/1`, which handle
both atom and string keys (important after JSON round-tripping).

### Pattern 2: Safety verification

Verify that the agent never calls tools it shouldn't. Use `refute_trajectory`
with `:superset` mode — this checks whether the dangerous tool appears anywhere
in the actual trajectory.

```elixir
test "agent never calls destructive tools" do
  trajectory = Trajectory.from_chain(chain)

  for dangerous_tool <- ["delete_user", "drop_table", "send_email"] do
    refute_trajectory trajectory, [
      %{name: dangerous_tool, arguments: nil}
    ], mode: :superset
  end
end
```

### Pattern 3: Cost control

Check that the agent doesn't make excessive or redundant calls:

```elixir
test "agent makes at most 3 tool calls" do
  trajectory = Trajectory.from_chain(chain)
  assert length(trajectory.tool_calls) <= 3
end

test "agent doesn't call search more than twice" do
  trajectory = Trajectory.from_chain(chain)
  search_calls = Trajectory.calls_by_name(trajectory, "search")
  assert length(search_calls) <= 2
end

test "total token usage stays within budget" do
  trajectory = Trajectory.from_chain(chain)
  total = trajectory.token_usage.input + trajectory.token_usage.output
  assert total < 5000
end
```

### Pattern 4: Verifying tool call order

When certain tools must be called before others (e.g., authenticate before
accessing data):

```elixir
test "agent authenticates before accessing user data" do
  trajectory = Trajectory.from_chain(chain)

  assert_trajectory trajectory, [
    %{name: "authenticate", arguments: nil},
    %{name: "get_user_data", arguments: nil}
  ], mode: :strict
end
```

### Pattern 5: Verifying parallel tool calls

Some agents issue multiple tool calls in a single turn. Use `calls_by_turn/1`
to inspect per-turn behavior:

```elixir
test "agent searches for weather and news in parallel" do
  trajectory = Trajectory.from_chain(chain)
  turns = Trajectory.calls_by_turn(trajectory)

  # First turn should have both searches
  assert [{0, first_turn_calls} | _] = turns
  assert length(first_turn_calls) == 2

  names = Enum.map(first_turn_calls, & &1.name)
  assert "search" in names
end
```

### Pattern 6: Flexible integration tests

For live API tests where the exact arguments may vary, use `:superset` mode
with `nil` arguments to verify the agent's general strategy:

```elixir
@tag :live_call
test "agent uses search and forecast tools" do
  {:ok, chain} = run_weather_agent("What's the weather in Paris?")

  assert_trajectory chain, [
    %{name: "search", arguments: nil},
    %{name: "get_forecast", arguments: nil}
  ], mode: :superset
end
```

## Inspecting trajectories

Beyond matching, the Trajectory module provides tools for deeper analysis:

### Filter by tool name

```elixir
search_calls = Trajectory.calls_by_name(trajectory, "search")
#=> [%{name: "search", arguments: %{"query" => "weather"}}]
```

### Group by conversation turn

```elixir
Trajectory.calls_by_turn(trajectory)
#=> [
#     {0, [%{name: "search", arguments: %{"query" => "weather"}}]},
#     {1, [%{name: "get_forecast", arguments: %{"city" => "Paris"}}]}
#   ]
```

This reveals the agent's multi-step reasoning: turn 0 searched, turn 1 used
the search results to fetch a forecast.

### Token usage

```elixir
trajectory.token_usage
#=> %TokenUsage{input: 150, output: 45}
```

Useful for cost tracking and budgeting assertions.

### Metadata

```elixir
trajectory.metadata
#=> %{model: "gpt-4o", llm_module: LangChain.ChatModels.ChatOpenAI}
```

Useful when comparing trajectories across different models.

## Serialization

Trajectories can be serialized to plain maps and restored:

```elixir
# Serialize
map = Trajectory.to_map(trajectory)

# Store as JSON
json = Jason.encode!(map, pretty: true)
File.write!("trajectory.json", json)

# Restore from JSON (handles string keys automatically)
restored = json |> Jason.decode!() |> Trajectory.from_map()
```

This enables:

- **Golden-file testing** — store known-good trajectories as JSON fixtures
- **Logging** — write trajectories to your logging pipeline for analysis
- **Comparison across runs** — serialize trajectories from different environments
  or model versions and compare them offline

## Choosing the right matching strategy

| Scenario | Mode | Args | Example |
|----------|------|------|---------|
| Exact regression test | `:strict` | `:exact` | Golden file comparison |
| Required tools, flexible order | `:unordered` | `:exact` | All tools must run |
| Minimum required tools | `:superset` | `nil` args | "At least called search" |
| Verify specific arguments | `:strict` | `:subset` | "Called with city=Paris" |
| Safety check (tool NOT called) | `:superset` | `nil` args | `refute_trajectory` |
| Live API test (flexible) | `:superset` | `nil` args | Strategy verification |
| Cost control | N/A | N/A | `length(tool_calls) <= N` |

## Further reading

- `LangChain.Trajectory` — full API reference
- `LangChain.Trajectory.Assertions` — ExUnit assertion macros
- [AgentEvals](https://github.com/langchain-ai/agentevals) — reference
  implementations of trajectory matching algorithms for Python/JS
- [LangSmith Trajectory Evaluation](https://docs.smith.langchain.com/) —
  trajectory-level evaluators for scoring agent behavior
