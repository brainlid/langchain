defmodule LangChain.TrajectoryAssertions do
  @moduledoc """
  ExUnit assertion helpers for trajectory comparison.

  ## Usage

      use LangChain.TrajectoryAssertions

      test "agent calls the right tools" do
        trajectory = Trajectory.from_chain(chain)

        assert_trajectory trajectory, [
          %{name: "search", arguments: %{"query" => "weather"}},
          %{name: "get_forecast", arguments: nil}
        ]
      end
  """

  @doc """
  Assert that a trajectory matches the expected tool call sequence.

  Accepts the same options as `LangChain.Trajectory.match?/3`:

    * `:mode` — `:strict` (default), `:unordered`, `:superset`
    * `:args` — `:exact` (default), `:subset`

  On failure, raises `ExUnit.AssertionError` with a diff showing expected vs
  actual tool calls.
  """
  defmacro assert_trajectory(actual, expected, opts \\ []) do
    quote do
      actual_val = unquote(actual)
      expected_val = unquote(expected)
      opts_val = unquote(opts)

      unless LangChain.Trajectory.match?(actual_val, expected_val, opts_val) do
        actual_calls = LangChain.TrajectoryAssertions.extract_tool_calls(actual_val)
        expected_calls = LangChain.TrajectoryAssertions.extract_tool_calls(expected_val)

        raise ExUnit.AssertionError,
          message: """
          Trajectory mismatch (mode: #{Keyword.get(opts_val, :mode, :strict)}, args: #{Keyword.get(opts_val, :args, :exact)})

          Expected:
          #{inspect(expected_calls, pretty: true)}

          Actual:
          #{inspect(actual_calls, pretty: true)}
          """
      end
    end
  end

  @doc false
  def extract_tool_calls(%LangChain.Trajectory{tool_calls: calls}), do: calls
  def extract_tool_calls(calls) when is_list(calls), do: calls

  defmacro __using__(_opts) do
    quote do
      import LangChain.TrajectoryAssertions
    end
  end
end
