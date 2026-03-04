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

      test "agent does not call dangerous tool" do
        trajectory = Trajectory.from_chain(chain)

        refute_trajectory trajectory, [
          %{name: "delete_all", arguments: nil}
        ], mode: :superset
      end
  """

  @doc """
  Assert that a trajectory matches the expected tool call sequence.

  Accepts the same options as `LangChain.Trajectory.matches?/3`:

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

      unless LangChain.Trajectory.matches?(actual_val, expected_val, opts_val) do
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

  @doc """
  Assert that a trajectory does NOT match the expected tool call sequence.

  Useful for verifying that specific tools were not called or that a
  particular call pattern did not occur.

  Accepts the same options as `assert_trajectory/3`.

  On failure, raises `ExUnit.AssertionError` indicating an unexpected match.
  """
  defmacro refute_trajectory(actual, expected, opts \\ []) do
    quote do
      actual_val = unquote(actual)
      expected_val = unquote(expected)
      opts_val = unquote(opts)

      if LangChain.Trajectory.matches?(actual_val, expected_val, opts_val) do
        matched_calls = LangChain.TrajectoryAssertions.extract_tool_calls(expected_val)

        raise ExUnit.AssertionError,
          message: """
          Unexpected trajectory match (mode: #{Keyword.get(opts_val, :mode, :strict)}, args: #{Keyword.get(opts_val, :args, :exact)})

          Did not expect to match:
          #{inspect(matched_calls, pretty: true)}
          """
      end
    end
  end

  @doc false
  def extract_tool_calls(%LangChain.Chains.LLMChain{} = chain) do
    chain |> LangChain.Trajectory.from_chain() |> extract_tool_calls()
  end

  def extract_tool_calls(%LangChain.Trajectory{tool_calls: calls}), do: calls
  def extract_tool_calls(calls) when is_list(calls), do: calls

  defmacro __using__(_opts) do
    quote do
      import LangChain.TrajectoryAssertions
    end
  end
end
