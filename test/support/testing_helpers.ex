defmodule LangChain.TestingHelpers do
  @doc """
  Collects all messages sent to the current test process and returns them as a
  list.

  This is useful for testing callbacks that send messages to the test process.
  It's a bit of a hack, but it's the best way I can think of to test callbacks
  that send an unspecified number of messages to the test process.
  """
  def collect_messages do
    collect_messages([])
  end

  defp collect_messages(acc) do
    receive do
      message -> collect_messages([message | acc])
    after
      0 -> Enum.reverse(acc)
    end
  end
end
