defmodule LangChain.TestingHelpers do
  @moduledoc """
  Shared testing helper functions used across test suites.
  """

  alias LangChain.Agents.FileSystemServer

  # Default test registry
  @test_registry LangChain.Test.Registry

  @doc """
  Returns the default test registry name.
  """
  def test_registry, do: @test_registry

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

  @doc """
  Helper to get a file entry from FileSystemServer's GenServer state.

  This is useful for inspecting the internal state of the filesystem in tests.

  ## Parameters

  - `agent_id` - The agent identifier
  - `path` - The file path to retrieve
  - `registry` - The registry name (optional, defaults to test registry)

  ## Returns

  The FileEntry struct or nil if not found.

  ## Examples

      entry = get_entry("agent-123", "/file.txt")
      assert entry.content == "test content"
  """
  def get_entry(agent_id, path, registry \\ @test_registry) do
    pid = FileSystemServer.whereis(registry, agent_id)
    state = :sys.get_state(pid)
    Map.get(state.files, path)
  end
end
