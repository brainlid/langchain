defmodule LangChain.TestingHelpers do
  @moduledoc """
  Shared testing helper functions used across test suites.
  """

  alias LangChain.ChatModels.ChatAnthropic

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
  Basic conversion of a Message to a DisplayMessage like data map.
  """
  def message_to_display_data(%LangChain.Message{} = message) do
    %{
      content_type: "text",
      role: to_string(message.role),
      content: LangChain.Message.ContentPart.parts_to_string(message.content)
    }
  end

  # Helper to create a mock model
  def mock_model do
    ChatAnthropic.new!(%{
      model: "claude-3-5-sonnet-20241022",
      api_key: "test_key"
    })
  end
end
