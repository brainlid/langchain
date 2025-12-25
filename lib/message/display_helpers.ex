defmodule LangChain.Message.DisplayHelpers do
  @moduledoc """
  Utilities for extracting displayable content from LangChain Messages.

  These helpers bridge the gap between LangChain Message structs and
  application-specific display schemas. They handle the complexity of:
  - Extracting text, thinking, tool_calls, and tool_results
  - Proper sequencing when a single Message contains multiple display items
  - Converting structs to maps with string keys (JSON-compatible)

  ## Usage in Generated Code

  Mix task templates should demonstrate this pattern:

      # In your LiveView or context module
      def persist_message(%Message{} = message, conversation_id) do
        message
        |> DisplayHelpers.extract_display_items()
        |> Enum.with_index()
        |> Enum.map(fn {item, sequence} ->
          attrs = Map.put(item, "sequence", sequence)
          create_display_message(conversation_id, attrs)
        end)
      end

  This gives users full control over their schema while providing
  library utilities that handle the extraction complexity.
  """

  alias LangChain.Message

  @doc """
  Extracts all displayable items from a Message.

  Returns a list of maps, each representing one displayable item.
  A single Message can produce multiple items (e.g., text + tool_calls).

  ## Return Format

  Each map contains atom keys:
  - `:type` - One of: :text, :thinking, :tool_call, :tool_result
  - `:message_type` - Role-based: :user, :assistant, :tool, :system
  - `:content` - Map with type-specific content (string keys for JSONB storage)

  The order of items in the list represents the display order.
  The caller should assign sequence numbers (0, 1, 2, ...) when persisting.

  **Note**: No mixed maps - top-level keys are atoms, content payload uses string keys.

  ## Examples

      # Assistant message with text and tool calls
      message = Message.new_assistant!(%{
        content: [ContentPart.text!("Let me search...")],
        tool_calls: [
          ToolCall.new!(%{call_id: "1", name: "search", arguments: %{q: "elixir"}}),
          ToolCall.new!(%{call_id: "2", name: "weather", arguments: %{city: "NYC"}})
        ]
      })

      DisplayHelpers.extract_display_items(message)
      # => [
      #   %{type: :text, message_type: :assistant, content: %{"text" => "Let me search..."}},
      #   %{type: :tool_call, message_type: :assistant, content: %{"call_id" => "1", "name" => "search", "arguments" => %{q: "elixir"}}},
      #   %{type: :tool_call, message_type: :assistant, content: %{"call_id" => "2", "name" => "weather", "arguments" => %{city: "NYC"}}}
      # ]

      # Tool result message with multiple results
      message = Message.new_tool_result!(%{
        tool_results: [
          ToolResult.new!(%{tool_call_id: "1", name: "search", content: "Found...", is_error: false}),
          ToolResult.new!(%{tool_call_id: "2", name: "weather", content: "Sunny", is_error: false})
        ]
      })

      DisplayHelpers.extract_display_items(message)
      # => [
      #   %{type: :tool_result, message_type: :tool, content: %{"tool_call_id" => "1", "name" => "search", "content" => "Found...", "is_error" => false}},
      #   %{type: :tool_result, message_type: :tool, content: %{"tool_call_id" => "2", "name" => "weather", "content" => "Sunny", "is_error" => false}}
      # ]
  """
  @spec extract_display_items(Message.t()) :: [map()]
  def extract_display_items(%Message{} = message) do
    items = []

    # Extract text/thinking content if present
    items = items ++ extract_content_items(message)

    # Extract tool_calls if present (assistant messages)
    items = items ++ extract_tool_call_items(message)

    # Extract tool_results if present (tool messages)
    items = items ++ extract_tool_result_items(message)

    items
  end

  # Extract text and thinking content from message.content
  defp extract_content_items(%Message{content: content, role: role}) do
    message_type = role_to_message_type(role)

    case content do
      # String content (simple text)
      text when is_binary(text) and text != "" ->
        [%{
          type: :text,
          message_type: message_type,
          content: %{"text" => text}
        }]

      # List of ContentParts (text, thinking, etc.)
      parts when is_list(parts) ->
        parts
        |> Enum.filter(fn part -> part.type in [:text, :thinking] end)
        |> Enum.reject(fn part -> is_nil(part.content) or part.content == "" end)
        |> Enum.map(fn part ->
          %{
            type: part.type,
            message_type: message_type,
            content: %{"text" => part.content}
          }
        end)

      _ ->
        []
    end
  end

  # Extract tool_calls into display items
  defp extract_tool_call_items(%Message{tool_calls: tool_calls, role: role})
       when is_list(tool_calls) and tool_calls != [] do
    message_type = role_to_message_type(role)

    Enum.map(tool_calls, fn tool_call ->
      %{
        type: :tool_call,
        message_type: message_type,
        content: %{
          "call_id" => tool_call.call_id,
          "name" => tool_call.name,
          "arguments" => tool_call.arguments
        }
      }
    end)
  end

  defp extract_tool_call_items(_message), do: []

  # Extract tool_results into display items
  defp extract_tool_result_items(%Message{tool_results: tool_results, role: role})
       when is_list(tool_results) and tool_results != [] do
    message_type = role_to_message_type(role)

    Enum.map(tool_results, fn tool_result ->
      # Extract content as string - it may be a list of ContentParts or a string
      content_str = extract_tool_result_content(tool_result.content)

      %{
        type: :tool_result,
        message_type: message_type,
        content: %{
          "tool_call_id" => tool_result.tool_call_id,
          "name" => tool_result.name,
          "content" => content_str,
          "is_error" => tool_result.is_error
        }
      }
    end)
  end

  defp extract_tool_result_items(_message), do: []

  # Extract tool result content as a string
  # ToolResult.content can be a string or a list of ContentParts
  defp extract_tool_result_content(content) when is_binary(content), do: content

  defp extract_tool_result_content(content) when is_list(content) do
    # Extract text from ContentParts
    content
    |> Enum.filter(fn part -> part.type == :text end)
    |> Enum.map(fn part -> part.content end)
    |> Enum.join("\n")
  end

  defp extract_tool_result_content(_), do: ""

  # Convert Message role to display message_type atom
  defp role_to_message_type(:system), do: :system
  defp role_to_message_type(:user), do: :user
  defp role_to_message_type(:assistant), do: :assistant
  defp role_to_message_type(:tool), do: :tool
end
