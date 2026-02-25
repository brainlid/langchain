defmodule LangChain.OpenTelemetry.MessageSerializer do
  @moduledoc """
  Serializes LangChain `Message` structs to JSON strings following the
  `gen_ai.input.messages` / `gen_ai.output.messages` GenAI Semantic Convention schema.

  The Erlang OpenTelemetry SDK uses flat key-value attributes, so messages are
  serialized as a JSON string rather than nested structures.
  """

  alias LangChain.Message
  alias LangChain.Message.ContentPart
  alias LangChain.Message.ToolCall

  @doc """
  Serializes a list of messages to a JSON string for `gen_ai.input.messages`.

  Each message is represented as a map with `role` and `content` keys.
  Tool calls are included under the `tool_calls` key when present.
  """
  @spec serialize_input(list(Message.t())) :: String.t()
  def serialize_input(messages) when is_list(messages) do
    messages
    |> Enum.map(&serialize_message/1)
    |> Jason.encode!()
  end

  @doc """
  Serializes a result (single message or list of messages) to a JSON string
  for `gen_ai.output.messages`.
  """
  @spec serialize_output(Message.t() | list(Message.t())) :: String.t()
  def serialize_output(%Message{} = message) do
    serialize_output([message])
  end

  def serialize_output(messages) when is_list(messages) do
    messages
    |> Enum.map(&serialize_message/1)
    |> Jason.encode!()
  end

  defp serialize_message(%Message{} = msg) do
    base = %{
      "role" => to_string(msg.role),
      "content" => serialize_content(msg.content)
    }

    case msg.tool_calls do
      [_ | _] = tool_calls ->
        Map.put(base, "tool_calls", Enum.map(tool_calls, &serialize_tool_call/1))

      _ ->
        base
    end
  end

  # Single text ContentPart -> plain string for cleaner GenAI spec output
  defp serialize_content([%ContentPart{type: :text} = part]) do
    part.content
  end

  defp serialize_content(parts) when is_list(parts) do
    Enum.map(parts, &serialize_content_part/1)
  end

  defp serialize_content(text) when is_binary(text), do: text
  defp serialize_content(nil), do: nil
  defp serialize_content(other), do: inspect(other)

  defp serialize_content_part(%ContentPart{type: :text} = part) do
    %{"type" => "text", "text" => part.content}
  end

  defp serialize_content_part(%ContentPart{type: :image_url} = part) do
    %{"type" => "image_url", "url" => part.content}
  end

  defp serialize_content_part(%ContentPart{type: :image} = part) do
    %{"type" => "image", "data" => part.content, "media" => part.options[:media]}
  end

  defp serialize_content_part(%ContentPart{} = part) do
    %{"type" => to_string(part.type), "content" => part.content}
  end

  defp serialize_content_part(other), do: inspect(other)

  defp serialize_tool_call(%ToolCall{} = tc) do
    %{
      "id" => tc.call_id,
      "type" => "function",
      "function" => %{
        "name" => tc.name,
        "arguments" => serialize_arguments(tc.arguments)
      }
    }
  end

  defp serialize_arguments(args) when is_map(args), do: Jason.encode!(args)
  defp serialize_arguments(args) when is_binary(args), do: args
  defp serialize_arguments(nil), do: "{}"
  defp serialize_arguments(args), do: inspect(args)
end
