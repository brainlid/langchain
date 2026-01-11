defmodule LangChain.Persistence.StateSerializer do
  @moduledoc """
  Handles serialization and deserialization of AgentServer state.

  Automatically handles:
  - Version migrations
  - Non-serializable data (PIDs, refs)
  - Consistent use of string keys (for JSONB compatibility)

  ## String Keys

  NOTE: This serializer uses string keys throughout to match PostgreSQL JSONB
  behavior. When JSONB maps are restored from the database, they will have
  string keys, not atoms. For consistency, the serializer exports to string keys
  and expects string keys on import.

  ## Agent Configuration vs Conversation State

  StateSerializer follows the principle: **Code lives in code. Data lives in data.**

  **What is serialized**:
  - ✅ Conversation state: messages, todos, metadata

  **What is NOT serialized**:
  - ❌ Agent configuration: middleware, tools, model
  - ❌ Runtime identifiers: agent_id

  Agent capabilities (middleware, tools, model) are code-defined by your application.
  When restoring a conversation, create the agent from your application code and
  restore only the conversation state.

  See `AgentServer` module documentation for complete restoration examples.

  ## Versioning

  The serialized state includes a version field that allows for future
  migrations of the state format. The current version is 1.
  """
  require Logger

  alias LangChain.Agents.{State, Agent, Todo}
  alias LangChain.Message
  alias LangChain.Message.{ContentPart, ToolCall, ToolResult}

  @current_version 1

  @doc """
  Get the current serialization format version.
  """
  def current_version, do: @current_version

  @doc """
  Serializes AgentServer state to a map with string keys.

  Returns a map with string keys suitable for storage in JSONB columns.

  Note: The `agent_id` is NOT included in the serialized state. The agent_id
  is a runtime identifier used for process registration and PubSub topics, not
  part of the conversation state. When restoring state, you must provide the
  agent_id to `deserialize_server_state/2` or `AgentServer.start_link_from_state/2`.
  """
  def serialize_server_state(%Agent{} = _agent, %State{} = state) do
    %{
      # Version stays at 1, but now only contains state
      "version" => @current_version,
      "state" => serialize_state(state),
      "serialized_at" => DateTime.utc_now() |> DateTime.to_iso8601()
    }
  end

  @doc """
  Deserializes a serialized state map back to a State struct.

  The agent configuration is NOT deserialized - agents must be created
  from your application code. This function only restores conversation
  state (messages, metadata).

  ## Parameters

  - `agent_id` - The agent_id to use (NOT serialized, provided by you)
  - `data` - The serialized state map (from JSONB or export)
  - `opts` - Options (currently unused, kept for compatibility)

  ## Returns

  - `{:ok, state}` - Successfully deserialized state
  - `{:error, reason}` - Deserialization failed
  """
  def deserialize_server_state(agent_id, data, _opts \\ [])
      when is_binary(agent_id) and is_map(data)
      when is_map(data) and is_binary(agent_id) do
    # Handle version migration if needed
    case maybe_migrate(data) do
      {:error, reason} ->
        {:error, reason}

      migrated_data ->
        # Only deserialize state, not agent config
        deserialize_state(agent_id, migrated_data["state"])
    end
  end

  @doc """
  Serializes a State struct to a map with string keys.
  """
  def serialize_state(%State{} = state) do
    %{
      "messages" => Enum.map(state.messages, &serialize_message/1),
      "todos" => Enum.map(state.todos, &Todo.to_map/1),
      "metadata" => serialize_map_to_string_keys(state.metadata)
    }
  end

  @doc """
  Deserializes a map with string keys into a State struct.

  The `agent_id` must be provided as it is not part of the serialized data
  (it's a runtime identifier for process registration and PubSub).

  Returns `{:ok, state}` on success or `{:error, reason}` on failure.
  """
  def deserialize_state(agent_id, data) when is_binary(agent_id) and is_map(data) do
    messages =
      case data["messages"] do
        messages when is_list(messages) ->
          Enum.map(messages, &deserialize_message/1)

        _ ->
          []
      end

    todos =
      case data["todos"] do
        todos when is_list(todos) ->
          Enum.map(todos, fn todo_map ->
            {:ok, todo} = Todo.from_map(todo_map)
            todo
          end)

        _ ->
          []
      end

    metadata =
      case data["metadata"] do
        metadata when is_map(metadata) -> metadata
        _ -> %{}
      end

    case State.new(%{agent_id: agent_id, messages: messages, todos: todos, metadata: metadata}) do
      {:ok, state} -> {:ok, state}
      {:error, changeset} -> {:error, {:invalid_state, changeset}}
    end
  end

  # Private Functions

  defp serialize_message(%Message{} = message) do
    base = %{
      "role" => to_string(message.role),
      "content" => serialize_content(message.content),
      "status" => if(message.status, do: to_string(message.status), else: nil)
    }

    # Add optional fields if present
    base = maybe_add_string_field(base, "name", message.name)
    base = maybe_add_string_field(base, "index", message.index)

    # Add tool_calls if present
    base =
      if message.tool_calls && length(message.tool_calls) > 0 do
        Map.put(base, "tool_calls", Enum.map(message.tool_calls, &serialize_tool_call/1))
      else
        base
      end

    # Add tool_results if present
    base =
      if message.tool_results && length(message.tool_results) > 0 do
        Map.put(base, "tool_results", Enum.map(message.tool_results, &serialize_tool_result/1))
      else
        base
      end

    base
  end

  defp deserialize_message(data) when is_map(data) do
    # Convert string keys to atom keys for Message.new
    attrs = %{
      role: String.to_existing_atom(data["role"] || "user"),
      content: deserialize_content(data["content"]),
      status: if(data["status"], do: String.to_existing_atom(data["status"]), else: nil)
    }

    # Add optional fields
    attrs = maybe_add_field(attrs, :name, data["name"])
    attrs = maybe_add_field(attrs, :index, data["index"])

    # Add tool_calls if present
    attrs =
      if data["tool_calls"] do
        Map.put(attrs, :tool_calls, Enum.map(data["tool_calls"], &deserialize_tool_call/1))
      else
        attrs
      end

    # Add tool_results if present
    attrs =
      if data["tool_results"] do
        Map.put(attrs, :tool_results, Enum.map(data["tool_results"], &deserialize_tool_result/1))
      else
        attrs
      end

    # Use Message.new! to create the message
    case Message.new(attrs) do
      {:ok, message} -> message
      {:error, _changeset} -> raise "Failed to deserialize message: #{inspect(data)}"
    end
  end

  defp serialize_content(content) when is_binary(content), do: content

  defp serialize_content(content) when is_list(content) do
    Enum.map(content, &serialize_content_part/1)
  end

  defp serialize_content(nil), do: nil

  defp deserialize_content(content) when is_binary(content), do: content

  defp deserialize_content(content) when is_list(content) do
    Enum.map(content, &deserialize_content_part/1)
  end

  defp deserialize_content(nil), do: nil

  defp serialize_content_part(%ContentPart{} = part) do
    %{
      "type" => to_string(part.type),
      "content" => part.content,
      "options" => serialize_options(part.options)
    }
    |> Enum.reject(fn {_k, v} -> is_nil(v) or v == [] or v == %{} end)
    |> Map.new()
  end

  # Convert options keyword list to a map with string keys for JSON serialization
  defp serialize_options(opts) when is_list(opts) and opts != [] do
    # Check if it's a keyword list (list of 2-tuples with atom keys)
    if Keyword.keyword?(opts) do
      opts
      |> Enum.map(fn {k, v} -> {to_string(k), v} end)
      |> Map.new()
    else
      opts
    end
  end

  defp serialize_options(opts) when is_map(opts), do: opts
  defp serialize_options(_), do: %{}

  defp deserialize_content_part(part) when is_map(part) do
    case ContentPart.new(%{
           type: String.to_existing_atom(part["type"] || "text"),
           content: part["content"],
           options: deserialize_options(part["options"])
         }) do
      {:ok, content_part} -> content_part
      {:error, _} -> raise "Failed to deserialize content part: #{inspect(part)}"
    end
  end

  # Convert options map with string keys back to keyword list with atom keys
  defp deserialize_options(opts) when is_map(opts) and map_size(opts) > 0 do
    opts
    |> Enum.map(fn {k, v} -> {String.to_atom(k), v} end)
    |> Keyword.new()
  end

  defp deserialize_options(opts) when is_list(opts), do: opts
  defp deserialize_options(_), do: []

  defp serialize_tool_call(%ToolCall{} = tool_call) do
    %{
      "call_id" => tool_call.call_id,
      "type" => to_string(tool_call.type),
      "name" => tool_call.name,
      "arguments" => serialize_map_to_string_keys(tool_call.arguments || %{}),
      "status" => to_string(tool_call.status)
    }
    |> Enum.reject(fn {_k, v} -> is_nil(v) end)
    |> Map.new()
  end

  defp deserialize_tool_call(data) when is_map(data) do
    case ToolCall.new(%{
           call_id: data["call_id"],
           type: String.to_existing_atom(data["type"] || "function"),
           name: data["name"],
           arguments: data["arguments"] || %{},
           status: String.to_existing_atom(data["status"] || "incomplete")
         }) do
      {:ok, tool_call} -> tool_call
      {:error, _} -> raise "Failed to deserialize tool call: #{inspect(data)}"
    end
  end

  defp serialize_tool_result(%ToolResult{} = tool_result) do
    %{
      "type" => to_string(tool_result.type),
      "tool_call_id" => tool_result.tool_call_id,
      "name" => tool_result.name,
      "content" => serialize_content(tool_result.content),
      "is_error" => tool_result.is_error
    }
    |> Enum.reject(fn {_k, v} -> is_nil(v) end)
    |> Map.new()
  end

  defp deserialize_tool_result(data) when is_map(data) do
    case ToolResult.new(%{
           type: String.to_existing_atom(data["type"] || "function"),
           tool_call_id: data["tool_call_id"],
           name: data["name"],
           content: deserialize_content(data["content"]),
           is_error: data["is_error"] || false
         }) do
      {:ok, tool_result} -> tool_result
      {:error, _} -> raise "Failed to deserialize tool result: #{inspect(data)}"
    end
  end

  defp serialize_map_to_string_keys(map) when is_map(map) do
    Map.new(map, fn
      {k, v} when is_atom(k) -> {to_string(k), serialize_value(v)}
      {k, v} when is_binary(k) -> {k, serialize_value(v)}
    end)
  end

  defp serialize_value(value) when is_struct(value) do
    # For structs, we'll just use their inspect representation as a string
    # This is not ideal for deserialization, but complex structs in middleware opts
    # are not typically needed for state restoration
    inspect(value)
  end

  defp serialize_value(value) when is_map(value), do: serialize_map_to_string_keys(value)
  defp serialize_value(value) when is_list(value), do: Enum.map(value, &serialize_value/1)
  defp serialize_value(value) when is_atom(value) and not is_nil(value), do: to_string(value)

  # Functions cannot be serialized to JSON - skip them
  # They will need to be recreated during deserialization from config
  defp serialize_value(value) when is_function(value), do: nil

  defp serialize_value(value), do: value

  defp maybe_add_string_field(map, _key, nil), do: map
  defp maybe_add_string_field(map, key, value), do: Map.put(map, key, value)

  defp maybe_add_field(map, _key, nil), do: map
  defp maybe_add_field(map, key, value), do: Map.put(map, key, value)

  defp maybe_migrate(%{"version" => 1} = data), do: data

  defp maybe_migrate(%{"version" => _other} = data) do
    # Future: implement migration from older versions
    # For now, just pass through
    data
  end

  defp maybe_migrate(data) do
    # No version field, assume version 1
    Map.put(data, "version", 1)
  end
end
