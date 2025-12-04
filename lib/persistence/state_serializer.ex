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

  ## Versioning

  The serialized state includes a version field that allows for future
  migrations of the state format. The current version is 1.
  """

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
  """
  def serialize_server_state(%Agent{} = agent, %State{} = state) do
    %{
      "version" => @current_version,
      "agent_id" => agent.agent_id,
      "state" => serialize_state(state),
      "agent_config" => serialize_agent_config(agent),
      "serialized_at" => DateTime.utc_now() |> DateTime.to_iso8601()
    }
  end

  @doc """
  Deserializes a map with string keys into AgentServer state components.

  Returns `{:ok, {agent, state}}` on success or `{:error, reason}` on failure.

  ## Options

  - `:custom_tools` - Map of tool name to LangChain.Function struct for custom tools
  """
  def deserialize_server_state(data, opts \\ []) when is_map(data) do
    # Handle version migration if needed
    data = maybe_migrate(data)

    custom_tools = Keyword.get(opts, :custom_tools, %{})

    with {:ok, state} <- deserialize_state(data["state"]),
         {:ok, agent} <-
           deserialize_agent_config(data["agent_config"], data["agent_id"], custom_tools) do
      {:ok, {agent, state}}
    else
      {:error, reason} -> {:error, reason}
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

  Returns `{:ok, state}` on success or `{:error, reason}` on failure.
  """
  def deserialize_state(data) when is_map(data) do
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

    case State.new(%{messages: messages, todos: todos, metadata: metadata}) do
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
      "options" => part.options || []
    }
    |> Enum.reject(fn {_k, v} -> is_nil(v) or v == [] end)
    |> Map.new()
  end

  defp deserialize_content_part(part) when is_map(part) do
    case ContentPart.new(%{
           type: String.to_existing_atom(part["type"] || "text"),
           content: part["content"],
           options: part["options"] || []
         }) do
      {:ok, content_part} -> content_part
      {:error, _} -> raise "Failed to deserialize content part: #{inspect(part)}"
    end
  end

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

  defp serialize_agent_config(%Agent{} = agent) do
    # Identify middleware tool names
    middleware_tool_names =
      agent.middleware
      |> Enum.flat_map(&LangChain.Agents.Middleware.get_tools/1)
      |> Enum.map(& &1.name)
      |> MapSet.new()

    # Extract custom tool names (not from middleware)
    custom_tool_names =
      agent.tools
      |> Enum.reject(fn tool -> MapSet.member?(middleware_tool_names, tool.name) end)
      |> Enum.map(& &1.name)

    %{
      "agent_id" => agent.agent_id,
      "model" => serialize_model(agent.model),
      "base_system_prompt" => agent.base_system_prompt,
      "custom_tool_names" => custom_tool_names,
      "middleware" => Enum.map(agent.middleware || [], &serialize_middleware/1),
      "name" => agent.name
    }
    |> Enum.reject(fn {_k, v} -> is_nil(v) or v == [] end)
    |> Map.new()
  end

  defp deserialize_agent_config(nil, agent_id, _custom_tools) do
    {:error, "agent_config is nil for agent #{agent_id}"}
  end

  defp deserialize_agent_config(data, agent_id, custom_tools) when is_map(data) do
    require Logger

    # Handle backward compatibility - if base_system_prompt not present, use system_prompt
    base_prompt = data["base_system_prompt"] || ""

    # Lookup custom tools by name
    custom_tool_names = data["custom_tool_names"] || []

    {resolved_tools, missing_tools} =
      Enum.split_with(custom_tool_names, fn name ->
        Map.has_key?(custom_tools, name)
      end)

    # Log warning for missing tools
    if missing_tools != [] do
      Logger.warning(
        "Agent #{agent_id} is missing custom tools: #{inspect(missing_tools)}. " <>
          "Provide these via custom_tools option to restore full functionality."
      )
    end

    # Get resolved tool structs
    base_tools = Enum.map(resolved_tools, fn name -> Map.get(custom_tools, name) end)

    attrs = %{
      agent_id: agent_id || data["agent_id"],
      model: deserialize_model(data["model"]),
      base_system_prompt: base_prompt,
      tools: base_tools,
      middleware: Enum.map(data["middleware"] || [], &deserialize_middleware/1),
      name: data["name"]
    }

    case Agent.new(attrs) do
      {:ok, agent} -> {:ok, agent}
      {:error, changeset} -> {:error, {:invalid_agent, changeset}}
    end
  end

  defp serialize_model(model) when is_struct(model) do
    %{
      "module" => to_string(model.__struct__),
      "model" => model.model
      # Never serialize API keys
    }
  end

  defp deserialize_model(%{"module" => module_name} = data) do
    # Convert module string to actual module
    module = String.to_existing_atom(module_name)

    # Create a new model instance
    # Note: API keys must be provided from config/environment
    case apply(module, :new, [%{model: data["model"]}]) do
      {:ok, model} -> model
      {:error, _} -> raise "Failed to deserialize model: #{inspect(data)}"
    end
  end

  defp serialize_middleware({module, opts}) when is_atom(module) do
    serialized_opts =
      cond do
        is_map(opts) -> serialize_map_to_string_keys(opts)
        is_list(opts) -> opts
        true -> %{}
      end

    %{
      "module" => to_string(module),
      "opts" => serialized_opts
    }
  end

  defp deserialize_middleware(%{"module" => module_name} = data) do
    module = String.to_existing_atom(module_name)
    opts = data["opts"] || %{}
    {module, opts}
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
