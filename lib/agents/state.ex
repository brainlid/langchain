defmodule LangChain.Agents.State do
  @moduledoc """
  Agent state structure for managing agent execution context.

  The state holds the complete context for an agent execution including:
  - Message history (list of `LangChain.Message` structs)
  - TODO list
  - Metadata

  **Note**: Files are managed separately by `FileSystemServer` and are not part of
  the agent's internal state. The FileSystemServer provides persistent storage with
  ETS and optional backend persistence (disk, database, S3, etc.).

  ## agent_id Management (Automatic)

  The `agent_id` field is a **runtime identifier** used for process registration and
  coordination. You don't need to set it when creating states—the library
  automatically injects it when you call `Agent.execute/2`, `Agent.resume/3`, or
  start an `AgentServer`.

  ### Why it's automatic

  The `agent_id` flows from the Agent struct (which is configuration) to the State
  (which is data). Making you synchronize them manually could be error-prone.

  ### For middleware developers

  When middleware receives state in hooks (`before_model`, `after_model`), the
  `agent_id` will already be set. If you create new state structs in middleware,
  copy the `agent_id` from the incoming state:

      def after_model(state, config) do
        updated_state = State.new!(%{
          agent_id: state.agent_id,  # Copy from incoming state
          messages: new_messages
        })
        {:ok, updated_state}
      end

  ## State Merging

  State merging follows specific rules:

  - **messages**: Appends new messages to existing list
  - **todos**: Replaces with new todos (merge handled by TodoList middleware)
  - **metadata**: Deep merges metadata maps
  - **agent_id**: Uses right if present, otherwise left (runtime identifier, not data)
  """

  use Ecto.Schema
  import Ecto.Changeset
  alias __MODULE__

  @primary_key false
  embedded_schema do
    # Agent identifier (set automatically by AgentServer during initialization)
    field :agent_id, :string
    # List of LangChain.Message structs
    field :messages, {:array, :any}, default: [], virtual: true
    field :todos, {:array, :map}, default: []
    field :metadata, :map, default: %{}
    # Interrupt data for HumanInTheLoop middleware
    field :interrupt_data, :map, default: nil, virtual: true
  end

  @type t :: %State{}

  @doc """
  Create a new agent state.

  Note: The `agent_id` field is optional when creating a state. The library
  automatically injects it when the state is passed to Agent.execute or
  AgentServer. This eliminates the need for manual agent_id synchronization.

  ## Examples

      # Create state without agent_id (recommended)
      state = State.new!(%{messages: [message]})

      # Library injects agent_id automatically
      {:ok, result_state} = Agent.execute(agent, state)
  """
  def new(attrs \\ %{}) do
    %State{}
    |> cast(attrs, [:agent_id, :messages, :todos, :metadata, :interrupt_data])
    |> apply_action(:insert)
  end

  @doc """
  Create a new agent state, raising on error.
  """
  def new!(attrs \\ %{}) do
    case new(attrs) do
      {:ok, state} -> state
      {:error, changeset} -> raise LangChain.LangChainError, changeset
    end
  end

  @doc """
  Deserializes state data from export_state/1.

  This is a convenience wrapper around StateSerializer.deserialize_state/2.

  **Important**: The `agent_id` is NOT serialized (it's a runtime identifier),
  so you MUST provide it when deserializing. This ensures the state can properly
  interact with AgentServer and middleware that rely on the agent_id.

  ## Examples

      # Load from database
      {:ok, state_data} = load_from_db(conversation_id)

      # Deserialize with agent_id
      {:ok, state} = State.from_serialized("my-agent-123", state_data["state"])

  ## Parameters

    - `agent_id` - The agent_id to use for this state (required)
    - `data` - The serialized state map (the "state" field from export_state)

  ## Returns

    - `{:ok, state}` - Successfully deserialized with agent_id set
    - `{:error, reason}` - Deserialization failed
  """
  def from_serialized(agent_id, data) when is_binary(agent_id) and is_map(data) do
    LangChain.Persistence.StateSerializer.deserialize_state(agent_id, data)
  end

  @doc """
  Merge two states together.

  This is used when combining state updates from tools, middleware, or subagents.

  ## Merge Rules

  - **messages**: Concatenates lists (left + right)
  - **todos**: Uses right if present, otherwise left
  - **metadata**: Deep merges maps

  ## Examples

      left = State.new!(%{messages: [%{role: "user", content: "hi"}]})
      right = State.new!(%{messages: [%{role: "assistant", content: "hello"}]})
      merged = State.merge_states(left, right)
      # merged now has both messages
  """
  def merge_states(left, right)

  def merge_states(%State{} = left, %State{} = right) do
    %State{
      agent_id: left.agent_id || right.agent_id,
      messages: merge_messages(left.messages, right.messages),
      todos: merge_todos(left.todos, right.todos),
      metadata: deep_merge_maps(left.metadata, right.metadata),
      interrupt_data: right.interrupt_data || left.interrupt_data
    }
  end

  def merge_states(%State{} = state, updates) when is_map(updates) do
    right = struct(State, updates)
    merge_states(state, right)
  end

  # Private merge functions

  defp merge_messages(left, right) when is_list(left) and is_list(right) do
    left ++ right
  end

  defp merge_messages(left, _right) when is_list(left), do: left
  defp merge_messages(_left, right) when is_list(right), do: right
  defp merge_messages(_left, _right), do: []

  # Replace todos with right if right is a list (even if empty - allows clearing)
  defp merge_todos(_left, right) when is_list(right), do: right
  defp merge_todos(left, _right) when is_list(left), do: left
  defp merge_todos(_left, _right), do: []

  defp deep_merge_maps(left, right) when is_nil(left), do: right
  defp deep_merge_maps(left, right) when is_nil(right), do: left

  defp deep_merge_maps(left, right) when is_map(left) and is_map(right) do
    Map.merge(left, right, fn _key, left_val, right_val ->
      if is_map(left_val) and is_map(right_val) do
        deep_merge_maps(left_val, right_val)
      else
        right_val
      end
    end)
  end

  defp deep_merge_maps(left, _right) when is_map(left), do: left
  defp deep_merge_maps(_left, right) when is_map(right), do: right
  defp deep_merge_maps(_left, _right), do: %{}

  @doc """
  Add a message to the state.

  Message must be a `LangChain.Message` struct.
  """
  def add_message(%State{} = state, %LangChain.Message{} = message) do
    add_messages(state, [message])
  end

  @doc """
  Add multiple messages to the state.

  Messages must be `LangChain.Message` structs.
  """
  def add_messages(%State{} = state, messages) when is_list(messages) do
    %{state | messages: state.messages ++ messages}
  end

  @doc """
  Set metadata value.
  """
  def put_metadata(%State{} = state, key, value) do
    %{state | metadata: Map.put(state.metadata, key, value)}
  end

  @doc """
  Get metadata value.
  """
  def get_metadata(%State{} = state, key, default \\ nil) do
    Map.get(state.metadata, key, default)
  end

  @doc """
  Add or update a TODO item.

  If a TODO with the same ID exists, it will be replaced at its current position.
  If the TODO ID doesn't exist, it will be appended to the end of the list.
  """
  def put_todo(%State{} = state, %LangChain.Agents.Todo{} = todo) do
    # Find the index of the existing TODO with the same ID
    existing_index =
      Enum.find_index(state.todos, fn
        %{id: id} -> id == todo.id
        _ -> false
      end)

    updated_todos =
      case existing_index do
        nil ->
          # Not found, append to the end
          state.todos ++ [todo]

        index ->
          # Found, replace at the same index position
          List.replace_at(state.todos, index, todo)
      end

    %{state | todos: updated_todos}
  end

  @doc """
  Get a TODO by ID.
  """
  def get_todo(%State{} = state, todo_id) when is_binary(todo_id) do
    Enum.find(state.todos, fn
      %{id: ^todo_id} -> true
      _ -> false
    end)
  end

  @doc """
  Remove a TODO by ID.
  """
  def delete_todo(%State{} = state, todo_id) when is_binary(todo_id) do
    updated_todos =
      Enum.reject(state.todos, fn
        %{id: ^todo_id} -> true
        _ -> false
      end)

    %{state | todos: updated_todos}
  end

  @doc """
  Get all TODOs with a specific status.
  """
  def get_todos_by_status(%State{} = state, status) when is_atom(status) do
    Enum.filter(state.todos, fn
      %{status: ^status} -> true
      _ -> false
    end)
  end

  @doc """
  Replace all TODOs.
  """
  def set_todos(%State{} = state, todos) when is_list(todos) do
    %{state | todos: todos}
  end

  @doc """
  Replace all messages.

  Useful for:
  - Thread restoration (restoring persisted messages)
  - Testing scenarios (setting sample messages)
  - Bulk message updates

  ## Parameters

  - `state` - The current State struct
  - `messages` - List of Message structs

  ## Examples

      messages = [
        Message.new_user!("Hello")
      ]
      state = State.set_messages(state, messages)
  """
  def set_messages(%State{} = state, messages) when is_list(messages) do
    %{state | messages: messages}
  end

  @doc """
  Reset the state to a clean slate.

  Clears:
  - All messages
  - All TODOs
  - All metadata

  **Note**: This function only resets the Agent's state structure. File state is managed
  separately by FileSystemServer and must be reset through AgentServer.reset/1 which
  coordinates the full reset process.

  ## Examples

      state = State.new!(%{
        messages: [msg1, msg2],
        todos: [todo1],
        metadata: %{config: "value"}
      })

      reset_state = State.reset(state)
      # reset_state has:
      # - messages: []
      # - todos: []
      # - metadata: %{} (cleared)
  """
  @spec reset(t()) :: t()
  def reset(%State{} = state) do
    %State{
      agent_id: state.agent_id,
      messages: [],
      todos: [],
      metadata: %{}
    }
  end
end
