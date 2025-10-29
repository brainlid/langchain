defmodule LangChain.Agents.State do
  @moduledoc """
  Agent state structure with support for middleware state.

  The state holds the complete context for an agent execution including:
  - Message history (list of `LangChain.Message` structs)
  - TODO list
  - Mock filesystem
  - Metadata
  - Middleware-specific state

  ## File Storage Format

  Files are stored with metadata for tracking creation and modification times:

      %{
        "path/to/file.txt" => %{
          content: "file contents as string",
          created_at: ~U[2025-10-24 12:00:00Z],
          modified_at: ~U[2025-10-24 12:00:00Z]
        }
      }

  ## State Merging

  State merging follows specific rules:

  - **messages**: Appends new messages to existing list
  - **todos**: Replaces with new todos (merge handled by TodoList middleware)
  - **files**: Merges file dictionaries (right side wins on conflicts, preserves FileData structure)
  - **metadata**: Deep merges metadata maps
  - **middleware_state**: Merges middleware-specific state
  """

  use Ecto.Schema
  import Ecto.Changeset
  alias __MODULE__

  @primary_key false
  embedded_schema do
    # List of LangChain.Message structs
    field :messages, {:array, :any}, default: [], virtual: true
    field :todos, {:array, :map}, default: []
    field :files, :map, default: %{}
    field :metadata, :map, default: %{}
    field :middleware_state, :map, default: %{}
  end

  @type t :: %State{}

  @doc """
  Create a new agent state.
  """
  def new(attrs \\ %{}) do
    %State{}
    |> cast(attrs, [:messages, :todos, :files, :metadata, :middleware_state])
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
  Merge two states together.

  This is used when combining state updates from tools, middleware, or subagents.

  ## Merge Rules

  - **messages**: Concatenates lists (left + right)
  - **todos**: Uses right if present, otherwise left
  - **files**: Merges maps with right side winning on conflicts
  - **metadata**: Deep merges maps
  - **middleware_state**: Merges maps with right side winning

  ## Examples

      left = State.new!(%{messages: [%{role: "user", content: "hi"}]})
      right = State.new!(%{messages: [%{role: "assistant", content: "hello"}]})
      merged = State.merge_states(left, right)
      # merged now has both messages
  """
  def merge_states(left, right)

  def merge_states(%State{} = left, %State{} = right) do
    %State{
      messages: merge_messages(left.messages, right.messages),
      todos: merge_todos(left.todos, right.todos),
      files: merge_files(left.files, right.files),
      metadata: deep_merge_maps(left.metadata, right.metadata),
      middleware_state: Map.merge(left.middleware_state, right.middleware_state)
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

  defp merge_todos(_left, right) when is_list(right) and right != [], do: right
  defp merge_todos(left, _right) when is_list(left), do: left
  defp merge_todos(_left, _right), do: []

  defp merge_files(left, right) when is_nil(left), do: right
  defp merge_files(left, right) when is_nil(right), do: left

  defp merge_files(left, right) when is_map(left) and is_map(right) do
    Map.merge(left, right)
  end

  defp merge_files(left, _right) when is_map(left), do: left
  defp merge_files(_left, right) when is_map(right), do: right
  defp merge_files(_left, _right), do: %{}

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
  Create FileData structure with metadata.

  Returns a map with content and timestamps:

      %{
        content: "file content",
        created_at: ~U[...],
        modified_at: ~U[...]
      }
  """
  def create_file_data(content) when is_binary(content) do
    now = DateTime.utc_now()

    %{
      content: content,
      created_at: now,
      modified_at: now
    }
  end

  @doc """
  Update FileData structure with new content, preserving created_at.

  If existing_file_data is nil, creates new FileData.
  """
  def update_file_data(content, existing_file_data \\ nil) when is_binary(content) do
    case existing_file_data do
      %{created_at: created_at} ->
        %{
          content: content,
          created_at: created_at,
          modified_at: DateTime.utc_now()
        }

      _ ->
        create_file_data(content)
    end
  end

  @doc """
  Extract content from FileData.

  Handles both old format (string) and new format (FileData map).
  """
  def extract_file_content(file_data) do
    case file_data do
      %{content: content} when is_binary(content) -> content
      content when is_binary(content) -> content
      _ -> nil
    end
  end

  @doc """
  Update file in the filesystem with metadata.

  Creates or updates a file with FileData structure including timestamps.
  """
  def put_file(%State{} = state, path, content) when is_binary(path) and is_binary(content) do
    existing_file_data = Map.get(state.files, path)
    file_data = update_file_data(content, existing_file_data)
    %{state | files: Map.put(state.files, path, file_data)}
  end

  @doc """
  Get file content from the filesystem.

  Returns the content string if file exists, nil otherwise.
  Handles both old format (string) and new format (FileData map).
  """
  def get_file(%State{} = state, path) when is_binary(path) do
    state.files
    |> Map.get(path)
    |> extract_file_content()
  end

  @doc """
  Get complete FileData for a file.

  Returns the full FileData map with metadata, or nil if not found.
  """
  def get_file_data(%State{} = state, path) when is_binary(path) do
    Map.get(state.files, path)
  end

  @doc """
  Delete file from the filesystem.
  """
  def delete_file(%State{} = state, path) when is_binary(path) do
    %{state | files: Map.delete(state.files, path)}
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
end
