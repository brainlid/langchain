defmodule LangChain.Agents.StateTest do
  use ExUnit.Case, async: true

  alias LangChain.Agents.State
  alias LangChain.Message

  doctest State

  describe "new/1" do
    test "creates a state with default values" do
      assert {:ok, state} = State.new()
      assert state.messages == []
      assert state.todos == []
      assert state.files == %{}
      assert state.metadata == %{}
      assert state.middleware_state == %{}
    end

    test "creates a state with provided attributes" do
      message = Message.new_user!("hello")

      attrs = %{
        messages: [message],
        todos: [%{id: "1", content: "task"}],
        files: %{"file.txt" => "content"},
        metadata: %{key: "value"}
      }

      assert {:ok, state} = State.new(attrs)
      assert state.messages == attrs.messages
      assert state.todos == attrs.todos
      assert state.files == attrs.files
      assert state.metadata == attrs.metadata
    end
  end

  describe "new!/1" do
    test "creates a state successfully" do
      state = State.new!()
      assert %State{} = state
    end

    test "creates state with attributes" do
      state = State.new!(%{messages: [Message.new_user!("hi")]})
      assert length(state.messages) == 1
    end
  end

  describe "merge_states/2 with two State structs" do
    test "merges messages by concatenating" do
      left = State.new!(%{messages: [Message.new_user!("hello")]})
      right = State.new!(%{messages: [Message.new_assistant!("hi")]})

      merged = State.merge_states(left, right)

      assert length(merged.messages) == 2
      assert merged.messages == left.messages ++ right.messages
    end

    test "merges todos by using right if present" do
      left = State.new!(%{todos: [%{id: "1", content: "old"}]})
      right = State.new!(%{todos: [%{id: "2", content: "new"}]})

      merged = State.merge_states(left, right)

      assert length(merged.todos) == 1
      assert Enum.at(merged.todos, 0).id == "2"
    end

    test "keeps left todos if right is empty" do
      left = State.new!(%{todos: [%{id: "1", content: "task"}]})
      right = State.new!(%{todos: []})

      merged = State.merge_states(left, right)

      assert length(merged.todos) == 1
      assert Enum.at(merged.todos, 0).id == "1"
    end

    test "merges files with right side winning" do
      left = State.new!(%{files: %{"a.txt" => "old", "b.txt" => "keep"}})
      right = State.new!(%{files: %{"a.txt" => "new", "c.txt" => "add"}})

      merged = State.merge_states(left, right)

      assert merged.files["a.txt"] == "new"
      assert merged.files["b.txt"] == "keep"
      assert merged.files["c.txt"] == "add"
    end

    test "handles nil files" do
      left = State.new!(%{files: %{"a.txt" => "content"}})
      right = State.new!()

      merged = State.merge_states(left, right)
      assert merged.files == %{"a.txt" => "content"}
    end

    test "deep merges metadata" do
      left = State.new!(%{metadata: %{a: 1, nested: %{x: 1, y: 2}}})
      right = State.new!(%{metadata: %{b: 2, nested: %{y: 3, z: 4}}})

      merged = State.merge_states(left, right)

      assert merged.metadata.a == 1
      assert merged.metadata.b == 2
      assert merged.metadata.nested.x == 1
      assert merged.metadata.nested.y == 3
      assert merged.metadata.nested.z == 4
    end

    test "merges middleware_state" do
      left = State.new!(%{middleware_state: %{m1: "a"}})
      right = State.new!(%{middleware_state: %{m2: "b"}})

      merged = State.merge_states(left, right)

      assert merged.middleware_state == %{m1: "a", m2: "b"}
    end
  end

  describe "merge_states/2 with map updates" do
    test "merges map into state" do
      state = State.new!(%{messages: [Message.new_user!("hello")]})

      merged = State.merge_states(state, %{files: %{"test.txt" => "content"}})

      assert merged.messages == state.messages
      assert merged.files == %{"test.txt" => "content"}
    end
  end

  describe "add_message/2" do
    test "adds a message to empty state" do
      state = State.new!()
      message = Message.new_user!("hello")

      updated = State.add_message(state, message)

      assert length(updated.messages) == 1
      assert Enum.at(updated.messages, 0) == message
    end

    test "appends message to existing messages" do
      state = State.new!(%{messages: [Message.new_user!("first")]})
      message = Message.new_assistant!("second")

      updated = State.add_message(state, message)

      assert length(updated.messages) == 2
      assert Enum.at(updated.messages, 1) == message
    end
  end

  describe "add_messages/2" do
    test "adds multiple messages" do
      state = State.new!()

      messages = [
        Message.new_user!("first"),
        Message.new_assistant!("second")
      ]

      updated = State.add_messages(state, messages)

      assert length(updated.messages) == 2
    end
  end

  describe "put_file/3" do
    test "adds a file to empty filesystem" do
      state = State.new!()
      updated = State.put_file(state, "/test.txt", "content")

      # Files now use FileData structure, use State.get_file to extract content
      assert State.get_file(updated, "/test.txt") == "content"
    end

    test "overwrites existing file" do
      state = State.new!(%{files: %{"/test.txt" => "old"}})
      updated = State.put_file(state, "/test.txt", "new")

      # Files now use FileData structure, use State.get_file to extract content
      assert State.get_file(updated, "/test.txt") == "new"
    end
  end

  describe "get_file/2" do
    test "retrieves existing file" do
      state = State.new!(%{files: %{"/test.txt" => "content"}})
      assert State.get_file(state, "/test.txt") == "content"
    end

    test "returns nil for missing file" do
      state = State.new!()
      assert State.get_file(state, "/missing.txt") == nil
    end
  end

  describe "delete_file/2" do
    test "removes file from filesystem" do
      state = State.new!(%{files: %{"/test.txt" => "content"}})
      updated = State.delete_file(state, "/test.txt")

      assert updated.files == %{}
    end

    test "handles deleting non-existent file" do
      state = State.new!()
      updated = State.delete_file(state, "/missing.txt")

      assert updated.files == %{}
    end
  end

  describe "put_metadata/3" do
    test "adds metadata" do
      state = State.new!()
      updated = State.put_metadata(state, :key, "value")

      assert updated.metadata.key == "value"
    end

    test "overwrites existing metadata" do
      state = State.new!(%{metadata: %{key: "old"}})
      updated = State.put_metadata(state, :key, "new")

      assert updated.metadata.key == "new"
    end
  end

  describe "get_metadata/3" do
    test "retrieves existing metadata" do
      state = State.new!(%{metadata: %{key: "value"}})
      assert State.get_metadata(state, :key) == "value"
    end

    test "returns nil for missing metadata" do
      state = State.new!()
      assert State.get_metadata(state, :missing) == nil
    end

    test "returns default for missing metadata" do
      state = State.new!()
      assert State.get_metadata(state, :missing, "default") == "default"
    end
  end

  describe "put_todo/2" do
    test "adds a new todo" do
      alias LangChain.Agents.Todo

      state = State.new!()
      todo = Todo.new!(%{id: "1", content: "Task", status: :pending})

      updated = State.put_todo(state, todo)

      assert length(updated.todos) == 1
      assert hd(updated.todos).id == "1"
    end

    test "replaces existing todo with same ID" do
      alias LangChain.Agents.Todo

      todo1 = Todo.new!(%{id: "1", content: "Original", status: :pending})
      state = State.new!(%{todos: [todo1]})

      todo2 = Todo.new!(%{id: "1", content: "Updated", status: :completed})
      updated = State.put_todo(state, todo2)

      assert length(updated.todos) == 1
      assert hd(updated.todos).content == "Updated"
      assert hd(updated.todos).status == :completed
    end

    test "maintains insertion order for todos" do
      alias LangChain.Agents.Todo

      state = State.new!()
      todo_c = Todo.new!(%{id: "c", content: "C"})
      todo_a = Todo.new!(%{id: "a", content: "A"})
      todo_b = Todo.new!(%{id: "b", content: "B"})

      updated =
        state
        |> State.put_todo(todo_c)
        |> State.put_todo(todo_a)
        |> State.put_todo(todo_b)

      ids = Enum.map(updated.todos, & &1.id)
      # Insertion order is preserved, not sorted by ID
      assert ids == ["c", "a", "b"]
    end

    test "updating a todo maintains its position in the list" do
      alias LangChain.Agents.Todo

      # Create initial list of todos
      todo1 = Todo.new!(%{id: "1", content: "First", status: :pending})
      todo2 = Todo.new!(%{id: "2", content: "Second", status: :pending})
      todo3 = Todo.new!(%{id: "3", content: "Third", status: :pending})

      state = State.new!(%{todos: [todo1, todo2, todo3]})

      # Update the middle todo
      updated_todo2 = Todo.new!(%{id: "2", content: "Second Updated", status: :completed})
      updated_state = State.put_todo(state, updated_todo2)

      # Check order is maintained
      ids = Enum.map(updated_state.todos, & &1.id)
      assert ids == ["1", "2", "3"], "Order should be preserved when updating"

      # Check the content was updated
      second = Enum.at(updated_state.todos, 1)
      assert second.content == "Second Updated"
      assert second.status == :completed

      # Update the first todo
      updated_todo1 = Todo.new!(%{id: "1", content: "First Updated", status: :in_progress})
      updated_state2 = State.put_todo(updated_state, updated_todo1)

      ids2 = Enum.map(updated_state2.todos, & &1.id)
      assert ids2 == ["1", "2", "3"], "Order should still be preserved"

      first = Enum.at(updated_state2.todos, 0)
      assert first.content == "First Updated"
      assert first.status == :in_progress
    end
  end

  describe "get_todo/2" do
    test "retrieves todo by ID" do
      alias LangChain.Agents.Todo

      todo = Todo.new!(%{id: "test", content: "Task"})
      state = State.new!(%{todos: [todo]})

      retrieved = State.get_todo(state, "test")
      assert retrieved.id == "test"
      assert retrieved.content == "Task"
    end

    test "returns nil for non-existent ID" do
      state = State.new!()
      assert State.get_todo(state, "missing") == nil
    end
  end

  describe "delete_todo/2" do
    test "removes todo by ID" do
      alias LangChain.Agents.Todo

      todo1 = Todo.new!(%{id: "1", content: "Keep"})
      todo2 = Todo.new!(%{id: "2", content: "Remove"})
      state = State.new!(%{todos: [todo1, todo2]})

      updated = State.delete_todo(state, "2")

      assert length(updated.todos) == 1
      assert hd(updated.todos).id == "1"
    end

    test "handles deleting non-existent todo" do
      alias LangChain.Agents.Todo

      todo = Todo.new!(%{id: "1", content: "Task"})
      state = State.new!(%{todos: [todo]})

      updated = State.delete_todo(state, "missing")

      assert length(updated.todos) == 1
    end
  end

  describe "get_todos_by_status/2" do
    test "filters todos by status" do
      alias LangChain.Agents.Todo

      todo1 = Todo.new!(%{id: "1", content: "Task 1", status: :pending})
      todo2 = Todo.new!(%{id: "2", content: "Task 2", status: :completed})
      todo3 = Todo.new!(%{id: "3", content: "Task 3", status: :pending})

      state = State.new!(%{todos: [todo1, todo2, todo3]})

      pending = State.get_todos_by_status(state, :pending)
      assert length(pending) == 2
      assert Enum.all?(pending, &(&1.status == :pending))

      completed = State.get_todos_by_status(state, :completed)
      assert length(completed) == 1
      assert hd(completed).status == :completed
    end

    test "returns empty list when no matches" do
      alias LangChain.Agents.Todo

      todo = Todo.new!(%{id: "1", content: "Task", status: :pending})
      state = State.new!(%{todos: [todo]})

      completed = State.get_todos_by_status(state, :completed)
      assert completed == []
    end
  end

  describe "set_todos/2" do
    test "replaces all todos" do
      alias LangChain.Agents.Todo

      old_todo = Todo.new!(%{id: "old", content: "Old"})
      state = State.new!(%{todos: [old_todo]})

      new_todos = [
        Todo.new!(%{id: "new1", content: "New 1"}),
        Todo.new!(%{id: "new2", content: "New 2"})
      ]

      updated = State.set_todos(state, new_todos)

      assert length(updated.todos) == 2
      ids = Enum.map(updated.todos, & &1.id)
      assert "new1" in ids
      assert "new2" in ids
      refute "old" in ids
    end

    test "can set empty list" do
      alias LangChain.Agents.Todo

      todo = Todo.new!(%{id: "1", content: "Task"})
      state = State.new!(%{todos: [todo]})

      updated = State.set_todos(state, [])
      assert updated.todos == []
    end
  end
end
