defmodule LangChain.Agents.TodoTest do
  use ExUnit.Case, async: true

  alias LangChain.Agents.Todo

  doctest Todo

  describe "new/1" do
    test "creates a todo with default values" do
      assert {:ok, todo} = Todo.new(%{content: "Test task"})
      assert todo.content == "Test task"
      assert todo.status == :pending
      assert is_binary(todo.id)
      assert String.length(todo.id) > 0
    end

    test "creates a todo with custom ID" do
      assert {:ok, todo} = Todo.new(%{id: "custom-123", content: "Task"})
      assert todo.id == "custom-123"
    end

    test "creates a todo with specified status" do
      assert {:ok, todo} = Todo.new(%{content: "Task", status: :in_progress})
      assert todo.status == :in_progress
    end

    test "accepts all valid statuses" do
      for status <- [:pending, :in_progress, :completed, :cancelled] do
        assert {:ok, todo} = Todo.new(%{content: "Task", status: status})
        assert todo.status == status
      end
    end

    test "generates unique IDs when not provided" do
      {:ok, todo1} = Todo.new(%{content: "Task 1"})
      {:ok, todo2} = Todo.new(%{content: "Task 2"})
      assert todo1.id != todo2.id
    end

    test "requires content" do
      assert {:error, changeset} = Todo.new(%{})
      assert "can't be blank" in errors_on(changeset).content
    end

    test "validates content length" do
      long_content = String.duplicate("a", 1001)
      assert {:error, changeset} = Todo.new(%{content: long_content})
      errors = errors_on(changeset).content
      assert is_list(errors)
      assert length(errors) > 0
      assert hd(errors) =~ "should be at most"
    end

    test "rejects invalid status" do
      assert {:error, changeset} = Todo.new(%{content: "Task", status: :invalid})
      assert "is invalid" in errors_on(changeset).status
    end

    test "rejects empty content" do
      assert {:error, changeset} = Todo.new(%{content: ""})
      errors = errors_on(changeset).content
      assert is_list(errors)
      assert length(errors) > 0
      # Either "can't be blank" or "should be at least 1 character(s)"
      assert hd(errors) =~ "blank" or hd(errors) =~ "at least"
    end
  end

  describe "new!/1" do
    test "creates a todo successfully" do
      todo = Todo.new!(%{content: "Task"})
      assert %Todo{} = todo
      assert todo.content == "Task"
    end

    test "raises on invalid data" do
      assert_raise LangChain.LangChainError, fn ->
        Todo.new!(%{content: ""})
      end
    end
  end

  describe "to_map/1" do
    test "converts todo to map" do
      todo = Todo.new!(%{id: "123", content: "Task", status: :in_progress})
      map = Todo.to_map(todo)

      assert map == %{
               id: "123",
               content: "Task",
               status: "in_progress"
             }
    end

    test "converts all statuses correctly" do
      for status <- [:pending, :in_progress, :completed, :cancelled] do
        todo = Todo.new!(%{content: "Task", status: status})
        map = Todo.to_map(todo)
        assert map.status == Atom.to_string(status)
      end
    end
  end

  describe "from_map/1" do
    test "creates todo from string-keyed map" do
      map = %{"id" => "123", "content" => "Task", "status" => "pending"}
      assert {:ok, todo} = Todo.from_map(map)
      assert todo.id == "123"
      assert todo.content == "Task"
      assert todo.status == :pending
    end

    test "creates todo from atom-keyed map" do
      map = %{id: "123", content: "Task", status: "completed"}
      assert {:ok, todo} = Todo.from_map(map)
      assert todo.id == "123"
      assert todo.status == :completed
    end

    test "parses all status strings" do
      for status_str <- ["pending", "in_progress", "completed", "cancelled"] do
        map = %{"content" => "Task", "status" => status_str}
        assert {:ok, todo} = Todo.from_map(map)
        assert is_atom(todo.status)
      end
    end

    test "generates ID if not provided in map" do
      map = %{"content" => "Task"}
      assert {:ok, todo} = Todo.from_map(map)
      assert is_binary(todo.id)
      assert String.length(todo.id) > 0
    end

    test "defaults to pending for invalid status string" do
      map = %{"content" => "Task", "status" => "invalid"}
      assert {:ok, todo} = Todo.from_map(map)
      assert todo.status == :pending
    end
  end

  describe "round-trip conversion" do
    test "to_map and from_map preserve data" do
      original = Todo.new!(%{content: "Important task", status: :in_progress})
      map = Todo.to_map(original)
      {:ok, restored} = Todo.from_map(map)

      assert restored.id == original.id
      assert restored.content == original.content
      assert restored.status == original.status
    end
  end

  # Helper to extract errors from changeset
  defp errors_on(changeset) do
    Ecto.Changeset.traverse_errors(changeset, fn {message, _opts} -> message end)
  end
end
