defmodule LangChain.Agents.Middleware.TodoListTest do
  use ExUnit.Case, async: true

  alias LangChain.Agents.{State, Todo}
  alias LangChain.Agents.Middleware.TodoList
  alias LangChain.Function

  describe "system_prompt/1" do
    test "returns TODO management prompt" do
      prompt = TodoList.system_prompt(nil)
      assert is_binary(prompt)
      assert prompt =~ "write_todos"
      assert prompt =~ "To-Do"
    end
  end

  describe "tools/1" do
    test "provides write_todos tool" do
      tools = TodoList.tools(nil)
      assert length(tools) == 1

      tool = hd(tools)
      assert %Function{} = tool
      assert tool.name == "write_todos"
    end

    test "write_todos tool has proper schema" do
      [tool] = TodoList.tools(nil)

      schema = tool.parameters_schema
      assert schema[:type] == "object" or schema["type"] == "object"

      properties = schema[:properties] || schema["properties"]
      required = schema[:required] || schema["required"]

      assert properties != nil
      assert required != nil
    end
  end

  describe "write_todos tool - replace mode" do
    test "replaces all todos when merge is false" do
      state =
        State.new!(%{todos: [%{"id" => "old", "content" => "Old task", "status" => "pending"}]})

      [tool] = TodoList.tools(nil)

      params = %{
        merge: false,
        todos: [
          %{"id" => "new1", "content" => "New task 1", "status" => "pending"},
          %{"id" => "new2", "content" => "New task 2", "status" => "in_progress"}
        ]
      }

      context = %{state: state}
      assert {:ok, message, updated_state} = tool.function.(params, context)

      assert is_binary(message)
      assert message =~ "replaced"
      assert message =~ "2 TODO"

      assert length(updated_state.todos) == 2
      assert Enum.all?(updated_state.todos, &match?(%Todo{}, &1))

      ids = Enum.map(updated_state.todos, & &1.id)
      assert "new1" in ids
      assert "new2" in ids
      refute "old" in ids
    end

    test "creates todos with proper status types" do
      state = State.new!()
      [tool] = TodoList.tools(nil)

      params = %{
        merge: false,
        todos: [
          %{"id" => "1", "content" => "Task 1", "status" => "pending"},
          %{"id" => "2", "content" => "Task 2", "status" => "in_progress"},
          %{"id" => "3", "content" => "Task 3", "status" => "completed"}
        ]
      }

      {:ok, _msg, updated_state} = tool.function.(params, %{state: state})

      assert Enum.at(updated_state.todos, 0).status == :pending
      assert Enum.at(updated_state.todos, 1).status == :in_progress
      assert Enum.at(updated_state.todos, 2).status == :completed
    end
  end

  describe "write_todos tool - merge mode" do
    test "merges with existing todos by ID" do
      existing_todo1 = Todo.new!(%{id: "1", content: "Keep me", status: :pending})
      existing_todo2 = Todo.new!(%{id: "2", content: "Update me", status: :pending})

      state = State.new!(%{todos: [existing_todo1, existing_todo2]})
      [tool] = TodoList.tools(nil)

      params = %{
        merge: true,
        todos: [
          %{"id" => "2", "content" => "Updated task", "status" => "completed"},
          %{"id" => "3", "content" => "New task", "status" => "pending"}
        ]
      }

      {:ok, message, updated_state} = tool.function.(params, %{state: state})

      assert message =~ "merged"
      assert length(updated_state.todos) == 3

      # Check that todo 1 was kept
      todo1 = Enum.find(updated_state.todos, &(&1.id == "1"))
      assert todo1.content == "Keep me"
      assert todo1.status == :pending

      # Check that todo 2 was updated
      todo2 = Enum.find(updated_state.todos, &(&1.id == "2"))
      assert todo2.content == "Updated task"
      assert todo2.status == :completed

      # Check that todo 3 was added
      todo3 = Enum.find(updated_state.todos, &(&1.id == "3"))
      assert todo3.content == "New task"
    end

    test "preserves existing todos when merging with non-overlapping IDs" do
      existing = Todo.new!(%{id: "keep", content: "Keep", status: :completed})
      state = State.new!(%{todos: [existing]})
      [tool] = TodoList.tools(nil)

      params = %{
        merge: true,
        todos: [%{"id" => "new", "content" => "New", "status" => "pending"}]
      }

      {:ok, _msg, updated_state} = tool.function.(params, %{state: state})

      assert length(updated_state.todos) == 2
      ids = Enum.map(updated_state.todos, & &1.id)
      assert "keep" in ids
      assert "new" in ids
    end
  end

  describe "write_todos tool - validation" do
    test "returns error for invalid todo data" do
      state = State.new!()
      [tool] = TodoList.tools(nil)

      params = %{
        merge: false,
        todos: [%{"id" => "1", "content" => "", "status" => "pending"}]
      }

      assert {:error, message} = tool.function.(params, %{state: state})
      assert message =~ "Failed to parse"
    end

    test "returns error when todos is not an array" do
      state = State.new!()
      [tool] = TodoList.tools(nil)

      params = %{
        merge: false,
        todos: "not an array"
      }

      assert {:error, message} = tool.function.(params, %{state: state})
      assert message =~ "must be an array"
    end

    test "handles invalid status gracefully" do
      state = State.new!()
      [tool] = TodoList.tools(nil)

      params = %{
        merge: false,
        todos: [%{"id" => "1", "content" => "Task", "status" => "invalid_status"}]
      }

      # Invalid status defaults to pending
      {:ok, _msg, updated_state} = tool.function.(params, %{state: state})
      assert hd(updated_state.todos).status == :pending
    end
  end

  describe "write_todos tool - response messages" do
    test "includes count and status summary in response" do
      state = State.new!()
      [tool] = TodoList.tools(nil)

      params = %{
        merge: false,
        todos: [
          %{"id" => "1", "content" => "Task 1", "status" => "pending"},
          %{"id" => "2", "content" => "Task 2", "status" => "pending"},
          %{"id" => "3", "content" => "Task 3", "status" => "in_progress"}
        ]
      }

      {:ok, message, _state} = tool.function.(params, %{state: state})

      assert message =~ "3 TODO"
      assert message =~ "2 pending"
      assert message =~ "1 in_progress"
    end

    test "indicates merge vs replace in message" do
      state = State.new!()
      [tool] = TodoList.tools(nil)

      params = %{
        merge: true,
        todos: [%{"id" => "1", "content" => "Task", "status" => "pending"}]
      }

      {:ok, merge_msg, _} = tool.function.(params, %{state: state})
      assert merge_msg =~ "merged"

      params_replace = %{merge: false, todos: params.todos}
      {:ok, replace_msg, _} = tool.function.(params_replace, %{state: state})
      assert replace_msg =~ "replaced"
    end
  end

  describe "integration with State helpers" do
    test "todos can be accessed with State.get_todo/2" do
      state = State.new!()
      [tool] = TodoList.tools(nil)

      params = %{
        merge: false,
        todos: [%{"id" => "test-id", "content" => "Task", "status" => "pending"}]
      }

      {:ok, _, updated_state} = tool.function.(params, %{state: state})

      todo = State.get_todo(updated_state, "test-id")
      assert todo.content == "Task"
    end

    test "todos can be filtered by status" do
      state = State.new!()
      [tool] = TodoList.tools(nil)

      params = %{
        merge: false,
        todos: [
          %{"id" => "1", "content" => "Task 1", "status" => "pending"},
          %{"id" => "2", "content" => "Task 2", "status" => "completed"},
          %{"id" => "3", "content" => "Task 3", "status" => "pending"}
        ]
      }

      {:ok, _, updated_state} = tool.function.(params, %{state: state})

      pending_todos = State.get_todos_by_status(updated_state, :pending)
      assert length(pending_todos) == 2

      completed_todos = State.get_todos_by_status(updated_state, :completed)
      assert length(completed_todos) == 1
    end
  end

  describe "auto-cleanup when all todos completed" do
    test "clears todo list when all todos marked as completed in replace mode" do
      state = State.new!()
      [tool] = TodoList.tools(nil)

      # Create todos with all completed
      params = %{
        merge: false,
        todos: [
          %{"id" => "1", "content" => "Task 1", "status" => "completed"},
          %{"id" => "2", "content" => "Task 2", "status" => "completed"}
        ]
      }

      {:ok, msg, updated_state} = tool.function.(params, %{state: state})

      # List should be cleared since all todos are completed
      assert updated_state.todos == []
      # Message should indicate list was cleared
      assert msg == "TODO list cleared - all tasks completed"
    end

    test "clears todo list when all todos marked as completed in merge mode" do
      # Start with one pending todo
      existing = Todo.new!(%{id: "1", content: "Task 1", status: :pending})
      state = State.new!(%{todos: [existing]})
      [tool] = TodoList.tools(nil)

      # Update the todo to completed
      params = %{
        merge: true,
        todos: [%{"id" => "1", "content" => "Task 1", "status" => "completed"}]
      }

      {:ok, msg, updated_state} = tool.function.(params, %{state: state})

      # List should be cleared since all todos are completed
      assert updated_state.todos == []
      # Message should indicate list was cleared
      assert msg == "TODO list cleared - all tasks completed"
    end

    test "keeps todo list when not all todos are completed" do
      state = State.new!()
      [tool] = TodoList.tools(nil)

      # Create todos with mixed status
      params = %{
        merge: false,
        todos: [
          %{"id" => "1", "content" => "Task 1", "status" => "completed"},
          %{"id" => "2", "content" => "Task 2", "status" => "pending"}
        ]
      }

      {:ok, _msg, updated_state} = tool.function.(params, %{state: state})

      # List should NOT be cleared since not all are completed
      assert length(updated_state.todos) == 2
    end

    test "keeps todo list when todos have in_progress status" do
      state = State.new!()
      [tool] = TodoList.tools(nil)

      params = %{
        merge: false,
        todos: [
          %{"id" => "1", "content" => "Task 1", "status" => "completed"},
          %{"id" => "2", "content" => "Task 2", "status" => "in_progress"}
        ]
      }

      {:ok, _msg, updated_state} = tool.function.(params, %{state: state})

      # List should NOT be cleared
      assert length(updated_state.todos) == 2
    end

    test "clears list immediately after completing final todo via merge" do
      # Start with two todos, one completed and one pending
      todo1 = Todo.new!(%{id: "1", content: "Task 1", status: :completed})
      todo2 = Todo.new!(%{id: "2", content: "Task 2", status: :pending})
      state = State.new!(%{todos: [todo1, todo2]})
      [tool] = TodoList.tools(nil)

      # Complete the final pending todo
      params = %{
        merge: true,
        todos: [%{"id" => "2", "content" => "Task 2", "status" => "completed"}]
      }

      {:ok, msg, updated_state} = tool.function.(params, %{state: state})

      # List should be cleared immediately
      assert updated_state.todos == []
      # Message should indicate list was cleared
      assert msg == "TODO list cleared - all tasks completed"
    end
  end
end
