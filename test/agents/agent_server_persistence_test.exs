defmodule LangChain.Agents.AgentServerPersistenceTest do
  use ExUnit.Case, async: false

  alias LangChain.Agents.{Agent, AgentServer, State, Todo}
  alias LangChain.Message
  alias LangChain.ChatModels.ChatOpenAI

  @registry LangChain.Agents.Registry

  setup do
    # Start the registry if not already started
    start_supervised!({Registry, keys: :unique, name: @registry})

    :ok
  end

  describe "export_state/1" do
    test "exports current agent state with string keys" do
      {:ok, model} = ChatOpenAI.new(%{model: "gpt-4", api_key: "test-key"})

      {:ok, agent} =
        Agent.new(%{
          agent_id: "export-test-1",
          model: model,
          system_prompt: "You are helpful"
        })

      {:ok, msg1} = Message.new_user!("Hello")
      {:ok, msg2} = Message.new_assistant!(%{content: "Hi there"})

      {:ok, todo} = Todo.new(%{content: "Task 1", status: :in_progress})

      state =
        State.new!(%{
          messages: [msg1, msg2],
          todos: [todo],
          metadata: %{session_id: "session-1"}
        })

      {:ok, _pid} =
        AgentServer.start_link(
          agent: agent,
          initial_state: state,
          name: AgentServer.get_name("export-test-1"),
          pubsub: nil
        )

      # Export state
      exported = AgentServer.export_state("export-test-1")

      # Check structure
      assert is_map(exported)
      assert Map.has_key?(exported, "version")
      assert Map.has_key?(exported, "agent_id")
      assert Map.has_key?(exported, "state")
      assert Map.has_key?(exported, "agent_config")
      assert Map.has_key?(exported, "serialized_at")

      # Check all keys are strings
      assert all_keys_are_strings?(exported)

      # Check content
      assert exported["agent_id"] == "export-test-1"
      assert length(exported["state"]["messages"]) == 2
      assert length(exported["state"]["todos"]) == 1

      # Cleanup
      AgentServer.stop("export-test-1")
    end

    test "exports state with empty messages and todos" do
      {:ok, model} = ChatOpenAI.new(%{model: "gpt-4", api_key: "test-key"})
      {:ok, agent} = Agent.new(%{agent_id: "export-test-2", model: model})

      {:ok, _pid} =
        AgentServer.start_link(
          agent: agent,
          name: AgentServer.get_name("export-test-2"),
          pubsub: nil
        )

      exported = AgentServer.export_state("export-test-2")

      assert exported["state"]["messages"] == []
      assert exported["state"]["todos"] == []
      assert exported["state"]["metadata"] == %{}

      AgentServer.stop("export-test-2")
    end
  end

  describe "restore_state/2" do
    test "restores agent state from exported data" do
      {:ok, model} = ChatOpenAI.new(%{model: "gpt-4", api_key: "test-key"})

      {:ok, agent} =
        Agent.new(%{
          agent_id: "restore-test-1",
          model: model,
          system_prompt: "Initial prompt"
        })

      {:ok, _pid} =
        AgentServer.start_link(
          agent: agent,
          name: AgentServer.get_name("restore-test-1"),
          pubsub: nil
        )

      # Create exported state to restore
      exported_state = %{
        "version" => 1,
        "agent_id" => "restore-test-1",
        "state" => %{
          "messages" => [
            %{"role" => "user", "content" => "Restored message"}
          ],
          "todos" => [
            %{
              "id" => "todo-1",
              "content" => "Restored task",
              "status" => "pending"
            }
          ],
          "metadata" => %{"restored" => true}
        },
        "agent_config" => %{
          "agent_id" => "restore-test-1",
          "model" => %{
            "module" => "Elixir.LangChain.ChatModels.ChatOpenAI",
            "model" => "gpt-4"
          },
          "system_prompt" => "Restored prompt",
          "tools" => [],
          "middleware" => []
        },
        "serialized_at" => "2025-11-29T10:30:00Z"
      }

      # Restore state
      :ok = AgentServer.restore_state("restore-test-1", exported_state)

      # Verify state was restored
      restored_state = AgentServer.get_state("restore-test-1")
      assert length(restored_state.messages) == 1
      assert hd(restored_state.messages).content == "Restored message"

      assert length(restored_state.todos) == 1
      assert hd(restored_state.todos).content == "Restored task"

      assert restored_state.metadata["restored"] == true

      AgentServer.stop("restore-test-1")
    end

    test "returns error for invalid state data" do
      {:ok, model} = ChatOpenAI.new(%{model: "gpt-4", api_key: "test-key"})
      {:ok, agent} = Agent.new(%{agent_id: "restore-test-2", model: model})

      {:ok, _pid} =
        AgentServer.start_link(
          agent: agent,
          name: AgentServer.get_name("restore-test-2"),
          pubsub: nil
        )

      # Try to restore invalid state
      invalid_state = %{
        "version" => 1,
        "agent_id" => "restore-test-2",
        "state" => %{
          "messages" => "not a list",
          # Invalid - should be list
          "todos" => [],
          "metadata" => %{}
        }
      }

      result = AgentServer.restore_state("restore-test-2", invalid_state)
      assert {:error, _reason} = result

      AgentServer.stop("restore-test-2")
    end
  end

  describe "start_link_from_state/2" do
    test "starts agent server with restored state" do
      # Create exported state
      exported_state = %{
        "version" => 1,
        "agent_id" => "from-state-test-1",
        "state" => %{
          "messages" => [
            %{"role" => "user", "content" => "Hello"},
            %{"role" => "assistant", "content" => "Hi"}
          ],
          "todos" => [
            %{
              "id" => "todo-1",
              "content" => "Task from state",
              "status" => "in_progress"
            }
          ],
          "metadata" => %{"session_id" => "restored-session"}
        },
        "agent_config" => %{
          "agent_id" => "from-state-test-1",
          "model" => %{
            "module" => "Elixir.LangChain.ChatModels.ChatOpenAI",
            "model" => "gpt-4"
          },
          "system_prompt" => "You are a helpful assistant",
          "temperature" => 0.8,
          "tools" => [],
          "middleware" => []
        },
        "serialized_at" => "2025-11-29T10:30:00Z"
      }

      # Start from state
      {:ok, pid} =
        AgentServer.start_link_from_state(
          exported_state,
          name: AgentServer.get_name("from-state-test-1"),
          pubsub: nil
        )

      assert is_pid(pid)

      # Verify state was restored
      state = AgentServer.get_state("from-state-test-1")
      assert length(state.messages) == 2
      assert hd(state.messages).content == "Hello"

      assert length(state.todos) == 1
      assert hd(state.todos).content == "Task from state"

      assert state.metadata["session_id"] == "restored-session"

      # Verify status is idle (ready for execution)
      status = AgentServer.get_status("from-state-test-1")
      assert status == :idle

      AgentServer.stop("from-state-test-1")
    end

    test "fails to start with invalid state data" do
      invalid_state = %{
        "version" => 1,
        "agent_id" => "from-state-test-2",
        "state" => %{
          "messages" => []
        }
        # Missing agent_config
      }

      result =
        AgentServer.start_link_from_state(
          invalid_state,
          name: AgentServer.get_name("from-state-test-2"),
          pubsub: nil
        )

      assert {:error, _reason} = result
    end
  end

  describe "round-trip export and restore" do
    test "export and start_link_from_state preserves complete state" do
      {:ok, model} = ChatOpenAI.new(%{model: "gpt-4", api_key: "test-key"})

      {:ok, agent} =
        Agent.new(%{
          agent_id: "roundtrip-test-1",
          model: model,
          system_prompt: "Original prompt"
        })

      {:ok, msg1} = Message.new_user!("User message")
      {:ok, msg2} = Message.new_assistant!(%{content: "Assistant response"})

      {:ok, todo1} = Todo.new(%{content: "Task 1", status: :in_progress})

      {:ok, todo2} = Todo.new(%{content: "Task 2", status: :completed})

      original_state =
        State.new!(%{
          messages: [msg1, msg2],
          todos: [todo1, todo2],
          metadata: %{key1: "value1", key2: 42}
        })

      # Start original server
      {:ok, _pid} =
        AgentServer.start_link(
          agent: agent,
          initial_state: original_state,
          name: AgentServer.get_name("roundtrip-test-1"),
          pubsub: nil
        )

      # Export state
      exported = AgentServer.export_state("roundtrip-test-1")

      # Stop original server
      AgentServer.stop("roundtrip-test-1")

      # Small delay to ensure process is stopped
      Process.sleep(50)

      # Start new server from exported state
      {:ok, _pid} =
        AgentServer.start_link_from_state(
          exported,
          name: AgentServer.get_name("roundtrip-test-1"),
          pubsub: nil
        )

      # Get restored state
      restored_state = AgentServer.get_state("roundtrip-test-1")

      # Compare messages
      assert length(restored_state.messages) == length(original_state.messages)

      Enum.zip(restored_state.messages, original_state.messages)
      |> Enum.each(fn {restored, original} ->
        assert restored.role == original.role
        assert restored.content == original.content
      end)

      # Compare todos
      assert length(restored_state.todos) == length(original_state.todos)

      Enum.zip(restored_state.todos, original_state.todos)
      |> Enum.each(fn {restored, original} ->
        assert restored.content == original.content
        assert restored.status == original.status
      end)

      # Metadata will have string keys after round-trip
      assert restored_state.metadata["key1"] == "value1"
      assert restored_state.metadata["key2"] == 42

      AgentServer.stop("roundtrip-test-1")
    end

    test "export, restore, and continue execution" do
      {:ok, model} = ChatOpenAI.new(%{model: "gpt-4", api_key: "test-key"})

      {:ok, agent} =
        Agent.new(%{
          agent_id: "continue-test-1",
          model: model,
          system_prompt: "You are helpful"
        })

      {:ok, msg1} = Message.new_user!("First message")

      state = State.new!(%{messages: [msg1]})

      # Start server
      {:ok, _pid} =
        AgentServer.start_link(
          agent: agent,
          initial_state: state,
          name: AgentServer.get_name("continue-test-1"),
          pubsub: nil
        )

      # Export state
      exported = AgentServer.export_state("continue-test-1")

      # Add another message
      {:ok, msg2} = Message.new_user!("Second message")
      :ok = AgentServer.add_message("continue-test-1", msg2)

      # Get updated state
      updated_state = AgentServer.get_state("continue-test-1")
      assert length(updated_state.messages) == 2

      # Restore to previous state
      :ok = AgentServer.restore_state("continue-test-1", exported)

      # Verify we're back to previous state
      restored_state = AgentServer.get_state("continue-test-1")
      assert length(restored_state.messages) == 1
      assert hd(restored_state.messages).content == "First message"

      AgentServer.stop("continue-test-1")
    end
  end

  # Helper function
  defp all_keys_are_strings?(map) when is_map(map) do
    Enum.all?(map, fn {key, value} ->
      is_binary(key) && all_keys_are_strings?(value)
    end)
  end

  defp all_keys_are_strings?(list) when is_list(list) do
    Enum.all?(list, &all_keys_are_strings?/1)
  end

  defp all_keys_are_strings?(_other), do: true
end
