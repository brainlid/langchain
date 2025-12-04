defmodule LangChain.Persistence.StateSerializerTest do
  use ExUnit.Case, async: true

  alias LangChain.Persistence.StateSerializer
  alias LangChain.Agents.{Agent, State, Todo}
  alias LangChain.Message
  alias LangChain.Message.{ToolCall, ToolResult}
  alias LangChain.ChatModels.ChatOpenAI

  describe "serialize_server_state/2" do
    test "serializes agent and state to map with string keys" do
      {:ok, model} = ChatOpenAI.new(%{model: "gpt-4", api_key: "test-key"})

      {:ok, agent} =
        Agent.new(%{
          agent_id: "agent-123",
          model: model,
          base_system_prompt: "You are helpful"
        })

      msg1 = Message.new_user!("Hello")
      msg2 = Message.new_assistant!(%{content: "Hi there"})

      {:ok, todo1} = Todo.new(%{content: "Task 1", status: :in_progress})

      state =
        State.new!(%{
          messages: [msg1, msg2],
          todos: [todo1],
          metadata: %{session_id: "session-1"}
        })

      result = StateSerializer.serialize_server_state(agent, state)

      # Check that result has string keys and correct values
      assert %{
               "version" => 1,
               "agent_id" => "agent-123",
               "state" => _state_data,
               "agent_config" => _agent_config,
               "serialized_at" => _serialized_at
             } = result

      # Check state has string keys
      assert %{"messages" => messages, "todos" => todos, "metadata" => metadata} =
               result["state"]

      # Check messages are serialized with string keys
      assert [first_message, _second_message] = messages
      assert %{"role" => "user", "content" => content} = first_message

      # Content can be a string or list of ContentParts
      assert content == "Hello" or content == [%{"type" => "text", "content" => "Hello"}]

      # Check todos are serialized with string keys
      assert [first_todo] = todos
      assert %{"content" => "Task 1", "status" => "in_progress"} = first_todo

      # Check metadata is serialized with string keys
      assert %{"session_id" => _} = metadata
    end

    test "handles empty state" do
      {:ok, model} = ChatOpenAI.new(%{model: "gpt-4", api_key: "test-key"})
      {:ok, agent} = Agent.new(%{agent_id: "agent-123", model: model})

      state = State.new!(%{})

      result = StateSerializer.serialize_server_state(agent, state)

      assert result["state"]["messages"] == []
      assert result["state"]["todos"] == []
      assert result["state"]["metadata"] == %{}
    end

    test "serializes messages with tool calls" do
      {:ok, model} = ChatOpenAI.new(%{model: "gpt-4", api_key: "test-key"})
      {:ok, agent} = Agent.new(%{agent_id: "agent-123", model: model})

      {:ok, tool_call} =
        ToolCall.new(%{
          call_id: "call-1",
          type: :function,
          name: "calculator",
          arguments: %{"expression" => "2 + 2"},
          status: :complete
        })

      {:ok, msg} = Message.new(%{role: :assistant, content: nil, tool_calls: [tool_call]})

      state = State.new!(%{messages: [msg]})

      result = StateSerializer.serialize_server_state(agent, state)

      assert [message] = result["state"]["messages"]
      assert [serialized_call] = message["tool_calls"]

      # All keys should be strings, check structure and values
      assert %{
               "call_id" => _,
               "type" => _,
               "name" => _,
               "arguments" => %{"expression" => "2 + 2"}
             } = serialized_call
    end

    test "does not serialize API keys" do
      {:ok, model} = ChatOpenAI.new(%{model: "gpt-4", api_key: "secret-key"})
      {:ok, agent} = Agent.new(%{agent_id: "agent-123", model: model})

      state = State.new!(%{})

      result = StateSerializer.serialize_server_state(agent, state)

      # API key should not be present in serialized form
      refute Map.has_key?(result["agent_config"]["model"], "api_key")
    end
  end

  describe "deserialize_server_state/1" do
    test "deserializes map with string keys back to agent and state" do
      # Create a serialized state with string keys (as would come from JSONB)
      serialized = %{
        "version" => 1,
        "agent_id" => "agent-123",
        "state" => %{
          "messages" => [
            %{"role" => "user", "content" => "Hello", "status" => "complete"},
            %{"role" => "assistant", "content" => "Hi there", "status" => "complete"}
          ],
          "todos" => [
            %{
              "id" => "todo-1",
              "content" => "Task 1",
              "status" => "in_progress"
            }
          ],
          "metadata" => %{"session_id" => "session-1"}
        },
        "agent_config" => %{
          "agent_id" => "agent-123",
          "model" => %{
            "module" => "Elixir.LangChain.ChatModels.ChatOpenAI",
            "model" => "gpt-4"
          },
          "base_system_prompt" => "You are helpful",
          "tools" => [],
          "middleware" => []
        },
        "serialized_at" => "2025-11-29T10:30:00Z"
      }

      {:ok, {agent, state}} = StateSerializer.deserialize_server_state(serialized)

      # Check agent
      assert agent.agent_id == "agent-123"

      # System prompt will be re-assembled with middleware, so just check it starts with base prompt
      assert String.starts_with?(agent.assembled_system_prompt, "You are helpful")
      assert agent.model.model == "gpt-4"

      # Check state
      assert [first_msg, _second_msg] = state.messages
      assert first_msg.role == :user
      # Content is converted to ContentPart list by Message.new
      assert [%LangChain.Message.ContentPart{content: "Hello"}] = first_msg.content

      assert [todo] = state.todos
      assert todo.content == "Task 1"
      assert todo.status == :in_progress

      # Metadata can have string keys (from JSONB)
      assert state.metadata["session_id"] == "session-1"
    end

    test "deserializes messages with tool calls" do
      serialized = %{
        "version" => 1,
        "agent_id" => "agent-123",
        "state" => %{
          "messages" => [
            %{
              "role" => "assistant",
              "content" => nil,
              "status" => "complete",
              "tool_calls" => [
                %{
                  "call_id" => "call-1",
                  "type" => "function",
                  "name" => "calculator",
                  "arguments" => %{"expression" => "2 + 2"},
                  "status" => "complete"
                }
              ]
            }
          ],
          "todos" => [],
          "metadata" => %{}
        },
        "agent_config" => %{
          "agent_id" => "agent-123",
          "model" => %{
            "module" => "Elixir.LangChain.ChatModels.ChatOpenAI",
            "model" => "gpt-4"
          },
          "tools" => [],
          "middleware" => []
        },
        "serialized_at" => "2025-11-29T10:30:00Z"
      }

      {:ok, {_agent, state}} = StateSerializer.deserialize_server_state(serialized)

      assert [message] = state.messages
      assert message.role == :assistant
      assert message.content == nil

      assert [tool_call] = message.tool_calls
      assert tool_call.call_id == "call-1"
      assert tool_call.type == :function
      assert tool_call.name == "calculator"
      # Arguments from JSONB will have string keys
      assert tool_call.arguments["expression"] == "2 + 2"
    end

    test "handles missing version field" do
      serialized = %{
        "agent_id" => "agent-123",
        "state" => %{
          "messages" => [],
          "todos" => [],
          "metadata" => %{}
        },
        "agent_config" => %{
          "agent_id" => "agent-123",
          "model" => %{
            "module" => "Elixir.LangChain.ChatModels.ChatOpenAI",
            "model" => "gpt-4"
          },
          "tools" => [],
          "middleware" => []
        }
      }

      # Should still work (assumes version 1)
      {:ok, {agent, state}} = StateSerializer.deserialize_server_state(serialized)

      assert agent.agent_id == "agent-123"
      assert state.messages == []
    end
  end

  describe "round-trip serialization" do
    test "serialize and deserialize preserves data" do
      {:ok, model} = ChatOpenAI.new(%{model: "gpt-4", api_key: "test-key"})

      {:ok, agent} =
        Agent.new(%{
          agent_id: "agent-123",
          model: model,
          base_system_prompt: "You are helpful"
        })

      msg1 = Message.new_user!("Hello")
      msg2 = Message.new_assistant!(%{content: "Hi there"})

      {:ok, todo1} = Todo.new(%{content: "Task 1", status: :in_progress})
      {:ok, todo2} = Todo.new(%{content: "Task 2", status: :pending})

      state =
        State.new!(%{
          messages: [msg1, msg2],
          todos: [todo1, todo2],
          metadata: %{session_id: "session-1", user_id: 42}
        })

      # Serialize
      serialized = StateSerializer.serialize_server_state(agent, state)

      # Deserialize
      {:ok, {restored_agent, restored_state}} =
        StateSerializer.deserialize_server_state(serialized)

      # Compare agent
      assert restored_agent.agent_id == agent.agent_id

      # System prompt will be re-assembled with middleware, so just check it starts with base prompt
      assert String.starts_with?(restored_agent.assembled_system_prompt, "You are helpful")
      assert restored_agent.model.model == agent.model.model

      # Compare state - pattern match to ensure same number of elements
      assert [_msg1, _msg2] = restored_state.messages
      assert [_orig_msg1, _orig_msg2] = state.messages

      Enum.zip(restored_state.messages, state.messages)
      |> Enum.each(fn {restored_msg, original_msg} ->
        assert restored_msg.role == original_msg.role
        assert restored_msg.content == original_msg.content
      end)

      assert [_todo1, _todo2] = restored_state.todos
      assert [_orig_todo1, _orig_todo2] = state.todos

      Enum.zip(restored_state.todos, state.todos)
      |> Enum.each(fn {restored_todo, original_todo} ->
        assert restored_todo.content == original_todo.content
        assert restored_todo.status == original_todo.status
      end)

      # Metadata from JSONB will have string keys
      assert restored_state.metadata["session_id"] == "session-1"
      assert restored_state.metadata["user_id"] == 42
    end

    test "serialize and deserialize with complex messages" do
      {:ok, model} = ChatOpenAI.new(%{model: "gpt-4", api_key: "test-key"})
      {:ok, agent} = Agent.new(%{agent_id: "agent-123", model: model})

      msg1 = Message.new_user!("Calculate 2 + 2")

      {:ok, tool_call} =
        ToolCall.new(%{
          call_id: "call-1",
          type: :function,
          name: "calculator",
          arguments: %{"expression" => "2 + 2"},
          status: :complete
        })

      {:ok, msg2} = Message.new(%{role: :assistant, content: nil, tool_calls: [tool_call]})

      {:ok, tool_result} =
        ToolResult.new(%{
          type: :function,
          tool_call_id: "call-1",
          name: "calculator",
          content: "4"
        })

      {:ok, msg3} = Message.new_tool_result(%{tool_results: [tool_result]})

      msg4 = Message.new_assistant!(%{content: "The result is 4"})

      state = State.new!(%{messages: [msg1, msg2, msg3, msg4]})

      # Serialize
      serialized = StateSerializer.serialize_server_state(agent, state)

      # Deserialize
      {:ok, {_restored_agent, restored_state}} =
        StateSerializer.deserialize_server_state(serialized)

      # Check messages
      assert [restored_msg1, restored_msg2, restored_msg3, restored_msg4] =
               restored_state.messages

      assert restored_msg1.role == :user
      # Content is converted to ContentPart list by Message.new
      assert [%LangChain.Message.ContentPart{content: "Calculate 2 + 2"}] = restored_msg1.content

      assert restored_msg2.role == :assistant
      assert restored_msg2.content == nil
      assert [restored_tool_call] = restored_msg2.tool_calls
      assert restored_tool_call.name == "calculator"

      assert restored_msg3.role == :tool
      assert restored_msg3.content == nil
      assert [_tool_result] = restored_msg3.tool_results

      assert restored_msg4.role == :assistant
      # Content is converted to ContentPart list by Message.new
      assert [%LangChain.Message.ContentPart{content: "The result is 4"}] = restored_msg4.content
    end
  end

  describe "string keys consistency" do
    test "all map keys in serialized output are strings" do
      {:ok, model} = ChatOpenAI.new(%{model: "gpt-4", api_key: "test-key"})

      {:ok, agent} =
        Agent.new(%{
          agent_id: "agent-123",
          model: model,
          base_system_prompt: "You are helpful"
        })

      msg = Message.new_user!("Hello")
      {:ok, todo} = Todo.new(%{content: "Task", status: :pending})

      state =
        State.new!(%{
          messages: [msg],
          todos: [todo],
          metadata: %{key: "value", nested: %{inner: "data"}}
        })

      serialized = StateSerializer.serialize_server_state(agent, state)

      # Check all keys are strings recursively
      assert all_keys_are_strings?(serialized)
    end

    test "deserialization works with string keys from JSONB" do
      # Simulate data coming from PostgreSQL JSONB (all string keys)
      jsonb_data = %{
        "version" => 1,
        "agent_id" => "agent-123",
        "state" => %{
          "messages" => [
            %{
              "role" => "user",
              "content" => "Hello",
              "status" => "complete"
            }
          ],
          "todos" => [],
          "metadata" => %{
            "string_key" => "value",
            "number_key" => 42,
            "nested" => %{"inner_key" => "inner_value"}
          }
        },
        "agent_config" => %{
          "agent_id" => "agent-123",
          "model" => %{
            "module" => "Elixir.LangChain.ChatModels.ChatOpenAI",
            "model" => "gpt-4"
          }
        },
        "serialized_at" => "2025-11-29T10:30:00Z"
      }

      # Should deserialize successfully
      {:ok, {agent, state}} = StateSerializer.deserialize_server_state(jsonb_data)

      assert agent.agent_id == "agent-123"
      assert [_message] = state.messages
      # Metadata stays with string keys (from JSONB)
      assert state.metadata["string_key"] == "value"
      assert state.metadata["number_key"] == 42
      assert state.metadata["nested"]["inner_key"] == "inner_value"
    end
  end

  # Helper function to check all keys are strings recursively
  defp all_keys_are_strings?(struct) when is_struct(struct) do
    # Structs are acceptable as values, skip checking them
    true
  end

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
