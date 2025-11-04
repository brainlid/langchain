defmodule LangChain.Agents.IntegrationTest do
  use ExUnit.Case
  use Mimic

  alias LangChain.Agents.Agent
  alias LangChain.Agents.State
  alias LangChain.Agents.FileSystemServer
  alias LangChain.ChatModels.ChatOpenAI
  alias LangChain.Message
  alias LangChain.Message.ToolCall

  describe "Agent with LLMChain integration" do
    setup do
      # Create a mock ChatOpenAI model
      model = ChatOpenAI.new!(%{model: "gpt-4", stream: false, temperature: 0})

      # Start the Registry for FileSystemServer
      {:ok, _registry} =
        start_supervised({Registry, keys: :unique, name: LangChain.Agents.Registry})

      {:ok, model: model}
    end

    test "executes successfully with tool calls that update todos", %{model: model} do
      # Stub the call to return a tool call for write_todos
      stub(ChatOpenAI, :call, fn _model, messages, _tools ->
        # First call: LLM calls write_todos tool
        case length(messages) do
          # First LLM call - return assistant message with tool call
          2 ->
            {:ok,
             [
               Message.new_assistant!(%{
                 role: :assistant,
                 tool_calls: [
                   ToolCall.new!(%{
                     call_id: "call_123",
                     name: "write_todos",
                     arguments: %{
                       "merge" => false,
                       "todos" => [
                         %{
                           "id" => "task_1",
                           "content" => "Write integration test",
                           "status" => "completed"
                         },
                         %{
                           "id" => "task_2",
                           "content" => "Verify state updates",
                           "status" => "in_progress"
                         }
                       ]
                     }
                   })
                 ]
               })
             ]}

          # Second LLM call (after tool execution) - return final response
          _ ->
            {:ok, [Message.new_assistant!("I've created the TODO list for you.")]}
        end
      end)

      # Create agent with TodoList middleware
      {:ok, agent} =
        Agent.new(
          model: model,
          system_prompt: "You are a helpful assistant."
        )

      # Initial state with user message
      initial_state =
        State.new!(%{
          messages: [
            Message.new_user!("Create a TODO list with 2 tasks")
          ]
        })

      # Execute the agent
      {:ok, final_state} = Agent.execute(agent, initial_state)

      # Verify todos were updated
      assert length(final_state.todos) == 2

      # Find the todos
      task_1 = Enum.find(final_state.todos, &(&1.id == "task_1"))
      task_2 = Enum.find(final_state.todos, &(&1.id == "task_2"))

      assert task_1.content == "Write integration test"
      assert task_1.status == :completed

      assert task_2.content == "Verify state updates"
      assert task_2.status == :in_progress

      # Verify messages were added
      assert length(final_state.messages) > 1

      # The last message should be the assistant's final response
      last_message = List.last(final_state.messages)
      assert last_message.role == :assistant
      assert Message.ContentPart.content_to_string(last_message.content) =~ "TODO"
    end

    test "executes successfully with filesystem tool calls", %{model: model} do
      # Stub the call to handle write_file tool
      stub(ChatOpenAI, :call, fn _model, messages, _tools ->
        case length(messages) do
          # First LLM call - return assistant message with write_file tool call
          2 ->
            {:ok,
             [
               Message.new_assistant!(%{
                 role: :assistant,
                 tool_calls: [
                   ToolCall.new!(%{
                     call_id: "call_456",
                     name: "write_file",
                     arguments: %{
                       "file_path" => "/test.txt",
                       "content" => "Hello, World!"
                     }
                   })
                 ]
               })
             ]}

          # Second LLM call (after tool execution) - return final response
          _ ->
            {:ok, [Message.new_assistant!("I've created the file test.txt for you.")]}
        end
      end)

      # Use a specific agent_id for this test
      agent_id = "test-agent-#{System.unique_integer([:positive])}"

      # Start FileSystemServer first (before creating agent)
      {:ok, fs_pid} = FileSystemServer.start_link(agent_id: agent_id)

      # Verify the server is running
      assert Process.alive?(fs_pid)

      # Verify we can reach the server
      assert FileSystemServer.whereis(agent_id) == fs_pid

      # Test that write_file works directly (paths must start with /)
      assert :ok == FileSystemServer.write_file(agent_id, "/direct_test.txt", "test content")
      assert FileSystemServer.file_exists?(agent_id, "/direct_test.txt")

      # Create agent with Filesystem middleware
      {:ok, agent} =
        Agent.new(
          agent_id: agent_id,
          model: model,
          system_prompt: "You are a helpful assistant."
        )

      # Initial state with user message
      initial_state =
        State.new!(%{
          messages: [
            Message.new_user!("Create a file called test.txt with 'Hello, World!' in it")
          ]
        })

      # Execute the agent
      {:ok, _final_state} = Agent.execute(agent, initial_state)

      # Verify file was created using FileSystemServer
      files = FileSystemServer.list_files(agent_id)
      assert "/test.txt" in files

      # Get the file content from FileSystemServer
      {:ok, file_content} = FileSystemServer.read_file(agent_id, "/test.txt")
      assert file_content == "Hello, World!"
    end

    test "handles multiple tool calls in sequence", %{model: model} do
      call_count = :counters.new(1, [])

      # Stub to handle multiple tool calls
      stub(ChatOpenAI, :call, fn _model, _messages, _tools ->
        count = :counters.get(call_count, 1)
        :counters.add(call_count, 1, 1)

        case count do
          # First call: create todos
          0 ->
            {:ok,
             [
               Message.new_assistant!(%{
                 role: :assistant,
                 tool_calls: [
                   ToolCall.new!(%{
                     call_id: "call_1",
                     name: "write_todos",
                     arguments: %{
                       "merge" => false,
                       "todos" => [
                         %{"id" => "task_1", "content" => "Write file", "status" => "pending"}
                       ]
                     }
                   })
                 ]
               })
             ]}

          # Second call: create file
          1 ->
            {:ok,
             [
               Message.new_assistant!(%{
                 role: :assistant,
                 tool_calls: [
                   ToolCall.new!(%{
                     call_id: "call_2",
                     name: "write_file",
                     arguments: %{
                       "file_path" => "/data.txt",
                       "content" => "Some data"
                     }
                   })
                 ]
               })
             ]}

          # Third call: update todos
          2 ->
            {:ok,
             [
               Message.new_assistant!(%{
                 role: :assistant,
                 tool_calls: [
                   ToolCall.new!(%{
                     call_id: "call_3",
                     name: "write_todos",
                     arguments: %{
                       "merge" => true,
                       "todos" => [
                         %{"id" => "task_1", "content" => "Write file", "status" => "completed"}
                       ]
                     }
                   })
                 ]
               })
             ]}

          # Final call: respond to user
          _ ->
            {:ok, [Message.new_assistant!("All done! I've completed the task.")]}
        end
      end)

      # Use a specific agent_id for this test
      agent_id = "test-agent-#{System.unique_integer([:positive])}"

      # Start FileSystemServer first (before creating agent)
      {:ok, _fs_pid} = FileSystemServer.start_link(agent_id: agent_id)

      # Create agent
      {:ok, agent} = Agent.new(agent_id: agent_id, model: model)

      # Initial state
      initial_state =
        State.new!(%{
          messages: [
            Message.new_user!("Create a file and track your progress")
          ]
        })

      # Execute
      {:ok, final_state} = Agent.execute(agent, initial_state)

      # Verify both state updates occurred
      assert length(final_state.todos) == 1
      todo = List.first(final_state.todos)
      assert todo.status == :completed

      # Verify file was created using FileSystemServer
      files = FileSystemServer.list_files(agent.agent_id)
      assert "/data.txt" in files

      {:ok, file_content} = FileSystemServer.read_file(agent.agent_id, "/data.txt")
      assert file_content == "Some data"
    end

    test "handles tool errors gracefully", %{model: model} do
      # Stub to try to write an existing file
      stub(ChatOpenAI, :call, fn _model, messages, _tools ->
        case length(messages) do
          2 ->
            # First call: try to create file
            {:ok,
             [
               Message.new_assistant!(%{
                 role: :assistant,
                 tool_calls: [
                   ToolCall.new!(%{
                     call_id: "call_1",
                     name: "write_file",
                     arguments: %{"file_path" => "/existing.txt", "content" => "content"}
                   })
                 ]
               })
             ]}

          3 ->
            # After successful write, try again (will fail)
            {:ok,
             [
               Message.new_assistant!(%{
                 role: :assistant,
                 tool_calls: [
                   ToolCall.new!(%{
                     call_id: "call_2",
                     name: "write_file",
                     arguments: %{"file_path" => "/existing.txt", "content" => "new content"}
                   })
                 ]
               })
             ]}

          _ ->
            {:ok, [Message.new_assistant!("The file already exists.")]}
        end
      end)

      # Use a specific agent_id for this test
      agent_id = "test-agent-#{System.unique_integer([:positive])}"

      # Start FileSystemServer first (before creating agent)
      {:ok, _fs_pid} = FileSystemServer.start_link(agent_id: agent_id)

      {:ok, agent} = Agent.new(agent_id: agent_id, model: model)

      initial_state =
        State.new!(%{
          messages: [Message.new_user!("Write a file twice")]
        })

      # Execute - should not crash
      {:ok, _final_state} = Agent.execute(agent, initial_state)

      # Verify the file was created only once using FileSystemServer
      files = FileSystemServer.list_files(agent.agent_id)
      assert "/existing.txt" in files

      {:ok, file_content} = FileSystemServer.read_file(agent.agent_id, "/existing.txt")
      assert file_content == "content"
    end
  end
end
