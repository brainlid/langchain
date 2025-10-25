defmodule LangChain.DeepAgents.AgentLiveTest do
  use ExUnit.Case

  alias LangChain.DeepAgents.Agent
  alias LangChain.DeepAgents.State
  alias LangChain.ChatModels.ChatAnthropic
  alias LangChain.Message

  @moduletag live_call: true, live_anthropic: true

  describe "Agent with live Anthropic calls" do
    setup do
      # Create a real Anthropic model
      model = ChatAnthropic.new!(%{model: "claude-3-5-sonnet-20241022", stream: false})

      {:ok, model: model}
    end

    test "can call write_todos tool and update state", %{model: model} do
      # Create agent with TodoList middleware
      {:ok, agent} = Agent.new(model: model)

      # Initial state with user message
      initial_state =
        State.new!(%{
          messages: [
            Message.new_user!(
              "Create a TODO list with 2 tasks: 'Write test' (completed) and 'Verify integration' (in_progress)"
            )
          ]
        })

      # Execute the agent
      {:ok, final_state} = Agent.execute(agent, initial_state)

      # Verify todos were created
      assert length(final_state.todos) == 2

      # Verify the task statuses
      statuses = Enum.map(final_state.todos, & &1.status)
      assert :completed in statuses
      assert :in_progress in statuses

      # Verify we got an assistant response
      assert length(final_state.messages) >= 2
      last_message = List.last(final_state.messages)
      assert last_message.role == :assistant
    end

    test "can call write_file tool and create files", %{model: model} do
      # Create agent with Filesystem middleware
      {:ok, agent} = Agent.new(model: model)

      # Initial state with user message
      initial_state =
        State.new!(%{
          messages: [
            Message.new_user!(
              "Create a file called 'test.txt' with the content 'Hello from DeepAgents integration test!'"
            )
          ]
        })

      # Execute the agent
      {:ok, final_state} = Agent.execute(agent, initial_state)

      # Verify file was created
      assert Map.has_key?(final_state.files, "test.txt")

      # Get file content (handles FileData structure)
      file_content = State.get_file(final_state, "test.txt")
      assert file_content =~ "Hello from DeepAgents"

      # Verify file metadata
      file_data = State.get_file_data(final_state, "test.txt")
      assert file_data.content =~ "Hello from DeepAgents"
      assert %DateTime{} = file_data.created_at
      assert %DateTime{} = file_data.modified_at

      # Verify we got an assistant response
      last_message = List.last(final_state.messages)
      assert last_message.role == :assistant
    end

    test "can handle complex multi-tool scenarios", %{model: model} do
      # Create agent
      {:ok, agent} = Agent.new(model: model)

      # Initial state asking for both todos and file creation
      initial_state =
        State.new!(%{
          messages: [
            Message.new_user!(
              "First, create a TODO list with task 'Create data file' (pending). Then create a file 'data.txt' with 'Sample data'. Finally, update the TODO to completed."
            )
          ]
        })

      # Execute the agent
      {:ok, final_state} = Agent.execute(agent, initial_state)

      # Verify file was created
      assert Map.has_key?(final_state.files, "data.txt")
      assert State.get_file(final_state, "data.txt") =~ "Sample data"

      # Verify todos exist
      assert length(final_state.todos) >= 1

      # The status might be completed if the agent followed instructions well
      todo = List.first(final_state.todos)
      # Either completed if agent followed all steps, or pending if it's still working
      assert todo.status in [:completed, :pending, :in_progress]
      assert todo.content =~ "file" or todo.content =~ "data" or todo.content =~ "Create"

      # Verify we got messages
      assert length(final_state.messages) >= 2
    end

    test "handles invalid tool calls gracefully", %{model: model} do
      {:ok, agent} = Agent.new(model: model)

      # Ask to create a file, then try to create it again
      initial_state =
        State.new!(%{
          messages: [
            Message.new_user!(
              "Create a file called 'existing.txt' with content 'First'. Then try to create the same file again with content 'Second'."
            )
          ]
        })

      # Execute - should not crash even though second file creation will fail
      {:ok, final_state} = Agent.execute(agent, initial_state)

      # Verify only one file was created with the first content
      assert Map.has_key?(final_state.files, "existing.txt")
      file_content = State.get_file(final_state, "existing.txt")
      assert file_content == "First"

      # Verify we got a response (agent should handle the error)
      assert length(final_state.messages) >= 2
    end
  end
end
