defmodule LangChain.Agents.AgentServerTest do
  use ExUnit.Case, async: false
  use Mimic

  alias LangChain.Agents.{Agent, AgentServer, State, Todo}
  alias LangChain.ChatModels.ChatAnthropic
  alias LangChain.Message

  setup :set_mimic_global
  setup :verify_on_exit!

  setup_all do
    # Copy Agent module for mocking
    Mimic.copy(Agent)
    :ok
  end

  # Helper to create a mock model
  defp mock_model do
    ChatAnthropic.new!(%{
      model: "claude-3-5-sonnet-20241022",
      api_key: "test_key"
    })
  end

  # Helper to create a simple agent
  defp create_test_agent(opts \\ []) do
    Agent.new!(
      Keyword.merge(
        [
          model: mock_model(),
          system_prompt: "Test agent",
          replace_default_middleware: true,
          middleware: []
        ],
        opts
      )
    )
  end

  describe "start_link/1" do
    test "starts server with agent and initial state" do
      agent = create_test_agent()
      initial_state = State.new!(%{messages: [Message.new_user!("Hello")]})

      assert {:ok, pid} =
               AgentServer.start_link(
                 agent: agent,
                 initial_state: initial_state,
                 pubsub: nil
               )

      assert Process.alive?(pid)

      # Verify state is accessible
      state = AgentServer.get_state(pid)
      assert length(state.messages) == 1
    end

    test "starts server with default empty state" do
      agent = create_test_agent()

      assert {:ok, pid} = AgentServer.start_link(agent: agent, pubsub: nil)

      state = AgentServer.get_state(pid)
      assert state.messages == []
    end

    test "starts with named registration" do
      agent = create_test_agent()

      assert {:ok, pid} =
               AgentServer.start_link(
                 agent: agent,
                 name: :test_agent_server,
                 pubsub: nil
               )

      assert Process.whereis(:test_agent_server) == pid
    end

    test "initializes with idle status" do
      agent = create_test_agent()

      {:ok, pid} = AgentServer.start_link(agent: agent, pubsub: nil)

      assert AgentServer.get_status(pid) == :idle
    end
  end

  describe "get_state/1" do
    test "returns current state" do
      agent = create_test_agent()
      msg = Message.new_user!("Test")
      initial_state = State.new!(%{messages: [msg]})

      {:ok, pid} = AgentServer.start_link(agent: agent, initial_state: initial_state, pubsub: nil)

      state = AgentServer.get_state(pid)
      assert length(state.messages) == 1
      assert hd(state.messages) == msg
    end
  end

  describe "get_status/1" do
    test "returns current status" do
      agent = create_test_agent()
      {:ok, pid} = AgentServer.start_link(agent: agent, pubsub: nil)

      assert AgentServer.get_status(pid) == :idle
    end
  end

  describe "get_info/1" do
    test "returns comprehensive server info" do
      agent = create_test_agent()
      initial_state = State.new!(%{messages: [Message.new_user!("Test")]})

      {:ok, pid} = AgentServer.start_link(agent: agent, initial_state: initial_state, pubsub: nil)

      info = AgentServer.get_info(pid)

      assert info.status == :idle
      assert info.state.messages == initial_state.messages
      assert info.interrupt_data == nil
      assert info.error == nil
    end
  end

  describe "execute/1" do
    setup do
      # Mock the Agent.execute to avoid real LLM calls
      agent = create_test_agent()
      {:ok, agent: agent}
    end

    test "executes agent successfully", %{agent: agent} do
      initial_state = State.new!(%{messages: [Message.new_user!("Hello")]})

      Agent
      |> expect(:execute, fn ^agent, state ->
        # Simulate adding an assistant response
        new_state = State.add_message(state, Message.new_assistant!(%{content: "Hi there!"}))
        {:ok, new_state}
      end)

      {:ok, pid} = AgentServer.start_link(agent: agent, initial_state: initial_state, pubsub: nil)

      assert :ok = AgentServer.execute(pid)

      # Wait for execution to complete
      Process.sleep(50)

      assert AgentServer.get_status(pid) == :completed
      state = AgentServer.get_state(pid)
      assert length(state.messages) == 2
    end

    test "transitions to running status immediately", %{agent: agent} do
      initial_state = State.new!(%{messages: [Message.new_user!("Hello")]})

      Agent
      |> stub(:execute, fn _agent, state ->
        Process.sleep(100)
        {:ok, state}
      end)

      {:ok, pid} = AgentServer.start_link(agent: agent, initial_state: initial_state, pubsub: nil)

      assert :ok = AgentServer.execute(pid)
      # Should immediately be running
      assert AgentServer.get_status(pid) == :running
    end

    test "returns error if not idle", %{agent: agent} do
      initial_state = State.new!(%{messages: [Message.new_user!("Hello")]})

      Agent
      |> stub(:execute, fn _agent, state ->
        Process.sleep(100)
        {:ok, state}
      end)

      {:ok, pid} = AgentServer.start_link(agent: agent, initial_state: initial_state, pubsub: nil)

      # First execution succeeds
      assert :ok = AgentServer.execute(pid)

      # Second execution while running fails
      assert {:error, _} = AgentServer.execute(pid)
    end

    test "handles agent execution error", %{agent: agent} do
      initial_state = State.new!(%{messages: [Message.new_user!("Hello")]})

      Agent
      |> expect(:execute, fn ^agent, _state ->
        {:error, "Something went wrong"}
      end)

      {:ok, pid} = AgentServer.start_link(agent: agent, initial_state: initial_state, pubsub: nil)

      assert :ok = AgentServer.execute(pid)

      # Wait for execution to complete
      Process.sleep(50)

      assert AgentServer.get_status(pid) == :error
      info = AgentServer.get_info(pid)
      assert info.error == "Something went wrong"
    end

    test "handles agent interrupt", %{agent: agent} do
      initial_state = State.new!(%{messages: [Message.new_user!("Write file")]})

      interrupt_data = %{
        action_requests: [
          %{tool_name: "write_file", arguments: %{"path" => "test.txt", "content" => "data"}}
        ],
        review_configs: %{}
      }

      Agent
      |> expect(:execute, fn ^agent, state ->
        {:interrupt, state, interrupt_data}
      end)

      {:ok, pid} = AgentServer.start_link(agent: agent, initial_state: initial_state, pubsub: nil)

      assert :ok = AgentServer.execute(pid)

      # Wait for execution to complete
      Process.sleep(50)

      assert AgentServer.get_status(pid) == :interrupted
      info = AgentServer.get_info(pid)
      assert info.interrupt_data == interrupt_data
    end
  end

  describe "resume/2" do
    setup do
      agent = create_test_agent()

      initial_state = State.new!(%{messages: [Message.new_user!("Write file")]})

      interrupt_data = %{
        action_requests: [
          %{tool_name: "write_file", arguments: %{"path" => "test.txt", "content" => "data"}}
        ],
        review_configs: %{}
      }

      # Mock execute to return interrupt
      Agent
      |> expect(:execute, fn ^agent, state ->
        {:interrupt, state, interrupt_data}
      end)

      {:ok, pid} = AgentServer.start_link(agent: agent, initial_state: initial_state, pubsub: nil)
      :ok = AgentServer.execute(pid)
      Process.sleep(50)

      {:ok, agent: agent, pid: pid}
    end

    test "resumes execution after interrupt", %{agent: agent, pid: pid} do
      decisions = [%{type: :approve}]

      Agent
      |> expect(:resume, fn ^agent, state, ^decisions ->
        new_state = State.add_message(state, Message.new_assistant!(%{content: "Done"}))
        {:ok, new_state}
      end)

      assert :ok = AgentServer.resume(pid, decisions)

      # Wait for resume to complete
      Process.sleep(50)

      assert AgentServer.get_status(pid) == :completed
    end

    test "returns error if not interrupted", %{agent: _agent} do
      # Create a new idle server
      agent = create_test_agent()
      {:ok, pid} = AgentServer.start_link(agent: agent, pubsub: nil)

      decisions = [%{type: :approve}]
      assert {:error, _} = AgentServer.resume(pid, decisions)
    end

    test "handles resume error", %{agent: agent, pid: pid} do
      decisions = [%{type: :approve}]

      Agent
      |> expect(:resume, fn ^agent, _state, ^decisions ->
        {:error, "Resume failed"}
      end)

      assert :ok = AgentServer.resume(pid, decisions)

      # Wait for resume to complete
      Process.sleep(50)

      assert AgentServer.get_status(pid) == :error
      info = AgentServer.get_info(pid)
      assert info.error == "Resume failed"
    end
  end

  describe "PubSub events" do
    setup do
      # Start a test PubSub with supervisor
      pubsub_name = :"test_pubsub_#{:erlang.unique_integer([:positive])}"
      {:ok, _} = start_supervised({Phoenix.PubSub, name: pubsub_name})

      agent = create_test_agent()
      initial_state = State.new!()

      {:ok, pid} =
        AgentServer.start_link(
          agent: agent,
          initial_state: initial_state,
          pubsub: Phoenix.PubSub,
          pubsub_name: pubsub_name,
          id: "test_agent_#{:erlang.unique_integer([:positive])}"
        )

      # Subscribe to events
      :ok = AgentServer.subscribe(pid)

      {:ok, agent: agent, pid: pid, pubsub_name: pubsub_name}
    end

    test "broadcasts status changes", %{agent: agent, pid: pid} do
      Agent
      |> expect(:execute, fn ^agent, state ->
        {:ok, state}
      end)

      :ok = AgentServer.execute(pid)

      # Should receive running status
      assert_receive {:status_changed, :running, nil}, 100

      # Should receive completed status
      assert_receive {:status_changed, :completed, _state}, 200
    end

    test "broadcasts file added event", %{agent: agent, pid: pid} do
      Agent
      |> expect(:execute, fn ^agent, state ->
        new_state = State.put_file(state, "test.txt", "Hello, World!")
        {:ok, new_state}
      end)

      :ok = AgentServer.execute(pid)

      # Should receive status change
      assert_receive {:status_changed, :running, nil}, 100

      # Should receive file added event
      assert_receive {:file_added, "test.txt", "Hello, World!"}, 200

      # Should receive completed status
      assert_receive {:status_changed, :completed, _state}, 200
    end

    test "broadcasts file updated event", %{agent: agent, pid: pid} do
      # Set initial file
      initial_state = State.new!() |> State.put_file("test.txt", "Version 1")

      # Update the server state
      :sys.replace_state(pid, fn server_state ->
        %{server_state | state: initial_state}
      end)

      Agent
      |> expect(:execute, fn ^agent, state ->
        new_state = State.put_file(state, "test.txt", "Version 2")
        {:ok, new_state}
      end)

      :ok = AgentServer.execute(pid)

      assert_receive {:status_changed, :running, nil}, 100
      assert_receive {:file_updated, "test.txt", "Version 2"}, 200
    end

    test "broadcasts file deleted event", %{agent: agent, pid: pid} do
      # Set initial file
      initial_state = State.new!() |> State.put_file("test.txt", "Data")

      :sys.replace_state(pid, fn server_state ->
        %{server_state | state: initial_state}
      end)

      Agent
      |> expect(:execute, fn ^agent, state ->
        new_state = State.delete_file(state, "test.txt")
        {:ok, new_state}
      end)

      :ok = AgentServer.execute(pid)

      assert_receive {:status_changed, :running, nil}, 100
      assert_receive {:file_deleted, "test.txt"}, 200
    end

    test "broadcasts todo created event", %{agent: agent, pid: pid} do
      Agent
      |> expect(:execute, fn ^agent, state ->
        todo = Todo.new!(%{content: "Write tests", status: :pending})
        new_state = State.put_todo(state, todo)
        {:ok, new_state}
      end)

      :ok = AgentServer.execute(pid)

      assert_receive {:status_changed, :running, nil}, 100
      assert_receive {:todo_created, todo}, 200
      assert todo.content == "Write tests"
    end

    test "broadcasts todo updated event", %{agent: agent, pid: pid} do
      # Set initial todo
      todo = Todo.new!(%{id: "test_id", content: "Write tests", status: :pending})
      initial_state = State.new!() |> State.put_todo(todo)

      :sys.replace_state(pid, fn server_state ->
        %{server_state | state: initial_state}
      end)

      Agent
      |> expect(:execute, fn ^agent, state ->
        updated_todo = Todo.new!(%{id: "test_id", content: "Write tests", status: :completed})
        new_state = State.put_todo(state, updated_todo)
        {:ok, new_state}
      end)

      :ok = AgentServer.execute(pid)

      assert_receive {:status_changed, :running, nil}, 100
      assert_receive {:todo_updated, updated_todo}, 200
      assert updated_todo.status == :completed
    end

    test "broadcasts todo deleted event", %{agent: agent, pid: pid} do
      # Set initial todo
      todo = Todo.new!(%{id: "test_id", content: "Write tests", status: :pending})
      initial_state = State.new!() |> State.put_todo(todo)

      :sys.replace_state(pid, fn server_state ->
        %{server_state | state: initial_state}
      end)

      Agent
      |> expect(:execute, fn ^agent, state ->
        new_state = State.delete_todo(state, "test_id")
        {:ok, new_state}
      end)

      :ok = AgentServer.execute(pid)

      assert_receive {:status_changed, :running, nil}, 100
      assert_receive {:todo_deleted, "test_id"}, 200
    end

    test "broadcasts interrupt status with data", %{agent: agent, pid: pid} do
      interrupt_data = %{
        action_requests: [%{tool_name: "write_file"}],
        review_configs: %{}
      }

      Agent
      |> expect(:execute, fn ^agent, state ->
        {:interrupt, state, interrupt_data}
      end)

      :ok = AgentServer.execute(pid)

      assert_receive {:status_changed, :running, nil}, 100
      assert_receive {:status_changed, :interrupted, ^interrupt_data}, 200
    end

    test "broadcasts error status", %{agent: agent, pid: pid} do
      Agent
      |> expect(:execute, fn ^agent, _state ->
        {:error, "Test error"}
      end)

      :ok = AgentServer.execute(pid)

      assert_receive {:status_changed, :running, nil}, 100
      assert_receive {:status_changed, :error, "Test error"}, 200
    end

    test "broadcasts multiple file changes in one execution", %{agent: agent, pid: pid} do
      Agent
      |> expect(:execute, fn ^agent, state ->
        new_state =
          state
          |> State.put_file("file1.txt", "Content 1")
          |> State.put_file("file2.txt", "Content 2")
          |> State.put_file("file3.txt", "Content 3")

        {:ok, new_state}
      end)

      :ok = AgentServer.execute(pid)

      assert_receive {:status_changed, :running, nil}, 100
      assert_receive {:file_added, "file1.txt", "Content 1"}, 200
      assert_receive {:file_added, "file2.txt", "Content 2"}, 200
      assert_receive {:file_added, "file3.txt", "Content 3"}, 200
      assert_receive {:status_changed, :completed, _state}, 200
    end
  end

  describe "stop/1" do
    test "stops the server gracefully" do
      agent = create_test_agent()
      {:ok, pid} = AgentServer.start_link(agent: agent, pubsub: nil)

      assert Process.alive?(pid)
      assert :ok = AgentServer.stop(pid)
      refute Process.alive?(pid)
    end
  end
end
