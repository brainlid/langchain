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
    # Generate unique agent_id if not provided
    agent_id = Keyword.get(opts, :agent_id, "test-agent-#{System.unique_integer([:positive])}")
    opts = Keyword.put(opts, :agent_id, agent_id)

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
      agent_id = agent.agent_id
      initial_state = State.new!(%{messages: [Message.new_user!("Hello")]})

      assert {:ok, pid} =
               AgentServer.start_link(
                 agent: agent,
                 initial_state: initial_state,
                 name: AgentServer.get_name(agent_id),
                 pubsub: nil
               )

      assert Process.alive?(pid)

      # Verify state is accessible using agent_id
      state = AgentServer.get_state(agent_id)
      assert length(state.messages) == 1
    end

    test "starts server with default empty state" do
      agent = create_test_agent()
      agent_id = agent.agent_id

      assert {:ok, _pid} =
               AgentServer.start_link(
                 agent: agent,
                 name: AgentServer.get_name(agent_id),
                 pubsub: nil
               )

      state = AgentServer.get_state(agent_id)
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
      agent_id = agent.agent_id

      {:ok, _pid} =
               AgentServer.start_link(
                 agent: agent,
                 name: AgentServer.get_name(agent_id),
                 pubsub: nil
               )

      assert AgentServer.get_status(agent_id) == :idle
    end
  end

  describe "get_state/1" do
    test "returns current state" do
      agent = create_test_agent()
      agent_id = agent.agent_id
      msg = Message.new_user!("Test")
      initial_state = State.new!(%{messages: [msg]})

      {:ok, _pid} =
        AgentServer.start_link(
          agent: agent,
          initial_state: initial_state,
          name: AgentServer.get_name(agent_id),
          pubsub: nil
        )

      state = AgentServer.get_state(agent_id)
      assert length(state.messages) == 1
      assert hd(state.messages) == msg
    end
  end

  describe "get_status/1" do
    test "returns current status" do
      agent = create_test_agent()
      agent_id = agent.agent_id

      {:ok, _pid} =
        AgentServer.start_link(
          agent: agent,
          name: AgentServer.get_name(agent_id),
          pubsub: nil
        )

      assert AgentServer.get_status(agent_id) == :idle
    end
  end

  describe "get_info/1" do
    test "returns comprehensive server info" do
      agent = create_test_agent()
      agent_id = agent.agent_id
      initial_state = State.new!(%{messages: [Message.new_user!("Test")]})

      {:ok, _pid} =
        AgentServer.start_link(
          agent: agent,
          initial_state: initial_state,
          name: AgentServer.get_name(agent_id),
          pubsub: nil
        )

      info = AgentServer.get_info(agent_id)

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
      agent_id = agent.agent_id
      {:ok, agent: agent, agent_id: agent_id}
    end

    test "executes agent successfully", %{agent: agent, agent_id: agent_id} do
      initial_state = State.new!(%{messages: [Message.new_user!("Hello")]})

      Agent
      |> expect(:execute, fn ^agent, state ->
        # Simulate adding an assistant response
        new_state = State.add_message(state, Message.new_assistant!(%{content: "Hi there!"}))
        {:ok, new_state}
      end)

      {:ok, _pid} =
        AgentServer.start_link(
          agent: agent,
          initial_state: initial_state,
          name: AgentServer.get_name(agent_id),
          pubsub: nil
        )

      assert :ok = AgentServer.execute(agent_id)

      # Wait for execution to complete
      Process.sleep(50)

      assert AgentServer.get_status(agent_id) == :completed
      state = AgentServer.get_state(agent_id)
      assert length(state.messages) == 2
    end

    test "transitions to running status immediately", %{agent: agent, agent_id: agent_id} do
      initial_state = State.new!(%{messages: [Message.new_user!("Hello")]})

      Agent
      |> stub(:execute, fn _agent, state ->
        Process.sleep(100)
        {:ok, state}
      end)

      {:ok, _pid} =
        AgentServer.start_link(
          agent: agent,
          initial_state: initial_state,
          name: AgentServer.get_name(agent_id),
          pubsub: nil
        )

      assert :ok = AgentServer.execute(agent_id)
      # Should immediately be running
      assert AgentServer.get_status(agent_id) == :running
    end

    test "returns error if not idle", %{agent: agent, agent_id: agent_id} do
      initial_state = State.new!(%{messages: [Message.new_user!("Hello")]})

      Agent
      |> stub(:execute, fn _agent, state ->
        Process.sleep(100)
        {:ok, state}
      end)

      {:ok, _pid} =
        AgentServer.start_link(
          agent: agent,
          initial_state: initial_state,
          name: AgentServer.get_name(agent_id),
          pubsub: nil
        )

      # First execution succeeds
      assert :ok = AgentServer.execute(agent_id)

      # Second execution while running fails
      assert {:error, _} = AgentServer.execute(agent_id)
    end

    test "handles agent execution error", %{agent: agent, agent_id: agent_id} do
      initial_state = State.new!(%{messages: [Message.new_user!("Hello")]})

      Agent
      |> expect(:execute, fn ^agent, _state ->
        {:error, "Something went wrong"}
      end)

      {:ok, _pid} =
        AgentServer.start_link(
          agent: agent,
          initial_state: initial_state,
          name: AgentServer.get_name(agent_id),
          pubsub: nil
        )

      assert :ok = AgentServer.execute(agent_id)

      # Wait for execution to complete
      Process.sleep(50)

      assert AgentServer.get_status(agent_id) == :error
      info = AgentServer.get_info(agent_id)
      assert info.error == "Something went wrong"
    end

    test "handles agent interrupt", %{agent: agent, agent_id: agent_id} do
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

      {:ok, _pid} =
        AgentServer.start_link(
          agent: agent,
          initial_state: initial_state,
          name: AgentServer.get_name(agent_id),
          pubsub: nil
        )

      assert :ok = AgentServer.execute(agent_id)

      # Wait for execution to complete
      Process.sleep(50)

      assert AgentServer.get_status(agent_id) == :interrupted
      info = AgentServer.get_info(agent_id)
      assert info.interrupt_data == interrupt_data
    end
  end

  describe "resume/2" do
    setup do
      agent = create_test_agent()
      agent_id = agent.agent_id

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

      {:ok, _pid} =
        AgentServer.start_link(
          agent: agent,
          initial_state: initial_state,
          name: AgentServer.get_name(agent_id),
          pubsub: nil
        )

      :ok = AgentServer.execute(agent_id)
      Process.sleep(50)

      {:ok, agent: agent, agent_id: agent_id}
    end

    test "resumes execution after interrupt", %{agent: agent, agent_id: agent_id} do
      decisions = [%{type: :approve}]

      Agent
      |> expect(:resume, fn ^agent, state, ^decisions ->
        new_state = State.add_message(state, Message.new_assistant!(%{content: "Done"}))
        {:ok, new_state}
      end)

      assert :ok = AgentServer.resume(agent_id, decisions)

      # Wait for resume to complete
      Process.sleep(50)

      assert AgentServer.get_status(agent_id) == :completed
    end

    test "returns error if not interrupted", %{agent: _agent, agent_id: setup_agent_id} do
      # Stop the server from setup first since it uses the default name
      :ok = AgentServer.stop(setup_agent_id)

      # Create a new idle server
      agent = create_test_agent()
      new_agent_id = agent.agent_id

      {:ok, _pid} =
        AgentServer.start_link(
          agent: agent,
          name: AgentServer.get_name(new_agent_id),
          pubsub: nil
        )

      decisions = [%{type: :approve}]
      assert {:error, _} = AgentServer.resume(new_agent_id, decisions)
    end

    test "handles resume error", %{agent: agent, agent_id: agent_id} do
      decisions = [%{type: :approve}]

      Agent
      |> expect(:resume, fn ^agent, _state, ^decisions ->
        {:error, "Resume failed"}
      end)

      assert :ok = AgentServer.resume(agent_id, decisions)

      # Wait for resume to complete
      Process.sleep(50)

      assert AgentServer.get_status(agent_id) == :error
      info = AgentServer.get_info(agent_id)
      assert info.error == "Resume failed"
    end
  end

  describe "PubSub events" do
    setup do
      # Start a test PubSub with supervisor
      pubsub_name = :"test_pubsub_#{:erlang.unique_integer([:positive])}"
      {:ok, _} = start_supervised({Phoenix.PubSub, name: pubsub_name})

      agent = create_test_agent()
      agent_id = agent.agent_id
      initial_state = State.new!()

      {:ok, _pid} =
        AgentServer.start_link(
          agent: agent,
          initial_state: initial_state,
          name: AgentServer.get_name(agent_id),
          pubsub: Phoenix.PubSub,
          pubsub_name: pubsub_name,
          id: "test_agent_#{:erlang.unique_integer([:positive])}"
        )

      # Subscribe to events
      :ok = AgentServer.subscribe(agent_id)

      {:ok, agent: agent, agent_id: agent_id, pubsub_name: pubsub_name}
    end

    test "broadcasts status changes", %{agent: agent, agent_id: agent_id} do
      Agent
      |> expect(:execute, fn ^agent, state ->
        {:ok, state}
      end)

      :ok = AgentServer.execute(agent_id)

      # Should receive running status
      assert_receive {:status_changed, :running, nil}, 100

      # Should receive completed status
      assert_receive {:status_changed, :completed, _state}, 200
    end

    # NOTE: File events are NOT broadcast by AgentServer anymore
    # Files are managed by FileSystemServer which has its own event handling

    test "broadcasts todo created event", %{agent: agent, agent_id: agent_id} do
      Agent
      |> expect(:execute, fn ^agent, state ->
        todo = Todo.new!(%{content: "Write tests", status: :pending})
        new_state = State.put_todo(state, todo)
        {:ok, new_state}
      end)

      :ok = AgentServer.execute(agent_id)

      assert_receive {:status_changed, :running, nil}, 100
      assert_receive {:todo_created, todo}, 200
      assert todo.content == "Write tests"
    end

    test "broadcasts todo updated event", %{agent: agent, agent_id: agent_id} do
      # Set initial todo
      todo = Todo.new!(%{id: "test_id", content: "Write tests", status: :pending})
      initial_state = State.new!() |> State.put_todo(todo)

      :sys.replace_state(GenServer.whereis(AgentServer.get_name(agent_id)), fn server_state ->
        %{server_state | state: initial_state}
      end)

      Agent
      |> expect(:execute, fn ^agent, state ->
        updated_todo = Todo.new!(%{id: "test_id", content: "Write tests", status: :completed})
        new_state = State.put_todo(state, updated_todo)
        {:ok, new_state}
      end)

      :ok = AgentServer.execute(agent_id)

      assert_receive {:status_changed, :running, nil}, 100
      assert_receive {:todo_updated, updated_todo}, 200
      assert updated_todo.status == :completed
    end

    test "broadcasts todo deleted event", %{agent: agent, agent_id: agent_id} do
      # Set initial todo
      todo = Todo.new!(%{id: "test_id", content: "Write tests", status: :pending})
      initial_state = State.new!() |> State.put_todo(todo)

      :sys.replace_state(GenServer.whereis(AgentServer.get_name(agent_id)), fn server_state ->
        %{server_state | state: initial_state}
      end)

      Agent
      |> expect(:execute, fn ^agent, state ->
        new_state = State.delete_todo(state, "test_id")
        {:ok, new_state}
      end)

      :ok = AgentServer.execute(agent_id)

      assert_receive {:status_changed, :running, nil}, 100
      assert_receive {:todo_deleted, "test_id"}, 200
    end

    test "broadcasts interrupt status with data", %{agent: agent, agent_id: agent_id} do
      interrupt_data = %{
        action_requests: [%{tool_name: "write_file"}],
        review_configs: %{}
      }

      Agent
      |> expect(:execute, fn ^agent, state ->
        {:interrupt, state, interrupt_data}
      end)

      :ok = AgentServer.execute(agent_id)

      assert_receive {:status_changed, :running, nil}, 100
      assert_receive {:status_changed, :interrupted, ^interrupt_data}, 200
    end

    test "broadcasts error status", %{agent: agent, agent_id: agent_id} do
      Agent
      |> expect(:execute, fn ^agent, _state ->
        {:error, "Test error"}
      end)

      :ok = AgentServer.execute(agent_id)

      assert_receive {:status_changed, :running, nil}, 100
      assert_receive {:status_changed, :error, "Test error"}, 200
    end

    # NOTE: File events are NOT broadcast by AgentServer anymore
    # Files are managed by FileSystemServer
  end

  describe "stop/1" do
    test "stops the server gracefully" do
      agent = create_test_agent()
      agent_id = agent.agent_id

      {:ok, pid} =
        AgentServer.start_link(
          agent: agent,
          name: AgentServer.get_name(agent_id),
          pubsub: nil
        )

      assert Process.alive?(pid)
      assert :ok = AgentServer.stop(agent_id)
      refute Process.alive?(pid)
    end
  end
end
