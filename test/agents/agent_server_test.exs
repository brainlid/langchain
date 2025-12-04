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
      Map.merge(
        %{
          model: mock_model(),
          base_system_prompt: "Test agent",
          replace_default_middleware: true,
          middleware: []
        },
        Enum.into(opts, %{})
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
      |> expect(:execute, fn ^agent, state, _opts ->
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
      |> expect(:execute, fn ^agent, _state, _opts ->
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
      |> expect(:execute, fn ^agent, state, _opts ->
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

  describe "cancel/1" do
    test "cancels a running task" do
      agent = create_test_agent()
      agent_id = agent.agent_id
      initial_state = State.new!(%{messages: [Message.new_user!("Hello")]})

      # Mock execute to take a long time so we can cancel it
      Agent
      |> expect(:execute, fn ^agent, state, _opts ->
        Process.sleep(1_000)
        {:ok, state}
      end)

      {:ok, _pid} =
        AgentServer.start_link(
          agent: agent,
          initial_state: initial_state,
          name: AgentServer.get_name(agent_id),
          pubsub: nil
        )

      # Start execution
      assert :ok = AgentServer.execute(agent_id)
      assert AgentServer.get_status(agent_id) == :running

      # Pause briefly to let the task spin up
      Process.sleep(50)
      # Cancel it
      assert :ok = AgentServer.cancel(agent_id)

      # Should be completed now
      assert AgentServer.get_status(agent_id) == :completed
    end

    test "returns error when not running" do
      agent = create_test_agent()
      agent_id = agent.agent_id

      {:ok, _pid} =
        AgentServer.start_link(
          agent: agent,
          name: AgentServer.get_name(agent_id),
          pubsub: nil
        )

      # Server is idle, cannot cancel
      assert {:error, msg} = AgentServer.cancel(agent_id)
      assert msg == "Cannot cancel, server is not running (status: idle)"
    end

    test "returns error when already completed" do
      agent = create_test_agent()
      agent_id = agent.agent_id
      initial_state = State.new!(%{messages: [Message.new_user!("Hello")]})

      Agent
      |> expect(:execute, fn ^agent, state, _opts ->
        {:ok, state}
      end)

      {:ok, _pid} =
        AgentServer.start_link(
          agent: agent,
          initial_state: initial_state,
          name: AgentServer.get_name(agent_id),
          pubsub: nil
        )

      # Execute and wait for completion
      assert :ok = AgentServer.execute(agent_id)
      Process.sleep(50)
      assert AgentServer.get_status(agent_id) == :completed

      # Try to cancel - should fail
      assert {:error, msg} = AgentServer.cancel(agent_id)
      assert msg == "Cannot cancel, server is not running (status: completed)"
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
      |> expect(:execute, fn ^agent, state, _opts ->
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
      |> expect(:resume, fn ^agent, state, ^decisions, _opts ->
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
      |> expect(:resume, fn ^agent, _state, ^decisions, _opts ->
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
      |> expect(:execute, fn ^agent, state, _opts ->
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

    test "broadcasts todos updated event when todo created", %{agent: agent, agent_id: agent_id} do
      Agent
      |> expect(:execute, fn ^agent, state, _opts ->
        todo = Todo.new!(%{content: "Write tests", status: :pending})
        new_state = State.put_todo(state, todo)
        {:ok, new_state}
      end)

      :ok = AgentServer.execute(agent_id)

      assert_receive {:status_changed, :running, nil}, 100
      assert_receive {:todos_updated, todos}, 200
      assert length(todos) == 1
      assert hd(todos).content == "Write tests"
    end

    test "broadcasts todos updated event when todo status changes", %{
      agent: agent,
      agent_id: agent_id
    } do
      # Set initial todo
      todo = Todo.new!(%{id: "test_id", content: "Write tests", status: :pending})
      initial_state = State.new!() |> State.put_todo(todo)

      :sys.replace_state(GenServer.whereis(AgentServer.get_name(agent_id)), fn server_state ->
        %{server_state | state: initial_state}
      end)

      Agent
      |> expect(:execute, fn ^agent, state, _opts ->
        updated_todo = Todo.new!(%{id: "test_id", content: "Write tests", status: :completed})
        new_state = State.put_todo(state, updated_todo)
        {:ok, new_state}
      end)

      :ok = AgentServer.execute(agent_id)

      assert_receive {:status_changed, :running, nil}, 100
      assert_receive {:todos_updated, todos}, 200
      assert length(todos) == 1
      assert hd(todos).status == :completed
      assert hd(todos).id == "test_id"
    end

    test "broadcasts todos updated event when todo deleted", %{agent: agent, agent_id: agent_id} do
      # Set initial todo
      todo = Todo.new!(%{id: "test_id", content: "Write tests", status: :pending})
      initial_state = State.new!() |> State.put_todo(todo)

      :sys.replace_state(GenServer.whereis(AgentServer.get_name(agent_id)), fn server_state ->
        %{server_state | state: initial_state}
      end)

      Agent
      |> expect(:execute, fn ^agent, state, _opts ->
        new_state = State.delete_todo(state, "test_id")
        {:ok, new_state}
      end)

      :ok = AgentServer.execute(agent_id)

      assert_receive {:status_changed, :running, nil}, 100
      assert_receive {:todos_updated, todos}, 200
      assert length(todos) == 0
    end

    test "broadcasts interrupt status with data", %{agent: agent, agent_id: agent_id} do
      interrupt_data = %{
        action_requests: [%{tool_name: "write_file"}],
        review_configs: %{}
      }

      Agent
      |> expect(:execute, fn ^agent, state, _opts ->
        {:interrupt, state, interrupt_data}
      end)

      :ok = AgentServer.execute(agent_id)

      assert_receive {:status_changed, :running, nil}, 100
      assert_receive {:status_changed, :interrupted, ^interrupt_data}, 200
    end

    test "broadcasts error status", %{agent: agent, agent_id: agent_id} do
      Agent
      |> expect(:execute, fn ^agent, _state, _opts ->
        {:error, "Test error"}
      end)

      :ok = AgentServer.execute(agent_id)

      assert_receive {:status_changed, :running, nil}, 100
      assert_receive {:status_changed, :error, "Test error"}, 200
    end

    test "broadcasts completed status when task is cancelled", %{agent: agent, agent_id: agent_id} do
      Agent
      |> expect(:execute, fn ^agent, state, _opts ->
        Process.sleep(1_000)
        {:ok, state}
      end)

      :ok = AgentServer.execute(agent_id)

      # Should receive running status
      assert_receive {:status_changed, :running, nil}, 100

      # let the task spin up
      Process.sleep(20)
      # Cancel the task
      :ok = AgentServer.cancel(agent_id)

      # Should receive completed status
      assert_receive {:status_changed, :completed, _state}, 100
    end
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

  describe "inactivity timeout" do
    test "initializes with default 5 minute timeout" do
      agent = create_test_agent()
      agent_id = agent.agent_id

      {:ok, _pid} =
        AgentServer.start_link(
          agent: agent,
          name: AgentServer.get_name(agent_id),
          pubsub: nil
        )

      status = AgentServer.get_inactivity_status(agent_id)
      assert status.inactivity_timeout == 300_000
      assert status.timer_active == true
      assert status.last_activity_at != nil
    end

    test "accepts custom timeout value" do
      agent = create_test_agent()
      agent_id = agent.agent_id

      {:ok, _pid} =
        AgentServer.start_link(
          agent: agent,
          name: AgentServer.get_name(agent_id),
          pubsub: nil,
          inactivity_timeout: 600_000
        )

      status = AgentServer.get_inactivity_status(agent_id)
      assert status.inactivity_timeout == 600_000
      assert status.timer_active == true
    end

    test "can disable timeout with nil" do
      agent = create_test_agent()
      agent_id = agent.agent_id

      {:ok, _pid} =
        AgentServer.start_link(
          agent: agent,
          name: AgentServer.get_name(agent_id),
          pubsub: nil,
          inactivity_timeout: nil
        )

      status = AgentServer.get_inactivity_status(agent_id)
      assert status.inactivity_timeout == nil
      assert status.timer_active == false
    end

    test "can disable timeout with :infinity" do
      agent = create_test_agent()
      agent_id = agent.agent_id

      {:ok, _pid} =
        AgentServer.start_link(
          agent: agent,
          name: AgentServer.get_name(agent_id),
          pubsub: nil,
          inactivity_timeout: :infinity
        )

      status = AgentServer.get_inactivity_status(agent_id)
      assert status.inactivity_timeout == :infinity
      assert status.timer_active == false
    end

    test "resets timer on execute" do
      agent = create_test_agent()
      agent_id = agent.agent_id

      Agent
      |> expect(:execute, fn ^agent, state, _opts ->
        # Simulate slow execution
        Process.sleep(50)
        {:ok, state}
      end)

      {:ok, _pid} =
        AgentServer.start_link(
          agent: agent,
          name: AgentServer.get_name(agent_id),
          pubsub: nil,
          inactivity_timeout: 1_000
        )

      # Get initial status
      status1 = AgentServer.get_inactivity_status(agent_id)
      initial_time = status1.last_activity_at

      # Wait a bit
      Process.sleep(50)

      # Execute - should reset timer
      :ok = AgentServer.execute(agent_id)
      Process.sleep(100)

      # Check that activity time was updated
      status2 = AgentServer.get_inactivity_status(agent_id)
      assert DateTime.compare(status2.last_activity_at, initial_time) == :gt
    end

    test "resets timer on add_message" do
      agent = create_test_agent()
      agent_id = agent.agent_id

      {:ok, _pid} =
        AgentServer.start_link(
          agent: agent,
          name: AgentServer.get_name(agent_id),
          pubsub: nil,
          inactivity_timeout: 1_000
        )

      # Get initial status
      status1 = AgentServer.get_inactivity_status(agent_id)
      initial_time = status1.last_activity_at

      # Wait a bit
      Process.sleep(50)

      # Add message - should reset timer
      :ok = AgentServer.add_message(agent_id, Message.new_user!("Hello"))

      # Check that activity time was updated
      status2 = AgentServer.get_inactivity_status(agent_id)
      assert DateTime.compare(status2.last_activity_at, initial_time) == :gt
    end

    # Note: The reset timer test is in agent_supervisor_test.exs where FileSystemServer is available

    test "time_since_activity increases over time" do
      agent = create_test_agent()
      agent_id = agent.agent_id

      {:ok, _pid} =
        AgentServer.start_link(
          agent: agent,
          name: AgentServer.get_name(agent_id),
          pubsub: nil,
          inactivity_timeout: 10_000
        )

      # Get initial time
      status1 = AgentServer.get_inactivity_status(agent_id)
      time1 = status1.time_since_activity

      # Wait a bit
      Process.sleep(100)

      # Check that time has increased
      status2 = AgentServer.get_inactivity_status(agent_id)
      time2 = status2.time_since_activity

      assert time2 > time1
      assert time2 >= 100
    end

    test "broadcasts shutdown event when timeout triggers" do
      # Start a test PubSub
      pubsub_name = :"test_pubsub_#{:erlang.unique_integer([:positive])}"
      {:ok, _} = start_supervised({Phoenix.PubSub, name: pubsub_name})

      agent = create_test_agent()
      agent_id = agent.agent_id

      # Short timeout for testing
      {:ok, pid} =
        AgentServer.start_link(
          agent: agent,
          name: AgentServer.get_name(agent_id),
          pubsub: Phoenix.PubSub,
          pubsub_name: pubsub_name,
          id: agent_id,
          inactivity_timeout: 100
        )

      # Subscribe to events
      :ok = AgentServer.subscribe(agent_id)

      # Monitor the process
      _ref = Process.monitor(pid)

      # Wait for timeout and shutdown event
      assert_receive {:agent_shutdown, shutdown_data}, 500
      assert shutdown_data.agent_id == agent_id
      assert shutdown_data.reason == :inactivity
      assert shutdown_data.last_activity_at != nil
      assert shutdown_data.shutdown_at != nil

      # Note: The actual shutdown happens at AgentSupervisor level,
      # so the process might not die in this standalone test
      # But the event should still be broadcast
    end
  end
end
