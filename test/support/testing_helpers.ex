defmodule LangChain.TestingHelpers do
  @moduledoc """
  Shared testing helper functions used across test suites.
  """

  alias LangChain.Agents.{Agent, AgentServer, AgentSupervisor, FileSystemServer}
  alias LangChain.ChatModels.ChatAnthropic

  @doc """
  Collects all messages sent to the current test process and returns them as a
  list.

  This is useful for testing callbacks that send messages to the test process.
  It's a bit of a hack, but it's the best way I can think of to test callbacks
  that send an unspecified number of messages to the test process.
  """
  def collect_messages do
    collect_messages([])
  end

  defp collect_messages(acc) do
    receive do
      message -> collect_messages([message | acc])
    after
      0 -> Enum.reverse(acc)
    end
  end

  @doc """
  Helper to get a file entry from FileSystemServer's GenServer state.

  This is useful for inspecting the internal state of the filesystem in tests.

  ## Parameters

  - `agent_id` - The agent identifier
  - `path` - The file path to retrieve

  ## Returns

  The FileEntry struct or nil if not found.

  ## Examples

      entry = get_entry("agent-123", "/file.txt")
      assert entry.content == "test content"
  """
  def get_entry(agent_id, path) do
    pid = FileSystemServer.whereis({:agent, agent_id})
    state = :sys.get_state(pid)
    Map.get(state.files, path)
  end

  @doc """
  Generate a new, unique agent_id.
  """
  def new_agent_id() do
    "test-agent-#{System.unique_integer()}"
  end

  @doc """
  Generate a new, unique test agent_id.

  This is an alias for `new_agent_id/0` with a more descriptive name for use in tests.
  """
  def generate_test_agent_id() do
    new_agent_id()
  end

  @doc """
  Basic conversion of a Message to a DisplayMessage like data map.
  """
  def message_to_display_data(%LangChain.Message{} = message) do
    %{
      content_type: "text",
      role: to_string(message.role),
      content: LangChain.Message.ContentPart.parts_to_string(message.content)
    }
  end

  @doc """
  A basic function compatible for assigning to a test agent's `:save_new_message_fn` callback function.
  """
  def basic_process_to_display_data(_conversation_id, %LangChain.Message{} = message) do
    {:ok, [message_to_display_data(message)]}
  end

  @doc """
  Start a test agent with proper configuration for testing.

  This helper follows the same pattern as Coordinator.do_start_session to ensure
  agents are properly initialized and ready before tests interact with them.

  ## Options

  - `:agent_id` - (required) Unique identifier for the agent
  - `:pubsub` - (required) Tuple of {module, name} for PubSub
  - `:conversation_id` - (optional) Conversation ID for message persistence
  - `:save_new_message_fn` - (optional) Callback function for message persistence
  - `:initial_state` - (optional) Initial agent state (defaults to empty state)

  ## Returns

  `{:ok, %{agent_id: agent_id, pid: pid}}` on success, or `{:error, reason}` on failure.

  ## Example

      {:ok, %{agent_id: agent_id, pid: pid}} = start_test_agent(
        agent_id: "test-123",
        pubsub: {Phoenix.PubSub, :test_pubsub},
        conversation_id: "conv-123",
        save_new_message_fn: &MyModule.save_message/2
      )
  """
  def start_test_agent(opts) do
    agent_id = Keyword.fetch!(opts, :agent_id)
    {pubsub_module, pubsub_name} = Keyword.fetch!(opts, :pubsub)

    # Optional callback configuration
    conversation_id = Keyword.get(opts, :conversation_id)
    save_new_message_fn = Keyword.get(opts, :save_new_message_fn)
    initial_state = Keyword.get(opts, :initial_state)

    # Subscribe to agent's PubSub topic so test can receive events
    # AgentServer broadcasts to "agent_server:#{agent_id}"
    topic = "agent_server:#{agent_id}"
    LangChain.PubSub.raw_subscribe(pubsub_module, pubsub_name, topic)

    # Create a minimal test agent
    model =
      ChatAnthropic.new!(%{
        model: "claude-3-5-sonnet-20241022",
        api_key: "test_key"
      })

    agent =
      Agent.new!(%{
        agent_id: agent_id,
        model: model,
        base_system_prompt: "Test agent",
        replace_default_middleware: true,
        middleware: []
      })

    # Build supervisor configuration (similar to Coordinator pattern)
    supervisor_name = AgentSupervisor.get_name(agent_id)

    supervisor_config = [
      name: supervisor_name,
      agent: agent,
      pubsub: {pubsub_module, pubsub_name}
    ]

    # Add initial_state if provided
    supervisor_config =
      if initial_state,
        do: Keyword.put(supervisor_config, :initial_state, initial_state),
        else: supervisor_config

    # Add callback configuration if provided
    supervisor_config =
      if conversation_id,
        do: Keyword.put(supervisor_config, :conversation_id, conversation_id),
        else: supervisor_config

    supervisor_config =
      if save_new_message_fn,
        do: Keyword.put(supervisor_config, :save_new_message_fn, save_new_message_fn),
        else: supervisor_config

    # Start supervisor synchronously to ensure agent is ready
    case AgentSupervisor.start_link_sync(supervisor_config) do
      {:ok, _supervisor_pid} ->
        pid = AgentServer.get_pid(agent_id)
        {:ok, %{agent_id: agent_id, pid: pid}}

      {:error, {:already_started, _supervisor_pid}} ->
        # Already started - return existing pid
        pid = AgentServer.get_pid(agent_id)
        {:ok, %{agent_id: agent_id, pid: pid}}

      {:error, reason} ->
        {:error, reason}
    end
  end

  @doc """
  Stop a test agent and clean up resources.

  ## Parameters

  - `agent_id` - The agent identifier

  ## Example

      stop_test_agent("test-123")
  """
  def stop_test_agent(agent_id) do
    AgentServer.stop(agent_id)
  end
end
