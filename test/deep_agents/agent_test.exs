defmodule LangChain.DeepAgents.AgentTest do
  use ExUnit.Case, async: true
  use Mimic

  alias LangChain.DeepAgents.{Agent, Middleware, State}
  alias LangChain.ChatModels.ChatAnthropic
  alias LangChain.Message

  # Test middleware for composition testing

  defmodule TestMiddleware1 do
    @behaviour Middleware

    @impl true
    def init(opts) do
      {:ok, %{name: Keyword.get(opts, :name, "test1")}}
    end

    @impl true
    def system_prompt(config) do
      "Prompt from #{config.name}"
    end

    @impl true
    def tools(_config) do
      [
        LangChain.Function.new!(%{
          name: "tool1",
          description: "Test tool 1",
          function: fn _args, _params -> {:ok, "result1"} end
        })
      ]
    end

    @impl true
    def before_model(state, config) do
      calls = Map.get(state, :before_calls, [])
      {:ok, Map.put(state, :before_calls, calls ++ [config.name])}
    end

    @impl true
    def after_model(state, config) do
      calls = Map.get(state, :after_calls, [])
      {:ok, Map.put(state, :after_calls, calls ++ [config.name])}
    end
  end

  defmodule TestMiddleware2 do
    @behaviour Middleware

    @impl true
    def init(opts) do
      {:ok, %{name: Keyword.get(opts, :name, "test2")}}
    end

    @impl true
    def system_prompt(config) do
      "Another prompt from #{config.name}"
    end

    @impl true
    def tools(_config) do
      [
        LangChain.Function.new!(%{
          name: "tool2",
          description: "Test tool 2",
          function: fn _args, _params -> {:ok, "result2"} end
        })
      ]
    end

    @impl true
    def before_model(state, config) do
      calls = Map.get(state, :before_calls, [])
      {:ok, Map.put(state, :before_calls, calls ++ [config.name])}
    end

    @impl true
    def after_model(state, config) do
      calls = Map.get(state, :after_calls, [])
      {:ok, Map.put(state, :after_calls, calls ++ [config.name])}
    end
  end

  defmodule ErrorMiddleware do
    @behaviour Middleware

    @impl true
    def before_model(_state, _config) do
      {:error, "before_model failed"}
    end
  end

  # Helper to create a mock model
  defp mock_model do
    ChatAnthropic.new!(%{
      model: "claude-3-5-sonnet-20241022",
      api_key: "test_key"
    })
  end

  describe "new/1" do
    test "creates agent with required model" do
      assert {:ok, agent} = Agent.new(model: mock_model())
      assert %Agent{} = agent
      assert agent.model != nil
      # System prompt now includes TodoList middleware prompt
      assert agent.system_prompt =~ "write_todos"
    end

    test "requires model parameter" do
      assert {:error, "Model is required"} = Agent.new()
    end

    test "creates agent with system prompt" do
      {:ok, agent} = Agent.new(model: mock_model(), system_prompt: "You are helpful.")
      # System prompt includes user prompt + TodoList middleware
      assert agent.system_prompt =~ "You are helpful"
      assert agent.system_prompt =~ "write_todos"
    end

    test "creates agent with custom name" do
      {:ok, agent} = Agent.new(model: mock_model(), name: "my-agent")
      assert agent.name == "my-agent"
    end

    test "creates agent with tools" do
      tool =
        LangChain.Function.new!(%{
          name: "custom_tool",
          description: "A custom tool",
          function: fn _args, _params -> {:ok, "result"} end
        })

      {:ok, agent} = Agent.new(model: mock_model(), tools: [tool])

      # Now includes custom tool + write_todos (TodoList) + 4 filesystem tools (ls, read_file, write_file, edit_file)
      assert length(agent.tools) == 6
      tool_names = Enum.map(agent.tools, & &1.name)
      assert "custom_tool" in tool_names
      assert "write_todos" in tool_names
      assert "ls" in tool_names
      assert "read_file" in tool_names
    end
  end

  describe "new!/1" do
    test "creates agent successfully" do
      agent = Agent.new!(model: mock_model())
      assert %Agent{} = agent
    end

    test "raises on error" do
      assert_raise LangChain.LangChainError, fn ->
        Agent.new!()
      end
    end
  end

  describe "middleware composition - default behavior" do
    test "appends user middleware to defaults" do
      # Defaults include TodoList, Filesystem, Summarization, and PatchToolCalls
      {:ok, agent} =
        Agent.new(
          model: mock_model(),
          middleware: [TestMiddleware1]
        )

      # Default middleware (TodoList + Filesystem + Summarization + PatchToolCalls) + TestMiddleware1 = 5
      assert length(agent.middleware) == 5
    end

    test "collects system prompts from middleware" do
      {:ok, agent} =
        Agent.new(
          model: mock_model(),
          system_prompt: "Base prompt",
          middleware: [
            {TestMiddleware1, [name: "first"]},
            {TestMiddleware2, [name: "second"]}
          ]
        )

      assert agent.system_prompt =~ "Base prompt"
      assert agent.system_prompt =~ "Prompt from first"
      assert agent.system_prompt =~ "Another prompt from second"
    end

    test "collects tools from middleware" do
      {:ok, agent} =
        Agent.new(
          model: mock_model(),
          middleware: [TestMiddleware1, TestMiddleware2]
        )

      # write_todos + 4 filesystem tools + tool1 + tool2 = 7
      assert length(agent.tools) == 7
      tool_names = Enum.map(agent.tools, & &1.name)
      assert "write_todos" in tool_names
      assert "ls" in tool_names
      assert "tool1" in tool_names
      assert "tool2" in tool_names
    end

    test "combines user tools with middleware tools" do
      user_tool =
        LangChain.Function.new!(%{
          name: "user_tool",
          description: "User tool",
          function: fn _args, _params -> {:ok, "result"} end
        })

      {:ok, agent} =
        Agent.new(
          model: mock_model(),
          tools: [user_tool],
          middleware: [TestMiddleware1]
        )

      # user_tool + write_todos + 4 filesystem tools + tool1 = 7
      assert length(agent.tools) == 7
      tool_names = Enum.map(agent.tools, & &1.name)
      assert "write_todos" in tool_names
      assert "user_tool" in tool_names
      assert "tool1" in tool_names
      assert "ls" in tool_names
    end
  end

  describe "middleware composition - replace defaults" do
    test "uses only provided middleware when replace_default_middleware is true" do
      {:ok, agent} =
        Agent.new(
          model: mock_model(),
          replace_default_middleware: true,
          middleware: [TestMiddleware1]
        )

      assert length(agent.middleware) == 1
      {module, _config} = hd(agent.middleware)
      assert module == TestMiddleware1
    end

    test "empty middleware when replace_default_middleware is true and no middleware provided" do
      {:ok, agent} =
        Agent.new(
          model: mock_model(),
          replace_default_middleware: true,
          middleware: []
        )

      assert agent.middleware == []
      assert agent.tools == []
      assert agent.system_prompt == ""
    end
  end

  describe "execute/2" do
    setup do
      # Mock ChatAnthropic.call to return a simple response
      stub(ChatAnthropic, :call, fn _model, _messages, _tools ->
        {:ok, [Message.new_assistant!("Mock response")]}
      end)

      :ok
    end

    test "executes with empty middleware" do
      {:ok, agent} =
        Agent.new(
          model: mock_model(),
          replace_default_middleware: true
        )

      initial_state = State.new!(%{messages: [Message.new_user!("Hello")]})

      assert {:ok, result_state} = Agent.execute(agent, initial_state)
      assert %State{} = result_state
      # Mock execution adds an assistant message
      assert length(result_state.messages) == 2
      assert Enum.at(result_state.messages, 1).role == :assistant
    end

    test "applies before_model hooks in order" do
      {:ok, agent} =
        Agent.new(
          model: mock_model(),
          middleware: [
            {TestMiddleware1, [name: "first"]},
            {TestMiddleware2, [name: "second"]}
          ]
        )

      initial_state = State.new!(%{messages: [Message.new_user!("Test")]})

      assert {:ok, result_state} = Agent.execute(agent, initial_state)
      assert result_state.before_calls == ["first", "second"]
    end

    test "applies after_model hooks in reverse order" do
      {:ok, agent} =
        Agent.new(
          model: mock_model(),
          middleware: [
            {TestMiddleware1, [name: "first"]},
            {TestMiddleware2, [name: "second"]}
          ]
        )

      initial_state = State.new!(%{messages: [Message.new_user!("Test")]})

      assert {:ok, result_state} = Agent.execute(agent, initial_state)
      # After hooks applied in reverse
      assert result_state.after_calls == ["second", "first"]
    end

    test "returns error if before_model hook fails" do
      {:ok, agent} =
        Agent.new(
          model: mock_model(),
          middleware: [ErrorMiddleware]
        )

      initial_state = State.new!(%{messages: [Message.new_user!("Test")]})

      assert {:error, "before_model failed"} = Agent.execute(agent, initial_state)
    end

    test "state flows through all hooks" do
      {:ok, agent} =
        Agent.new(
          model: mock_model(),
          middleware: [
            {TestMiddleware1, [name: "mw1"]},
            {TestMiddleware2, [name: "mw2"]}
          ]
        )

      initial_state = State.new!(%{messages: [Message.new_user!("Test")]})

      assert {:ok, result_state} = Agent.execute(agent, initial_state)

      # Verify hooks were called
      assert result_state.before_calls == ["mw1", "mw2"]
      assert result_state.after_calls == ["mw2", "mw1"]

      # Verify original messages preserved
      first_message = Enum.at(result_state.messages, 0)
      assert first_message.role == :user
      assert Message.ContentPart.content_to_string(first_message.content) == "Test"

      # Verify mock response added
      assert Enum.at(result_state.messages, 1).role == :assistant
    end
  end

  describe "integration tests" do
    setup do
      # Mock ChatAnthropic.call to return a simple response
      stub(ChatAnthropic, :call, fn _model, _messages, _tools ->
        {:ok, [Message.new_assistant!("Mock response")]}
      end)

      :ok
    end

    test "complete workflow with system prompt, tools, and middleware" do
      custom_tool =
        LangChain.Function.new!(%{
          name: "calculator",
          description: "Calculate things",
          function: fn _args, _params -> {:ok, "42"} end
        })

      {:ok, agent} =
        Agent.new(
          model: mock_model(),
          name: "math-agent",
          system_prompt: "You are a math assistant.",
          tools: [custom_tool],
          middleware: [
            {TestMiddleware1, [name: "logging"]},
            {TestMiddleware2, [name: "validation"]}
          ]
        )

      # Verify agent structure
      assert agent.name == "math-agent"
      assert agent.system_prompt =~ "math assistant"
      assert agent.system_prompt =~ "logging"
      assert agent.system_prompt =~ "validation"
      # calculator + write_todos + 4 filesystem tools + tool1 + tool2 = 8
      assert length(agent.tools) == 8

      # Execute
      initial_state = State.new!(%{messages: [Message.new_user!("What is 2+2?")]})
      assert {:ok, result_state} = Agent.execute(agent, initial_state)

      # Verify execution
      assert length(result_state.messages) == 2
      assert result_state.before_calls == ["logging", "validation"]
      assert result_state.after_calls == ["validation", "logging"]
    end

    test "agent with no middleware works correctly" do
      {:ok, agent} =
        Agent.new(
          model: mock_model(),
          system_prompt: "Simple agent",
          replace_default_middleware: true
        )

      initial_state = State.new!(%{messages: [Message.new_user!("Hi")]})
      assert {:ok, result_state} = Agent.execute(agent, initial_state)

      assert length(result_state.messages) == 2
      # No middleware hooks were called, so these keys don't exist
      assert Map.get(result_state, :before_calls) == nil
      assert Map.get(result_state, :after_calls) == nil
    end
  end

  describe "TodoList middleware integration" do
    test "agents include TodoList middleware by default" do
      {:ok, agent} = Agent.new(model: mock_model())

      # Should have TodoList middleware in the stack
      assert length(agent.middleware) > 0

      # Should have write_todos tool
      tool_names = Enum.map(agent.tools, & &1.name)
      assert "write_todos" in tool_names
    end

    test "TodoList middleware can be excluded" do
      {:ok, agent} =
        Agent.new(
          model: mock_model(),
          replace_default_middleware: true,
          middleware: []
        )

      # No middleware
      assert agent.middleware == []
      assert agent.tools == []
    end

    test "system prompt includes TODO instructions" do
      {:ok, agent} = Agent.new(model: mock_model())

      assert agent.system_prompt =~ "write_todos"
    end
  end
end
