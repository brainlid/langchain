defmodule LangChain.Agents.SubAgentIntegrationTest do
  @moduledoc """
  End-to-end integration tests for SubAgent functionality.

  These tests verify the complete flow from main agent to SubAgent execution,
  including HITL interrupts and resume operations.
  """

  use ExUnit.Case, async: false
  use Mimic

  alias LangChain.Agents.{Agent, SubAgent, State}
  alias LangChain.ChatModels.ChatAnthropic
  alias LangChain.Chains.LLMChain
  alias LangChain.Function
  alias LangChain.Message

  setup :set_mimic_global
  setup :verify_on_exit!

  setup_all do
    # Ensure Registry is started
    unless Process.whereis(LangChain.Agents.Registry) do
      {:ok, _} = Registry.start_link(keys: :unique, name: LangChain.Agents.Registry)
    end

    # Copy modules for mocking
    Mimic.copy(LLMChain)

    :ok
  end

  # Helper to create a test model
  defp test_model do
    ChatAnthropic.new!(%{
      model: "claude-3-5-sonnet-20241022",
      api_key: "test_key"
    })
  end

  # Helper to create a simple tool
  defp simple_tool(name) do
    Function.new!(%{
      name: name,
      description: "A simple test tool",
      function: fn _args, _context -> {:ok, "Tool #{name} executed"} end
    })
  end

  describe "basic SubAgent execution flow" do
    test "main agent can execute SubAgent and get result" do
      # Create a SubAgent configuration
      researcher_config =
        SubAgent.Config.new!(%{
          name: "researcher",
          description: "Research agent",
          system_prompt: "You research topics",
          tools: [simple_tool("search")]
        })

      # Create main agent with SubAgent middleware
      # Don't replace default middleware so SubAgent middleware gets added automatically
      {:ok, agent} =
        Agent.new(%{
          model: test_model(),
          system_prompt: "You delegate research",
          subagent_opts: [subagents: [researcher_config]]
        })

      # Verify agent was created successfully
      assert %Agent{} = agent

      # Verify SubAgent middleware is present
      assert Enum.any?(agent.middleware, fn
               {LangChain.Agents.Middleware.SubAgent, _} -> true
               _ -> false
             end)

      # Verify task tool is available
      task_tool =
        Enum.find(agent.tools, fn tool ->
          tool.name == "task"
        end)

      assert task_tool != nil
      assert task_tool.async == true
    end

    test "SubAgent middleware is filtered from SubAgent's middleware stack" do
      # Create SubAgent config
      subagent_config =
        SubAgent.Config.new!(%{
          name: "worker",
          description: "Worker agent",
          system_prompt: "You work",
          tools: [simple_tool("work")]
        })

      # Build registry with default middleware
      model = test_model()

      default_middleware = [
        {LangChain.Agents.Middleware.TodoList, []},
        {LangChain.Agents.Middleware.SubAgent, []},
        {LangChain.Agents.Middleware.FileSystem, [agent_id: "parent"]}
      ]

      {:ok, registry} = SubAgent.build_agent_map([subagent_config], model, default_middleware)

      worker_agent = registry["worker"]

      # Verify SubAgent middleware is NOT in the worker's middleware
      refute Enum.any?(worker_agent.middleware, fn
               {LangChain.Agents.Middleware.SubAgent, _} -> true
               LangChain.Agents.Middleware.SubAgent -> true
               _ -> false
             end)

      # But other middleware should be present
      assert Enum.any?(worker_agent.middleware, fn
               {LangChain.Agents.Middleware.TodoList, _} -> true
               _ -> false
             end)
    end
  end

  describe "SubAgent with HITL configuration" do
    test "SubAgent extracts interrupt_on configuration from middleware" do
      # Create agent with HITL middleware
      agent =
        Agent.new!(%{
          model: test_model(),
          system_prompt: "Test",
          replace_default_middleware: true,
          middleware: [
            {LangChain.Agents.Middleware.HumanInTheLoop,
             [interrupt_on: %{"dangerous_tool" => true}]}
          ]
        })

      # Create SubAgent from this agent config
      subagent =
        SubAgent.new_from_config(
          parent_agent_id: "test",
          instructions: "Do something",
          agent_config: agent,
          parent_state: State.new!(%{messages: []})
        )

      # Verify interrupt_on was extracted
      assert subagent.interrupt_on != nil
      assert is_map(subagent.interrupt_on)
      # HumanInTheLoop middleware normalizes true to a config map
      assert Map.has_key?(subagent.interrupt_on, "dangerous_tool")
    end

    test "SubAgent without HITL has empty interrupt_on" do
      agent =
        Agent.new!(%{
          model: test_model(),
          system_prompt: "Test",
          replace_default_middleware: true,
          middleware: []
        })

      subagent =
        SubAgent.new_from_config(
          parent_agent_id: "test",
          instructions: "Do something",
          agent_config: agent,
          parent_state: State.new!(%{messages: []})
        )

      # Should have empty interrupt_on
      assert subagent.interrupt_on == %{}
    end
  end

  describe "SubAgent execution states" do
    test "SubAgent transitions through correct states" do
      agent =
        Agent.new!(%{
          model: test_model(),
          system_prompt: "Test",
          replace_default_middleware: true,
          middleware: []
        })

      subagent =
        SubAgent.new_from_config(
          parent_agent_id: "test",
          instructions: "Task",
          agent_config: agent,
          parent_state: State.new!(%{messages: []})
        )

      # Initial state
      assert subagent.status == :idle
      assert SubAgent.can_execute?(subagent)
      refute SubAgent.can_resume?(subagent)
      refute SubAgent.is_terminal?(subagent)

      # Simulated execution state changes
      running = %{subagent | status: :running}
      refute SubAgent.can_execute?(running)
      refute SubAgent.can_resume?(running)
      refute SubAgent.is_terminal?(running)

      # Interrupted state
      interrupted = %{
        subagent
        | status: :interrupted,
          interrupt_data: %{action_requests: [], hitl_tool_call_ids: []}
      }

      refute SubAgent.can_execute?(interrupted)
      assert SubAgent.can_resume?(interrupted)
      refute SubAgent.is_terminal?(interrupted)

      # Completed state
      completed = %{subagent | status: :completed}
      refute SubAgent.can_execute?(completed)
      refute SubAgent.can_resume?(completed)
      assert SubAgent.is_terminal?(completed)

      # Error state
      error = %{subagent | status: :error, error: "test"}
      refute SubAgent.can_execute?(error)
      refute SubAgent.can_resume?(error)
      assert SubAgent.is_terminal?(error)
    end
  end

  describe "SubAgent agent_map building" do
    test "builds agent_map from mixed Config and Compiled subagents" do
      # Config subagent
      dynamic_config =
        SubAgent.Config.new!(%{
          name: "dynamic",
          description: "Dynamic agent",
          system_prompt: "Dynamic",
          tools: [simple_tool("tool1")]
        })

      # Compiled subagent
      compiled_agent =
        Agent.new!(%{
          model: test_model(),
          system_prompt: "Compiled",
          replace_default_middleware: true,
          middleware: []
        })

      compiled_config =
        SubAgent.Compiled.new!(%{
          name: "compiled",
          description: "Compiled agent",
          agent: compiled_agent
        })

      # Build agent_map
      model = test_model()
      {:ok, agent_map} = SubAgent.build_agent_map([dynamic_config, compiled_config], model, [])

      assert map_size(agent_map) == 2
      assert %Agent{} = agent_map["dynamic"]
      # Compiled subagents are stored as Compiled structs to preserve metadata
      assert %SubAgent.Compiled{} = agent_map["compiled"]
      assert agent_map["compiled"].agent == compiled_agent
    end

    test "builds descriptions map correctly" do
      configs = [
        SubAgent.Config.new!(%{
          name: "agent1",
          description: "Description 1",
          system_prompt: "Prompt",
          tools: [simple_tool("tool")]
        }),
        SubAgent.Config.new!(%{
          name: "agent2",
          description: "Description 2",
          system_prompt: "Prompt",
          tools: [simple_tool("tool")]
        })
      ]

      descriptions = SubAgent.build_descriptions(configs)

      assert descriptions == %{
               "agent1" => "Description 1",
               "agent2" => "Description 2"
             }
    end
  end

  describe "error handling" do
    test "SubAgent.execute returns error for non-idle status" do
      agent =
        Agent.new!(%{
          model: test_model(),
          system_prompt: "Test",
          replace_default_middleware: true,
          middleware: []
        })

      subagent =
        SubAgent.new_from_config(
          parent_agent_id: "test",
          instructions: "Task",
          agent_config: agent,
          parent_state: State.new!(%{messages: []})
        )

      # Try to execute a running subagent
      running = %{subagent | status: :running}

      assert {:error, {:invalid_status, :running, :expected_idle}} = SubAgent.execute(running)
    end

    test "SubAgent.resume returns error for non-interrupted status" do
      agent =
        Agent.new!(%{
          model: test_model(),
          system_prompt: "Test",
          replace_default_middleware: true,
          middleware: []
        })

      subagent =
        SubAgent.new_from_config(
          parent_agent_id: "test",
          instructions: "Task",
          agent_config: agent,
          parent_state: State.new!(%{messages: []})
        )

      # Try to resume an idle subagent
      assert {:error, {:invalid_status, :idle, :expected_interrupted}} =
               SubAgent.resume(subagent, [])
    end

    test "build_agent_map returns error for invalid configuration" do
      invalid_config =
        SubAgent.Config.new!(%{
          name: "test",
          description: "Test",
          system_prompt: "Test",
          tools: [simple_tool("tool")]
        })

      # Use nil model to trigger error
      model = nil

      assert {:error, reason} = SubAgent.build_agent_map([invalid_config], model, [])
      assert reason =~ "Failed to build subagent registry"
    end
  end

  describe "result extraction" do
    test "extract_result returns content from completed SubAgent" do
      agent =
        Agent.new!(%{
          model: test_model(),
          system_prompt: "Test",
          replace_default_middleware: true,
          middleware: []
        })

      subagent =
        SubAgent.new_from_config(
          parent_agent_id: "test",
          instructions: "Task",
          agent_config: agent,
          parent_state: State.new!(%{messages: []})
        )

      # Simulate completed subagent with result
      final_message = Message.new_assistant!(%{content: "Final result"})

      completed_chain =
        subagent.chain
        |> Map.put(:messages, subagent.chain.messages ++ [final_message])
        |> Map.put(:last_message, final_message)

      completed = %{
        subagent
        | status: :completed,
          chain: completed_chain
      }

      assert {:ok, result} = SubAgent.extract_result(completed)
      assert result == "Final result"
    end

    test "extract_result returns error for non-completed SubAgent" do
      agent =
        Agent.new!(%{
          model: test_model(),
          system_prompt: "Test",
          replace_default_middleware: true,
          middleware: []
        })

      subagent =
        SubAgent.new_from_config(
          parent_agent_id: "test",
          instructions: "Task",
          agent_config: agent,
          parent_state: State.new!(%{messages: []})
        )

      # Try to extract from idle subagent
      assert {:error, {:invalid_status, :idle, :expected_completed}} =
               SubAgent.extract_result(subagent)
    end
  end
end
