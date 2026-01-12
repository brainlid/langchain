defmodule LangChain.Agents.SubAgentTest do
  use ExUnit.Case, async: true

  alias LangChain.Agents.SubAgent
  alias LangChain.Agents.Agent
  alias LangChain.Agents.MiddlewareEntry
  alias LangChain.Function
  alias LangChain.ChatModels.ChatAnthropic

  # Import SubAgent structs directly to avoid module name conflicts
  alias LangChain.Agents.SubAgent.Config, as: SubAgentConfig
  alias LangChain.Agents.SubAgent.Compiled, as: SubAgentCompiled

  # Helper to create a test tool
  defp test_tool(name \\ "test_tool") do
    Function.new!(%{
      name: name,
      description: "A test tool",
      function: fn _args, _context -> {:ok, "result"} end
    })
  end

  # Helper to create a test model
  defp test_model do
    ChatAnthropic.new!(%{
      model: "claude-3-5-sonnet-20241022",
      api_key: "test_key"
    })
  end

  # Helper to create a test agent
  defp test_agent do
    Agent.new!(%{
      model: test_model(),
      system_prompt: "Test agent",
      replace_default_middleware: true,
      middleware: []
    })
  end

  describe "SubAgentConfig.new/1" do
    test "creates config with required fields" do
      attrs = %{
        name: "test-agent",
        description: "A test agent",
        system_prompt: "You are a test agent",
        tools: [test_tool()]
      }

      assert {:ok, config} = SubAgentConfig.new(attrs)
      assert config.name == "test-agent"
      assert config.description == "A test agent"
      assert config.system_prompt == "You are a test agent"
      assert length(config.tools) == 1
      assert config.model == nil
      assert config.middleware == []
      assert config.interrupt_on == nil
    end

    test "creates config with optional fields" do
      model = test_model()
      middleware = [{SomeMiddleware, []}]

      attrs = %{
        name: "custom-agent",
        description: "Custom agent",
        system_prompt: "Custom prompt",
        tools: [test_tool()],
        model: model,
        middleware: middleware,
        interrupt_on: %{"write_file" => true}
      }

      assert {:ok, config} = SubAgentConfig.new(attrs)
      assert config.model == model
      assert config.middleware == middleware
      assert config.interrupt_on == %{"write_file" => true}
    end

    test "requires name field" do
      attrs = %{
        description: "Test",
        system_prompt: "Test",
        tools: [test_tool()]
      }

      assert {:error, changeset} = SubAgentConfig.new(attrs)
      assert %{name: ["can't be blank"]} = errors_on(changeset)
    end

    test "requires description field" do
      attrs = %{
        name: "test",
        system_prompt: "Test",
        tools: [test_tool()]
      }

      assert {:error, changeset} = SubAgentConfig.new(attrs)
      assert %{description: ["can't be blank"]} = errors_on(changeset)
    end

    test "requires system_prompt field" do
      attrs = %{
        name: "test",
        description: "Test",
        tools: [test_tool()]
      }

      assert {:error, changeset} = SubAgentConfig.new(attrs)
      assert %{system_prompt: ["can't be blank"]} = errors_on(changeset)
    end

    test "requires tools field" do
      attrs = %{
        name: "test",
        description: "Test",
        system_prompt: "Test"
      }

      assert {:error, changeset} = SubAgentConfig.new(attrs)
      errors = errors_on(changeset)
      assert errors[:tools]

      assert "can't be blank" in errors[:tools] or
               "must contain at least one tool" in errors[:tools]
    end

    test "validates tools must be non-empty list" do
      attrs = %{
        name: "test",
        description: "Test",
        system_prompt: "Test",
        tools: []
      }

      assert {:error, changeset} = SubAgentConfig.new(attrs)
      assert %{tools: ["must contain at least one tool"]} = errors_on(changeset)
    end

    test "validates tools must be Function structs" do
      attrs = %{
        name: "test",
        description: "Test",
        system_prompt: "Test",
        tools: ["not a function"]
      }

      assert {:error, changeset} = SubAgentConfig.new(attrs)
      assert %{tools: ["must be a list of LangChain.Function structs"]} = errors_on(changeset)
    end

    test "validates name length" do
      attrs = %{
        name: String.duplicate("x", 101),
        description: "Test",
        system_prompt: "Test",
        tools: [test_tool()]
      }

      assert {:error, changeset} = SubAgentConfig.new(attrs)
      assert %{name: [_]} = errors_on(changeset)
    end

    test "validates description length" do
      attrs = %{
        name: "test",
        description: String.duplicate("x", 501),
        system_prompt: "Test",
        tools: [test_tool()]
      }

      assert {:error, changeset} = SubAgentConfig.new(attrs)
      assert %{description: [_]} = errors_on(changeset)
    end

    test "validates system_prompt length" do
      attrs = %{
        name: "test",
        description: "Test",
        system_prompt: String.duplicate("x", 10_001),
        tools: [test_tool()]
      }

      assert {:error, changeset} = SubAgentConfig.new(attrs)
      assert %{system_prompt: [_]} = errors_on(changeset)
    end

    test "accepts multiple tools" do
      attrs = %{
        name: "test",
        description: "Test",
        system_prompt: "Test",
        tools: [test_tool("tool1"), test_tool("tool2"), test_tool("tool3")]
      }

      assert {:ok, config} = SubAgentConfig.new(attrs)
      assert length(config.tools) == 3
    end
  end

  describe "SubAgentConfig.new!/1" do
    test "returns config on success" do
      attrs = %{
        name: "test",
        description: "Test",
        system_prompt: "Test",
        tools: [test_tool()]
      }

      assert %SubAgentConfig{} = SubAgentConfig.new!(attrs)
    end

    test "raises on error" do
      attrs = %{
        name: "test",
        description: "Test",
        system_prompt: "Test"
      }

      assert_raise LangChain.LangChainError, fn ->
        SubAgentConfig.new!(attrs)
      end
    end
  end

  describe "SubAgentCompiled.new/1" do
    test "creates compiled config with required fields" do
      agent = test_agent()

      attrs = %{
        name: "custom-agent",
        description: "Custom pre-built agent",
        agent: agent
      }

      assert {:ok, compiled} = SubAgentCompiled.new(attrs)
      assert compiled.name == "custom-agent"
      assert compiled.description == "Custom pre-built agent"
      assert compiled.agent == agent
    end

    test "requires name field" do
      attrs = %{
        description: "Test",
        agent: test_agent()
      }

      assert {:error, changeset} = SubAgentCompiled.new(attrs)
      assert %{name: ["can't be blank"]} = errors_on(changeset)
    end

    test "requires description field" do
      attrs = %{
        name: "test",
        agent: test_agent()
      }

      assert {:error, changeset} = SubAgentCompiled.new(attrs)
      assert %{description: ["can't be blank"]} = errors_on(changeset)
    end

    test "requires agent field" do
      attrs = %{
        name: "test",
        description: "Test"
      }

      assert {:error, changeset} = SubAgentCompiled.new(attrs)
      errors = errors_on(changeset)
      assert errors[:agent]
      assert "can't be blank" in errors[:agent]
    end

    test "validates agent must be Agent struct" do
      attrs = %{
        name: "test",
        description: "Test",
        agent: "not an agent"
      }

      assert {:error, changeset} = SubAgentCompiled.new(attrs)
      assert %{agent: ["must be a LangChain.Agents.Agent struct"]} = errors_on(changeset)
    end

    test "validates name length" do
      attrs = %{
        name: String.duplicate("x", 101),
        description: "Test",
        agent: test_agent()
      }

      assert {:error, changeset} = SubAgentCompiled.new(attrs)
      assert %{name: [_]} = errors_on(changeset)
    end

    test "validates description length" do
      attrs = %{
        name: "test",
        description: String.duplicate("x", 501),
        agent: test_agent()
      }

      assert {:error, changeset} = SubAgentCompiled.new(attrs)
      assert %{description: [_]} = errors_on(changeset)
    end
  end

  describe "SubAgentCompiled.new!/1" do
    test "returns compiled on success" do
      attrs = %{
        name: "test",
        description: "Test",
        agent: test_agent()
      }

      assert %SubAgentCompiled{} = SubAgentCompiled.new!(attrs)
    end

    test "raises on error" do
      attrs = %{
        name: "test",
        description: "Test"
      }

      assert_raise LangChain.LangChainError, fn ->
        SubAgentCompiled.new!(attrs)
      end
    end
  end

  describe "SubAgentCompiled initial_messages" do
    test "creates compiled with initial_messages" do
      agent = test_agent()

      messages = [
        LangChain.Message.new_user!("Test question"),
        LangChain.Message.new_assistant!("Test response")
      ]

      attrs = %{
        name: "test",
        description: "Test",
        agent: agent,
        initial_messages: messages
      }

      assert {:ok, compiled} = SubAgentCompiled.new(attrs)
      assert length(compiled.initial_messages) == 2
      assert Enum.at(compiled.initial_messages, 0).role == :user
      assert Enum.at(compiled.initial_messages, 1).role == :assistant
    end

    test "defaults initial_messages to empty list when not provided" do
      attrs = %{
        name: "test",
        description: "Test",
        agent: test_agent()
      }

      assert {:ok, compiled} = SubAgentCompiled.new(attrs)
      assert compiled.initial_messages == []
    end

    test "treats nil initial_messages as empty list" do
      attrs = %{
        name: "test",
        description: "Test",
        agent: test_agent(),
        initial_messages: nil
      }

      assert {:ok, compiled} = SubAgentCompiled.new(attrs)
      assert compiled.initial_messages == []
    end

    test "validates initial_messages must be list of Message structs" do
      attrs = %{
        name: "test",
        description: "Test",
        agent: test_agent(),
        initial_messages: ["not a message"]
      }

      assert {:error, changeset} = SubAgentCompiled.new(attrs)
      assert %{initial_messages: ["must be a list of Message structs"]} = errors_on(changeset)
    end

    test "validates initial_messages must be a list" do
      attrs = %{
        name: "test",
        description: "Test",
        agent: test_agent(),
        initial_messages: "not a list"
      }

      assert {:error, changeset} = SubAgentCompiled.new(attrs)
      errors = errors_on(changeset)

      # Should have an error on initial_messages field (Ecto returns "is invalid" for type mismatch)
      assert errors[:initial_messages]
      assert is_list(errors[:initial_messages])
      assert length(errors[:initial_messages]) > 0
    end

    test "accepts empty list for initial_messages" do
      attrs = %{
        name: "test",
        description: "Test",
        agent: test_agent(),
        initial_messages: []
      }

      assert {:ok, compiled} = SubAgentCompiled.new(attrs)
      assert compiled.initial_messages == []
    end

    test "validates mixed list with non-Message items" do
      message = LangChain.Message.new_user!("Valid message")

      attrs = %{
        name: "test",
        description: "Test",
        agent: test_agent(),
        initial_messages: [message, "not a message"]
      }

      assert {:error, changeset} = SubAgentCompiled.new(attrs)
      assert %{initial_messages: ["must be a list of Message structs"]} = errors_on(changeset)
    end
  end

  describe "build_agent_map/3" do
    test "builds registry from Config subagents" do
      model = test_model()
      middleware = []

      configs = [
        SubAgentConfig.new!(%{
          name: "agent1",
          description: "First agent",
          system_prompt: "First",
          tools: [test_tool()]
        }),
        SubAgentConfig.new!(%{
          name: "agent2",
          description: "Second agent",
          system_prompt: "Second",
          tools: [test_tool()]
        })
      ]

      assert {:ok, registry} = SubAgent.build_agent_map(configs, model, middleware)
      assert map_size(registry) == 2
      assert %Agent{} = registry["agent1"]
      assert %Agent{} = registry["agent2"]
    end

    test "builds registry from Compiled subagents" do
      model = test_model()
      agent1 = test_agent()
      agent2 = test_agent()

      compiled1 =
        SubAgentCompiled.new!(%{
          name: "custom1",
          description: "Custom 1",
          agent: agent1
        })

      compiled2 =
        SubAgentCompiled.new!(%{
          name: "custom2",
          description: "Custom 2",
          agent: agent2
        })

      configs = [compiled1, compiled2]

      assert {:ok, registry} = SubAgent.build_agent_map(configs, model, [])
      assert map_size(registry) == 2
      # Registry now stores entire Compiled struct to preserve metadata
      assert %SubAgentCompiled{} = registry["custom1"]
      assert registry["custom1"].agent == agent1
      assert %SubAgentCompiled{} = registry["custom2"]
      assert registry["custom2"].agent == agent2
    end

    test "builds registry from mixed Config and Compiled" do
      model = test_model()
      pre_built = test_agent()

      compiled_config =
        SubAgentCompiled.new!(%{
          name: "compiled",
          description: "Compiled agent",
          agent: pre_built
        })

      configs = [
        SubAgentConfig.new!(%{
          name: "dynamic",
          description: "Dynamic agent",
          system_prompt: "Dynamic",
          tools: [test_tool()]
        }),
        compiled_config
      ]

      assert {:ok, registry} = SubAgent.build_agent_map(configs, model, [])
      assert map_size(registry) == 2
      assert %Agent{} = registry["dynamic"]
      # Registry now stores entire Compiled struct
      assert %SubAgentCompiled{} = registry["compiled"]
      assert registry["compiled"].agent == pre_built
    end

    test "Config subagent uses default model when not specified" do
      model = test_model()

      configs = [
        SubAgentConfig.new!(%{
          name: "agent",
          description: "Agent",
          system_prompt: "Prompt",
          tools: [test_tool()]
        })
      ]

      assert {:ok, registry} = SubAgent.build_agent_map(configs, model, [])
      agent = registry["agent"]
      assert agent.model == model
    end

    test "Config subagent uses its own model when specified" do
      default_model = test_model()
      custom_model = ChatAnthropic.new!(%{model: "claude-3-opus-20240229", api_key: "test"})

      configs = [
        SubAgentConfig.new!(%{
          name: "agent",
          description: "Agent",
          system_prompt: "Prompt",
          tools: [test_tool()],
          model: custom_model
        })
      ]

      assert {:ok, registry} = SubAgent.build_agent_map(configs, default_model, [])
      agent = registry["agent"]
      assert agent.model == custom_model
    end

    test "Config subagent gets middleware from config" do
      model = test_model()

      configs = [
        SubAgentConfig.new!(%{
          name: "agent",
          description: "Agent",
          system_prompt: "Prompt",
          tools: [test_tool()],
          middleware: [{CustomMiddleware, []}]
        })
      ]

      assert {:ok, registry} = SubAgent.build_agent_map(configs, model, [])
      agent = registry["agent"]
      # Middleware should be appended
      assert Enum.any?(agent.middleware, fn
               %MiddlewareEntry{module: CustomMiddleware} -> true
               _ -> false
             end)
    end

    test "returns error on invalid configuration" do
      # This would happen if instantiation fails
      # Invalid model
      model = nil

      configs = [
        SubAgentConfig.new!(%{
          name: "agent",
          description: "Agent",
          system_prompt: "Prompt",
          tools: [test_tool()]
        })
      ]

      assert {:error, reason} = SubAgent.build_agent_map(configs, model, [])
      assert reason =~ "Failed to build subagent registry"
    end

    test "handles empty configs list" do
      model = test_model()

      assert {:ok, registry} = SubAgent.build_agent_map([], model, [])
      assert registry == %{}
    end
  end

  describe "build_agent_map!/3" do
    test "returns registry on success" do
      model = test_model()

      configs = [
        SubAgentConfig.new!(%{
          name: "agent",
          description: "Agent",
          system_prompt: "Prompt",
          tools: [test_tool()]
        })
      ]

      registry = SubAgent.build_agent_map!(configs, model, [])
      assert is_map(registry)
      assert map_size(registry) == 1
    end

    test "raises on error" do
      model = nil

      configs = [
        SubAgentConfig.new!(%{
          name: "agent",
          description: "Agent",
          system_prompt: "Prompt",
          tools: [test_tool()]
        })
      ]

      assert_raise LangChain.LangChainError, fn ->
        SubAgent.build_agent_map!(configs, model, [])
      end
    end
  end

  describe "build_descriptions/1" do
    test "builds descriptions map from configs" do
      configs = [
        SubAgentConfig.new!(%{
          name: "agent1",
          description: "First agent description",
          system_prompt: "Prompt",
          tools: [test_tool()]
        }),
        SubAgentConfig.new!(%{
          name: "agent2",
          description: "Second agent description",
          system_prompt: "Prompt",
          tools: [test_tool()]
        })
      ]

      descriptions = SubAgent.build_descriptions(configs)

      assert descriptions == %{
               "agent1" => "First agent description",
               "agent2" => "Second agent description"
             }
    end

    test "works with Compiled configs" do
      configs = [
        SubAgentCompiled.new!(%{
          name: "custom",
          description: "Custom agent description",
          agent: test_agent()
        })
      ]

      descriptions = SubAgent.build_descriptions(configs)
      assert descriptions == %{"custom" => "Custom agent description"}
    end

    test "works with mixed configs" do
      configs = [
        SubAgentConfig.new!(%{
          name: "dynamic",
          description: "Dynamic description",
          system_prompt: "Prompt",
          tools: [test_tool()]
        }),
        SubAgentCompiled.new!(%{
          name: "compiled",
          description: "Compiled description",
          agent: test_agent()
        })
      ]

      descriptions = SubAgent.build_descriptions(configs)

      assert descriptions == %{
               "dynamic" => "Dynamic description",
               "compiled" => "Compiled description"
             }
    end

    test "handles empty list" do
      assert SubAgent.build_descriptions([]) == %{}
    end
  end

  describe "subagent_middleware_stack/2" do
    test "filters out SubAgent middleware from defaults" do
      default_middleware = [
        {LangChain.Agents.Middleware.TodoList, []},
        {LangChain.Agents.Middleware.SubAgent, []},
        {LangChain.Agents.Middleware.FileSystem, []}
      ]

      result = SubAgent.subagent_middleware_stack(default_middleware)

      assert length(result) == 2

      refute Enum.any?(result, fn
               {LangChain.Agents.Middleware.SubAgent, _} -> true
               _ -> false
             end)
    end

    test "filters out SubAgent middleware without config tuple" do
      default_middleware = [
        {LangChain.Agents.Middleware.TodoList, []},
        LangChain.Agents.Middleware.SubAgent,
        {LangChain.Agents.Middleware.FileSystem, []}
      ]

      result = SubAgent.subagent_middleware_stack(default_middleware)

      assert length(result) == 2
      refute Enum.any?(result, &match?(LangChain.Agents.Middleware.SubAgent, &1))
    end

    test "appends additional middleware" do
      default_middleware = [
        {LangChain.Agents.Middleware.TodoList, []}
      ]

      additional = [{CustomMiddleware, []}]

      result = SubAgent.subagent_middleware_stack(default_middleware, additional)

      assert length(result) == 2
      assert Enum.at(result, 1) == {CustomMiddleware, []}
    end

    test "preserves other middleware" do
      default_middleware = [
        {LangChain.Agents.Middleware.TodoList, []},
        {LangChain.Agents.Middleware.FileSystem, []},
        {LangChain.Agents.Middleware.PatchToolCalls, []}
      ]

      result = SubAgent.subagent_middleware_stack(default_middleware)

      assert length(result) == 3

      assert Enum.all?(result, fn
               {mod, _} -> mod != LangChain.Agents.Middleware.SubAgent
             end)
    end

    test "handles empty default middleware" do
      result = SubAgent.subagent_middleware_stack([])
      assert result == []
    end

    test "handles empty additional middleware" do
      default_middleware = [{SomeMiddleware, []}]
      result = SubAgent.subagent_middleware_stack(default_middleware, [])
      assert result == [{SomeMiddleware, []}]
    end

    test "filters out SubAgent middleware from MiddlewareEntry structs" do
      # Create initialized middleware list (as it would be in agent.middleware)
      {:ok, agent} =
        Agent.new(%{
          model: test_model(),
          middleware: [
            LangChain.Agents.Middleware.TodoList,
            LangChain.Agents.Middleware.SubAgent,
            LangChain.Agents.Middleware.FileSystem
          ]
        })

      # agent.middleware is now a list of MiddlewareEntry structs
      assert Enum.all?(agent.middleware, &match?(%MiddlewareEntry{}, &1))

      result = SubAgent.subagent_middleware_stack(agent.middleware)

      # SubAgent middleware should be filtered out
      refute Enum.any?(result, fn
        %MiddlewareEntry{module: LangChain.Agents.Middleware.SubAgent} -> true
        _ -> false
      end)

      # Other middleware should remain
      assert Enum.any?(result, fn
        %MiddlewareEntry{module: LangChain.Agents.Middleware.TodoList} -> true
        _ -> false
      end)

      assert Enum.any?(result, fn
        %MiddlewareEntry{module: LangChain.Agents.Middleware.FileSystem} -> true
        _ -> false
      end)

      # Should have 2 items (TodoList and FileSystem) - SubAgent filtered out
      # Note: Agent.new adds other default middleware, so we check for at least these
      assert length(result) >= 2

      # Verify NO SubAgent middleware in any format
      refute Enum.any?(result, &SubAgent.is_subagent_middleware?/1)
    end

    test "filters out SubAgent middleware from mixed formats" do
      # Mix of raw specs and MiddlewareEntry structs
      middleware_entry = %MiddlewareEntry{
        id: LangChain.Agents.Middleware.SubAgent,
        module: LangChain.Agents.Middleware.SubAgent,
        config: %{}
      }

      default_middleware = [
        {LangChain.Agents.Middleware.TodoList, []},
        middleware_entry,
        LangChain.Agents.Middleware.FileSystem
      ]

      result = SubAgent.subagent_middleware_stack(default_middleware)

      # SubAgent should be filtered out
      assert length(result) == 2
      refute Enum.any?(result, &SubAgent.is_subagent_middleware?/1)
    end
  end

  describe "is_subagent_middleware?/1" do
    test "returns true for raw SubAgent module" do
      assert SubAgent.is_subagent_middleware?(LangChain.Agents.Middleware.SubAgent)
    end

    test "returns true for SubAgent tuple" do
      assert SubAgent.is_subagent_middleware?({LangChain.Agents.Middleware.SubAgent, []})
      assert SubAgent.is_subagent_middleware?({LangChain.Agents.Middleware.SubAgent, [opt: 1]})
    end

    test "returns true for SubAgent MiddlewareEntry struct" do
      entry = %MiddlewareEntry{
        id: LangChain.Agents.Middleware.SubAgent,
        module: LangChain.Agents.Middleware.SubAgent,
        config: %{}
      }

      assert SubAgent.is_subagent_middleware?(entry)
    end

    test "returns false for other middleware modules" do
      refute SubAgent.is_subagent_middleware?(LangChain.Agents.Middleware.TodoList)
      refute SubAgent.is_subagent_middleware?({LangChain.Agents.Middleware.TodoList, []})
    end

    test "returns false for other MiddlewareEntry structs" do
      entry = %MiddlewareEntry{
        id: LangChain.Agents.Middleware.TodoList,
        module: LangChain.Agents.Middleware.TodoList,
        config: %{}
      }

      refute SubAgent.is_subagent_middleware?(entry)
    end

    test "returns false for unknown types" do
      refute SubAgent.is_subagent_middleware?("string")
      refute SubAgent.is_subagent_middleware?(123)
      refute SubAgent.is_subagent_middleware?(%{})
    end
  end

  describe "extract_middleware_module/1" do
    test "extracts module from raw atom" do
      assert SubAgent.extract_middleware_module(LangChain.Agents.Middleware.TodoList) ==
               LangChain.Agents.Middleware.TodoList
    end

    test "extracts module from tuple" do
      assert SubAgent.extract_middleware_module({LangChain.Agents.Middleware.TodoList, []}) ==
               LangChain.Agents.Middleware.TodoList

      assert SubAgent.extract_middleware_module(
               {LangChain.Agents.Middleware.FileSystem, [opt: 1]}
             ) ==
               LangChain.Agents.Middleware.FileSystem
    end

    test "extracts module from MiddlewareEntry struct" do
      entry = %MiddlewareEntry{
        id: LangChain.Agents.Middleware.TodoList,
        module: LangChain.Agents.Middleware.TodoList,
        config: %{some: "config"}
      }

      assert SubAgent.extract_middleware_module(entry) == LangChain.Agents.Middleware.TodoList
    end

    test "returns nil for unknown formats" do
      assert SubAgent.extract_middleware_module("string") == nil
      assert SubAgent.extract_middleware_module(123) == nil
      assert SubAgent.extract_middleware_module(%{}) == nil
      assert SubAgent.extract_middleware_module([]) == nil
    end
  end

  describe "new_from_config/1" do
    test "creates SubAgent with proper initialization" do
      agent_config = test_agent()

      parent_state = %{messages: []}

      subagent =
        SubAgent.new_from_config(
          parent_agent_id: "test-parent",
          instructions: "Do something",
          agent_config: agent_config,
          parent_state: parent_state
        )

      assert %SubAgent{} = subagent
      assert subagent.status == :idle
      assert subagent.parent_agent_id == "test-parent"
      assert String.starts_with?(subagent.id, "test-parent-sub-")
      assert subagent.chain != nil
      assert subagent.interrupt_on == %{}
    end

    test "stores interrupt_on from agent_config middleware" do
      agent_config =
        Agent.new!(%{
          model: test_model(),
          system_prompt: "Test",
          replace_default_middleware: true,
          middleware: [
            {LangChain.Agents.Middleware.HumanInTheLoop,
             [interrupt_on: %{"dangerous_tool" => true}]}
          ]
        })

      subagent =
        SubAgent.new_from_config(
          parent_agent_id: "test",
          instructions: "Task",
          agent_config: agent_config,
          parent_state: %{messages: []}
        )

      # HumanInTheLoop middleware transforms true into a config map
      assert subagent.interrupt_on == %{
               "dangerous_tool" => %{allowed_decisions: [:approve, :edit, :reject]}
             }
    end
  end

  describe "new_from_compiled/1" do
    test "creates SubAgent from compiled agent" do
      compiled_agent = test_agent()

      subagent =
        SubAgent.new_from_compiled(
          parent_agent_id: "test-parent",
          instructions: "Do something",
          compiled_agent: compiled_agent,
          parent_state: %{messages: []}
        )

      assert %SubAgent{} = subagent
      assert subagent.status == :idle
      assert subagent.parent_agent_id == "test-parent"
      assert subagent.chain != nil
    end

    test "stores interrupt_on from compiled_agent middleware" do
      compiled_agent =
        Agent.new!(%{
          model: test_model(),
          system_prompt: "Test",
          replace_default_middleware: true,
          middleware: [
            {LangChain.Agents.Middleware.HumanInTheLoop, [interrupt_on: %{"write_file" => true}]}
          ]
        })

      subagent =
        SubAgent.new_from_compiled(
          parent_agent_id: "test",
          instructions: "Task",
          compiled_agent: compiled_agent,
          parent_state: %{messages: []}
        )

      # HumanInTheLoop middleware transforms true into a config map
      assert subagent.interrupt_on == %{
               "write_file" => %{allowed_decisions: [:approve, :edit, :reject]}
             }
    end

    test "includes initial_messages in chain when provided" do
      compiled_agent = test_agent()

      initial_messages = [
        LangChain.Message.new_user!("Previous question"),
        LangChain.Message.new_assistant!("Previous answer")
      ]

      subagent =
        SubAgent.new_from_compiled(
          parent_agent_id: "test-parent",
          instructions: "New task",
          compiled_agent: compiled_agent,
          initial_messages: initial_messages,
          parent_state: %{messages: []}
        )

      assert %SubAgent{} = subagent
      # Chain should contain: system message + initial_messages + user instruction
      # Exact count depends on system_prompt (may be nil or present)
      assert length(subagent.chain.messages) >= 3

      # Verify initial messages are included in chain (check for content match)
      # Extract text content from messages (content is a list of ContentPart structs)
      message_texts =
        Enum.flat_map(subagent.chain.messages, fn msg ->
          Enum.map(msg.content, fn
            %LangChain.Message.ContentPart{type: :text, content: text} -> text
            _ -> ""
          end)
        end)

      assert "Previous question" in message_texts
      assert "Previous answer" in message_texts
      assert "New task" in message_texts
    end

    test "works with empty initial_messages list" do
      compiled_agent = test_agent()

      subagent =
        SubAgent.new_from_compiled(
          parent_agent_id: "test-parent",
          instructions: "Task",
          compiled_agent: compiled_agent,
          initial_messages: [],
          parent_state: %{messages: []}
        )

      assert %SubAgent{} = subagent
      assert subagent.chain != nil
    end
  end

  describe "SubAgent status checks" do
    test "can_execute?/1 returns true for idle SubAgent" do
      agent_config = test_agent()

      subagent =
        SubAgent.new_from_config(
          parent_agent_id: "test",
          instructions: "Task",
          agent_config: agent_config,
          parent_state: %{messages: []}
        )

      assert SubAgent.can_execute?(subagent)
    end

    test "can_execute?/1 returns false for non-idle SubAgent" do
      agent_config = test_agent()

      subagent =
        SubAgent.new_from_config(
          parent_agent_id: "test",
          instructions: "Task",
          agent_config: agent_config,
          parent_state: %{messages: []}
        )

      running_subagent = %{subagent | status: :running}
      refute SubAgent.can_execute?(running_subagent)
    end

    test "can_resume?/1 returns true for interrupted SubAgent" do
      agent_config = test_agent()

      subagent =
        SubAgent.new_from_config(
          parent_agent_id: "test",
          instructions: "Task",
          agent_config: agent_config,
          parent_state: %{messages: []}
        )

      interrupted_subagent = %{
        subagent
        | status: :interrupted,
          interrupt_data: %{action_requests: [], hitl_tool_call_ids: []}
      }

      assert SubAgent.can_resume?(interrupted_subagent)
    end

    test "can_resume?/1 returns false for non-interrupted SubAgent" do
      agent_config = test_agent()

      subagent =
        SubAgent.new_from_config(
          parent_agent_id: "test",
          instructions: "Task",
          agent_config: agent_config,
          parent_state: %{messages: []}
        )

      refute SubAgent.can_resume?(subagent)
    end

    test "is_terminal?/1 returns true for completed SubAgent" do
      agent_config = test_agent()

      subagent =
        SubAgent.new_from_config(
          parent_agent_id: "test",
          instructions: "Task",
          agent_config: agent_config,
          parent_state: %{messages: []}
        )

      completed_subagent = %{subagent | status: :completed}
      assert SubAgent.is_terminal?(completed_subagent)
    end

    test "is_terminal?/1 returns true for error SubAgent" do
      agent_config = test_agent()

      subagent =
        SubAgent.new_from_config(
          parent_agent_id: "test",
          instructions: "Task",
          agent_config: agent_config,
          parent_state: %{messages: []}
        )

      error_subagent = %{subagent | status: :error, error: "test error"}
      assert SubAgent.is_terminal?(error_subagent)
    end

    test "is_terminal?/1 returns false for non-terminal SubAgent" do
      agent_config = test_agent()

      subagent =
        SubAgent.new_from_config(
          parent_agent_id: "test",
          instructions: "Task",
          agent_config: agent_config,
          parent_state: %{messages: []}
        )

      refute SubAgent.is_terminal?(subagent)
    end
  end

  describe "execute/1" do
    test "returns error for non-idle SubAgent" do
      agent_config = test_agent()

      subagent =
        SubAgent.new_from_config(
          parent_agent_id: "test",
          instructions: "Task",
          agent_config: agent_config,
          parent_state: %{messages: []}
        )

      running_subagent = %{subagent | status: :running}

      assert {:error, {:invalid_status, :running, :expected_idle}} =
               SubAgent.execute(running_subagent)
    end
  end

  describe "resume/2" do
    test "returns error for non-interrupted SubAgent" do
      agent_config = test_agent()

      subagent =
        SubAgent.new_from_config(
          parent_agent_id: "test",
          instructions: "Task",
          agent_config: agent_config,
          parent_state: %{messages: []}
        )

      assert {:error, {:invalid_status, :idle, :expected_interrupted}} =
               SubAgent.resume(subagent, [])
    end
  end

  # Helper function to extract errors from changeset
  defp errors_on(changeset) do
    Ecto.Changeset.traverse_errors(changeset, fn {msg, opts} ->
      Enum.reduce(opts, msg, fn {key, value}, acc ->
        # Use inspect for complex types that don't implement String.Chars
        str_value =
          if is_binary(value) or is_number(value) or is_atom(value) do
            to_string(value)
          else
            inspect(value)
          end

        String.replace(acc, "%{#{key}}", str_value)
      end)
    end)
  end
end
