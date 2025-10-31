defmodule LangChain.Agents.SubAgentTest do
  use ExUnit.Case, async: true

  alias LangChain.Agents.{SubAgent, Agent, State, Todo}
  alias LangChain.Function
  alias LangChain.ChatModels.ChatAnthropic
  alias LangChain.Message

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
    Agent.new!(
      model: test_model(),
      system_prompt: "Test agent",
      replace_default_middleware: true,
      middleware: []
    )
  end

  describe "SubAgent.SubAgentConfig.new/1" do
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

  describe "SubAgent.SubAgentConfig.new!/1" do
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

  describe "SubAgent.SubAgentCompiled.new/1" do
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

  describe "SubAgent.SubAgentCompiled.new!/1" do
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

  describe "SubAgent.build_registry/3" do
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

      assert {:ok, registry} = SubAgent.build_registry(configs, model, middleware)
      assert map_size(registry) == 2
      assert %Agent{} = registry["agent1"]
      assert %Agent{} = registry["agent2"]
    end

    test "builds registry from Compiled subagents" do
      model = test_model()
      agent1 = test_agent()
      agent2 = test_agent()

      configs = [
        SubAgentCompiled.new!(%{
          name: "custom1",
          description: "Custom 1",
          agent: agent1
        }),
        SubAgentCompiled.new!(%{
          name: "custom2",
          description: "Custom 2",
          agent: agent2
        })
      ]

      assert {:ok, registry} = SubAgent.build_registry(configs, model, [])
      assert map_size(registry) == 2
      assert registry["custom1"] == agent1
      assert registry["custom2"] == agent2
    end

    test "builds registry from mixed Config and Compiled" do
      model = test_model()
      pre_built = test_agent()

      configs = [
        SubAgentConfig.new!(%{
          name: "dynamic",
          description: "Dynamic agent",
          system_prompt: "Dynamic",
          tools: [test_tool()]
        }),
        SubAgentCompiled.new!(%{
          name: "compiled",
          description: "Compiled agent",
          agent: pre_built
        })
      ]

      assert {:ok, registry} = SubAgent.build_registry(configs, model, [])
      assert map_size(registry) == 2
      assert %Agent{} = registry["dynamic"]
      assert registry["compiled"] == pre_built
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

      assert {:ok, registry} = SubAgent.build_registry(configs, model, [])
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

      assert {:ok, registry} = SubAgent.build_registry(configs, default_model, [])
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

      assert {:ok, registry} = SubAgent.build_registry(configs, model, [])
      agent = registry["agent"]
      # Middleware should be appended
      assert Enum.any?(agent.middleware, fn
               {CustomMiddleware, _} -> true
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

      assert {:error, reason} = SubAgent.build_registry(configs, model, [])
      assert reason =~ "Failed to build subagent registry"
    end

    test "handles empty configs list" do
      model = test_model()

      assert {:ok, registry} = SubAgent.build_registry([], model, [])
      assert registry == %{}
    end
  end

  describe "SubAgent.build_registry!/3" do
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

      registry = SubAgent.build_registry!(configs, model, [])
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
        SubAgent.build_registry!(configs, model, [])
      end
    end
  end

  describe "SubAgent.build_descriptions/1" do
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

  describe "SubAgent.subagent_middleware_stack/2" do
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
  end

  describe "SubAgent.excluded_state_keys/0" do
    test "returns the list of excluded state keys" do
      keys = SubAgent.excluded_state_keys()
      assert :messages in keys
      assert :todos in keys
      assert length(keys) == 2
    end
  end

  describe "SubAgent.prepare_subagent_state/2" do
    test "creates fresh conversation with task description" do
      parent_state =
        State.new!(%{
          messages: [
            Message.new_user!("Original task"),
            Message.new_assistant!("Response")
          ],
          files: %{"data.txt" => "content"},
          todos: [Todo.new!(%{content: "Parent todo"})]
        })

      subagent_state =
        SubAgent.prepare_subagent_state(
          "Research renewable energy",
          parent_state
        )

      # Fresh conversation with single user message
      assert length(subagent_state.messages) == 1
      assert [msg] = subagent_state.messages
      assert msg.role == :user
      # Content can be string or list of ContentParts
      content_text =
        case msg.content do
          text when is_binary(text) -> text
          [%{content: text} | _] -> text
          _ -> ""
        end

      assert content_text =~ "renewable energy"
    end

    test "inherits filesystem from parent" do
      parent_state =
        State.new!(%{
          files: %{
            "file1.txt" => "content1",
            "file2.txt" => "content2"
          }
        })

      subagent_state =
        SubAgent.prepare_subagent_state(
          "Do something",
          parent_state
        )

      assert subagent_state.files == parent_state.files
    end

    test "starts with empty todos" do
      parent_state =
        State.new!(%{
          todos: [
            Todo.new!(%{content: "Parent todo 1"}),
            Todo.new!(%{content: "Parent todo 2"})
          ]
        })

      subagent_state =
        SubAgent.prepare_subagent_state(
          "Do something",
          parent_state
        )

      assert subagent_state.todos == []
    end

    test "inherits metadata from parent" do
      parent_state =
        State.new!(%{
          metadata: %{key: "value", nested: %{data: 123}}
        })

      subagent_state =
        SubAgent.prepare_subagent_state(
          "Do something",
          parent_state
        )

      assert subagent_state.metadata == parent_state.metadata
    end

    test "inherits middleware_state from parent" do
      parent_state =
        State.new!(%{
          middleware_state: %{some_middleware: %{config: "value"}}
        })

      subagent_state =
        SubAgent.prepare_subagent_state(
          "Do something",
          parent_state
        )

      assert subagent_state.middleware_state == parent_state.middleware_state
    end
  end

  describe "SubAgent.extract_subagent_result/2" do
    test "extracts final message from result state" do
      result_state =
        State.new!(%{
          messages: [
            Message.new_user!("Task"),
            Message.new_assistant!("Intermediate"),
            Message.new_assistant!("Final result")
          ]
        })

      result = SubAgent.extract_subagent_result(result_state)

      content_text =
        case result.final_message.content do
          text when is_binary(text) -> text
          [%{content: text} | _] -> text
          _ -> ""
        end

      assert content_text == "Final result"
    end

    test "excludes messages from state updates" do
      result_state =
        State.new!(%{
          messages: [Message.new_assistant!("Final")],
          files: %{"output.txt" => "data"}
        })

      result = SubAgent.extract_subagent_result(result_state)

      refute Map.has_key?(result.state_updates, :messages)
      assert result.state_updates[:files] == %{"output.txt" => "data"}
    end

    test "excludes todos from state updates" do
      result_state =
        State.new!(%{
          messages: [Message.new_assistant!("Final")],
          todos: [Todo.new!(%{content: "SubAgent todo"})]
        })

      result = SubAgent.extract_subagent_result(result_state)

      refute Map.has_key?(result.state_updates, :todos)
    end

    test "includes files in state updates" do
      result_state =
        State.new!(%{
          messages: [Message.new_assistant!("Final")],
          files: %{"output.txt" => "result data"}
        })

      result = SubAgent.extract_subagent_result(result_state)

      assert result.state_updates[:files] == %{"output.txt" => "result data"}
    end

    test "includes metadata in state updates" do
      result_state =
        State.new!(%{
          messages: [Message.new_assistant!("Final")],
          metadata: %{computed: true, value: 42}
        })

      result = SubAgent.extract_subagent_result(result_state)

      assert result.state_updates[:metadata] == %{computed: true, value: 42}
    end

    test "includes tool_call_id when provided" do
      result_state =
        State.new!(%{
          messages: [Message.new_assistant!("Final")]
        })

      result = SubAgent.extract_subagent_result(result_state, "call_123")

      assert result.tool_call_id == "call_123"
    end

    test "handles nil tool_call_id" do
      result_state =
        State.new!(%{
          messages: [Message.new_assistant!("Final")]
        })

      result = SubAgent.extract_subagent_result(result_state)

      assert result.tool_call_id == nil
    end
  end

  describe "SubAgent.merge_subagent_result/2" do
    test "merges files from subagent" do
      parent_state =
        State.new!(%{
          files: %{"a.txt" => "A"}
        })

      subagent_result = %{
        final_message: Message.new_assistant!("Done"),
        state_updates: %{
          files: %{"b.txt" => "B"}
        }
      }

      merged = SubAgent.merge_subagent_result(parent_state, subagent_result)

      assert merged.files == %{"a.txt" => "A", "b.txt" => "B"}
    end

    test "subagent files override parent files" do
      parent_state =
        State.new!(%{
          files: %{"shared.txt" => "Parent version"}
        })

      subagent_result = %{
        final_message: Message.new_assistant!("Done"),
        state_updates: %{
          files: %{"shared.txt" => "SubAgent version"}
        }
      }

      merged = SubAgent.merge_subagent_result(parent_state, subagent_result)

      assert merged.files["shared.txt"] == "SubAgent version"
    end

    test "deep merges metadata" do
      parent_state =
        State.new!(%{
          metadata: %{step: 1, parent_key: "value"}
        })

      subagent_result = %{
        final_message: Message.new_assistant!("Done"),
        state_updates: %{
          metadata: %{step: 2, subagent_key: "new"}
        }
      }

      merged = SubAgent.merge_subagent_result(parent_state, subagent_result)

      assert merged.metadata == %{
               step: 2,
               parent_key: "value",
               subagent_key: "new"
             }
    end

    test "preserves parent messages" do
      parent_state =
        State.new!(%{
          messages: [
            Message.new_user!("Parent task"),
            Message.new_assistant!("Parent response")
          ]
        })

      subagent_result = %{
        final_message: Message.new_assistant!("SubAgent result"),
        state_updates: %{}
      }

      merged = SubAgent.merge_subagent_result(parent_state, subagent_result)

      # Parent messages unchanged
      assert length(merged.messages) == 2
      assert merged.messages == parent_state.messages
    end

    test "preserves parent todos" do
      parent_state =
        State.new!(%{
          todos: [Todo.new!(%{content: "Parent todo"})]
        })

      subagent_result = %{
        final_message: Message.new_assistant!("Done"),
        state_updates: %{}
      }

      merged = SubAgent.merge_subagent_result(parent_state, subagent_result)

      assert length(merged.todos) == 1
      assert merged.todos == parent_state.todos
    end

    test "handles missing files in state updates" do
      parent_state =
        State.new!(%{
          files: %{"a.txt" => "A"}
        })

      subagent_result = %{
        final_message: Message.new_assistant!("Done"),
        state_updates: %{}
      }

      merged = SubAgent.merge_subagent_result(parent_state, subagent_result)

      assert merged.files == %{"a.txt" => "A"}
    end

    test "handles missing metadata in state updates" do
      parent_state =
        State.new!(%{
          metadata: %{key: "value"}
        })

      subagent_result = %{
        final_message: Message.new_assistant!("Done"),
        state_updates: %{}
      }

      merged = SubAgent.merge_subagent_result(parent_state, subagent_result)

      assert merged.metadata == %{key: "value"}
    end
  end

  # Helper function to extract errors from changeset
  defp errors_on(changeset) do
    Ecto.Changeset.traverse_errors(changeset, fn {msg, opts} ->
      Enum.reduce(opts, msg, fn {key, value}, acc ->
        String.replace(acc, "%{#{key}}", to_string(value))
      end)
    end)
  end
end
