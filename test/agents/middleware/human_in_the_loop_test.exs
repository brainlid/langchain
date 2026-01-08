defmodule LangChain.Agents.Middleware.HumanInTheLoopTest do
  use LangChain.BaseCase, async: true

  alias LangChain.Agents.Middleware.HumanInTheLoop
  alias LangChain.Agents.State
  alias LangChain.Message
  alias LangChain.Message.ToolCall

  setup do
    %{agent_id: generate_test_agent_id()}
  end

  # Test helper: creates state with interrupt_data populated
  # This simulates what happens during agent execution when HITL detects tools needing approval
  defp setup_interrupted_state(state, config) do
    case HumanInTheLoop.check_for_interrupt(state, config) do
      {:interrupt, interrupt_data} ->
        %{state | interrupt_data: interrupt_data}

      :continue ->
        state
    end
  end

  describe "init/1" do
    test "normalizes simple boolean configuration" do
      opts = [
        interrupt_on: %{
          "write_file" => true,
          "delete_file" => false
        }
      ]

      assert {:ok, config} = HumanInTheLoop.init(opts)

      assert config.interrupt_on["write_file"] == %{
               allowed_decisions: [:approve, :edit, :reject]
             }

      assert config.interrupt_on["delete_file"] == %{allowed_decisions: []}
    end

    test "normalizes advanced configuration with allowed_decisions" do
      opts = [
        interrupt_on: %{
          "write_file" => %{allowed_decisions: [:approve, :reject]},
          "read_file" => %{allowed_decisions: [:approve]}
        }
      ]

      assert {:ok, config} = HumanInTheLoop.init(opts)

      assert config.interrupt_on["write_file"] == %{allowed_decisions: [:approve, :reject]}
      assert config.interrupt_on["read_file"] == %{allowed_decisions: [:approve]}
    end

    test "handles empty configuration" do
      opts = [interrupt_on: %{}]
      assert {:ok, config} = HumanInTheLoop.init(opts)
      assert config.interrupt_on == %{}
    end

    test "handles missing interrupt_on option" do
      opts = []
      assert {:ok, config} = HumanInTheLoop.init(opts)
      assert config.interrupt_on == %{}
    end

    test "adds default decisions if map config missing allowed_decisions" do
      opts = [
        interrupt_on: %{
          "write_file" => %{some_other_field: "value"}
        }
      ]

      assert {:ok, config} = HumanInTheLoop.init(opts)

      assert config.interrupt_on["write_file"] == %{
               some_other_field: "value",
               allowed_decisions: [:approve, :edit, :reject]
             }
    end
  end

  describe "check_for_interrupt/2" do
    test "returns :continue when no tool calls", %{agent_id: agent_id} do
      messages = [
        Message.new_system!("You are helpful"),
        Message.new_user!("Hello"),
        Message.new_assistant!("Hi there!")
      ]

      state = State.new!(%{agent_id: agent_id, messages: messages})
      config = %{interrupt_on: %{}}

      assert :continue = HumanInTheLoop.check_for_interrupt(state, config)
    end

    test "returns :continue when tool not in interrupt_on", %{agent_id: agent_id} do
      tool_call =
        ToolCall.new!(%{
          call_id: "123",
          name: "read_file",
          arguments: %{"path" => "test.txt"}
        })

      messages = [
        Message.new_user!("Read the file"),
        Message.new_assistant!(%{tool_calls: [tool_call]})
      ]

      state = State.new!(%{agent_id: agent_id, messages: messages})
      config = %{interrupt_on: %{"write_file" => %{allowed_decisions: [:approve]}}}

      assert :continue = HumanInTheLoop.check_for_interrupt(state, config)
    end

    test "returns :continue when tool has empty allowed_decisions", %{agent_id: agent_id} do
      tool_call =
        ToolCall.new!(%{
          call_id: "123",
          name: "read_file",
          arguments: %{"path" => "test.txt"}
        })

      messages = [
        Message.new_user!("Read the file"),
        Message.new_assistant!(%{tool_calls: [tool_call]})
      ]

      state = State.new!(%{agent_id: agent_id, messages: messages})
      config = %{interrupt_on: %{"read_file" => %{allowed_decisions: []}}}

      assert :continue = HumanInTheLoop.check_for_interrupt(state, config)
    end

    test "generates interrupt for single tool call", %{agent_id: agent_id} do
      tool_call =
        ToolCall.new!(%{
          call_id: "call_123",
          name: "write_file",
          arguments: %{"path" => "test.txt", "content" => "Hello"}
        })

      messages = [
        Message.new_user!("Write a file"),
        Message.new_assistant!(%{tool_calls: [tool_call]})
      ]

      state = State.new!(%{agent_id: agent_id, messages: messages})

      config = %{
        interrupt_on: %{
          "write_file" => %{allowed_decisions: [:approve, :edit, :reject]}
        }
      }

      assert {:interrupt, interrupt_data} = HumanInTheLoop.check_for_interrupt(state, config)

      assert %{
               action_requests: [action],
               review_configs: review_configs,
               hitl_tool_call_ids: hitl_ids
             } = interrupt_data

      assert action == %{
               tool_call_id: "call_123",
               tool_name: "write_file",
               arguments: %{"path" => "test.txt", "content" => "Hello"}
             }

      assert review_configs["write_file"] == %{allowed_decisions: [:approve, :edit, :reject]}
      assert hitl_ids == ["call_123"]
    end

    test "generates interrupt for multiple tool calls", %{agent_id: agent_id} do
      tool_call1 =
        ToolCall.new!(%{
          call_id: "call_1",
          name: "write_file",
          arguments: %{"path" => "file1.txt"}
        })

      tool_call2 =
        ToolCall.new!(%{
          call_id: "call_2",
          name: "delete_file",
          arguments: %{"path" => "old.txt"}
        })

      messages = [
        Message.new_user!("Update files"),
        Message.new_assistant!(%{tool_calls: [tool_call1, tool_call2]})
      ]

      state = State.new!(%{agent_id: agent_id, messages: messages})

      config = %{
        interrupt_on: %{
          "write_file" => %{allowed_decisions: [:approve, :reject]},
          "delete_file" => %{allowed_decisions: [:approve, :reject]}
        }
      }

      assert {:interrupt, interrupt_data} = HumanInTheLoop.check_for_interrupt(state, config)

      assert %{
               action_requests: actions,
               review_configs: review_configs,
               hitl_tool_call_ids: hitl_ids
             } = interrupt_data

      assert length(actions) == 2

      assert Enum.at(actions, 0) == %{
               tool_call_id: "call_1",
               tool_name: "write_file",
               arguments: %{"path" => "file1.txt"}
             }

      assert Enum.at(actions, 1) == %{
               tool_call_id: "call_2",
               tool_name: "delete_file",
               arguments: %{"path" => "old.txt"}
             }

      assert review_configs["write_file"] == %{allowed_decisions: [:approve, :reject]}
      assert review_configs["delete_file"] == %{allowed_decisions: [:approve, :reject]}
      assert hitl_ids == ["call_1", "call_2"]
    end

    test "generates interrupt only for configured tools", %{agent_id: agent_id} do
      tool_call1 =
        ToolCall.new!(%{
          call_id: "call_1",
          name: "read_file",
          arguments: %{"path" => "test.txt"}
        })

      tool_call2 =
        ToolCall.new!(%{
          call_id: "call_2",
          name: "write_file",
          arguments: %{"path" => "output.txt"}
        })

      messages = [
        Message.new_user!("Process file"),
        Message.new_assistant!(%{tool_calls: [tool_call1, tool_call2]})
      ]

      state = State.new!(%{agent_id: agent_id, messages: messages})

      config = %{
        interrupt_on: %{
          "write_file" => %{allowed_decisions: [:approve, :reject]}
          # read_file not configured
        }
      }

      assert {:interrupt, interrupt_data} = HumanInTheLoop.check_for_interrupt(state, config)

      # Should only interrupt for write_file
      assert %{action_requests: [action], hitl_tool_call_ids: hitl_ids} = interrupt_data
      assert action.tool_name == "write_file"
      assert action.tool_call_id == "call_2"

      # hitl_tool_call_ids should only contain write_file's ID
      assert hitl_ids == ["call_2"]
    end

    test "mixed tools: HITL first, non-HITL last", %{agent_id: agent_id} do
      tool_call1 =
        ToolCall.new!(%{
          call_id: "call_1",
          name: "delete_file",
          arguments: %{"path" => "old.txt"}
        })

      tool_call2 =
        ToolCall.new!(%{
          call_id: "call_2",
          name: "read_file",
          arguments: %{"path" => "test.txt"}
        })

      tool_call3 =
        ToolCall.new!(%{
          call_id: "call_3",
          name: "write_file",
          arguments: %{"path" => "output.txt"}
        })

      messages = [
        Message.new_user!("Process files"),
        Message.new_assistant!(%{tool_calls: [tool_call1, tool_call2, tool_call3]})
      ]

      state = State.new!(%{agent_id: agent_id, messages: messages})

      config = %{
        interrupt_on: %{
          "delete_file" => %{allowed_decisions: [:approve, :reject]},
          "write_file" => %{allowed_decisions: [:approve, :edit, :reject]}
          # read_file not configured - should execute without approval
        }
      }

      assert {:interrupt, interrupt_data} = HumanInTheLoop.check_for_interrupt(state, config)

      # Should interrupt for delete_file and write_file only
      assert %{action_requests: actions, hitl_tool_call_ids: hitl_ids} = interrupt_data
      assert length(actions) == 2
      assert Enum.at(actions, 0).tool_name == "delete_file"
      assert Enum.at(actions, 1).tool_name == "write_file"

      # hitl_tool_call_ids should contain both HITL tool IDs
      assert hitl_ids == ["call_1", "call_3"]
    end

    test "mixed tools: non-HITL first, HITL last", %{agent_id: agent_id} do
      tool_call1 =
        ToolCall.new!(%{
          call_id: "call_1",
          name: "read_file",
          arguments: %{"path" => "test.txt"}
        })

      tool_call2 =
        ToolCall.new!(%{
          call_id: "call_2",
          name: "delete_file",
          arguments: %{"path" => "old.txt"}
        })

      messages = [
        Message.new_user!("Process file"),
        Message.new_assistant!(%{tool_calls: [tool_call1, tool_call2]})
      ]

      state = State.new!(%{agent_id: agent_id, messages: messages})

      config = %{
        interrupt_on: %{
          "delete_file" => %{allowed_decisions: [:approve, :reject]}
          # read_file not configured
        }
      }

      assert {:interrupt, interrupt_data} = HumanInTheLoop.check_for_interrupt(state, config)

      # Should only interrupt for delete_file
      assert %{action_requests: [action], hitl_tool_call_ids: hitl_ids} = interrupt_data
      assert action.tool_name == "delete_file"
      assert action.tool_call_id == "call_2"

      # hitl_tool_call_ids should only contain delete_file's ID
      assert hitl_ids == ["call_2"]
    end

    test "mixed tools: non-HITL in the middle", %{agent_id: agent_id} do
      tool_call1 =
        ToolCall.new!(%{
          call_id: "call_1",
          name: "write_file",
          arguments: %{"path" => "output1.txt"}
        })

      tool_call2 =
        ToolCall.new!(%{
          call_id: "call_2",
          name: "read_file",
          arguments: %{"path" => "test.txt"}
        })

      tool_call3 =
        ToolCall.new!(%{
          call_id: "call_3",
          name: "delete_file",
          arguments: %{"path" => "old.txt"}
        })

      messages = [
        Message.new_user!("Process files"),
        Message.new_assistant!(%{tool_calls: [tool_call1, tool_call2, tool_call3]})
      ]

      state = State.new!(%{agent_id: agent_id, messages: messages})

      config = %{
        interrupt_on: %{
          "write_file" => %{allowed_decisions: [:approve, :edit, :reject]},
          "delete_file" => %{allowed_decisions: [:approve, :reject]}
          # read_file not configured
        }
      }

      assert {:interrupt, interrupt_data} = HumanInTheLoop.check_for_interrupt(state, config)

      # Should interrupt for write_file and delete_file
      assert %{action_requests: actions, hitl_tool_call_ids: hitl_ids} = interrupt_data
      assert length(actions) == 2
      assert Enum.at(actions, 0).tool_name == "write_file"
      assert Enum.at(actions, 1).tool_name == "delete_file"

      # hitl_tool_call_ids should contain both HITL tool IDs
      assert hitl_ids == ["call_1", "call_3"]
    end

    test "uses last assistant message with tool calls", %{agent_id: agent_id} do
      tool_call1 =
        ToolCall.new!(%{
          call_id: "call_old",
          name: "write_file",
          arguments: %{"path" => "old.txt"}
        })

      tool_call2 =
        ToolCall.new!(%{
          call_id: "call_new",
          name: "write_file",
          arguments: %{"path" => "new.txt"}
        })

      messages = [
        Message.new_user!("First request"),
        Message.new_assistant!(%{tool_calls: [tool_call1]}),
        Message.new_user!("Second request"),
        Message.new_assistant!(%{tool_calls: [tool_call2]})
      ]

      state = State.new!(%{agent_id: agent_id, messages: messages})

      config = %{
        interrupt_on: %{
          "write_file" => %{allowed_decisions: [:approve]}
        }
      }

      assert {:interrupt, interrupt_data} = HumanInTheLoop.check_for_interrupt(state, config)

      # Should only use the last assistant message's tool call
      assert %{action_requests: [action], hitl_tool_call_ids: hitl_ids} = interrupt_data
      assert action.tool_call_id == "call_new"
      assert hitl_ids == ["call_new"]
    end
  end

  describe "process_decisions/3" do
    setup %{agent_id: agent_id} do
      tool_call1 =
        ToolCall.new!(%{
          call_id: "call_1",
          name: "write_file",
          arguments: %{"path" => "file1.txt", "content" => "Hello"}
        })

      tool_call2 =
        ToolCall.new!(%{
          call_id: "call_2",
          name: "delete_file",
          arguments: %{"path" => "old.txt"}
        })

      messages = [
        Message.new_user!("Update files"),
        Message.new_assistant!(%{tool_calls: [tool_call1, tool_call2]})
      ]

      initial_state = State.new!(%{agent_id: agent_id, messages: messages})

      config = %{
        interrupt_on: %{
          "write_file" => %{allowed_decisions: [:approve, :edit, :reject]},
          "delete_file" => %{allowed_decisions: [:approve, :reject]}
        }
      }

      # Populate interrupt_data using test helper
      state = setup_interrupted_state(initial_state, config)

      %{state: state, config: config}
    end

    test "validates approve decisions", %{state: state, config: config} do
      decisions = [
        %{type: :approve},
        %{type: :approve}
      ]

      assert {:ok, ^state} = HumanInTheLoop.process_decisions(state, decisions, config)

      # State should be unchanged - no tool result message added
      assert [_user, _assistant] = state.messages
    end

    test "validates edit decision", %{state: state, config: config} do
      decisions = [
        %{type: :edit, arguments: %{"path" => "modified.txt", "content" => "Modified"}},
        %{type: :approve}
      ]

      assert {:ok, ^state} = HumanInTheLoop.process_decisions(state, decisions, config)

      # State should be unchanged - no tool result message added
      assert [_user, _assistant] = state.messages
    end

    test "validates reject decision", %{state: state, config: config} do
      decisions = [
        %{type: :reject},
        %{type: :approve}
      ]

      assert {:ok, ^state} = HumanInTheLoop.process_decisions(state, decisions, config)

      # State should be unchanged - no tool result message added
      assert [_user, _assistant] = state.messages
    end

    test "validates mixed decisions", %{state: state, config: config} do
      decisions = [
        %{type: :edit, arguments: %{"path" => "new.txt"}},
        %{type: :reject}
      ]

      assert {:ok, ^state} = HumanInTheLoop.process_decisions(state, decisions, config)

      # State should be unchanged - no tool result message added
      assert [_user, _assistant] = state.messages
    end

    test "returns error when decision count mismatches tool call count", %{
      state: state,
      config: config
    } do
      # Only 1 decision for 2 tool calls
      decisions = [%{type: :approve}]

      assert {:error, reason} = HumanInTheLoop.process_decisions(state, decisions, config)
      assert reason =~ "Decision count"
      assert reason =~ "does not match"
    end

    test "returns error when decision type is missing", %{state: state, config: config} do
      decisions = [
        # Missing type field
        %{},
        %{type: :approve}
      ]

      assert {:error, reason} = HumanInTheLoop.process_decisions(state, decisions, config)
      assert reason =~ "missing required 'type' field"
    end

    test "returns error when decision type not allowed for tool", %{state: state, config: config} do
      decisions = [
        %{type: :approve},
        # edit not allowed for delete_file
        %{type: :edit, arguments: %{}}
      ]

      assert {:error, reason} = HumanInTheLoop.process_decisions(state, decisions, config)
      assert reason =~ "Decision type 'edit' not allowed"
      assert reason =~ "delete_file"
    end

    test "returns error when edit decision missing arguments", %{state: state, config: config} do
      decisions = [
        # Missing arguments field
        %{type: :edit},
        %{type: :approve}
      ]

      assert {:error, reason} = HumanInTheLoop.process_decisions(state, decisions, config)
      assert reason =~ "type 'edit' must include 'arguments' field"
    end

    test "returns error when no tool calls found in state", %{agent_id: agent_id} do
      messages = [
        Message.new_user!("Hello"),
        Message.new_assistant!("Hi there!")
      ]

      state = State.new!(%{agent_id: agent_id, messages: messages})
      config = %{interrupt_on: %{}}
      decisions = [%{type: :approve}]

      assert {:error, reason} = HumanInTheLoop.process_decisions(state, decisions, config)
      assert reason =~ "No interrupt data found"
    end

    test "uses default decisions when tool not in config", %{agent_id: agent_id} do
      tool_call1 =
        ToolCall.new!(%{
          call_id: "call_1",
          name: "write_file",
          arguments: %{"path" => "file1.txt", "content" => "Hello"}
        })

      tool_call2 =
        ToolCall.new!(%{
          call_id: "call_2",
          name: "delete_file",
          arguments: %{"path" => "old.txt"}
        })

      messages = [
        Message.new_user!("Update files"),
        Message.new_assistant!(%{tool_calls: [tool_call1, tool_call2]})
      ]

      initial_state = State.new!(%{agent_id: agent_id, messages: messages})

      # Config doesn't include write_file or delete_file
      config = %{interrupt_on: %{}}

      # Manually populate interrupt_data with default decisions
      # Since config is empty, we need to simulate what would happen if tools were configured
      interrupt_data = %{
        action_requests: [
          %{tool_call_id: "call_1", tool_name: "write_file", arguments: %{}},
          %{tool_call_id: "call_2", tool_name: "delete_file", arguments: %{}}
        ],
        review_configs: %{},
        hitl_tool_call_ids: ["call_1", "call_2"]
      }

      state = %{initial_state | interrupt_data: interrupt_data}

      decisions = [
        %{type: :approve},
        %{type: :approve}
      ]

      # Should still work with default allowed_decisions
      assert {:ok, _updated_state} = HumanInTheLoop.process_decisions(state, decisions, config)
    end

    test "validates against default decisions when tool not in config", %{agent_id: agent_id} do
      tool_call1 =
        ToolCall.new!(%{
          call_id: "call_1",
          name: "write_file",
          arguments: %{"path" => "file1.txt", "content" => "Hello"}
        })

      tool_call2 =
        ToolCall.new!(%{
          call_id: "call_2",
          name: "delete_file",
          arguments: %{"path" => "old.txt"}
        })

      messages = [
        Message.new_user!("Update files"),
        Message.new_assistant!(%{tool_calls: [tool_call1, tool_call2]})
      ]

      initial_state = State.new!(%{agent_id: agent_id, messages: messages})

      config = %{interrupt_on: %{}}

      # Manually populate interrupt_data with default decisions
      interrupt_data = %{
        action_requests: [
          %{tool_call_id: "call_1", tool_name: "write_file", arguments: %{}},
          %{tool_call_id: "call_2", tool_name: "delete_file", arguments: %{}}
        ],
        review_configs: %{},
        hitl_tool_call_ids: ["call_1", "call_2"]
      }

      state = %{initial_state | interrupt_data: interrupt_data}

      # Try an invalid decision type
      decisions = [
        %{type: :invalid_type},
        %{type: :approve}
      ]

      assert {:error, reason} = HumanInTheLoop.process_decisions(state, decisions, config)
      assert reason =~ "not allowed"
    end
  end

  describe "process_decisions/3 with mixed tools" do
    test "validates decisions for HITL tools only (non-HITL first)", %{agent_id: agent_id} do
      tool_call1 =
        ToolCall.new!(%{
          call_id: "call_1",
          name: "read_file",
          arguments: %{"path" => "test.txt"}
        })

      tool_call2 =
        ToolCall.new!(%{
          call_id: "call_2",
          name: "write_file",
          arguments: %{"path" => "output.txt"}
        })

      messages = [
        Message.new_user!("Process file"),
        Message.new_assistant!(%{tool_calls: [tool_call1, tool_call2]})
      ]

      state = State.new!(%{agent_id: agent_id, messages: messages})

      config = %{
        interrupt_on: %{
          "write_file" => %{allowed_decisions: [:approve, :edit, :reject]}
          # read_file not configured - should not require decision
        }
      }

      # Simulate the interrupt flow to populate interrupt_data
      state_with_interrupt_data = setup_interrupted_state(state, config)

      # Only need 1 decision for write_file (not 2 for all tools)
      decisions = [
        %{type: :approve}
      ]

      assert {:ok, _state} =
               HumanInTheLoop.process_decisions(state_with_interrupt_data, decisions, config)
    end

    test "validates decisions for HITL tools only (non-HITL last)", %{agent_id: agent_id} do
      tool_call1 =
        ToolCall.new!(%{
          call_id: "call_1",
          name: "delete_file",
          arguments: %{"path" => "old.txt"}
        })

      tool_call2 =
        ToolCall.new!(%{
          call_id: "call_2",
          name: "read_file",
          arguments: %{"path" => "test.txt"}
        })

      messages = [
        Message.new_user!("Process file"),
        Message.new_assistant!(%{tool_calls: [tool_call1, tool_call2]})
      ]

      state = State.new!(%{agent_id: agent_id, messages: messages})

      config = %{
        interrupt_on: %{
          "delete_file" => %{allowed_decisions: [:approve, :reject]}
          # read_file not configured
        }
      }

      # Simulate the interrupt flow
      state_with_interrupt_data = setup_interrupted_state(state, config)

      # Only need 1 decision for delete_file
      decisions = [
        %{type: :approve}
      ]

      assert {:ok, _state} =
               HumanInTheLoop.process_decisions(state_with_interrupt_data, decisions, config)
    end

    test "validates decisions for HITL tools only (non-HITL in middle)", %{agent_id: agent_id} do
      tool_call1 =
        ToolCall.new!(%{
          call_id: "call_1",
          name: "write_file",
          arguments: %{"path" => "output1.txt"}
        })

      tool_call2 =
        ToolCall.new!(%{
          call_id: "call_2",
          name: "read_file",
          arguments: %{"path" => "test.txt"}
        })

      tool_call3 =
        ToolCall.new!(%{
          call_id: "call_3",
          name: "delete_file",
          arguments: %{"path" => "old.txt"}
        })

      messages = [
        Message.new_user!("Process files"),
        Message.new_assistant!(%{tool_calls: [tool_call1, tool_call2, tool_call3]})
      ]

      state = State.new!(%{agent_id: agent_id, messages: messages})

      config = %{
        interrupt_on: %{
          "write_file" => %{allowed_decisions: [:approve, :edit, :reject]},
          "delete_file" => %{allowed_decisions: [:approve, :reject]}
          # read_file not configured
        }
      }

      # Simulate the interrupt flow
      state_with_interrupt_data = setup_interrupted_state(state, config)

      # Need 2 decisions for write_file and delete_file (not 3 for all tools)
      decisions = [
        %{type: :approve},
        %{type: :reject}
      ]

      assert {:ok, _state} =
               HumanInTheLoop.process_decisions(state_with_interrupt_data, decisions, config)
    end

    test "returns error when decision count exceeds HITL tool count", %{agent_id: agent_id} do
      tool_call1 =
        ToolCall.new!(%{
          call_id: "call_1",
          name: "read_file",
          arguments: %{"path" => "test.txt"}
        })

      tool_call2 =
        ToolCall.new!(%{
          call_id: "call_2",
          name: "write_file",
          arguments: %{"path" => "output.txt"}
        })

      messages = [
        Message.new_user!("Process file"),
        Message.new_assistant!(%{tool_calls: [tool_call1, tool_call2]})
      ]

      state = State.new!(%{agent_id: agent_id, messages: messages})

      config = %{
        interrupt_on: %{
          "write_file" => %{allowed_decisions: [:approve, :reject]}
        }
      }

      # Simulate the interrupt flow
      state_with_interrupt_data = setup_interrupted_state(state, config)

      # Providing 2 decisions when only 1 HITL tool
      decisions = [
        %{type: :approve},
        %{type: :approve}
      ]

      assert {:error, reason} =
               HumanInTheLoop.process_decisions(state_with_interrupt_data, decisions, config)

      assert reason =~ "Decision count (2) does not match HITL tool count (1)"
    end

    test "returns error when decision count is less than HITL tool count", %{agent_id: agent_id} do
      tool_call1 =
        ToolCall.new!(%{
          call_id: "call_1",
          name: "write_file",
          arguments: %{"path" => "output1.txt"}
        })

      tool_call2 =
        ToolCall.new!(%{
          call_id: "call_2",
          name: "read_file",
          arguments: %{"path" => "test.txt"}
        })

      tool_call3 =
        ToolCall.new!(%{
          call_id: "call_3",
          name: "delete_file",
          arguments: %{"path" => "old.txt"}
        })

      messages = [
        Message.new_user!("Process files"),
        Message.new_assistant!(%{tool_calls: [tool_call1, tool_call2, tool_call3]})
      ]

      state = State.new!(%{agent_id: agent_id, messages: messages})

      config = %{
        interrupt_on: %{
          "write_file" => %{allowed_decisions: [:approve, :edit, :reject]},
          "delete_file" => %{allowed_decisions: [:approve, :reject]}
        }
      }

      # Simulate the interrupt flow
      state_with_interrupt_data = setup_interrupted_state(state, config)

      # Providing only 1 decision when 2 HITL tools
      decisions = [
        %{type: :approve}
      ]

      assert {:error, reason} =
               HumanInTheLoop.process_decisions(state_with_interrupt_data, decisions, config)

      assert reason =~ "Decision count (1) does not match HITL tool count (2)"
    end

    test "returns error when interrupt_data is missing", %{agent_id: agent_id} do
      tool_call =
        ToolCall.new!(%{
          call_id: "call_1",
          name: "write_file",
          arguments: %{"path" => "output.txt"}
        })

      messages = [
        Message.new_user!("Write file"),
        Message.new_assistant!(%{tool_calls: [tool_call]})
      ]

      # State without interrupt_data
      state = State.new!(%{agent_id: agent_id, messages: messages})

      config = %{
        interrupt_on: %{
          "write_file" => %{allowed_decisions: [:approve, :reject]}
        }
      }

      decisions = [%{type: :approve}]

      assert {:error, reason} = HumanInTheLoop.process_decisions(state, decisions, config)
      assert reason =~ "No interrupt data found in state"
    end
  end
end
