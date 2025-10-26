defmodule LangChain.DeepAgents.Middleware.HumanInTheLoopTest do
  use ExUnit.Case, async: true

  alias LangChain.DeepAgents.Middleware.HumanInTheLoop
  alias LangChain.DeepAgents.State
  alias LangChain.Message
  alias LangChain.Message.ToolCall

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

  describe "after_model/2" do
    test "returns state unchanged when no messages" do
      state = State.new!(%{messages: []})
      config = %{interrupt_on: %{}}

      assert {:ok, ^state} = HumanInTheLoop.after_model(state, config)
    end

    test "returns state unchanged when no tool calls" do
      messages = [
        Message.new_system!("You are helpful"),
        Message.new_user!("Hello"),
        Message.new_assistant!("Hi there!")
      ]

      state = State.new!(%{messages: messages})
      config = %{interrupt_on: %{}}

      assert {:ok, ^state} = HumanInTheLoop.after_model(state, config)
    end

    test "returns state unchanged when tool not in interrupt_on" do
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

      state = State.new!(%{messages: messages})
      config = %{interrupt_on: %{"write_file" => %{allowed_decisions: [:approve]}}}

      assert {:ok, ^state} = HumanInTheLoop.after_model(state, config)
    end

    test "returns state unchanged when tool has empty allowed_decisions" do
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

      state = State.new!(%{messages: messages})
      config = %{interrupt_on: %{"read_file" => %{allowed_decisions: []}}}

      assert {:ok, ^state} = HumanInTheLoop.after_model(state, config)
    end

    test "generates interrupt for single tool call" do
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

      state = State.new!(%{messages: messages})

      config = %{
        interrupt_on: %{
          "write_file" => %{allowed_decisions: [:approve, :edit, :reject]}
        }
      }

      assert {:interrupt, ^state, interrupt_data} = HumanInTheLoop.after_model(state, config)

      assert %{action_requests: [action], review_configs: review_configs} = interrupt_data

      assert action == %{
               tool_call_id: "call_123",
               tool_name: "write_file",
               arguments: %{"path" => "test.txt", "content" => "Hello"}
             }

      assert review_configs["write_file"] == %{allowed_decisions: [:approve, :edit, :reject]}
    end

    test "generates interrupt for multiple tool calls" do
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

      state = State.new!(%{messages: messages})

      config = %{
        interrupt_on: %{
          "write_file" => %{allowed_decisions: [:approve, :reject]},
          "delete_file" => %{allowed_decisions: [:approve, :reject]}
        }
      }

      assert {:interrupt, ^state, interrupt_data} = HumanInTheLoop.after_model(state, config)

      assert %{action_requests: actions, review_configs: review_configs} = interrupt_data
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
    end

    test "generates interrupt only for configured tools" do
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

      state = State.new!(%{messages: messages})

      config = %{
        interrupt_on: %{
          "write_file" => %{allowed_decisions: [:approve, :reject]}
          # read_file not configured
        }
      }

      assert {:interrupt, ^state, interrupt_data} = HumanInTheLoop.after_model(state, config)

      # Should only interrupt for write_file
      assert %{action_requests: [action]} = interrupt_data
      assert action.tool_name == "write_file"
    end

    test "uses last assistant message with tool calls" do
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

      state = State.new!(%{messages: messages})

      config = %{
        interrupt_on: %{
          "write_file" => %{allowed_decisions: [:approve]}
        }
      }

      assert {:interrupt, ^state, interrupt_data} = HumanInTheLoop.after_model(state, config)

      # Should only use the last assistant message's tool call
      assert %{action_requests: [action]} = interrupt_data
      assert action.tool_call_id == "call_new"
    end
  end

  describe "process_decisions/3" do
    setup do
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

      state = State.new!(%{messages: messages})

      config = %{
        interrupt_on: %{
          "write_file" => %{allowed_decisions: [:approve, :edit, :reject]},
          "delete_file" => %{allowed_decisions: [:approve, :reject]}
        }
      }

      %{state: state, config: config}
    end

    test "processes approve decision", %{state: state, config: config} do
      decisions = [
        %{type: :approve},
        %{type: :approve}
      ]

      assert {:ok, updated_state} = HumanInTheLoop.process_decisions(state, decisions, config)

      # Should have added tool result message
      assert [_user, _assistant, tool_message] = updated_state.messages
      assert tool_message.role == :tool
      assert length(tool_message.tool_results) == 2

      [result1, result2] = tool_message.tool_results

      assert result1.tool_call_id == "call_1"
      assert result1.name == "write_file"

      # Content is stored as ContentParts
      assert [%Message.ContentPart{content: content1}] = result1.content
      assert content1 =~ "approved for execution"

      assert result2.tool_call_id == "call_2"
      assert result2.name == "delete_file"
    end

    test "processes edit decision", %{state: state, config: config} do
      decisions = [
        %{type: :edit, arguments: %{"path" => "modified.txt", "content" => "Modified"}},
        %{type: :approve}
      ]

      assert {:ok, updated_state} = HumanInTheLoop.process_decisions(state, decisions, config)

      assert [_user, _assistant, tool_message] = updated_state.messages
      [result1, _result2] = tool_message.tool_results

      assert result1.tool_call_id == "call_1"
      assert result1.name == "write_file"

      # Content is stored as ContentParts
      assert [%Message.ContentPart{content: content1}] = result1.content
      assert content1 =~ "edited arguments"
      assert content1 =~ "modified.txt"
    end

    test "processes reject decision", %{state: state, config: config} do
      decisions = [
        %{type: :reject},
        %{type: :approve}
      ]

      assert {:ok, updated_state} = HumanInTheLoop.process_decisions(state, decisions, config)

      assert [_user, _assistant, tool_message] = updated_state.messages
      [result1, _result2] = tool_message.tool_results

      assert result1.tool_call_id == "call_1"
      assert result1.name == "write_file"

      # Content is stored as ContentParts
      assert [%Message.ContentPart{content: content1}] = result1.content
      assert content1 =~ "rejected by human reviewer"
    end

    test "processes mixed decisions", %{state: state, config: config} do
      decisions = [
        %{type: :edit, arguments: %{"path" => "new.txt"}},
        %{type: :reject}
      ]

      assert {:ok, updated_state} = HumanInTheLoop.process_decisions(state, decisions, config)

      assert [_user, _assistant, tool_message] = updated_state.messages
      assert length(tool_message.tool_results) == 2

      [result1, result2] = tool_message.tool_results

      # Content is stored as ContentParts
      assert [%Message.ContentPart{content: content1}] = result1.content
      assert content1 =~ "edited arguments"

      assert [%Message.ContentPart{content: content2}] = result2.content
      assert content2 =~ "rejected"
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

    test "returns error when no tool calls found in state" do
      messages = [
        Message.new_user!("Hello"),
        Message.new_assistant!("Hi there!")
      ]

      state = State.new!(%{messages: messages})
      config = %{interrupt_on: %{}}
      decisions = [%{type: :approve}]

      assert {:error, reason} = HumanInTheLoop.process_decisions(state, decisions, config)
      assert reason =~ "No tool calls found"
    end

    test "uses default decisions when tool not in config", %{state: state} do
      # Config doesn't include write_file or delete_file
      config = %{interrupt_on: %{}}

      decisions = [
        %{type: :approve},
        %{type: :approve}
      ]

      # Should still work with default allowed_decisions
      assert {:ok, _updated_state} = HumanInTheLoop.process_decisions(state, decisions, config)
    end

    test "validates against default decisions when tool not in config", %{state: state} do
      config = %{interrupt_on: %{}}

      # Try an invalid decision type
      decisions = [
        %{type: :invalid_type},
        %{type: :approve}
      ]

      assert {:error, reason} = HumanInTheLoop.process_decisions(state, decisions, config)
      assert reason =~ "not allowed"
    end
  end
end
