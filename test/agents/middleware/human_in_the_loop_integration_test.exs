defmodule LangChain.Agents.Middleware.HumanInTheLoopIntegrationTest do
  use ExUnit.Case, async: true

  alias LangChain.Agents.Agent
  alias LangChain.Agents.State
  alias LangChain.Message
  alias LangChain.Function
  alias LangChain.ChatModels.ChatAnthropic

  defp create_test_model do
    # Create a simple model for testing
    # In real tests, this would be mocked
    ChatAnthropic.new!(%{
      model: "claude-3-5-sonnet-20241022",
      temperature: 0
    })
  end

  defp create_write_file_tool do
    Function.new!(%{
      name: "write_file",
      description: "Write content to a file",
      parameters_schema: %{
        type: "object",
        properties: %{
          path: %{type: "string", description: "File path"},
          content: %{type: "string", description: "File content"}
        },
        required: ["path", "content"]
      },
      function: fn args, _context ->
        # Mock implementation
        {:ok, "File written: #{args["path"]}"}
      end
    })
  end

  defp create_read_file_tool do
    Function.new!(%{
      name: "read_file",
      description: "Read a file",
      parameters_schema: %{
        type: "object",
        properties: %{
          path: %{type: "string", description: "File path"}
        },
        required: ["path"]
      },
      function: fn args, _context ->
        {:ok, "File content from: #{args["path"]}"}
      end
    })
  end

  describe "Agent integration with HumanInTheLoop" do
    test "agent includes HITL middleware when interrupt_on is configured" do
      {:ok, agent} =
        Agent.new(
          model: create_test_model(),
          tools: [create_write_file_tool()],
          interrupt_on: %{
            "write_file" => true
          }
        )

      # Check that HITL middleware is in the stack
      assert Enum.any?(agent.middleware, fn {module, _config} ->
               module == LangChain.Agents.Middleware.HumanInTheLoop
             end)
    end

    test "agent excludes HITL middleware when interrupt_on is nil" do
      {:ok, agent} =
        Agent.new(
          model: create_test_model(),
          tools: [create_write_file_tool()]
          # No interrupt_on configured
        )

      # Check that HITL middleware is NOT in the stack
      refute Enum.any?(agent.middleware, fn {module, _config} ->
               module == LangChain.Agents.Middleware.HumanInTheLoop
             end)
    end

    @tag :live_call
    test "agent handles interrupt return from execute" do
      {:ok, agent} =
        Agent.new(
          model: create_test_model(),
          tools: [create_write_file_tool()],
          interrupt_on: %{
            "write_file" => %{allowed_decisions: [:approve, :reject]}
          }
        )

      # Create state with assistant message containing tool call
      tool_call =
        LangChain.Message.ToolCall.new!(%{
          call_id: "test_123",
          name: "write_file",
          arguments: %{"path" => "test.txt", "content" => "Hello World"}
        })

      messages = [
        Message.new_user!("Write a test file"),
        Message.new_assistant!(%{
          content: "I'll write the file now.",
          tool_calls: [tool_call]
        })
      ]

      state = State.new!(%{messages: messages})

      # Execute should return an interrupt
      # Note: This bypasses the LLM call by providing pre-built messages
      # In a real scenario, execute would call the model first
      result = Agent.execute(agent, state)

      case result do
        {:interrupt, interrupted_state, interrupt_data} ->
          # Verify interrupt data structure
          assert %{action_requests: [action], review_configs: configs} = interrupt_data

          assert action == %{
                   tool_call_id: "test_123",
                   tool_name: "write_file",
                   arguments: %{"path" => "test.txt", "content" => "Hello World"}
                 }

          assert configs["write_file"] == %{allowed_decisions: [:approve, :reject]}

          # Verify state is preserved
          assert interrupted_state.messages == messages

        {:ok, _state} ->
          # If we get OK, it means no interrupt occurred (possibly due to other middleware)
          flunk("Expected interrupt but got ok")

        {:error, reason} ->
          flunk("Unexpected error: #{inspect(reason)}")
      end
    end

    test "agent resume processes decisions correctly" do
      {:ok, agent} =
        Agent.new(
          model: create_test_model(),
          tools: [create_write_file_tool()],
          interrupt_on: %{
            "write_file" => true
          }
        )

      # Create interrupted state
      tool_call =
        LangChain.Message.ToolCall.new!(%{
          call_id: "call_456",
          name: "write_file",
          arguments: %{"path" => "output.txt", "content" => "Data"}
        })

      messages = [
        Message.new_user!("Save the data"),
        Message.new_assistant!(%{tool_calls: [tool_call]})
      ]

      interrupted_state = State.new!(%{messages: messages})

      # Resume with approval decision
      decisions = [%{type: :approve}]

      case Agent.resume(agent, interrupted_state, decisions) do
        {:ok, resumed_state} ->
          # Should have tool result message added
          assert length(resumed_state.messages) == 3
          assert [_user, _assistant, tool_msg] = resumed_state.messages
          assert tool_msg.role == :tool
          assert [result] = tool_msg.tool_results
          assert result.tool_call_id == "call_456"
          assert result.name == "write_file"

          # Content is stored as ContentParts
          assert [%Message.ContentPart{content: content}] = result.content
          assert content =~ "approved for execution"

        {:error, reason} ->
          flunk("Resume failed: #{inspect(reason)}")
      end
    end

    test "agent resume handles edit decision" do
      {:ok, agent} =
        Agent.new(
          model: create_test_model(),
          tools: [create_write_file_tool()],
          interrupt_on: %{
            "write_file" => %{allowed_decisions: [:approve, :edit, :reject]}
          }
        )

      tool_call =
        LangChain.Message.ToolCall.new!(%{
          call_id: "call_789",
          name: "write_file",
          arguments: %{"path" => "draft.txt", "content" => "Original"}
        })

      messages = [
        Message.new_user!("Create draft"),
        Message.new_assistant!(%{tool_calls: [tool_call]})
      ]

      interrupted_state = State.new!(%{messages: messages})

      # Resume with edit decision
      decisions = [
        %{
          type: :edit,
          arguments: %{"path" => "final.txt", "content" => "Edited"}
        }
      ]

      assert {:ok, resumed_state} = Agent.resume(agent, interrupted_state, decisions)

      assert [_user, _assistant, tool_msg] = resumed_state.messages
      [result] = tool_msg.tool_results

      assert result.tool_call_id == "call_789"

      # Content is stored as ContentParts
      assert [%Message.ContentPart{content: content}] = result.content
      assert content =~ "edited arguments"
      assert content =~ "final.txt"
      assert content =~ "Edited"
    end

    test "agent resume handles reject decision" do
      {:ok, agent} =
        Agent.new(
          model: create_test_model(),
          tools: [create_write_file_tool()],
          interrupt_on: %{
            "write_file" => true
          }
        )

      tool_call =
        LangChain.Message.ToolCall.new!(%{
          call_id: "call_reject",
          name: "write_file",
          arguments: %{"path" => "sensitive.txt"}
        })

      messages = [
        Message.new_user!("Write sensitive file"),
        Message.new_assistant!(%{tool_calls: [tool_call]})
      ]

      interrupted_state = State.new!(%{messages: messages})

      # Resume with reject decision
      decisions = [%{type: :reject}]

      assert {:ok, resumed_state} = Agent.resume(agent, interrupted_state, decisions)

      assert [_user, _assistant, tool_msg] = resumed_state.messages
      [result] = tool_msg.tool_results

      assert result.tool_call_id == "call_reject"

      # Content is stored as ContentParts
      assert [%Message.ContentPart{content: content}] = result.content
      assert content =~ "rejected by human reviewer"
    end

    test "agent resume returns error for invalid decisions" do
      {:ok, agent} =
        Agent.new(
          model: create_test_model(),
          tools: [create_write_file_tool()],
          interrupt_on: %{
            "write_file" => %{allowed_decisions: [:approve, :reject]}
          }
        )

      tool_call =
        LangChain.Message.ToolCall.new!(%{
          call_id: "call_invalid",
          name: "write_file",
          arguments: %{"path" => "test.txt"}
        })

      messages = [
        Message.new_user!("Write file"),
        Message.new_assistant!(%{tool_calls: [tool_call]})
      ]

      interrupted_state = State.new!(%{messages: messages})

      # Try to edit when only approve/reject allowed
      decisions = [%{type: :edit, arguments: %{"path" => "new.txt"}}]

      assert {:error, reason} = Agent.resume(agent, interrupted_state, decisions)
      assert reason =~ "not allowed"
    end

    test "agent resume returns error when HITL not configured" do
      {:ok, agent} =
        Agent.new(
          model: create_test_model(),
          tools: [create_write_file_tool()]
          # No interrupt_on
        )

      messages = [Message.new_user!("Hello")]
      state = State.new!(%{messages: messages})
      decisions = [%{type: :approve}]

      assert {:error, reason} = Agent.resume(agent, state, decisions)
      assert reason =~ "does not have HumanInTheLoop middleware"
    end

    @tag :live_call
    test "selective interruption - only configured tools interrupt" do
      {:ok, agent} =
        Agent.new(
          model: create_test_model(),
          tools: [create_write_file_tool(), create_read_file_tool()],
          interrupt_on: %{
            "write_file" => true
            # read_file not configured - should not interrupt
          }
        )

      write_call =
        LangChain.Message.ToolCall.new!(%{
          call_id: "write_1",
          name: "write_file",
          arguments: %{"path" => "out.txt"}
        })

      read_call =
        LangChain.Message.ToolCall.new!(%{
          call_id: "read_1",
          name: "read_file",
          arguments: %{"path" => "in.txt"}
        })

      messages = [
        Message.new_user!("Process files"),
        Message.new_assistant!(%{tool_calls: [read_call, write_call]})
      ]

      state = State.new!(%{messages: messages})

      result = Agent.execute(agent, state)

      case result do
        {:interrupt, _state, interrupt_data} ->
          # Should only interrupt for write_file
          assert %{action_requests: [action]} = interrupt_data
          assert action.tool_name == "write_file"

        _ ->
          flunk("Expected interrupt")
      end
    end

    test "multiple tool calls with mixed decisions" do
      {:ok, agent} =
        Agent.new(
          model: create_test_model(),
          tools: [create_write_file_tool()],
          interrupt_on: %{
            "write_file" => true
          }
        )

      tool_call1 =
        LangChain.Message.ToolCall.new!(%{
          call_id: "call_1",
          name: "write_file",
          arguments: %{"path" => "file1.txt", "content" => "Data 1"}
        })

      tool_call2 =
        LangChain.Message.ToolCall.new!(%{
          call_id: "call_2",
          name: "write_file",
          arguments: %{"path" => "file2.txt", "content" => "Data 2"}
        })

      messages = [
        Message.new_user!("Write both files"),
        Message.new_assistant!(%{tool_calls: [tool_call1, tool_call2]})
      ]

      interrupted_state = State.new!(%{messages: messages})

      # Approve first, edit second
      decisions = [
        %{type: :approve},
        %{type: :edit, arguments: %{"path" => "modified.txt", "content" => "Modified"}}
      ]

      assert {:ok, resumed_state} = Agent.resume(agent, interrupted_state, decisions)

      assert [_user, _assistant, tool_msg] = resumed_state.messages
      assert length(tool_msg.tool_results) == 2

      [result1, result2] = tool_msg.tool_results

      assert result1.tool_call_id == "call_1"

      # Content is stored as ContentParts
      assert [%Message.ContentPart{content: content1}] = result1.content
      assert content1 =~ "approved"

      assert result2.tool_call_id == "call_2"

      assert [%Message.ContentPart{content: content2}] = result2.content
      assert content2 =~ "edited"
      assert content2 =~ "modified.txt"
    end
  end

  describe "configuration validation" do
    test "accepts valid interrupt_on map" do
      assert {:ok, _agent} =
               Agent.new(
                 model: create_test_model(),
                 tools: [create_write_file_tool()],
                 interrupt_on: %{
                   "write_file" => true,
                   "delete_file" => %{allowed_decisions: [:approve, :reject]}
                 }
               )
    end

    test "handles empty interrupt_on map" do
      assert {:ok, agent} =
               Agent.new(
                 model: create_test_model(),
                 tools: [create_write_file_tool()],
                 interrupt_on: %{}
               )

      # HITL middleware should still be added even with empty config
      assert Enum.any?(agent.middleware, fn {module, _config} ->
               module == LangChain.Agents.Middleware.HumanInTheLoop
             end)
    end

    test "handles nil interrupt_on by not adding middleware" do
      assert {:ok, agent} =
               Agent.new(
                 model: create_test_model(),
                 tools: [create_write_file_tool()],
                 interrupt_on: nil
               )

      refute Enum.any?(agent.middleware, fn {module, _config} ->
               module == LangChain.Agents.Middleware.HumanInTheLoop
             end)
    end
  end
end
