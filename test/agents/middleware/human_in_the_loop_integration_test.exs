defmodule LangChain.Agents.Middleware.HumanInTheLoopIntegrationTest do
  use ExUnit.Case, async: true
  use Mimic

  alias LangChain.Agents.Agent
  alias LangChain.Agents.State
  alias LangChain.Message
  alias LangChain.Message.ToolCall
  alias LangChain.Function
  alias LangChain.ChatModels.ChatAnthropic

  setup :verify_on_exit!

  defp create_test_model do
    ChatAnthropic.new!(%{
      model: "claude-3-5-sonnet-20241022",
      temperature: 0,
      stream: false
    })
  end

  defp mock_llm_with_tool_call(tool_name, tool_args) do
    # Mock the LLM to return a message with a tool call on first call,
    # then a final message without tool calls on subsequent calls
    tool_call =
      ToolCall.new!(%{
        call_id: "test_call_#{:rand.uniform(10000)}",
        name: tool_name,
        arguments: tool_args
      })

    response_with_tool =
      Message.new_assistant!(%{
        content: "I'll use the tool.",
        tool_calls: [tool_call]
      })

    final_response =
      Message.new_assistant!(%{
        content: "Done! I've completed the task."
      })

    ChatAnthropic
    |> stub(:call, fn _model, messages, _tools ->
      # Check if there's a tool result message in the conversation
      # If yes, this is the second call (after tool execution), return final message
      # If no, this is the first call, return tool call
      has_tool_result =
        Enum.any?(messages, fn msg ->
          case msg do
            %{role: :tool} -> true
            _ -> false
          end
        end)

      if has_tool_result do
        {:ok, [final_response]}
      else
        {:ok, [response_with_tool]}
      end
    end)

    tool_call
  end

  defp mock_llm_with_multiple_tool_calls(tool_specs) do
    # Mock the LLM to return multiple tool calls on first call,
    # then a final message without tool calls on subsequent calls
    tool_calls =
      Enum.map(tool_specs, fn {tool_name, tool_args} ->
        ToolCall.new!(%{
          call_id: "test_call_#{:rand.uniform(10000)}",
          name: tool_name,
          arguments: tool_args
        })
      end)

    response_with_tools =
      Message.new_assistant!(%{
        content: "I'll use multiple tools.",
        tool_calls: tool_calls
      })

    final_response =
      Message.new_assistant!(%{
        content: "Done! I've completed all the tasks."
      })

    ChatAnthropic
    |> stub(:call, fn _model, messages, _tools ->
      # Check if there's a tool result message in the conversation
      # If yes, this is the second call (after tool execution), return final message
      # If no, this is the first call, return tool calls
      has_tool_result =
        Enum.any?(messages, fn msg ->
          case msg do
            %{role: :tool} -> true
            _ -> false
          end
        end)

      if has_tool_result do
        {:ok, [final_response]}
      else
        {:ok, [response_with_tools]}
      end
    end)

    tool_calls
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

    test "agent execute returns interrupt, resume processes decisions and executes tools" do
      # Mock LLM to return a tool call
      tool_call =
        mock_llm_with_tool_call("write_file", %{
          "path" => "output.txt",
          "content" => "Data"
        })

      {:ok, agent} =
        Agent.new(
          model: create_test_model(),
          tools: [create_write_file_tool()],
          interrupt_on: %{
            "write_file" => true
          },
          replace_default_middleware: true,
          middleware: [
            {LangChain.Agents.Middleware.HumanInTheLoop, [interrupt_on: %{"write_file" => true}]}
          ]
        )

      # Create initial state with user message
      initial_state = State.new!(%{messages: [Message.new_user!("Save the data")]})

      # Execute should return interrupt WITHOUT executing tools
      result = Agent.execute(agent, initial_state)

      assert {:interrupt, interrupted_state, interrupt_data} = result

      # Verify interrupt data
      assert %{action_requests: [action], review_configs: _configs} = interrupt_data
      assert action.tool_name == "write_file"
      assert action.arguments == %{"path" => "output.txt", "content" => "Data"}

      # Tool should NOT have been executed yet
      # (we can't easily verify this without tracking, but the interrupt proves it)

      # Now resume with approval decision
      decisions = [%{type: :approve}]

      case Agent.resume(agent, interrupted_state, decisions) do
        {:ok, resumed_state} ->
          # Should have tool result message AND final assistant message
          # User + Assistant (tool call) + Tool result + Assistant (final) = 4
          assert length(resumed_state.messages) == 4

          assert [_user, _assistant_with_tools, tool_msg, final_assistant] =
                   resumed_state.messages

          # Check tool result message
          assert tool_msg.role == :tool
          assert [result] = tool_msg.tool_results
          assert result.tool_call_id == tool_call.call_id
          assert result.name == "write_file"

          # Content is the actual tool output (not placeholder)
          assert [%Message.ContentPart{content: content}] = result.content
          assert content =~ "File written: output.txt"

          # Check final assistant message (no tool calls)
          assert final_assistant.role == :assistant
          assert final_assistant.tool_calls == [] || final_assistant.tool_calls == nil
          assert [%Message.ContentPart{content: final_content}] = final_assistant.content
          assert final_content =~ "Done"

        {:error, reason} ->
          flunk("Resume failed: #{inspect(reason)}")
      end
    end

    test "agent resume handles edit decision" do
      # Mock LLM to return a tool call
      tool_call =
        mock_llm_with_tool_call("write_file", %{
          "path" => "draft.txt",
          "content" => "Original"
        })

      {:ok, agent} =
        Agent.new(
          model: create_test_model(),
          tools: [create_write_file_tool()],
          interrupt_on: %{
            "write_file" => %{allowed_decisions: [:approve, :edit, :reject]}
          },
          replace_default_middleware: true,
          middleware: [
            {LangChain.Agents.Middleware.HumanInTheLoop,
             [interrupt_on: %{"write_file" => %{allowed_decisions: [:approve, :edit, :reject]}}]}
          ]
        )

      initial_state = State.new!(%{messages: [Message.new_user!("Create draft")]})

      # Execute should return interrupt
      {:interrupt, interrupted_state, _interrupt_data} = Agent.execute(agent, initial_state)

      # Resume with edit decision
      decisions = [
        %{
          type: :edit,
          arguments: %{"path" => "final.txt", "content" => "Edited"}
        }
      ]

      assert {:ok, resumed_state} = Agent.resume(agent, interrupted_state, decisions)

      # Should have tool result + final assistant message
      assert length(resumed_state.messages) == 4
      assert [_user, _assistant, tool_msg, _final] = resumed_state.messages
      [result] = tool_msg.tool_results

      assert result.tool_call_id == tool_call.call_id

      # Content is the actual tool output with EDITED arguments
      assert [%Message.ContentPart{content: content}] = result.content
      assert content =~ "File written: final.txt"
    end

    test "agent resume handles reject decision" do
      # Mock LLM to return a tool call
      tool_call =
        mock_llm_with_tool_call("write_file", %{
          "path" => "sensitive.txt",
          "content" => "Secret"
        })

      {:ok, agent} =
        Agent.new(
          model: create_test_model(),
          tools: [create_write_file_tool()],
          interrupt_on: %{
            "write_file" => true
          },
          replace_default_middleware: true,
          middleware: [
            {LangChain.Agents.Middleware.HumanInTheLoop, [interrupt_on: %{"write_file" => true}]}
          ]
        )

      initial_state = State.new!(%{messages: [Message.new_user!("Write sensitive file")]})

      # Execute should return interrupt
      {:interrupt, interrupted_state, _interrupt_data} = Agent.execute(agent, initial_state)

      # Resume with reject decision
      decisions = [%{type: :reject}]

      assert {:ok, resumed_state} = Agent.resume(agent, interrupted_state, decisions)

      # Should have tool result + final assistant message
      assert length(resumed_state.messages) == 4
      assert [_user, _assistant, tool_msg, _final] = resumed_state.messages
      [result] = tool_msg.tool_results

      assert result.tool_call_id == tool_call.call_id

      # Content shows rejection (tool was NOT executed)
      assert [%Message.ContentPart{content: content}] = result.content
      assert content =~ "rejected by a human reviewer"
    end

    test "agent resume returns error for invalid decisions" do
      # Mock LLM to return a tool call
      mock_llm_with_tool_call("write_file", %{"path" => "test.txt", "content" => "data"})

      {:ok, agent} =
        Agent.new(
          model: create_test_model(),
          tools: [create_write_file_tool()],
          interrupt_on: %{
            "write_file" => %{allowed_decisions: [:approve, :reject]}
          }
        )

      # Execute to get interrupt
      initial_state = State.new!(%{messages: [Message.new_user!("Write file")]})

      assert {:interrupt, interrupted_state, _interrupt_data} =
               Agent.execute(agent, initial_state)

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
      # Mock LLM to return multiple tool calls
      [tool_call1, tool_call2] =
        mock_llm_with_multiple_tool_calls([
          {"write_file", %{"path" => "file1.txt", "content" => "Data 1"}},
          {"write_file", %{"path" => "file2.txt", "content" => "Data 2"}}
        ])

      {:ok, agent} =
        Agent.new(
          model: create_test_model(),
          tools: [create_write_file_tool()],
          interrupt_on: %{
            "write_file" => true
          },
          replace_default_middleware: true,
          middleware: [
            {LangChain.Agents.Middleware.HumanInTheLoop, [interrupt_on: %{"write_file" => true}]}
          ]
        )

      initial_state = State.new!(%{messages: [Message.new_user!("Write both files")]})

      # Execute should return interrupt
      {:interrupt, interrupted_state, _interrupt_data} = Agent.execute(agent, initial_state)

      # Approve first, edit second
      decisions = [
        %{type: :approve},
        %{type: :edit, arguments: %{"path" => "modified.txt", "content" => "Modified"}}
      ]

      assert {:ok, resumed_state} = Agent.resume(agent, interrupted_state, decisions)

      # Should have tool result + final assistant message
      assert length(resumed_state.messages) == 4
      assert [_user, _assistant, tool_msg, _final] = resumed_state.messages
      assert length(tool_msg.tool_results) == 2

      [result1, result2] = tool_msg.tool_results

      assert result1.tool_call_id == tool_call1.call_id

      # First tool executed with original arguments
      assert [%Message.ContentPart{content: content1}] = result1.content
      assert content1 =~ "File written: file1.txt"

      assert result2.tool_call_id == tool_call2.call_id

      # Second tool executed with EDITED arguments
      assert [%Message.ContentPart{content: content2}] = result2.content
      assert content2 =~ "File written: modified.txt"
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
