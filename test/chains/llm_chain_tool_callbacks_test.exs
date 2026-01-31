defmodule LangChain.Chains.LLMChainToolCallbacksTest do
  use ExUnit.Case

  alias LangChain.Chains.LLMChain
  alias LangChain.{Function, Message}
  alias LangChain.Message.ToolCall
  alias LangChain.ChatModels.ChatAnthropic

  describe "tool execution callbacks" do
    test "fires on_tool_execution_started callback" do
      test_pid = self()

      # Create a tool with display_text
      tool =
        Function.new!(%{
          name: "test_tool",
          description: "A test tool",
          display_text: "Testing something",
          parameters_schema: %{type: "object", properties: %{}},
          function: fn _args, _ctx -> {:ok, "result"} end
        })

      # Create callback handler
      callbacks = %{
        on_tool_execution_started: fn _chain, tool_call, function ->
          send(test_pid, {:started, tool_call.name, function.display_text})
        end
      }

      # Create chain with tool and callback
      chain =
        LLMChain.new!(%{
          llm: ChatAnthropic.new!(%{model: "claude-sonnet-4-5-20250929"}),
          tools: [tool],
          callbacks: [callbacks]
        })

      # Simulate a message with tool call
      message =
        Message.new_assistant!(%{
          content: "Let me test",
          tool_calls: [
            ToolCall.new!(%{call_id: "call_1", name: "test_tool", arguments: %{}})
          ]
        })

      chain = LLMChain.add_message(chain, message)

      # Execute tools
      _updated_chain = LLMChain.execute_tool_calls(chain)

      # Verify callback was fired
      assert_received {:started, "test_tool", "Testing something"}
    end

    test "fires on_tool_execution_completed callback on success" do
      test_pid = self()

      # Create a successful tool
      tool =
        Function.new!(%{
          name: "success_tool",
          description: "Always succeeds",
          display_text: "Running success tool",
          parameters_schema: %{type: "object", properties: %{}},
          function: fn _args, _ctx -> {:ok, "success!"} end
        })

      callbacks = %{
        on_tool_execution_completed: fn _chain, tool_call, tool_result ->
          send(test_pid, {:completed, tool_call.name, tool_result})
        end
      }

      chain =
        LLMChain.new!(%{
          llm: ChatAnthropic.new!(%{model: "claude-sonnet-4-5-20250929"}),
          tools: [tool],
          callbacks: [callbacks]
        })

      message =
        Message.new_assistant!(%{
          content: "Test",
          tool_calls: [
            ToolCall.new!(%{call_id: "call_2", name: "success_tool", arguments: %{}})
          ]
        })

      chain = LLMChain.add_message(chain, message)
      _updated_chain = LLMChain.execute_tool_calls(chain)

      assert_received {:completed, "success_tool", tool_result}
      assert tool_result.is_error == false
    end

    test "fires on_tool_execution_failed callback when tool raises error" do
      test_pid = self()

      # Create a failing tool
      tool =
        Function.new!(%{
          name: "fail_tool",
          description: "Always fails",
          display_text: "Running fail tool",
          parameters_schema: %{type: "object", properties: %{}},
          function: fn _args, _ctx -> {:error, "Something went wrong"} end
        })

      callbacks = %{
        on_tool_execution_failed: fn _chain, tool_call, error ->
          send(test_pid, {:failed, tool_call.name, error})
        end,
        on_tool_execution_completed: fn _chain, tool_call, _tool_result ->
          send(test_pid, {:completed, tool_call.name})
        end
      }

      chain =
        LLMChain.new!(%{
          llm: ChatAnthropic.new!(%{model: "claude-sonnet-4-5-20250929"}),
          tools: [tool],
          callbacks: [callbacks]
        })

      message =
        Message.new_assistant!(%{
          content: "Test",
          tool_calls: [
            ToolCall.new!(%{call_id: "call_3", name: "fail_tool", arguments: %{}})
          ]
        })

      chain = LLMChain.add_message(chain, message)
      _updated_chain = LLMChain.execute_tool_calls(chain)

      # Error can be a string or list of ContentParts
      assert_received {:failed, "fail_tool", _error}

      # Ensure on_tool_execution_completed did NOT fire
      refute_received {:completed, "fail_tool"}
    end

    test "fires on_tool_execution_failed callback for invalid tool" do
      test_pid = self()

      callbacks = %{
        on_tool_execution_failed: fn _chain, tool_call, error ->
          send(test_pid, {:failed, tool_call.name, error})
        end
      }

      chain =
        LLMChain.new!(%{
          llm: ChatAnthropic.new!(%{model: "claude-sonnet-4-5-20250929"}),
          tools: [],
          callbacks: [callbacks]
        })

      message =
        Message.new_assistant!(%{
          content: "Test",
          tool_calls: [
            ToolCall.new!(%{call_id: "call_4", name: "nonexistent_tool", arguments: %{}})
          ]
        })

      chain = LLMChain.add_message(chain, message)
      _updated_chain = LLMChain.execute_tool_calls(chain)

      assert_received {:failed, "nonexistent_tool", _error}
    end

    test "fires callbacks for multiple tools in correct order" do
      test_pid = self()

      tool1 =
        Function.new!(%{
          name: "tool1",
          description: "First tool",
          display_text: "Tool 1",
          parameters_schema: %{type: "object", properties: %{}},
          function: fn _args, _ctx -> {:ok, "result1"} end
        })

      tool2 =
        Function.new!(%{
          name: "tool2",
          description: "Second tool",
          display_text: "Tool 2",
          parameters_schema: %{type: "object", properties: %{}},
          function: fn _args, _ctx -> {:ok, "result2"} end
        })

      callbacks = %{
        on_tool_execution_started: fn _chain, tool_call, _function ->
          send(test_pid, {:started, tool_call.name})
        end,
        on_tool_execution_completed: fn _chain, tool_call, _tool_result ->
          send(test_pid, {:completed, tool_call.name})
        end
      }

      chain =
        LLMChain.new!(%{
          llm: ChatAnthropic.new!(%{model: "claude-sonnet-4-5-20250929"}),
          tools: [tool1, tool2],
          callbacks: [callbacks]
        })

      message =
        Message.new_assistant!(%{
          content: "Test",
          tool_calls: [
            ToolCall.new!(%{call_id: "call_5", name: "tool1", arguments: %{}}),
            ToolCall.new!(%{call_id: "call_6", name: "tool2", arguments: %{}})
          ]
        })

      chain = LLMChain.add_message(chain, message)
      _updated_chain = LLMChain.execute_tool_calls(chain)

      # All started callbacks should fire before completed callbacks
      assert_received {:started, "tool1"}
      assert_received {:started, "tool2"}
      assert_received {:completed, "tool1"}
      assert_received {:completed, "tool2"}
    end

    test "fires callbacks in execute_tool_calls_with_decisions for approved tools" do
      test_pid = self()

      tool =
        Function.new!(%{
          name: "test_tool",
          description: "A test tool",
          display_text: "Testing with decisions",
          parameters_schema: %{type: "object", properties: %{}},
          function: fn _args, _ctx -> {:ok, "result"} end
        })

      callbacks = %{
        on_tool_execution_started: fn _chain, tool_call, function ->
          send(test_pid, {:started, tool_call.name, function.display_text})
        end,
        on_tool_execution_completed: fn _chain, tool_call, _tool_result ->
          send(test_pid, {:completed, tool_call.name})
        end
      }

      chain =
        LLMChain.new!(%{
          llm: ChatAnthropic.new!(%{model: "claude-sonnet-4-5-20250929"}),
          tools: [tool],
          callbacks: [callbacks]
        })

      tool_calls = [
        ToolCall.new!(%{call_id: "call_7", name: "test_tool", arguments: %{}})
      ]

      decisions = [%{type: :approve}]

      _updated_chain = LLMChain.execute_tool_calls_with_decisions(chain, tool_calls, decisions)

      assert_received {:started, "test_tool", "Testing with decisions"}
      assert_received {:completed, "test_tool"}
    end

    test "fires on_tool_execution_failed callback for rejected tools in HITL" do
      test_pid = self()

      tool =
        Function.new!(%{
          name: "test_tool",
          description: "A test tool",
          display_text: "Testing rejection",
          parameters_schema: %{type: "object", properties: %{}},
          function: fn _args, _ctx -> {:ok, "should not execute"} end
        })

      callbacks = %{
        on_tool_execution_failed: fn _chain, tool_call, error ->
          send(test_pid, {:failed, tool_call.name, error})
        end
      }

      chain =
        LLMChain.new!(%{
          llm: ChatAnthropic.new!(%{model: "claude-sonnet-4-5-20250929"}),
          tools: [tool],
          callbacks: [callbacks]
        })

      tool_calls = [
        ToolCall.new!(%{call_id: "call_8", name: "test_tool", arguments: %{}})
      ]

      decisions = [%{type: :reject}]

      _updated_chain = LLMChain.execute_tool_calls_with_decisions(chain, tool_calls, decisions)

      assert_received {:failed, "test_tool", error}
      assert error =~ "rejected by a human reviewer"
    end

    test "fires callbacks for edited tool calls in HITL" do
      test_pid = self()

      tool =
        Function.new!(%{
          name: "test_tool",
          description: "A test tool",
          display_text: "Testing editing",
          parameters_schema: %{
            type: "object",
            properties: %{value: %{type: "string"}}
          },
          function: fn args, _ctx -> {:ok, "received: #{args["value"]}"} end
        })

      callbacks = %{
        on_tool_execution_started: fn _chain, tool_call, _function ->
          send(test_pid, {:started, tool_call.arguments})
        end,
        on_tool_execution_completed: fn _chain, _tool_call, tool_result ->
          send(test_pid, {:completed, tool_result})
        end
      }

      chain =
        LLMChain.new!(%{
          llm: ChatAnthropic.new!(%{model: "claude-sonnet-4-5-20250929"}),
          tools: [tool],
          callbacks: [callbacks]
        })

      tool_calls = [
        ToolCall.new!(%{
          call_id: "call_9",
          name: "test_tool",
          arguments: %{"value" => "original"}
        })
      ]

      decisions = [%{type: :edit, arguments: %{"value" => "edited"}}]

      _updated_chain = LLMChain.execute_tool_calls_with_decisions(chain, tool_calls, decisions)

      # Should receive the edited arguments
      assert_received {:started, %{"value" => "edited"}}
      assert_received {:completed, _tool_result}
    end

    test "async tools fire callbacks after all complete" do
      test_pid = self()

      async_tool =
        Function.new!(%{
          name: "async_tool",
          description: "An async tool",
          display_text: "Async processing",
          async: true,
          parameters_schema: %{type: "object", properties: %{}},
          function: fn _args, _ctx ->
            # Simulate async work
            Process.sleep(10)
            {:ok, "async result"}
          end
        })

      callbacks = %{
        on_tool_execution_started: fn _chain, tool_call, _function ->
          send(test_pid, {:started, tool_call.name, System.monotonic_time(:millisecond)})
        end,
        on_tool_execution_completed: fn _chain, tool_call, _tool_result ->
          send(test_pid, {:completed, tool_call.name, System.monotonic_time(:millisecond)})
        end
      }

      chain =
        LLMChain.new!(%{
          llm: ChatAnthropic.new!(%{model: "claude-sonnet-4-5-20250929"}),
          tools: [async_tool],
          callbacks: [callbacks]
        })

      message =
        Message.new_assistant!(%{
          content: "Test",
          tool_calls: [
            ToolCall.new!(%{call_id: "call_10", name: "async_tool", arguments: %{}})
          ]
        })

      chain = LLMChain.add_message(chain, message)
      _updated_chain = LLMChain.execute_tool_calls(chain)

      assert_received {:started, "async_tool", _start_time}
      assert_received {:completed, "async_tool", _end_time}
    end
  end

  describe "tool identification callbacks during streaming" do
    test "fires on_tool_call_identified when tool name detected in streaming delta" do
      test_pid = self()

      tool =
        Function.new!(%{
          name: "search_web",
          display_text: "Searching web",
          description: "Search the web",
          parameters_schema: %{type: "object", properties: %{}},
          function: fn _args, _ctx -> {:ok, "results"} end
        })

      callbacks = %{
        on_tool_call_identified: fn _chain, tool_call, function ->
          send(test_pid, {:identified, tool_call.name, function.display_text})
        end
      }

      chain =
        LLMChain.new!(%{
          llm: ChatAnthropic.new!(%{model: "claude-sonnet-4-5-20250929"}),
          tools: [tool],
          callbacks: [callbacks]
        })

      # Simulate streaming: first delta has tool name but no call_id
      delta =
        LangChain.MessageDelta.new!(%{
          role: :assistant,
          tool_calls: [
            LangChain.Message.ToolCall.new!(%{index: 0, name: "search_web"})
          ]
        })

      _updated_chain = LLMChain.merge_delta(chain, delta)

      assert_received {:identified, "search_web", "Searching web"}
    end

    test "does not fire on_tool_call_identified twice for same tool" do
      test_pid = self()

      tool =
        Function.new!(%{
          name: "file_read",
          display_text: "Reading file",
          description: "Read a file",
          parameters_schema: %{type: "object", properties: %{}},
          function: fn _args, _ctx -> {:ok, "content"} end
        })

      callbacks = %{
        on_tool_call_identified: fn _chain, tool_call, _function ->
          send(test_pid, {:identified, tool_call.name})
        end
      }

      chain =
        LLMChain.new!(%{
          llm: ChatAnthropic.new!(%{model: "claude-sonnet-4-5-20250929"}),
          tools: [tool],
          callbacks: [callbacks]
        })

      # First delta with tool name
      delta1 =
        LangChain.MessageDelta.new!(%{
          role: :assistant,
          tool_calls: [
            LangChain.Message.ToolCall.new!(%{index: 0, name: "file_read"})
          ]
        })

      chain = LLMChain.merge_delta(chain, delta1)

      # Second delta with same tool (adding arguments)
      delta2 =
        LangChain.MessageDelta.new!(%{
          role: :assistant,
          tool_calls: [
            LangChain.Message.ToolCall.new!(%{
              index: 0,
              name: "file_read",
              arguments: "{\"path\":"
            })
          ]
        })

      _updated_chain = LLMChain.merge_delta(chain, delta2)

      # Should only receive one identification
      assert_received {:identified, "file_read"}
      refute_received {:identified, "file_read"}
    end

    test "fires both on_tool_call_identified and on_tool_execution_started in sequence" do
      test_pid = self()

      tool =
        Function.new!(%{
          name: "calculator",
          display_text: "Calculating",
          description: "Do math",
          parameters_schema: %{type: "object", properties: %{}},
          function: fn _args, _ctx -> {:ok, "42"} end
        })

      callbacks = %{
        on_tool_call_identified: fn _chain, tool_call, function ->
          send(test_pid, {:identified, tool_call.name, function.display_text})
        end,
        on_tool_execution_started: fn _chain, tool_call, function ->
          send(test_pid, {:started, tool_call.name, function.display_text})
        end
      }

      chain =
        LLMChain.new!(%{
          llm: ChatAnthropic.new!(%{model: "claude-sonnet-4-5-20250929"}),
          tools: [tool],
          callbacks: [callbacks]
        })

      # Simulate streaming delta
      delta =
        LangChain.MessageDelta.new!(%{
          role: :assistant,
          tool_calls: [
            LangChain.Message.ToolCall.new!(%{index: 0, name: "calculator"})
          ]
        })

      chain = LLMChain.merge_delta(chain, delta)

      # Should receive identification callback
      assert_received {:identified, "calculator", "Calculating"}

      # Now simulate execution
      message =
        Message.new_assistant!(%{
          content: "Let me calculate",
          tool_calls: [
            LangChain.Message.ToolCall.new!(%{
              call_id: "call_123",
              name: "calculator",
              arguments: %{}
            })
          ]
        })

      chain = LLMChain.add_message(chain, message)
      _updated_chain = LLMChain.execute_tool_calls(chain)

      # Should also receive execution started callback
      assert_received {:started, "calculator", "Calculating"}
    end

    test "handles tool not found in tool map during identification" do
      test_pid = self()

      callbacks = %{
        on_tool_call_identified: fn _chain, tool_call, _function ->
          send(test_pid, {:identified, tool_call.name})
        end
      }

      chain =
        LLMChain.new!(%{
          llm: ChatAnthropic.new!(%{model: "claude-sonnet-4-5-20250929"}),
          tools: [],
          callbacks: [callbacks]
        })

      # Delta with unknown tool
      delta =
        LangChain.MessageDelta.new!(%{
          role: :assistant,
          tool_calls: [
            LangChain.Message.ToolCall.new!(%{index: 0, name: "unknown_tool"})
          ]
        })

      _updated_chain = LLMChain.merge_delta(chain, delta)

      # Should not fire callback for unknown tool
      refute_received {:identified, _}
    end
  end
end
