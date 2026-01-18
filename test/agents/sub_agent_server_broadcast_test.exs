defmodule LangChain.Agents.SubAgentServerBroadcastTest do
  @moduledoc """
  Tests for SubAgentServer event broadcasting via parent AgentServer.

  These tests verify that SubAgentServer broadcasts lifecycle events
  (started, status_changed, completed, error) through the parent AgentServer's
  debug topic.
  """
  use LangChain.BaseCase, async: false
  use Mimic

  alias LangChain.Agents.{AgentServer, SubAgent, SubAgentServer, State}
  alias LangChain.ChatModels.ChatAnthropic
  alias LangChain.Function
  alias LangChain.Message
  alias LangChain.Message.ContentPart
  alias LangChain.Message.ToolCall

  setup :set_mimic_global
  setup :verify_on_exit!

  setup do
    # Start PubSub for testing
    pubsub_name = :"test_pubsub_#{System.unique_integer([:positive])}"
    debug_pubsub_name = :"test_debug_pubsub_#{System.unique_integer([:positive])}"

    {:ok, _} =
      Phoenix.PubSub.Supervisor.start_link(name: pubsub_name, adapter_name: Phoenix.PubSub.PG2)

    {:ok, _} =
      Phoenix.PubSub.Supervisor.start_link(
        name: debug_pubsub_name,
        adapter_name: Phoenix.PubSub.PG2
      )

    %{
      pubsub_name: pubsub_name,
      debug_pubsub_name: debug_pubsub_name
    }
  end

  # Helper to create parent agent with debug pubsub
  defp start_parent_agent(context) do
    parent_agent = create_test_agent()

    {:ok, _pid} =
      AgentServer.start_link(
        agent: parent_agent,
        pubsub: {Phoenix.PubSub, context.pubsub_name},
        debug_pubsub: {Phoenix.PubSub, context.debug_pubsub_name}
      )

    # Subscribe to parent's debug topic
    debug_topic = "agent_server:debug:#{parent_agent.agent_id}"
    Phoenix.PubSub.subscribe(context.debug_pubsub_name, debug_topic)

    parent_agent
  end

  # Helper to create a SubAgent struct with parent
  defp create_subagent(parent_agent_id, opts \\ []) do
    agent = Keyword.get(opts, :agent, create_test_agent())
    instructions = Keyword.get(opts, :instructions, "Do something")
    parent_state = Keyword.get(opts, :parent_state, State.new!(%{}))

    SubAgent.new_from_config(
      parent_agent_id: parent_agent_id,
      instructions: instructions,
      agent_config: agent,
      parent_state: parent_state
    )
  end

  # Helper to create a simple test tool
  defp create_test_tool do
    Function.new!(%{
      name: "get_weather",
      description: "Get the weather for a location",
      parameters_schema: %{
        type: "object",
        properties: %{
          location: %{type: "string", description: "The location to get weather for"}
        },
        required: ["location"]
      },
      function: fn %{"location" => location}, _context ->
        {:ok, "Weather in #{location}: Sunny, 72°F"}
      end
    })
  end

  # Helper to create an agent with tools
  defp create_test_agent_with_tools do
    create_test_agent(tools: [create_test_tool()])
  end

  describe "subagent_started event" do
    test "broadcasts subagent_started on init via parent", context do
      parent_agent = start_parent_agent(context)
      subagent = create_subagent(parent_agent.agent_id)

      # Start subagent server
      {:ok, _pid} = SubAgentServer.start_link(subagent: subagent)

      # Should receive started event via parent's topic
      assert_receive {:agent, {:debug, {:subagent, sub_id, {:subagent_started, data}}}}, 100

      assert sub_id == subagent.id
      assert data.parent_id == parent_agent.agent_id
      assert data.id == subagent.id
      assert is_binary(data.name)
      assert is_list(data.tools)
      assert is_binary(data.model)
    end

    test "includes instructions in initial_messages in started metadata", context do
      parent_agent = start_parent_agent(context)
      instructions = "Research renewable energy impacts on climate"
      subagent = create_subagent(parent_agent.agent_id, instructions: instructions)

      {:ok, _pid} = SubAgentServer.start_link(subagent: subagent)

      assert_receive {:agent, {:debug, {:subagent, _, {:subagent_started, data}}}}, 100
      # Instructions are included as a user message in initial_messages
      assert is_list(data.initial_messages)
      user_message = Enum.find(data.initial_messages, fn msg -> msg.role == :user end)
      assert user_message != nil
      # Content is a list of ContentPart structs
      assert instructions == ContentPart.parts_to_string(user_message.content)
    end
  end

  describe "subagent_status_changed event" do
    test "broadcasts status change to running on execute", context do
      parent_agent = start_parent_agent(context)
      subagent = create_subagent(parent_agent.agent_id)

      {:ok, _pid} = SubAgentServer.start_link(subagent: subagent)

      # Consume the started event
      assert_receive {:agent, {:debug, {:subagent, _, {:subagent_started, _}}}}, 100

      # Mock ChatAnthropic.call
      ChatAnthropic
      |> stub(:call, fn _model, _messages, _callbacks ->
        {:ok, [Message.new_assistant!("Done")]}
      end)

      # Execute (spawns a task to avoid blocking test)
      Task.async(fn -> SubAgentServer.execute(subagent.id) end)

      # Should receive running status
      assert_receive {:agent, {:debug, {:subagent, sub_id, {:subagent_status_changed, :running}}}},
                     1000

      assert sub_id == subagent.id
    end
  end

  describe "subagent_llm_message event" do
    test "broadcasts llm_message for simple assistant response", context do
      parent_agent = start_parent_agent(context)
      subagent = create_subagent(parent_agent.agent_id)

      {:ok, _pid} = SubAgentServer.start_link(subagent: subagent)

      # Consume started event
      assert_receive {:agent, {:debug, {:subagent, _, {:subagent_started, _}}}}, 100

      # Mock ChatAnthropic.call at the deepest level to let LLMChain machinery run
      # This ensures callbacks are actually fired through the normal code path
      ChatAnthropic
      |> stub(:call, fn _model, _messages, _callbacks ->
        {:ok, [Message.new_assistant!("Task completed successfully")]}
      end)

      {:ok, _result} = SubAgentServer.execute(subagent.id)

      # Consume running status
      assert_receive {:agent, {:debug, {:subagent, _, {:subagent_status_changed, :running}}}}

      # Should receive llm_message event for the assistant message
      # This verifies the full callback chain works: SubAgentServer -> SubAgent -> LLMChain -> callback
      assert_receive {:agent, {:debug, {:subagent, sub_id, {:subagent_llm_message, message}}}}

      assert sub_id == subagent.id
      assert message.role == :assistant
      # Content may be a list of ContentParts or a string
      content_text =
        case message.content do
          text when is_binary(text) -> text
          parts when is_list(parts) -> Enum.map_join(parts, "", & &1.content)
        end

      assert content_text == "Task completed successfully"
    end

    test "broadcasts llm_message events during tool call execution loop", context do
      parent_agent = start_parent_agent(context)
      # Create subagent with a tool so we can exercise the full execution loop
      agent_with_tools = create_test_agent_with_tools()
      subagent = create_subagent(parent_agent.agent_id, agent: agent_with_tools)

      {:ok, _pid} = SubAgentServer.start_link(subagent: subagent)

      # Consume started event
      assert_receive {:agent, {:debug, {:subagent, _, {:subagent_started, _}}}}, 100

      # Mock ChatAnthropic.call to return different responses on each call:
      # First call: returns a tool call
      # Second call: returns a final assistant message
      call_counter = :counters.new(1, [:atomics])

      ChatAnthropic
      |> stub(:call, fn _model, _messages, _callbacks ->
        call_count = :counters.get(call_counter, 1) + 1
        :counters.put(call_counter, 1, call_count)

        if call_count == 1 do
          # First call: return a tool call
          tool_call = ToolCall.new!(%{
            call_id: "call_test_123",
            name: "get_weather",
            arguments: Jason.encode!(%{"location" => "San Francisco"})
          })

          {:ok, [Message.new_assistant!(%{tool_calls: [tool_call]})]}
        else
          # Second call: return final response
          {:ok, [Message.new_assistant!("The weather in San Francisco is sunny and 72°F.")]}
        end
      end)

      {:ok, result} = SubAgentServer.execute(subagent.id)
      assert result == "The weather in San Francisco is sunny and 72°F."

      # Consume running status
      assert_receive {:agent, {:debug, {:subagent, _, {:subagent_status_changed, :running}}}}

      # Should receive llm_message event for the first assistant message (with tool calls)
      assert_receive {:agent, {:debug, {:subagent, sub_id, {:subagent_llm_message, tool_call_msg}}}}

      assert sub_id == subagent.id
      assert tool_call_msg.role == :assistant
      assert length(tool_call_msg.tool_calls) == 1
      assert hd(tool_call_msg.tool_calls).name == "get_weather"

      # Should receive llm_message event for the tool result
      assert_receive {:agent, {:debug, {:subagent, ^sub_id, {:subagent_llm_message, tool_result_msg}}}}

      assert tool_result_msg.role == :tool
      assert length(tool_result_msg.tool_results) == 1
      tool_result = hd(tool_result_msg.tool_results)
      assert tool_result.tool_call_id == "call_test_123"
      assert tool_result.name == "get_weather"

      # Should receive llm_message event for the final assistant message
      assert_receive {:agent, {:debug, {:subagent, ^sub_id, {:subagent_llm_message, final_msg}}}}

      assert final_msg.role == :assistant
      content_text = ContentPart.content_to_string(final_msg.content)
      assert content_text == "The weather in San Francisco is sunny and 72°F."
    end
  end

  describe "subagent_completed event" do
    test "broadcasts completion with result and duration", context do
      parent_agent = start_parent_agent(context)
      subagent = create_subagent(parent_agent.agent_id)

      {:ok, _pid} = SubAgentServer.start_link(subagent: subagent)

      # Consume started event
      assert_receive {:agent, {:debug, {:subagent, _, {:subagent_started, _}}}}, 100

      # Mock ChatAnthropic.call
      ChatAnthropic
      |> stub(:call, fn _model, _messages, _callbacks ->
        {:ok, [Message.new_assistant!("Task completed successfully")]}
      end)

      {:ok, result} = SubAgentServer.execute(subagent.id)
      assert result == "Task completed successfully"

      # Consume running status
      assert_receive {:agent, {:debug, {:subagent, _, {:subagent_status_changed, :running}}}},
                     1000

      # Should receive completed event
      assert_receive {:agent, {:debug, {:subagent, sub_id, {:subagent_completed, data}}}}, 100

      assert sub_id == subagent.id
      assert data.id == subagent.id
      assert data.result == "Task completed successfully"
      assert is_integer(data.duration_ms)
      assert data.duration_ms >= 0
    end
  end

  describe "subagent_error event" do
    test "broadcasts error on execution failure", context do
      parent_agent = start_parent_agent(context)
      subagent = create_subagent(parent_agent.agent_id)

      {:ok, _pid} = SubAgentServer.start_link(subagent: subagent)

      # Consume started event
      assert_receive {:agent, {:debug, {:subagent, _, {:subagent_started, _}}}}, 100

      # Mock ChatAnthropic.call to return error
      ChatAnthropic
      |> stub(:call, fn _model, _messages, _callbacks ->
        {:error, LangChain.LangChainError.exception(message: "API error: rate limit")}
      end)

      {:error, reason} = SubAgentServer.execute(subagent.id)
      assert %LangChain.LangChainError{message: "API error: rate limit"} = reason

      # Consume running status
      assert_receive {:agent, {:debug, {:subagent, _, {:subagent_status_changed, :running}}}},
                     1000

      # Should receive error event
      assert_receive {:agent, {:debug, {:subagent, sub_id, {:subagent_error, error_reason}}}},
                     1000

      assert sub_id == subagent.id
      assert %LangChain.LangChainError{message: "API error: rate limit"} = error_reason
    end
  end

  describe "no-op when parent has no debug_pubsub" do
    test "subagent starts without error when parent lacks debug_pubsub", context do
      # Start parent WITHOUT debug_pubsub
      parent_agent = create_test_agent()

      {:ok, _pid} =
        AgentServer.start_link(
          agent: parent_agent,
          pubsub: {Phoenix.PubSub, context.pubsub_name}
          # No debug_pubsub configured
        )

      # Subscribe to debug topic anyway (shouldn't receive anything)
      debug_topic = "agent_server:debug:#{parent_agent.agent_id}"
      Phoenix.PubSub.subscribe(context.debug_pubsub_name, debug_topic)

      subagent = create_subagent(parent_agent.agent_id)
      {:ok, _pid} = SubAgentServer.start_link(subagent: subagent)

      # Should NOT receive any events (parent has no debug_pubsub)
      refute_receive {:agent, {:debug, {:subagent, _, _}}}, 200
    end

    test "subagent executes without error when parent lacks debug_pubsub", context do
      # Start parent WITHOUT debug_pubsub
      parent_agent = create_test_agent()

      {:ok, _pid} =
        AgentServer.start_link(
          agent: parent_agent,
          pubsub: {Phoenix.PubSub, context.pubsub_name}
        )

      subagent = create_subagent(parent_agent.agent_id)
      {:ok, _pid} = SubAgentServer.start_link(subagent: subagent)

      # Mock ChatAnthropic.call
      ChatAnthropic
      |> stub(:call, fn _model, _messages, _callbacks ->
        {:ok, [Message.new_assistant!("Done")]}
      end)

      # Execute should still work
      {:ok, result} = SubAgentServer.execute(subagent.id)
      assert result == "Done"
    end
  end

  describe "resume broadcasts events" do
    test "broadcasts events during resume after HITL approval", context do
      parent_agent = start_parent_agent(context)

      # Create a tool that requires HITL approval
      file_write_tool =
        Function.new!(%{
          name: "file_write",
          description: "Write content to a file",
          parameters_schema: %{
            type: "object",
            properties: %{
              path: %{type: "string", description: "File path"},
              content: %{type: "string", description: "Content to write"}
            },
            required: ["path", "content"]
          },
          function: fn %{"path" => path, "content" => content}, _context ->
            {:ok, "Wrote #{byte_size(content)} bytes to #{path}"}
          end
        })

      # Create agent with the file_write tool and HITL config
      agent_with_hitl =
        create_test_agent(
          tools: [file_write_tool],
          middleware: [
            {LangChain.Agents.Middleware.HumanInTheLoop,
             interrupt_on: %{"file_write" => true}}
          ]
        )

      # Create subagent - new_from_config extracts interrupt_on from middleware
      subagent = create_subagent(parent_agent.agent_id, agent: agent_with_hitl)

      # Build a tool call that would have triggered HITL interrupt
      tool_call =
        ToolCall.new!(%{
          call_id: "call_file_write_1",
          name: "file_write",
          arguments: Jason.encode!(%{"path" => "test.txt", "content" => "hello world"})
        })

      # Add the assistant message with tool call to the chain
      # This simulates the state after execute returned {:interrupt, ...}
      tool_call_message = Message.new_assistant!(%{tool_calls: [tool_call]})
      chain_with_tool_call = LangChain.Chains.LLMChain.add_message(subagent.chain, tool_call_message)

      # Build the interrupt_data that would have been created
      action_request = %{
        tool_call_id: "call_file_write_1",
        tool_name: "file_write",
        arguments: %{"path" => "test.txt", "content" => "hello world"}
      }

      # Create the interrupted subagent state
      interrupted_subagent = %{
        subagent
        | status: :interrupted,
          chain: chain_with_tool_call,
          interrupt_data: %{
            action_requests: [action_request],
            hitl_tool_call_ids: ["call_file_write_1"]
          }
      }

      {:ok, _pid} = SubAgentServer.start_link(subagent: interrupted_subagent)

      # Consume started event
      assert_receive {:agent, {:debug, {:subagent, _, {:subagent_started, _}}}}, 100

      # Mock ChatAnthropic.call for the completion response after tool execution
      ChatAnthropic
      |> stub(:call, fn _model, _messages, _callbacks ->
        {:ok, [Message.new_assistant!("I've written the file test.txt successfully.")]}
      end)

      # Resume with approval decision
      decisions = [%{type: :approve}]
      {:ok, result} = SubAgentServer.resume(interrupted_subagent.id, decisions)
      assert result == "I've written the file test.txt successfully."

      # Should receive running status when resume starts
      assert_receive {:agent, {:debug, {:subagent, sub_id, {:subagent_status_changed, :running}}}}
      assert sub_id == interrupted_subagent.id

      # Should receive llm_message event for the tool result (from executing the approved tool)
      assert_receive {:agent, {:debug, {:subagent, ^sub_id, {:subagent_llm_message, tool_result_msg}}}}
      assert tool_result_msg.role == :tool
      assert length(tool_result_msg.tool_results) == 1
      tool_result = hd(tool_result_msg.tool_results)
      assert tool_result.tool_call_id == "call_file_write_1"
      assert tool_result.name == "file_write"

      # Should receive llm_message event for the final assistant response
      assert_receive {:agent, {:debug, {:subagent, ^sub_id, {:subagent_llm_message, assistant_msg}}}}
      assert assistant_msg.role == :assistant
      content_text = ContentPart.content_to_string(assistant_msg.content)
      assert content_text == "I've written the file test.txt successfully."

      # Should receive completed event
      assert_receive {:agent, {:debug, {:subagent, ^sub_id, {:subagent_completed, data}}}}

      assert data.result == "I've written the file test.txt successfully."
      assert is_integer(data.duration_ms)
    end
  end
end
