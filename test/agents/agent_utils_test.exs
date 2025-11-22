defmodule LangChain.Agents.AgentUtilsTest do
  use ExUnit.Case, async: true

  alias LangChain.Agents.AgentUtils
  alias LangChain.Message
  alias LangChain.Message.ToolCall
  alias LangChain.Chains.LLMChain

  # Helper to create a mock chain with tool calls
  defp create_chain_with_tool_calls(tool_calls) do
    assistant_message =
      Message.new_assistant!(%{
        content: "I'll use these tools",
        tool_calls: tool_calls
      })

    %LLMChain{
      last_message: assistant_message,
      messages: [assistant_message]
    }
  end

  # Helper to create a mock chain with no tool calls
  defp create_chain_with_no_tool_calls do
    assistant_message = Message.new_assistant!(%{content: "Just a response"})

    %LLMChain{
      last_message: assistant_message,
      messages: [assistant_message]
    }
  end

  # Helper to create a tool call
  defp create_tool_call(name, call_id, arguments \\ %{}) do
    %ToolCall{
      name: name,
      call_id: call_id,
      arguments: arguments
    }
  end

  describe "check_for_hitl_interrupt/2" do
    test "returns :continue when no tool calls present" do
      chain = create_chain_with_no_tool_calls()
      interrupt_on = %{"write_file" => true}

      assert AgentUtils.check_for_hitl_interrupt(chain, interrupt_on) == :continue
    end

    test "returns :continue when tool calls are not in interrupt_on config" do
      tool_calls = [
        create_tool_call("read_file", "call_1", %{"path" => "test.txt"})
      ]

      chain = create_chain_with_tool_calls(tool_calls)
      # Only write_file needs approval
      interrupt_on = %{"write_file" => true}

      assert AgentUtils.check_for_hitl_interrupt(chain, interrupt_on) == :continue
    end

    test "returns :continue when tool has explicit false in config" do
      tool_calls = [
        create_tool_call("read_file", "call_1", %{"path" => "test.txt"})
      ]

      chain = create_chain_with_tool_calls(tool_calls)
      # Explicitly no approval needed
      interrupt_on = %{"read_file" => false}

      assert AgentUtils.check_for_hitl_interrupt(chain, interrupt_on) == :continue
    end

    test "returns interrupt when tool requires approval (boolean true)" do
      tool_calls = [
        create_tool_call("write_file", "call_1", %{"path" => "test.txt", "content" => "data"})
      ]

      chain = create_chain_with_tool_calls(tool_calls)
      interrupt_on = %{"write_file" => true}

      assert {:interrupt, interrupt_data} =
               AgentUtils.check_for_hitl_interrupt(chain, interrupt_on)

      assert %{action_requests: action_requests, hitl_tool_call_ids: hitl_ids} = interrupt_data
      assert length(action_requests) == 1

      [request] = action_requests
      assert request.tool_call_id == "call_1"
      assert request.tool_name == "write_file"
      assert request.arguments == %{"path" => "test.txt", "content" => "data"}

      assert hitl_ids == ["call_1"]
    end

    test "returns interrupt when tool requires approval (config map)" do
      tool_calls = [
        create_tool_call("delete_file", "call_2", %{"path" => "dangerous.txt"})
      ]

      chain = create_chain_with_tool_calls(tool_calls)
      interrupt_on = %{"delete_file" => %{allowed_decisions: [:approve, :reject]}}

      assert {:interrupt, interrupt_data} =
               AgentUtils.check_for_hitl_interrupt(chain, interrupt_on)

      assert %{action_requests: action_requests, hitl_tool_call_ids: hitl_ids} = interrupt_data
      assert length(action_requests) == 1

      [request] = action_requests
      assert request.tool_call_id == "call_2"
      assert request.tool_name == "delete_file"

      assert hitl_ids == ["call_2"]
    end

    test "filters mixed tool calls, only returns HITL tools" do
      tool_calls = [
        create_tool_call("read_file", "call_1", %{"path" => "safe.txt"}),
        create_tool_call("write_file", "call_2", %{"path" => "important.txt"}),
        create_tool_call("search", "call_3", %{"query" => "data"})
      ]

      chain = create_chain_with_tool_calls(tool_calls)

      interrupt_on = %{
        # No approval
        "read_file" => false,
        # Needs approval
        "write_file" => true,
        # No approval
        "search" => false
      }

      assert {:interrupt, interrupt_data} =
               AgentUtils.check_for_hitl_interrupt(chain, interrupt_on)

      assert %{action_requests: action_requests, hitl_tool_call_ids: hitl_ids} = interrupt_data
      assert length(action_requests) == 1

      [request] = action_requests
      assert request.tool_name == "write_file"
      assert request.tool_call_id == "call_2"

      assert hitl_ids == ["call_2"]
    end

    test "handles multiple HITL tools" do
      tool_calls = [
        create_tool_call("write_file", "call_1", %{"path" => "file1.txt"}),
        create_tool_call("delete_file", "call_2", %{"path" => "file2.txt"}),
        create_tool_call("write_file", "call_3", %{"path" => "file3.txt"})
      ]

      chain = create_chain_with_tool_calls(tool_calls)

      interrupt_on = %{
        "write_file" => true,
        "delete_file" => true
      }

      assert {:interrupt, interrupt_data} =
               AgentUtils.check_for_hitl_interrupt(chain, interrupt_on)

      assert %{action_requests: action_requests, hitl_tool_call_ids: hitl_ids} = interrupt_data
      assert length(action_requests) == 3

      # Verify all three are in action_requests
      assert Enum.all?(action_requests, fn req ->
               req.tool_name in ["write_file", "delete_file"]
             end)

      assert length(hitl_ids) == 3
      assert "call_1" in hitl_ids
      assert "call_2" in hitl_ids
      assert "call_3" in hitl_ids
    end

    test "returns :continue when interrupt_on is empty map" do
      tool_calls = [
        create_tool_call("write_file", "call_1", %{"path" => "test.txt"})
      ]

      chain = create_chain_with_tool_calls(tool_calls)
      interrupt_on = %{}

      assert AgentUtils.check_for_hitl_interrupt(chain, interrupt_on) == :continue
    end

    test "handles chain with last_message as non-assistant" do
      # Chain where last message is not assistant
      user_message = Message.new_user!("Hello")

      chain = %LLMChain{
        last_message: user_message,
        messages: [user_message]
      }

      interrupt_on = %{"write_file" => true}

      assert AgentUtils.check_for_hitl_interrupt(chain, interrupt_on) == :continue
    end
  end

  describe "build_full_decisions/4" do
    test "auto-approves all tools when no HITL tools" do
      all_tool_calls = [
        create_tool_call("read_file", "call_1"),
        create_tool_call("search", "call_2"),
        create_tool_call("list_files", "call_3")
      ]

      hitl_tool_call_ids = []
      human_decisions = []
      action_requests = []

      full_decisions =
        AgentUtils.build_full_decisions(
          all_tool_calls,
          hitl_tool_call_ids,
          human_decisions,
          action_requests
        )

      assert length(full_decisions) == 3
      assert Enum.all?(full_decisions, fn decision -> decision == %{type: :approve} end)
    end

    test "mixes human decisions with auto-approvals" do
      all_tool_calls = [
        create_tool_call("read_file", "call_1"),
        create_tool_call("write_file", "call_2"),
        create_tool_call("search", "call_3")
      ]

      # Only write_file needed HITL approval
      hitl_tool_call_ids = ["call_2"]

      action_requests = [
        %{tool_call_id: "call_2", tool_name: "write_file", arguments: %{}}
      ]

      human_decisions = [
        %{type: :edit, arguments: %{"path" => "modified.txt"}}
      ]

      full_decisions =
        AgentUtils.build_full_decisions(
          all_tool_calls,
          hitl_tool_call_ids,
          human_decisions,
          action_requests
        )

      assert length(full_decisions) == 3

      # First tool (read_file) - auto-approved
      assert Enum.at(full_decisions, 0) == %{type: :approve}

      # Second tool (write_file) - human decision
      assert Enum.at(full_decisions, 1) == %{type: :edit, arguments: %{"path" => "modified.txt"}}

      # Third tool (search) - auto-approved
      assert Enum.at(full_decisions, 2) == %{type: :approve}
    end

    test "handles multiple HITL tools with different decisions" do
      all_tool_calls = [
        create_tool_call("write_file", "call_1"),
        create_tool_call("read_file", "call_2"),
        create_tool_call("delete_file", "call_3"),
        create_tool_call("search", "call_4")
      ]

      # write_file and delete_file needed HITL
      hitl_tool_call_ids = ["call_1", "call_3"]

      action_requests = [
        %{tool_call_id: "call_1", tool_name: "write_file", arguments: %{}},
        %{tool_call_id: "call_3", tool_name: "delete_file", arguments: %{}}
      ]

      human_decisions = [
        # Approve write_file
        %{type: :approve},
        # Reject delete_file
        %{type: :reject}
      ]

      full_decisions =
        AgentUtils.build_full_decisions(
          all_tool_calls,
          hitl_tool_call_ids,
          human_decisions,
          action_requests
        )

      assert length(full_decisions) == 4

      # write_file - human approved
      assert Enum.at(full_decisions, 0) == %{type: :approve}

      # read_file - auto-approved
      assert Enum.at(full_decisions, 1) == %{type: :approve}

      # delete_file - human rejected
      assert Enum.at(full_decisions, 2) == %{type: :reject}

      # search - auto-approved
      assert Enum.at(full_decisions, 3) == %{type: :approve}
    end

    test "handles all HITL tools" do
      all_tool_calls = [
        create_tool_call("write_file", "call_1"),
        create_tool_call("delete_file", "call_2")
      ]

      hitl_tool_call_ids = ["call_1", "call_2"]

      action_requests = [
        %{tool_call_id: "call_1", tool_name: "write_file", arguments: %{}},
        %{tool_call_id: "call_2", tool_name: "delete_file", arguments: %{}}
      ]

      human_decisions = [
        %{type: :approve},
        %{type: :edit, arguments: %{"path" => "safe.txt"}}
      ]

      full_decisions =
        AgentUtils.build_full_decisions(
          all_tool_calls,
          hitl_tool_call_ids,
          human_decisions,
          action_requests
        )

      assert length(full_decisions) == 2
      assert Enum.at(full_decisions, 0) == %{type: :approve}
      assert Enum.at(full_decisions, 1) == %{type: :edit, arguments: %{"path" => "safe.txt"}}
    end

    test "preserves order of tool calls" do
      all_tool_calls = [
        create_tool_call("tool_a", "call_1"),
        create_tool_call("tool_b", "call_2"),
        create_tool_call("tool_c", "call_3"),
        create_tool_call("tool_d", "call_4")
      ]

      # tool_b and tool_d need HITL
      hitl_tool_call_ids = ["call_2", "call_4"]

      action_requests = [
        %{tool_call_id: "call_2", tool_name: "tool_b", arguments: %{}},
        %{tool_call_id: "call_4", tool_name: "tool_d", arguments: %{}}
      ]

      human_decisions = [
        %{type: :approve, note: "decision_for_b"},
        %{type: :reject, note: "decision_for_d"}
      ]

      full_decisions =
        AgentUtils.build_full_decisions(
          all_tool_calls,
          hitl_tool_call_ids,
          human_decisions,
          action_requests
        )

      # Verify order is preserved
      # tool_a - auto
      assert Enum.at(full_decisions, 0) == %{type: :approve}
      # tool_b - human
      assert Enum.at(full_decisions, 1) == %{type: :approve, note: "decision_for_b"}
      # tool_c - auto
      assert Enum.at(full_decisions, 2) == %{type: :approve}
      # tool_d - human
      assert Enum.at(full_decisions, 3) == %{type: :reject, note: "decision_for_d"}
    end
  end

  describe "get_tool_calls_from_last_message/1" do
    test "returns tool calls from assistant message" do
      tool_calls = [
        create_tool_call("write_file", "call_1"),
        create_tool_call("read_file", "call_2")
      ]

      chain = create_chain_with_tool_calls(tool_calls)

      [result_1, result_2] = AgentUtils.get_tool_calls_from_last_message(chain)

      assert result_1.name == "write_file"
      assert result_1.call_id == "call_1"
      assert result_2.name == "read_file"
      assert result_2.call_id == "call_2"
    end

    test "returns empty list when no tool calls" do
      chain = create_chain_with_no_tool_calls()

      result = AgentUtils.get_tool_calls_from_last_message(chain)

      assert result == []
    end

    test "returns empty list when last message is not assistant" do
      user_message = Message.new_user!("Hello")

      chain = %LLMChain{
        last_message: user_message,
        messages: [user_message]
      }

      result = AgentUtils.get_tool_calls_from_last_message(chain)

      assert result == []
    end

    test "returns empty list when tool_calls is nil" do
      assistant_message = Message.new_assistant!(%{content: "Response", tool_calls: nil})

      chain = %LLMChain{
        last_message: assistant_message,
        messages: [assistant_message]
      }

      result = AgentUtils.get_tool_calls_from_last_message(chain)

      assert result == []
    end

    test "returns empty list when tool_calls is empty list" do
      assistant_message = Message.new_assistant!(%{content: "Response", tool_calls: []})

      chain = %LLMChain{
        last_message: assistant_message,
        messages: [assistant_message]
      }

      result = AgentUtils.get_tool_calls_from_last_message(chain)

      assert result == []
    end

    test "handles single tool call" do
      tool_calls = [create_tool_call("write_file", "call_1", %{"path" => "test.txt"})]

      chain = create_chain_with_tool_calls(tool_calls)

      [result] = AgentUtils.get_tool_calls_from_last_message(chain)

      assert result.name == "write_file"
      assert result.arguments == %{"path" => "test.txt"}
    end
  end
end
