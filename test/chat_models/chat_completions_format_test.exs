defmodule LangChain.ChatModels.ChatCompletionsFormatTest do
  use ExUnit.Case, async: true

  alias LangChain.ChatModels.ChatCompletionsFormat
  alias LangChain.Message
  alias LangChain.Message.ContentPart
  alias LangChain.Message.ToolCall
  alias LangChain.Message.ToolResult
  alias LangChain.MessageDelta
  alias LangChain.Function

  describe "item_for_api/2 system role" do
    test "sends system messages as :system by default" do
      assert %{"role" => :system} =
               ChatCompletionsFormat.item_for_api(Message.new_system!("Be brief."))
    end

    test "sends system messages under the given system_role" do
      assert %{"role" => :developer} =
               ChatCompletionsFormat.item_for_api(Message.new_system!("Be brief."),
                 system_role: :developer
               )
    end

    test "system_role does not change other roles" do
      opts = [system_role: :developer]

      assert %{"role" => :user} =
               ChatCompletionsFormat.item_for_api(Message.new_user!("Hi"), opts)

      assert %{"role" => :assistant} =
               ChatCompletionsFormat.item_for_api(Message.new_assistant!("Hello"), opts)

      tool_msg =
        Message.new_tool_result!(%{
          tool_results: [ToolResult.new!(%{tool_call_id: "call_1", content: "42"})]
        })

      assert [%{"role" => :tool}] = ChatCompletionsFormat.item_for_api(tool_msg, opts)
    end
  end

  describe "messages_for_api/2" do
    test "expands a tool message into one message per result, in order" do
      messages = [
        Message.new_system!("System"),
        Message.new_user!("Use both tools"),
        Message.new_tool_result!(%{
          tool_results: [
            ToolResult.new!(%{tool_call_id: "call_1", content: "first"}),
            ToolResult.new!(%{tool_call_id: "call_2", content: "second"})
          ]
        }),
        Message.new_user!("Thanks")
      ]

      assert [
               %{"role" => :system},
               %{"role" => :user},
               %{"role" => :tool, "tool_call_id" => "call_1", "content" => [first]},
               %{"role" => :tool, "tool_call_id" => "call_2", "content" => [second]},
               %{"role" => :user}
             ] = ChatCompletionsFormat.messages_for_api(messages)

      assert first == %{"type" => "text", "text" => "first"}
      assert second == %{"type" => "text", "text" => "second"}
    end

    test "passes system_role through to each message" do
      assert [%{"role" => :developer}, %{"role" => :user}] =
               ChatCompletionsFormat.messages_for_api(
                 [Message.new_system!("System"), Message.new_user!("Hi")],
                 system_role: :developer
               )
    end
  end

  describe "content_parts_for_api/1" do
    test "omits thinking and unsupported parts" do
      parts = [
        ContentPart.thinking!("Let me think."),
        ContentPart.new!(%{type: :unsupported, content: "opaque", options: []}),
        ContentPart.text!("The answer.")
      ]

      assert [%{"type" => "text", "text" => "The answer."}] =
               ChatCompletionsFormat.content_parts_for_api(parts)
    end
  end

  describe "tools_for_api/1" do
    test "returns an empty list for nil" do
      assert [] == ChatCompletionsFormat.tools_for_api(nil)
    end

    test "wraps each function as a function tool" do
      fun =
        Function.new!(%{
          name: "get_weather",
          description: "Weather lookup",
          parameters_schema: %{"type" => "object", "properties" => %{}},
          function: fn _args, _context -> {:ok, "sunny"} end
        })

      assert [
               %{
                 "type" => "function",
                 "function" => %{
                   "name" => "get_weather",
                   "description" => "Weather lookup",
                   "parameters" => %{"type" => "object", "properties" => %{}}
                 }
               }
             ] = ChatCompletionsFormat.tools_for_api([fun])
    end
  end

  describe "process_response/1 with reasoning_content" do
    test "a full message puts thinking ahead of the answer" do
      data = %{
        "index" => 0,
        "finish_reason" => "stop",
        "message" => %{
          "role" => "assistant",
          "content" => "9.9 is larger.",
          "reasoning_content" => "Comparing tenths."
        }
      }

      assert %Message{content: [thinking, text]} = ChatCompletionsFormat.process_response(data)
      assert %ContentPart{type: :thinking, content: "Comparing tenths."} = thinking
      assert %ContentPart{type: :text, content: "9.9 is larger."} = text
    end

    test "a full message with tool calls keeps both thinking and calls" do
      data = %{
        "index" => 0,
        "finish_reason" => "tool_calls",
        "message" => %{
          "role" => "assistant",
          "content" => nil,
          "reasoning_content" => "I should look it up.",
          "tool_calls" => [
            %{
              "id" => "call_1",
              "type" => "function",
              "function" => %{"name" => "lookup", "arguments" => "{}"}
            }
          ]
        }
      }

      assert %Message{content: [thinking], tool_calls: [call]} =
               ChatCompletionsFormat.process_response(data)

      assert %ContentPart{type: :thinking, content: "I should look it up."} = thinking
      assert %ToolCall{name: "lookup", call_id: "call_1", status: :complete} = call
    end

    test "a delta carrying thinking targets content index 0" do
      data = %{
        "index" => 0,
        "delta" => %{"role" => "assistant", "content" => nil, "reasoning_content" => "Hmm"}
      }

      assert %MessageDelta{index: 0, content: %ContentPart{type: :thinking, content: "Hmm"}} =
               ChatCompletionsFormat.process_response(data)
    end

    test "a delta carrying answer text after thinking targets content index 1" do
      data = %{
        "index" => 0,
        "delta" => %{"content" => "Answer", "reasoning_content" => nil}
      }

      assert %MessageDelta{index: 1, content: "Answer"} =
               ChatCompletionsFormat.process_response(data)
    end
  end

  describe "decode_stream/2" do
    test "buffers a JSON object split across chunks" do
      first = ~s(data: {"id":"1","choices":[{"index":0,"delta":{"content":"Hel)

      assert {[], buffer} = ChatCompletionsFormat.decode_stream({first, ""})
      assert buffer != ""

      second = ~s(lo"}}]}\n\ndata: [DONE]\n\n)

      assert {[%{"choices" => [%{"delta" => %{"content" => "Hello"}}]}], ""} =
               ChatCompletionsFormat.decode_stream({second, buffer})
    end

    test "ignores SSE comment lines" do
      raw = ~s(: KEEPALIVE\n\ndata: {"id":"1","choices":[]}\n\n)

      assert {[%{"id" => "1"}], ""} = ChatCompletionsFormat.decode_stream({raw, ""})
    end
  end
end
