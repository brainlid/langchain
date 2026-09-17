defmodule LangChain.OpenTelemetry.MessageSerializerTest do
  use ExUnit.Case, async: true

  alias LangChain.Chains.LLMChain
  alias LangChain.Chains.LLMChain.Mode.Steps
  alias LangChain.ChatModels.ChatOpenAI
  alias LangChain.Message
  alias LangChain.Message.ContentPart
  alias LangChain.Message.ToolCall
  alias LangChain.Message.ToolResult
  alias LangChain.MessageExpansion
  alias LangChain.OpenTelemetry.MessageSerializer

  describe "serialize_input/1" do
    test "serializes text messages" do
      messages = [
        Message.new_system!("Be helpful"),
        Message.new_user!("Hello")
      ]

      json = MessageSerializer.serialize_input(messages)
      decoded = Jason.decode!(json)

      assert [
               %{"role" => "system", "content" => "Be helpful"},
               %{"role" => "user", "content" => "Hello"}
             ] = decoded
    end

    test "serializes multi-part content" do
      messages = [
        Message.new_user!([
          ContentPart.text!("What is this?"),
          ContentPart.image_url!("https://example.com/img.png")
        ])
      ]

      json = MessageSerializer.serialize_input(messages)
      decoded = Jason.decode!(json)

      assert [%{"role" => "user", "content" => content}] = decoded

      assert [%{"type" => "text", "text" => "What is this?"}, %{"type" => "image_url"} | _] =
               content
    end

    test "serializes assistant messages with tool calls" do
      tool_call =
        ToolCall.new!(%{
          call_id: "call-1",
          name: "calculator",
          arguments: %{"x" => 1}
        })

      msg = Message.new_assistant!(%{tool_calls: [tool_call]})

      json = MessageSerializer.serialize_input([msg])
      decoded = Jason.decode!(json)

      assert [%{"role" => "assistant", "tool_calls" => [tc]}] = decoded
      assert tc["id"] == "call-1"
      assert tc["type"] == "function"
      assert tc["function"]["name"] == "calculator"
    end

    test "expands tool results in conversation order with their call IDs and error content" do
      messages = [
        Message.new_assistant!(%{
          tool_calls: [
            ToolCall.new!(%{call_id: "call-1", name: "lookup", arguments: %{}}),
            ToolCall.new!(%{call_id: "call-2", name: "lookup", arguments: %{}})
          ]
        }),
        Message.new_tool_result!(%{
          tool_results: [
            ToolResult.new!(%{tool_call_id: "call-1", content: "Found a match"}),
            ToolResult.new!(%{
              tool_call_id: "call-2",
              content: "No match found",
              is_error: true
            })
          ]
        }),
        Message.new_assistant!(%{content: "One match was found"})
      ]

      assert [
               %{
                 "role" => "assistant",
                 "tool_calls" => [%{"id" => "call-1"}, %{"id" => "call-2"}]
               },
               %{"role" => "tool", "tool_call_id" => "call-1", "content" => "Found a match"},
               %{"role" => "tool", "tool_call_id" => "call-2", "content" => "No match found"},
               %{"role" => "assistant", "content" => "One match was found"}
             ] = messages |> MessageSerializer.serialize_input() |> Jason.decode!()
    end

    test "serializes tool content parts without application-only or reasoning data" do
      message =
        Message.new_tool_result!(%{
          tool_results: [
            ToolResult.new!(%{
              tool_call_id: "call-1",
              content: [
                ContentPart.text!("A chart"),
                ContentPart.image_url!("https://example.com/chart.png"),
                ContentPart.new!(%{type: :thinking, content: "private reasoning"})
              ],
              processed_content: %{internal: "application data"},
              display_text: "UI text"
            })
          ]
        })

      assert [
               %{
                 "role" => "tool",
                 "tool_call_id" => "call-1",
                 "content" => [
                   %{"type" => "text", "text" => "A chart"},
                   %{"type" => "image_url", "url" => "https://example.com/chart.png"}
                 ]
               }
             ] == [message] |> MessageSerializer.serialize_input() |> Jason.decode!()
    end

    test "an unapplied expansion serializes the result's fallback content only" do
      {:ok, result} =
        MessageExpansion.expand(
          "THE MATERIAL",
          [Message.new_assistant!("THE MATERIAL"), Message.new_user!("Use the content above.")],
          result_content: "Loaded 1 document."
        )

      message =
        Message.new_tool_result!(%{
          tool_results: [%ToolResult{result | tool_call_id: "call-1", name: "load_reference"}]
        })

      assert [%{"role" => "tool", "tool_call_id" => "call-1", "content" => "THE MATERIAL"}] ==
               [message] |> MessageSerializer.serialize_input() |> Jason.decode!()
    end

    test "an applied expansion serializes the trimmed result followed by the inserted messages" do
      {:ok, result} =
        MessageExpansion.expand(
          "THE MATERIAL",
          [Message.new_assistant!("THE MATERIAL"), Message.new_user!("Use the content above.")],
          result_content: "Loaded 1 document."
        )

      tool_call = ToolCall.new!(%{call_id: "call-1", name: "load_reference", arguments: %{}})

      chain =
        %{llm: ChatOpenAI.new!()}
        |> LLMChain.new!()
        |> LLMChain.add_message(Message.new_user!("What does the policy say?"))
        |> LLMChain.add_message(Message.new_assistant!(%{tool_calls: [tool_call]}))
        |> LLMChain.add_message(
          Message.new_tool_result!(%{
            tool_results: [%ToolResult{result | tool_call_id: "call-1", name: "load_reference"}]
          })
        )

      assert {:continue, expanded} = Steps.expand_tool_results({:continue, chain})

      assert [
               %{"role" => "user", "content" => "What does the policy say?"},
               %{"role" => "assistant", "tool_calls" => [%{"id" => "call-1"}]},
               %{"role" => "tool", "tool_call_id" => "call-1", "content" => "Loaded 1 document."},
               %{"role" => "assistant", "content" => "THE MATERIAL"},
               %{"role" => "user", "content" => "Use the content above."}
             ] = expanded.messages |> MessageSerializer.serialize_input() |> Jason.decode!()
    end

    test "serializes empty list" do
      assert MessageSerializer.serialize_input([]) == "[]"
    end

    test "serializes an :image content part with its data and media type" do
      b64 = Base.encode64(<<1, 2, 3>>)

      messages = [
        Message.new_user!([
          ContentPart.text!("look:"),
          ContentPart.image!(b64, media: :png)
        ])
      ]

      json = MessageSerializer.serialize_input(messages)
      decoded = Jason.decode!(json)

      assert [%{"role" => "user", "content" => parts}] = decoded

      assert %{"type" => "image", "data" => ^b64, "media" => "png"} =
               Enum.find(parts, &(&1["type"] == "image"))
    end
  end

  describe "serialize_output/1" do
    test "serializes a tool-result message with legacy string content" do
      message = %Message{
        role: :tool,
        tool_results: [%ToolResult{tool_call_id: "call-1", content: "Found a match"}]
      }

      assert [%{"role" => "tool", "tool_call_id" => "call-1", "content" => "Found a match"}] =
               message |> MessageSerializer.serialize_output() |> Jason.decode!()
    end

    test "serializes a single message" do
      msg = Message.new_assistant!(%{content: "Hello!"})

      json = MessageSerializer.serialize_output(msg)
      decoded = Jason.decode!(json)

      assert [%{"role" => "assistant", "content" => "Hello!"}] = decoded
    end

    test "serializes a list of messages" do
      messages = [
        Message.new_assistant!(%{content: "First"}),
        Message.new_assistant!(%{content: "Second"})
      ]

      json = MessageSerializer.serialize_output(messages)
      decoded = Jason.decode!(json)

      assert [
               %{"role" => "assistant", "content" => "First"},
               %{"role" => "assistant", "content" => "Second"}
             ] = decoded
    end

    test "handles nil content" do
      tool_call =
        ToolCall.new!(%{
          call_id: "call-1",
          name: "search",
          arguments: %{"query" => "test"}
        })

      msg = Message.new_assistant!(%{tool_calls: [tool_call]})

      json = MessageSerializer.serialize_output(msg)
      decoded = Jason.decode!(json)

      assert [%{"role" => "assistant", "content" => nil, "tool_calls" => [_]}] = decoded
    end

    test "passes already-serialized (binary) tool-call arguments through verbatim" do
      # Some providers hand back tool arguments as a raw JSON string rather than a
      # decoded map; those must be emitted as-is, not re-encoded.
      msg = %Message{
        role: :assistant,
        content: nil,
        tool_calls: [%ToolCall{call_id: "c1", name: "lookup", arguments: ~s({"q":"raw"})}]
      }

      json = MessageSerializer.serialize_output(msg)
      assert [%{"tool_calls" => [tc]}] = Jason.decode!(json)
      assert tc["function"]["arguments"] == ~s({"q":"raw"})
    end

    test "serializes nil tool-call arguments as an empty JSON object string" do
      msg = %Message{
        role: :assistant,
        content: nil,
        tool_calls: [%ToolCall{call_id: "c1", name: "noop", arguments: nil}]
      }

      json = MessageSerializer.serialize_output(msg)
      assert [%{"tool_calls" => [tc]}] = Jason.decode!(json)
      assert tc["function"]["arguments"] == "{}"
    end
  end

  describe "thinking/reasoning content filtering" do
    test "filters out thinking content parts from multi-part output" do
      msg = %Message{
        role: :assistant,
        content: [
          ContentPart.new!(%{type: :thinking, content: "Let me reason about this..."}),
          ContentPart.text!("Here is the answer")
        ]
      }

      json = MessageSerializer.serialize_output(msg)
      decoded = Jason.decode!(json)

      assert [%{"role" => "assistant", "content" => [text_part]}] = decoded
      assert text_part == %{"type" => "text", "text" => "Here is the answer"}
    end

    test "filters out unsupported content parts (e.g. redacted_thinking)" do
      msg = %Message{
        role: :assistant,
        content: [
          ContentPart.new!(%{
            type: :unsupported,
            content: nil,
            options: [type: "redacted_thinking"]
          }),
          ContentPart.text!("The response")
        ]
      }

      json = MessageSerializer.serialize_output(msg)
      decoded = Jason.decode!(json)

      assert [%{"role" => "assistant", "content" => [text_part]}] = decoded
      assert text_part == %{"type" => "text", "text" => "The response"}
    end

    test "filters thinking from input messages too" do
      messages = [
        %Message{
          role: :assistant,
          content: [
            ContentPart.new!(%{type: :thinking, content: "reasoning..."}),
            ContentPart.text!("visible text")
          ]
        }
      ]

      json = MessageSerializer.serialize_input(messages)
      decoded = Jason.decode!(json)

      assert [%{"role" => "assistant", "content" => [text_part]}] = decoded
      assert text_part == %{"type" => "text", "text" => "visible text"}
    end
  end
end
