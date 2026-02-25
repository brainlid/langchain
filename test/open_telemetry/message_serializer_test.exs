defmodule LangChain.OpenTelemetry.MessageSerializerTest do
  use ExUnit.Case, async: true

  alias LangChain.Message
  alias LangChain.Message.ContentPart
  alias LangChain.Message.ToolCall
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

    test "serializes empty list" do
      assert MessageSerializer.serialize_input([]) == "[]"
    end
  end

  describe "serialize_output/1" do
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
  end
end
