defmodule LangChain.MessageTest do
  use ExUnit.Case
  doctest LangChain.Message
  alias LangChain.Message
  alias LangChain.Message.ToolCall
  alias LangChain.Message.UserContentPart

  describe "new/1" do
    test "works with minimal attrs" do
      assert {:ok, %Message{} = msg} =
               Message.new(%{"role" => "system", "content" => "hello!", "index" => 0})

      assert msg.role == :system
      assert msg.content == "hello!"
      assert msg.index == 0
    end

    test "accepts atom keys and role enum" do
      assert {:ok, %Message{} = msg} = Message.new(%{role: :system, content: "hello!"})
      assert msg.role == :system
      assert msg.content == "hello!"
    end

    test "returns error when invalid" do
      assert {:error, changeset} = Message.new(%{"role" => nil})
      refute changeset.valid?
      assert {"can't be blank", _} = changeset.errors[:role]
    end

    test "adds error when tool arguments are valid JSON but not a map" do
      json = Jason.encode!([true, 1, 2, 3])

      assert {:error, changeset} =
               Message.new_assistant(%{
                 tool_calls: [
                   ToolCall.new!(%{
                     call_id: "call_abc123",
                     name: "my_fun",
                     arguments: json
                   })
                 ],
                 status: :complete
               })

      refute changeset.valid?

      assert {"arguments: a json object is expected for tool arguments", _} =
               changeset.errors[:tool_calls]
    end

    test "adds error to arguments if it fails to parse" do
      assert {:error, changeset} =
               Message.new_assistant(%{
                 tool_calls: [
                   ToolCall.new!(%{
                     call_id: "call_abc123",
                     name: "my_fun",
                     arguments: "invalid"
                   })
                 ],
                 status: :complete
               })

      refute changeset.valid?
      assert {"arguments: invalid json", _} = changeset.errors[:tool_calls]
    end
  end

  describe "validations" do
    test "allows blank content for assistant message" do
      assert {:ok, %Message{} = msg} = Message.new(%{role: :assistant, content: nil})
      assert msg.role == :assistant
      assert msg.content == nil
    end

    test "require content for system and user messages" do
      assert {:error, changeset} = Message.new(%{role: :system, content: nil})
      assert {"can't be blank", _} = changeset.errors[:content]

      assert {:error, changeset} = Message.new(%{role: :user, content: nil})
      assert {"can't be blank", _} = changeset.errors[:content]
    end

    test "requires content to be text or UserContentParts when a user message" do
      # can be a content part
      part = UserContentPart.text!("Hi")
      {:ok, message} = Message.new_user([part])
      assert message.content == [part]

      # can be a string
      {:ok, message} = Message.new_user("Hi")
      assert message.content == "Hi"

      # content parts not allowed for other role types
      {:error, changeset} = Message.new_assistant(%{content: [part]})
      assert {"is invalid for role assistant", _} = changeset.errors[:content]

      {:error, changeset} = Message.new_system([part])
      assert {"is invalid for role system", _} = changeset.errors[:content]

      {:error, changeset} = Message.new(%{role: :tool, call_id: "tool_123", content: [part]})
      assert {"is invalid for role tool", _} = changeset.errors[:content]
    end

    test "content can be nil when an assistant message (tool calls)" do
      tool_call = ToolCall.new!(%{call_id: "1", name: "hello"})

      {:ok, message} =
        Message.new_assistant(%{
          content: nil,
          tool_calls: [tool_call]
        })

      assert message.content == nil
      # the tool call gets completed
      assert message.tool_calls == [Map.put(tool_call, :status, :complete)]
    end
  end

  describe "new_system/1" do
    test "creates a system message" do
      assert {:ok, %Message{role: :system} = msg} = Message.new_system("You are an AI.")
      assert msg.content == "You are an AI."
    end

    test "provides default content" do
      assert {:ok, msg} = Message.new_system()
      assert msg.content == "You are a helpful assistant."
    end

    test "requires content" do
      assert_raise LangChain.LangChainError, "content: can't be blank", fn ->
        Message.new_system!(nil)
      end
    end
  end

  describe "new_user/1" do
    test "creates a user message" do
      assert {:ok, %Message{role: :user} = msg} = Message.new_user("Hello!")
      assert msg.content == "Hello!"
    end

    test "requires content" do
      assert {:error, changeset} = Message.new_user(nil)
      assert {"can't be blank", _} = changeset.errors[:content]
    end

    test "accepts list of UserContentParts for content" do
      assert {:ok, %Message{} = msg} =
               Message.new_user([
                 UserContentPart.text!("Describe what is in this image:"),
                 UserContentPart.image!(:base64.encode("fake_image_data"))
               ])

      assert msg.role == :user

      assert msg.content == [
               %UserContentPart{type: :text, content: "Describe what is in this image:"},
               %UserContentPart{type: :image, content: "ZmFrZV9pbWFnZV9kYXRh", options: []}
             ]
    end

    test "does not accept invalid contents" do
      assert {:error, changeset} = Message.new_user(123)
      assert {"must be text or a list of UserContentParts", _} = changeset.errors[:content]

      assert {:error, changeset} = Message.new_user([123, "ABC"])
      assert {"must be text or a list of UserContentParts", _} = changeset.errors[:content]

      assert {:error, changeset} = Message.new_user([UserContentPart.text!("CCC"), "invalid"])
      assert {"must be text or a list of UserContentParts", _} = changeset.errors[:content]
    end
  end

  describe "new_user!/1" do
    test "creates a user message" do
      assert %Message{role: :user} = msg = Message.new_user!("Hello!")
      assert msg.content == "Hello!"
      assert msg.status == :complete
    end

    test "requires content" do
      assert_raise LangChain.LangChainError, "content: can't be blank", fn ->
        Message.new_user!(nil)
      end
    end
  end

  describe "new_assistant/1" do
    test "creates a assistant message" do
      assert {:ok, %Message{role: :assistant} = msg} =
               Message.new_assistant(%{content: "Greetings non-AI!", status: "complete"})

      assert msg.content == "Greetings non-AI!"
      assert msg.status == :complete
    end

    test "creates a cancelled assistant message" do
      assert {:ok, %Message{role: :assistant} = msg} =
               Message.new_assistant(%{content: "Greetings ", status: :cancelled})

      assert msg.content == "Greetings "
      assert msg.status == :cancelled
    end

    test "does not require content" do
      assert {:ok, %Message{role: :assistant, content: nil}} =
               Message.new_assistant(%{content: nil})
    end

    test "creates a tool_call execution request" do
      assert {:ok, %Message{} = msg} =
               Message.new_assistant(%{
                 tool_calls: [
                   ToolCall.new!(%{
                     call_id: "call_abc123",
                     name: "my_fun",
                     arguments: Jason.encode!(%{name: "Tim", age: 40})
                   })
                 ]
               })

      [call] = msg.tool_calls
      assert call.name == "my_fun"
      assert call.arguments == %{"name" => "Tim", "age" => 40}
      assert msg.role == :assistant
      assert msg.content == nil
      assert msg.status == :complete

      assert Message.is_tool_call?(msg)
    end
  end

  describe "new_assistant!/1" do
    test "creates a assistant message" do
      assert %Message{role: :assistant} = msg = Message.new_assistant!(%{content: "Hello!"})
      assert msg.content == "Hello!"
    end
  end

   describe "new_tool/1" do
    test "creates a tool response message" do
      assert {:ok, %Message{role: :tool} = msg} = Message.new_tool("my_fun", "APP ANSWER")
      assert msg.tool_call_id == "my_fun"
      assert msg.content == "APP ANSWER"
      assert msg.is_error == false
    end

    test "flags message as is_error true when option passed" do
      assert {:ok, %Message{role: :tool} = msg} = Message.new_tool("my_fun", "STUFF BROKE!", is_error: true)
      assert msg.tool_call_id == "my_fun"
      assert msg.content == "STUFF BROKE!"
      assert msg.is_error == true
    end
  end

  describe "new_tool!/1" do
    test "creates a function response message" do
      assert %Message{role: :tool} = msg = Message.new_tool!("test_fun", "RESULT")
      assert msg.tool_call_id == "test_fun"
      assert msg.content == "RESULT"
    end
  end
end
