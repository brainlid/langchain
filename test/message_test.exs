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
                   tool_id: "call_abc123",
                   name: "my_fun",
                   arguments: json
                 ],
                 status: :complete
               })

      refute changeset.valid?
      assert {"unexpected JSON arguments format", _} = changeset.errors[:arguments]
    end

    test "adds error to arguments if it fails to parse" do
      assert {:error, changeset} =
               Message.new(%{
                 role: :assistant,
                 function_name: "my_fun",
                 arguments: "invalid"
               })

      refute changeset.valid?
      assert {"invalid JSON function arguments", _} = changeset.errors[:arguments]
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

      {:error, changeset} = Message.new(%{role: :tool, tool_id: "tool_123", content: [part]})
      assert {"is invalid for role tool", _} = changeset.errors[:content]
    end

    test "requires tool_call to be present when no content" do
      {:error, changeset} =
        Message.new_assistant(%{
          content: nil,
          tool_calls: []
        })

      assert {"is required when no tool_calls", _} = changeset.errors[:content]
    end

    test "content can be nil when an assistant message (tool calls)" do
      tool_call = ToolCall.new!(%{tool_id: "1", name: "hello"})

      {:ok, message} =
        Message.new_assistant(%{
          content: nil,
          tool_calls: [tool_call]
        })

      assert message.content == nil
      assert message.tool_calls == [tool_call]
    end

    test "requires function_name when role is function" do
      assert {:error, changeset} = Message.new(%{role: :function, function_name: nil})
      assert {"can't be blank", _} = changeset.errors[:function_name]
    end

    test "function_name cannot be set when when role is not function" do
      assert {:error, changeset} = Message.new(%{role: :user, function_name: "test"})
      assert {"can't be set with role :user", _} = changeset.errors[:function_name]

      assert {:error, changeset} = Message.new(%{role: :system, function_name: "test"})
      assert {"can't be set with role :system", _} = changeset.errors[:function_name]
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
               %UserContentPart{type: :image, content: "ZmFrZV9pbWFnZV9kYXRh"}
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

    test "accepts list of UserContentParts for content" do
      assert msg =
               Message.new_user!([
                 UserContentPart.text!("Describe what is in this image:"),
                 UserContentPart.image!(:base64.encode("fake_image_data"))
               ])

      assert msg.role == :user

      assert msg.content == [
               %UserContentPart{type: :text, content: "Describe what is in this image:"},
               %UserContentPart{type: :image, content: "ZmFrZV9pbWFnZV9kYXRh"}
             ]
    end
  end

  describe "new_assistant/1" do
    test "creates a assistant message" do
      assert {:ok, %Message{role: :assistant} = msg} =
               Message.new_assistant("Greetings non-AI!", "complete")

      assert msg.content == "Greetings non-AI!"
      assert msg.status == :complete
    end

    test "creates a cancelled assistant message" do
      assert {:ok, %Message{role: :assistant} = msg} =
               Message.new_assistant("Greetings ", :cancelled)

      assert msg.content == "Greetings "
      assert msg.status == :cancelled
    end

    test "does not require content" do
      assert {:ok, %Message{role: :assistant, content: nil}} = Message.new_assistant(nil)
    end
  end

  describe "new_assistant!/1" do
    test "creates a assistant message" do
      assert %Message{role: :assistant} = msg = Message.new_assistant!("Hello!")
      assert msg.content == "Hello!"
    end
  end

  describe "new_function_call/1" do
    test "creates a function_call execution request message" do
      assert {:ok, %Message{role: :assistant} = msg} =
               Message.new_function_call("my_fun", Jason.encode!(%{name: "Tim", age: 40}))

      assert msg.function_name == "my_fun"
      assert msg.arguments == %{"name" => "Tim", "age" => 40}
      assert msg.content == nil
      assert Message.is_function_call?(msg)
    end

    test "requires function_name" do
      assert {:error, changeset} = Message.new_function_call("", Jason.encode!(%{}))
      assert {"is required when arguments are given", _} = changeset.errors[:function_name]
    end

    test "returns error if JSON parsing fails" do
      assert {:error, changeset} = Message.new_function_call("", "invalid")
      assert {"Failed to parse arguments: \"invalid\"", _} = changeset.errors[:arguments]
    end
  end

  describe "new_function_call!/1" do
    test "creates a function_call execution request message" do
      assert %Message{role: :assistant} =
               msg = Message.new_function_call!("fun_name", Jason.encode!(%{name: "Herman"}))

      assert msg.function_name == "fun_name"
      assert msg.arguments == %{"name" => "Herman"}
      assert msg.content == nil
      assert Message.is_function_call?(msg)
    end
  end

  describe "new_function/1" do
    test "creates a function response message" do
      assert {:ok, %Message{role: :function} = msg} = Message.new_function("my_fun", "APP ANSWER")
      assert msg.function_name == "my_fun"
      assert msg.content == "APP ANSWER"
    end

    test "requires function_name" do
      assert {:error, changeset} = Message.new_function("", "answer")
      assert {"can't be blank", _} = changeset.errors[:function_name]
    end
  end

  describe "new_function!/1" do
    test "creates a function response message" do
      assert %Message{role: :function} = msg = Message.new_function!("test_fun", "RESULT")
      assert msg.function_name == "test_fun"
      assert msg.content == "RESULT"
    end
  end
end
