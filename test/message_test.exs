defmodule Langchain.MessageTest do
  use ExUnit.Case
  doctest Langchain.Message
  alias Langchain.Message

  describe "new/1" do
    test "works with minimal attrs" do
      assert {:ok, %Message{} = msg} = Message.new(%{"role" => "system", "content" => "hello!"})
      assert msg.role == :system
      assert msg.content == "hello!"
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

    test "parses arguments when complete" do
      json = Jason.encode!(%{name: "Tim", age: 40})

      assert {:ok, %Message{role: :function_call} = msg} =
               Message.new(%{
                 role: :function_call,
                 function_name: "my_fun",
                 complete: true,
                 arguments: json
               })

      assert msg.role == :function_call
      assert msg.function_name == "my_fun"
      assert msg.arguments == %{"name" => "Tim", "age" => 40}
      assert msg.complete
      assert msg.content == nil
    end

    test "adds error to arguments when message not complete" do
      json = Jason.encode!(%{name: "Tim", age: 40})

      assert {:error, changeset} =
               Message.new(%{
                 role: :function_call,
                 complete: false,
                 function_name: "my_fun",
                 arguments: json
               })

      refute changeset.valid?

      assert {"cannot parse function arguments on incomplete message", _} =
               changeset.errors[:arguments]
    end

    test "adds error to arguments when valid JSON but not a map" do
      json = Jason.encode!([true, 1, 2, 3])

      assert {:error, changeset} =
               Message.new(%{
                 role: :function_call,
                 complete: true,
                 function_name: "my_fun",
                 arguments: json
               })

      refute changeset.valid?
      assert {"unexpected JSON arguments format", _} = changeset.errors[:arguments]
    end

    test "adds error to arguments if it fails to parse" do
      assert {:error, changeset} =
               Message.new(%{
                 role: :function_call,
                 complete: true,
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

    test "requires function_name when role is function_call or function" do
      assert {:error, changeset} = Message.new(%{role: :function_call, function_name: nil})
      assert {"can't be blank", _} = changeset.errors[:function_name]

      assert {:error, changeset} = Message.new(%{role: :function, function_name: nil})
      assert {"can't be blank", _} = changeset.errors[:function_name]
    end

    test "function_name cannot be set when when role is neither function_call nor function" do
      assert {:error, changeset} = Message.new(%{role: :user, function_name: "test"})
      assert {"can't be set with role :user", _} = changeset.errors[:function_name]

      assert {:error, changeset} = Message.new(%{role: :system, function_name: "test"})
      assert {"can't be set with role :system", _} = changeset.errors[:function_name]

      assert {:error, changeset} = Message.new(%{role: :assistant, function_name: "test"})
      assert {"can't be set with role :assistant", _} = changeset.errors[:function_name]
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
      assert_raise Langchain.LangchainError, "content: can't be blank", fn ->
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
  end

  describe "new_user!/1" do
    test "creates a user message" do
      assert %Message{role: :user} = msg = Message.new_user!("Hello!")
      assert msg.content == "Hello!"
    end

    test "requires content" do
      assert_raise Langchain.LangchainError, "content: can't be blank", fn ->
        Message.new_user!(nil)
      end
    end
  end

  describe "new_assistant/1" do
    test "creates a assistant message" do
      assert {:ok, %Message{role: :assistant} = msg} = Message.new_assistant("Greetings non-AI!")
      assert msg.content == "Greetings non-AI!"
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
      assert {:ok, %Message{role: :function_call} = msg} =
               Message.new_function_call("my_fun", Jason.encode!(%{name: "Tim", age: 40}))

      assert msg.function_name == "my_fun"
      assert msg.arguments == %{"name" => "Tim", "age" => 40}
      assert msg.content == nil
    end

    test "requires function_name" do
      assert {:error, changeset} = Message.new_function_call("", Jason.encode!(%{}))
      assert {"can't be blank", _} = changeset.errors[:function_name]
    end

    test "returns error if JSON parsing fails" do
      assert {:error, changeset} = Message.new_function_call("", "invalid")
      assert {"Failed to parse arguments: \"invalid\"", _} = changeset.errors[:arguments]
    end
  end

  describe "new_function_call!/1" do
    test "creates a function_call execution request message" do
      assert %Message{role: :function_call} =
               msg = Message.new_function_call!("fun_name", Jason.encode!(%{name: "Herman"}))

      assert msg.function_name == "fun_name"
      assert msg.arguments == %{"name" => "Herman"}
      assert msg.content == nil
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
