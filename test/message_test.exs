defmodule LangChain.MessageTest do
  use ExUnit.Case
  doctest LangChain.Message
  alias LangChain.Message
  alias LangChain.Message.ToolCall
  alias LangChain.Message.ToolResult
  alias LangChain.Message.ContentPart
  alias LangChain.PromptTemplate
  alias LangChain.LangChainError

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

    test "requires content to be text or ContentParts when a user message" do
      # can be a content part
      part = ContentPart.text!("Hi")
      {:ok, message} = Message.new_user([part])
      assert message.content == [part]

      # can be a string
      {:ok, message} = Message.new_user("Hi")
      assert message.content == "Hi"

      # content parts not allowed for other role types
      {:error, changeset} = Message.new_system([part])
      assert {"is invalid for role system", _} = changeset.errors[:content]

      {:error, changeset} =
        Message.new(%{
          role: :tool,
          tool_results: [ToolResult.new!(%{tool_call_id: "call_123", content: "woof"})],
          content: [part]
        })

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

    test "accepts list of ContentParts for content" do
      assert {:ok, %Message{} = msg} =
               Message.new_user([
                 ContentPart.text!("Describe what is in this image:"),
                 ContentPart.image!(:base64.encode("fake_image_data"))
               ])

      assert msg.role == :user

      assert msg.content == [
               %ContentPart{type: :text, content: "Describe what is in this image:"},
               %ContentPart{type: :image, content: "ZmFrZV9pbWFnZV9kYXRh", options: []}
             ]
    end

    test "accepts PromptTemplates in content list" do
      assert {:ok, %Message{} = msg} =
               Message.new_user([
                 PromptTemplate.from_template!(
                   "My name is <%= @name %> and here's a picture of me:"
                 ),
                 ContentPart.image!(:base64.encode("fake_image_data"))
               ])

      assert msg.role == :user

      assert msg.content == [
               %PromptTemplate{text: "My name is <%= @name %> and here's a picture of me:"},
               %ContentPart{type: :image, content: "ZmFrZV9pbWFnZV9kYXRh", options: []}
             ]
    end

    test "does not accept invalid contents" do
      assert {:error, changeset} = Message.new_user(123)
      assert {"must be text or a list of ContentParts", _} = changeset.errors[:content]

      assert {:error, changeset} = Message.new_user([123, "ABC"])
      assert {"must be text or a list of ContentParts", _} = changeset.errors[:content]

      assert {:error, changeset} = Message.new_user([ContentPart.text!("CCC"), "invalid"])
      assert {"must be text or a list of ContentParts", _} = changeset.errors[:content]
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

  describe "new_tool_result/1" do
    test "creates a tool response message" do
      result =
        ToolResult.new!(%{
          tool_call_id: "call_123",
          content: "STUFF_BROKE!",
          is_error: true
        })

      {:ok, %Message{} = msg} = Message.new_tool_result(%{tool_results: [result]})

      assert msg.role == :tool
      assert [result] == msg.tool_results
    end
  end

  describe "new_tool_result!/1" do
    test "creates a tool response message" do
      result = ToolResult.new!(%{tool_call_id: "call_123", content: "RESULT"})

      %Message{} = msg = Message.new_tool_result!(%{tool_results: [result]})

      assert msg.role == :tool
      assert [result] == msg.tool_results
    end
  end

  describe "append_tool_result/2" do
    test "appends a ToolResult to a tool message" do
      result1 =
        ToolResult.new!(%{
          tool_call_id: "call_123",
          name: "hello_world",
          content: "Hello world!"
        })

      result2 =
        ToolResult.new!(%{
          tool_call_id: "call_234",
          name: "hello_world",
          content: "Hello world! x2"
        })

      message =
        %{tool_results: [result1]}
        |> Message.new_tool_result!()
        |> Message.append_tool_result(result2)

      assert [result1, result2] == message.tool_results
    end

    test "raises error adding ToolResult to other roles" do
      user_message = Message.new_user!("Hi")

      result =
        ToolResult.new!(%{
          tool_call_id: "call_123",
          name: "hello_world",
          content: "Hello world!"
        })

      assert_raise LangChainError, "Can only append tool results to a tool role message.", fn ->
        Message.append_tool_result(user_message, result)
      end
    end
  end

  describe "is_tool_call?/1" do
    test "returns true when a tool call" do
      msg =
        Message.new_assistant!(%{
          tool_calls: [
            ToolCall.new!(%{
              call_id: "call_abc123",
              name: "my_fun",
              arguments: nil
            })
          ]
        })

      assert Message.is_tool_call?(msg)
    end

    test "returns false when not" do
      refute Message.is_tool_call?(Message.new_assistant!(%{content: "Howdy"}))
    end
  end

  describe "is_tool_related?/1" do
    test "returns true when a tool call" do
      msg =
        Message.new_assistant!(%{
          tool_calls: [
            ToolCall.new!(%{
              call_id: "call_abc123",
              name: "my_fun",
              arguments: nil
            })
          ]
        })

      assert Message.is_tool_related?(msg)
    end

    test "returns true when a tool result" do
      msg =
        Message.new_tool_result!(%{
          tool_results: [
            ToolResult.new!(%{
              tool_call_id: "call_123",
              name: "hello_world",
              content: "Hello world!"
            })
          ]
        })

      assert Message.is_tool_related?(msg)
    end

    test "returns false when regular message" do
      refute Message.is_tool_related?(Message.new_user!("Hi"))
      refute Message.is_tool_related?(Message.new_assistant!(%{content: "Hello"}))
    end
  end

  describe "tool_had_errors?/1" do
    test "returns true when any tool result had an error" do
      msg =
        Message.new_tool_result!(%{
          tool_results: [
            ToolResult.new!(%{
              tool_call_id: "call_123",
              name: "hello_world",
              content: "ERROR!",
              is_error: true
            })
          ]
        })

      assert Message.tool_had_errors?(msg)

      msg =
        Message.new_tool_result!(%{
          tool_results: [
            ToolResult.new!(%{
              tool_call_id: "call_123",
              name: "hello_world",
              content: "Hello world!"
            }),
            ToolResult.new!(%{
              tool_call_id: "call_123",
              name: "hello_world",
              content: "ERROR!",
              is_error: true
            })
          ]
        })

      assert Message.tool_had_errors?(msg)
    end

    test "returns false when all tool results succeeded" do
      msg =
        Message.new_tool_result!(%{
          tool_results: [
            ToolResult.new!(%{
              tool_call_id: "call_123",
              name: "hello_world",
              content: "Hello world!"
            })
          ]
        })

      refute Message.tool_had_errors?(msg)
    end

    test "returns false when not a tool response" do
      refute Message.tool_had_errors?(Message.new_assistant!(%{content: "Howdy"}))
    end
  end
end
