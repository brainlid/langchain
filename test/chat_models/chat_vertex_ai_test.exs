defmodule ChatModels.ChatVertexAITest do
  alias LangChain.ChatModels.ChatVertexAI
  use LangChain.BaseCase

  doctest LangChain.ChatModels.ChatVertexAI
  alias LangChain.ChatModels.ChatVertexAI
  alias LangChain.Message
  alias LangChain.Message.ContentPart
  alias LangChain.Message.ToolCall
  alias LangChain.Message.ToolResult
  alias LangChain.MessageDelta
  alias LangChain.Function

  setup do
    {:ok, hello_world} =
      Function.new(%{
        name: "hello_world",
        description: "Give a hello world greeting.",
        function: fn _args, _context -> {:ok, "Hello world!"} end
      })

    %{hello_world: hello_world}
  end

  describe "new/1" do
    test "works with minimal attr" do
      assert {:ok, %ChatVertexAI{} = vertex_ai} =
               ChatVertexAI.new(%{"model" => "gemini-pro", "endpoint" => "http://localhost:1234/"})

      assert vertex_ai.model == "gemini-pro"
    end

    test "returns error when invalid" do
      assert {:error, changeset} = ChatVertexAI.new(%{"model" => nil})
      refute changeset.valid?
      assert {"can't be blank", _} = changeset.errors[:model]
    end
  end

  describe "for_api/3" do
    setup do
      {:ok, vertex_ai} =
        ChatVertexAI.new(%{
          "model" => "gemini-pro",
          "endpoint" => "http://localhost:1234/",
          "temperature" => 1.0,
          "top_p" => 1.0,
          "top_k" => 1.0
        })

      %{vertex_ai: vertex_ai}
    end

    test "generates a map for an API call", %{vertex_ai: vertex_ai} do
      data = ChatVertexAI.for_api(vertex_ai, [], [])
      assert %{"contents" => [], "generationConfig" => config} = data
      assert %{"temperature" => 1.0, "topK" => 1.0, "topP" => 1.0} = config
    end

    test "generates a map containing user and assistant messages", %{vertex_ai: vertex_ai} do
      user_message = "Hello Assistant!"
      assistant_message = "Hello User!"

      data =
        ChatVertexAI.for_api(
          vertex_ai,
          [
            Message.new_user!(user_message),
            Message.new_assistant!(assistant_message)
          ],
          []
        )

      assert %{"contents" => [msg1, msg2]} = data
      assert %{"role" => :user, "parts" => [%{"text" => ^user_message}]} = msg1
      assert %{"role" => :model, "parts" => [%{"text" => ^assistant_message}]} = msg2
    end

    test "generates a map containing function and function call messages", %{vertex_ai: vertex_ai} do
      message = "Can you do an action for me?"
      arguments = %{"args" => "data"}
      function_result = %{"result" => "data"}

      data =
        ChatVertexAI.for_api(
          vertex_ai,
          [
            Message.new_user!(message),
            Message.new_assistant!(%{
              tool_calls: [
                ToolCall.new!(%{
                  call_id: "call_123",
                  name: "userland_action",
                  arguments: Jason.encode!(arguments)
                })
              ]
            }),
            Message.new_tool_result!(%{
              tool_results: [
                ToolResult.new!(%{
                  tool_call_id: "call_123",
                  name: "userland_action",
                  content: Jason.encode!(function_result)
                })
              ]
            })
          ],
          []
        )

      assert %{"contents" => [msg1, msg2, msg3]} = data
      assert %{"role" => :user, "parts" => [%{"text" => ^message}]} = msg1
      assert %{"role" => :model, "parts" => [tool_call]} = msg2
      assert %{"role" => :function, "parts" => [tool_result]} = msg3

      assert %{
               "functionCall" => %{
                 "args" => ^arguments,
                 "name" => "userland_action"
               }
             } = tool_call

      assert %{
               "functionResponse" => %{
                 "name" => "userland_action",
                 "response" => ^function_result
               }
             } = tool_result
    end

    test "expands system messages into two", %{vertex_ai: vertex_ai} do
      message = "These are some instructions."

      data = ChatVertexAI.for_api(vertex_ai, [Message.new_system!(message)], [])

      assert %{"contents" => [msg1, msg2]} = data
      assert %{"role" => :user, "parts" => [%{"text" => ^message}]} = msg1
      assert %{"role" => :model, "parts" => [%{"text" => ""}]} = msg2
    end

    test "generates a map containing function declarations", %{
      vertex_ai: vertex_ai,
      hello_world: hello_world
    } do
      data = ChatVertexAI.for_api(vertex_ai, [], [hello_world])

      assert %{"contents" => []} = data
      assert %{"tools" => [tool_call]} = data

      assert %{
               "functionDeclarations" => [
                 %{
                   "name" => "hello_world",
                   "description" => "Give a hello world greeting.",
                   "parameters" => %{"properties" => %{}, "type" => "object"}
                 }
               ]
             } = tool_call
    end
  end

  describe "do_process_response/2" do
    test "handles receiving a message" do
      response = %{
        "candidates" => [
          %{
            "content" => %{"role" => "model", "parts" => [%{"text" => "Hello User!"}]},
            "finishReason" => "STOP",
            "index" => 0
          }
        ]
      }

      assert [%Message{} = struct] = ChatVertexAI.do_process_response(response)
      assert struct.role == :assistant
      [%ContentPart{type: :text, content: "Hello User!"}] = struct.content
      assert struct.index == 0
      assert struct.status == :complete
    end

    test "error if receiving non-text content" do
      response = %{
        "candidates" => [
          %{
            "content" => %{"role" => "bad_role", "parts" => [%{"text" => "Hello user"}]},
            "finishReason" => "STOP",
            "index" => 0
          }
        ]
      }

      assert [{:error, error_string}] = ChatVertexAI.do_process_response(response)
      assert error_string == "role: is invalid"
    end

    test "handles receiving function calls" do
      args = %{"args" => "data"}

      response = %{
        "candidates" => [
          %{
            "content" => %{
              "role" => "model",
              "parts" => [%{"functionCall" => %{"args" => args, "name" => "hello_world"}}]
            },
            "finishReason" => "STOP",
            "index" => 0
          }
        ]
      }

      assert [%Message{} = struct] = ChatVertexAI.do_process_response(response)
      assert struct.role == :assistant
      assert struct.index == 0
      [call] = struct.tool_calls
      assert call.name == "hello_world"
      assert call.arguments == args
    end

    test "handles receiving MessageDeltas as well" do
      response = %{
        "candidates" => [
          %{
            "content" => %{
              "role" => "model",
              "parts" => [%{"text" => "This is the first part of a mes"}]
            },
            "finishReason" => "STOP",
            "index" => 0
          }
        ]
      }

      assert [%MessageDelta{} = struct] = ChatVertexAI.do_process_response(response, MessageDelta)
      assert struct.role == :assistant
      assert struct.content == "This is the first part of a mes"
      assert struct.index == 0
      assert struct.status == :incomplete
    end

    test "handles API error messages" do
      response = %{
        "error" => %{
          "code" => 400,
          "message" => "Invalid request",
          "status" => "INVALID_ARGUMENT"
        }
      }

      assert {:error, error_string} = ChatVertexAI.do_process_response(response)
      assert error_string == "Invalid request"
    end

    test "handles Jason.DecodeError" do
      response = {:error, %Jason.DecodeError{}}

      assert {:error, error_string} = ChatVertexAI.do_process_response(response)
      assert "Received invalid JSON:" <> _ = error_string
    end

    test "handles unexpected response with error" do
      response = %{}
      assert {:error, "Unexpected response"} = ChatVertexAI.do_process_response(response)
    end
  end

  describe "filter_parts_for_types/2" do
    test "returns a single functionCall type" do
      parts = [
        %{"text" => "I think I'll call this function."},
        %{
          "functionCall" => %{
            "args" => %{"args" => "data"},
            "name" => "userland_action"
          }
        }
      ]

      assert [%{"text" => _}] = ChatVertexAI.filter_parts_for_types(parts, ["text"])

      assert [%{"functionCall" => _}] =
               ChatVertexAI.filter_parts_for_types(parts, ["functionCall"])
    end

    test "returns a set of types" do
      parts = [
        %{"text" => "I think I'll call this function."},
        %{
          "functionCall" => %{
            "args" => %{"args" => "data"},
            "name" => "userland_action"
          }
        }
      ]

      assert parts == ChatVertexAI.filter_parts_for_types(parts, ["text", "functionCall"])
    end
  end

  describe "get_message_contents/1" do
    test "returns basic text as a ContentPart" do
      message = Message.new_user!("Howdy!")

      result = ChatVertexAI.get_message_contents(message)

      assert result == [%{"text" => "Howdy!"}]
    end

    test "supports a list of ContentParts" do
      message =
        Message.new_user!([
          ContentPart.new!(%{type: :text, content: "Hello!"}),
          ContentPart.new!(%{type: :text, content: "What's up?"})
        ])

      result = ChatVertexAI.get_message_contents(message)

      assert result == [
               %{"text" => "Hello!"},
               %{"text" => "What's up?"}
             ]
    end
  end

  describe "serialize_config/2" do
    test "does not include the API key or callbacks" do
      model = ChatVertexAI.new!(%{model: "gemini-pro", endpoint: "http://localhost:1234/"})
      result = ChatVertexAI.serialize_config(model)
      assert result["version"] == 1
      refute Map.has_key?(result, "api_key")
      refute Map.has_key?(result, "callbacks")
    end

    test "creates expected map" do
      model =
        ChatVertexAI.new!(%{
          model: "gemini-pro",
          endpoint: "http://localhost:1234/"
        })

      result = ChatVertexAI.serialize_config(model)

      assert result == %{
               "endpoint" => "http://localhost:1234/",
               "model" => "gemini-pro",
               "module" => "Elixir.LangChain.ChatModels.ChatVertexAI",
               "receive_timeout" => 60000,
               "stream" => false,
               "temperature" => 0.9,
               "top_k" => 1.0,
               "top_p" => 1.0,
               "version" => 1,
               "json_response" => false
             }
    end
  end
end
