defmodule ChatModels.ChatGoogleAITest do
  alias LangChain.ChatModels.ChatGoogleAI
  use LangChain.BaseCase

  doctest LangChain.ChatModels.ChatGoogleAI
  alias LangChain.ChatModels.ChatGoogleAI
  alias LangChain.Message
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
      assert {:ok, %ChatGoogleAI{} = google_ai} = ChatGoogleAI.new(%{"model" => "gemini-pro"})
      assert google_ai.model == "gemini-pro"
    end

    test "returns error when invalid" do
      assert {:error, changeset} = ChatGoogleAI.new(%{"model" => nil})
      refute changeset.valid?
      assert {"can't be blank", _} = changeset.errors[:model]
    end

    test "supports overriding the API endpoint" do
      override_url = "http://localhost:1234/"

      model =
        ChatGoogleAI.new!(%{
          endpoint: override_url
        })

      assert model.endpoint == override_url
    end

    test "supports overriding the API version" do
      version = "v1"

      model =
        ChatGoogleAI.new!(%{
          version: version
        })

      assert model.version == version
    end
  end

  describe "for_api/3" do
    setup do
      {:ok, google_ai} =
        ChatGoogleAI.new(%{
          "model" => "gemini-pro",
          "temperature" => 1.0,
          "top_p" => 1.0,
          "top_k" => 1.0
        })

      %{google_ai: google_ai}
    end

    test "generates a map for an API call", %{google_ai: google_ai} do
      data = ChatGoogleAI.for_api(google_ai, [], [])
      assert data["contents"] == []
      assert data["generationConfig"]["temperature"] == 1.0
      assert data["generationConfig"]["topP"] == 1.0
      assert data["generationConfig"]["topK"] == 1.0
    end

    test "generates a map containing user and assistant messages", %{google_ai: google_ai} do
      user_message = "Hello Assistant!"
      assistant_message = "Hello User!"

      data =
        ChatGoogleAI.for_api(
          google_ai,
          [Message.new_user!(user_message), Message.new_assistant!(assistant_message)],
          []
        )

      assert get_in(data, ["contents", Access.at(0), "role"]) == :user

      assert get_in(data, ["contents", Access.at(0), "parts", Access.at(0), "text"]) ==
               user_message

      assert get_in(data, ["contents", Access.at(1), "role"]) == :model

      assert get_in(data, ["contents", Access.at(1), "parts", Access.at(0), "text"]) ==
               assistant_message
    end

    test "generates a map containing function and function call messages", %{google_ai: google_ai} do
      message = "Can you do an action for me?"
      arguments = %{"args" => "data"}
      function_result = %{"result" => "data"}

      data =
        ChatGoogleAI.for_api(
          google_ai,
          [
            Message.new_user!(message),
            Message.new_function_call!("userland_action", Jason.encode!(arguments)),
            Message.new_function!("userland_action", function_result)
          ],
          []
        )

      assert get_in(data, ["contents", Access.at(0), "role"]) == :user

      assert get_in(data, ["contents", Access.at(0), "parts", Access.at(0), "text"]) ==
               message

      assert get_in(data, ["contents", Access.at(1), "role"]) == :model

      assert get_in(data, [
               "contents",
               Access.at(1),
               "parts",
               Access.at(0),
               "functionCall",
               "name"
             ]) == "userland_action"

      assert get_in(data, [
               "contents",
               Access.at(1),
               "parts",
               Access.at(0),
               "functionCall",
               "args"
             ]) == arguments

      assert get_in(data, ["contents", Access.at(2), "role"]) == :function

      assert get_in(data, [
               "contents",
               Access.at(2),
               "parts",
               Access.at(0),
               "functionResponse",
               "name"
             ]) == "userland_action"

      assert get_in(data, [
               "contents",
               Access.at(2),
               "parts",
               Access.at(0),
               "functionResponse",
               "response"
             ]) == function_result
    end

    test "expands system messages into two", %{google_ai: google_ai} do
      message = "These are some instructions."

      data = ChatGoogleAI.for_api(google_ai, [Message.new_system!(message)], [])
      assert get_in(data, ["contents", Access.at(0), "role"]) == :user

      assert get_in(data, ["contents", Access.at(0), "parts", Access.at(0), "text"]) ==
               message

      assert get_in(data, ["contents", Access.at(1), "role"]) == :model

      assert get_in(data, ["contents", Access.at(1), "parts", Access.at(0), "text"]) == ""
    end

    test "generates a map containing function declarations", %{
      google_ai: google_ai,
      hello_world: hello_world
    } do
      data = ChatGoogleAI.for_api(google_ai, [], [hello_world])
      assert data["contents"] == []

      assert get_in(data, [
               "tools",
               Access.at(0),
               "functionDeclarations",
               Access.at(0),
               "name"
             ]) ==
               "hello_world"

      assert get_in(data, [
               "tools",
               Access.at(0),
               "functionDeclarations",
               Access.at(0),
               "description"
             ]) ==
               "Give a hello world greeting."
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

      assert [%Message{} = struct] = ChatGoogleAI.do_process_response(response)
      assert struct.role == :assistant
      assert struct.content == "Hello User!"
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

      assert [{:error, error_string}] = ChatGoogleAI.do_process_response(response)
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

      assert [%Message{} = struct] = ChatGoogleAI.do_process_response(response)
      assert struct.role == :assistant
      assert struct.index == 0
      assert struct.function_name == "hello_world"
      assert struct.arguments == args
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

      assert [%MessageDelta{} = struct] = ChatGoogleAI.do_process_response(response, MessageDelta)
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

      assert {:error, error_string} = ChatGoogleAI.do_process_response(response)
      assert error_string == "Invalid request"
    end

    test "handles Jason.DecodeError" do
      response = {:error, %Jason.DecodeError{}}

      assert {:error, error_string} = ChatGoogleAI.do_process_response(response)
      assert "Received invalid JSON:" <> _ = error_string
    end

    test "handles unexpected response with error" do
      response = %{}
      assert {:error, "Unexpected response"} = ChatGoogleAI.do_process_response(response)
    end
  end
end
