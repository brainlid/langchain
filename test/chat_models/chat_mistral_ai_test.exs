defmodule LangChain.ChatModels.ChatMistralAITest do
  alias Langchain.ChatModels.ChatMistralAI
  use LangChain.BaseCase

  alias LangChain.Message
  alias LangChain.MessageDelta

  setup do
    model = ChatMistralAI.new!(%{"model" => "mistral-tiny"})
    %{model: model}
  end

  describe "new/1" do
    test "works with minimal attr" do
      assert {:ok, %ChatMistralAI{} = mistral_ai} =
               ChatMistralAI.new(%{"model" => "mistral-tiny"})

      assert mistral_ai.model == "mistral-tiny"
    end

    test "returns error when invalid" do
      assert {:error, changeset} = ChatMistralAI.new(%{"model" => nil})
      refute changeset.valid?
      assert {"can't be blank", _} = changeset.errors[:model]
    end

    test "supports overriding the API endpoint" do
      override_url = "http://localhost:1234/"

      model =
        ChatMistralAI.new!(%{
          "model" => "mistral-tiny",
          "endpoint" => override_url
        })

      assert model.endpoint == override_url
    end
  end

  describe "for_api/3" do
    setup do
      {:ok, mistral_ai} =
        ChatMistralAI.new(%{
          model: "mistral-tiny",
          temperature: 1.0,
          top_p: 1.0,
          max_tokens: 100,
          safe_prompt: true,
          random_seed: 42
        })

      %{mistral_ai: mistral_ai}
    end

    test "generates a map for an API call", %{mistral_ai: mistral_ai} do
      data = ChatMistralAI.for_api(mistral_ai, [], [])

      assert data ==
               %{
                 model: "mistral-tiny",
                 temperature: 1.0,
                 top_p: 1.0,
                 messages: [],
                 stream: false,
                 max_tokens: 100,
                 safe_prompt: true,
                 random_seed: 42
               }
    end

    test "generates a map containing user and assistant messages", %{mistral_ai: mistral_ai} do
      user_message = "Hello Assistant!"
      assistant_message = "Hello User!"

      data =
        ChatMistralAI.for_api(
          mistral_ai,
          [Message.new_user!(user_message), Message.new_assistant!(assistant_message)],
          []
        )

      assert get_in(data, [:messages, Access.at(0), "role"]) == :user

      assert get_in(data, [:messages, Access.at(0), "content"]) == user_message

      assert get_in(data, [:messages, Access.at(1), "role"]) == :assistant

      assert get_in(data, [:messages, Access.at(1), "content"]) == assistant_message
    end
  end

  describe "do_process_response/2" do
    test "handles receiving a message", %{model: model} do
      response = %{
        "choices" => [
          %{
            "message" => %{
              "role" => "assistant",
              "content" => "Hello User!"
            },
            "finish_reason" => "stop",
            "index" => 0
          }
        ]
      }

      assert [%Message{} = struct] = ChatMistralAI.do_process_response(model, response)
      assert struct.role == :assistant
      assert struct.content == "Hello User!"
      assert struct.index == 0
      assert struct.status == :complete
    end

    test "errors with invalid role", %{model: model} do
      response = %{
        "choices" => [
          %{
            "message" => %{
              "role" => "unknown role",
              "content" => "Hello User!"
            },
            "finish_reason" => "stop",
            "index" => 0
          }
        ]
      }

      assert [{:error, "role: is invalid"}] = ChatMistralAI.do_process_response(model, response)
    end

    test "handles receiving MessageDeltas as well", %{model: model} do
      response = %{
        "choices" => [
          %{
            "delta" => %{
              "role" => "assistant",
              "content" => "This is the first part of a mes"
            },
            "finish_reason" => nil,
            "index" => 0
          }
        ]
      }

      assert [%MessageDelta{} = struct] = ChatMistralAI.do_process_response(model, response)

      assert struct.role == :assistant
      assert struct.content == "This is the first part of a mes"
      assert struct.index == 0
      assert struct.status == :incomplete
    end

    test "handles API error messages", %{model: model} do
      response = %{
        "error" => %{
          "code" => 400,
          "message" => "Invalid request",
          "status" => "INVALID_ARGUMENT"
        }
      }

      assert {:error, error_string} = ChatMistralAI.do_process_response(model, response)
      assert error_string == "Invalid request"
    end

    test "handles Jason.DecodeError", %{model: model} do
      response = {:error, %Jason.DecodeError{}}

      assert {:error, error_string} = ChatMistralAI.do_process_response(model, response)
      assert "Received invalid JSON:" <> _ = error_string
    end

    test "handles unexpected response with error", %{model: model} do
      response = %{}
      assert {:error, "Unexpected response"} = ChatMistralAI.do_process_response(model, response)
    end
  end

  describe "serialize_config/2" do
    test "does not include the API key or callbacks" do
      model = ChatMistralAI.new!(%{model: "mistral-tiny"})
      result = ChatMistralAI.serialize_config(model)
      assert result["version"] == 1
      refute Map.has_key?(result, "api_key")
      refute Map.has_key?(result, "callbacks")
    end

    test "creates expected map" do
      model =
        ChatMistralAI.new!(%{
          model: "mistral-tiny",
          temperature: 1.0,
          top_p: 1.0,
          max_tokens: 100,
          safe_prompt: true,
          random_seed: 42
        })

      result = ChatMistralAI.serialize_config(model)

      assert result == %{
               "endpoint" => "https://api.mistral.ai/v1/chat/completions",
               "model" => "mistral-tiny",
               "max_tokens" => 100,
               "module" => "Elixir.Langchain.ChatModels.ChatMistralAI",
               "receive_timeout" => 60000,
               "stream" => false,
               "temperature" => 1.0,
               "random_seed" => 42,
               "safe_prompt" => true,
               "top_p" => 1.0,
               "version" => 1
             }
    end
  end
end
