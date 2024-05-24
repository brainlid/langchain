defmodule ChatModels.ChatOllamaAITest do
  use LangChain.BaseCase

  doctest LangChain.ChatModels.ChatOllamaAI
  alias LangChain.ChatModels.ChatOllamaAI

  describe "new/1" do
    test "works with minimal attributes" do
      assert {:ok, %ChatOllamaAI{} = ollama_ai} = ChatOllamaAI.new(%{"model" => "llama2:latest"})
      assert ollama_ai.model == "llama2:latest"
      assert ollama_ai.endpoint == "http://localhost:11434/api/chat"
    end

    test "returns errors given invalid attributes" do
      assert {:error, changeset} =
               ChatOllamaAI.new(%{"model" => nil, "temperature" => 4.4, "mirostat_eta" => 4.4})

      refute changeset.valid?
      assert {"can't be blank", _} = changeset.errors[:model]
      assert {"must be less than or equal to %{number}", _} = changeset.errors[:temperature]
      assert {"must be less than or equal to %{number}", _} = changeset.errors[:mirostat_eta]
    end

    test "supports overriding the API endpoint" do
      override_url = "http://localhost:99999/api/chat"

      model =
        ChatOllamaAI.new!(%{
          endpoint: override_url
        })

      assert model.endpoint == override_url
    end
  end

  describe "for_api/3" do
    setup do
      {:ok, ollama_ai} =
        ChatOllamaAI.new(%{
          "model" => "llama2:latest",
          "temperature" => 0.4,
          "stream" => false,
          "seed" => 0,
          "num_ctx" => 2048,
          "num_predict" => 128,
          "repeat_last_n" => 64,
          "repeat_penalty" => 1.1,
          "mirostat" => 0,
          "mirostat_eta" => 0.1,
          "mirostat_tau" => 5.0,
          "num_gqa" => 8,
          "num_gpu" => 1,
          "num_thread" => 0,
          "receive_timeout" => 300_000,
          "stop" => "",
          "tfs_z" => 0.0,
          "top_k" => 0,
          "top_p" => 0.0
        })

      %{ollama_ai: ollama_ai}
    end

    test "generates a map for an API call with no messages", %{ollama_ai: ollama_ai} do
      data = ChatOllamaAI.for_api(ollama_ai, [], [])
      assert data.model == "llama2:latest"
      assert data.temperature == 0.4
      assert data.stream == false
      assert data.messages == []
      assert data.seed == 0
      assert data.num_ctx == 2048
      assert data.num_predict == 128
      assert data.repeat_last_n == 64
      assert data.repeat_penalty == 1.1
      assert data.mirostat == 0
      assert data.mirostat_eta == 0.1
      assert data.mirostat_tau == 5.0
      assert data.num_gqa == 8
      assert data.num_gpu == 1
      assert data.num_thread == 0
      assert data.receive_timeout == 300_000
      # TODO: figure out why this is field is is being cast to nil instead of empty string
      assert data.stop == nil
      assert data.tfs_z == 0.0
      assert data.top_k == 0
      assert data.top_p == 0.0
    end

    test "generates a map for an API call with a single message", %{ollama_ai: ollama_ai} do
      user_message = "What color is the sky?"

      data = ChatOllamaAI.for_api(ollama_ai, [Message.new_user!(user_message)], [])
      assert data.model == "llama2:latest"
      assert data.temperature == 0.4

      assert [%{"content" => "What color is the sky?", "role" => :user}] = data.messages
    end

    test "generates a map for an API call with user and system messages", %{ollama_ai: ollama_ai} do
      user_message = "What color is the sky?"
      system_message = "You are a weather man"

      data =
        ChatOllamaAI.for_api(
          ollama_ai,
          [Message.new_system!(system_message), Message.new_user!(user_message)],
          []
        )

      assert data.model == "llama2:latest"
      assert data.temperature == 0.4

      assert [
               %{"role" => :system} = system_msg,
               %{"role" => :user} = user_msg
             ] = data.messages

      assert system_msg["content"] == "You are a weather man"
      assert user_msg["content"] == "What color is the sky?"
    end
  end

  describe "call/2" do
    @tag live_call: true, live_ollama_ai: true
    test "basic content example with no streaming" do
      {:ok, chat} =
        ChatOllamaAI.new(%{
          model: "llama2:latest",
          temperature: 1,
          seed: 0,
          stream: false
        })

      {:ok, %Message{role: :assistant, content: response}} =
        ChatOllamaAI.call(chat, [
          Message.new_user!("Return the response 'Colorful Threads'.")
        ])

      assert response =~ "Colorful Threads"
    end

    @tag live_call: true, live_ollama_ai: true
    test "basic content example with streaming" do
      {:ok, chat} =
        ChatOllamaAI.new(%{
          model: "llama2:latest",
          temperature: 1,
          seed: 0,
          stream: true
        })

      result =
        ChatOllamaAI.call(chat, [
          Message.new_user!("Return the response 'Colorful Threads'.")
        ])

      assert {:ok, deltas} = result
      assert length(deltas) > 0

      deltas_except_last = Enum.slice(deltas, 0..-2//-1)

      for delta <- deltas_except_last do
        assert delta.__struct__ == LangChain.MessageDelta
        assert is_binary(delta.content)
        assert delta.status == :incomplete
        assert delta.role == :assistant
      end

      last_delta = Enum.at(deltas, -1)
      assert last_delta.__struct__ == LangChain.Message
      assert is_nil(last_delta.content)
      assert last_delta.status == :complete
      assert last_delta.role == :assistant
    end

    @tag live_call: true, live_ollama_ai: true
    test "returns an error when given an invalid payload" do
      invalid_model = "invalid"

      {:error, reason} =
        ChatOllamaAI.call(%ChatOllamaAI{model: invalid_model}, [
          Message.new_user!("Return the response 'Colorful Threads'.")
        ])

      assert reason == "model '#{invalid_model}' not found, try pulling it first"
    end
  end

  describe "do_process_response/1" do
    test "handles receiving a non streamed message result" do
      response = %{
        "model" => "llama2",
        "created_at" => "2024-01-15T23:02:24.087444Z",
        "message" => %{
          "role" => "assistant",
          "content" => "Greetings!"
        },
        "done" => true,
        "total_duration" => 12_323_379_834,
        "load_duration" => 6_889_264_834,
        "prompt_eval_count" => 26,
        "prompt_eval_duration" => 91_493_000,
        "eval_count" => 362,
        "eval_duration" => 5_336_241_000
      }

      assert %Message{} = struct = ChatOllamaAI.do_process_response(response)
      assert struct.role == :assistant
      assert struct.content == "Greetings!"
      assert struct.index == nil
    end

    test "handles receiving a streamed message result" do
      response = %{
        "model" => "llama2",
        "created_at" => "2024-01-15T23:02:24.087444Z",
        "message" => %{
          "role" => "assistant",
          "content" => "Gre"
        },
        "done" => false
      }

      assert %MessageDelta{} = struct = ChatOllamaAI.do_process_response(response)
      assert struct.role == :assistant
      assert struct.content == "Gre"
      assert struct.status == :incomplete
    end
  end
end
