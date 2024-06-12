defmodule LangChain.ChatModels.ChatBumblebeeTest do
  use LangChain.BaseCase
  import LangChain.Utils.ApiOverride

  doctest LangChain.ChatModels.ChatBumblebee
  alias LangChain.ChatModels.ChatBumblebee
  alias LangChain.Message
  alias LangChain.TokenUsage

  describe "new/1" do
    test "works with minimal attr" do
      assert {:ok, %ChatBumblebee{} = model} = ChatBumblebee.new(%{"serving" => SomeModule})
      assert model.serving == SomeModule
      assert model.stream == true
      assert model.seed == nil
    end

    test "returns error when invalid" do
      assert {:error, changeset} = ChatBumblebee.new(%{"serving" => nil})
      refute changeset.valid?
      assert {"can't be blank", _} = changeset.errors[:serving]
    end
  end

  describe "call/4" do
    test "supports API override" do
      set_api_override({:ok, [Message.new_assistant!("Colorful Threads")], :on_llm_new_message})

      # https://js.langchain.com/docs/modules/models/chat/
      {:ok, chat} = ChatBumblebee.new(%{serving: Fake})

      {:ok, [%Message{role: :assistant, content: response}]} =
        ChatBumblebee.call(chat, [
          Message.new_user!("Return the response 'Colorful Threads'.")
        ])

      assert response =~ "Colorful Threads"
    end
  end

  describe "do_process_response/3" do
    setup do
      handler = %{
        on_llm_new_delta: fn _model, delta ->
          send(self(), {:callback_delta, delta})
        end,
        on_llm_new_message: fn _model, message ->
          send(self(), {:callback_message, message})
        end,
        on_llm_token_usage: fn _model, usage ->
          send(self(), {:callback_usage, usage})
        end,
      }

      %{handler: handler}
    end

    test "handles non-streamed full text response", %{handler: handler} do
      model = ChatBumblebee.new!(%{serving: Fake, stream: false, callbacks: [handler]})

      response = %{
        results: [
          %{
            text: "Hello.",
            token_summary: %{input: 38, output: 4, padding: 4058}
          }
        ]
      }

      expected_message = Message.new_assistant!(%{content: "Hello.", status: :complete})

      [message] = ChatBumblebee.do_process_response(response, model)
      assert message == expected_message

      assert_received {:callback_message, data}
      assert data == expected_message
    end

    test "handles stream when stream: false", %{handler: handler} do
      model = ChatBumblebee.new!(%{serving: Fake, stream: false, callbacks: [handler]})

      expected_message = Message.new_assistant!(%{content: "Hello.", status: :complete})

      stream =
        ["He", "ll", "o", ".", {:done, %{input: 38, output: 4, padding: 4058}}]
        |> Stream.map(& &1)

      [message] = ChatBumblebee.do_process_response(stream, model)
      assert message == expected_message

      assert_received {:callback_message, data}
      assert data == expected_message
    end

    test "handles a stream when stream: false and no stream_done requested", %{
      handler: handler
    } do
      model = ChatBumblebee.new!(%{serving: Fake, stream: false, callbacks: [handler]})

      expected_message = Message.new_assistant!(%{content: "Hello.", status: :complete})

      stream =
        ["He", "ll", "o", "."]
        |> Stream.map(& &1)

      [message] = ChatBumblebee.do_process_response(stream, model)
      assert message == expected_message

      assert_received {:callback_message, data}
      assert data == expected_message
    end

    test "handles a stream when stream: true and no stream_done requested", %{
      handler: handler
    } do
      model = ChatBumblebee.new!(%{serving: Fake, stream: true, callbacks: [handler]})

      expected_deltas = [
        MessageDelta.new!(%{content: "Hel", status: :incomplete, role: :assistant}),
        MessageDelta.new!(%{content: "lo.", status: :incomplete, role: :assistant})
      ]

      stream =
        ["Hel", "lo."]
        |> Stream.map(& &1)

      assert [deltas] = ChatBumblebee.do_process_response(stream, model)
      assert deltas == expected_deltas

      assert_received {:callback_delta, data_1}
      assert data_1 == MessageDelta.new!(%{content: "Hel", role: :assistant, status: :incomplete})

      assert_received {:callback_delta, data_2}
      assert data_2 == MessageDelta.new!(%{content: "lo.", role: :assistant, status: :incomplete})

      refute_received {:callback_delta, _data_3}
    end

    test "handles stream when stream: true", %{handler: handler} do
      model = ChatBumblebee.new!(%{serving: Fake, stream: true, callbacks: [handler]})

      expected_deltas = [
        %MessageDelta{content: "He", status: :incomplete, role: :assistant},
        %MessageDelta{content: "ll", status: :incomplete, role: :assistant},
        %MessageDelta{content: "o", status: :incomplete, role: :assistant},
        %MessageDelta{content: ".", status: :incomplete, role: :assistant},
        %MessageDelta{content: nil, status: :complete, role: :assistant}
      ]

      stream =
        ["He", "ll", "o", ".", {:done, %{input: 38, output: 4, padding: 4058}}]
        |> Stream.map(& &1)

      # expect the deltas to be in an outer
      assert [deltas] = ChatBumblebee.do_process_response(stream, model)
      assert deltas == expected_deltas

      assert_received {:callback_delta, data_1}
      assert data_1 == MessageDelta.new!(%{content: "He", role: :assistant, status: :incomplete})

      assert_received {:callback_delta, data_2}
      assert data_2 == MessageDelta.new!(%{content: "ll", role: :assistant, status: :incomplete})

      assert_received {:callback_delta, data_3}
      assert data_3 == MessageDelta.new!(%{content: "o", role: :assistant, status: :incomplete})

      assert_received {:callback_delta, data_4}
      assert data_4 == MessageDelta.new!(%{content: ".", role: :assistant, status: :incomplete})

      assert_received {:callback_delta, data_5}
      assert data_5 == MessageDelta.new!(%{role: :assistant, status: :complete})

      assert_received {:callback_usage, usage}
      assert %TokenUsage{input: 38, output: 4} = usage
    end
  end
end
