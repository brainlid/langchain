defmodule Langchain.ChatModels.ChatOpenAITest do
  use Langchain.BaseCase

  doctest Langchain.ChatModels.ChatOpenAI
  alias Langchain.ChatModels.ChatOpenAI
  alias Langchain.Functions.Function
  alias Langchain.MessageDelta

  setup do
    {:ok, hello_world} =
      Function.new(%{
        name: "hello_world",
        description: "Give a hello world greeting.",
        function: fn -> IO.puts("Hello world!") end
      })

    %{hello_world: hello_world}
  end

  describe "new/1" do
    test "works with minimal attr" do
      assert {:ok, %ChatOpenAI{} = openai} = ChatOpenAI.new(%{"model" => "gpt-3.5-turbo-0613"})
      assert openai.model == "gpt-3.5-turbo-0613"
    end

    test "returns error when invalid" do
      assert {:error, changeset} = ChatOpenAI.new(%{"model" => nil})
      refute changeset.valid?
      assert {"can't be blank", _} = changeset.errors[:model]
    end
  end

  describe "for_api/3" do
    test "generates a map for an API call" do
      {:ok, openai} =
        ChatOpenAI.new(%{
          "model" => "gpt-3.5-turbo-0613",
          "temperature" => 1,
          "frequency_penalty" => 0.5
        })

      data = ChatOpenAI.for_api(openai, [], [])
      assert data.model == "gpt-3.5-turbo-0613"
      assert data.temperature == 1
      assert data.frequency_penalty == 0.5
    end
  end

  describe "call/2" do
    @tag :live_call
    test "basic content example" do
      # set_fake_llm_response({:ok, Message.new_assistant("\n\nRainbow Sox Co.")})

      # https://js.langchain.com/docs/modules/models/chat/
      {:ok, chat} = ChatOpenAI.new(%{temperature: 1})

      {:ok, [%Message{role: :assistant, content: response}]} =
        ChatOpenAI.call(chat, [
          Message.new_user!("Return the response 'Colorful Threads'.")
        ])

      assert response == "Colorful Threads"
    end

    @tag :live_call
    test "executing a function", %{hello_world: hello_world} do
      {:ok, chat} = ChatOpenAI.new(%{verbose: true})

      {:ok, message} =
        Message.new_user("Only using the functions you have been provided with, give a greeting.")

      {:ok, [message]} = ChatOpenAI.call(chat, [message], [hello_world])

      assert %Message{role: :function_call} = message
      assert message.arguments == %{}
      assert message.content == nil
    end

    @tag :live_call
    test "executes callback function when data is streamed" do
      callback = fn %MessageDelta{} = delta ->
        send(self(), {:message_delta, delta})
      end

      # https://js.langchain.com/docs/modules/models/chat/
      {:ok, chat} = ChatOpenAI.new(%{temperature: 1, stream: true, callback_fn: callback})

      {:ok, _post_results} =
        ChatOpenAI.call(chat, [
          Message.new_user!("Return the response 'Hi'.")
        ])

      # we expect to receive the response over 3 delta messages
      assert_receive {:message_delta, delta_1}, 500
      assert_receive {:message_delta, delta_2}, 500
      assert_receive {:message_delta, delta_3}, 500

      # IO.inspect(delta_1)
      # IO.inspect(delta_2)
      # IO.inspect(delta_3)

      merged =
        delta_1
        |> MessageDelta.merge_delta(delta_2)
        |> MessageDelta.merge_delta(delta_3)

      assert merged.role == :assistant
      assert merged.content == "Hi"
      assert merged.complete
    end

    @tag :live_call
    test "executes callback function when data is NOT streamed" do
      callback = fn [%Message{} = new_message] ->
        send(self(), {:message_received, new_message})
      end

      # https://js.langchain.com/docs/modules/models/chat/
      # NOTE streamed. Should receive complete message.
      {:ok, chat} = ChatOpenAI.new(%{temperature: 1, stream: false, callback_fn: callback})

      {:ok, [message]} =
        ChatOpenAI.call(chat, [
          Message.new_user!("Return the response 'Hi'.")
        ])

      assert message.content == "Hi"
      assert_receive {:message_received, received_item}, 500
      assert %Message{} = received_item
      assert received_item.role == :assistant
      assert received_item.content == "Hi"
    end
  end

  describe "do_process_response/1" do
    test "handles receiving a message" do
      response = %{
        "message" => %{"role" => "assistant", "content" => "Greetings!", "index" => 1},
        "finish_reason" => "stop"
      }

      assert %Message{} = struct = ChatOpenAI.do_process_response(response)
      assert struct.role == :assistant
      assert struct.content == "Greetings!"
      assert struct.index == 1
    end

    test "handles receiving a function_call message" do
      response = %{
        "finish_reason" => "function_call",
        "index" => 0,
        "message" => %{
          "content" => nil,
          "function_call" => %{"arguments" => "{}", "name" => "hello_world"},
          "role" => "assistant"
        }
      }

      assert %Message{} = struct = ChatOpenAI.do_process_response(response)

      assert struct.role == :function_call
      assert struct.content == nil
      assert struct.function_name == "hello_world"
      assert struct.arguments == %{}
    end

    test "handles error from server that the max length has been reached"

    test "handles receiving a delta message for a content message at different parts" do
      delta_content = Langchain.Fixtures.raw_deltas_for_content()

      msg_1 = Enum.at(delta_content, 0)
      msg_2 = Enum.at(delta_content, 1)
      msg_10 = Enum.at(delta_content, 10)

      expected_1 = %MessageDelta{
        content: nil,
        index: 0,
        function_name: nil,
        role: :assistant,
        arguments: nil,
        complete: false
      }

      [%MessageDelta{} = delta_1] = ChatOpenAI.do_process_response(msg_1)
      assert delta_1 == expected_1

      expected_2 = %MessageDelta{
        content: "Hello",
        index: 0,
        function_name: nil,
        role: :unknown,
        arguments: nil,
        complete: false
      }

      [%MessageDelta{} = delta_2] = ChatOpenAI.do_process_response(msg_2)
      assert delta_2 == expected_2

      expected_10 = %MessageDelta{
        content: nil,
        index: 0,
        function_name: nil,
        role: :unknown,
        arguments: nil,
        complete: true
      }

      [%MessageDelta{} = delta_10] = ChatOpenAI.do_process_response(msg_10)
      assert delta_10 == expected_10
    end

    test "handles receiving a delta message for a function_call" do
      delta_function = Langchain.Fixtures.raw_deltas_for_function_call()

      msg_1 = Enum.at(delta_function, 0)
      msg_2 = Enum.at(delta_function, 1)
      msg_3 = Enum.at(delta_function, 2)

      expected_1 = %MessageDelta{
        content: nil,
        index: 0,
        function_name: "hello_world",
        role: :function_call,
        arguments: "",
        complete: false
      }

      [%MessageDelta{} = delta_1] = ChatOpenAI.do_process_response(msg_1)
      assert delta_1 == expected_1

      expected_2 = %MessageDelta{
        content: nil,
        index: 0,
        function_name: nil,
        role: :function_call,
        arguments: "{}",
        complete: false
      }

      [%MessageDelta{} = delta_2] = ChatOpenAI.do_process_response(msg_2)
      assert delta_2 == expected_2

      expected_3 = %MessageDelta{
        content: nil,
        index: 0,
        function_name: nil,
        role: :unknown,
        arguments: nil,
        complete: true
      }

      # it should not trim the arguments text
      [%MessageDelta{} = delta_3] = ChatOpenAI.do_process_response(msg_3)
      assert delta_3 == expected_3
    end

    test "handles receiving error message from server"

    test "return multiple responses when given multiple choices" do
      # received multiple responses because multiples were requested.
      response = %{
        "choices" => [
          %{
            "message" => %{"role" => "assistant", "content" => "Greetings!", "index" => 1},
            "finish_reason" => "stop"
          },
          %{
            "message" => %{"role" => "assistant", "content" => "Howdy!", "index" => 1},
            "finish_reason" => "stop"
          }
        ]
      }

      [msg1, msg2] = ChatOpenAI.do_process_response(response)
      assert %Message{role: :assistant, index: 1} = msg1
      assert %Message{role: :assistant, index: 1} = msg2
      assert msg1.content == "Greetings!"
      assert msg2.content == "Howdy!"
    end
  end

  describe "streaming examples" do
    @tag :live_call
    test "supports streaming response calling function with args" do
      callback = fn data ->
        IO.inspect(data, label: "DATA")
        send(self(), {:streamed_fn, data})
      end

      {:ok, chat} = ChatOpenAI.new(%{stream: true, callback_fn: callback, verbose: true})

      {:ok, message} =
        Message.new_user("Answer the following math question: What is 100 + 300 - 200?")

      response = ChatOpenAI.do_api_request(chat, [message], [Langchain.Tools.Calculator.new!()])

      IO.inspect(response, label: "OPEN AI POST RESPONSE")

      assert_receive {:streamed_fn, received_data}, 300
      assert %MessageDelta{} = received_data
      assert received_data.role == :function_call
      # wait for the response to be received
      Process.sleep(500)
    end
  end

  # TODO: TEST that a non-streaming result could return content with "finish_reason" => "length". If so,
  #      I would need to store content on a message AND flag the length error.

end
