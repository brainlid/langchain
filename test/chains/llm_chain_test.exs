defmodule Langchain.Chains.LLMChainTest do
  use Langchain.BaseCase

  doctest Langchain.Chains.LLMChain
  alias Langchain.ChatModels.ChatOpenAI
  alias Langchain.Chains.LLMChain
  alias Langchain.PromptTemplate
  alias Langchain.Function
  alias Langchain.Message
  alias Langchain.MessageDelta

  setup do
    {:ok, chat} = ChatOpenAI.new(%{temperature: 0})

    {:ok, function} =
      Function.new(%{
        name: "hello_world",
        description: "Responds with a greeting.",
        function: fn -> IO.puts("Hello world!") end
      })

    %{chat: chat, function: function}
  end

  describe "new/1" do
    test "works with minimal setup", %{chat: chat} do
      assert {:ok, %LLMChain{} = chain} = LLMChain.new(%{llm: chat})

      assert chain.llm == chat
    end

    test "accepts and includes functions to list and map", %{chat: chat, function: function} do
      assert {:ok, %LLMChain{} = chain} =
               LLMChain.new(%{
                 prompt: "Execute the hello_world function",
                 llm: chat,
                 functions: [function]
               })

      assert chain.llm == chat
      # include them in the list
      assert chain.functions == [function]
      # functions get mapped to a dictionary by name
      assert chain.function_map == %{"hello_world" => function}
    end
  end

  describe "add_functions/2" do
    test "adds a list of functions to the LLM list and map", %{chat: chat, function: function} do
      assert {:ok, %LLMChain{} = chain} =
               LLMChain.new(%{prompt: "Execute the hello_world function", llm: chat})

      assert chain.functions == []

      # test adding when empty
      updated_chain = LLMChain.add_functions(chain, [function])
      # includes function in the list and map
      assert updated_chain.functions == [function]
      assert updated_chain.function_map == %{"hello_world" => function}

      # test adding more when not empty
      {:ok, howdy_fn} =
        Function.new(%{
          name: "howdy",
          description: "Say howdy.",
          function: fn -> IO.puts("HOWDY!!") end
        })

      updated_chain2 = LLMChain.add_functions(updated_chain, [howdy_fn])
      # includes function in the list and map
      assert updated_chain2.functions == [function, howdy_fn]
      assert updated_chain2.function_map == %{"hello_world" => function, "howdy" => howdy_fn}
    end
  end

  describe "JS inspired test" do
    @tag :live_call
    test "live POST usage with LLM" do
      # https://js.langchain.com/docs/modules/chains/llm_chain

      prompt =
        PromptTemplate.from_template!(
          "Suggest one good name for a company that makes <%= @product %>?"
        )

      # We can construct an LLMChain from a PromptTemplate and an LLM.
      {:ok, updated_chain, response} =
        %{llm: ChatOpenAI.new!(%{temperature: 1, stream: false}), verbose: true}
        |> LLMChain.new!()
        |> LLMChain.apply_prompt_templates([prompt], %{product: "colorful socks"})
        |> LLMChain.run()

      assert %Message{role: :assistant} = response
      assert updated_chain.last_message == response

      # TODO: What does a streamed call_chat return?
      # success that it was submitted?
      # it's a blocking call while the callback function fires.
    end

    @tag :live_call
    test "live STREAM usage with LLM" do
      # https://js.langchain.com/docs/modules/chains/llm_chain

      prompt =
        PromptTemplate.from_template!(
          "Suggest one good name for a company that makes <%= @product %>?"
        )

      callback = fn %MessageDelta{} = delta ->
        send(self(), {:test_stream_deltas, delta})
      end

      model = ChatOpenAI.new!(%{temperature: 1, stream: true, callback_fn: callback})

      # We can construct an LLMChain from a PromptTemplate and an LLM.
      {:ok, updated_chain, response} =
        %{llm: model, verbose: true}
        |> LLMChain.new!()
        |> LLMChain.apply_prompt_templates([prompt], %{product: "colorful socks"})
        |> LLMChain.run()

      assert %Message{role: :assistant} = response
      assert updated_chain.last_message == response
      IO.inspect(response, label: "RECEIVED MESSAGE")

      # we should have received at least one callback message delta
      assert_received {:test_stream_deltas, delta_1}
      assert %MessageDelta{role: :assistant, complete: false} = delta_1
    end

    test "non-live not-streamed usage test" do
      # https://js.langchain.com/docs/modules/chains/llm_chain

      prompt =
        PromptTemplate.from_template!(
          "What is a good name for a company that makes <%= @product %>?"
        )

      # Made NOT LIVE here
      fake_message = Message.new!(%{role: :assistant, content: "Socktastic!", complete: true})
      set_api_override({:ok, [fake_message]})

      # We can construct an LLMChain from a PromptTemplate and an LLM.
      {:ok, %LLMChain{} = updated_chain, message} =
        %{llm: ChatOpenAI.new!(%{stream: false})}
        |> LLMChain.new!()
        |> LLMChain.apply_prompt_templates([prompt], %{product: "colorful socks"})
        # The result is an updated LLMChain with a last_message set, also the received message is returned
        |> LLMChain.run()

      assert updated_chain.needs_response == false
      assert updated_chain.last_message == message
      assert updated_chain.last_message == fake_message
    end

    test "non-live STREAM usage test" do
      # https://js.langchain.com/docs/modules/chains/llm_chain

      prompt =
        PromptTemplate.from_template!(
          "Suggest one good name for a company that makes <%= @product %>?"
        )

      callback = fn %MessageDelta{} = delta ->
        send(self(), {:fake_stream_deltas, delta})
      end

      model = ChatOpenAI.new!(%{temperature: 1, stream: true, callback_fn: callback})

      # Made NOT LIVE here
      fake_messages = [
        [MessageDelta.new!(%{role: :assistant, content: nil, complete: false})],
        [MessageDelta.new!(%{content: "Socktastic!", complete: false})],
        [MessageDelta.new!(%{content: nil, complete: true})]
      ]
      set_api_override({:ok, fake_messages})

      # We can construct an LLMChain from a PromptTemplate and an LLM.
      {:ok, updated_chain, response} =
        %{llm: model, verbose: false}
        |> LLMChain.new!()
        |> LLMChain.apply_prompt_templates([prompt], %{product: "colorful socks"})
        |> LLMChain.run()

      assert %Message{role: :assistant, content: "Socktastic!"} = response
      assert updated_chain.last_message == response
      IO.inspect(response, label: "RECEIVED MESSAGE")

      # we should have received at least one callback message delta
      assert_received {:fake_stream_deltas, delta_1}
      assert %MessageDelta{role: :assistant, complete: false} = delta_1
    end
  end

  describe "apply_delta/2" do
    setup do
      # https://js.langchain.com/docs/modules/chains/llm_chain#usage-with-chat-models
      {:ok, chat} = ChatOpenAI.new()
      {:ok, chain} = LLMChain.new(%{prompt: [], llm: chat, verbose: true})

      %{chain: chain}
    end

    test "when the first delta, assigns it to `delta`", %{chain: chain} do
      delta = MessageDelta.new!(%{role: :assistant, content: "Greetings from"})

      assert chain.delta == nil
      updated_chain = LLMChain.apply_delta(chain, delta)
      assert updated_chain.delta == delta
    end

    test "merges to existing delta and returns merged on struct", %{chain: chain} do
      updated_chain =
        chain
        |> LLMChain.apply_delta(
          MessageDelta.new!(%{role: :assistant, content: "Greetings from "})
        )
        |> LLMChain.apply_delta(MessageDelta.new!(%{content: "your "}))

      assert updated_chain.delta.content == "Greetings from your "
    end

    test "when final delta received, transforms to a message and applies it", %{chain: chain} do
      assert chain.messages == []

      updated_chain =
        chain
        |> LLMChain.apply_delta(
          MessageDelta.new!(%{role: :assistant, content: "Greetings from "})
        )
        |> LLMChain.apply_delta(MessageDelta.new!(%{content: "your "}))
        |> LLMChain.apply_delta(MessageDelta.new!(%{content: "favorite "}))
        |> LLMChain.apply_delta(MessageDelta.new!(%{content: "assistant.", complete: true}))

      # the delta is complete and removed from the chain
      assert updated_chain.delta == nil
      # the delta is converted to a message and applied to the messages
      assert [%Message{} = new_message] = updated_chain.messages
      assert new_message.role == :assistant
      assert new_message.content == "Greetings from your favorite assistant."
    end

    test "applies list of deltas for function_call with arguments", %{chain: chain} do
      deltas = Langchain.Fixtures.deltas_for_function_call("calculator")

      updated_chain =
        Enum.reduce(deltas, chain, fn delta, acc ->
          # apply each successive delta to the chain
          LLMChain.apply_delta(acc, delta)
        end)

      assert updated_chain.delta == nil
      last = updated_chain.last_message
      assert last.role == :function_call
      assert last.function_name == "calculator"
      assert last.arguments == %{"expression" => "100 + 300 - 200"}
      assert updated_chain.messages == [last]
    end
  end

  describe "apply_message/2" do
    setup do
      # https://js.langchain.com/docs/modules/chains/llm_chain#usage-with-chat-models
      {:ok, chat} = ChatOpenAI.new()
      {:ok, chain} = LLMChain.new(%{prompt: [], llm: chat, verbose: true})

      %{chain: chain}
    end

    test "appends a message and stores as last_message", %{chain: chain} do
      assert chain.messages == []

      # start with user message
      user_msg = Message.new_user!("Howdy!")
      updated_chain = LLMChain.apply_message(chain, user_msg)
      assert updated_chain.messages == [user_msg]
      assert updated_chain.last_message == user_msg

      # add assistant response
      assist_msg = Message.new_assistant!("Well hello to you too.")
      updated_chain = LLMChain.apply_message(updated_chain, assist_msg)
      assert updated_chain.messages == [user_msg, assist_msg]
      assert updated_chain.last_message == assist_msg
    end

    test "correctly sets the needs_response flag", %{chain: chain} do
      # after applying a message with role of :user, :function_call, or
      # :function, it should set need_response to true.
      user_msg = Message.new_user!("Howdy!")
      updated_chain = LLMChain.apply_message(chain, user_msg)
      assert updated_chain.needs_response

      function_call_msg = Message.new_function_call!("hello_world", "{}")
      updated_chain = LLMChain.apply_message(chain, function_call_msg)
      assert updated_chain.needs_response

      function_msg = Message.new_function!("hello_world", "Hello world!")
      updated_chain = LLMChain.apply_message(chain, function_msg)
      assert updated_chain.needs_response

      # set to false with a :system or :assistant message.
      system_msg = Message.new_system!("You are an overly optimistic assistant.")
      updated_chain = LLMChain.apply_message(chain, system_msg)
      refute updated_chain.needs_response

      assistant_msg = Message.new_assistant!("Yes, that's correct.")
      updated_chain = LLMChain.apply_message(chain, assistant_msg)
      refute updated_chain.needs_response
    end
  end

  describe "apply_prompt_templates/3" do
    test "transforms a list of messages and prompt templates into messages" do
      templates = [
        Message.new_system!("You are a helpful assistant"),
        PromptTemplate.new!(%{
          role: :user,
          text: "Give a brief description of <%= @subject %>."
        })
      ]

      {:ok, chat} = ChatOpenAI.new()
      {:ok, chain} = LLMChain.new(%{prompt: [], llm: chat})
      updated = LLMChain.apply_prompt_templates(chain, templates, %{subject: "Pomeranians"})
      assert length(updated.messages) == 2
      assert [%Message{role: :system}, %Message{role: :user} = user_msg] = updated.messages
      assert user_msg.content == "Give a brief description of Pomeranians."
      assert updated.last_message == user_msg
      assert updated.needs_response
    end
  end

  describe "quick_prompt/2" do
    test "creates the needed underlying messages and applies them" do
      {:ok, chat} = ChatOpenAI.new()
      {:ok, chain} = LLMChain.new(%{llm: chat})
      updated = LLMChain.quick_prompt(chain, "Hello!")
      assert length(updated.messages) == 2
      assert [%Message{role: :system}, %Message{role: :user} = user_msg] = updated.messages
      assert user_msg.content == "Hello!"
      assert updated.last_message == user_msg
      assert updated.needs_response
    end
  end

  # TODO: Sequential chains
  # https://js.langchain.com/docs/modules/chains/sequential_chain

  # TODO: Index related chains
  # https://js.langchain.com/docs/modules/chains/index_related_chains/

  # TODO: OpenAI Function Chains
  # https://js.langchain.com/docs/modules/chains/openai_functions/

  # TODO: Other Chains
  # https://js.langchain.com/docs/modules/chains/other_chains/
end
