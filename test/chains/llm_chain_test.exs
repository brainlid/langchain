defmodule LangChain.Chains.LLMChainTest do
  use LangChain.BaseCase
  use Mimic

  doctest LangChain.Chains.LLMChain

  import LangChain.Fixtures
  alias LangChain.ChatModels.ChatOpenAI
  alias LangChain.ChatModels.ChatAnthropic
  alias LangChain.Chains.LLMChain
  alias LangChain.PromptTemplate
  alias LangChain.Function
  alias LangChain.TokenUsage
  alias LangChain.Message
  alias LangChain.Message.ContentPart
  alias LangChain.Message.ToolCall
  alias LangChain.Message.ToolResult
  alias LangChain.MessageDelta
  alias LangChain.LangChainError
  alias LangChain.MessageProcessors.JsonProcessor
  alias LangChain.Utils

  @anthropic_test_model "claude-3-opus-20240229"

  setup do
    {:ok, chat} = ChatOpenAI.new(%{temperature: 0})

    chain = LLMChain.new!(%{llm: chat})

    hello_world =
      Function.new!(%{
        name: "hello_world",
        description: "Responds with a greeting.",
        function: fn _args, _context -> "Hello world!" end
      })

    greet =
      Function.new!(%{
        name: "greet",
        description: "Greet a person.",
        function: fn %{"name" => name}, _context -> "Hi #{name}!" end
      })

    sync =
      Function.new!(%{
        name: "do_thing",
        description: "Do something by only synchronously.",
        function: fn _args, _context -> "Did something." end,
        async: false
      })

    fail_func =
      Function.new!(%{
        name: "fail_func",
        description: "Return a function failure response.",
        function: fn _args, _context ->
          {:error, "Not what I wanted"}
        end
      })

    get_date =
      Function.new!(%{
        name: "get_date",
        description: "Returns the date as YYYY-MM-DD",
        function: fn _args, _context ->
          # return as a formatted string and the date struct
          date = Date.new!(2024, 11, 1)

          # Format the date as YYYY-MM-DD
          formatted_date = Calendar.strftime(date, "%Y-%m-%d")

          {:ok, formatted_date, date}
        end
      })

    # on setup, delete the Process dictionary key for each test run
    Process.delete(:test_func_failed_once)

    fail_once =
      Function.new!(%{
        name: "fail_once",
        description: "Return a function that fails once and succeeds on the second request.",
        # make it `async: false` so the test process remains the same and our
        # process dictionary hack will work.
        async: false,
        function: fn _args, _context ->
          # uses the process dictionary to store the test state
          #
          # if we haven't failed once yet, do that. After failing once, the
          # state is changed and it will pass the next time.
          if false == Process.get(:test_func_failed_once, false) do
            Process.put(:test_func_failed_once, true)
            {:error, "Not what I wanted"}
          else
            # already failed once, return a success
            {:ok, "It worked this time"}
          end
        end
      })

    %{
      chat: chat,
      chain: chain,
      hello_world: hello_world,
      greet: greet,
      sync: sync,
      fail_func: fail_func,
      fail_once: fail_once,
      get_date: get_date
    }
  end

  def fake_success_processor(%LLMChain{} = _chain, %Message{} = message) do
    {:cont, %Message{message | processed_content: message.processed_content <> " *"}}
  end

  def fake_fail_processor(%LLMChain{} = _chain, %Message{} = _message) do
    {:halt, Message.new_user!("ERROR: I reject your message!")}
  end

  def fake_raise_processor(%LLMChain{} = _chain, %Message{} = _message) do
    raise RuntimeError, "BOOM! Processor exploded"
  end

  describe "new/1" do
    test "works with minimal setup", %{chat: chat} do
      assert {:ok, %LLMChain{} = chain} = LLMChain.new(%{llm: chat})

      assert chain.llm == chat
    end

    test "accepts and includes tools to list and map", %{chat: chat, hello_world: hello_world} do
      assert {:ok, %LLMChain{} = chain} =
               LLMChain.new(%{
                 prompt: "Execute the hello_world tool",
                 llm: chat,
                 tools: [hello_world]
               })

      assert chain.llm == chat
      # include them in the list
      assert chain.tools == [hello_world]
      # tools get mapped to a dictionary by name
      assert chain._tool_map == %{"hello_world" => hello_world}
    end

    test "requires `llm`" do
      assert {:error, changeset} = LLMChain.new(%{llm: nil})
      assert {"can't be blank", _} = changeset.errors[:llm]
    end
  end

  describe "new!/1" do
    test "works with minimal setup", %{chat: chat} do
      assert %LLMChain{} = chain = LLMChain.new!(%{llm: chat})
      assert chain.llm == chat
    end

    test "requires `llm`" do
      assert_raise LangChainError, "llm: can't be blank", fn ->
        LLMChain.new!(%{llm: nil})
      end
    end
  end

  describe "add_tools/2" do
    test "adds a list of tools to the LLM list and map", %{chat: chat, hello_world: hello_world} do
      assert {:ok, %LLMChain{} = chain} =
               LLMChain.new(%{prompt: "Execute the hello_world tool", llm: chat})

      assert chain.tools == []

      # test adding when empty
      updated_chain = LLMChain.add_tools(chain, [hello_world])
      # includes tool in the list and map
      assert updated_chain.tools == [hello_world]
      assert updated_chain._tool_map == %{"hello_world" => hello_world}

      # test adding more when not empty
      {:ok, howdy_fn} =
        Function.new(%{
          name: "howdy",
          description: "Say howdy.",
          function: fn _args, _context -> "HOWDY!!" end
        })

      updated_chain2 = LLMChain.add_tools(updated_chain, [howdy_fn])
      # includes function in the list and map
      assert updated_chain2.tools == [hello_world, howdy_fn]
      assert updated_chain2._tool_map == %{"hello_world" => hello_world, "howdy" => howdy_fn}
    end
  end

  describe "message_processors/2" do
    test "assigns processor list to the struct", %{chain: chain} do
      assert chain.message_processors == []

      processors = [JsonProcessor.new!()]
      updated_chain = LLMChain.message_processors(chain, processors)
      assert updated_chain.message_processors == processors
    end
  end

  describe "cancelled_delta/1" do
    test "does nothing when no delta is present" do
      model = ChatOpenAI.new!(%{temperature: 1, stream: true})

      # We can construct an LLMChain from a PromptTemplate and an LLM.
      chain = LLMChain.new!(%{llm: model, verbose: false})
      assert chain.delta == nil

      new_chain = LLMChain.cancel_delta(chain, :cancelled)
      assert new_chain == chain
    end

    test "remove delta and adds cancelled message" do
      model = ChatOpenAI.new!(%{temperature: 1, stream: true})

      # Made NOT LIVE here
      fake_messages = [
        [MessageDelta.new!(%{role: :assistant, content: nil, status: :incomplete})],
        [MessageDelta.new!(%{content: "Sock", status: :incomplete})]
      ]

      expect(ChatOpenAI, :call, fn _model, _messages, _tools ->
        {:ok, fake_messages}
      end)

      # We can construct an LLMChain from a PromptTemplate and an LLM.
      {:ok, updated_chain} =
        %{llm: model, verbose: false}
        |> LLMChain.new!()
        |> LLMChain.add_message(
          Message.new_user!("What is a good name for a company that makes colorful socks?")
        )
        |> LLMChain.run()

      assert %MessageDelta{} = updated_chain.delta
      new_chain = LLMChain.cancel_delta(updated_chain, :cancelled)
      assert new_chain.delta == nil

      content = [ContentPart.text!("Sock")]

      assert %Message{
               role: :assistant,
               content: ^content,
               status: :cancelled
             } =
               new_chain.last_message
    end
  end

  describe "JS inspired test" do
    @tag live_call: true, live_open_ai: true
    test "live POST usage with LLM" do
      # https://js.langchain.com/docs/modules/chains/llm_chain

      handler = %{
        on_llm_new_delta: fn %LLMChain{} = _chain, deltas ->
          send(self(), {:test_stream_deltas, deltas})
        end,
        on_message_processed: fn _chain, message ->
          send(self(), {:test_stream_message, message})
        end
      }

      prompt =
        PromptTemplate.from_template!(
          "Suggest one good name for a company that makes <%= @product %>?"
        )

      # We can construct an LLMChain from a PromptTemplate and an LLM.
      model = ChatOpenAI.new!(%{temperature: 1, seed: 0, stream: false})

      {:ok, updated_chain} =
        %{llm: model}
        |> LLMChain.new!()
        |> LLMChain.add_callback(handler)
        |> LLMChain.apply_prompt_templates([prompt], %{product: "colorful socks"})
        |> LLMChain.run()

      assert %Message{role: :assistant} = updated_chain.last_message

      assert_received {:on_message_processed, message}
      assert %Message{role: :assistant} = message
      # the final returned message should match the callback message
      assert message == updated_chain.last_message
      # we should have received the final combined message
      refute_received {:test_stream_message, _delta}
    end

    @tag live_call: true, live_open_ai: true
    test "live STREAM usage with LLM" do
      # https://js.langchain.com/docs/modules/chains/llm_chain

      prompt =
        PromptTemplate.from_template!(
          "Suggest one good name for a company that makes <%= @product %>?"
        )

      handler = %{
        on_llm_new_delta: fn %LLMChain{} = _chain, deltas ->
          send(self(), {:test_stream_deltas, deltas})
        end,
        on_message_processed: fn _chain, message ->
          send(self(), {:test_stream_message, message})
        end
      }

      model =
        ChatOpenAI.new!(%{
          temperature: 1,
          seed: 0,
          stream: true,
          stream_options: %{include_usage: true}
        })

      # model = ChatAnthropic.new!(%{temperature: 1, seed: 0, stream: true, verbose_api: true})

      # We can construct an LLMChain from a PromptTemplate and an LLM.
      {:ok, updated_chain} =
        %{llm: model, verbose: false}
        |> LLMChain.new!()
        |> LLMChain.add_callback(handler)
        |> LLMChain.apply_prompt_templates([prompt], %{product: "colorful socks"})
        |> LLMChain.run()

      assert %Message{role: :assistant} = updated_chain.last_message

      # we should have received at least one callback message delta
      assert_received {:test_stream_deltas, deltas}
      assert %MessageDelta{role: :assistant, status: :incomplete} = List.first(deltas)

      # we should have received the final combined message
      assert_received {:test_stream_message, message}
      assert %Message{role: :assistant} = message
      # the final returned message should match the callback message
      assert message.content == updated_chain.last_message.content
    end

    test "non-live not-streamed usage test" do
      # https://js.langchain.com/docs/modules/chains/llm_chain

      prompt =
        PromptTemplate.from_template!(
          "What is a good name for a company that makes <%= @product %>?"
        )

      # Made NOT LIVE here
      fake_message = Message.new!(%{role: :assistant, content: "Socktastic!", status: :complete})

      expect(ChatOpenAI, :call, fn _model, _messages, _tools ->
        {:ok, [fake_message]}
      end)

      # We can construct an LLMChain from a PromptTemplate and an LLM.
      {:ok, %LLMChain{} = updated_chain} =
        %{llm: ChatOpenAI.new!(%{stream: false})}
        |> LLMChain.new!()
        |> LLMChain.apply_prompt_templates([prompt], %{product: "colorful socks"})
        # The result is an updated LLMChain with a last_message set, also the received message is returned
        |> LLMChain.run()

      assert updated_chain.needs_response == false
      assert updated_chain.last_message == fake_message
    end

    test "non-live STREAM usage test" do
      # testing that a set of deltas can be processed, they fire events, and the
      # full message processed event fires from the chain.

      # https://js.langchain.com/docs/modules/chains/llm_chain

      prompt =
        PromptTemplate.from_template!(
          "Suggest one good name for a company that makes <%= @product %>?"
        )

      llm_handler = %{
        on_llm_new_delta: fn _model, %MessageDelta{} = delta ->
          send(self(), {:fake_stream_deltas, delta})
        end
      }

      chain_handler = %{
        on_message_processed: fn _model, %Message{} = fake_full_message ->
          send(self(), {:fake_full_message, fake_full_message})
        end
      }

      model = ChatOpenAI.new!(%{temperature: 1, stream: true, callbacks: [llm_handler]})

      # Made NOT LIVE here
      fake_messages = [
        [MessageDelta.new!(%{role: :assistant, content: nil, status: :incomplete})],
        [MessageDelta.new!(%{content: "Socktastic!", status: :incomplete})],
        [MessageDelta.new!(%{content: nil, status: :complete})]
      ]

      expect(ChatOpenAI, :call, fn _model, _messages, _tools ->
        {:ok, fake_messages}
      end)

      # We can construct an LLMChain from a PromptTemplate and an LLM.
      {:ok, updated_chain} =
        %{llm: model, verbose: false, callbacks: [chain_handler]}
        |> LLMChain.new!()
        |> LLMChain.apply_prompt_templates([prompt], %{product: "colorful socks"})
        |> LLMChain.run()

      content = [ContentPart.text!("Socktastic!")]

      assert %Message{role: :assistant, content: ^content, status: :complete} =
               updated_chain.last_message

      # we should have received a message for the completed, combined message
      assert_received {:fake_full_message, message}
      assert %Message{role: :assistant, content: ^content} = message
    end
  end

  describe "merge_delta/2" do
    setup do
      # https://js.langchain.com/docs/modules/chains/llm_chain#usage-with-chat-models
      {:ok, chat} = ChatOpenAI.new()
      {:ok, chain} = LLMChain.new(%{prompt: [], llm: chat, verbose: false})

      %{chain: chain}
    end

    test "when the first delta, assigns it to `delta`", %{chain: chain} do
      delta = MessageDelta.new!(%{role: :assistant, content: ContentPart.text!("Greetings from")})

      assert chain.delta == nil
      updated_chain = LLMChain.merge_delta(chain, delta)
      assert updated_chain.delta.merged_content == [ContentPart.text!("Greetings from")]
      assert updated_chain.delta.content == nil
      assert updated_chain.delta.role == :assistant
      assert updated_chain.delta.status == :incomplete
    end

    test "merges to existing delta and returns merged on struct", %{chain: chain} do
      updated_chain =
        chain
        |> LLMChain.merge_delta(
          MessageDelta.new!(%{role: :assistant, content: ContentPart.text!("Greetings from ")})
        )
        |> LLMChain.merge_delta(MessageDelta.new!(%{content: ContentPart.text!("your ")}))

      assert updated_chain.delta.merged_content == [ContentPart.text!("Greetings from your ")]
    end

    test "applies a token usage struct to the chain", %{chain: chain} do
      assert chain.messages == []

      updated_chain =
        chain
        |> LLMChain.merge_delta(
          MessageDelta.new!(%{role: :assistant, content: "Greetings from "})
        )
        |> LLMChain.merge_delta(MessageDelta.new!(%{content: "your "}))
        |> LLMChain.merge_delta(MessageDelta.new!(%{content: "favorite "}))
        |> LLMChain.merge_delta(TokenUsage.new!(%{input: 10, output: 5}))

      assert %MessageDelta{
               role: :assistant,
               status: :incomplete,
               content: nil,
               merged_content: [
                 %ContentPart{
                   type: :text,
                   content: "Greetings from your favorite ",
                   options: []
                 }
               ],
               tool_calls: nil,
               metadata: %{usage: %TokenUsage{input: 10, output: 5, raw: %{}}}
             } = updated_chain.delta
    end

    test "applies list of deltas for tool_call with arguments", %{chain: chain} do
      deltas = deltas_for_tool_call("calculator")

      updated_chain =
        Enum.reduce(deltas, chain, fn delta, acc ->
          # apply each successive delta to the chain
          LLMChain.merge_delta(acc, delta)
        end)

      assert %MessageDelta{
               role: :assistant,
               content: nil,
               index: 0,
               status: :complete,
               metadata: nil,
               tool_calls: [
                 %ToolCall{
                   status: :incomplete,
                   type: :function,
                   call_id: "call_IBDsG5rtgR9rt1CNrWkPMvXG",
                   name: "calculator",
                   arguments: "{\n \"expression\": \"100 + 300 - 200\"}",
                   index: 0
                 }
               ],
               merged_content: []
             } = updated_chain.delta
    end

    test "cancels the current delta when applying an overloaded error", %{chain: chain} do
      assert chain.messages == []

      updated_chain =
        chain
        |> LLMChain.merge_delta(
          MessageDelta.new!(%{role: :assistant, content: "Greetings from "})
        )
        |> LLMChain.merge_delta(MessageDelta.new!(%{content: "your "}))
        |> LLMChain.merge_delta(MessageDelta.new!(%{content: "favorite "}))
        |> LLMChain.merge_delta(
          {:error, LangChainError.exception(type: "overloaded", message: "Overloaded")}
        )

      # the delta is complete and removed from the chain
      assert updated_chain.delta == nil
      # the delta is converted to a message and applied to the messages
      assert [%Message{} = new_message] = updated_chain.messages
      assert new_message.role == :assistant
      assert new_message.content == [ContentPart.text!("Greetings from your favorite ")]
      assert new_message.status == :cancelled
    end
  end

  describe "apply_deltas/2" do
    test "applies list of deltas" do
      deltas = [
        [
          %LangChain.MessageDelta{
            content: nil,
            status: :incomplete,
            index: 0,
            role: :assistant,
            tool_calls: [
              %LangChain.Message.ToolCall{
                status: :incomplete,
                type: :function,
                call_id: "call_abc123",
                name: "find_by_code",
                arguments: nil,
                index: 0
              }
            ]
          }
        ],
        [
          %LangChain.MessageDelta{
            content: nil,
            status: :incomplete,
            index: 0,
            role: :unknown,
            tool_calls: [
              %LangChain.Message.ToolCall{
                status: :incomplete,
                type: :function,
                call_id: nil,
                name: nil,
                arguments: "{\"",
                index: 0
              }
            ]
          }
        ],
        [
          %LangChain.MessageDelta{
            content: nil,
            status: :incomplete,
            index: 0,
            role: :unknown,
            tool_calls: [
              %LangChain.Message.ToolCall{
                status: :incomplete,
                type: :function,
                call_id: nil,
                name: nil,
                arguments: "code",
                index: 0
              }
            ]
          }
        ],
        [
          %LangChain.MessageDelta{
            content: nil,
            status: :incomplete,
            index: 0,
            role: :unknown,
            tool_calls: [
              %LangChain.Message.ToolCall{
                status: :incomplete,
                type: :function,
                call_id: nil,
                name: nil,
                arguments: "\":\"",
                index: 0
              }
            ]
          }
        ],
        [
          %LangChain.MessageDelta{
            content: nil,
            status: :incomplete,
            index: 0,
            role: :unknown,
            tool_calls: [
              %LangChain.Message.ToolCall{
                status: :incomplete,
                type: :function,
                call_id: nil,
                name: nil,
                arguments: "don",
                index: 0
              }
            ]
          }
        ],
        [
          %LangChain.MessageDelta{
            content: nil,
            status: :incomplete,
            index: 0,
            role: :unknown,
            tool_calls: [
              %LangChain.Message.ToolCall{
                status: :incomplete,
                type: :function,
                call_id: nil,
                name: nil,
                arguments: "ate",
                index: 0
              }
            ]
          }
        ],
        [
          %LangChain.MessageDelta{
            content: nil,
            status: :incomplete,
            index: 0,
            role: :unknown,
            tool_calls: [
              %LangChain.Message.ToolCall{
                status: :incomplete,
                type: :function,
                call_id: nil,
                name: nil,
                arguments: "\"}",
                index: 0
              }
            ]
          }
        ],
        [
          %LangChain.MessageDelta{
            content: nil,
            status: :complete,
            index: 0,
            role: :unknown,
            tool_calls: nil
          }
        ],
        %LangChain.TokenUsage{
          input: 15,
          output: 4,
          raw: %{
            "completion_tokens" => 4,
            "completion_tokens_details" => %{
              "accepted_prediction_tokens" => 0,
              "audio_tokens" => 0,
              "reasoning_tokens" => 0,
              "rejected_prediction_tokens" => 0
            },
            "prompt_tokens" => 15,
            "prompt_tokens_details" => %{"audio_tokens" => 0, "cached_tokens" => 0},
            "total_tokens" => 19
          }
        }
      ]

      chain = LLMChain.new!(%{llm: ChatOpenAI.new!()})
      updated_chain = LLMChain.apply_deltas(chain, deltas)

      assert updated_chain.delta == nil
      %Message{} = last = updated_chain.last_message
      assert last.role == :assistant
      [%ToolCall{} = tool_call] = last.tool_calls
      assert tool_call.name == "find_by_code"
      assert tool_call.arguments == %{"code" => "donate"}
      assert updated_chain.messages == [last]
      assert %TokenUsage{input: 15, output: 4, raw: %{}} = last.metadata.usage
    end
  end

  describe "add_message/2" do
    setup do
      # https://js.langchain.com/docs/modules/chains/llm_chain#usage-with-chat-models
      {:ok, chat} = ChatOpenAI.new()
      {:ok, chain} = LLMChain.new(%{prompt: [], llm: chat, verbose: false})

      %{chain: chain}
    end

    test "appends a message and stores as last_message", %{chain: chain} do
      assert chain.messages == []

      # start with user message
      user_msg = Message.new_user!("Howdy!")
      updated_chain = LLMChain.add_message(chain, user_msg)
      assert updated_chain.messages == [user_msg]
      assert updated_chain.last_message == user_msg

      # add assistant response
      assist_msg = Message.new_assistant!(%{content: "Well hello to you too."})
      updated_chain = LLMChain.add_message(updated_chain, assist_msg)
      assert updated_chain.messages == [user_msg, assist_msg]
      assert updated_chain.last_message == assist_msg
    end

    test "correctly sets the needs_response flag", %{chain: chain} do
      # after applying a message with role of :user, :function_call, or
      # :function, it should set need_response to true.
      user_msg = Message.new_user!("Howdy!")
      updated_chain = LLMChain.add_message(chain, user_msg)
      assert updated_chain.needs_response

      call_msg = new_function_call!("call_abc123", "hello_world", "{}")
      # function_call_msg = Message.new_function_call!("hello_world", "{}")
      updated_chain = LLMChain.add_message(chain, call_msg)
      assert updated_chain.needs_response

      tool_msg =
        Message.new_tool_result!(%{
          tool_results: [
            ToolResult.new!(%{tool_call_id: "call_abc123", content: "Hello world!"})
          ]
        })

      updated_chain = LLMChain.add_message(chain, tool_msg)
      assert updated_chain.needs_response

      # set to false with a :system or :assistant message.
      system_msg = Message.new_system!("You are an overly optimistic assistant.")
      updated_chain = LLMChain.add_message(chain, system_msg)
      refute updated_chain.needs_response

      assistant_msg = Message.new_assistant!(%{content: "Yes, that's correct."})
      updated_chain = LLMChain.add_message(chain, assistant_msg)
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
      assert user_msg.content == [ContentPart.text!("Give a brief description of Pomeranians.")]
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
      assert user_msg.content == [ContentPart.text!("Hello!")]
      assert updated.last_message == user_msg
      assert updated.needs_response
    end
  end

  describe "run_message_processors/2" do
    test "continues when no processors given", %{chain: chain} do
      assert chain.message_processors == []
      message = Message.new_assistant!(%{content: "Initial"})

      final_message = LLMChain.run_message_processors(chain, message)
      assert final_message == message
    end

    test "applies a single processor to the message", %{chain: chain} do
      chain =
        LLMChain.message_processors(chain, [
          &fake_success_processor/2
        ])

      message = Message.new_assistant!(%{content: "Initial"})

      final_message = LLMChain.run_message_processors(chain, message)
      assert final_message.processed_content == "Initial *"
    end

    test "applies successive processors", %{chain: chain} do
      chain =
        LLMChain.message_processors(chain, [
          &fake_success_processor/2,
          &fake_success_processor/2,
          &fake_success_processor/2
        ])

      message = Message.new_assistant!(%{content: "Initial"})

      final_message = LLMChain.run_message_processors(chain, message)
      assert final_message.processed_content == "Initial * * *"
    end

    test "returns :halted and a new message when :halt returned", %{
      chain: chain
    } do
      chain =
        LLMChain.message_processors(chain, [
          &fake_success_processor/2,
          &fake_success_processor/2,
          &fake_fail_processor/2
        ])

      message = Message.new_assistant!(%{content: "Initial"})

      {:halted, failed_message, new_message} = LLMChain.run_message_processors(chain, message)
      assert failed_message.role == :assistant
      assert failed_message.processed_content == "Initial * *"

      assert new_message.role == :user
      assert new_message.content == [ContentPart.text!("ERROR: I reject your message!")]
    end

    test "handles an exception raised in processor", %{chain: chain} do
      chain =
        LLMChain.message_processors(chain, [
          &fake_success_processor/2,
          &fake_raise_processor/2
        ])

      message = Message.new_assistant!(%{content: "Initial"})

      {:halted, final_message} = LLMChain.run_message_processors(chain, message)

      assert final_message.content ==
               [
                 ContentPart.text!(
                   "ERROR: An exception was raised! Exception: %RuntimeError{message: \"BOOM! Processor exploded\"}"
                 )
               ]
    end

    test "does nothing on other message roles", %{chain: chain} do
      chain =
        LLMChain.message_processors(chain, [
          &fake_success_processor/2
        ])

      message = Message.new_user!("Howdy!")

      assert message == LLMChain.run_message_processors(chain, message)
    end
  end

  describe "process_message/2" do
    test "runs message processors, adds to chain, fires callbacks on final message", %{
      chain: chain
    } do
      handler = %{
        on_message_processed: fn _chain, %Message{} = message ->
          send(self(), {:processed_message_callback, message})
        end,
        on_llm_token_usage: fn _chain, %TokenUsage{} = usage ->
          send(self(), {:token_usage_callback, usage})
        end
      }

      # internal hack to assign a callback. Verifying it gets executed.
      chain = %LLMChain{chain | callbacks: [handler]}

      chain =
        LLMChain.message_processors(chain, [
          &fake_success_processor/2,
          &fake_success_processor/2
        ])

      message =
        Message.new_assistant!(%{
          content: "Initial",
          metadata: %{usage: %TokenUsage{input: 10, output: 15}}
        })

      updated_chain = LLMChain.process_message(chain, message)
      [msg1] = updated_chain.messages
      assert msg1.processed_content == "Initial * *"

      # Expect callback with the updated message
      assert_received {:processed_message_callback, ^msg1}
      assert_received {:token_usage_callback, %TokenUsage{input: 10, output: 15}}
    end

    test "when halted, adds original message plus new message returned from processor and fires 2 callbacks",
         %{chain: chain} do
      # Verifying it gets executed.
      handler = %{
        on_message_processing_error: fn _chain, item ->
          send(self(), {:processing_error_callback, item})
        end,
        on_error_message_created: fn _chain, item ->
          send(self(), {:error_message_created_callback, item})
        end
      }

      assert chain.current_failure_count == 0

      chain =
        chain
        |> LLMChain.add_callback(handler)
        |> LLMChain.message_processors([
          &fake_success_processor/2,
          &fake_fail_processor/2
        ])

      message = Message.new_assistant!(%{content: "Initial"})

      updated_chain = LLMChain.process_message(chain, message)
      # the failure count is incremented
      assert updated_chain.current_failure_count == 1
      [msg1, msg2] = updated_chain.messages
      # includes the message that errored at the point it was before failure
      assert msg1.processed_content == "Initial *"
      # adds a new message with the processor response message
      assert msg2.content == [ContentPart.text!("ERROR: I reject your message!")]

      # Expect callback with the original assistant message
      assert_received {:processing_error_callback, ^msg1}
      # Expect callback with the new user message
      assert_received {:error_message_created_callback, ^msg2}
    end

    test "on successful processing, clears or resets the failure count", %{chain: chain} do
      chain =
        chain
        |> LLMChain.increment_current_failure_count()
        |> LLMChain.increment_current_failure_count()
        |> LLMChain.message_processors([&fake_success_processor/2])

      assert chain.current_failure_count == 2

      message = Message.new_assistant!(%{content: "Initial"})

      updated_chain = LLMChain.process_message(chain, message)
      # the failure count is reset
      assert updated_chain.current_failure_count == 0
      [msg1] = updated_chain.messages
      # includes the message that errored at the point it was before failure
      assert msg1.content == [ContentPart.text!("Initial")]
      assert msg1.processed_content == "Initial *"
    end
  end

  describe "run/1" do
    test "returns an error when running without messages", %{chain: chain} do
      assert chain.messages == []

      {:error, error_chain, error} = LLMChain.run(chain)
      assert error_chain == chain
      assert error.message == "LLMChain cannot be run without messages"
    end

    @tag live_call: true, live_open_ai: true
    test "custom_context is passed to a custom function" do
      # map of data we want to be passed as `context` to the function when
      # executed.
      custom_context = %{
        "user_id" => 123,
        "hairbrush" => "drawer",
        "dog" => "backyard",
        "sandwich" => "kitchen"
      }

      test_pid = self()

      # a custom Elixir function made available to the LLM
      custom_fn =
        Function.new!(%{
          name: "item_location",
          description: "Returns the location of the requested element or item.",
          parameters_schema: %{
            type: "object",
            properties: %{
              thing: %{
                type: "string",
                description: "The thing whose location is being requested."
              }
            },
            required: ["thing"]
          },
          function: fn %{"thing" => thing} = arguments, context ->
            send(test_pid, {:function_run, arguments, context})
            # our context is a pretend item/location location map
            {:ok, context[thing]}
          end
        })

      # create and run the chain
      {:ok, updated_chain} =
        LLMChain.new!(%{
          llm: ChatOpenAI.new!(%{seed: 0}),
          custom_context: custom_context,
          verbose: false
        })
        |> LLMChain.add_tools(custom_fn)
        |> LLMChain.add_message(Message.new_user!("Where is the hairbrush located?"))
        |> LLMChain.run(mode: :while_needs_response)

      [_tool_call, _tool_result, %Message{} = final_message] = updated_chain.exchanged_messages

      assert updated_chain.last_message == final_message
      assert final_message.role == :assistant

      assert "The hairbrush is located in the drawer." ==
               ContentPart.content_to_string(final_message.content)

      # assert our custom function was executed with custom_context supplied
      assert_received {:function_run, arguments, context}
      assert context == custom_context
      assert arguments == %{"thing" => "hairbrush"}
    end

    @tag live_call: true, live_open_ai: true
    test "NON-STREAMING handles receiving an error when no messages sent" do
      # create and run the chain
      {:error, _updated_chain, %LangChainError{} = reason} =
        LLMChain.new!(%{
          llm: ChatOpenAI.new!(%{seed: 0, stream: false}),
          verbose: false
        })
        |> LLMChain.run()

      assert reason.type == nil
      assert reason.message == "LLMChain cannot be run without messages"
    end

    @tag live_call: true, live_open_ai: true
    test "STREAMING handles receiving an error when no messages sent" do
      # create and run the chain
      {:error, _updated_chain, %LangChainError{} = reason} =
        LLMChain.new!(%{
          llm: ChatOpenAI.new!(%{seed: 0, stream: true}),
          verbose: false
        })
        |> LLMChain.run()

      assert reason.type == nil
      assert reason.message == "LLMChain cannot be run without messages"
    end

    # runs until tools are evaluated
    @tag live_call: true, live_open_ai: true
    test "handles content response + function call" do
      test_pid = self()

      message =
        Message.new_user!("""
        Please pull the list of available fly_regions and return them to me. List as:

        - (region_abbreviation) Region Name
        """)

      regions_function =
        Function.new!(%{
          name: "fly_regions",
          description:
            "List the currently available regions an app can be deployed to in JSON format.",
          function: fn _args, _context ->
            send(test_pid, {:function_called, "fly_regions"})

            [
              %{name: "ams", location: "Amsterdam, Netherlands"},
              %{name: "arn", location: "Stockholm, Sweden"},
              %{name: "atl", location: "Atlanta, Georgia (US)"},
              %{name: "dfw", location: "Dallas, Texas (US)"},
              %{name: "fra", location: "Frankfurt, Germany"},
              %{name: "iad", location: "Ashburn, Virginia (US)"},
              %{name: "lax", location: "Los Angeles, California (US)"},
              %{name: "nrt", location: "Tokyo, Japan"},
              %{name: "ord", location: "Chicago, Illinois (US)"},
              %{name: "yul", location: "Montreal, Canada"},
              %{name: "yyz", location: "Toronto, Canada"}
            ]
            |> Jason.encode!()
          end
        })

      {:ok, updated_chain} =
        LLMChain.new!(%{
          llm: ChatOpenAI.new!(%{seed: 0, stream: false}),
          custom_context: nil,
          verbose: false
        })
        |> LLMChain.add_tools(regions_function)
        |> LLMChain.add_message(message)
        |> LLMChain.run(mode: :while_needs_response)

      [_tool_call, _tool_result, %Message{} = final_response] = updated_chain.exchanged_messages

      # the final_response should contain data returned from the function
      assert final_response == updated_chain.last_message
      assert ContentPart.content_to_string(final_response.content) =~ "Germany"
      assert ContentPart.content_to_string(final_response.content) =~ "fra"
      assert final_response.role == :assistant
      assert_received {:function_called, "fly_regions"}
    end

    test "ignores empty lists in the list of messages" do
      # Made NOT LIVE here
      expect(ChatOpenAI, :call, fn _model, _prompt, _tools ->
        {:ok,
         [
           [],
           [
             MessageDelta.new!(%{content: "Hello ", role: :assistant}),
             [],
             MessageDelta.new!(%{content: "World", role: :assistant})
           ]
         ]}
      end)

      model = ChatOpenAI.new!(%{stream: true, model: "gpt-4o-mini"})

      assert {:ok, updated_chain} =
               LLMChain.new!(%{llm: model})
               |> LLMChain.add_messages([Message.new_user!("Hi")])
               |> LLMChain.run()

      assert %MessageDelta{
               merged_content: [
                 %ContentPart{
                   type: :text,
                   content: "Hello World",
                   options: []
                 }
               ],
               status: :incomplete,
               role: :assistant
             } =
               updated_chain.delta
    end

    test "returns error when receives overloaded from Anthropic" do
      # Made NOT LIVE here
      expect(ChatAnthropic, :call, fn _model, _prompt, _tools ->
        {:error, LangChainError.exception(type: "overloaded", message: "Overloaded (from test)")}
      end)

      model = ChatAnthropic.new!(%{stream: true, model: @anthropic_test_model})

      assert {:error, _updated_chain, reason} =
               LLMChain.new!(%{llm: model})
               |> LLMChain.add_messages([Message.new_user!("Hi")])
               |> LLMChain.run()

      assert reason.type == "overloaded"
      assert reason.message == "Overloaded (from test)"
    end

    test "errors when messages have PromptTemplates" do
      messages = [
        PromptTemplate.new!(%{
          role: :system,
          text: "You are my personal assistant named <%= @assistant_name %>."
        })
      ]

      # errors when trying to send a PromptTemplate
      assert_raise LangChainError, ~r/PromptTemplates must be/, fn ->
        LLMChain.new!(%{
          llm: ChatOpenAI.new!(%{seed: 0}),
          verbose: false
        })
        |> LLMChain.add_messages(messages)
        |> LLMChain.run()
      end
    end

    test "mode: :while_needs_response - increments current_failure_count on parse failure", %{
      chain: chain
    } do
      # Made NOT LIVE here
      fake_messages = [
        Message.new_assistant!(%{content: "Not what you wanted"})
      ]

      # expect it to be called 3 times
      expect(ChatOpenAI, :call, 3, fn _model, _messages, _tools ->
        {:ok, fake_messages}
      end)

      messages = [
        Message.new_user!("Say what I want you to say.")
      ]

      {:error, error_chain, %LangChainError{} = reason} =
        chain
        |> LLMChain.message_processors([JsonProcessor.new!()])
        |> LLMChain.add_messages(messages)
        # run repeatedly
        |> LLMChain.run(mode: :while_needs_response)

      assert error_chain.current_failure_count == 3
      assert reason.type == "exceeded_failure_count"
      assert reason.message == "Exceeded max failure count"

      [m1, m2, m3, m4, m5, m6, m7] = error_chain.messages

      assert m1.role == :user
      assert m1.content == [ContentPart.text!("Say what I want you to say.")]

      assert m2.role == :assistant
      assert m2.content == [ContentPart.text!("Not what you wanted")]
      assert m2.processed_content == "Not what you wanted"

      assert m3.role == :user

      assert m3.content == [
               ContentPart.text!(
                 "ERROR: Invalid JSON data: unexpected byte at position 0: 0x4E (\"N\")"
               )
             ]

      assert m4.role == :assistant
      assert m4.content == [ContentPart.text!("Not what you wanted")]
      assert m4.processed_content == "Not what you wanted"

      assert m5.role == :user

      assert m5.content == [
               ContentPart.text!(
                 "ERROR: Invalid JSON data: unexpected byte at position 0: 0x4E (\"N\")"
               )
             ]

      assert m6.role == :assistant
      assert m6.content == [ContentPart.text!("Not what you wanted")]
      assert m6.processed_content == "Not what you wanted"

      assert m7.role == :user

      assert m7.content == [
               ContentPart.text!(
                 "ERROR: Invalid JSON data: unexpected byte at position 0: 0x4E (\"N\")"
               )
             ]
    end

    test "mode: :while_needs_response - fires callbacks for failed messages correctly" do
      handler = %{
        on_message_processing_error: fn _chain, data ->
          send(self(), {:processing_error_callback, data})
        end,
        on_error_message_created: fn _chain, data ->
          send(self(), {:error_message_created_callback, data})
        end,
        on_retries_exceeded: fn chain ->
          send(self(), {:retries_exceeded_callback, chain})
        end
      }

      # Made NOT LIVE here
      fake_messages = [
        Message.new_assistant!(%{content: "Not what you wanted"})
      ]

      # expects to be called 2 times
      expect(ChatOpenAI, :call, 2, fn _model, _messages, _tools ->
        {:ok, fake_messages}
      end)

      chain =
        LLMChain.new!(%{
          llm: ChatOpenAI.new!(%{temperature: 0}),
          verbose: false,
          max_retry_count: 2,
          callbacks: [handler]
        })

      {:error, error_chain, %LangChainError{} = reason} =
        chain
        |> LLMChain.message_processors([JsonProcessor.new!()])
        |> LLMChain.add_messages([
          Message.new_user!("Say what I want you to say.")
        ])
        # run repeatedly
        |> LLMChain.run(mode: :while_needs_response)

      assert error_chain.current_failure_count == 2
      assert reason.type == "exceeded_failure_count"
      assert reason.message == "Exceeded max failure count"

      [m1, m2, m3, m4, m5] = error_chain.messages

      assert m1.role == :user
      assert m1.content == [ContentPart.text!("Say what I want you to say.")]

      assert m2.role == :assistant
      assert m2.content == [ContentPart.text!("Not what you wanted")]
      assert m2.processed_content == "Not what you wanted"

      assert m3.role == :user

      assert m3.content == [
               ContentPart.text!(
                 "ERROR: Invalid JSON data: unexpected byte at position 0: 0x4E (\"N\")"
               )
             ]

      assert m4.role == :assistant
      assert m4.content == [ContentPart.text!("Not what you wanted")]
      assert m4.processed_content == "Not what you wanted"

      assert m5.role == :user

      assert m5.content == [
               ContentPart.text!(
                 "ERROR: Invalid JSON data: unexpected byte at position 0: 0x4E (\"N\")"
               )
             ]

      assert_received {:processing_error_callback, ^m2}
      assert_received {:error_message_created_callback, ^m3}
      assert_received {:processing_error_callback, ^m4}
      assert_received {:error_message_created_callback, ^m5}
      assert_received {:retries_exceeded_callback, ^error_chain}
      refute_received {:processing_error_callback, _data}
      refute_received {:error_message_created_callback, _data}
    end

    test "mode: :until_success - message needs processing, succeeds", %{chain: chain} do
      # Made NOT LIVE here
      fake_messages = [
        Message.new_assistant!(%{content: Jason.encode!(%{value: "abc"})})
      ]

      expect(ChatOpenAI, :call, fn _model, _messages, _tools ->
        {:ok, fake_messages}
      end)

      {:ok, updated_chain} =
        chain
        |> LLMChain.message_processors([JsonProcessor.new!()])
        |> LLMChain.add_message(Message.new_system!())
        |> LLMChain.add_message(Message.new_user!("What's the value in JSON?"))
        |> LLMChain.run(mode: :until_success)

      # stopped after processing a successful assistant response
      assert updated_chain.last_message.role == :assistant
      assert updated_chain.last_message.processed_content == %{"value" => "abc"}
    end

    test "mode: :until_success - message needs processing, fails, then succeeds", %{chat: chat} do
      # Made NOT LIVE here - handles two consecutive calls
      expect(ChatOpenAI, :call, fn _model, _messages, _tools ->
        {:ok, [Message.new_assistant!(%{content: "invalid"})]}
      end)

      expect(ChatOpenAI, :call, fn _model, _messages, _tools ->
        {:ok,
         [
           Message.new_assistant!(%{content: Jason.encode!(%{value: "abc"})})
         ]}
      end)

      {:ok, updated_chain} =
        %{llm: chat}
        |> LLMChain.new!()
        |> LLMChain.message_processors([JsonProcessor.new!()])
        |> LLMChain.add_message(Message.new_system!())
        |> LLMChain.add_message(Message.new_user!("What's the value in JSON?"))
        |> LLMChain.run(mode: :until_success)

      # stopped after processing a successful assistant response
      assert updated_chain.last_message.role == :assistant
      assert updated_chain.last_message.processed_content == %{"value" => "abc"}
    end

    test "mode: :until_success - tool call returns failure once, then succeeds", %{
      fail_once: fail_once
    } do
      # Made NOT LIVE here
      fake_messages = [
        new_function_calls!([
          ToolCall.new!(%{call_id: "call_fake123", name: "fail_once", arguments: nil})
        ])
      ]

      expect(ChatOpenAI, :call, 2, fn _model, _messages, _tools ->
        {:ok, fake_messages}
      end)

      {:ok, updated_chain} =
        %{llm: ChatOpenAI.new!(%{stream: false}), verbose: false}
        |> LLMChain.new!()
        |> LLMChain.add_tools([fail_once])
        |> LLMChain.add_message(Message.new_system!())
        |> LLMChain.add_message(Message.new_user!("Execute the fail_once tool."))
        |> LLMChain.run(mode: :until_success)

      [tool_call_1, tool_result_1, tool_call_2, tool_result_2] = updated_chain.exchanged_messages

      assert %Message{
               tool_calls: [
                 %LangChain.Message.ToolCall{
                   status: :complete,
                   type: :function,
                   call_id: "call_fake123",
                   name: "fail_once"
                 }
               ]
             } = tool_call_1

      assert %Message{
               tool_results: [
                 %LangChain.Message.ToolResult{
                   type: :function,
                   tool_call_id: "call_fake123",
                   name: "fail_once",
                   content: [%ContentPart{type: :text, content: "Not what I wanted", options: []}],
                   # failed
                   is_error: true
                 }
               ]
             } = tool_result_1

      assert %Message{
               tool_calls: [
                 %LangChain.Message.ToolCall{
                   status: :complete,
                   type: :function,
                   call_id: "call_fake123",
                   name: "fail_once"
                 }
               ]
             } = tool_call_2

      assert %Message{
               status: :complete,
               role: :tool,
               tool_results: [
                 %LangChain.Message.ToolResult{
                   type: :function,
                   tool_call_id: "call_fake123",
                   name: "fail_once",
                   content: [
                     %ContentPart{type: :text, content: "It worked this time", options: []}
                   ],
                   # passed
                   is_error: false
                 }
               ]
             } = tool_result_2

      assert updated_chain.last_message.role == :tool
      assert [%ToolResult{is_error: false}] = updated_chain.last_message.tool_results
      assert updated_chain.current_failure_count == 0
    end

    test "mode: :until_success - multiple tool_calls in one message. One succeeds and the other fails then succeeds",
         %{
           hello_world: hello_world,
           fail_once: fail_once
         } do
      # Made NOT LIVE here
      fake_messages = [
        new_function_calls!([
          ToolCall.new!(%{call_id: "call_fakeABC", name: "hello_world", arguments: nil}),
          ToolCall.new!(%{call_id: "call_fake123", name: "fail_once", arguments: nil})
        ])
      ]

      expect(ChatOpenAI, :call, 2, fn _model, _messages, _tools ->
        {:ok, fake_messages}
      end)

      {:ok, updated_chain} =
        %{llm: ChatOpenAI.new!(%{stream: false}), verbose: false}
        |> LLMChain.new!()
        |> LLMChain.add_tools([hello_world, fail_once])
        |> LLMChain.add_message(Message.new_system!())
        |> LLMChain.add_message(Message.new_user!("Execute the fail_once tool."))
        |> LLMChain.run(mode: :until_success)

      assert updated_chain.last_message.role == :tool

      assert [%ToolResult{is_error: false}, %ToolResult{is_error: false}] =
               updated_chain.last_message.tool_results

      assert updated_chain.current_failure_count == 0
    end

    test "mode: :until_success - fails after max count", %{fail_func: fail_func} do
      # Made NOT LIVE here
      fake_messages = [
        new_function_calls!([
          ToolCall.new!(%{call_id: "call_fake123", name: "fail_func", arguments: nil})
        ])
      ]

      # expect 3 calls
      expect(ChatOpenAI, :call, 3, fn _model, _messages, _tools ->
        {:ok, fake_messages}
      end)

      {:error, updated_chain, %LangChainError{} = reason} =
        %{llm: ChatOpenAI.new!(%{stream: false}), verbose: false}
        |> LLMChain.new!()
        |> LLMChain.add_tools([fail_func])
        |> LLMChain.add_message(Message.new_system!())
        |> LLMChain.add_message(Message.new_user!("Execute the fail_func tool."))
        |> LLMChain.run(mode: :until_success)

      assert reason.type == "exceeded_failure_count"
      assert reason.message == "Exceeded max failure count"
      assert updated_chain.current_failure_count == 3
    end

    test "mode: :step - last message is user message -> returns tool calls and stops", %{
      chain: chain,
      hello_world: hello_world
    } do
      fake_messages = [
        new_function_calls!([
          ToolCall.new!(%{call_id: "call_fake123", name: "hello_world", arguments: nil})
        ])
      ]

      expect(ChatOpenAI, :call, fn _model, _messages, _tools ->
        {:ok, fake_messages}
      end)

      {:ok, updated_chain} =
        chain
        |> LLMChain.add_tools([hello_world])
        |> LLMChain.add_message(Message.new_system!())
        |> LLMChain.add_message(Message.new_user!("Say hello!"))
        |> LLMChain.run(mode: :step)

      assert updated_chain.last_message.role == :assistant

      assert [%ToolCall{name: "hello_world", call_id: "call_fake123"}] =
               updated_chain.last_message.tool_calls

      assert updated_chain.needs_response == true

      assert length(updated_chain.exchanged_messages) == 1
      assert hd(updated_chain.exchanged_messages) == updated_chain.last_message
    end

    test "mode: :step - last message is tool call -> executes tool and returns tool results", %{
      chain: chain,
      hello_world: hello_world
    } do
      tool_call_message = new_function_call!("call_fake123", "hello_world", "{}")

      chain_with_tool_call =
        chain
        |> LLMChain.add_tools([hello_world])
        |> LLMChain.add_message(Message.new_system!())
        |> LLMChain.add_message(Message.new_user!("Say hello!"))
        |> LLMChain.add_message(tool_call_message)

      {:ok, updated_chain} = LLMChain.run(chain_with_tool_call, mode: :step)

      assert updated_chain.last_message.role == :tool

      assert [
               %ToolResult{
                 content: [
                   %LangChain.Message.ContentPart{
                     type: :text,
                     content: "Hello world!",
                     options: []
                   }
                 ],
                 tool_call_id: "call_fake123",
                 is_error: false
               }
             ] =
               updated_chain.last_message.tool_results

      assert updated_chain.needs_response == true

      assert length(updated_chain.exchanged_messages) == 1
      assert hd(updated_chain.exchanged_messages) == updated_chain.last_message
    end

    test "mode: :step - last message is tool result -> returns assistant comment", %{
      chain: chain,
      hello_world: hello_world
    } do
      tool_result = ToolResult.new!(%{tool_call_id: "call_fake123", content: "Hello world!"})
      tool_result_message = Message.new_tool_result!(%{content: nil, tool_results: [tool_result]})

      chain_with_tool_result =
        chain
        |> LLMChain.add_tools([hello_world])
        |> LLMChain.add_message(Message.new_system!())
        |> LLMChain.add_message(Message.new_user!("Say hello!"))
        |> LLMChain.add_message(new_function_call!("call_fake123", "hello_world", "{}"))
        |> LLMChain.add_message(tool_result_message)

      fake_messages = [
        Message.new_assistant!(%{content: "I said hello using the hello_world function!"})
      ]

      expect(ChatOpenAI, :call, fn _model, _messages, _tools ->
        {:ok, fake_messages}
      end)

      {:ok, updated_chain} = LLMChain.run(chain_with_tool_result, mode: :step)

      assert updated_chain.last_message.role == :assistant

      assert updated_chain.last_message.content == [
               %LangChain.Message.ContentPart{
                 type: :text,
                 content: "I said hello using the hello_world function!",
                 options: []
               }
             ]

      assert updated_chain.last_message.tool_calls == []
      assert updated_chain.needs_response == false

      assert length(updated_chain.exchanged_messages) == 1
      assert hd(updated_chain.exchanged_messages) == updated_chain.last_message
    end

    test "mode: :step - multiple steps to complete a full interaction", %{
      chain: chain,
      greet: greet
    } do
      expect(ChatOpenAI, :call, fn _model, _messages, _tools ->
        {:ok,
         [
           new_function_calls!([
             ToolCall.new!(%{
               call_id: "call_greet",
               name: "greet",
               arguments: %{"name" => "Alice"}
             })
           ])
         ]}
      end)

      {:ok, step1_chain} =
        chain
        |> LLMChain.add_tools([greet])
        |> LLMChain.add_message(Message.new_system!())
        |> LLMChain.add_message(Message.new_user!("Please greet Alice"))
        |> LLMChain.run(mode: :step)

      assert step1_chain.last_message.role == :assistant
      assert [%ToolCall{name: "greet"}] = step1_chain.last_message.tool_calls
      assert step1_chain.needs_response == true

      {:ok, step2_chain} = LLMChain.run(step1_chain, mode: :step)

      assert step2_chain.last_message.role == :tool

      assert [
               %ToolResult{
                 content: [
                   %LangChain.Message.ContentPart{type: :text, content: "Hi Alice!", options: []}
                 ],
                 is_error: false
               }
             ] =
               step2_chain.last_message.tool_results

      assert step2_chain.needs_response == true

      expect(ChatOpenAI, :call, fn _model, _messages, _tools ->
        {:ok, [Message.new_assistant!(%{content: "I've greeted Alice for you!"})]}
      end)

      {:ok, step3_chain} = LLMChain.run(step2_chain, mode: :step)

      assert step3_chain.last_message.role == :assistant

      assert step3_chain.last_message.content == [
               %LangChain.Message.ContentPart{
                 type: :text,
                 content: "I've greeted Alice for you!",
                 options: []
               }
             ]

      assert step3_chain.last_message.tool_calls == []
      assert step3_chain.needs_response == false
    end

    test "mode: :step - supports fallbacks", %{chain: chain, hello_world: hello_world} do
      expect(ChatOpenAI, :call, fn _model, _messages, _tools ->
        {:error, LangChainError.exception(type: "rate_limited", message: "Rate limited")}
      end)

      expect(ChatAnthropic, :call, fn _model, _messages, _tools ->
        {:ok,
         [
           new_function_calls!([
             ToolCall.new!(%{call_id: "call_fallback", name: "hello_world", arguments: nil})
           ])
         ]}
      end)

      {:ok, updated_chain} =
        chain
        |> LLMChain.add_tools([hello_world])
        |> LLMChain.add_message(Message.new_system!())
        |> LLMChain.add_message(Message.new_user!("Say hello!"))
        |> LLMChain.run(
          mode: :step,
          with_fallbacks: [ChatAnthropic.new!(%{stream: false})]
        )

      assert updated_chain.last_message.role == :assistant

      assert [%ToolCall{name: "hello_world", call_id: "call_fallback"}] =
               updated_chain.last_message.tool_calls
    end

    test "mode: :step - handles tool execution errors", %{chain: chain, fail_func: fail_func} do
      tool_call_message = new_function_call!("call_fail", "fail_func", "{}")

      chain_with_tool_call =
        chain
        |> LLMChain.add_tools([fail_func])
        |> LLMChain.add_message(Message.new_system!())
        |> LLMChain.add_message(Message.new_user!("Execute the failing function"))
        |> LLMChain.add_message(tool_call_message)

      {:ok, updated_chain} = LLMChain.run(chain_with_tool_call, mode: :step)

      assert updated_chain.last_message.role == :tool

      assert [
               %ToolResult{
                 content: [
                   %LangChain.Message.ContentPart{
                     type: :text,
                     content: "Not what I wanted",
                     options: []
                   }
                 ],
                 is_error: true
               }
             ] =
               updated_chain.last_message.tool_results

      assert updated_chain.current_failure_count == 1
      assert updated_chain.needs_response == true
    end

    test "mode: :step - processes message processors correctly", %{chain: chain} do
      fake_messages = [
        Message.new_assistant!(%{content: "Initial response"})
      ]

      expect(ChatOpenAI, :call, fn _model, _messages, _tools ->
        {:ok, fake_messages}
      end)

      {:ok, updated_chain} =
        chain
        |> LLMChain.message_processors([&fake_success_processor/2])
        |> LLMChain.add_message(Message.new_system!())
        |> LLMChain.add_message(Message.new_user!("Test message"))
        |> LLMChain.run(mode: :step)

      assert updated_chain.last_message.processed_content == "Initial response *"

      assert updated_chain.needs_response == false
    end

    test "with_fallbacks: re-runs with next LLM after first fails" do
      # Made NOT LIVE here - handles two calls
      expect(ChatOpenAI, :call, fn _model, _messages, _tools ->
        # IO.puts "FAKE OpenAI ERROR RESULT RETURNED"
        {:error,
         LangChainError.exception(type: "too_many_requests", message: "Too many requests!")}
      end)

      expect(ChatAnthropic, :call, fn _model, _messages, _tools ->
        {:ok,
         [
           Message.new_assistant!(%{content: "fallback worked!"})
         ]}
      end)

      {:ok, updated_chain} =
        %{llm: ChatOpenAI.new!(%{stream: false})}
        |> LLMChain.new!()
        |> LLMChain.add_message(Message.new_system!())
        |> LLMChain.add_message(Message.new_user!("Why is the sky blue?"))
        |> LLMChain.run(with_fallbacks: [ChatAnthropic.new!(%{stream: false})])

      # stopped after processing a successful assistant response
      assert updated_chain.last_message.role == :assistant
      assert updated_chain.last_message.content == [ContentPart.text!("fallback worked!")]
    end

    test "with_fallbacks: runs each LLM option and returns when all failed" do
      # Made NOT LIVE here - handles two calls
      expect(ChatOpenAI, :call, fn _model, _messages, _tools ->
        # IO.puts "FAKE OpenAI ERROR RESULT RETURNED"
        {:error,
         LangChainError.exception(type: "too_many_requests", message: "Too many requests!")}
      end)

      expect(ChatAnthropic, :call, fn _model, _messages, _tools ->
        {:error, LangChainError.exception(type: "overloaded", message: "Overloaded")}
      end)

      {:error, _updated_chain, reason} =
        %{llm: ChatOpenAI.new!(%{stream: false})}
        |> LLMChain.new!()
        |> LLMChain.add_message(Message.new_system!())
        |> LLMChain.add_message(Message.new_user!("Why is the sky blue?"))
        |> LLMChain.run(with_fallbacks: [ChatAnthropic.new!(%{stream: false})])

      assert %LangChainError{
               type: "all_fallbacks_failed",
               message: "Failed all attempts to generate response"
             } == reason
    end

    test "with_fallbacks: runs before_fallback function and uses the resulting chain" do
      # Made NOT LIVE here - handles two calls
      expect(ChatOpenAI, :call, fn _model, _messages, _tools ->
        # IO.puts "FAKE OpenAI ERROR RESULT RETURNED"
        {:error,
         LangChainError.exception(type: "too_many_requests", message: "Too many requests!")}
      end)

      expect(ChatAnthropic, :call, fn _model, _messages, _tools ->
        {:ok, Message.new_assistant!(%{content: "Claude says it's because it's not red."})}
      end)

      {:ok, updated_chain} =
        %{llm: ChatOpenAI.new!(%{stream: false})}
        |> LLMChain.new!()
        |> LLMChain.add_message(Message.new_system!("OpenAI system prompt"))
        |> LLMChain.add_message(Message.new_user!("Why is the sky blue?"))
        |> LLMChain.run(
          with_fallbacks: [
            ChatAnthropic.new!(%{stream: false})
          ],
          before_fallback: fn chain ->
            send(self(), :before_fallback_fired)

            case chain.llm do
              %ChatAnthropic{} ->
                # replace the system message
                %LLMChain{
                  chain
                  | messages:
                      Utils.replace_system_message!(
                        chain.messages,
                        Message.new_system!("Anthropic system prompt")
                      )
                }

              _open_ai ->
                chain
            end
          end
        )

      assert [system_msg | _rest] = updated_chain.messages
      assert system_msg.role == :system
      assert system_msg.content == [ContentPart.text!("Anthropic system prompt")]
      assert updated_chain.last_message.role == :assistant

      assert updated_chain.last_message.content == [
               ContentPart.text!("Claude says it's because it's not red.")
             ]

      assert_received :before_fallback_fired
    end
  end

  describe "run_until_tool_used/3" do
    test "supports multiple tool calls being made and stopping when the specific tool is called",
         %{greet: greet, sync: do_thing} do
      # Made NOT LIVE here
      expect(ChatOpenAI, :call, fn _model, _messages, _tools ->
        {:ok,
         new_function_calls!([
           ToolCall.new!(%{
             call_id: "call_fakeGreet",
             name: "greet",
             arguments: %{"name" => "Tim"}
           })
         ])}
      end)

      expect(ChatOpenAI, :call, fn _model, _messages, _tools ->
        {:ok,
         new_function_calls!([
           ToolCall.new!(%{call_id: "call_fakeDoThing", name: "do_thing", arguments: nil})
         ])}
      end)

      {:ok, updated_chain, tool_result} =
        %{llm: ChatOpenAI.new!(%{stream: false}), verbose: false}
        |> LLMChain.new!()
        |> LLMChain.add_tools([greet, do_thing])
        |> LLMChain.add_message(Message.new_system!())
        |> LLMChain.add_message(Message.new_user!("Say hello and then call do_thing."))
        |> LLMChain.run_until_tool_used("do_thing")

      assert updated_chain.last_message.role == :tool

      assert %ToolResult{is_error: false} = tool_result
      assert tool_result.name == "do_thing"

      assert updated_chain.current_failure_count == 0
    end

    test "supports multiple tool calls being made and stopping when the specific tool from the tool list is called",
         %{greet: greet, sync: do_thing, hello_world: hello_world} do
      # Made NOT LIVE here
      expect(ChatOpenAI, :call, fn _model, _messages, _tools ->
        {:ok,
         new_function_calls!([
           ToolCall.new!(%{
             call_id: "call_fakeGreet",
             name: "greet",
             arguments: %{"name" => "Tim"}
           })
         ])}
      end)

      expect(ChatOpenAI, :call, fn _model, _messages, _tools ->
        {:ok,
         new_function_calls!([
           ToolCall.new!(%{call_id: "call_hello_world", name: "hello_world", arguments: nil})
         ])}
      end)

      {:ok, updated_chain, tool_result} =
        %{llm: ChatOpenAI.new!(%{stream: false}), verbose: false}
        |> LLMChain.new!()
        |> LLMChain.add_tools([greet, do_thing, hello_world])
        |> LLMChain.add_message(Message.new_system!())
        |> LLMChain.add_message(Message.new_user!("Say hello and then call hello_world."))
        |> LLMChain.run_until_tool_used(["do_thing", "hello_world"])

      assert updated_chain.last_message.role == :tool

      assert %ToolResult{is_error: false} = tool_result
      assert tool_result.name == "hello_world"

      assert updated_chain.current_failure_count == 0
    end

    test "supports stopping after max_runs attempts", %{greet: greet, sync: do_thing} do
      # Made NOT LIVE here
      expect(ChatOpenAI, :call, 3, fn _model, _messages, _tools ->
        {:ok,
         new_function_calls!([
           ToolCall.new!(%{call_id: "call_fake123", name: "greet", arguments: %{"name" => "Tim"}})
         ])}
      end)

      {:error, _updated_chain, error} =
        %{llm: ChatOpenAI.new!(%{stream: false}), verbose: false}
        |> LLMChain.new!()
        |> LLMChain.add_tools([greet, do_thing])
        |> LLMChain.add_message(Message.new_system!())
        |> LLMChain.add_message(Message.new_user!("Say hello and then call do_thing."))
        # will not be called. Tool call is not made.
        |> LLMChain.run_until_tool_used("do_thing", max_runs: 3)

      assert error.type == "exceeded_max_runs"
      assert error.message == "Exceeded maximum number of runs"
    end

    test "returns error when tool_name does not exist in available tools", %{greet: greet} do
      {:error, _updated_chain, error} =
        %{llm: ChatOpenAI.new!(%{stream: false}), verbose: false}
        |> LLMChain.new!()
        |> LLMChain.add_tools([greet])
        |> LLMChain.add_message(Message.new_system!())
        |> LLMChain.add_message(Message.new_user!("Say hello to Tim."))
        |> LLMChain.run_until_tool_used("non_existent_tool", max_runs: 1)

      assert error.type == "invalid_tool_name"
      assert error.message == "Tool name 'non_existent_tool' not found in available tools"
    end
  end

  describe "increment_current_failure_count/1" do
    test "increments the current_failure_count", %{chain: chain} do
      updated_chain_1 =
        chain
        |> LLMChain.increment_current_failure_count()

      assert updated_chain_1.current_failure_count == 1

      updated_chain_2 =
        updated_chain_1
        |> LLMChain.increment_current_failure_count()

      assert updated_chain_2.current_failure_count == 2
    end
  end

  describe "reset_current_failure_count/1" do
    test "resets the current_failure_count to 0", %{chain: chain} do
      updated_chain =
        chain
        |> LLMChain.increment_current_failure_count()
        |> LLMChain.increment_current_failure_count()
        |> LLMChain.reset_current_failure_count()

      assert updated_chain.current_failure_count == 0
    end
  end

  describe "update_custom_context/3" do
    test "updates using merge by default" do
      chain =
        LLMChain.new!(%{
          llm: ChatOpenAI.new!(%{stream: false}),
          custom_context: %{existing: "a", count: 1}
        })

      updated_1 = LLMChain.update_custom_context(chain, %{count: 5})
      assert updated_1.custom_context == %{existing: "a", count: 5}

      updated_2 = LLMChain.update_custom_context(updated_1, %{more: true}, as: :merge)
      assert updated_2.custom_context == %{existing: "a", count: 5, more: true}
    end

    test "handles update when custom_context is nil" do
      chain =
        LLMChain.new!(%{
          llm: ChatOpenAI.new!(%{stream: false}),
          custom_context: nil
        })

      assert chain.custom_context == nil

      updated = LLMChain.update_custom_context(chain, %{some: :thing})
      assert updated.custom_context == %{some: :thing}
    end

    test "support updates using replace" do
      chain =
        LLMChain.new!(%{
          llm: ChatOpenAI.new!(%{stream: false}),
          custom_context: %{count: 1}
        })

      updated = LLMChain.update_custom_context(chain, %{color: "blue"}, as: :replace)
      assert updated.custom_context == %{color: "blue"}
    end
  end

  describe "execute_tool_calls/2" do
    test "returns chain unmodified if no tool calls" do
      chain =
        LLMChain.new!(%{
          llm: ChatOpenAI.new!(%{stream: false}),
          custom_context: %{count: 1}
        })
        |> LLMChain.add_message(Message.new_system!())
        |> LLMChain.add_message(Message.new_assistant!(%{content: "What's up?"}))

      assert chain == LLMChain.execute_tool_calls(chain)
    end

    test "fires a single tool call that generates expected Tool result message", %{
      hello_world: hello_world
    } do
      chain =
        LLMChain.new!(%{
          llm: ChatOpenAI.new!(%{stream: false}),
          custom_context: %{count: 1}
        })
        |> LLMChain.add_tools(hello_world)
        |> LLMChain.add_message(Message.new_system!())
        |> LLMChain.add_message(Message.new_user!("Say hello!"))
        |> LLMChain.add_message(new_function_call!("call_fake123", "hello_world", "{}"))

      updated_chain = LLMChain.execute_tool_calls(chain)

      assert %Message{role: :tool} = updated_chain.last_message
      # result of execution
      [%ToolResult{} = result] = updated_chain.last_message.tool_results
      assert result.content == [ContentPart.text!("Hello world!")]
      # tool response is linked to original call
      assert result.tool_call_id == "call_fake123"
    end

    test "supports executing multiple tool calls from a single request and returns results in a single message",
         %{
           hello_world: hello_world,
           greet: greet
         } do
      test_pid = self()

      handler = %{
        on_message_processed: fn _chain, tool_msg ->
          send(test_pid, {:message_processed_callback_fired, tool_msg})
        end,
        on_tool_response_created: fn _chain, tool_msg ->
          send(test_pid, {:response_created_callback_fired, tool_msg})
        end
      }

      chain =
        LLMChain.new!(%{
          llm: ChatOpenAI.new!(%{stream: false}),
          custom_context: %{count: 1}
        })
        |> LLMChain.add_tools([hello_world, greet])
        |> LLMChain.add_message(Message.new_system!())
        |> LLMChain.add_message(Message.new_user!("Say hello!"))
        |> LLMChain.add_message(
          new_function_calls!([
            ToolCall.new!(%{
              call_id: "call_fake123",
              name: "greet",
              arguments: %{"name" => "Tim"}
            }),
            ToolCall.new!(%{call_id: "call_fake234", name: "hello_world", arguments: nil}),
            ToolCall.new!(%{
              call_id: "call_fake345",
              name: "greet",
              arguments: %{"name" => "Jane"}
            })
          ])
        )

      # hookup callback and execute the tools
      updated_chain =
        chain
        |> LLMChain.add_callback(handler)
        |> LLMChain.execute_tool_calls()

      %Message{role: :tool} = tool_message = updated_chain.last_message

      [tool1, tool2, tool3] = tool_message.tool_results

      assert_receive {:message_processed_callback_fired, callback_message}
      assert %Message{role: :tool} = callback_message

      assert_receive {:response_created_callback_fired, callback_message}
      assert %Message{role: :tool} = callback_message
      assert [tool1, tool2, tool3] == callback_message.tool_results

      [%ToolResult{} = result1, result2, result3] = tool_message.tool_results

      assert result1.content == [ContentPart.text!("Hi Tim!")]
      assert result1.tool_call_id == "call_fake123"
      assert result1.is_error == false

      assert result2.content == [ContentPart.text!("Hello world!")]
      assert result2.tool_call_id == "call_fake234"
      assert result2.is_error == false

      assert result3.content == [ContentPart.text!("Hi Jane!")]
      assert result3.tool_call_id == "call_fake345"
      assert result3.is_error == false
    end

    test "executes tool calls to synchronous functions", %{sync: sync} do
      assert sync.async == false

      chain =
        LLMChain.new!(%{
          llm: ChatOpenAI.new!(%{stream: false}),
          custom_context: %{count: 1}
        })
        |> LLMChain.add_tools(sync)
        |> LLMChain.add_message(Message.new_system!())
        |> LLMChain.add_message(new_function_call!("call_fake123", "do_thing", "{}"))

      updated_chain = LLMChain.execute_tool_calls(chain)

      %Message{role: :tool} = result_message = updated_chain.last_message
      # result of execution
      [%ToolResult{} = result] = result_message.tool_results
      assert result.tool_call_id == "call_fake123"
      assert result.is_error == false
    end

    test "catches exceptions from executed function and returns Tool result with error message" do
      error_function =
        Function.new!(%{
          name: "go_time",
          description: "Raises an exception.",
          function: fn _args, _context -> raise RuntimeError, "Stuff went boom!" end
        })

      chain =
        LLMChain.new!(%{
          llm: ChatOpenAI.new!(%{stream: false}),
          custom_context: %{count: 1}
        })
        |> LLMChain.add_tools(error_function)
        |> LLMChain.add_message(Message.new_system!())
        |> LLMChain.add_message(Message.new_user!("It's go time!"))
        |> LLMChain.add_message(new_function_call!("call_fake123", "go_time", "{}"))

      updated_chain = LLMChain.execute_tool_calls(chain)

      assert updated_chain.last_message.role == :tool
      [%ToolResult{} = result] = updated_chain.last_message.tool_results

      assert result.content == [
               ContentPart.text!(
                 "ERROR: (RuntimeError) Stuff went boom! at test/chains/llm_chain_test.exs:#{__ENV__.line - 20}: anonymous fn/2 in LangChain.Chains.LLMChainTest.\"test execute_tool_calls/2 catches exceptions from executed function and returns Tool result with error message\"/1"
               )
             ]

      assert result.is_error == true
    end

    test "returns error tool result when tool_call is a hallucination" do
      chain =
        LLMChain.new!(%{
          llm: ChatOpenAI.new!(%{stream: false}),
          custom_context: %{count: 1}
        })
        # NOTE: No tools added
        |> LLMChain.add_message(Message.new_system!())
        |> LLMChain.add_message(Message.new_user!("It's go time!"))
        |> LLMChain.add_message(new_function_call!("call_fake123", "greet", %{"name" => "Tim"}))

      updated_chain = LLMChain.execute_tool_calls(chain)
      %Message{role: :tool} = result_message = updated_chain.last_message

      # increments current_failure_count
      assert updated_chain.current_failure_count == 1

      # result of execution
      [%ToolResult{} = result] = result_message.tool_results
      assert result.content == [ContentPart.text!("Tool call made to greet but tool not found")]
      # tool response is linked to original call
      assert result.tool_call_id == "call_fake123"
      assert result.is_error == true
    end

    test "on successful execution, clears resets the failure count", %{
      chain: chain,
      hello_world: hello_world
    } do
      chain =
        chain
        |> LLMChain.add_tools(hello_world)
        |> LLMChain.increment_current_failure_count()
        |> LLMChain.increment_current_failure_count()
        |> LLMChain.add_message(new_function_call!("call_fake123", "hello_world", "{}"))

      assert chain.current_failure_count == 2

      updated_chain = LLMChain.execute_tool_calls(chain)
      %Message{role: :tool} = updated_chain.last_message

      # resets the current_failure_count after processing successfully
      assert updated_chain.current_failure_count == 0
    end

    test "supports returning processed_content to ToolResult", %{chain: chain, get_date: get_date} do
      chain =
        chain
        |> LLMChain.add_tools(get_date)
        |> LLMChain.add_message(new_function_call!("call_fake123", "get_date", "{}"))

      updated_chain = LLMChain.execute_tool_calls(chain)
      # get the 1 expected tool result
      %Message{role: :tool, tool_results: [%ToolResult{} = result]} = updated_chain.last_message
      assert result.name == "get_date"
      assert result.content == [ContentPart.text!("2024-11-01")]
      assert result.processed_content == ~D[2024-11-01]
    end

    test "supports returning a ToolResult from a tool execution", %{chain: chain} do
      returns_tool_result =
        Function.new!(%{
          name: "explicit_return",
          description: "Returns a fully setup ToolResult",
          display_text: "Explicit return",
          function: fn _args, _context ->
            {:ok,
             %ToolResult{
               content: [ContentPart.text!("Hello!", cache_control: true)],
               options: [custom: 1]
             }}
          end
        })

      chain =
        chain
        |> LLMChain.add_tools(returns_tool_result)
        |> LLMChain.add_message(new_function_call!("test-call-id-192", "explicit_return", "{}"))

      updated_chain = LLMChain.execute_tool_calls(chain)
      # get the 1 expected tool result
      %Message{role: :tool, tool_results: [%ToolResult{} = result]} = updated_chain.last_message
      assert result.tool_call_id == "test-call-id-192"
      assert result.name == "explicit_return"
      assert result.display_text == "Explicit return"
      assert result.content == [ContentPart.text!("Hello!", cache_control: true)]
      assert result.processed_content == nil
      assert result.options == [custom: 1]
    end
  end

  describe "add_callback/2" do
    test "appends a callback handler to the list", %{chat: chat} do
      handler1 = %{on_message_processed: fn _chain, _msg -> IO.puts("PROCESSED 1!") end}
      handler2 = %{on_message_processed: fn _chain, _msg -> IO.puts("PROCESSED 2!") end}

      chain =
        %{llm: chat}
        |> LLMChain.new!()
        |> LLMChain.add_callback(handler1)

      assert chain.callbacks == [handler1]

      updated_chain = LLMChain.add_callback(chain, handler2)
      assert updated_chain.callbacks == [handler1, handler2]
    end

    test "appends a callback handler to the chain's LLM", %{chat: chat} do
      handler1 = %{on_llm_new_message: fn %LLMChain{} = _chain, _msg -> IO.puts("MESSAGE 1!") end}
      handler2 = %{on_llm_new_message: fn %LLMChain{} = _chain, _msg -> IO.puts("MESSAGE 2!") end}

      # none to start with
      assert chat.callbacks == []

      chain =
        %{llm: chat}
        |> LLMChain.new!()
        |> LLMChain.add_callback(handler1)
        |> LLMChain.add_callback(handler2)

      assert chain.callbacks == [handler1, handler2]
    end
  end

  # TODO: Sequential chains
  # https://js.langchain.com/docs/modules/chains/sequential_chain

  # TODO: Index related chains
  # https://js.langchain.com/docs/modules/chains/index_related_chains/

  # TODO: Other Chains
  # https://js.langchain.com/docs/modules/chains/other_chains/
end
