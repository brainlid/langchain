defmodule LangChain.Chains.LLMChain.ModeTest do
  use LangChain.BaseCase
  use Mimic

  alias LangChain.Chains.LLMChain
  alias LangChain.ChatModels.ChatOpenAI
  alias LangChain.Message
  alias LangChain.Message.ToolCall
  alias LangChain.Function

  setup :verify_on_exit!

  defmodule TestMode do
    @behaviour LangChain.Chains.LLMChain.Mode

    @impl true
    def run(chain, opts) do
      case LLMChain.execute_step(chain) do
        {:ok, updated_chain} ->
          if Keyword.get(opts, :return_extra) do
            {:ok, updated_chain, :test_extra}
          else
            {:ok, updated_chain}
          end

        error ->
          error
      end
    end
  end

  defmodule PauseMode do
    @behaviour LangChain.Chains.LLMChain.Mode

    @impl true
    def run(chain, _opts) do
      {:pause, chain}
    end
  end

  setup do
    {:ok, chat} = ChatOpenAI.new(%{temperature: 0})
    chain = LLMChain.new!(%{llm: chat})
    %{chat: chat, chain: chain}
  end

  describe "execute_step/1" do
    test "executes single LLM call and returns assistant message", %{chain: chain} do
      expect(ChatOpenAI, :call, fn _model, _messages, _tools ->
        {:ok, [Message.new_assistant!("Hello!")]}
      end)

      chain = LLMChain.add_message(chain, Message.new_user!("Hi"))

      assert {:ok, updated_chain} = LLMChain.execute_step(chain)

      assert LangChain.Message.ContentPart.parts_to_string(updated_chain.last_message.content) ==
               "Hello!"

      assert updated_chain.last_message.role == :assistant
      assert updated_chain.needs_response == false
    end

    test "executes single LLM call with tool calls and sets needs_response", %{chain: chain} do
      tool_call = ToolCall.new!(%{call_id: "call_1", name: "hello_world", arguments: %{}})

      expect(ChatOpenAI, :call, fn _model, _messages, _tools ->
        {:ok, [Message.new_assistant!(%{tool_calls: [tool_call]})]}
      end)

      hello_world =
        Function.new!(%{
          name: "hello_world",
          description: "Says hello",
          function: fn _args, _context -> "Hello world!" end
        })

      chain =
        chain
        |> LLMChain.add_tools(hello_world)
        |> LLMChain.add_message(Message.new_user!("Hi"))

      assert {:ok, updated_chain} = LLMChain.execute_step(chain)
      assert updated_chain.needs_response == true
      assert Message.is_tool_call?(updated_chain.last_message)
    end

    test "returns error when max retries exceeded", %{chain: chain} do
      chain =
        chain
        |> LLMChain.add_message(Message.new_user!("Hi"))
        |> Map.put(:current_failure_count, 3)
        |> Map.put(:max_retry_count, 3)

      assert {:error, _chain, %LangChain.LangChainError{type: "exceeded_failure_count"}} =
               LLMChain.execute_step(chain)
    end
  end

  describe "run/2 with custom mode module" do
    test "dispatches to module's run/2", %{chain: chain} do
      expect(ChatOpenAI, :call, fn _model, _messages, _tools ->
        {:ok, [Message.new_assistant!("Hello!")]}
      end)

      chain = LLMChain.add_message(chain, Message.new_user!("Hi"))

      assert {:ok, updated_chain} = LLMChain.run(chain, mode: TestMode)

      assert LangChain.Message.ContentPart.parts_to_string(updated_chain.last_message.content) ==
               "Hello!"
    end

    test "passes through 3-tuple from custom mode", %{chain: chain} do
      expect(ChatOpenAI, :call, fn _model, _messages, _tools ->
        {:ok, [Message.new_assistant!("Hello!")]}
      end)

      chain = LLMChain.add_message(chain, Message.new_user!("Hi"))

      assert {:ok, _chain, :test_extra} =
               LLMChain.run(chain, mode: TestMode, return_extra: true)
    end

    test "passes through pause from custom mode", %{chain: chain} do
      chain = LLMChain.add_message(chain, Message.new_user!("Hi"))

      assert {:pause, _chain} = LLMChain.run(chain, mode: PauseMode)
    end

    test "passes opts to custom mode", %{chain: chain} do
      expect(ChatOpenAI, :call, fn _model, _messages, _tools ->
        {:ok, [Message.new_assistant!("Hello!")]}
      end)

      chain = LLMChain.add_message(chain, Message.new_user!("Hi"))

      # Without return_extra, should return 2-tuple
      assert {:ok, _chain} = LLMChain.run(chain, mode: TestMode)
    end
  end
end
