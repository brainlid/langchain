defmodule LangChain.CallbacksIntegrationTest do
  @moduledoc """
  Pins down how callbacks registered on a chat model relate to callbacks
  registered on the `LLMChain` that runs it.

  These run the real `ChatOpenAI.call/3` against a mocked `Req.post/1`, so the
  provider's own callback firing is exercised rather than stubbed out.
  """
  use ExUnit.Case, async: false
  use Mimic

  alias LangChain.Chains.LLMChain
  alias LangChain.ChatModels.ChatOpenAI
  alias LangChain.LangChainError
  alias LangChain.Message

  @response_body %{
    "id" => "chatcmpl-abc123",
    "object" => "chat.completion",
    "model" => "gpt-4o-mini",
    "choices" => [
      %{
        "message" => %{"role" => "assistant", "content" => "Hi"},
        "finish_reason" => "stop",
        "index" => 0
      }
    ],
    "usage" => %{"prompt_tokens" => 11, "completion_tokens" => 22, "total_tokens" => 33}
  }

  setup do
    stub(Req, :post, fn _req -> {:ok, %Req.Response{status: 200, body: @response_body}} end)
    :ok
  end

  # A model-tier handler. Chat models fire these with a single argument.
  defp model_handler(test_pid) do
    %{
      on_llm_new_message: fn message -> send(test_pid, {:model_tier, message}) end
    }
  end

  # A chain-tier handler. LLMChain curries itself in as the first argument.
  defp chain_handler(test_pid) do
    %{
      on_llm_new_message: fn _chain, message -> send(test_pid, {:chain_tier, message}) end
    }
  end

  defp run_chain(llm, chain_callbacks) do
    %{llm: llm, callbacks: chain_callbacks}
    |> LLMChain.new!()
    |> LLMChain.add_message(Message.new_user!("Hi"))
    |> LLMChain.run()
  end

  describe "calling a chat model directly" do
    test "callbacks assigned through new/1 fire" do
      test_pid = self()

      llm =
        ChatOpenAI.new!(%{
          model: "gpt-4o-mini",
          stream: false,
          callbacks: [model_handler(test_pid)]
        })

      assert {:ok, _} = ChatOpenAI.call(llm, [Message.new_user!("Hi")], [])

      assert_received {:model_tier, %Message{}}
    end
  end

  describe "handler arity" do
    test "a chain-tier (arity 2) handler assigned to a model fails the whole call" do
      test_pid = self()

      # Chat models fire their own callbacks with one argument, so a handler
      # written in the two-argument chain style blows up when invoked. The
      # BadArityError is caught by Callbacks.fire/3 and re-raised as a
      # LangChainError, which surfaces as a failed LLM call rather than a
      # skipped callback.
      chain_style_handler = %{
        on_llm_ratelimit_info: fn _model, headers -> send(test_pid, {:ratelimit, headers}) end
      }

      llm =
        ChatOpenAI.new!(%{
          model: "gpt-4o-mini",
          stream: false,
          callbacks: [chain_style_handler]
        })

      assert {:error, %LangChainError{message: message}} =
               ChatOpenAI.call(llm, [Message.new_user!("Hi")], [])

      assert message =~ "Callback handler for :on_llm_ratelimit_info raised an exception"
      assert message =~ "BadArityError"
      refute_received {:ratelimit, _}
    end

    test "a model-tier (arity 1) handler receives the ratelimit info" do
      test_pid = self()

      llm =
        ChatOpenAI.new!(%{
          model: "gpt-4o-mini",
          stream: false,
          callbacks: [%{on_llm_ratelimit_info: fn info -> send(test_pid, {:ratelimit, info}) end}]
        })

      assert {:ok, _} = ChatOpenAI.call(llm, [Message.new_user!("Hi")], [])

      assert_received {:ratelimit, %{}}
    end
  end

  describe "running the same model through an LLMChain" do
    test "model callbacks still fire; the chain does not discard them" do
      test_pid = self()

      llm =
        ChatOpenAI.new!(%{
          model: "gpt-4o-mini",
          stream: false,
          callbacks: [model_handler(test_pid)]
        })

      assert {:ok, %LLMChain{}} = run_chain(llm, [])

      assert_received {:model_tier, %Message{}}
    end

    test "chain callbacks fire" do
      test_pid = self()

      llm = ChatOpenAI.new!(%{model: "gpt-4o-mini", stream: false})

      assert {:ok, %LLMChain{}} = run_chain(llm, [chain_handler(test_pid)])

      assert_received {:chain_tier, %Message{}}
    end

    test "with both set, both fire" do
      test_pid = self()

      llm =
        ChatOpenAI.new!(%{
          model: "gpt-4o-mini",
          stream: false,
          callbacks: [model_handler(test_pid)]
        })

      assert {:ok, %LLMChain{}} = run_chain(llm, [chain_handler(test_pid)])

      assert_received {:chain_tier, %Message{}}
      assert_received {:model_tier, %Message{}}
    end

    test "the chain appends its own handlers to the model's, keeping both" do
      test_pid = self()

      llm =
        ChatOpenAI.new!(%{
          model: "gpt-4o-mini",
          stream: false,
          callbacks: [model_handler(test_pid)]
        })

      # capture what the chain actually passes down to the model
      expect(ChatOpenAI, :call, fn model, _messages, _tools ->
        send(test_pid, {:model_as_called, model.callbacks})
        {:ok, [Message.new_assistant!("Hi")]}
      end)

      assert {:ok, %LLMChain{}} = run_chain(llm, [chain_handler(test_pid)])

      # the model's own handler is first, the chain's wrapped handler follows
      assert_received {:model_as_called, [model_own, chain_wrapped]}
      assert is_function(model_own.on_llm_new_message, 1)
      assert is_function(chain_wrapped.on_llm_new_message, 1)
    end

    test "a model handler is not called twice when the chain has none" do
      test_pid = self()

      llm =
        ChatOpenAI.new!(%{
          model: "gpt-4o-mini",
          stream: false,
          callbacks: [model_handler(test_pid)]
        })

      assert {:ok, %LLMChain{}} = run_chain(llm, [])

      assert_received {:model_tier, %Message{}}
      refute_received {:model_tier, _}
    end
  end
end
