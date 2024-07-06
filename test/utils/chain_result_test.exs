defmodule LangChain.Utils.ChainResultTest do
  use ExUnit.Case

  doctest LangChain.Utils.ChainResult

  alias LangChain.Utils.ChainResult
  alias LangChain.Chains.LLMChain
  alias LangChain.Message
  alias LangChain.LangChainError
  alias LangChain.Message.ContentPart

  describe "to_string/1" do
    test "passes an error tuple through" do
      assert {:error, %LLMChain{}, "original error"} ==
               ChainResult.to_string({:error, %LLMChain{}, "original error"})
    end

    test "returns {:ok, answer} when valid" do
      chain = %LLMChain{last_message: Message.new_assistant!("the answer")}
      assert {:ok, "the answer"} == ChainResult.to_string(chain)
    end

    test "returns {:ok, answer} when message content parts are valid" do
      chain = %LLMChain{
        last_message: Message.new_assistant!(%{content: [ContentPart.text!("the answer")]})
      }

      assert {:ok, "the answer"} == ChainResult.to_string(chain)
    end

    test "returns error when no last message" do
      chain = %LLMChain{last_message: nil}
      assert {:error, chain, "No last message"} == ChainResult.to_string(chain)
    end

    test "returns error when incomplete last message" do
      chain = %LLMChain{
        last_message: Message.new!(%{role: :assistant, content: "Incomplete", status: :length})
      }

      assert {:error, chain, "Message is incomplete"} == ChainResult.to_string(chain)
    end

    test "returns error when last message is not from assistant" do
      chain = %LLMChain{
        last_message: Message.new_user!("The question")
      }

      assert {:error, chain, "Message is not from assistant"} == ChainResult.to_string(chain)
    end

    test "handles an LLMChain.run/2 success result" do
      message = Message.new_assistant!("the answer")
      chain = %LLMChain{last_message: message}
      assert {:ok, "the answer"} == ChainResult.to_string({:ok, chain, message})
    end
  end

  describe "to_string!/1" do
    test "returns string when valid" do
      chain = %LLMChain{last_message: Message.new_assistant!("the answer")}
      assert "the answer" == ChainResult.to_string!(chain)
    end

    test "raises LangChainError when invalid" do
      chain = %LLMChain{last_message: nil}

      assert_raise LangChainError, "No last message", fn ->
        ChainResult.to_string!(chain)
      end
    end
  end

  describe "to_map/3" do
    test "writes string result to map when valid" do
      data = %{thing: "one"}
      chain = %LLMChain{last_message: Message.new_assistant!("the answer")}
      assert {:ok, result} = ChainResult.to_map(chain, data, :answer)
      assert %{thing: "one", answer: "the answer"} == result
    end

    test "returns error tuple with reason when invalid" do
      data = %{thing: "one"}
      chain = %LLMChain{last_message: nil}
      assert {:error, _chain, "No last message"} = ChainResult.to_map(chain, data, :answer)
    end
  end

  describe "to_map!/3" do
    test "writes string result to map when valid" do
      data = %{thing: "one"}
      chain = %LLMChain{last_message: Message.new_assistant!("the answer")}
      result = ChainResult.to_map!(chain, data, :answer)
      assert %{thing: "one", answer: "the answer"} == result
    end

    test "raises error tuple with reason when invalid" do
      data = %{thing: "one"}
      chain = %LLMChain{last_message: nil}

      assert_raise LangChainError, "No last message", fn ->
        ChainResult.to_map!(chain, data, :answer)
      end
    end
  end
end
