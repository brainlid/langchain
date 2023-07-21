defmodule Langchain.Tools.CalculatorTest do
  alias Langchain.Chains.LLMChain
  use Langchain.BaseCase

  doctest Langchain.Tools.Calculator
  alias Langchain.Tools.Calculator
  alias Langchain.Functions.Function
  alias Langchain.ChatModels.ChatOpenAI

  describe "new/0" do
    test "defines the function correctly" do
      assert {:ok, %Function{} = function} = Calculator.new()
      assert function.name == "calculator"
      assert function.description == "Perform basic math calculations"
      assert function.function != nil

      assert function.parameters_schema == %{
               type: "object",
               properties: %{
                 expression: %{type: "string", description: "A simple mathematical expression."}
               },
               required: ["expression"]
             }
    end

    test "assigned function can be executed" do
      {:ok, calc} = Calculator.new()
      assert "3" == calc.function.(%{"expression" => "1 + 2"}, nil)
    end
  end

  describe "new!/0" do
    test "returns the function" do
      assert %Function{name: "calculator"} = Calculator.new!()
    end
  end

  describe "execute/2" do
    test "evaluates the expression returning the result" do
      assert "14" == Calculator.execute(%{"expression" => "1 + 2 + 3 + (2 * 4)"}, nil)
    end

    test "returns an error when evaluation fails" do
      assert "ERROR" == Calculator.execute(%{"expression" => "cow + dog"}, nil)
    end
  end

  @tag :live_call
  describe "live test" do
    test "works when used with a live LLM" do
      {:ok, chat} = ChatOpenAI.new(%{temperature: 0})

      {:ok, chain} =
        LLMChain.new(%{
          llm: chat,
          # prompt: ["How many cookies will I get share my 35 cookies among 4 friends and myself? Additionally, what is 5 to the power of 2?"],
          prompt: ["Answer the following math question: What is 100 + 300 - 200?"],
          verbose: true
        })

      chain = LLMChain.add_functions(chain, Calculator.new!())
      {:ok, %Message{} = message} = LLMChain.call_chat(chain)
      assert message.role == :function_call
      assert message.arguments == %{"expression" => "100 + 300 - 200"}
    end
  end
end
