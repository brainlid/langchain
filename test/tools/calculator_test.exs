defmodule Langchain.Tools.CalculatorTest do
  alias Langchain.Chains.LLMChain
  use Langchain.BaseCase

  doctest Langchain.Tools.Calculator
  alias Langchain.Tools.Calculator
  alias Langchain.Function
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

  describe "live test" do
    @tag :live_call
    test "performs repeated calls until complete with a live LLM" do
      callback = fn %Message{} = msg ->
        send(self(), {:callback_msg, msg})
      end

      {:ok, updated_chain, %Message{} = message} =
        LLMChain.new!(%{
          llm: ChatOpenAI.new!(%{temperature: 0, callback_fn: callback}),
          verbose: true
        })
        |> LLMChain.add_message(
          Message.new_user!("Answer the following math question: What is 100 + 300 - 200?")
        )
        |> LLMChain.add_functions(Calculator.new!())
        |> LLMChain.run(while_needs_response: true)

      assert updated_chain.last_message == message
      assert message.role == :assistant
      assert message.content == "The answer is 200."

      # assert received multiple messages as callbacks
      assert_received {:callback_msg, message}
      assert message.role == :function_call
      assert message.function_name == "calculator"
      assert message.arguments == %{"expression" => "100 + 300 - 200"}

      assert_received {:callback_msg, message}
      assert message.role == :assistant
      assert message.content == "The answer is 200."

      assert updated_chain.last_message == message
    end
  end
end
