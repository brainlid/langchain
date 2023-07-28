defmodule Langchain.Tools.CalculatorTest do
  alias Langchain.Chains.LLMChain
  use Langchain.BaseCase

  doctest Langchain.Tools.Calculator
  alias Langchain.Tools.Calculator
  alias Langchain.Function
  alias Langchain.ChatModels.ChatOpenAI
  alias Langchain.PromptTemplate

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
      {:ok, updated_chain_1, %Message{} = message} =
        LLMChain.new!(%{
          llm: ChatOpenAI.new!(%{temperature: 0}),
          messages: [
            Message.new_user!("Answer the following math question: What is 100 + 300 - 200?")
          ],
          verbose: true
        })
        |> LLMChain.add_functions(Calculator.new!())
        |> LLMChain.run()

      assert updated_chain_1.last_message == message
      assert message.role == :function_call
      assert message.arguments == %{"expression" => "100 + 300 - 200"}

      # execute/evaluate the requested function and respond
      {:ok, updated_chain_2, %Message{} = final_answer} =
        updated_chain_1
        |> LLMChain.execute_function()
        |> LLMChain.run()

      assert final_answer.role == :assistant
      assert final_answer.content == "The answer is 200."

      # LLMChain.run(while_needs_response: true)
      # LLMChain.run(until_complete: true)
      # LLMChain.run(until_response: true)

      #TODO: create a "run_until_answered"? Keeps evaluating functions and resubmitting.
      # TODO: Would ideally be run in a separate task/process with message passing for results.
    end
  end
end
