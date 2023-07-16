defmodule Langchain.Tools.CalculatorTest do
  use Langchain.BaseCase

  doctest Langchain.Tools.Calculator
  alias Langchain.Tools.Calculator
  alias Langchain.Functions.Function

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

  describe "execute/2" do
    test "evaluates the expression returning the result" do
      assert "14" == Calculator.execute(%{"expression" => "1 + 2 + 3 + (2 * 4)"}, nil)
    end

    test "returns an error when evaluation fails" do
      assert "ERROR" == Calculator.execute(%{"expression" => "cow + dog"}, nil)
    end
  end
end
