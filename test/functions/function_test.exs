defmodule Langchain.Functions.FunctionTest do
  use ExUnit.Case

  doctest Langchain.Functions.Function

  alias Langchain.Functions.Function
  alias Langchain.Functions.FunctionParameter
  alias Langchain.ForOpenAIApi

  defp hello_world(_args, _context) do
    "Hello world!"
  end

  describe "new/1" do
    test "works with minimal attrs" do
      assert {:ok, %Function{} = fun} = Function.new(%{"name" => "hello_world"})
      assert fun.name == "hello_world"
    end

    test "returns error when invalid" do
      assert {:error, changeset} = Function.new(%{"name" => nil})
      refute changeset.valid?
      assert {"can't be blank", _} = changeset.errors[:name]
    end

    test "supports name, description, an 1 param" do
      {:ok, fun} =
        Function.new(%{
          "name" => "say_hi",
          "description" => "Provide a friendly greeting.",
          "parameters" => [%{name: "p1", type: "string"}]
        })

      assert fun.name == "say_hi"
      assert fun.description == "Provide a friendly greeting."
      assert fun.parameters == [%FunctionParameter{name: "p1", type: "string"}]
    end

    test "supports required param" do
      {:ok, fun} =
        Function.new(%{
          "name" => "say_hi",
          "parameters" => [%{name: "greeting", type: "string"}],
          "required" => ["greeting"]
        })

      assert fun.name == "say_hi"
      assert fun.required == ["greeting"]
    end

    test "correctly parses with additional params" do
      {:ok, fun} =
        Function.new(%{
          "name" => "say_hi",
          "description" => "Provide a friendly greeting.",
          "parameters" => [
            %{name: "p1", type: "string"},
            %{name: "p2", type: "number", description: "Param 2"},
            %{name: "p3", type: "string", enum: ["yellow", "red", "green"]}
          ]
        })

      assert fun.name == "say_hi"
      assert fun.description == "Provide a friendly greeting."

      assert fun.parameters == [
               %FunctionParameter{name: "p1", type: "string"},
               %FunctionParameter{name: "p2", type: "number", description: "Param 2"},
               %FunctionParameter{name: "p3", type: "string", enum: ["yellow", "red", "green"]}
             ]
    end

    test "assigns the function to execute" do
      {:ok, fun} = Function.new(%{"name" => "hello_world", "function" => &hello_world/2})
      assert is_function(fun.function)
    end
  end

  describe "OpenAI - for_api/1" do
    test "works with minimal definition and no parameters" do
      {:ok, fun} = Function.new(%{"name" => "hello_world"})

      result = ForOpenAIApi.for_api(fun)
      # result = Function.for_api(fun)

      assert result == %{
               "description" => nil,
               "name" => "hello_world",
               "parameters" => %{"properties" => %{}, "type" => "object"},
               "required" => nil
             }
    end

    test "supports parameters" do
      {:ok, fun} =
        Function.new(%{
          "name" => "say_hi",
          "description" => "Provide a friendly greeting.",
          "parameters" => [
            %{name: "p1", type: "string"},
            %{name: "p2", type: "number", description: "Param 2"},
            %{name: "p3", type: "string", enum: ["yellow", "red", "green"]}
          ],
          "required" => ["p1"]
        })

      # result = Function.for_api(fun)
      result = ForOpenAIApi.for_api(fun)

      assert result == %{
               "name" => "say_hi",
               "description" => "Provide a friendly greeting.",
               "parameters" => %{
                 "properties" => %{
                   "p1" => %{"description" => nil, "type" => "string"},
                   "p2" => %{"description" => "Param 2", "type" => "number"},
                   "p3" => %{
                     "description" => nil,
                     "enum" => ["yellow", "red", "green"],
                     "type" => "string"
                   }
                 },
                 "type" => "object"
               },
               "required" => ["p1"]
             }
    end

    test "does not include the function to execute" do
      {:ok, fun} = Function.new(%{"name" => "hello_world", "function" => &hello_world/2})
      # result = Function.for_api(fun)
      result = ForOpenAIApi.for_api(fun)
      refute Map.has_key?(result, "function")
    end
  end
end
