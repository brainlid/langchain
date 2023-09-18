defmodule LangChain.FunctionTest do
  use ExUnit.Case

  doctest LangChain.Function

  alias LangChain.Function
  alias LangChain.ForOpenAIApi

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

    test "supports name, description, and parameter schema def" do
      schema_def = %{
        type: "object",
        properties: %{
          info: %{
            type: "object",
            properties: %{
              name: %{type: "string"}
            },
            required: ["name"]
          }
        },
        required: ["info"]
      }

      {:ok, fun} =
        Function.new(%{
          "name" => "say_hi",
          "description" => "Provide a friendly greeting.",
          "parameters_schema" => schema_def
        })

      assert fun.name == "say_hi"
      assert fun.description == "Provide a friendly greeting."
      assert fun.parameters_schema == schema_def
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
               "name" => "hello_world",
               "description" => nil,
               #  NOTE: Sends the required empty parameter definition when none set
               "parameters" => %{"properties" => %{}, "type" => "object"}
             }
    end

    test "supports parameters" do
      params_def = %{
        "type" => "object",
        "properties" => %{
          "p1" => %{"description" => nil, "type" => "string"},
          "p2" => %{"description" => "Param 2", "type" => "number"},
          "p3" => %{
            "description" => nil,
            "enum" => ["yellow", "red", "green"],
            "type" => "string"
          }
        },
        "required" => ["p1"]
      }

      {:ok, fun} =
        Function.new(%{
          "name" => "say_hi",
          "description" => "Provide a friendly greeting.",
          "parameters_schema" => params_def
        })

      # result = Function.for_api(fun)
      result = ForOpenAIApi.for_api(fun)

      assert result == %{
               "name" => "say_hi",
               "description" => "Provide a friendly greeting.",
               "parameters" => params_def
             }
    end

    test "does not include the function to execute" do
      # don't try and send an Elixir function ref through to the API
      {:ok, fun} = Function.new(%{"name" => "hello_world", "function" => &hello_world/2})
      # result = Function.for_api(fun)
      result = ForOpenAIApi.for_api(fun)
      refute Map.has_key?(result, "function")
    end
  end
end
