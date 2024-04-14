defmodule LangChain.FunctionTest do
  use ExUnit.Case

  doctest LangChain.Function

  alias LangChain.Function

  defp hello_world(_args, _context) do
    "Hello world!"
  end

  describe "new/1" do
    test "works with minimal attrs" do
      assert {:ok, %Function{} = fun} = Function.new(%{"name" => "hello_world"})
      assert fun.name == "hello_world"
      assert fun.async == true
    end

    test "allows for tracking async setting" do
      assert {:ok, %Function{} = fun} = Function.new(%{"name" => "hello_world", "async"=> false})
      assert fun.async == false
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

  describe "get_display_text/3" do
    setup do
      functions = [
        Function.new!(%{name: "speak", display_text: "Speaking..."}),
        Function.new!(%{name: "walk", display_text: "Walking..."})
      ]

      %{functions: functions}
    end

    test "finds and returns function display text", %{functions: functions} do
      assert "Speaking..." == Function.get_display_text(functions, "speak")
      assert "Walking..." == Function.get_display_text(functions, "walk")
    end

    test "when function not found, returns default text", %{functions: functions} do
      assert "Perform action" == Function.get_display_text(functions, "missing")
      assert "Other" == Function.get_display_text(functions, "missing", "Other")
    end
  end
end
