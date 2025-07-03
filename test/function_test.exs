defmodule LangChain.FunctionTest do
  use ExUnit.Case

  doctest LangChain.Function

  alias LangChain.Function

  defp hello_world(_args, _context) do
    "Hello world!"
  end

  defp returns_context(_args, %{result: :exception}) do
    raise RuntimeError, "fake exception"
  end

  defp returns_context(_args, %{result: result}) do
    result
  end

  describe "new/1" do
    test "works with minimal attrs" do
      assert {:ok, %Function{} = fun} =
               Function.new(%{"name" => "hello_world", "function" => &hello_world/2})

      assert fun.name == "hello_world"
      assert fun.async == false
    end

    test "allows for tracking async setting" do
      assert {:ok, %Function{} = fun} =
               Function.new(%{
                 "name" => "hello_world",
                 "function" => &hello_world/2,
                 "async" => false
               })

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
          "parameters_schema" => schema_def,
          "function" => &hello_world/2
        })

      assert fun.name == "say_hi"
      assert fun.description == "Provide a friendly greeting."
      assert fun.parameters_schema == schema_def
    end

    test "assigns the function to execute" do
      {:ok, fun} = Function.new(%{"name" => "hello_world", "function" => &hello_world/2})
      assert is_function(fun.function)
    end

    test "validates that an Elixir function was provided" do
      {:error, changeset} = Function.new(%{"name" => "hello_world", "function" => "stuff"})
      assert {"is not an Elixir function", _} = changeset.errors[:function]
    end

    test "validates arity of the assigned Elixir function" do
      {:error, changeset} = Function.new(%{"name" => "hello_world", "function" => fn -> :ok end})
      assert {"expected arity of 2 but has arity 0", _} = changeset.errors[:function]
    end
  end

  describe "get_display_text/3" do
    setup do
      functions = [
        Function.new!(%{name: "speak", display_text: "Speaking...", function: &hello_world/2}),
        Function.new!(%{name: "walk", display_text: "Walking...", function: &hello_world/2})
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

  describe "execute/3" do
    test "executes the Elixir function and returns the result" do
      function = Function.new!(%{name: "returns_context", function: &returns_context/2})
      result = Function.execute(function, %{}, %{result: {:ok, "SUCCESS"}})
      assert result == {:ok, "SUCCESS"}
    end

    test "normalizes responses to {:ok, result} and {:error, reason}" do
      function = Function.new!(%{name: "returns_context", function: &returns_context/2})

      # returns an :ok tuple as-is
      result = Function.execute(function, %{}, %{result: {:ok, "SUCCESS"}})
      assert result == {:ok, "SUCCESS"}

      # returns a string wrapped in :ok tuple
      result = Function.execute(function, %{}, %{result: "SUCCESS"})
      assert result == {:ok, "SUCCESS"}

      # returns an error tuple
      result = Function.execute(function, %{}, %{result: {:error, "FAILED"}})
      assert result == {:error, "FAILED"}

      # makes error structs as a string
      result = Function.execute(function, %{}, %{result: {:error, Date.new!(2024, 04, 01)}})
      assert result == {:error, "~D[2024-04-01]"}

      # rescues an exception and returns as string text
      result = Function.execute(function, %{}, %{result: :exception})

      assert result ==
               {:error,
                "ERROR: (RuntimeError) fake exception at test/function_test.exs:13: LangChain.FunctionTest.returns_context/2"}

      # returns an error when anything else is returned
      result = Function.execute(function, %{}, %{result: 123})
      assert result == {:error, "An unexpected response was returned from the tool."}
    end
  end
end
