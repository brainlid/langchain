defmodule LangChain.FunctionTest do
  use ExUnit.Case

  doctest LangChain.Function

  alias LangChain.Function
  alias LangChain.Message.ContentPart

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
                "ERROR: (RuntimeError) fake exception at test/function_test.exs:14: LangChain.FunctionTest.returns_context/2"}

      # returns an error when anything else is returned
      result = Function.execute(function, %{}, %{result: 123})
      assert result == {:error, "An unexpected response was returned from the tool."}
    end

    test "handles multi-part responses from tools" do
      # Simulate a tool that returns an image with text metadata
      image_tool = fn _args, _context ->
        [
          ContentPart.text!("Generated visualization of sales data"),
          ContentPart.image!("base64encodedimagedata==", media: :jpg)
        ]
      end

      function = Function.new!(%{name: "generate_chart", function: image_tool})
      result = Function.execute(function, %{}, %{})

      assert {:ok, parts} = result
      assert is_list(parts)
      assert length(parts) == 2

      assert %ContentPart{type: :text, content: "Generated visualization of sales data"} =
               Enum.at(parts, 0)

      assert %ContentPart{type: :image, content: "base64encodedimagedata=="} = Enum.at(parts, 1)
    end

    test "validates required parameters before execution" do
      alias LangChain.FunctionParam

      function =
        Function.new!(%{
          name: "create_record",
          parameters: [
            FunctionParam.new!(%{name: "record_id", type: :string, required: true}),
            FunctionParam.new!(%{name: "content", type: :string, required: true}),
            FunctionParam.new!(%{name: "active", type: :boolean, required: false})
          ],
          function: &returns_context/2
        })

      # Test with missing required parameters
      result =
        Function.execute(
          function,
          %{"active" => true, "display_name" => "Some Name"},
          %{result: {:ok, "SUCCESS"}}
        )

      assert {:error, message} = result
      assert message =~ "Missing required parameters"
      assert message =~ "Required parameters:"
      assert message =~ "record_id"
      assert message =~ "content"
    end

    test "allows execution when all required parameters are present" do
      alias LangChain.FunctionParam

      function =
        Function.new!(%{
          name: "create_record",
          parameters: [
            FunctionParam.new!(%{name: "record_id", type: :string, required: true}),
            FunctionParam.new!(%{name: "content", type: :string, required: true}),
            FunctionParam.new!(%{name: "active", type: :boolean, required: false})
          ],
          function: &returns_context/2
        })

      # Test with all required parameters present
      result =
        Function.execute(
          function,
          %{"record_id" => "123", "content" => "Test content", "active" => true},
          %{result: {:ok, "SUCCESS"}}
        )

      assert result == {:ok, "SUCCESS"}
    end

    test "allows execution when no required parameters are defined" do
      alias LangChain.FunctionParam

      function =
        Function.new!(%{
          name: "optional_params",
          parameters: [
            FunctionParam.new!(%{name: "option1", type: :string, required: false}),
            FunctionParam.new!(%{name: "option2", type: :boolean, required: false})
          ],
          function: &returns_context/2
        })

      # All parameters are optional, should execute even with no args
      result = Function.execute(function, %{}, %{result: {:ok, "SUCCESS"}})
      assert result == {:ok, "SUCCESS"}
    end

    test "skips validation when using parameters_schema instead of parameters" do
      function =
        Function.new!(%{
          name: "with_schema",
          parameters_schema: %{
            type: "object",
            properties: %{field: %{type: "string"}},
            required: ["field"]
          },
          function: &returns_context/2
        })

      # parameters_schema validation is not implemented yet, so it should execute
      result = Function.execute(function, %{}, %{result: {:ok, "SUCCESS"}})
      assert result == {:ok, "SUCCESS"}
    end

    test "allows execution when required params are present with extra params" do
      alias LangChain.FunctionParam

      function =
        Function.new!(%{
          name: "create_record",
          parameters: [
            FunctionParam.new!(%{name: "record_id", type: :string, required: true}),
            FunctionParam.new!(%{name: "content", type: :string, required: true})
          ],
          function: &returns_context/2
        })

      # Test with required parameters plus extra unexpected ones
      result =
        Function.execute(
          function,
          %{
            "record_id" => "123",
            "content" => "Test content",
            "active" => true,
            "display_name" => "Some Name",
            "extra_field" => "unexpected"
          },
          %{result: {:ok, "SUCCESS"}}
        )

      assert result == {:ok, "SUCCESS"}
    end

    test "allows execution with mix of required, optional, and extra params" do
      alias LangChain.FunctionParam

      function =
        Function.new!(%{
          name: "update_record",
          parameters: [
            FunctionParam.new!(%{name: "record_id", type: :string, required: true}),
            FunctionParam.new!(%{name: "title", type: :string, required: false}),
            FunctionParam.new!(%{name: "enabled", type: :boolean, required: false})
          ],
          function: &returns_context/2
        })

      # Required + optional + extra params should work
      result =
        Function.execute(
          function,
          %{
            "record_id" => "456",
            "title" => "New Title",
            "extra_metadata" => %{"foo" => "bar"},
            "unknown_field" => 123
          },
          %{result: {:ok, "SUCCESS"}}
        )

      assert result == {:ok, "SUCCESS"}
    end

    test "validates only required params, ignores missing optional params" do
      alias LangChain.FunctionParam

      function =
        Function.new!(%{
          name: "partial_update",
          parameters: [
            FunctionParam.new!(%{name: "id", type: :string, required: true}),
            FunctionParam.new!(%{name: "optional1", type: :string, required: false}),
            FunctionParam.new!(%{name: "optional2", type: :boolean, required: false})
          ],
          function: &returns_context/2
        })

      # Only required param provided, optional ones missing - should work
      result =
        Function.execute(
          function,
          %{"id" => "789"},
          %{result: {:ok, "SUCCESS"}}
        )

      assert result == {:ok, "SUCCESS"}
    end
  end
end
