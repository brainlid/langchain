defmodule LangChain.FunctionTest do
  use ExUnit.Case

  doctest LangChain.Function

  alias LangChain.Function
  alias LangChain.FunctionParam
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

      assert {:error, message, {%RuntimeError{}, [_ | _]}} = result

      assert message ==
               "ERROR: (RuntimeError) fake exception at test/function_test.exs:15: LangChain.FunctionTest.returns_context/2"

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

    test "validates required parameters declared with atom-keyed parameters_schema" do
      # This is the shape used by LangChain.Tools.Calculator: atom keys for the
      # schema itself AND for the property names.
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

      assert {:error, message} = Function.execute(function, %{}, %{result: {:ok, "SUCCESS"}})
      assert message =~ "Missing required parameters"
      assert message =~ "Required parameters: field"
      assert message =~ "Accepted parameters: field"
    end

    test "validates required parameters declared with string-keyed parameters_schema" do
      function =
        Function.new!(%{
          name: "with_schema",
          parameters_schema: %{
            "type" => "object",
            "properties" => %{"field" => %{"type" => "string"}},
            "required" => ["field"]
          },
          function: &returns_context/2
        })

      assert {:error, message} = Function.execute(function, %{}, %{result: {:ok, "SUCCESS"}})
      assert message =~ "Required parameters: field"

      assert {:ok, "SUCCESS"} =
               Function.execute(function, %{"field" => "x"}, %{result: {:ok, "SUCCESS"}})
    end

    test "executes when parameters_schema declares no required parameters" do
      function =
        Function.new!(%{
          name: "with_schema",
          parameters_schema: %{
            type: "object",
            properties: %{field: %{type: "string"}}
          },
          function: &returns_context/2
        })

      assert {:ok, "SUCCESS"} = Function.execute(function, %{}, %{result: {:ok, "SUCCESS"}})
    end

    test "omits the accepted parameters list when the schema has no properties" do
      function =
        Function.new!(%{
          name: "with_schema",
          parameters_schema: %{type: "object", required: ["field"]},
          function: &returns_context/2
        })

      assert {:error, message} = Function.execute(function, %{"other" => 1}, %{})
      assert message =~ "Required parameters: field"
      # We don't know which names are valid, so we can't call anything unrecognized.
      refute message =~ "Accepted parameters"
      refute message =~ "Unrecognized parameters"
    end

    test "names a renamed argument and suggests the parameter it meant" do
      # The motivating case: jaro_distance("file_path", "path") is 0.0, so a
      # jaro-only suggestion would silently produce no hint here.
      function =
        Function.new!(%{
          name: "read_document",
          parameters_schema: %{
            type: "object",
            properties: %{
              path: %{type: "string"},
              offset: %{type: "integer"},
              limit: %{type: "integer"}
            },
            required: ["path"]
          },
          function: &returns_context/2
        })

      assert {:error, message} = Function.execute(function, %{"file_path" => "/a.md"}, %{})
      assert message =~ "Missing parameters: path"
      assert message =~ "Unrecognized parameters: file_path (did you mean \"path\"?)"
      assert message =~ "Accepted parameters:"
      assert message =~ "path"
    end

    test "names an unrecognized argument without a suggestion when nothing is similar" do
      function =
        Function.new!(%{
          name: "read_document",
          parameters_schema: %{
            type: "object",
            properties: %{path: %{type: "string"}},
            required: ["path"]
          },
          function: &returns_context/2
        })

      assert {:error, message} = Function.execute(function, %{"query" => "abc"}, %{})
      assert message =~ "Unrecognized parameters: query"
      refute message =~ "did you mean"
    end

    test "omits the unrecognized line when every supplied argument is declared" do
      function =
        Function.new!(%{
          name: "read_document",
          parameters: [
            FunctionParam.new!(%{name: "path", type: :string, required: true}),
            FunctionParam.new!(%{name: "limit", type: :integer})
          ],
          function: &returns_context/2
        })

      assert {:error, message} = Function.execute(function, %{"limit" => 5}, %{})
      assert message =~ "Missing parameters: path"
      refute message =~ "Unrecognized parameters"
    end

    test "does not suggest a parameter that was already supplied" do
      function =
        Function.new!(%{
          name: "search",
          parameters: [
            FunctionParam.new!(%{name: "pattern", type: :string, required: true}),
            FunctionParam.new!(%{name: "file_path", type: :string, required: true})
          ],
          function: &returns_context/2
        })

      assert {:error, message} =
               Function.execute(function, %{"file_path" => "/a.md", "patern" => "TODO"}, %{})

      assert message =~ "Unrecognized parameters: patern (did you mean \"pattern\"?)"
    end

    test "lists required parameters in declaration order" do
      function =
        Function.new!(%{
          name: "create_record",
          parameters: [
            FunctionParam.new!(%{name: "alpha", type: :string, required: true}),
            FunctionParam.new!(%{name: "beta", type: :string, required: true}),
            FunctionParam.new!(%{name: "gamma", type: :string, required: true})
          ],
          function: &returns_context/2
        })

      assert {:error, message} = Function.execute(function, %{}, %{})
      assert message =~ "Required parameters: alpha, beta, gamma"
    end

    test "treats nil arguments as an empty map instead of raising BadMapError" do
      function =
        Function.new!(%{
          name: "create_record",
          parameters: [
            FunctionParam.new!(%{name: "record_id", type: :string, required: true})
          ],
          function: &returns_context/2
        })

      assert {:error, message} = Function.execute(function, nil, %{})
      assert message =~ "Missing required parameters"
      assert message =~ "record_id"
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

  describe "new/1 with :parse_args" do
    test "accepts a 1-arity anonymous function" do
      parser = fn args -> {:ok, args} end

      assert {:ok, %Function{} = fun} =
               Function.new(%{
                 name: "with_parser",
                 function: &hello_world/2,
                 parse_args: parser
               })

      assert fun.parse_args == parser
    end

    test "rejects an anonymous function with the wrong arity" do
      assert {:error, changeset} =
               Function.new(%{
                 name: "bad_parser",
                 function: &hello_world/2,
                 parse_args: fn _a, _b -> :ok end
               })

      assert {"expected arity of 1 but has arity 2", _} = changeset.errors[:parse_args]
    end

    test "rejects something that isn't an Elixir function" do
      assert {:error, changeset} =
               Function.new(%{
                 name: "bad_parser",
                 function: &hello_world/2,
                 parse_args: "not a function"
               })

      assert {"is not an Elixir function", _} = changeset.errors[:parse_args]
    end

    test ":parse_args defaults to nil" do
      assert {:ok, %Function{parse_args: nil}} =
               Function.new(%{name: "no_parser", function: &hello_world/2})
    end
  end

  describe "execute/3 with :parse_args" do
    # A tool body that returns the parsed arguments it was handed. Lets us
    # observe whether `parse_args` actually transformed the arguments.
    defp echo_args(args, _context), do: {:ok, "received: #{inspect(args)}", args}

    test "absent parser leaves arguments untouched" do
      function = Function.new!(%{name: "echo", function: &echo_args/2})

      assert {:ok, _llm, %{"a" => 1}} = Function.execute(function, %{"a" => 1}, nil)
    end

    test ":ok return passes original arguments through to the function body" do
      parser = fn _args -> :ok end
      function = Function.new!(%{name: "echo", function: &echo_args/2, parse_args: parser})

      assert {:ok, _llm, %{"a" => 1}} = Function.execute(function, %{"a" => 1}, nil)
    end

    test "{:ok, parsed} return replaces arguments handed to the function body" do
      # Parser coerces string keys → atom keys and narrows the shape — the
      # function body should receive the parsed map, not the raw input.
      parser = fn %{"value" => v} -> {:ok, %{value: String.to_integer(v)}} end
      function = Function.new!(%{name: "echo", function: &echo_args/2, parse_args: parser})

      assert {:ok, _llm, %{value: 42}} = Function.execute(function, %{"value" => "42"}, nil)
    end

    test "{:error, reason} short-circuits without running the function body" do
      parent = self()

      function =
        Function.new!(%{
          name: "rejecter",
          function: fn _args, _context ->
            send(parent, :function_ran)
            {:ok, "should not reach here"}
          end,
          parse_args: fn _args -> {:error, "device_task_id is required"} end
        })

      assert {:error, "device_task_id is required"} =
               Function.execute(function, %{"value" => "anything"}, nil)

      refute_received :function_ran
    end

    test "{:error, term} return that isn't a binary is stringified" do
      function =
        Function.new!(%{
          name: "rejecter",
          function: &echo_args/2,
          parse_args: fn _args -> {:error, %{validation: :failed}} end
        })

      assert {:error, "%{validation: :failed}"} = Function.execute(function, %{}, nil)
    end

    test "unexpected return shape is normalized to an :error tuple" do
      function =
        Function.new!(%{
          name: "weird",
          function: &echo_args/2,
          parse_args: fn _args -> :totally_invalid end
        })

      assert {:error, "parse_args returned unexpected shape: :totally_invalid"} =
               Function.execute(function, %{}, nil)
    end

    test "a rescued exception is returned alongside the model-facing message" do
      function =
        Function.new!(%{
          name: "raiser",
          function: fn _args, _ctx -> raise RuntimeError, "kaboom" end
        })

      assert {:error, message, {exception, stacktrace}} = Function.execute(function, %{}, nil)
      assert %RuntimeError{message: "kaboom"} = exception
      assert [{LangChain.FunctionTest, _fun, _arity, _loc} | _] = stacktrace
      assert message =~ "RuntimeError"
      assert message =~ "kaboom"
    end

    test "a rescued :parse_args exception is returned alongside the message" do
      function =
        Function.new!(%{
          name: "raiser",
          function: &echo_args/2,
          parse_args: fn _args -> raise ArgumentError, "bad args" end
        })

      assert {:error, message, {%ArgumentError{message: "bad args"}, [_ | _]}} =
               Function.execute(function, %{}, nil)

      assert message =~ "bad args"
    end

    test "an error the tool returns deliberately stays a two-element tuple" do
      function =
        Function.new!(%{
          name: "returns_error",
          function: fn _args, _ctx -> {:error, "not found"} end
        })

      assert {:error, "not found"} = Function.execute(function, %{}, nil)
    end

    test "a missing required parameter stays a two-element tuple" do
      function =
        Function.new!(%{
          name: "needs_path",
          parameters_schema: %{
            type: "object",
            properties: %{path: %{type: "string"}},
            required: ["path"]
          },
          function: fn _args, _ctx -> {:ok, "unreachable"} end
        })

      assert {:error, message} = Function.execute(function, %{}, nil)
      assert message =~ "path"
    end

    test "an exception raised inside :parse_args is caught and reported" do
      function =
        Function.new!(%{
          name: "raiser",
          function: &echo_args/2,
          parse_args: fn _args -> raise ArgumentError, "boom" end
        })

      assert {:error, "ERROR: " <> message, {_exception, _stacktrace}} =
               Function.execute(function, %{}, nil)

      assert message =~ "boom"
    end

    test ":parse_args replaces the built-in required-params check" do
      # A supplied parser owns argument validation outright, so the built-in
      # required-key check is skipped entirely and the parser decides.
      parent = self()

      function =
        Function.new!(%{
          name: "parser_owns_validation",
          parameters: [FunctionParam.new!(%{name: "id", type: :string, required: true})],
          function: &echo_args/2,
          parse_args: fn args ->
            send(parent, :parser_ran)
            {:ok, args}
          end
        })

      # "id" is required and absent, but the parser accepted the call, so it runs.
      assert {:ok, _llm_result, %{}} = Function.execute(function, %{}, nil)
      assert_received :parser_ran
    end

    test ":parse_args owns the message for a missing required parameter" do
      function =
        Function.new!(%{
          name: "parser_owns_validation",
          parameters_schema: %{
            type: "object",
            properties: %{coolio: %{type: "string"}},
            required: ["coolio"]
          },
          function: &echo_args/2,
          parse_args: fn args ->
            if Map.has_key?(args, "coolio"),
              do: {:ok, args},
              else: {:error, "PARSER: coolio is required"}
          end
        })

      assert {:error, "PARSER: coolio is required"} =
               Function.execute(function, %{"value" => 1}, nil)
    end

    test "an exception from the body keeps its original formatting when a parser is set" do
      # The parser approved these arguments, so a body that still can't read
      # them is a tool bug rather than a model mistake. Report it as such.
      function =
        Function.new!(%{
          name: "parser_owns_validation",
          parameters_schema: %{
            type: "object",
            properties: %{path: %{type: "string"}, limit: %{type: "integer"}},
            required: ["path"]
          },
          function: fn args, _ctx -> Map.fetch!(args, "limit") end,
          parse_args: fn args -> {:ok, args} end
        })

      assert {:error, message, {_exception, _stacktrace}} =
               Function.execute(function, %{"path" => "/a.md"}, nil)

      assert message =~ "KeyError"
      refute message =~ "does not accept"
    end
  end

  describe "required_param_names/1 and accepted_param_names/1" do
    test "read from a list of FunctionParams" do
      function =
        Function.new!(%{
          name: "demo",
          parameters: [
            FunctionParam.new!(%{name: "path", type: :string, required: true}),
            FunctionParam.new!(%{name: "limit", type: :integer}),
            FunctionParam.new!(%{name: "offset", type: :integer, required: true})
          ],
          function: &hello_world/2
        })

      assert Function.required_param_names(function) == ["path", "offset"]
      assert Function.accepted_param_names(function) == ["path", "limit", "offset"]
    end

    test "read from an atom-keyed parameters_schema" do
      function =
        Function.new!(%{
          name: "demo",
          parameters_schema: %{
            type: "object",
            properties: %{path: %{type: "string"}, limit: %{type: "integer"}},
            required: ["path"]
          },
          function: &hello_world/2
        })

      assert Function.required_param_names(function) == ["path"]
      assert function |> Function.accepted_param_names() |> Enum.sort() == ["limit", "path"]
    end

    test "read from a string-keyed parameters_schema" do
      function =
        Function.new!(%{
          name: "demo",
          parameters_schema: %{
            "type" => "object",
            "properties" => %{"path" => %{"type" => "string"}},
            "required" => ["path"]
          },
          function: &hello_world/2
        })

      assert Function.required_param_names(function) == ["path"]
      assert Function.accepted_param_names(function) == ["path"]
    end

    test "return empty lists when nothing is declared" do
      function = Function.new!(%{name: "demo", function: &hello_world/2})

      assert Function.required_param_names(function) == []
      assert Function.accepted_param_names(function) == []
    end

    test "tolerate a schema missing required or properties" do
      function =
        Function.new!(%{
          name: "demo",
          parameters_schema: %{type: "object"},
          function: &hello_world/2
        })

      assert Function.required_param_names(function) == []
      assert Function.accepted_param_names(function) == []
    end
  end

  describe "execute/3 with an argument error raised by the tool" do
    defp fetches_optional(args, _context) do
      Map.fetch!(args, "limit")
    end

    defp fetches_unrelated_map(_args, _context) do
      Map.fetch!(%{"a" => 1}, "totally_unrelated")
    end

    defp head_matches_path(%{"path" => path}, _context), do: {:ok, path}

    test "reports a renamed optional argument instead of a stack frame" do
      function =
        Function.new!(%{
          name: "read_document",
          parameters_schema: %{
            type: "object",
            properties: %{path: %{type: "string"}, limit: %{type: "integer"}},
            required: ["path"]
          },
          function: &fetches_optional/2
        })

      assert {:error, message, {_exception, _stacktrace}} =
               Function.execute(function, %{"path" => "/a.md", "lim" => 5}, %{})

      assert message =~ "does not accept"
      assert message =~ "Unrecognized parameters: lim (did you mean \"limit\"?)"
      assert message =~ "Accepted parameters:"
      # The tool's source location must not leak to the model.
      refute message =~ "KeyError"
      refute message =~ ".ex:"
    end

    test "reports a missing optional argument by name" do
      function =
        Function.new!(%{
          name: "read_document",
          parameters_schema: %{
            type: "object",
            properties: %{path: %{type: "string"}, limit: %{type: "integer"}},
            required: ["path"]
          },
          function: &fetches_optional/2
        })

      assert {:error, message, {_exception, _stacktrace}} =
               Function.execute(function, %{"path" => "/a.md"}, %{})

      assert message =~ "needs the \"limit\" argument"
      assert message =~ "Accepted parameters:"
      refute message =~ "KeyError"
    end

    test "reports a FunctionClauseError raised by the declared tool function" do
      function =
        Function.new!(%{
          name: "read_document",
          parameters_schema: %{
            type: "object",
            properties: %{path: %{type: "string"}},
            required: []
          },
          function: &head_matches_path/2
        })

      assert {:error, message, {_exception, _stacktrace}} =
               Function.execute(function, %{"file_path" => "/a.md"}, %{})

      assert message =~ "does not accept"
      assert message =~ "did you mean \"path\"?"
      refute message =~ "FunctionClauseError"
    end

    test "leaves a KeyError on an unrelated map with its original formatting" do
      function =
        Function.new!(%{
          name: "read_document",
          parameters_schema: %{
            type: "object",
            properties: %{path: %{type: "string"}},
            required: ["path"]
          },
          function: &fetches_unrelated_map/2
        })

      assert {:error, message, {_exception, _stacktrace}} =
               Function.execute(function, %{"path" => "/a.md"}, %{})

      assert message =~ "KeyError"
      assert message =~ "totally_unrelated"
    end

    test "falls back to the original formatting when no parameters are declared" do
      function = Function.new!(%{name: "read_document", function: &fetches_optional/2})

      assert {:error, message, {_exception, _stacktrace}} =
               Function.execute(function, %{"path" => "/a.md"}, %{})

      assert message =~ "KeyError"
    end
  end
end
