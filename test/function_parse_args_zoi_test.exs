defmodule LangChain.FunctionParseArgsZoiTest do
  @moduledoc """
  Integration test exercising `LangChain.Function`'s `:parse_args` hook against
  a real schema library (Zoi). Zoi is declared as an optional dependency of
  LangChain, so this test demonstrates that the hook is library-agnostic and
  that downstream apps can wire any "parse, don't validate" library to it.
  """
  use ExUnit.Case, async: true

  alias LangChain.Function

  # A Zoi schema is built at module load time so the parser closure can capture
  # it. This mirrors how a real tool module would define its argument shape.
  @args_schema Zoi.object(%{device_task_id: Zoi.string(min_length: 1)}, coerce: true)

  # Parser function that adapts Zoi's return contract to `parse_args/0`. Zoi
  # already returns `{:ok, map}` / `{:error, [Zoi.Error{}]}`, so the only work
  # left is rendering the error list to a string the LLM can read.
  defp parse_args(args) do
    case Zoi.parse(@args_schema, args) do
      {:ok, parsed} ->
        {:ok, parsed}

      {:error, errors} when is_list(errors) ->
        {:error, format_zoi_errors(errors)}
    end
  end

  defp format_zoi_errors(errors) do
    Enum.map_join(errors, "; ", fn %Zoi.Error{path: path, message: msg} ->
      "#{Enum.join(path, ".")}: #{msg}"
    end)
  end

  # Tool body that returns the parsed args it received. Makes assertions
  # straightforward and demonstrates that the parsed (atom-key, coerced)
  # arguments flow through.
  defp echo_args(args, _context), do: {:ok, "received: #{inspect(args)}", args}

  describe "Zoi as a :parse_args parser" do
    test "successful parse hands coerced, atom-keyed args to the function body" do
      function =
        Function.new!(%{
          name: "diagnose_device_task",
          function: &echo_args/2,
          parse_args: &parse_args/1
        })

      # The LLM sends string keys. Zoi's `coerce: true` accepts that and
      # produces atom keys, so the function body should see the parsed shape.
      args_from_llm = %{"device_task_id" => "dt-42"}

      assert {:ok, _llm, %{device_task_id: "dt-42"}} =
               Function.execute(function, args_from_llm, nil)
    end

    test "missing required field is rejected before the function body runs" do
      parent = self()

      function =
        Function.new!(%{
          name: "diagnose_device_task",
          function: fn _args, _ctx ->
            send(parent, :function_ran)
            {:ok, "should not be reached"}
          end,
          parse_args: &parse_args/1
        })

      assert {:error, reason} = Function.execute(function, %{}, nil)
      assert reason == "device_task_id: is required"

      refute_received :function_ran
    end

    test "constraint violation is rejected with the Zoi error message" do
      function =
        Function.new!(%{
          name: "diagnose_device_task",
          function: &echo_args/2,
          parse_args: &parse_args/1
        })

      assert {:error, reason} =
               Function.execute(function, %{"device_task_id" => ""}, nil)

      assert reason == "device_task_id: too small: must have at least 1 character(s)"
    end
  end
end
