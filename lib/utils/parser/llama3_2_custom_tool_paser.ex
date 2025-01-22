if Code.ensure_loaded?(NimbleParsec) do
  defmodule Langchain.Utils.Parser.LLAMA_3_2_CustomToolParser do
    import NimbleParsec

    # Basic components
    whitespace = ascii_char([?\s, ?\t]) |> ignore()
    optional_whitespace = whitespace |> repeat() |> ignore()

    # Function name parser
    function_name =
      ascii_string([?a..?z, ?A..?Z, ?_], 1)
      |> concat(ascii_string([?a..?z, ?A..?Z, ?0..?9, ?_], min: 0))
      |> reduce({Enum, :join, []})
      |> unwrap_and_tag(:function_name)

    # Parameter value parsers
    quoted_string =
      ignore(ascii_char([?', ?"]))
      |> ascii_string([not: ?\", not: ?'], min: 0)
      |> ignore(ascii_char([?', ?"]))

    number = integer(min: 1)

    param_value = choice([quoted_string, number])

    # Parameter key-value pair
    param_pair =
      optional_whitespace
      |> ascii_string([?a..?z, ?A..?Z, ?_], min: 1)
      |> ignore(string("="))
      |> concat(param_value)
      |> tag(:param)

    # Parameters list
    parameters =
      param_pair
      |> repeat(ignore(string(",")) |> concat(param_pair))
      |> wrap()
      |> unwrap_and_tag(:parameters)

    parse_function_call =
      function_name
      |> ignore(string("("))
      |> optional(parameters)
      |> ignore(string(")"))
      |> wrap()
      |> unwrap_and_tag(:function_call)

    # Complete function call parser
    defparsec(
      :parse_function_calls,
      ignore(string("["))
      |> optional(parse_function_call)
      |> repeat(ignore(string(",")) |> concat(parse_function_call))
      |> ignore(string("]"))
      |> eos()
    )

    defp map_params(function_call) do
      function_call
      |> Keyword.get(:parameters, [])
      |> Enum.map(fn {:param, [key, value]} -> {key, value} end)
      |> Map.new()
    end

    defp map_function({:function_call, function_call}) do
      %{
        function_name: Keyword.get(function_call, :function_name),
        parameters: map_params(function_call)
      }
    end

    def parse(input) when is_binary(input) do
      case parse_function_calls(input) do
        {:ok, result, "", _, _, _} ->
          {:ok, Enum.map(result, &map_function/1)}

        {:error, reason, _, _, _, _} ->
          {:error, reason}
      end
    end
  end
end
