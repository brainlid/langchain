if Code.ensure_loaded?(NimbleParsec) do
  defmodule LangChain.Utils.Parser.LLAMA_3_1_CustomToolParser do
    import NimbleParsec

    # Core components
    start_tag = string("<function=")
    end_tag = string("</function>")

    # Function name: letters, numbers, and underscores
    function_name =
      ascii_string([?a..?z, ?A..?Z, ?0..?9, ?_], min: 1)
      |> unwrap_and_tag(:function_name)

    # Parameters: capture everything between > and </function> as JSON
    parameters =
      ignore(string(">"))
      |> ascii_string([not: ?<], min: 0)
      |> map({Jason, :decode!, []})
      |> unwrap_and_tag(:parameters)

    # Complete function call parser
    defparsec(
      :parse_function_call,
      ignore(start_tag)
      |> concat(function_name)
      |> concat(parameters)
      |> ignore(end_tag)
      |> eos()
    )

    # Helper function for easier parsing
    def parse(input) when is_binary(input) do
      try do
        case parse_function_call(input) do
          {:ok, result, "", _, _, _} -> {:ok, Map.new(result)}
          {:error, reason, _, _, _, _} -> {:error, reason}
        end
      rescue
        Jason.DecodeError -> {:error, "Invalid JSON in parameters"}
      end
    end
  end
end
