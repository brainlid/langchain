defmodule Langchain.Tools.Calculator do
  require Logger
  alias Langchain.Functions.Function

  def new() do
    Function.new(%{
      name: "calculator",
      description: "Perform basic math calculations",
      parameters: [
        %{name: "expression", type: "string", description: "A simple mathematical expression."}
      ],
      required: "expression",
      function: &execute/2
    })
  end

  # def name, do: "calculator"

  # @doc """
  # Define the calculator tool using a JSON Schema.
  # """
  # def define do
  #   # JSON Schema definition of the function. The name, description, and the
  #   # parameters it takes.
  #   %{
  #     "name" => name(),
  #     "description" => "Perform basic math calculations",
  #     "parameters" => %{
  #       "type" => "object",
  #       "properties" => %{
  #         "expression" => %{
  #           "type" => "string",
  #           "description" => "A simple mathematical expression."
  #         }
  #       },
  #       "required" => ["expression"]
  #     }
  #   }
  # end

  # # TODO: Should the parse be separate from the execute?

  # @spec parse(json :: String.t()) :: {:ok, number()} | {:error, String.t()}
  # def parse(json) do
  #   case Jason.decode(json) do
  #     {:ok, args} ->
  #       execute(args)

  #     {:error, reason} ->
  #       Logger.error("Error receiving calculator arguments! Reason: #{inspect(reason)}")
  #       {:error, "error"}
  #   end
  # end

  @doc """
  Function that performs the calculation.
  """
  @spec execute(args :: %{String.t() => any()}, context :: map()) :: String.t()
  def execute(%{"expression" => expr} = _args, _context) do
    case Abacus.eval(expr) do
      {:ok, number} ->
        to_string(number)

      {:error, reason} ->
        Logger.warn(
          "Calculator tool errored in eval of #{inspect(expr)}. Reason: #{inspect(reason)}"
        )

        "ERROR"
    end
  end
end
