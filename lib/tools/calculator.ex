defmodule Langchain.Tools.Calculator do
  @moduledoc """
  Defines a Calculator tool for performing basic math calculations.

  Defines a function to expose to an LLM and provides the `execute/2` function
  for evaluating it the function when executed by an LLM.
  """
  require Logger
  alias Langchain.Function

  @doc """
  Defines the "calculator" function.
  """
  @spec new() :: {:ok, Function.t()} | {:error, Ecto.Changeset.t()}
  def new() do
    Function.new(%{
      name: "calculator",
      description: "Perform basic math calculations",
      parameters_schema: %{
        type: "object",
        properties: %{
          expression: %{type: "string", description: "A simple mathematical expression."}
        },
        required: ["expression"]
      },
      function: &execute/2
    })
  end

  @spec new!() :: Function.t() | no_return()
  def new!() do
    case new() do
      {:ok, function} ->
        function

      {:error, changeset} ->
        raise Langchain.LangchainError, changeset
    end
  end

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

  @doc """
  Performs the calculation specified in the expression and returns the response
  to be used by the the LLM.
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
