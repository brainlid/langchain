defmodule Langchain.Tools.Calculator do
  @moduledoc """
  Defines a Calculator tool for performing basic math calculations.

  This is an example of a pre-built `Langchain.Function` that is designed and
  configured for a specific purpose.

  This defines a function to expose to an LLM and provides an implementation for
  the `execute/2` function for evaluating when an LLM executes the function.

  When using the `Calculator` tool, you will either need to:

  * make repeated calls to run the chain as the tool is called and the results
  are then made available to the LLM before it returns the final result.
  * OR run the chain using the `while_needs_response: true` option like this:
    `Langchain.LLMChain.run(chain, while_needs_response: true)`


  ## Example

  The following is an example that uses a prompt where math is needed. What
  follows is the verbose log output.

      {:ok, updated_chain, %Message{} = message} =
        %{llm: ChatOpenAI.new!(%{temperature: 0}), verbose: true}
        |> LLMChain.new!()
        |> LLMChain.add_message(
          Message.new_user!("Answer the following math question: What is 100 + 300 - 200?")
        )
        |> LLMChain.add_functions(Calculator.new!())
        |> LLMChain.run(while_needs_response: true)

  Verbose log output:

      LLM: %Langchain.ChatModels.ChatOpenAI{
        endpoint: "https://api.openai.com/v1/chat/completions",
        model: "gpt-3.5-turbo",
        temperature: 0.0,
        frequency_penalty: 0.0,
        receive_timeout: 60000,
        n: 1,
        stream: false
      }
      MESSAGES: [
        %Langchain.Message{
          content: "Answer the following math question: What is 100 + 300 - 200?",
          index: nil,
          status: :complete,
          role: :user,
          function_name: nil,
          arguments: nil
        }
      ]
      FUNCTIONS: [
        %Langchain.Function{
          name: "calculator",
          description: "Perform basic math calculations",
          function: #Function<0.108164323/2 in Langchain.Tools.Calculator.execute>,
          parameters_schema: %{
            properties: %{
              expression: %{
                description: "A simple mathematical expression.",
                type: "string"
              }
            },
            required: ["expression"],
            type: "object"
          }
        }
      ]
      SINGLE MESSAGE RESPONSE: %Langchain.Message{
        content: nil,
        index: 0,
        status: :complete,
        role: :assistant,
        function_name: "calculator",
        arguments: %{"expression" => "100 + 300 - 200"}
      }
      EXECUTING FUNCTION: "calculator"
      FUNCTION RESULT: "200"
      SINGLE MESSAGE RESPONSE: %Langchain.Message{
        content: "The answer to the math question \"What is 100 + 300 - 200?\" is 200.",
        index: 0,
        status: :complete,
        role: :assistant,
        function_name: nil,
        arguments: nil
      }

  """
  require Logger
  alias Langchain.Function

  @doc """
  Define the "calculator" function. Returns a success/failure response.
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

  @doc """
  Define the "calculator" function. Raises an exception if function creation fails.
  """
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
        Logger.warning(
          "Calculator tool errored in eval of #{inspect(expr)}. Reason: #{inspect(reason)}"
        )

        "ERROR"
    end
  end
end
