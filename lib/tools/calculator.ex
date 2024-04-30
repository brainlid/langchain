defmodule LangChain.Tools.Calculator do
  @moduledoc """
  Defines a Calculator tool for performing basic math calculations.

  This is an example of a pre-built `LangChain.Function` that is designed and
  configured for a specific purpose.

  This defines a function to expose to an LLM and provides an implementation for
  the `execute/2` function for evaluating when an LLM executes the function.

  When using the `Calculator` tool, you will either need to:

  * make repeated calls to run the chain as the tool is called and the results
  are then made available to the LLM before it returns the final result.
  * OR run the chain using the `while_needs_response: true` option like this:
    `LangChain.LLMChain.run(chain, while_needs_response: true)`


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

      LLM: %LangChain.ChatModels.ChatOpenAI{
        endpoint: "https://api.openai.com/v1/chat/completions",
        model: "gpt-3.5-turbo",
        api_key: nil,
        temperature: 0.0,
        frequency_penalty: 0.0,
        receive_timeout: 60000,
        seed: 0,
        n: 1,
        json_response: false,
        stream: false,
        max_tokens: nil,
        user: nil
      }
      MESSAGES: [
        %LangChain.Message{
          content: "Answer the following math question: What is 100 + 300 - 200?",
          index: nil,
          status: :complete,
          role: :user,
          name: nil,
          tool_calls: [],
          tool_results: nil
        }
      ]
      TOOLS: [
        %LangChain.Function{
          name: "calculator",
          description: "Perform basic math calculations or expressions",
          display_text: nil,
          function: #Function<0.75045395/2 in LangChain.Tools.Calculator.execute>,
          async: true,
          parameters_schema: %{
            type: "object",
            required: ["expression"],
            properties: %{
              expression: %{
                type: "string",
                description: "A simple mathematical expression"
              }
            }
          },
          parameters: []
        }
      ]
      SINGLE MESSAGE RESPONSE: %LangChain.Message{
        content: nil,
        index: 0,
        status: :complete,
        role: :assistant,
        name: nil,
        tool_calls: [
          %LangChain.Message.ToolCall{
            status: :complete,
            type: :function,
            call_id: "call_NlHbo4R5NXTA6lHyjLdGQN9p",
            name: "calculator",
            arguments: %{"expression" => "100 + 300 - 200"},
            index: nil
          }
        ],
        tool_results: nil
      }
      EXECUTING FUNCTION: "calculator"
      FUNCTION RESULT: "200"
      TOOL RESULTS: %LangChain.Message{
        content: nil,
        index: nil,
        status: :complete,
        role: :tool,
        name: nil,
        tool_calls: [],
        tool_results: [
          %LangChain.Message.ToolResult{
            type: :function,
            tool_call_id: "call_NlHbo4R5NXTA6lHyjLdGQN9p",
            name: "calculator",
            content: "200",
            display_text: nil,
            is_error: false
          }
        ]
      }
      SINGLE MESSAGE RESPONSE: %LangChain.Message{
        content: "The result of the math question \"100 + 300 - 200\" is 200.",
        index: 0,
        status: :complete,
        role: :assistant,
        name: nil,
        tool_calls: [],
        tool_results: nil
      }
  """
  require Logger
  alias LangChain.Function

  @doc """
  Define the "calculator" function. Returns a success/failure response.
  """
  @spec new() :: {:ok, Function.t()} | {:error, Ecto.Changeset.t()}
  def new() do
    Function.new(%{
      name: "calculator",
      description: "Perform basic math calculations or expressions",
      parameters_schema: %{
        type: "object",
        properties: %{
          expression: %{type: "string", description: "A simple mathematical expression"}
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
        raise LangChain.LangChainError, changeset
    end
  end

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
