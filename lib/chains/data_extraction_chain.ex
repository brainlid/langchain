defmodule Langchain.Chains.DataExtractionChain do
  @moduledoc """
  Define an LLMChain for performing data extraction from a body of text.

  Provide the schema for desired information to be parsed into. It is treated as
  though there are 0 to many instances of the data structure being described so
  it returns information as an array.

  Originally based on:
  - https://github.com/hwchase17/langchainjs/blob/main/langchain/src/chains/openai_functions/extraction.ts#L42
  """
  use Ecto.Schema
  require Logger
  alias Langchain.PromptTemplate
  alias Langchain.Message
  alias Langchain.Chains.LLMChain

  @extraction_template ~s"Extract and save the relevant entities mentioned in the following passage together with their properties.

  Passage:
  <%= @input %>"

  @doc """
  Run the data extraction chain.
  """
  @spec run(ChatOpenAI.t(), schema :: map(), prompt :: [any()], opts :: Keyword.t()) ::
          {:ok, result :: [any()]} | {:error, String.t()}
  def run(llm, schema, prompt, opts \\ []) do
    verbose = Keyword.get(opts, :verbose, false)

    try do
      messages =
        [
          Message.new_system!(
            "You are a helpful assistant that extracts structured data from text passages. Only use the functions you have been provided with."
          ),
          PromptTemplate.new!(%{role: :user, text: @extraction_template})
        ]
        |> PromptTemplate.to_messages!(%{input: prompt})

      {:ok, chain} = LLMChain.new(%{llm: llm, verbose: verbose})

      chain
      |> LLMChain.add_functions(build_extract_function(schema))
      |> LLMChain.add_messages(messages)
      |> LLMChain.run()
      |> case do
        {:ok, _updated_chain, %Message{role: :function_call, arguments: %{"info" => info}}}
        when is_list(info) ->
          {:ok, info}

        other ->
          IO.inspect(other, label: "???????")
          {:error, "Unexpected response. #{inspect(other)}"}
      end
    rescue
      exception ->
        Logger.warning(
          "Caught unexpected exception in DataExtractionChain. Error: #{inspect(exception)}"
        )

        {:error, "Unexpected error in DataExtractionChain. Check logs for details."}
    end
  end

  @doc """
  Build the function to expose to the LLM that can be called for data
  extraction.
  """
  @spec build_extract_function(schema :: map()) :: Function.t() | no_return()
  def build_extract_function(schema) do
    Langchain.Function.new!(%{
      name: "information_extraction",
      description: "Extracts the relevant information from the passage.",
      parameters_schema: %{
        type: "object",
        properties: %{
          info: %{
            type: "array",
            items: schema
          }
        },
        required: ["info"]
      }
    })
  end
end
