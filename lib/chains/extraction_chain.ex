defmodule Langchain.Chains.ExtractionChain do
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
  @spec run(LLMChain.t(), schema :: map(), prompt :: [any()], opts :: Keyword.t()) :: {:ok, result :: [any()]} | {:error, String.t()}
  def run(llm, schema, prompt, opts \\ []) do
    verbose = Keyword.get(opts, :verbose, false)
    messages = [
      Message.new_system!("You are a helpful assistant that extracts structured data from text passages. Only use the functions you have been provided with."),
      PromptTemplate.new!(%{role: :user, text: @extraction_template})
    ]
    {:ok, function} = build_extract_function(schema)

    with {:ok, chain} <-
           LLMChain.new(%{prompt: messages, llm: llm, functions: [function], verbose: verbose}),
         {:ok, %{"info" => info}} when is_list(info) <- LLMChain.call_chat(chain, %{input: prompt}) do
      {:ok, info}
    else
      other ->
        IO.inspect(other, label: "???????")
        {:error, "Unexpected response. #{inspect other}"}
    end
  end

  @doc """
  Build the function to expose to the LLM that can be called for data
  extraction.
  """
  @spec build_extract_function(schema :: map()) ::
          {:ok, Function.t()} | {:error, Ecto.Changeset.t()}
  def build_extract_function(schema) do
    Langchain.Functions.Function.new(%{
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
