defmodule Langchain.Chains.ExtractionChain do
  @doc """
  Define an LLMChain
  """
  use Ecto.Schema
  import Ecto.Changeset
  require Logger
  alias Langchain.PromptTemplate
  alias __MODULE__
  alias Langchain.Message
  alias Langchain.Chains.LLMChain

  @extraction_template ~s"Extract and save the relevant entities mentioned in the following passage together with their properties.

  Passage:
  <%= @input %>"

  # TODO: Return a %LLMChain{} that is setup with the function and knows the base prompt. Also handles parsing the response.
  # stream: false,

  # TODO: Need to either store the schema definition on the generic struct (like a map of nested, custom config) or only pass it into the call that runs the process.

  def run(llm, schema, prompt, opts \\ []) do
    verbose = Keyword.get(opts, :verbose, false)
    messages = [
      Message.new_system!("You are a helpful assistant that extracts structured data from text passages. Only use the functions you have been provided with."),
      PromptTemplate.new!(%{role: :user, text: @extraction_template})
    ]
    {:ok, function} = build_extract_function(schema)

    with {:ok, chain} <-
           LLMChain.new(%{prompt: messages, llm: llm, functions: [function], verbose: verbose}),
         {:ok, %{"info" => info}} <- LLMChain.call_chat(chain, %{input: prompt}) do
      {:ok, info}
    else
      other ->
        IO.inspect(other, label: "???????")
    end
  end

  # TODO: Move this to a different module?
  # https://github.com/hwchase17/langchainjs/blob/main/langchain/src/chains/openai_functions/extraction.ts#L42

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
      # TODO: Need to test support for this. Pass in a schema definition of type, properties and required.
      # function getExtractionFunctions(schema: FunctionParameters) {
      #   return [
      #     {
      #       name: "information_extraction",
      #       description: "Extracts the relevant information from the passage.",
      #       parameters: {
      #         type: "object",
      #         properties: {
      #           info: {
      #             type: "array",
      #             items: {
      #               type: schema.type,
      #               properties: schema.properties,
      #               required: schema.required,
      #             },
      #           },
      #         },
      #         required: ["info"],
      #       },
      #     },
      #   ];
      # }
      # ],
      # required: ["info"]
    })
  end

  # export function createExtractionChain(
  #   schema: FunctionParameters,
  #   llm: ChatOpenAI
  # ) {
  #   const functions = getExtractionFunctions(schema);
  #   const prompt = PromptTemplate.fromTemplate(_EXTRACTION_TEMPLATE);
  #   const outputParser = new JsonKeyOutputFunctionsParser({ attrName: "info" });
  #   return new LLMChain({
  #     llm,
  #     prompt,
  #     llmKwargs: { functions },
  #     outputParser,
  #     tags: ["openai_functions", "extraction"],
  #   });
  # }
end
