defmodule LangChain.Chains.DataExtractionChain do
  @moduledoc """
  Defines an LLMChain for performing data extraction from a body of text.

  Provide the schema for desired information to be parsed into. It is treated as
  though there are 0 to many instances of the data structure being described so
  information is returned as an array.

  Originally based on:
  - https://github.com/hwchase17/langchainjs/blob/main/langchain/src/chains/openai_functions/extraction.ts#L42

  ## Example

      # JSONSchema definition of data we want to capture or extract.
      schema_parameters = %{
        type: "object",
        properties: %{
          person_name: %{type: "string"},
          person_age: %{type: "number"},
          person_hair_color: %{type: "string"},
          dog_name: %{type: "string"},
          dog_breed: %{type: "string"}
        },
        required: []
      }

      # Model setup
      {:ok, chat} = ChatOpenAI.new(%{temperature: 0})

      # run the chain on the text information
      data_prompt =
        "Alex is 5 feet tall. Claudia is 4 feet taller than Alex and jumps higher than him.
        Claudia is a brunette and Alex is blonde. Alex's dog Frosty is a labrador and likes to play hide and seek."

      {:ok, result} = LangChain.Chains.DataExtractionChain.run(chat, schema_parameters, data_prompt)

      # Example result
      [
        %{
          "dog_breed" => "labrador",
          "dog_name" => "Frosty",
          "person_age" => nil,
          "person_hair_color" => "blonde",
          "person_name" => "Alex"
        },
        %{
          "dog_breed" => nil,
          "dog_name" => nil,
          "person_age" => nil,
          "person_hair_color" => "brunette",
          "person_name" => "Claudia"
        }
      ]
  """
  use Ecto.Schema
  require Logger
  alias LangChain.PromptTemplate
  alias LangChain.Message
  alias LangChain.Chains.LLMChain
  alias LangChain.ChatModels.ChatOpenAI

  @function_name "information_extraction"
  @extraction_template ~s"Extract and save the relevant entities mentioned in the following passage together with their properties. Use the value `null` when missing in the passage.

  Passage:
  <%= @input %>"

  @doc """
  Run the data extraction chain.
  """
  @spec run(ChatOpenAI.t(), json_schema :: map(), prompt :: [any()], opts :: Keyword.t()) ::
          {:ok, result :: [any()]} | {:error, String.t()}
  def run(llm, json_schema, prompt, opts \\ []) do
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
      |> LLMChain.add_functions(build_extract_function(json_schema))
      |> LLMChain.add_messages(messages)
      |> LLMChain.run()
      |> case do
        {:ok, _updated_chain,
         %Message{
           role: :assistant,
           function_name: @function_name,
           arguments: %{"info" => info}
         }}
        when is_list(info) ->
          {:ok, info}

        other ->
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
  @spec build_extract_function(json_schema :: map()) :: LangChain.Function.t() | no_return()
  def build_extract_function(json_schema) do
    LangChain.Function.new!(%{
      name: @function_name,
      description: "Extracts the relevant information from the passage.",
      parameters_schema: %{
        type: "object",
        properties: %{
          info: %{
            type: "array",
            items: json_schema
          }
        },
        required: ["info"]
      }
    })
  end
end
