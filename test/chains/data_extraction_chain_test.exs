defmodule Langchain.Chains.DataExtractionChainTest do
  use Langchain.BaseCase

  doctest Langchain.Chains.DataExtractionChain
  alias Langchain.ChatModels.ChatOpenAI

  # Extraction - https://js.langchain.com/docs/modules/chains/openai_functions/extraction
  @tag :live_call
  test "data extraction chain" do
    # JSONSchema definition
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

    # run the chain, chain.run(prompt to extract data from)
    data_prompt =
      "Alex is 5 feet tall. Claudia is 4 feet taller than Alex and jumps higher than him.
       Claudia is a brunette and Alex is blonde. Alex's dog Frosty is a labrador and likes to play hide and seek."

    {:ok, result} = Langchain.Chains.DataExtractionChain.run(chat, schema_parameters, data_prompt, verbose: true)

    assert result == [
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
  end
end
