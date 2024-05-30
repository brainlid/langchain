defmodule LangChain.MessageProcessors.JsonProcessorTest do
  use LangChain.BaseCase
  alias LangChain.Chains.LLMChain
  alias LangChain.Message
  alias LangChain.ChatModels.ChatOpenAI
  alias LangChain.MessageProcessors.JsonProcessor

  doctest LangChain.MessageProcessors.JsonProcessor

  @json_xml_regex ~r/<json>(.*?)<\/json>/s
  @json_backticks_json_regex ~r/```json(.*?)```/s
  @json_backticks_regex ~r/```(.*?)```/s

  setup _ do
    {:ok, model} = ChatOpenAI.new(%{temperature: 0})
    {:ok, chain} = LLMChain.new(%{llm: model})

    %{model: model, chain: chain}
  end

  describe "new/0" do
    test "returns a function ready for use in a chain", %{chain: chain} do
      data = %{"subject" => "RE: Things that happen", "body" => "Main message body"}

      processor = JsonProcessor.new!()
      message = Message.new_assistant!(%{content: Jason.encode!(data)})
      assert {:continue, updated_message} = processor.(chain, message)

      # contents are converted back from JSON to a map
      assert updated_message.content == data
    end
  end

  describe "new/1" do
    test "returns a curried function with the regex ready", %{chain: chain} do
      data = %{"subject" => "RE: Things that happen", "body" => "Main message body"}

      message_text = """
      Here's your requested JSON data:

      ```json
      #{Jason.encode!(data)}
      ```

      I hope it meets your needs!
      """

      message = Message.new_assistant!(%{content: message_text})

      processor = JsonProcessor.new!(@json_backticks_json_regex)
      assert {:continue, updated_message} = processor.(chain, message)

      # contents are converted back from JSON to a map
      assert updated_message.content == data
    end
  end

  describe "run/2" do
    test "converts JSON text to a map", %{chain: chain} do
      data = %{"subject" => "RE: Things that happen", "body" => "Main message body"}

      message = Message.new_assistant!(%{content: Jason.encode!(data)})
      assert {:continue, updated_message} = JsonProcessor.run(chain, message)
      # contents are converted back from JSON to a map
      assert updated_message.content == data
    end

    test "halts and returns error when JSON parse fails", %{chain: chain} do
      invalid_data = "{\"body\":\"Main message body\",\"subj"
      message = Message.new_assistant!(%{content: invalid_data})
      assert {:halt, updated_chain, returned_message} = JsonProcessor.run(chain, message)
      # chain is not affected
      assert updated_chain == chain
      # contents are converted back from JSON to a map
      assert returned_message.role == :user

      assert returned_message.content ==
               "ERROR: Invalid JSON data: unexpected end of input at position 33"
    end
  end

  describe "run/4" do
    test "converts JSON content from <json> tags", %{chain: chain} do
      data = %{"subject" => "RE: Things that happen", "body" => "Main message body"}

      message_text = """
      Here's your requested JSON data:

      <json>
      #{Jason.encode!(data)}
      </json>

      I hope it meets your needs!
      """

      message = Message.new_assistant!(%{content: message_text})

      assert {:continue, updated_message} = JsonProcessor.run(chain, message, @json_xml_regex)
      # json is extracted and converted to a map
      assert updated_message.content == data
    end

    test "converts JSON content from ```json fences", %{chain: chain} do
      data = %{"subject" => "RE: Things that happen", "body" => "Main message body"}

      message_text = """
      Here's your requested JSON data:

      ```json
      #{Jason.encode!(data)}
      ```

      I hope it meets your needs!
      """

      message = Message.new_assistant!(%{content: message_text})

      assert {:continue, updated_message} =
               JsonProcessor.run(chain, message, @json_backticks_json_regex)

      # json is extracted and converted to a map
      assert updated_message.content == data
    end

    test "converts JSON content from ``` fences", %{chain: chain} do
      data = %{"subject" => "RE: Things that happen", "body" => "Main message body"}

      message_text = """
      Here's your requested JSON data:

      ```
      #{Jason.encode!(data)}
      ```

      I hope it meets your needs!
      """

      message = Message.new_assistant!(%{content: message_text})

      assert {:continue, updated_message} =
               JsonProcessor.run(chain, message, @json_backticks_regex)

      # json is extracted and converted to a map
      assert updated_message.content == data
    end

    test "supports when code blocks are at start and end", %{chain: chain} do
      data = %{"subject" => "RE: Things that happen", "body" => "Main message body"}

      message_text = "```json\n#{Jason.encode!(data)}\n```"

      message = Message.new_assistant!(%{content: message_text})

      assert {:continue, updated_message} =
               JsonProcessor.run(chain, message, @json_backticks_json_regex)

      # json is extracted and converted to a map
      assert updated_message.content == data
    end

    test "halts when expected JSON content not found", %{chain: chain} do
      data = %{"subject" => "RE: Things that happen", "body" => "Main message body"}

      message_text = "There is content, but no JSON to be found!"

      message = Message.new_assistant!(%{content: message_text})

      assert {:halt, updated_chain, returned_message} =
               JsonProcessor.run(chain, message, @json_backticks_json_regex)

      # the chain is not updated
      assert updated_chain == chain
      # json is extracted and converted to a map
      assert returned_message.role == :user
      assert returned_message.content == "ERROR: No JSON found"
    end

    test "halts when JSON content does not parse", %{chain: chain} do
      message_text = "```json\n{\"thing\"```"

      message = Message.new_assistant!(%{content: message_text})

      assert {:halt, updated_chain, returned_message} =
               JsonProcessor.run(chain, message, @json_backticks_json_regex)

      # the chain is not updated
      assert updated_chain == chain
      # json is extracted and converted to a map
      assert returned_message.role == :user

      assert returned_message.content ==
               "ERROR: Invalid JSON data: unexpected end of input at position 9"
    end
  end
end
