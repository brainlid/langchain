defmodule LangChain.MessageProcessors.JsonProcessor do
  @moduledoc """
  A built-in Message processor that processes a received Message for JSON
  contents.

  When successful, the assistant message's JSON contents are processed into a
  map and set on `processed_content`. No additional validation or processing of
  the data is done in by this processor.

  When JSON data is expected but not received, or the received JSON is invalid
  or incomplete, a new user `Message` struct is returned with a text error
  message for the LLM so it can try again to correct the error and return a
  valid response.

  There are multiple ways to extract JSON content.

  When the JSON data is reliably returned as the only response, this extracts it
  to an Elixir Map:

      message = Message.new_assistant!(%{content: "{\"value\": 123}"})

      # process the message for JSON content
      {:cont, updated_chain, updated_message} =
        JsonProcessor.run(chain, message)

  The updated message will be an assistant message where content is a map:

      updated_message.content
      #=> %{"value" => 123}

  Some models are unable to reliably return a JSON response without adding some
  commentary. For that situation, instruct the model how to format the JSON
  content. Depending on the model, one of these formats may work better than
  another:

      # bracketing the data with XML tags
      <json>
      {"value": 123}
      </json>

      # markdown style code fences with json language
      ```json
      {"value": 123}
      ```

      # markdown style code fences (no language)
      ```
      {"value": 123}
      ```

  When the LLM adds commentary with the data, it may appear like this:

      The answer to your question in JSON is:

      ```json
      {"value": 123}
      ```

  We can still extract the JSON data in a situation like this. We provide a
  Regex to use for extracting the data from whatever starting and ending text
  the LLM was instructed to use.

  Examples:

      ~r/<json>(.*?)<\/json>/s
      ~r/```json(.*?)```/s
      ~r/```(.*?)```/s

  The "```json" formatted one is processed like this:

      {:cont, updated_chain, updated_message} =
        JsonProcessor.run(chain, message, ~r/```json(.*?)```/s)

  """
  alias LangChain.Chains.LLMChain
  alias LangChain.Message

  @doc """
  Returns a function for use in a `LangChain.Chains.LLMChain.add_processors/2`.
  """
  def new!() do
    # Doesn't need any currying.
    &run/2
  end

  @doc """
  Returns a wrapped (curried) function for use in a
  `LangChain.Chains.LLMChain.add_processors/2` that includes the configured
  Regex to use for extracting JSON content.

  The Regex pattern is used with the `:all_but_first` capture option to extract
  just the internal capture text.
  """
  def new!(%Regex{} = regex) do
    # Curry in the regex option
    &run(&1, &2, regex)
  end

  @doc """
  Run the JSON Processor on a message. The response indicates what should be
  done with the message.

  Response values:

  - `{:cont, %Message{}}` - The returned message replaces the one being
    processed and no additional processors are run.
  - `{:halt, %Message{}}` - Future processors are skipped. The Message is
    returned as a response to the LLM for reporting errors.
  """
  @spec run(LLMChain.t(), Message.t()) ::
          {:cont, Message.t()} | {:halt, Message.t()}
  def run(%LLMChain{} = chain, %Message{} = message) do
    case Jason.decode(message.processed_content) do
      {:ok, parsed} ->
        if chain.verbose, do: IO.puts("Parsed JSON text to a map")
        {:cont, %Message{message | processed_content: parsed}}

      {:error, %Jason.DecodeError{} = error} ->
        error_message = Jason.DecodeError.message(error)
        {:halt, Message.new_user!("ERROR: Invalid JSON data: #{error_message}")}
    end
  end

  def run(%LLMChain{} = chain, %Message{} = message, regex_pattern) do
    case Regex.run(regex_pattern, message.processed_content, capture: :all_but_first) do
      [json] ->
        if chain.verbose, do: IO.puts("Extracted JSON text from message")
        # run recursive call on just the extracted JSON
        run(chain, %Message{message | processed_content: json})

      _ ->
        {:halt, Message.new_user!("ERROR: No JSON found")}
    end
  end
end
