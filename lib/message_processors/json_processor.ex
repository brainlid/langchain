defmodule LangChain.MessageProcessors.JsonProcessor do
  @moduledoc """
  A built-in Message processor that processes a received Message for JSON
  contents.

  When successful, the assistant message is replaced with one containing the
  parsed JSON data as an Elixir map data structure. No additional validation or
  processing of the data is done in by this processor.

  When JSON data is expected but not received, or the received JSON is invalid
  or incomplete, a new user `Message` struct is returned with a text error
  message for the LLM so it can try again to correct the error and return a
  valid response.

  There are multiple ways to extract JSON content.

  When the JSON data is reliably returned as the only response, this extracts it
  to an Elixir Map:

      message = Message.new_assistant!(%{content: "{\"value\": 123}"})

      # process the message for JSON content
      {:continue, updated_chain, updated_message} =
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

      {:continue, updated_chain, updated_message} =
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

  - `{:continue, %Message{}}` - The returned message replaces the one being
    processed and no additional processors are run.
  - `{:halt, %LLMChain{}, %Message{}}` - Pre-processors are halted. An updated
    LLMChain can be returned and the included Message is returned as a response
    to the LLM. This is for handling errors.
  """
  @spec run(LLMChain.t(), Message.t()) ::
          {:continue, Message.t()} | {:halt, LLMChain.t(), Message.t()}
  def run(%LLMChain{} = chain, %Message{} = message) do
    case Jason.decode(message.content) do
      {:ok, parsed} ->
        {:continue, %Message{message | content: parsed}}

      {:error, %Jason.DecodeError{} = error} ->
        error_message = Jason.DecodeError.message(error)
        {:halt, chain, Message.new_user!("ERROR: Invalid JSON data: #{error_message}")}
    end
  end

  def run(%LLMChain{} = chain, %Message{} = message, regex_pattern) do
    case Regex.run(regex_pattern, message.content, capture: :all_but_first) do
      [json] ->
        if chain.verbose, do: IO.puts("Extracted JSON text from message")
        # run recursive call on just the extracted JSON
        run(chain, %Message{message | content: json})

      _ ->
        {:halt, chain, Message.new_user!("ERROR: No JSON found")}
    end
  end
end
