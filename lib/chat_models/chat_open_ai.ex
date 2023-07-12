defmodule Langchain.ChatModels.ChatOpenAI do
  @moduledoc """
  Represents the OpenAI ChatModel.

  Parses and validates inputs for making a request from the API.

  https://platform.openai.com/docs/api-reference/chat/create

  - https://github.com/openai/openai-cookbook/blob/main/examples/How_to_call_functions_with_chat_models.ipynb
  """
  use Ecto.Schema
  require Logger
  import Ecto.Changeset
  import Langchain.Utils.ApiOverride
  alias __MODULE__
  alias Langchain.Message
  alias Langchain.ForOpenAIApi
  alias Langchain.Utils

  @primary_key false
  embedded_schema do
    field(:endpoint, :string, default: "https://api.openai.com/v1/chat/completions")
    field(:model, :string, default: "gpt-3.5-turbo")
    field(:temperature, :float, default: 0.0)
    field(:frequency_penalty, :float, default: 0.0)
    # How many chat completion choices to generate for each input message.
    field(:n, :integer, default: 1)
    field(:stream, :boolean, default: false)

    # embeds_many :messages, OpenAIMessage
    # embeds_many :functions, OpenAIFunctions
  end

  @type t :: %ChatOpenAI{}

  @create_fields [:model, :temperature, :frequency_penalty, :n, :stream]
  @required_fields [:model]

  @spec new(attrs :: map()) :: {:ok, t} | {:error, Ecto.Changeset.t()}
  def new(%{} = attrs \\ %{}) do
    %ChatOpenAI{}
    |> cast(attrs, @create_fields)
    |> common_validation()
    |> apply_action(:insert)
  end

  def common_validation(changeset) do
    changeset
    |> validate_required(@required_fields)
    |> validate_number(:temperature, greater_than_or_equal_to: 0, less_than_or_equal_to: 2)
    |> validate_number(:frequency_penalty, greater_than_or_equal_to: -2, less_than_or_equal_to: 2)
    |> validate_number(:n, greater_than_or_equal_to: 1)
  end

  @doc """
  Return the params to send in an API request.
  """
  @spec for_api(t, message :: [map()], functions :: [map()]) :: %{atom() => any()}
  def for_api(%ChatOpenAI{} = openai, messages, functions) do
    %{
      model: openai.model,
      temperature: openai.temperature,
      frequency_penalty: openai.frequency_penalty,
      n: openai.n,
      stream: openai.stream,
      messages: Enum.map(messages, &ForOpenAIApi.for_api/1)
    }
    |> Utils.conditionally_add_to_map(:functions, get_functions_for_api(functions))
  end

  defp get_functions_for_api(nil), do: []

  defp get_functions_for_api(functions) do
    Enum.map(functions, &ForOpenAIApi.for_api/1)
  end

  @doc """
  Call the API passing the ChatOpenAI struct with configuration, plus either a
  simple message or the list of messages to act as the prompt. Optionally pass
  in a list of functions available to the LLM for requesting execution in
  response.
  """
  @spec call(t(), String.t() | [Message.t()], [Function.t()]) ::
          {:ok, map()} | {:error, String.t()}
  def call(openai, prompt, functions \\ [])

  def call(%ChatOpenAI{} = openai, prompt, functions) when is_binary(prompt) do
    messages =
      [
        Message.new_system!(),
        Message.new_user!(prompt)
      ]
      |> dbg()

    call(openai, messages, functions)
  end

  def call(%ChatOpenAI{} = openai, messages, functions) when is_list(messages) do
    if override_api_return?() do
      Logger.warning("Found override API response. Will not make live API call.")

      case get_api_override() do
        {:ok, response} ->
          response

        _other ->
          raise LangchainError,
                "An unexpected fake API response was set. Should be an `{:ok, value}`"
      end
    else
      # TODO: make api request
      with %Req.Response{status: 200, body: data} <-
             do_api_request(openai, messages, functions),
           {:ok, parsed} <- do_process_response(data) do
        # TODO: callbacks?
        # TODO: handle parsed? What to return? return parsed?
        {:ok, parsed}

        # TODO: return a ChatState? Updated with messages added?
      else
        %Req.Response{} = error_response ->
          do_process_error_response(error_response)

          # error condition: stopped because token length is too long

          # %{"choices" => [%{"finish_reason" => "length"}]} ->
          #   # it stopped because it reached too many tokens. Update it to not
          #   # show as edited.
          #   message = Messages.get_message_by_index!(conversation_id, index)
          #   {:ok, _updated} = Messages.update_message(message, %{edited: false})
          #   send(pid, {:chat_error, "Stopped for length"})
          #   :ok

          # # error condition: we got an error response from ChatGPT
          # %{"error" => %{"message" => message}} ->
          #   send(pid, {:chat_error, message})

          # # error condition: something else went wrong. May not have been able
          # # to reach the server, etc.
          # other ->
          #   Logger.error("ChatGPT failure: #{inspect(other)}")
          #   send(pid, {:chat_error, inspect(other)})
          #   {:error, "Unexpected response from API"}
      end
    end
  end

  def do_api_request(%ChatOpenAI{stream: false} = openai, messages, functions) do
    Req.post!(openai.endpoint,
      json: for_api(openai, messages, functions),
      auth: {:bearer, System.fetch_env!("OPENAPI_KEY")}
    )
  end

  # Parse a new message response
  def do_process_response(%{"choices" => [%{"finish_reason" => "stop", "message" => message}]}) do
    case Message.new(message) do
      {:ok, message} ->
        {:ok, message}

      {:error, changeset} ->
        {:error, Utils.changeset_error_to_string(changeset)}
    end
  end

  def do_process_error_response(%Req.Response{
        status: status,
        body: %{"error" => %{"message" => message}}
      }) do
    Logger.error("OpenAI error status #{inspect(status)}. Reason: #{inspect(message)}")
    {:error, inspect(message)}
  end

  def do_process_error_response(%Req.Response{status: status, body: data}) do
    Logger.error(
      "DIDN'T RECEIVE A SUCCESS FROM API. Status: #{inspect(status)}, Body: #{inspect(data)}"
    )

    {:error, "Unexpected response"}
  end

  # def do_process_response(data) do
  #   # handle the received data
  #   case data do
  #     # starting a new assistant response
  #     %{"choices" => [%{"delta" => %{"role" => "assistant"}}]} ->
  #       {:ok, message} =
  #         Messages.create_message(conversation_id, %{"role" => "assistant", "index" => index})

  #       send(pid, {:chat_response, message, {:start, self()}})

  #     # adding data to a response
  #     %{"choices" => [%{"delta" => %{"content" => content}}]} ->
  #       message = Messages.get_message_by_index!(conversation_id, index)
  #       new_content = (message.content || "") <> content
  #       {:ok, updated} = Messages.update_message(message, %{content: new_content})
  #       send(pid, {:chat_response, updated, {:update, self()}})
  #       nil

  #     # Execute a function_call
  #     %{
  #       "choices" => [
  #         %{
  #           "finish_reason" => "function_call",
  #           "message" => %{"function_call" => %{"arguments" => raw_args, "name" => name}}
  #         }
  #       ]
  #     } ->
  #       # NOTE: JSON from the LLM may not be valid. Handle that situation.
  #       # TODO: May receive multiple function calls in a single response.
  #       Langchain.Tools.Calculator.parse(raw_args)

  #     # TODO:
  #     # # Step 4: send the info on the function call and function response to GPT
  #     # messages.append(response_message)  # extend conversation with assistant's reply
  #     # messages.append(
  #     #     {
  #     #         "role": "function",
  #     #         "name": function_name,
  #     #         "content": function_response,
  #     #     }
  #     # )  # extend conversation with function response
  #     # second_response = openai.ChatCompletion.create(
  #     #     model="gpt-3.5-turbo-0613",
  #     #     messages=messages,
  #     # )  # get a new response from GPT where it can see the function response
  #     # return second_response

  #     # TODO: Include "Only use the functions you have been provided with." in system message.

  #     # response is finished
  #     %{"choices" => [%{"delta" => %{}, "finish_reason" => "stop"}]} ->
  #       # we received the final message. Flag it as "original" and unedited
  #       message = Messages.get_message_by_index!(conversation_id, index)
  #       {:ok, updated} = Messages.update_message(message, %{edited: false})
  #       send(pid, {:chat_response, updated, {:done, self()}})
  #       :ok

  #     # error condition: stopped because token length is too long
  #     %{"choices" => [%{"finish_reason" => "length"}]} ->
  #       # it stopped because it reached too many tokens. Update it to not
  #       # show as edited.
  #       message = Messages.get_message_by_index!(conversation_id, index)
  #       {:ok, _updated} = Messages.update_message(message, %{edited: false})
  #       send(pid, {:chat_error, "Stopped for length"})
  #       :ok

  #     # error condition: we got an error response from ChatGPT
  #     %{"error" => %{"message" => message}} ->
  #       send(pid, {:chat_error, message})

  #     # error condition: something else went wrong. May not have been able
  #     # to reach the server, etc.
  #     other ->
  #       Logger.error("ChatGPT failure: #{inspect(other)}")
  #       send(pid, {:chat_error, inspect(other)})
  #   end
  # end


  #TODO: Move this to a different module?
  # https://github.com/hwchase17/langchainjs/blob/main/langchain/src/chains/openai_functions/extraction.ts#L42
  @extraction_template ~s"Extract and save the relevant entities mentioned in the following passage together with their properties.

  Passage:
  <%= @input %>"

  def get_extraction_functions() do
    Langchain.Functions.new(%{
      name: "information_extraction",
      description: "Extracts the relevant information from the passage.",
      parameters: [
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
      ],
      required: ["info"]
    })
  end
end
