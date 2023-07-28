defmodule Langchain.ChatModels.ChatOpenAI do
  @moduledoc """
  Represents the OpenAI ChatModel.

  Parses and validates inputs for making a request from the API.

  https://platform.openai.com/docs/api-reference/chat/create

  - https://github.com/openai/openai-cookbook/blob/main/examples/How_to_call_functions_with_chat_models.ipynb

  Converts responses into more explicit data structures.
  """
  use Ecto.Schema
  require Logger
  import Ecto.Changeset
  import Langchain.Utils.ApiOverride
  alias __MODULE__
  alias Langchain.Message
  alias Langchain.LangchainError
  alias Langchain.ForOpenAIApi
  alias Langchain.Utils
  alias Langchain.MessageDelta

  # NOTE: As of gpt-4 and gpt-3.5, only one function_call is issued at a time
  # even when multiple requests could be issued based on the prompt.

  @primary_key false
  embedded_schema do
    field :endpoint, :string, default: "https://api.openai.com/v1/chat/completions"
    field :model, :string, default: "gpt-4"
    # field :model, :string, default: "gpt-3.5-turbo"
    field :temperature, :float, default: 0.0
    field :frequency_penalty, :float, default: 0.0
    # How many chat completion choices to generate for each input message.
    field :n, :integer, default: 1
    field :stream, :boolean, default: false

    # A callback function to execute when a message is received.
    field :callback_fn, :any, virtual: true
  end

  @type t :: %ChatOpenAI{}

  @create_fields [:model, :temperature, :frequency_penalty, :n, :stream, :callback_fn]
  @required_fields [:model]

  @doc """
  Setup a ChatOpenAI client configuration.
  """
  @spec new(attrs :: map()) :: {:ok, t} | {:error, Ecto.Changeset.t()}
  def new(%{} = attrs \\ %{}) do
    %ChatOpenAI{}
    |> cast(attrs, @create_fields)
    |> common_validation()
    |> apply_action(:insert)
  end

  @doc """
  Setup a ChatOpenAI client configuration and return it or raise an error if invalid.
  """
  @spec new!(attrs :: map()) :: t() | no_return()
  def new!(attrs \\ %{}) do
    case new(attrs) do
      {:ok, chain} ->
        chain

      {:error, changeset} ->
        raise LangchainError, changeset
    end
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
    messages = [
      Message.new_system!(),
      Message.new_user!(prompt)
    ]

    call(openai, messages, functions)
  end

  def call(%ChatOpenAI{} = openai, messages, functions) when is_list(messages) do
    if override_api_return?() do
      Logger.warning("Found override API response. Will not make live API call.")

      case get_api_override() do
        {:ok, {:ok, data} = response} ->
          # fire callback for face responses too
          fire_callback(openai, data)
          response

        _other ->
          raise LangchainError,
                "An unexpected fake API response was set. Should be an `{:ok, value}`"
      end
    else
      # make base api request and perform high-level success/failure checks
      case do_api_request(openai, messages, functions) do
        %Req.Response{status: 200, body: parsed_data} ->
          {:ok, parsed_data}

        %Req.Response{} = error_response ->
          do_process_error_response(error_response)
      end

      # with %Req.Response{status: 200, body: data} <-
      #        do_api_request(openai, messages, functions),
      #      {:ok, parsed} <- do_process_response(data) do
      #   # TODO: callbacks?
      #   # TODO: handle parsed? What to return? return parsed?
      #   {:ok, parsed}

      #   # TODO: return a ChatState? Updated with messages added?
      # else
      #   %Req.Response{} = error_response ->
      #     do_process_error_response(error_response)

      #     # error condition: stopped because token length is too long

      #     # %{"choices" => [%{"finish_reason" => "length"}]} ->
      #     #   # it stopped because it reached too many tokens. Update it to not
      #     #   # show as edited.
      #     #   message = Messages.get_message_by_index!(conversation_id, index)
      #     #   {:ok, _updated} = Messages.update_message(message, %{edited: false})
      #     #   send(pid, {:chat_error, "Stopped for length"})
      #     #   :ok

      #     # # error condition: we got an error response from ChatGPT
      #     # %{"error" => %{"message" => message}} ->
      #     #   send(pid, {:chat_error, message})

      #     # # error condition: something else went wrong. May not have been able
      #     # # to reach the server, etc.
      #     # other ->
      #     #   Logger.error("ChatGPT failure: #{inspect(other)}")
      #     #   send(pid, {:chat_error, inspect(other)})
      #     #   {:error, "Unexpected response from API"}
      # end
    end
  end

  # Make the API request from the OpenAI server.
  #
  # If `stream: false`, the completed message is returned.
  #
  # If `stream: true`, the `callback_fn` is executed for the returned MessageDelta
  # responses.
  #
  # Executes the callback function passing the response only parsed to the data
  # structures.
  @doc false
  def do_api_request(%ChatOpenAI{stream: false} = openai, messages, functions) do
    response =
      Req.post!(openai.endpoint,
        json: for_api(openai, messages, functions),
        auth: {:bearer, System.fetch_env!("OPENAPI_KEY")}
      )

    # parse the body and return it as parsed structs
    case response do
      %Req.Response{status: 200, body: data} ->
        body = do_process_response(data)
        fire_callback(openai, body)
        %Req.Response{response | body: body}

      other ->
        other
    end
  end

  def do_api_request(%ChatOpenAI{stream: true} = openai, messages, functions) do
    finch_fun = fn request, finch_request, finch_name, finch_options ->
      resp_fun = fn
        {:status, status}, response ->
          %{response | status: status}

        {:headers, headers}, response ->
          %{response | headers: headers}

        {:data, data}, response ->
          # cleanup data because it isn't structured well for JSON.
          body = decode_streamed_data(data)
          # execute the callback function for each MessageDelta
          fire_callback(openai, body)
          old_body = if response.body == "", do: [], else: response.body

          # Returns %Req.Response{} where the body contains ALL the stream delta
          # chunks converted to MessageDelta structs. The body is a list of lists like this...
          #
          # body: [
          #         [
          #           %Langchain.MessageDelta{
          #             content: nil,
          #             index: 0,
          #             function_name: nil,
          #             role: :assistant,
          #             arguments: nil,
          #             complete: false
          #           }
          #         ],
          #         ...
          #       ]
          #
          # The reason for the inner list is for each entry in the "n" choices. By default only 1.
          %{response | body: old_body ++ body}
      end

      # TODO: Merge deltas and return that as the result? Return the raw combined request body? Not even sure we care about it.
      # TODO: Create callback function to receive deltas
      # TODO: Chain tracks last message and merges them through callback.
      #      - callback needs to be received by a LiveView.

      case Finch.stream(finch_request, finch_name, Req.Response.new(), resp_fun, finch_options) do
        {:ok, response} ->
          {request, response}

        {:error, exception} ->
          Logger.error("Failed request to API: #{inspect(exception)}")
          {request, exception}
      end
    end

    # NOTE: The POST response includes a list of body messages that were
    # received during the streaming process. However, the messages in the
    # response all come at once when the stream is complete. It is blocking
    # until it completes. This means the streaming call should happen in a
    # separate process from the UI and the callback function will process the
    # chunks and should notify the UI process of the additional data.
    Req.post!(openai.endpoint,
      json: for_api(openai, messages, functions),
      auth: {:bearer, System.fetch_env!("OPENAPI_KEY")},
      finch_request: finch_fun
    )
  end

  defp decode_streamed_data(data) do
    # Data comes back like this:
    #
    # "data: {\"id\":\"chatcmpl-7e8yp1xBhriNXiqqZ0xJkgNrmMuGS\",\"object\":\"chat.completion.chunk\",\"created\":1689801995,\"model\":\"gpt-4-0613\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":null,\"function_call\":{\"name\":\"calculator\",\"arguments\":\"\"}},\"finish_reason\":null}]}\n\n
    #  data: {\"id\":\"chatcmpl-7e8yp1xBhriNXiqqZ0xJkgNrmMuGS\",\"object\":\"chat.completion.chunk\",\"created\":1689801995,\"model\":\"gpt-4-0613\",\"choices\":[{\"index\":0,\"delta\":{\"function_call\":{\"arguments\":\"{\\n\"}},\"finish_reason\":null}]}\n\n"
    #
    # In that form, the data is not ready to be interpreted as JSON. Let's clean
    # it up first.

    data
    |> String.split("data: ")
    |> Enum.map(fn str ->
      str
      |> String.trim()
      |> case do
        "" ->
          :empty

        "[DONE]" ->
          :empty

        json ->
          json
          |> Jason.decode!()
          |> do_process_response()
      end
    end)
    # returning a list of elements. "junk" elements were replaced with `:empty`.
    # Filter those out down and return the final list of MessageDelta structs.
    |> Enum.filter(fn d -> d != :empty end)
  end

  # fire the callback if present.
  defp fire_callback(%ChatOpenAI{callback_fn: nil, stream: true}, _body) do
    Logger.warning("Streaming call requested but no callback function was given.")
  end

  defp fire_callback(%ChatOpenAI{callback_fn: nil}, _body), do: :ok

  defp fire_callback(%ChatOpenAI{callback_fn: callback_fn}, body) when is_function(callback_fn) do
    # OPTIONAL: Execute callback function
    body
    |> List.flatten()
    |> Enum.each(fn item -> callback_fn.(item) end)
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

  # Parse a new message response
  def do_process_response(%{"choices" => choices}) when is_list(choices) do
    # process each response individually. Return a list of all processed choices
    for choice <- choices do
      do_process_response(choice)
    end
  end

  def do_process_response(%{"finish_reason" => "stop", "message" => message, "index" => index}) do
    case Message.new(Map.merge(message, %{"complete" => true, "index" => index})) do
      {:ok, message} ->
        message

      {:error, changeset} ->
        {:error, Utils.changeset_error_to_string(changeset)}
    end
  end

  def do_process_response(
        %{
          "finish_reason" => "function_call",
          "message" => %{"function_call" => %{"arguments" => raw_args, "name" => name}}
        } = data
      ) do
    case Message.new(%{
           "role" => "function_call",
           "function_name" => name,
           "arguments" => raw_args,
           "complete" => true,
           "index" => data["index"]
         }) do
      {:ok, message} ->
        message

      {:error, changeset} ->
        {:error, Utils.changeset_error_to_string(changeset)}
    end
  end

  def do_process_response(
        %{"delta" => delta_body, "finish_reason" => finish, "index" => index} = _msg
      ) do
    complete =
      case finish do
        nil ->
          false

        "stop" ->
          true

        "function_call" ->
          true

        other ->
          Logger.warning("Unsupported finish_reason in delta message. Reason: #{inspect(other)}")
          false
      end

    function_name =
      case delta_body do
        %{"function_call" => %{"name" => name}} -> name
        _other -> nil
      end

    arguments =
      case delta_body do
        %{"function_call" => %{"arguments" => args}} when is_binary(args) -> args
        _other -> nil
      end

    # more explicitly interpret the role. We treat a "function_call" as a a role
    # while OpenAI addresses it as an "assistant". Technically, they are correct
    # that the assistant is issuing the function_call.
    role =
      case delta_body do
        %{"function_call" => _data} -> "function_call"
        %{"role" => role} -> role
        _other -> "unknown"
      end

    data =
      delta_body
      |> Map.put("role", role)
      |> Map.put("index", index)
      |> Map.put("complete", complete)
      |> Map.put("function_name", function_name)
      |> Map.put("arguments", arguments)

    case MessageDelta.new(data) do
      {:ok, message} ->
        message

      {:error, changeset} ->
        {:error, Utils.changeset_error_to_string(changeset)}
    end
  end

  def do_process_response(%{"choices" => [%{"finish_reason" => "length"}]}) do
    {:error, "Stopped for length"}
  end

  # TODO: "last_message reference? Does the "delta" include a message index?

  # NOTE: Full delta message:
  # - %{"delta" => %{"content" => " assist"}, "finish_reason" => nil, "index" => 0}

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
end
