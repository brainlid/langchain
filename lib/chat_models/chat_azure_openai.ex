defmodule LangChain.ChatModels.ChatAzureOpenAI do
  @moduledoc """
  Represents the [Azure OpenAI ChatModel](https://learn.microsoft.com/en-us/azure/ai-services/openai/).

  Parses and validates inputs for making a requests from the Azure OpenAI Chat API.

  Converts responses into more specialized `LangChain` data structures.

  - https://github.com/openai/openai-cookbook/blob/main/examples/How_to_call_functions_with_chat_models.ipynb

  """
  use Ecto.Schema
  require Logger
  import Ecto.Changeset
  import LangChain.Utils.ApiOverride
  alias __MODULE__
  alias LangChain.Config
  alias LangChain.Message
  alias LangChain.LangChainError
  alias LangChain.ForOpenAIApi
  alias LangChain.Utils
  alias LangChain.MessageDelta

  # NOTE: As of gpt-4 and gpt-3.5, only one function_call is issued at a time
  # even when multiple requests could be issued based on the prompt.

  # allow up to 2 minutes for response.
  @receive_timeout 60_000

  @primary_key false
  embedded_schema do
    # endpoint is formatted as follows
    # https://#{your-resource-name}.openai.azure.com/openai/deployments/#{model-name}/chat/completions?api-version=2023-08-01-preview
    field :endpoint, :string
    # API key for OpenAI. If not set, will use global api key. Allows for usage
    # of a different API key per-call if desired. For instance, allowing a
    # customer to provide their own.
    field :api_key, :string
    # What sampling temperature to use, between 0 and 2. Higher values like 0.8
    # will make the output more random, while lower values like 0.2 will make it
    # more focused and deterministic.
    field :temperature, :float, default: 1.0
    # Number between -2.0 and 2.0. Positive values penalize new tokens based on
    # their existing frequency in the text so far, decreasing the model's
    # likelihood to repeat the same line verbatim.
    field :frequency_penalty, :float, default: 0.0
    # Duration in seconds for the response to be received. When streaming a very
    # lengthy response, a longer time limit may be required. However, when it
    # goes on too long by itself, it tends to hallucinate more.
    field :receive_timeout, :integer, default: @receive_timeout
    # How many chat completion choices to generate for each input message.
    field :n, :integer, default: 1
    field :stream, :boolean, default: false
  end

  @type t :: %ChatAzureOpenAI{}

  @type call_response :: {:ok, Message.t() | [Message.t()]} | {:error, String.t()}
  @type callback_data ::
          {:ok, Message.t() | MessageDelta.t() | [Message.t() | MessageDelta.t()]}
          | {:error, String.t()}

  @create_fields [
    :endpoint,
    :temperature,
    :frequency_penalty,
    :api_key,
    :n,
    :stream,
    :receive_timeout
  ]
  @required_fields [:endpoint]

  @spec get_api_key(t) :: String.t()
  defp get_api_key(%ChatAzureOpenAI{api_key: api_key}) do
    # if no API key is set default to `""` which will raise an API error
    api_key || Config.resolve(:azure_openai_key, "")
  end

  @doc """
  Setup a ChatAzureOpenAI client configuration.
  """
  @spec new(attrs :: map()) :: {:ok, t} | {:error, Ecto.Changeset.t()}
  def new(%{} = attrs \\ %{}) do
    %ChatAzureOpenAI{}
    |> cast(attrs, @create_fields)
    |> common_validation()
    |> apply_action(:insert)
  end

  @doc """
  Setup a ChatAzureOpenAI client configuration and return it or raise an error if invalid.
  """
  @spec new!(attrs :: map()) :: t() | no_return()
  def new!(attrs \\ %{}) do
    case new(attrs) do
      {:ok, chain} ->
        chain

      {:error, changeset} ->
        raise LangChainError, changeset
    end
  end

  defp common_validation(changeset) do
    changeset
    |> validate_required(@required_fields)
    |> validate_number(:temperature, greater_than_or_equal_to: 0, less_than_or_equal_to: 2)
    |> validate_number(:frequency_penalty, greater_than_or_equal_to: -2, less_than_or_equal_to: 2)
    |> validate_number(:n, greater_than_or_equal_to: 1)
    |> validate_number(:receive_timeout, greater_than_or_equal_to: 0)
  end

  @doc """
  Return the params formatted for an API request.
  """
  @spec for_api(t, message :: [map()], functions :: [map()]) :: %{atom() => any()}
  def for_api(%ChatAzureOpenAI{} = openai, messages, functions) do
    %{
      temperature: openai.temperature,
      frequency_penalty: openai.frequency_penalty,
      n: openai.n,
      stream: openai.stream,
      messages: Enum.map(messages, &ForOpenAIApi.for_api/1),
      function_call: "auto"
    }
    |> Utils.conditionally_add_to_map(:functions, get_functions_for_api(functions))
  end

  defp get_functions_for_api(nil), do: []

  defp get_functions_for_api(functions) do
    Enum.map(functions, &ForOpenAIApi.for_api/1)
  end

  @doc """
  Calls the OpenAI API passing the ChatAzureOpenAI struct with configuration, plus
  either a simple message or the list of messages to act as the prompt.

  Optionally pass in a list of functions available to the LLM for requesting
  execution in response.

  Optionally pass in a callback function that can be executed as data is
  received from the API.

  **NOTE:** This function *can* be used directly, but the primary interface
  should be through `LangChain.Chains.LLMChain`. The `ChatAzureOpenAI` module is more focused on
  translating the `LangChain` data structures to and from the OpenAI API.

  Another benefit of using `LangChain.Chains.LLMChain` is that it combines the
  storage of messages, adding functions, adding custom context that should be
  passed to functions, and automatically applying `LangChain.MessageDelta`
  structs as they are are received, then converting those to the full
  `LangChain.Message` once fully complete.
  """
  @spec call(
          t(),
          String.t() | [Message.t()],
          [LangChain.Function.t()],
          nil | (Message.t() | MessageDelta.t() -> any())
        ) :: call_response()
  def call(openai, prompt, functions \\ [], callback_fn \\ nil)

  def call(%ChatAzureOpenAI{} = openai, prompt, functions, callback_fn) when is_binary(prompt) do
    messages = [
      Message.new_system!(),
      Message.new_user!(prompt)
    ]

    call(openai, messages, functions, callback_fn)
  end

  def call(%ChatAzureOpenAI{} = openai, messages, functions, callback_fn) when is_list(messages) do
    if override_api_return?() do
      Logger.warning("Found override API response. Will not make live API call.")

      case get_api_override() do
        {:ok, {:ok, data} = response} ->
          # fire callback for fake responses too
          fire_callback(openai, data, callback_fn)
          response

        _other ->
          raise LangChainError,
                "An unexpected fake API response was set. Should be an `{:ok, value}`"
      end
    else
      try do
        # make base api request and perform high-level success/failure checks
        case do_api_request(openai, messages, functions, callback_fn) do
          {:error, reason} ->
            {:error, reason}

          parsed_data ->
            {:ok, parsed_data}
        end
      rescue
        err in LangChainError ->
          {:error, err.message}
      end
    end
  end

  # Make the API request from the Azure OpenAI server.
  #
  # The result of the function is:
  #
  # - `result` - where `result` is a data-structure like a list or map.
  # - `{:error, reason}` - Where reason is a string explanation of what went wrong.
  #
  # If a callback_fn is provided, it will fire with each

  # When `stream: true` is
  # If `stream: false`, the completed message is returned.
  #
  # If `stream: true`, the `callback_fn` is executed for the returned MessageDelta
  # responses.
  #
  # Executes the callback function passing the response only parsed to the data
  # structures.
  # Retries the request up to 3 times on transient errors with a 1 second delay
  @doc false
  @spec do_api_request(t(), [Message.t()], [Function.t()], (any() -> any())) ::
          list() | struct() | {:error, String.t()}
  def do_api_request(%ChatAzureOpenAI{stream: false} = openai, messages, functions, callback_fn) do

    req =
      Req.new(
        url: openai.endpoint,
        json: for_api(openai, messages, functions),
        headers: [{:api_key, get_api_key(openai)}],
        receive_timeout: openai.receive_timeout,
        retry: :transient,
        max_retries: 3,
        retry_delay: fn attempt -> 300 * attempt end
      )

    req
    |> Req.post()
    # parse the body and return it as parsed structs
    |> case do
      {:ok, %Req.Response{body: data}} ->
        case do_process_response(data) do
          {:error, reason} ->
            {:error, reason}

          result ->
            fire_callback(openai, result, callback_fn)
            result
        end

      {:error, %Mint.TransportError{reason: :timeout}} ->
        {:error, "Request timed out"}

      other ->
        Logger.error("Unexpected and unhandled API response! #{inspect(other)}")
        other
    end
  end

  def do_api_request(%ChatAzureOpenAI{stream: true} = openai, messages, functions, callback_fn) do
    finch_fun = fn request, finch_request, finch_name, finch_options ->
      resp_fun = fn
        {:status, status}, response ->
          %{response | status: status}

        {:headers, headers}, response ->
          %{response | headers: headers}

        {:data, raw_data}, response ->
          # JSON strings can be split over multiple responses
          # we need to buffer the response until we have valid
          # JSON strings to decode. JSON strings are delimited by two new lines
          old_buffer = Req.Response.get_private(response, :buffer, "")
          {items, new_buffer} = next_lines(old_buffer <> raw_data)
          response = Req.Response.put_private(response, :buffer, new_buffer)

          # we need to ensure that errors do get passed through for handling
          # by decode_streamed_data
          # this will happen when the is nothing but an error in the body
          # this results in an empty items list and the error in the new_buffer
          items =
            case response.status do
              400 -> [new_buffer]
              _ -> items
            end

          new_datas =
            items
            |> Enum.flat_map(fn raw_item ->
              # cleanup data because it isn't structured well for JSON.
              new_data = decode_streamed_data(raw_item)

              Logger.debug("API data item:- #{inspect(new_data)}")

              # execute the callback function for each MessageDelta
              new_data
              |> Enum.each(fn data_item ->
                fire_callback(openai, data_item, callback_fn)
              end)

              new_data
            end)

          old_body = if response.body == "", do: [], else: response.body

          # Returns %Req.Response{} where the body contains ALL the stream delta
          # chunks converted to MessageDelta structs. The body is a list of lists like this...
          #
          # body: [
          #         [
          #           %LangChain.MessageDelta{
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
          case new_datas do
            [[]] ->
              %{response | body: old_body}

            _ ->
              %{response | body: old_body ++ new_datas}
          end
      end

      case Finch.stream(finch_request, finch_name, Req.Response.new(), resp_fun, finch_options) do
        {:ok, response} ->
          {request, response}

        {:error, %Mint.TransportError{reason: :timeout}} ->
          {request, LangChainError.exception("Request timed out")}

        {:error, exception} ->
          Logger.error("Failed request to API: #{inspect(exception)}")
          {request, exception}
      end
    end

    req =
      Req.new(
        url: openai.endpoint,
        json: for_api(openai, messages, functions),
        headers: [{:api_key, get_api_key(openai)}],
        receive_timeout: openai.receive_timeout,
        retry: :transient,
        max_retries: 3,
        retry_delay: fn attempt -> 300 * attempt end,
        finch_request: finch_fun
      )

    # NOTE: The POST response includes a list of body messages that were
    # received during the streaming process. However, the messages in the
    # response all come at once when the stream is complete. It is blocking
    # until it completes. This means the streaming call should happen in a
    # separate process from the UI and the callback function will process the
    # chunks and should notify the UI process of the additional data.
    req
    |> Req.post()
    |> case do
      {:ok, %Req.Response{body: data}} ->
        data

      {:error, %LangChainError{message: reason}} ->
        {:error, reason}

      other ->
        Logger.error(
          "Unhandled and unexpected response from streamed post call. #{inspect(other)}"
        )

        {:error, "Unexpected response. #{inspect(other)}"}
    end
  end

  # JSON strings are delimited by two new lines
  # work through the string and break out JSON strings
  # next_lines implementation is shamelessly inspired/plucked gratefully from
  # https://github.com/poeticoding/httpstream_articles/blob/36bc2167b7024a990b04b28f9447fb9bc0e0310e/lib/http_stream.ex#L78
  defp next_lines(chunk, prev \\ ""), do: next_lines(chunk, prev, [])

  defp next_lines(<<"\n\n"::utf8, rest::binary>>, prev, lines) do
    next_lines(rest, "", [prev <> "\n\n" | lines])
  end

  defp next_lines(<<c::utf8, rest::binary>>, prev, lines) do
    next_lines(rest, <<prev::binary, c::utf8>>, lines)
  end

  defp next_lines(<<>>, prev, lines), do: {Enum.reverse(lines), prev}

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
          |> Jason.decode()
          |> case do
            {:ok, parsed} ->
              parsed

            {:error, reason} ->
              {:error, reason}
          end
          |> do_process_response()
      end
    end)
    # returning a list of elements. "junk" elements were replaced with `:empty`.
    # Filter those out down and return the final list of MessageDelta structs.
    |> Enum.filter(fn d -> d != :empty end)
    # if there was a single error returned in a list, flatten it out to just
    # return the error
    |> case do
      [{:error, reason}] ->
        raise LangChainError, reason

      other ->
        other
    end
  end

  # fire the callback if present.
  @spec fire_callback(
          t(),
          data :: callback_data() | [callback_data()],
          (callback_data() -> any())
        ) :: :ok
  defp fire_callback(%ChatAzureOpenAI{stream: true}, _data, nil) do
    Logger.warning("Streaming call requested but no callback function was given.")
    :ok
  end

  defp fire_callback(%ChatAzureOpenAI{}, _data, nil), do: :ok

  defp fire_callback(%ChatAzureOpenAI{}, data, callback_fn) when is_function(callback_fn) do
    # OPTIONAL: Execute callback function
    data
    |> List.flatten()
    |> Enum.each(fn item -> callback_fn.(item) end)

    :ok
  end

  # Parse a new message response
  @doc false
  @spec do_process_response(data :: %{String.t() => any()} | {:error, any()}) ::
          Message.t()
          | [Message.t()]
          | MessageDelta.t()
          | [MessageDelta.t()]
          | {:error, String.t()}
  def do_process_response(%{"choices" => choices}) when is_list(choices) do
    # process each response individually. Return a list of all processed choices
    for choice <- choices do
      do_process_response(choice)
    end
  end

  def do_process_response(
        %{
          "finish_reason" => "function_call",
          "message" => %{"function_call" => %{"arguments" => raw_args, "name" => name}}
        } = data
      ) do
    case Message.new(%{
           "role" => "assistant",
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
    status =
      case finish do
        nil ->
          :incomplete

        "stop" ->
          :complete

        "length" ->
          :length

        "function_call" ->
          :complete

        other ->
          Logger.warning("Unsupported finish_reason in delta message. Reason: #{inspect(other)}")
          nil
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
        %{"role" => role} -> role
        _other -> "unknown"
      end

    data =
      delta_body
      |> Map.put("role", role)
      |> Map.put("index", index)
      |> Map.put("status", status)
      |> Map.put("function_name", function_name)
      |> Map.put("arguments", arguments)

    case MessageDelta.new(data) do
      {:ok, message} ->
        message

      {:error, changeset} ->
        {:error, Utils.changeset_error_to_string(changeset)}
    end
  end

  def do_process_response(%{
        "finish_reason" => finish_reason,
        "message" => message,
        "index" => index
      }) do
    status =
      case finish_reason do
        "stop" ->
          :complete

        "length" ->
          :length

        other ->
          Logger.warning("Unsupported finish_reason in message. Reason: #{inspect(other)}")
          nil
      end

    case Message.new(Map.merge(message, %{"status" => status, "index" => index})) do
      {:ok, message} ->
        message

      {:error, changeset} ->
        {:error, Utils.changeset_error_to_string(changeset)}
    end
  end

  def do_process_response(%{"error" => %{"message" => reason}}) do
    Logger.error("Received error from API: #{inspect(reason)}")
    {:error, reason}
  end

  def do_process_response({:error, %Jason.DecodeError{} = response}) do
    error_message = "Received invalid JSON: #{inspect(response)}"
    Logger.error(error_message)
    {:error, error_message}
  end

  def do_process_response(other) do
    Logger.error("Trying to process an unexpected response. #{inspect(other)}")
    {:error, "Unexpected response"}
  end
end
