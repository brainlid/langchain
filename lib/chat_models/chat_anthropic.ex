defmodule LangChain.ChatModels.ChatAnthropic do
  @moduledoc """
  Module for interacting with [Anthropic models](https://docs.anthropic.com/claude/docs/models-overview#claude-3-a-new-generation-of-ai).

  Parses and validates inputs for making requests to [Anthropic's messages API](https://docs.anthropic.com/claude/reference/messages_post).

  Converts responses into more specialized `LangChain` data structures.
  """
  use Ecto.Schema
  require Logger
  import Ecto.Changeset
  import LangChain.Utils.ApiOverride
  alias __MODULE__
  alias LangChain.Config
  alias LangChain.ChatModels.ChatModel
  alias LangChain.LangChainError
  alias LangChain.Message
  alias LangChain.MessageDelta
  alias LangChain.Utils

  @behaviour ChatModel

  # allow up to 1 minute for response.
  @receive_timeout 60_000

  @primary_key false
  embedded_schema do
    # API endpoint to use. Defaults to Anthropic's API
    field :endpoint, :string, default: "https://api.anthropic.com/v1/messages"

    # API key for Anthropic. If not set, will use global api key. Allows for usage
    # of a different API key per-call if desired. For instance, allowing a
    # customer to provide their own.
    field :api_key, :string

    # https://docs.anthropic.com/claude/reference/versions
    field :api_version, :string, default: "2023-06-01"

    # Duration in seconds for the response to be received. When streaming a very
    # lengthy response, a longer time limit may be required. However, when it
    # goes on too long by itself, it tends to hallucinate more.
    field :receive_timeout, :integer, default: @receive_timeout

    # field :model, :string, default: "claude-3-haiku-20240307"
    field :model, :string, default: "claude-3-haiku-20240307"

    # The maximum tokens allowed
    # This field is required to be present in the API request.
    # For now, all Claude models support max of 4096, which makes this default easy.
    field :max_tokens, :integer, default: 4096

    # Amount of randomness injected into the response. Ranges from 0.0 to 1.0. Defaults to 1.0.
    # Use temperature closer to 0.0 for analytical / multiple choice, and closer to 1.0 for
    # creative and generative tasks.
    field :temperature, :float, default: 1.0

    # Use nucleus sampling.
    # Recommended for advanced use cases only. You usually only need to use temperature.
    #
    # https://towardsdatascience.com/how-to-sample-from-language-models-682bceb97277
    #
    field :top_p, :float

    # Only sample from the top K options for each subsequent token.
    # Recommended for advanced use cases only. You usually only need to use temperature.
    #
    # https://towardsdatascience.com/how-to-sample-from-language-models-682bceb97277
    #
    field :top_k, :integer

    # Whether to stream the response
    field :stream, :boolean, default: false
  end

  @type t :: %ChatAnthropic{}

  @create_fields [
    :endpoint,
    :api_key,
    :api_version,
    :receive_timeout,
    :model,
    :max_tokens,
    :temperature,
    :top_p,
    :top_k,
    :stream
  ]
  @required_fields [:endpoint, :model]

  @spec get_api_key(t()) :: String.t()
  defp get_api_key(%ChatAnthropic{api_key: api_key}) do
    # if no API key is set default to `""` which will raise an error
    api_key || Config.resolve(:anthropic_key, "")
  end

  @doc """
  Setup a ChatAnthropic client configuration.
  """
  @spec new(attrs :: map()) :: {:ok, t} | {:error, Ecto.Changeset.t()}
  def new(%{} = attrs \\ %{}) do
    %ChatAnthropic{}
    |> cast(attrs, @create_fields)
    |> common_validation()
    |> apply_action(:insert)
  end

  @doc """
  Setup a ChatAnthropic client configuration and return it or raise an error if invalid.
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
    |> validate_number(:temperature, greater_than_or_equal_to: 0, less_than_or_equal_to: 1)
    |> validate_number(:receive_timeout, greater_than_or_equal_to: 0)
  end

  @doc """
  Return the params formatted for an API request.
  """
  @spec for_api(t, message :: [map()]) :: %{atom() => any()}
  def for_api(%ChatAnthropic{} = anthropic, messages) do
    %{
      model: anthropic.model,
      temperature: anthropic.temperature,
      stream: anthropic.stream,
      messages: messages_for_api(messages)
    }
    # Anthropic sets the `system` message on the request body, not as part of the messages list.
    |> Utils.conditionally_add_to_map(:system, get_system_message_content(messages))
    |> Utils.conditionally_add_to_map(:max_tokens, anthropic.max_tokens)
    |> Utils.conditionally_add_to_map(:top_p, anthropic.top_p)
    |> Utils.conditionally_add_to_map(:top_k, anthropic.top_k)
  end

  # Unlike OpenAI, Anthropic only supports one system message.
  defp get_system_message_content(messages) do
    case Enum.find(messages, &(&1.role == :system)) do
      nil ->
        nil

      %Message{content: content} ->
        content
    end
  end

  # Anthropic currently only supports two roles in their message list, "user" and "assistant".
  defp messages_for_api(messages) do
    messages
    |> Enum.filter(&(&1.role in [:user, :assistant]))
    |> Enum.map(&Map.take(&1, [:role, :content]))
  end

  @doc """
  Calls the Anthropic API passing the ChatAnthropic struct with configuration, plus
  either a simple message or the list of messages to act as the prompt.

  Optionally pass in a callback function that can be executed as data is
  received from the API.

  **NOTE:** This function *can* be used directly, but the primary interface
  should be through `LangChain.Chains.LLMChain`. The `ChatAnthropic` module is more focused on
  translating the `LangChain` data structures to and from the Anthropic API.

  Another benefit of using `LangChain.Chains.LLMChain` is that it combines the
  storage of messages, adding functions, adding custom context that should be
  passed to functions, and automatically applying `LangChain.MessageDelta`
  structs as they are are received, then converting those to the full
  `LangChain.Message` once fully complete.
  """
  @impl ChatModel
  def call(anthropic, prompt, functions \\ [], callback_fn \\ nil)

  def call(%ChatAnthropic{} = anthropic, prompt, functions, callback_fn) when is_binary(prompt) do
    messages = [
      Message.new_system!(),
      Message.new_user!(prompt)
    ]

    call(anthropic, messages, functions, callback_fn)
  end

  def call(%ChatAnthropic{} = anthropic, messages, functions, callback_fn)
      when is_list(messages) do
    if override_api_return?() do
      Logger.warning("Found override API response. Will not make live API call.")

      case get_api_override() do
        {:ok, {:ok, data} = response} ->
          # fire callback for fake responses too
          Utils.fire_callback(anthropic, data, callback_fn)
          response

        # fake error response
        {:ok, {:error, _reason} = response} ->
          response

        _other ->
          raise LangChainError,
                "An unexpected fake API response was set. Should be an `{:ok, value}`"
      end
    else
      try do
        # make base api request and perform high-level success/failure checks
        case do_api_request(anthropic, messages, functions, callback_fn) do
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

  # Call Anthropic's API.
  #
  # The result of the function is:
  #
  # - `result` - where `result` is a data-structure like a list or map.
  # - `{:error, reason}` - Where reason is a string explanation of what went wrong.
  #
  # If a callback_fn is provided, it will fire with each
  #
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
  def do_api_request(anthropic, messages, functions, callback_fn, retry_count \\ 3)

  def do_api_request(_anthropic, _messages, _functions, _callback_fn, 0) do
    raise LangChainError, "Retries exceeded. Connection failed."
  end

  def do_api_request(
        %ChatAnthropic{stream: false} = anthropic,
        messages,
        functions,
        callback_fn,
        retry_count
      ) do
    req =
      Req.new(
        url: anthropic.endpoint,
        json: for_api(anthropic, messages),
        headers: headers(get_api_key(anthropic), anthropic.api_version),
        receive_timeout: anthropic.receive_timeout,
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
            Utils.fire_callback(anthropic, result, callback_fn)
            result
        end

      {:error, %Mint.TransportError{reason: :timeout}} ->
        {:error, "Request timed out"}

      {:error, %Mint.TransportError{reason: :closed}} ->
        # Force a retry by making a recursive call decrementing the counter
        Logger.debug(fn -> "Mint connection closed: retry count = #{inspect(retry_count)}" end)
        do_api_request(anthropic, messages, functions, callback_fn, retry_count - 1)

      other ->
        Logger.error("Unexpected and unhandled API response! #{inspect(other)}")
        other
    end
  end

  def do_api_request(
        %ChatAnthropic{stream: true} = anthropic,
        messages,
        functions,
        callback_fn,
        retry_count
      ) do
    Req.new(
      url: anthropic.endpoint,
      json: for_api(anthropic, messages),
      headers: headers(get_api_key(anthropic), anthropic.api_version),
      receive_timeout: anthropic.receive_timeout
    )
    |> Req.post(
      into:
        Utils.handle_stream_fn(
          anthropic,
          &do_process_response/1,
          callback_fn,
          fn {chunk, buffer}, process_response_fn ->
            {parsed_events, incomplete} =
              ChatAnthropic.StreamingChunkDecoder.decode(chunk, buffer)

            parsed_events
            |> Enum.reduce({[], incomplete}, fn parsed_data, {processed_data, incomplete} = acc ->
              case process_response_fn.(parsed_data) do
                :ignore ->
                  acc

                result ->
                  # Anthropic's `messages` endpoint does not support generating N messages, only one.
                  # However, we wrap the result in an array to match the other providers who return a
                  # list of lists of deltas.
                  result = [result]
                  {processed_data ++ [result], incomplete}
              end
            end)
          end
        )
    )
    |> case do
      {:ok, %Req.Response{body: data}} ->
        data

      {:error, %LangChainError{message: reason}} ->
        {:error, reason}

      {:error, %Mint.TransportError{reason: :timeout}} ->
        {:error, "Request timed out"}

      {:error, %Mint.TransportError{reason: :closed}} ->
        # Force a retry by making a recursive call decrementing the counter
        Logger.debug(fn -> "Mint connection closed: retry count = #{inspect(retry_count)}" end)
        do_api_request(anthropic, messages, functions, callback_fn, retry_count - 1)

      other ->
        Logger.error(
          "Unhandled and unexpected response from streamed post call. #{inspect(other)}"
        )

        {:error, "Unexpected response"}
    end
  end

  defp headers(api_key, api_version) do
    %{
      "x-api-key" => api_key,
      "content-type" => "application/json",
      "anthropic-version" => api_version
    }
  end

  # Parse a new message response
  @doc false
  @spec do_process_response(data :: %{String.t() => any()} | {:error, any()}) ::
          Message.t()
          | [Message.t()]
          | MessageDelta.t()
          | [MessageDelta.t()]
          | {:error, String.t()}
  def do_process_response(%{
        "type" => "message",
        "content" => [%{"type" => "text", "text" => content}],
        "stop_reason" => stop_reason
      }) do
    %{
      role: :assistant,
      content: content,
      status: stop_reason_to_status(stop_reason)
    }
    |> Message.new()
    |> to_response()
  end

  def do_process_response(%{
        "type" => "content_block_start",
        "content_block" => %{"type" => "text", "text" => content}
      }) do
    %{
      role: :assistant,
      content: content,
      status: :incomplete
    }
    |> MessageDelta.new()
    |> to_response()
  end

  def do_process_response(%{
        "type" => "content_block_delta",
        "delta" => %{"type" => "text_delta", "text" => content}
      }) do
    %{
      role: :assistant,
      content: content,
      status: :incomplete
    }
    |> MessageDelta.new()
    |> to_response()
  end

  def do_process_response(%{
        "type" => "message_delta",
        "delta" => %{"stop_reason" => stop_reason}
      }) do
    %{
      role: :assistant,
      content: "",
      status: stop_reason_to_status(stop_reason)
    }
    |> MessageDelta.new()
    |> to_response()
  end

  # These streaming events don't contain any useful data to return to callers. Ignore.
  def do_process_response(%{"type" => type})
      when type in ["message_start", "ping", "content_block_stop", "message_stop"] do
    :ignore
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

  defp to_response({:ok, message}), do: message
  defp to_response({:error, changeset}), do: {:error, Utils.changeset_error_to_string(changeset)}

  defp stop_reason_to_status("end_turn"), do: :complete
  defp stop_reason_to_status("max_tokens"), do: :length
  defp stop_reason_to_status("stop_sequence"), do: :complete

  defp stop_reason_to_status(other) do
    Logger.warning("Unsupported stop_reason. Reason: #{inspect(other)}")
    nil
  end

  defmodule StreamingChunkDecoder do
    @moduledoc false

    # Data comes back from Anthropic in event/data pairs. See the following example:
    #
    #     event: message_start
    #     data: {"type": "message_start", "message": {"id": "msg_1nZdL29xx5MUA1yADyHTEsnR8uuvGzszyY", "type": "message", "role": "assistant", "content": [], "model": "claude-3-opus-20240229", "stop_reason": null, "stop_sequence": null, "usage": {"input_tokens": 25, "output_tokens": 1}}}
    #
    #     event: content_block_start
    #     data: {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}}
    #
    #     event: ping
    #     data: {"type": "ping"}
    #
    #     event: content_block_delta
    #     data: {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "Hello"}}
    #
    #     event: content_block_delta
    #     data: {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "!"}}
    #
    #     event: content_block_stop
    #     data: {"type": "content_block_stop", "index": 0}
    #
    #     event: message_delta
    #     data: {"type": "message_delta", "delta": {"stop_reason": "end_turn", "stop_sequence":null, "usage":{"output_tokens": 15}}}
    #
    #     event: message_stop
    #     data: {"type": "message_stop"}
    #
    #
    # See Anthropic documentation: https://docs.anthropic.com/claude/reference/messages-streaming

    # This captures `event` and `data`
    @event_data_re ~r/^\s*event:\s*(?<event>\S*)\s*data:\s*(?<data>[^\n]+)\s*$/

    @doc false
    def decode(chunk, buffer) do
      ((buffer || "") <> chunk)
      # Split using the event separator to try to group event/data blocks together.
      |> String.split("\n\n")
      |> Enum.reduce({[], ""}, fn raw_event, {parsed_data_list, incomplete} ->
        case do_decode(raw_event) do
          :ignore ->
            {parsed_data_list, incomplete}

          {:complete, parsed_data} ->
            {parsed_data_list ++ [parsed_data], incomplete}

          # If we assume Anthropic is not sending corrupt events, then this can only
          # ever happen with the last item in the list resulting from the "\n\n" split.
          {:incomplete, remaining} ->
            {parsed_data_list, remaining}
        end
      end)
    end

    # When splitting a chunk, we end up with empty strings. Ignore those.
    defp do_decode("") do
      :ignore
    end

    # This handles the case where `raw_event` looks like a "complete" event, e.g.:
    #
    #     event: content_block_start\ndata: {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}}
    #
    # The above example contains the entire `event` and `data` values. We can pluck the `event` and `data`
    # values out of this chunk using a regular expression. If the regular expression matches and we're able
    # to JSON decode the `data`, then we have successfully parsed the event.
    #
    # However, this function can potentially be passed "incomplete" events. One example would be the following:
    #
    #     event: content_block_delta\ndata: {"type": "content_block_delta", "inde
    #
    # The above will match the regular expression, but will not successfully JSON decode.
    #
    defp do_decode(raw_event) do
      case Regex.named_captures(@event_data_re, raw_event) do
        %{"event" => _event, "data" => data} ->
          case Jason.decode(data) do
            {:ok, parsed_data} ->
              {:complete, parsed_data}

            {:error, _reason} ->
              {:incomplete, raw_event}
          end

        _ ->
          {:incomplete, raw_event}
      end
    end
  end
end
