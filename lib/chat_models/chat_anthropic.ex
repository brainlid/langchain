defmodule LangChain.ChatModels.ChatAnthropic do
  @moduledoc """
  Module for interacting with [Anthropic models](https://docs.anthropic.com/claude/docs/models-overview#claude-3-a-new-generation-of-ai).

  Parses and validates inputs for making requests to [Anthropic's messages API](https://docs.anthropic.com/claude/reference/messages_post).

  Converts responses into more specialized `LangChain` data structures.

  ## Callbacks

  See the set of available callback: `LangChain.ChatModels.LLMCallbacks`

  ### Rate Limit API Response Headers

  Anthropic returns rate limit information in the response headers. Those can be
  accessed using an LLM callback like this:

      handlers = %{
        on_llm_ratelimit_info: fn _model, headers ->
          IO.inspect(headers)
        end
      }

      {:ok, chat} = ChatAnthropic.new(%{callbacks: [handlers]})

  When a request is received, something similar to the following will be output
  to the console.

      %{
        "anthropic-ratelimit-requests-limit" => ["50"],
        "anthropic-ratelimit-requests-remaining" => ["49"],
        "anthropic-ratelimit-requests-reset" => ["2024-06-08T04:28:30Z"],
        "anthropic-ratelimit-tokens-limit" => ["50000"],
        "anthropic-ratelimit-tokens-remaining" => ["50000"],
        "anthropic-ratelimit-tokens-reset" => ["2024-06-08T04:28:30Z"],
        "request-id" => ["req_1234"]
      }

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
  alias LangChain.Message.ContentPart
  alias LangChain.Message.ToolCall
  alias LangChain.Message.ToolResult
  alias LangChain.MessageDelta
  alias LangChain.TokenUsage
  alias LangChain.Function
  alias LangChain.FunctionParam
  alias LangChain.Utils
  alias LangChain.Callbacks

  @behaviour ChatModel

  @current_config_version 1

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

    # A list of maps for callback handlers
    field :callbacks, {:array, :map}, default: []
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
    :stream,
    :callbacks
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
  @spec for_api(t, message :: [map()], ChatModel.tools()) :: %{atom() => any()}
  def for_api(%ChatAnthropic{} = anthropic, messages, tools) do
    # separate the system message from the rest. Handled separately.
    {system, messages} = split_system_message(messages)

    system_text =
      case system do
        nil ->
          nil

        %Message{role: :system, content: content} ->
          content
      end

    messages =
      messages
      |> Enum.map(&for_api/1)
      |> post_process_and_combine_messages()

    %{
      model: anthropic.model,
      temperature: anthropic.temperature,
      stream: anthropic.stream,
      messages: messages
    }
    # Anthropic sets the `system` message on the request body, not as part of the messages list.
    |> Utils.conditionally_add_to_map(:system, system_text)
    |> Utils.conditionally_add_to_map(:tools, get_tools_for_api(tools))
    |> Utils.conditionally_add_to_map(:max_tokens, anthropic.max_tokens)
    |> Utils.conditionally_add_to_map(:top_p, anthropic.top_p)
    |> Utils.conditionally_add_to_map(:top_k, anthropic.top_k)
  end

  defp get_tools_for_api(nil), do: []

  defp get_tools_for_api(tools) do
    Enum.map(tools, fn
      %Function{} = function ->
        for_api(function)
    end)
  end

  # Unlike OpenAI, Anthropic only supports one system message.
  @doc false
  @spec split_system_message([Message.t()]) :: {nil | Message.t(), [Message.t()]} | no_return()
  def split_system_message(messages) do
    # split the messages into "system" and "other". Error if more than 1 system
    # message. Return the other messages as a separate list.
    {system, other} = Enum.split_with(messages, &(&1.role == :system))

    if length(system) > 1 do
      raise LangChainError, "Anthropic only supports a single System message"
    end

    {List.first(system), other}
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
  def call(anthropic, prompt, functions \\ [])

  def call(%ChatAnthropic{} = anthropic, prompt, functions) when is_binary(prompt) do
    messages = [
      Message.new_system!(),
      Message.new_user!(prompt)
    ]

    call(anthropic, messages, functions)
  end

  def call(%ChatAnthropic{} = anthropic, messages, functions)
      when is_list(messages) do
    if override_api_return?() do
      Logger.warning("Found override API response. Will not make live API call.")

      case get_api_override() do
        {:ok, {:ok, data, callback_name}} ->
          # fire callback for fake responses too
          Callbacks.fire(anthropic.callbacks, callback_name, [anthropic, data])
          # return the data portion
          {:ok, data}

        # fake error response
        {:ok, {:error, _reason} = response} ->
          response

        _other ->
          raise LangChainError,
                "An unexpected fake API response was set. Should be an `{:ok, value, nil_or_callback_name}`"
      end
    else
      try do
        # make base api request and perform high-level success/failure checks
        case do_api_request(anthropic, messages, functions) do
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
  # If `stream: false`, the completed message is returned.
  #
  # Retries the request up to 3 times on transient errors with a 1 second delay
  @doc false
  @spec do_api_request(t(), [Message.t()], ChatModel.tools(), (any() -> any())) ::
          list() | struct() | {:error, String.t()}
  def do_api_request(anthropic, messages, tools, retry_count \\ 3)

  def do_api_request(_anthropic, _messages, _functions, 0) do
    raise LangChainError, "Retries exceeded. Connection failed."
  end

  def do_api_request(
        %ChatAnthropic{stream: false} = anthropic,
        messages,
        tools,
        retry_count
      ) do
    req =
      Req.new(
        url: anthropic.endpoint,
        json: for_api(anthropic, messages, tools),
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
      {:ok, %Req.Response{body: data} = response} ->
        Callbacks.fire(anthropic.callbacks, :on_llm_ratelimit_info, [
          anthropic,
          get_ratelimit_info(response.headers)
        ])

        Callbacks.fire(anthropic.callbacks, :on_llm_token_usage, [
          anthropic,
          get_token_usage(data)
        ])

        case do_process_response(anthropic, data) do
          {:error, reason} ->
            {:error, reason}

          result ->
            Callbacks.fire(anthropic.callbacks, :on_llm_new_message, [anthropic, result])
            result
        end

      {:error, %Req.TransportError{reason: :timeout}} ->
        {:error, "Request timed out"}

      {:error, %Req.TransportError{reason: :closed}} ->
        # Force a retry by making a recursive call decrementing the counter
        Logger.debug(fn -> "Mint connection closed: retry count = #{inspect(retry_count)}" end)
        do_api_request(anthropic, messages, tools, retry_count - 1)

      other ->
        Logger.error("Unexpected and unhandled API response! #{inspect(other)}")
        other
    end
  end

  def do_api_request(
        %ChatAnthropic{stream: true} = anthropic,
        messages,
        tools,
        retry_count
      ) do
    Req.new(
      url: anthropic.endpoint,
      json: for_api(anthropic, messages, tools),
      headers: headers(get_api_key(anthropic), anthropic.api_version),
      receive_timeout: anthropic.receive_timeout
    )
    |> Req.post(
      into:
        Utils.handle_stream_fn(anthropic, &decode_stream/1, &do_process_response(anthropic, &1))
    )
    |> case do
      {:ok, %Req.Response{body: data} = response} ->
        Callbacks.fire(anthropic.callbacks, :on_llm_ratelimit_info, [
          anthropic,
          get_ratelimit_info(response.headers)
        ])

        data

      {:error, %LangChainError{message: reason}} ->
        {:error, reason}

      {:error, %Req.TransportError{reason: :timeout}} ->
        {:error, "Request timed out"}

      {:error, %Req.TransportError{reason: :closed}} ->
        # Force a retry by making a recursive call decrementing the counter
        Logger.debug(fn -> "Mint connection closed: retry count = #{inspect(retry_count)}" end)
        do_api_request(anthropic, messages, tools, retry_count - 1)

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
      "anthropic-version" => api_version,
      # https://docs.anthropic.com/claude/docs/tool-use - requires this header during beta
      "anthropic-beta" => "tools-2024-04-04"
    }
  end

  # Parse a new message response
  @doc false
  @spec do_process_response(t(), data :: %{String.t() => any()} | {:error, any()}) ::
          Message.t()
          | [Message.t()]
          | MessageDelta.t()
          | [MessageDelta.t()]
          | {:error, String.t()}
  def do_process_response(_model, %{
        "role" => "assistant",
        "content" => contents,
        "stop_reason" => stop_reason
      }) do
    new_message =
      %{
        role: :assistant,
        status: stop_reason_to_status(stop_reason)
      }
      |> Message.new()
      |> to_response()

    # reduce over the contents and accumulate to the message
    Enum.reduce(contents, new_message, fn content, acc ->
      do_process_content_response(acc, content)
    end)
  end

  def do_process_response(_model, %{
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

  def do_process_response(_model, %{
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

  def do_process_response(
        model,
        %{
          "type" => "message_delta",
          "delta" => %{"stop_reason" => stop_reason},
          "usage" => _usage
        } = data
      ) do
    # if we received usage data, fire any callbacks for it.
    Callbacks.fire(model.callbacks, :on_llm_token_usage, [model, get_token_usage(data)])

    %{
      role: :assistant,
      content: "",
      status: stop_reason_to_status(stop_reason)
    }
    |> MessageDelta.new()
    |> to_response()
  end

  def do_process_response(_model, %{"error" => %{"message" => reason}}) do
    Logger.error("Received error from API: #{inspect(reason)}")
    {:error, reason}
  end

  def do_process_response(_model, {:error, %Jason.DecodeError{} = response}) do
    error_message = "Received invalid JSON: #{inspect(response)}"
    Logger.error(error_message)
    {:error, error_message}
  end

  def do_process_response(_model, other) do
    Logger.error("Trying to process an unexpected response. #{inspect(other)}")
    {:error, "Unexpected response"}
  end

  # for parsing a list of received content JSON objects
  defp do_process_content_response(%Message{} = message, %{"type" => "text", "text" => ""}),
    do: message

  defp do_process_content_response(%Message{} = message, %{"type" => "text", "text" => text}) do
    %Message{message | content: text}
  end

  defp do_process_content_response(
         %Message{} = message,
         %{"type" => "tool_use", "id" => call_id, "name" => name} = call
       ) do
    arguments =
      case call["input"] do
        # when properties is an empty string, treat it as nil
        %{"properties" => ""} ->
          nil

        # when properties is an empty map, treat it as nil
        %{"properties" => %{} = props} when props == %{} ->
          nil

        # when an empty map, return nil
        %{} = data when data == %{} ->
          nil

        # when a map with data
        %{} = data ->
          data
      end

    %Message{
      message
      | tool_calls:
          message.tool_calls ++
            [
              ToolCall.new!(%{
                type: :function,
                call_id: call_id,
                name: name,
                arguments: arguments,
                status: :complete
              })
            ]
    }
  end

  defp do_process_content_response({:error, _reason} = error, _content) do
    error
  end

  defp to_response({:ok, message}), do: message
  defp to_response({:error, changeset}), do: {:error, Utils.changeset_error_to_string(changeset)}

  defp stop_reason_to_status("end_turn"), do: :complete
  defp stop_reason_to_status("tool_use"), do: :complete
  defp stop_reason_to_status("max_tokens"), do: :length
  defp stop_reason_to_status("stop_sequence"), do: :complete

  defp stop_reason_to_status(other) do
    Logger.warning("Unsupported stop_reason. Reason: #{inspect(other)}")
    nil
  end

  @doc false
  def decode_stream({chunk, buffer}) do
    # Combine the incoming data with the buffered incomplete data
    combined_data = buffer <> chunk
    # Split data by double newline to find complete messages
    entries = String.split(combined_data, "\n\n", trim: true)

    # The last part may be incomplete if it doesn't end with "\n\n"
    {to_process, incomplete} =
      if String.ends_with?(combined_data, "\n\n") do
        {entries, ""}
      else
        # process all but the last, keep the last as incomplete
        {Enum.slice(entries, 0..-2//1), List.last(entries)}
      end

    processed =
      to_process
      # Trim whitespace from each line
      |> Stream.map(&String.trim/1)
      # Ignore empty lines
      |> Stream.reject(&(&1 == ""))
      # Filter lines based on some condition
      |> Stream.filter(&relevant_event?/1)
      # Split the event from the data into separate lines
      |> Stream.map(&extract_data(&1))
      |> Enum.reduce([], fn json, done ->
        json
        |> Jason.decode()
        |> case do
          {:ok, parsed} ->
            # wrap each parsed response into an array of 1. This matches the
            # return type of some LLMs where they return `n` number of responses.
            # This is for compatibility.
            # {done ++ Enum.map(parsed, &([&1])), ""}
            done ++ [parsed]

          {:error, reason} ->
            Logger.error("Failed to JSON decode streamed data: #{inspect(reason)}")
            done
        end
      end)

    {processed, incomplete}
  end

  defp relevant_event?("event: content_block_delta\n" <> _rest), do: true
  defp relevant_event?("event: content_block_start\n" <> _rest), do: true
  defp relevant_event?("event: message_delta\n" <> _rest), do: true
  # ignoring
  defp relevant_event?("event: message_start\n" <> _rest), do: false
  defp relevant_event?("event: ping\n" <> _rest), do: false
  defp relevant_event?("event: content_block_stop\n" <> _rest), do: false
  defp relevant_event?("event: message_stop\n" <> _rest), do: false
  # catch-all for when we miss something
  defp relevant_event?(event) do
    Logger.error("Unsupported event received when parsing Anthropic response: #{inspect(event)}")
    false
  end

  # process data for an event
  defp extract_data("event: " <> line) do
    [_prefix, json] = String.split(line, "data: ", trim: true)
    json
  end

  # assumed the response is JSON. Return as-is
  defp extract_data(json), do: json

  @doc """
  Convert a LangChain structure to the expected map of data for the OpenAI API.
  """
  @spec for_api(Message.t() | ContentPart.t() | Function.t()) ::
          %{String.t() => any()} | no_return()
  def for_api(%Message{role: :assistant, tool_calls: calls} = msg)
      when is_list(calls) and calls != [] do
    text_content =
      if is_binary(msg.content) do
        [
          %{
            "type" => "text",
            "text" => msg.content
          }
        ]
      else
        []
      end

    tool_calls = Enum.map(calls, &for_api(&1))

    %{
      "role" => "assistant",
      "content" => text_content ++ tool_calls
    }
  end

  def for_api(%Message{role: :tool, tool_results: results}) when is_list(results) do
    # convert ToolResult into the expected format for Anthropic.
    #
    # A tool result is returned as a list within the content of a user message.
    tool_results = Enum.map(results, &for_api(&1))

    %{
      "role" => "user",
      "content" => tool_results
    }
  end

  # when content is plain text
  def for_api(%Message{content: content} = msg) when is_binary(content) do
    %{
      "role" => Atom.to_string(msg.role),
      "content" => msg.content
    }
  end

  def for_api(%Message{role: :user, content: content}) when is_list(content) do
    %{
      "role" => "user",
      "content" => Enum.map(content, &for_api(&1))
    }
  end

  def for_api(%ContentPart{type: :text} = part) do
    %{"type" => "text", "text" => part.content}
  end

  def for_api(%ContentPart{type: :image} = part) do
    media =
      case Keyword.fetch!(part.options || [], :media) do
        :png ->
          "image/png"

        :gif ->
          "image/gif"

        :jpg ->
          "image/jpeg"

        :jpeg ->
          "image/jpeg"

        :webp ->
          "image/webp"

        value when is_binary(value) ->
          value

        other ->
          message = "Received unsupported media type for ContentPart: #{inspect(other)}"
          Logger.error(message)
          raise LangChainError, message
      end

    %{
      "type" => "image",
      "source" => %{
        "type" => "base64",
        "data" => part.content,
        "media_type" => media
      }
    }
  end

  def for_api(%ContentPart{type: :image_url} = _part) do
    raise LangChainError, "Anthropic does not support image_url"
  end

  # Function support
  def for_api(%Function{} = fun) do
    # I'm here
    %{
      "name" => fun.name,
      "input_schema" => get_parameters(fun)
    }
    |> Utils.conditionally_add_to_map("description", fun.description)
  end

  # ToolCall support
  def for_api(%ToolCall{} = call) do
    %{
      "type" => "tool_use",
      "id" => call.call_id,
      "name" => call.name,
      "input" => call.arguments || %{}
    }
  end

  # ToolResult support
  def for_api(%ToolResult{} = result) do
    %{
      "type" => "tool_result",
      "tool_use_id" => result.tool_call_id,
      "content" => result.content
    }
    |> Utils.conditionally_add_to_map("is_error", result.is_error)
  end

  defp get_parameters(%Function{parameters: [], parameters_schema: nil} = _fun) do
    %{
      "type" => "object",
      "properties" => %{}
    }
  end

  defp get_parameters(%Function{parameters: [], parameters_schema: schema} = _fun)
       when is_map(schema) do
    schema
  end

  defp get_parameters(%Function{parameters: params} = _fun) do
    FunctionParam.to_parameters_schema(params)
  end

  @doc """
  After all the messages have been converted using `for_api/1`, this combines
  multiple sequential tool response messages. The Anthropic API is very strict
  about user, assistant, user, assistant sequenced messages.
  """
  def post_process_and_combine_messages(messages) do
    messages
    |> Enum.reverse()
    |> Enum.reduce([], fn
      # when two "user" role messages are listed together, combine them. This
      # can happen because multiple ToolCalls require multiple tool response
      # messages, but Anthropic does those as a User message and strictly
      # enforces that multiple user messages in a row are not permitted.
      %{"role" => "user"} = item, [%{"role" => "user"} = prev | rest] = _acc ->
        updated_prev = merge_user_messages(item, prev)
        # merge current item into the previous and return the updated list
        # updated_prev = Map.put(prev, "content", item["content"] ++ prev["content"])
        [updated_prev | rest]

      item, acc ->
        [item | acc]
    end)
  end

  # Merge the two user messages
  defp merge_user_messages(%{"role" => "user"} = item, %{"role" => "user"} = prev) do
    item = get_merge_friendly_user_content(item)
    prev = get_merge_friendly_user_content(prev)

    Map.put(prev, "content", item["content"] ++ prev["content"])
  end

  defp get_merge_friendly_user_content(%{"role" => "user", "content" => content} = item)
       when is_binary(content) do
    # replace the string content with text object
    Map.put(item, "content", [%{"type" => "text", "text" => content}])
  end

  defp get_merge_friendly_user_content(%{"role" => "user", "content" => content} = item)
       when is_list(content) do
    item
  end

  defp get_ratelimit_info(response_headers) do
    # extract out all the ratelimit response headers
    #
    #  https://docs.anthropic.com/en/api/rate-limits#response-headers
    {return, _} =
      Map.split(response_headers, [
        "anthropic-ratelimit-requests-limit",
        "anthropic-ratelimit-requests-remaining",
        "anthropic-ratelimit-requests-reset",
        "anthropic-ratelimit-tokens-limit",
        "anthropic-ratelimit-tokens-remaining",
        "anthropic-ratelimit-tokens-reset",
        "retry-after",
        "request-id"
      ])

    return
  end

  defp get_token_usage(%{"usage" => usage} = _response_body) do
    # extract out the reported response token usage
    #
    #    defp get_token_usage(%{"usage" => usage} = _response_body) do
    # extract out the reported response token usage
    #
    #  https://platform.openai.com/docs/api-reference/chat/object#chat/object-usage
    TokenUsage.new!(%{
      input: Map.get(usage, "input_tokens"),
      output: Map.get(usage, "output_tokens")
    })
  end

  defp get_token_usage(_response_body), do: %{}

  @doc """
  Generate a config map that can later restore the model's configuration.
  """
  @impl ChatModel
  @spec serialize_config(t()) :: %{String.t() => any()}
  def serialize_config(%ChatAnthropic{} = model) do
    Utils.to_serializable_map(
      model,
      [
        :endpoint,
        :model,
        :api_version,
        :temperature,
        :max_tokens,
        :receive_timeout,
        :top_p,
        :top_k,
        :stream
      ],
      @current_config_version
    )
  end

  @doc """
  Restores the model from the config.
  """
  @impl ChatModel
  def restore_from_map(%{"version" => 1} = data) do
    ChatAnthropic.new(data)
  end
end
