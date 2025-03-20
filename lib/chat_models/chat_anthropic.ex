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

      handler = %{
        on_llm_ratelimit_info: fn _chain, headers ->
          IO.inspect(headers)
        end
      }

      %{llm: ChatAnthropic.new!(%{model: "..."})}
      |> LLMChain.new!()
      # ... add messages ...
      |> LLMChain.add_callback(handler)
      |> LLMChain.run()

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

  ## Tool Choice

  Anthropic supports forcing a tool to be used.
  - https://docs.anthropic.com/en/docs/build-with-claude/tool-use#forcing-tool-use

  This is supported through the `tool_choice` options. It takes a plain Elixir map to provide the configuration.

  By default, the LLM will choose a tool call if a tool is available and it determines it is needed. That's the "auto" mode.

  ### Example
  For the LLM's response to make a tool call of the "get_weather" function.

      ChatAnthropic.new(%{
        model: "...",
        tool_choice: %{"type" => "tool", "name" => "get_weather"}
      })

  ## AWS Bedrock Support

  Anthropic Claude is supported in [AWS Bedrock](https://docs.aws.amazon.com/bedrock/latest/userguide/what-is-bedrock.html).

  To configure `ChatAnthropic` for use on AWS Bedrock:

  1. Request [Model Access](https://console.aws.amazon.com/bedrock/home?#/modelaccess) to get access to the Anthropic models you intend to use.
  2. Using your AWS Console, create an Access Key for your application.
  3. Set the key values in your `AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY` ENVs.
  4. Get the Model ID for the model you intend to use. [Base Models](https://console.aws.amazon.com/bedrock/home?#/models)
  5. Refer to `LangChain.Utils.BedrockConfig` for setting up the Bedrock authentication credentials for your environment.
  6. Setup your ChatAnthropic similar to the following:

      alias LangChain.ChatModels.ChatAnthropic

      ChatAnthropic.new!(%{
        model: "anthropic.claude-3-5-sonnet-20241022-v2:0",
        bedrock: BedrockConfig.from_application_env!()
      })

  """
  use Ecto.Schema
  require Logger
  import Ecto.Changeset
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
  alias LangChain.Utils.BedrockStreamDecoder
  alias LangChain.Utils.BedrockConfig

  @behaviour ChatModel

  @current_config_version 1

  @default_cache_control_block %{"type" => "ephemeral"}

  # allow up to 1 minute for response.
  @receive_timeout 60_000

  @primary_key false
  embedded_schema do
    # API endpoint to use. Defaults to Anthropic's API
    field :endpoint, :string, default: "https://api.anthropic.com/v1/messages"

    # Configuration for AWS Bedrock. Configure this instead of endpoint & api_key if you want to use Bedrock.
    embeds_one :bedrock, BedrockConfig

    # API key for Anthropic. If not set, will use global api key. Allows for usage
    # of a different API key per-call if desired. For instance, allowing a
    # customer to provide their own.
    field :api_key, :string, redact: true

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

    # A list of maps for callback handlers (treat as private)
    field :callbacks, {:array, :map}, default: []

    # Tool choice option
    field :tool_choice, :map

    # Beta headers
    # https://docs.anthropic.com/claude/docs/tool-use - requires tools-2024-04-04 header during beta
    field :beta_headers, {:array, :string}, default: ["tools-2024-04-04"]
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
    :tool_choice,
    :beta_headers
  ]
  @required_fields [:endpoint, :model]

  @doc """
  Setup a ChatAnthropic client configuration.
  """
  @spec new(attrs :: map()) :: {:ok, t} | {:error, Ecto.Changeset.t()}
  def new(%{} = attrs \\ %{}) do
    %ChatAnthropic{}
    |> cast(attrs, @create_fields)
    |> cast_embed(:bedrock)
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
    {system, messages} =
      Utils.split_system_message(
        messages,
        "Anthropic only supports a single System message, however, you may use multiple ContentParts for the System message to indicate where prompt caching should be used."
      )

    system_text =
      case system do
        nil ->
          nil

        %Message{role: :system, content: [_ | _]} = message ->
          for_api(message)

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
    |> Utils.conditionally_add_to_map(:tool_choice, get_tool_choice(anthropic))
    |> Utils.conditionally_add_to_map(:max_tokens, anthropic.max_tokens)
    |> Utils.conditionally_add_to_map(:top_p, anthropic.top_p)
    |> Utils.conditionally_add_to_map(:top_k, anthropic.top_k)
    |> maybe_transform_for_bedrock(anthropic.bedrock)
  end

  defp maybe_transform_for_bedrock(body, nil), do: body

  defp maybe_transform_for_bedrock(body, %BedrockConfig{} = bedrock) do
    body
    |> Map.put(:anthropic_version, bedrock.anthropic_version)
    |> Map.drop([:model, :stream])
  end

  defp get_tool_choice(%ChatAnthropic{
         tool_choice: %{"type" => "tool", "name" => name} = _tool_choice
       })
       when is_binary(name) and byte_size(name) > 0,
       do: %{"type" => "tool", "name" => name}

  defp get_tool_choice(%ChatAnthropic{tool_choice: %{"type" => type} = _tool_choice})
       when is_binary(type) and byte_size(type) > 0,
       do: %{"type" => type}

  defp get_tool_choice(%ChatAnthropic{}), do: nil

  defp get_tools_for_api(nil), do: []

  defp get_tools_for_api(tools) do
    Enum.map(tools, fn
      %Function{} = function ->
        for_api(function)
    end)
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

  def call(%ChatAnthropic{} = anthropic, messages, functions) when is_list(messages) do
    try do
      # make base api request and perform high-level success/failure checks
      case do_api_request(anthropic, messages, functions) do
        {:error, %LangChainError{} = error} ->
          {:error, error}

        parsed_data ->
          {:ok, parsed_data}
      end
    rescue
      err in LangChainError ->
        {:error, err}
    end
  end

  # Call Anthropic's API.
  #
  # The result of the function is:
  #
  # - `result` - where `result` is a data-structure like a list or map.
  # - `{:error, %LangChainError{} = reason}` - An `LangChain.LangChainError` exception with an explanation of what went wrong.
  #
  # If `stream: false`, the completed message is returned.
  #
  # Retries the request up to 3 times on transient errors with a 1 second delay
  @doc false
  @spec do_api_request(t(), [Message.t()], ChatModel.tools(), non_neg_integer()) ::
          list() | struct() | {:error, LangChainError.t()} | no_return()
  def do_api_request(anthropic, messages, tools, retry_count \\ 3)

  def do_api_request(_anthropic, _messages, _functions, 0) do
    raise LangChainError,
      type: "retries_exceeded",
      message: "Retries exceeded. Connection failed."
  end

  def do_api_request(
        %ChatAnthropic{stream: false} = anthropic,
        messages,
        tools,
        retry_count
      ) do
    req =
      Req.new(
        url: url(anthropic),
        json: for_api(anthropic, messages, tools),
        headers: headers(anthropic),
        receive_timeout: anthropic.receive_timeout,
        retry: :transient,
        max_retries: 3,
        retry_delay: fn attempt -> 300 * attempt end,
        aws_sigv4: aws_sigv4_opts(anthropic.bedrock)
      )

    req
    |> Req.post()
    # parse the body and return it as parsed structs
    |> case do
      {:ok, %Req.Response{status: 200, body: data} = response} ->
        Callbacks.fire(anthropic.callbacks, :on_llm_ratelimit_info, [
          get_ratelimit_info(response.headers)
        ])

        Callbacks.fire(anthropic.callbacks, :on_llm_token_usage, [
          get_token_usage(data)
        ])

        case do_process_response(anthropic, data) do
          {:error, reason} ->
            {:error, reason}

          result ->
            Callbacks.fire(anthropic.callbacks, :on_llm_new_message, [result])
            result
        end

      {:ok, %Req.Response{status: 529}} ->
        {:error, LangChainError.exception(type: "overloaded", message: "Overloaded")}

      {:error, %Req.TransportError{reason: :timeout} = err} ->
        {:error,
         LangChainError.exception(type: "timeout", message: "Request timed out", original: err)}

      {:error, %Req.TransportError{reason: :closed}} ->
        # Force a retry by making a recursive call decrementing the counter
        Logger.debug(fn -> "Mint connection closed: retry count = #{inspect(retry_count)}" end)
        do_api_request(anthropic, messages, tools, retry_count - 1)

      {:error, %LangChainError{}} = error ->
        # pass through the already handled exception
        error

      other ->
        message = "Unexpected and unhandled API response! #{inspect(other)}"
        Logger.error(message)
        {:error, LangChainError.exception(type: "unexpected_response", message: message)}
    end
  end

  def do_api_request(
        %ChatAnthropic{stream: true} = anthropic,
        messages,
        tools,
        retry_count
      ) do
    Req.new(
      url: url(anthropic),
      json: for_api(anthropic, messages, tools),
      headers: headers(anthropic),
      receive_timeout: anthropic.receive_timeout,
      aws_sigv4: aws_sigv4_opts(anthropic.bedrock)
    )
    |> Req.post(
      into:
        Utils.handle_stream_fn(
          anthropic,
          &decode_stream(anthropic, &1),
          &do_process_response(anthropic, &1)
        )
    )
    |> case do
      {:ok, %Req.Response{body: data} = response} ->
        Callbacks.fire(anthropic.callbacks, :on_llm_ratelimit_info, [
          get_ratelimit_info(response.headers)
        ])

        data

      # The error tuple was successfully received from the API. Unwrap it and
      # return it as an error.
      {:ok, {:error, %LangChainError{} = error}} ->
        {:error, error}

      {:error, %Req.TransportError{reason: :timeout} = err} ->
        {:error,
         LangChainError.exception(type: "timeout", message: "Request timed out", original: err)}

      {:error, %Req.TransportError{reason: :closed}} ->
        # Force a retry by making a recursive call decrementing the counter
        Logger.debug(fn -> "Mint connection closed: retry count = #{inspect(retry_count)}" end)
        do_api_request(anthropic, messages, tools, retry_count - 1)

      {:error, %LangChainError{}} = error ->
        # pass through the already handled exception
        error

      other ->
        message = "Unhandled and unexpected response from streamed post call. #{inspect(other)}"
        Logger.error(message)
        {:error, LangChainError.exception(type: "unexpected_response", message: message)}
    end
  end

  defp aws_sigv4_opts(nil), do: nil
  defp aws_sigv4_opts(%BedrockConfig{} = bedrock), do: BedrockConfig.aws_sigv4_opts(bedrock)

  @spec get_api_key(binary() | nil) :: String.t()
  defp get_api_key(api_key) do
    # if no API key is set default to `""` which will raise an error
    api_key || Config.resolve(:anthropic_key, "")
  end

  defp headers(%ChatAnthropic{
         bedrock: nil,
         api_key: api_key,
         api_version: api_version,
         beta_headers: beta_headers
       }) do
    %{
      "x-api-key" => get_api_key(api_key),
      "content-type" => "application/json",
      "anthropic-version" => api_version
    }
    |> Utils.conditionally_add_to_map(
      "anthropic-beta",
      if(!Enum.empty?(beta_headers), do: Enum.join(beta_headers, ","))
    )
  end

  defp headers(%ChatAnthropic{bedrock: %BedrockConfig{}}) do
    %{
      "content-type" => "application/json",
      "accept" => "application/json"
    }
  end

  defp url(%ChatAnthropic{bedrock: nil} = anthropic) do
    anthropic.endpoint
  end

  defp url(%ChatAnthropic{bedrock: %BedrockConfig{} = bedrock, stream: stream} = anthropic) do
    BedrockConfig.url(bedrock, model: anthropic.model, stream: stream)
  end

  # Parse a new message response
  @doc false
  @spec do_process_response(t(), data :: %{String.t() => any()} | {:error, any()}) ::
          Message.t()
          | [Message.t()]
          | MessageDelta.t()
          | [MessageDelta.t()]
          | {:error, LangChainError.t()}
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

  def do_process_response(_model, %{
        "type" => "content_block_start",
        "index" => tool_index,
        "content_block" => %{"type" => "tool_use", "id" => call_id, "name" => tool_name}
      }) do
    %{
      role: :assistant,
      status: :incomplete,
      tool_calls: [
        ToolCall.new!(%{
          type: :function,
          name: tool_name,
          call_id: call_id,
          index: tool_index
        })
      ]
    }
    |> MessageDelta.new()
    |> to_response()
  end

  def do_process_response(_model, %{
        "type" => "content_block_delta",
        "index" => tool_index,
        "delta" => %{"type" => "input_json_delta", "partial_json" => partial_json}
      }) do
    %{
      role: :assistant,
      status: :incomplete,
      tool_calls: [
        ToolCall.new!(%{
          arguments: partial_json,
          index: tool_index
        })
      ]
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
    Callbacks.fire(model.callbacks, :on_llm_token_usage, [get_token_usage(data)])

    %{
      role: :assistant,
      content: "",
      status: stop_reason_to_status(stop_reason)
    }
    |> MessageDelta.new()
    |> to_response()
  end

  def do_process_response(_model, %{
        "type" => "error",
        "error" => %{"type" => type, "message" => reason}
      }) do
    Logger.error("Received error from API: #{inspect(reason)}")
    {:error, LangChainError.exception(type: type, message: reason)}
  end

  def do_process_response(_model, %{"error" => %{"message" => reason} = error}) do
    Logger.error("Received error from API: #{inspect(reason)}")
    {:error, LangChainError.exception(type: error["type"], message: reason)}
  end

  def do_process_response(_model, {:error, %Jason.DecodeError{} = response}) do
    error_message = "Received invalid JSON: #{inspect(response)}"
    Logger.error(error_message)

    {:error,
     LangChainError.exception(type: "invalid_json", message: error_message, original: response)}
  end

  def do_process_response(%ChatAnthropic{bedrock: %BedrockConfig{}}, %{
        "message" => "Too many requests" <> _rest = message
      }) do
    # the error isn't wrapped in an error JSON object. tsk, tsk
    {:error, LangChainError.exception(type: "too_many_requests", message: message)}
  end

  def do_process_response(%ChatAnthropic{bedrock: %BedrockConfig{}}, %{"message" => message}) do
    {:error, LangChainError.exception(message: "Received error from API: #{message}")}
  end

  def do_process_response(%ChatAnthropic{bedrock: %BedrockConfig{}}, %{
        bedrock_exception: exceptions
      }) do
    {:error,
     LangChainError.exception(message: "Stream exception received: #{inspect(exceptions)}")}
  end

  def do_process_response(_model, other) do
    Logger.error("Trying to process an unexpected response. #{inspect(other)}")

    {:error,
     LangChainError.exception(type: "unexpected_response", message: "Unexpected response")}
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

  defp to_response({:error, %Ecto.Changeset{} = changeset}),
    do: {:error, LangChainError.exception(changeset)}

  defp stop_reason_to_status("end_turn"), do: :complete
  defp stop_reason_to_status("tool_use"), do: :complete
  defp stop_reason_to_status("max_tokens"), do: :length
  defp stop_reason_to_status("stop_sequence"), do: :complete

  defp stop_reason_to_status(other) do
    Logger.warning("Unsupported stop_reason. Reason: #{inspect(other)}")
    nil
  end

  @doc false
  def decode_stream(%ChatAnthropic{bedrock: nil}, {chunk, buffer}) do
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
  defp relevant_event?("event: error\n" <> _rest), do: true
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

  @doc false
  def decode_stream(%ChatAnthropic{bedrock: %BedrockConfig{}}, {chunk, buffer}, chunks \\ []) do
    {chunks, remaining} = BedrockStreamDecoder.decode_stream({chunk, buffer}, chunks)

    chunks =
      Enum.filter(chunks, fn chunk ->
        Map.has_key?(chunk, :bedrock_exception) || relevant_event?("event: #{chunk["type"]}\n")
      end)

    {chunks, remaining}
  end

  @doc """
  Convert a LangChain structure to the expected map of data for the Anthropic API.
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

  def for_api(%Message{role: :system, content: content}) when is_list(content) do
    Enum.map(content, &for_api(&1))
  end

  def for_api(%ContentPart{type: :text} = part) do
    case Keyword.fetch(part.options || [], :cache_control) do
      :error ->
        %{"type" => "text", "text" => part.content}

      {:ok, setting} ->
        setting = if setting == true, do: @default_cache_control_block, else: setting
        %{"type" => "text", "text" => part.content, "cache_control" => setting}
    end
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
    case Keyword.fetch(result.options || [], :cache_control) do
      :error ->
        %{
          "type" => "tool_result",
          "tool_use_id" => result.tool_call_id,
          "content" => result.content
        }

      {:ok, setting} ->
        setting = if setting == true, do: @default_cache_control_block, else: setting

        %{
          "type" => "tool_result",
          "tool_use_id" => result.tool_call_id,
          "content" => result.content,
          "cache_control" => setting
        }
    end
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
    # if prompt caching has been used the response will also contain
    # "cache_creation_input_tokens" and "cache_read_input_tokens"
    TokenUsage.new!(%{
      input: Map.get(usage, "input_tokens"),
      output: Map.get(usage, "output_tokens"),
      raw: usage
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
        :stream,
        :beta_headers
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
