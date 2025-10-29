defmodule LangChain.ChatModels.ChatAnthropic do
  @moduledoc """
  Module for interacting with [Anthropic models](https://docs.anthropic.com/claude/docs/models-overview#claude-3-a-new-generation-of-ai).

  Parses and validates inputs for making requests to [Anthropic's messages API](https://docs.anthropic.com/claude/reference/messages_post).

  Converts responses into more specialized `LangChain` data structures.

  ## Callbacks

  See the set of available callbacks: `LangChain.Chains.ChainCallbacks`

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

  ### Token Usage

  Anthropic returns token usage information as part of the response body. The
  `LangChain.TokenUsage` is added to the `metadata` of the `LangChain.Message`
  and `LangChain.MessageDelta` structs that are processed under the `:usage`
  key.

  ```elixir
  %LangChain.MessageDelta{
    content: [],
    status: :incomplete,
    index: nil,
    role: :assistant,
    tool_calls: nil,
    metadata: %{
            usage: %LangChain.TokenUsage{
              input: 55,
              output: 4,
              raw: %{
                "cache_creation_input_tokens" => 0,
                "cache_read_input_tokens" => 0,
                "input_tokens" => 55,
                "output_tokens" => 4
              }
            }
    }
  }
  ```

  The `TokenUsage` data is accumulated for `MessageDelta` structs and the final usage information will be on the `LangChain.Message`.

  ## Tool Choice

  Anthropic supports forcing a tool to be used.
  - https://docs.anthropic.com/en/docs/build-with-claude/tool-use#forcing-tool-use

  This is supported through the `tool_choice` options. It takes a plain Elixir map to provide the configuration.

  By default, the LLM will choose a tool call if a tool is available and it determines it is needed. That's the "auto" mode.

  ### Example
  Force the LLM's response to make a tool call of the "get_weather" function.

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

  ## Thinking

  Models like Claude 3.7 Sonnet introduced a hybrid approach which allows for "thinking" and reasoning.
  See the [Anthropic thinking documentation](https://docs.anthropic.com/en/docs/build-with-claude/extended-thinking)
  for up-to-date instructions on the usage.

  For instance, enabling thinking may require the `temperature` to be set to `1` and other settings like `topP` may not be allowed.

  The model supports a `:thinking` attribute where the data is a map that matches the structure in the
  [Anthropic documentation](https://docs.anthropic.com/en/api/messages#body-thinking). It is passed along as-is.

  **Example:**

      # Enable thinking and budget 2,000 tokens for the thinking space.
      model = ChatAnthropic.new!(%{
        model: "claude-3-7-sonnet-latest",
        thinking: %{type: "enabled", budget_tokens: 2000}
      })

      # Disable thinking
      model = ChatAnthropic.new!(%{
        model: "claude-3-7-sonnet-latest",
        thinking: %{type: "disabled"}
      })

  As of the documentation for Claude 3.7 Sonnet, the minimum budget for thinking is 1024 tokens.

  ## Prompt Caching

  Anthropic supports [prompt caching](https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching) to
  reduce costs and latency for frequently repeated content. Prompt caching works by caching large blocks of
  content that are likely to be reused across multiple requests.

  Prompt caching is configured through the `cache_control` option in `ContentPart` options. It can be applied
  to both system messages, regular user messages, tool results, and tool definitions.

  Anthropic limits a conversation to max of 4 cache_control blocks and will refuse to service requests with more.

  ### Basic Usage

  Setting `cache_control: true` is a shortcut for the default ephemeral cache control:

      # System message with caching
      Message.new_system!([
        ContentPart.text!("You are an AI assistant analyzing literary works."),
        ContentPart.text!("<large document content>", cache_control: true)
      ])

      # User message with caching
      Message.new_user!([
        ContentPart.text!("Please analyze this document:"),
        ContentPart.text!("<large document content>", cache_control: true)
      ])

  This will set a single cache breakpoint that will include your functions (processed first) and system message.
  Anthropic limits conversations to a maximum of 4 cache_control blocks.

  For multi-turn conversations, turning on message_caching (see below) will add a second cache breakpoint and
  give you higher cache utilization and response times. Writing to the cache increases write costs so this setting
  is not on by default.

  ### Supported Content Types

  Prompt caching can be applied to:
  - Text content in system messages
  - Text content in user messages
  - Tool results in the `content` field when returning a list of `ContentPart` structs.
  - Tool definitions in the `options` field when creating a `Function` struct.

  For more information, see the [Anthropic prompt caching documentation](https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching).

  ### Advanced Cache Control

  For more explicit control over caching parameters, you can provide a map instead of `true`:

      ContentPart.text!("content", cache_control: %{"type" => "ephemeral", "ttl" => "1h"})

  When `cache_control: true` is used, it automatically expands to `%{"type" => "ephemeral"}` in the API request.
  If you need specific cache control settings like TTL, providing them explicitly preserves the exact values
  sent to the API.

  The default is "5m" for 5 minutes but supports "1h" for 1 hour depending on your account.

  ### Automatic Message Caching for Multi-Turn Conversations

  For multi-turn conversations, Anthropic recommends placing a cache breakpoint on the last user message
  to achieve high cache hit rates. The `:cache_messages` option automates this pattern by automatically
  adding `cache_control` to the last user message's last text content part.

  This is particularly effective for conversations because as new turns are added, the cache breakpoint
  automatically moves to the newest message, maximizing cache utilization across the conversation history.

  With :cache_messages enabled:
  - The cache breakpoint is only added to the **last user message** in the conversation
  - It's added to the **last text ContentPart** within that message (skips tool_result types)
  - This works alongside system message caching and tool caching

  #### Enabling Message Caching

  Message caching is disabled by default since writing to the cache increases write costs. Whether this
  is worth it depends on your situation.

  To enable automatic message caching, set `cache_messages: %{enabled: true}`:

      model = ChatAnthropic.new!(%{
        model: "claude-3-5-sonnet-20241022",
        cache_messages: %{enabled: true}
      })

  This will automatically add cache_control to the last text ContentPart of the last user message
  in every API request.

  #### With Custom TTL

  You can specify a custom TTL (time-to-live) for the cache breakpoint:

      model = ChatAnthropic.new!(%{
        model: "claude-3-5-sonnet-20241022",
        cache_messages: %{enabled: true, ttl: "1h"}
      })

  Supported TTL values are "5m" (5 minutes, default) and "1h" (1 hour), depending on your account settings.

  NOTE: Cache writes with a 5m TTL have cost 1.25X whereas 1h writes cost 3X.

  #### Multi-Turn Conversation Example

  As you add messages to a conversation, the cache breakpoint automatically moves to the latest user message:

      # Turn 1 - cache breakpoint on first message
      {:ok, chain} =
        LLMChain.new!(%{llm: model})
        |> LLMChain.add_message(Message.new_user!("What is machine learning?"))
        |> LLMChain.run()

      # Turn 2 - cache breakpoint moves to second message
      # Previous messages are now in the cache
      {:ok, chain} =
        chain
        |> LLMChain.add_message(Message.new_user!("Can you give me an example?"))
        |> LLMChain.run()

  This achieves ~90-100% cache hit rates in multi-turn conversations, significantly reducing latency and costs.

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

  # Added "thinking" support - https://docs.anthropic.com/en/api/messages#body-thinking

  # TODO: https://docs.anthropic.com/en/api/messages#body-messages - Messages support for images:
  # > We currently support the base64 source type for images, and the image/jpeg, image/png, image/gif, and image/webp media types.

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

    # The maximum tokens allowed for generating a response. This field is
    # required to be present in the API request. The default is a max of 4096
    # tokens, which was the max most Claude models could generate for a long
    # time.
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

    # Supported on "thinking" models like Claude 3.7 and later.
    field :thinking, :map

    # Tool choice option
    field :tool_choice, :map

    # Beta headers
    # https://docs.anthropic.com/claude/docs/tool-use - requires tools-2024-04-04 header during beta
    field :beta_headers, {:array, :string}, default: ["tools-2024-04-04"]

    # Additional level of raw api request and response data
    field :verbose_api, :boolean, default: false

    # Automatically cache messages in multi-turn conversations.
    # Set to %{enabled: true} to add cache_control to the last user message's last ContentPart.
    # Can include TTL: %{enabled: true, ttl: "1h"} or %{enabled: true, ttl: "5m"}
    # Set to %{enabled: false} or nil to disable automatic message caching.
    field :cache_messages, :map
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
    :thinking,
    :tool_choice,
    :beta_headers,
    :verbose_api,
    :cache_messages
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

  def get_system_text(nil) do
    get_system_text(Message.new_system!())
  end

  def get_system_text(%Message{role: :system, content: content} = _message)
      when is_binary(content) do
    [%{"type" => "text", "text" => content}]
  end

  def get_system_text(%Message{role: :system, content: content} = _message)
      when is_list(content) do
    Enum.map(content, &content_part_for_api/1)
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

    messages =
      messages
      |> Enum.map(&message_for_api/1)
      |> post_process_and_combine_messages(anthropic.cache_messages)

    %{
      model: anthropic.model,
      temperature: anthropic.temperature,
      stream: anthropic.stream,
      messages: messages,
      # Anthropic sets the `system` message on the request body, not as part of
      # the messages list.
      system: get_system_text(system)
    }
    |> Utils.conditionally_add_to_map(:tools, get_tools_for_api(tools))
    |> Utils.conditionally_add_to_map(:tool_choice, get_tool_choice(anthropic))
    |> Utils.conditionally_add_to_map(:max_tokens, anthropic.max_tokens)
    |> Utils.conditionally_add_to_map(:top_p, anthropic.top_p)
    |> Utils.conditionally_add_to_map(:top_k, anthropic.top_k)
    |> Utils.conditionally_add_to_map(:thinking, anthropic.thinking)
    |> maybe_transform_for_bedrock(anthropic.bedrock)
  end

  defp maybe_transform_for_bedrock(body, nil), do: body

  defp maybe_transform_for_bedrock(body, %BedrockConfig{} = bedrock) do
    body
    |> Map.put(:anthropic_version, bedrock.anthropic_version)
    |> Map.drop([:model, :stream])
  end

  defp get_tool_choice(%ChatAnthropic{
         tool_choice: %{"type" => "tool", "name" => name} = tool_choice
       })
       when is_binary(name) and byte_size(name) > 0 do
    %{"type" => "tool", "name" => name}
    |> Utils.conditionally_add_to_map(
      "disable_parallel_tool_use",
      Map.get(tool_choice, "disable_parallel_tool_use")
    )
  end

  defp get_tool_choice(%ChatAnthropic{tool_choice: %{"type" => type} = tool_choice})
       when is_binary(type) and byte_size(type) > 0 do
    %{"type" => type}
    |> Utils.conditionally_add_to_map(
      "disable_parallel_tool_use",
      Map.get(tool_choice, "disable_parallel_tool_use")
    )
  end

  defp get_tool_choice(%ChatAnthropic{}), do: nil

  defp get_tools_for_api(nil), do: []

  defp get_tools_for_api(tools) do
    Enum.map(tools, fn
      %Function{} = function ->
        function_for_api(function)
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
    metadata = %{
      model: anthropic.model,
      message_count: length(messages),
      tools_count: length(functions)
    }

    LangChain.Telemetry.span([:langchain, :llm, :call], metadata, fn ->
      try do
        # Track the prompt being sent
        LangChain.Telemetry.llm_prompt(
          %{system_time: System.system_time()},
          %{model: anthropic.model, messages: messages}
        )

        # make base api request and perform high-level success/failure checks
        case do_api_request(anthropic, messages, functions) do
          {:error, %LangChainError{} = error} ->
            {:error, error}

          parsed_data ->
            # Track the response being received
            LangChain.Telemetry.llm_response(
              %{system_time: System.system_time()},
              %{model: anthropic.model, response: parsed_data}
            )

            {:ok, parsed_data}
        end
      rescue
        err in LangChainError ->
          {:error, err}
      end
    end)
  end

  @doc """
  Determine if an error should be retried. If `true`, a fallback LLM may be
  used. If `false`, the error is understood to be more fundamental with the
  request rather than a service issue and it should not be retried or fallback
  to another service.
  """
  @impl ChatModel
  @spec retry_on_fallback?(LangChainError.t()) :: boolean()
  def retry_on_fallback?(%LangChainError{type: "rate_limited"}), do: true
  def retry_on_fallback?(%LangChainError{type: "rate_limit_exceeded"}), do: true
  def retry_on_fallback?(%LangChainError{type: "overloaded"}), do: true
  def retry_on_fallback?(%LangChainError{type: "timeout"}), do: true
  def retry_on_fallback?(%LangChainError{type: "invalid_request_error"}), do: false
  def retry_on_fallback?(_), do: false

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
    raw_data = for_api(anthropic, messages, tools)

    if anthropic.verbose_api do
      IO.inspect(raw_data, label: "RAW DATA BEING SUBMITTED")
    end

    req =
      Req.new(
        url: url(anthropic),
        json: raw_data,
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
        if anthropic.verbose_api do
          IO.inspect(response, label: "RAW REQ RESPONSE")
        end

        Callbacks.fire(anthropic.callbacks, :on_llm_response_headers, [response.headers])

        Callbacks.fire(anthropic.callbacks, :on_llm_ratelimit_info, [
          get_ratelimit_info(response.headers)
        ])

        case do_process_response(anthropic, data) do
          {:error, reason} ->
            {:error, reason}

          result ->
            Callbacks.fire(anthropic.callbacks, :on_llm_new_message, [result])

            result
        end

      {:ok, %Req.Response{status: 429} = response} ->
        rate_limit_info = get_ratelimit_info(response.headers)

        Callbacks.fire(anthropic.callbacks, :on_llm_response_headers, [response.headers])

        Callbacks.fire(anthropic.callbacks, :on_llm_ratelimit_info, [
          rate_limit_info
        ])

        # Rate limit exceeded
        {:error,
         LangChainError.exception(
           type: "rate_limit_exceeded",
           message: "Rate limit exceeded",
           original: rate_limit_info
         )}

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
    raw_data = for_api(anthropic, messages, tools)

    if anthropic.verbose_api do
      IO.inspect(raw_data, label: "RAW DATA BEING SUBMITTED")
    end

    # Track the prompt being sent for streaming
    LangChain.Telemetry.llm_prompt(
      %{system_time: System.system_time(), streaming: true},
      %{model: anthropic.model, messages: messages}
    )

    Req.new(
      url: url(anthropic),
      json: raw_data,
      headers: headers(anthropic),
      receive_timeout: anthropic.receive_timeout,
      aws_sigv4: aws_sigv4_opts(anthropic.bedrock),
      retry: :transient
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
        Callbacks.fire(anthropic.callbacks, :on_llm_response_headers, [response.headers])

        Callbacks.fire(anthropic.callbacks, :on_llm_ratelimit_info, [
          get_ratelimit_info(response.headers)
        ])

        # Track the stream completion
        LangChain.Telemetry.emit_event(
          [:langchain, :llm, :response, streaming: true],
          %{system_time: System.system_time()},
          %{model: anthropic.model}
        )

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
        "stop_reason" => stop_reason,
        "type" => "message",
        "usage" => usage
      }) do
    new_message =
      %{
        role: :assistant,
        content: [],
        status: stop_reason_to_status(stop_reason)
      }
      |> Message.new()
      |> TokenUsage.set_wrapped(get_token_usage(usage))
      |> to_response()

    # reduce over the contents and accumulate to the message
    Enum.reduce(contents, new_message, fn content, acc ->
      do_process_content_response(acc, content)
    end)
  end

  def do_process_response(_model, %{
        "type" => "message_start",
        "message" => %{
          "type" => "message",
          "role" => role,
          "content" => content,
          "usage" => usage
        }
      }) do
    %{
      role: role,
      content: content,
      status: :incomplete
    }
    |> MessageDelta.new()
    |> TokenUsage.set_wrapped(get_token_usage(usage))
    |> to_response()
  end

  def do_process_response(_model, %{
        "type" => "content_block_start",
        "index" => index,
        "content_block" => %{
          "type" => "thinking",
          "thinking" => content,
          "signature" => signature
        }
      }) do
    %{
      role: :assistant,
      content:
        ContentPart.new!(%{type: :thinking, content: content, options: [signature: signature]}),
      status: :incomplete,
      index: index
    }
    |> MessageDelta.new()
    |> to_response()
  end

  def do_process_response(_model, %{
        "type" => "content_block_start",
        "index" => index,
        "content_block" => %{"type" => "text", "text" => content}
      }) do
    %{
      role: :assistant,
      content: ContentPart.text!(content),
      status: :incomplete,
      index: index
    }
    |> MessageDelta.new()
    |> to_response()
  end

  def do_process_response(_model, %{
        "type" => "content_block_start",
        "index" => index,
        "content_block" => %{"type" => "redacted_thinking", "data" => content}
      }) do
    %{
      role: :assistant,
      content:
        ContentPart.new!(%{
          type: :unsupported,
          content: content,
          options: [type: "redacted_thinking"]
        }),
      status: :incomplete,
      index: index
    }
    |> MessageDelta.new()
    |> to_response()
  end

  def do_process_response(_model, %{
        "type" => "content_block_delta",
        "index" => index,
        "delta" => %{"type" => "text_delta", "text" => content}
      }) do
    %{
      role: :assistant,
      content: ContentPart.text!(content),
      status: :incomplete,
      index: index
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

  def do_process_response(_model, %{
        "type" => "content_block_delta",
        "index" => content_index,
        "delta" => %{"type" => "thinking_delta", "thinking" => thinking}
      }) do
    %{
      role: :assistant,
      status: :incomplete,
      index: content_index,
      content: ContentPart.new!(%{type: :thinking, content: thinking})
    }
    |> MessageDelta.new()
    |> to_response()
  end

  def do_process_response(_model, %{
        "type" => "content_block_delta",
        "index" => content_index,
        "delta" => %{"type" => "signature_delta", "signature" => signature}
      }) do
    %{
      role: :assistant,
      status: :incomplete,
      index: content_index,
      content: ContentPart.new!(%{type: :thinking, options: [signature: signature]})
    }
    |> MessageDelta.new()
    |> to_response()
  end

  def do_process_response(
        _model,
        %{
          "type" => "message_delta",
          "delta" => %{"stop_reason" => stop_reason},
          "usage" => usage
        } = _data
      ) do
    %{
      role: :assistant,
      content: nil,
      status: stop_reason_to_status(stop_reason)
    }
    |> MessageDelta.new()
    |> TokenUsage.set_wrapped(get_token_usage(usage))
    |> to_response()
  end

  def do_process_response(
        _model,
        %{
          "type" => "error",
          "error" => %{"type" => type, "message" => reason}
        } = response
      ) do
    Logger.error("Received error from API: #{inspect(response)}")
    {:error, LangChainError.exception(type: type, message: reason, original: response)}
  end

  def do_process_response(_model, %{"error" => %{"message" => reason} = error} = response) do
    Logger.error("Received error from API: #{inspect(response)}")
    {:error, LangChainError.exception(type: error["type"], message: reason, original: response)}
  end

  def do_process_response(_model, {:error, %Jason.DecodeError{} = response}) do
    error_message = "Received invalid JSON: #{inspect(response)}"
    Logger.error(error_message)

    {:error,
     LangChainError.exception(type: "invalid_json", message: error_message, original: response)}
  end

  def do_process_response(
        %ChatAnthropic{bedrock: %BedrockConfig{}},
        %{
          "message" => "Too many requests" <> _rest = message
        } = response
      ) do
    # the error isn't wrapped in an error JSON object. tsk, tsk
    {:error,
     LangChainError.exception(type: "too_many_requests", message: message, original: response)}
  end

  def do_process_response(
        %ChatAnthropic{bedrock: %BedrockConfig{}},
        %{"message" => message} = response
      ) do
    {:error,
     LangChainError.exception(message: "Received error from API: #{message}", original: response)}
  end

  def do_process_response(
        %ChatAnthropic{bedrock: %BedrockConfig{}},
        %{
          bedrock_exception: exceptions
        } = response
      ) do
    {:error,
     LangChainError.exception(
       message: "Stream exception received: #{inspect(exceptions)}",
       original: response
     )}
  end

  def do_process_response(_model, other) do
    Logger.error("Failed to process an unexpected response. #{inspect(other)}")

    {:error,
     LangChainError.exception(
       type: "unexpected_response",
       message: "Unexpected response",
       original: other
     )}
  end

  # for parsing a list of received content JSON objects
  defp do_process_content_response(%Message{} = message, %{"type" => "text", "text" => ""}),
    do: message

  defp do_process_content_response(%Message{} = message, %{"type" => "text", "text" => text}) do
    %Message{message | content: message.content ++ [ContentPart.text!(text)]}
  end

  defp do_process_content_response(%Message{} = message, %{
         "type" => "redacted_thinking",
         "data" => data
       }) do
    parts = message.content || []

    %Message{
      message
      | content:
          parts ++
            [
              ContentPart.new!(%{
                type: :unsupported,
                content: data,
                options: [type: "redacted_thinking"]
              })
            ]
    }
  end

  defp do_process_content_response(%Message{} = message, %{
         "type" => "thinking",
         "thinking" => thinking,
         "signature" => signature
       }) do
    parts = message.content || []

    %Message{
      message
      | content:
          parts ++
            [
              ContentPart.new!(%{
                type: :thinking,
                content: thinking,
                options: [signature: signature]
              })
            ]
    }
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
          (message.tool_calls || []) ++
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
  def parse_stream_events(%ChatAnthropic{bedrock: nil}, {chunk, buffer}) do
    # Combine the incoming data with any buffered incomplete data
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
      |> Enum.reduce([], fn chunk, acc ->
        # Split chunk into lines
        lines = String.split(chunk, "\n", trim: true)

        # Find the data line that contains the JSON. The data contains all the
        # information we need so we skip the "event: " lines.
        case Enum.find(lines, &String.starts_with?(&1, "data: ")) do
          nil ->
            acc

          data_line ->
            # Extract and parse the JSON data
            json = String.replace_prefix(data_line, "data: ", "")

            case Jason.decode(json) do
              {:ok, parsed} -> acc ++ [parsed]
              {:error, _} -> acc
            end
        end
      end)

    {processed, incomplete}
  end

  @doc false
  def decode_stream(%ChatAnthropic{bedrock: nil} = model, {chunk, buffer}) do
    if model.verbose_api do
      IO.inspect(chunk, label: "RCVD RAW CHUNK")
    end

    {to_process, incomplete} = parse_stream_events(model, {chunk, buffer})

    if model.verbose_api do
      IO.inspect(to_process, label: "RAW TO PROCESS")
    end

    processed = Enum.filter(to_process, &relevant_event?/1)

    if model.verbose_api do
      IO.inspect(processed, label: "READY TO PROCESS")
    end

    {processed, incomplete}
  end

  def relevant_event?(%{"type" => "message_start"}), do: true
  def relevant_event?(%{"type" => "content_block_delta"}), do: true
  def relevant_event?(%{"type" => "content_block_start"}), do: true
  def relevant_event?(%{"type" => "message_delta"}), do: true
  def relevant_event?(%{"type" => "error"}), do: true
  # ignoring
  def relevant_event?(%{"type" => "ping"}), do: false
  def relevant_event?(%{"type" => "content_block_stop"}), do: false
  def relevant_event?(%{"type" => "message_stop"}), do: false
  # catch-all for when we miss something
  def relevant_event?(event) do
    Logger.error("Unsupported event received when parsing Anthropic response: #{inspect(event)}")
    false
  end

  @doc false
  def decode_stream(%ChatAnthropic{bedrock: %BedrockConfig{}}, {chunk, buffer}, chunks \\ []) do
    {chunks, remaining} = BedrockStreamDecoder.decode_stream({chunk, buffer}, chunks)

    chunks =
      Enum.filter(chunks, fn chunk ->
        Map.has_key?(chunk, :bedrock_exception) || relevant_event?(chunk)
      end)

    {chunks, remaining}
  end

  @doc """
  Convert a LangChain structure to the expected map of data for the Anthropic API.
  """
  @spec for_api(Message.t() | ContentPart.t() | Function.t()) ::
          %{String.t() => any()} | no_return()
  # def for_api(%Message{role: :assistant, tool_calls: calls} = msg)
  #     when is_list(calls) and calls != [] do
  #   text_content =
  #     if is_binary(msg.content) do
  #       [
  #         %{
  #           "type" => "text",
  #           "text" => msg.content
  #         }
  #       ]
  #     else
  #       []
  #     end

  #   tool_calls = Enum.map(calls, &for_api(&1))

  #   %{
  #     "role" => "assistant",
  #     "content" => text_content ++ tool_calls
  #   }
  # end

  # def for_api(%Message{role: :tool, tool_results: results}) when is_list(results) do
  #   # convert ToolResult into the expected format for Anthropic.
  #   #
  #   # A tool result is returned as a list within the content of a user message.
  #   tool_results = Enum.map(results, &for_api(&1))

  #   %{
  #     "role" => "user",
  #     "content" => tool_results
  #   }
  # end

  # # when content is plain text
  # def for_api(%Message{content: content} = msg) when is_binary(content) do
  #   %{
  #     "role" => Atom.to_string(msg.role),
  #     "content" => [msg.content |> ContentPart.text!() |> content_part_for_api()]
  #   }
  # end

  # def for_api(%Message{role: :user, content: content}) when is_list(content) do
  #   %{
  #     "role" => "user",
  #     "content" => Enum.map(content, &content_part_for_api(&1))
  #   }
  # end

  # def for_api(%Message{role: role, content: content}) when is_list(content) do
  #   %{
  #     "role" => Atom.to_string(role),
  #     "content" =>
  #       content
  #       |> Enum.map(&content_part_for_api(&1))
  #       |> Enum.reject(&is_nil/1)
  #   }
  # end

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
      "content" => content_parts_for_api(result.content)
    }
    |> Utils.conditionally_add_to_map("is_error", result.is_error)
    |> Utils.conditionally_add_to_map("cache_control", get_cache_control_setting(result.options))
  end

  @doc """
  Convert a Function to the format expected by the Anthropic API.
  """
  @spec function_for_api(Function.t()) :: map() | no_return()
  def function_for_api(%Function{} = fun) do
    # I'm here
    %{
      "name" => fun.name,
      "input_schema" => get_parameters(fun)
    }
    |> Utils.conditionally_add_to_map("description", fun.description)
    |> Utils.conditionally_add_to_map("cache_control", get_cache_control_setting(fun.options))
  end

  @doc """
  Converts a Message to the format expected by the Anthropic API.
  """
  def message_for_api(%Message{role: :assistant, tool_calls: calls} = msg)
      when is_list(calls) and calls != [] do
    text_content = content_parts_for_api(msg.content)

    tool_calls = Enum.map(calls, &for_api(&1))

    %{
      "role" => "assistant",
      "content" => text_content ++ tool_calls
    }
  end

  def message_for_api(%Message{role: :tool, tool_results: results}) when is_list(results) do
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
  def message_for_api(%Message{content: content} = msg) when is_binary(content) do
    %{
      "role" => Atom.to_string(msg.role),
      "content" => [msg.content |> ContentPart.text!() |> content_part_for_api()]
    }
  end

  def message_for_api(%Message{role: :user, content: content}) when is_list(content) do
    %{
      "role" => "user",
      "content" => Enum.map(content, &content_part_for_api(&1))
    }
  end

  def message_for_api(%Message{role: role, content: content}) when is_list(content) do
    %{
      "role" => Atom.to_string(role),
      "content" =>
        content
        |> Enum.map(&content_part_for_api(&1))
        |> Enum.reject(&is_nil/1)
    }
  end

  @doc """
  Converts a list of ContentParts to the format expected by the Anthropic API.
  """
  def content_parts_for_api(contents)
  def content_parts_for_api(nil), do: []

  def content_parts_for_api(contents) when is_list(contents) do
    Enum.map(contents, &content_part_for_api/1)
  end

  def content_parts_for_api(content) when is_binary(content) do
    [
      %{
        "type" => "text",
        "text" => content
      }
    ]
  end

  # Get the cache control setting from the options.
  #
  # If the setting is true, return the default cache control block.
  # If the setting is false, return nil.
  # If the setting is a map, return the map.
  #
  # If the setting is not provided, return nil.
  defp get_cache_control_setting(options) do
    case Keyword.fetch(options || [], :cache_control) do
      :error ->
        nil

      {:ok, setting} ->
        if setting == true do
          @default_cache_control_block
        else
          setting
        end
    end
  end

  @doc """
  Converts a ContentPart to the format expected by the Anthropic API.

  Handles different content types:
  - `:text` - Converts to a text content part, optionally with cache control settings
  - `:thinking` - Converts to a thinking content part with required signature
  - `:unsupported` - Handles custom content types specified in options
  - `:image` - Converts to an image content part with base64 data and media type
  - `:image_url` - Raises an error as Anthropic doesn't support image URLs

  ## Options

  For `:text` type:
  - `:cache_control` - When provided, adds cache control settings to the content

  For `:thinking` type:
  - `:signature` - Required signature for thinking content

  For `:unsupported` type:
  - `:type` - Required string specifying the custom content type

  For `:image` type:
  - `:media` - Required media type (`:png`, `:jpg`, `:jpeg`, `:gif`, `:webp`, or a string)

  Returns `nil` for unsupported content without required options.
  """
  @spec content_part_for_api(ContentPart.t()) :: map() | nil | no_return()
  def content_part_for_api(%ContentPart{type: :text} = part) do
    %{"type" => "text", "text" => part.content}
    |> Utils.conditionally_add_to_map("cache_control", get_cache_control_setting(part.options))
  end

  def content_part_for_api(%ContentPart{type: :thinking} = part) do
    # Handle thinking content with signature
    case Keyword.fetch(part.options || [], :signature) do
      :error ->
        # Without a valid signature, we can't send thinking content
        Logger.warning("Thinking ContentPart without signature will be omitted: #{inspect(part)}")
        nil

      {:ok, signature} ->
        # Thinking content with signature
        %{"type" => "thinking", "thinking" => part.content, "signature" => signature}
    end
  end

  def content_part_for_api(%ContentPart{type: :unsupported} = part) do
    # Handle unsupported content types by using the type provided in options
    case Keyword.fetch(part.options || [], :type) do
      :error ->
        # If no type is provided, log a warning and return nil
        Logger.warning(
          "Unsupported ContentPart without type specification will be omitted: #{inspect(part)}"
        )

        nil

      {:ok, type} when is_binary(type) ->
        # Use the specified type from options and pass the content as data
        %{"type" => type, "data" => part.content}
    end
  end

  def content_part_for_api(%ContentPart{type: :image} = part) do
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

  def content_part_for_api(%ContentPart{type: :image_url} = _part) do
    raise LangChainError, "Anthropic does not support image_url"
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

  When `cache_messages` is set, it also adds cache_control to the last user message's
  last ContentPart to enable efficient caching in multi-turn conversations.
  """
  def post_process_and_combine_messages(messages, cache_messages \\ nil) do
    messages
    |> Enum.reverse()
    |> Enum.with_index()
    |> Enum.reduce([], fn
      # when two "user" role messages are listed together, combine them. This
      # can happen because multiple ToolCalls require multiple tool response
      # messages, but Anthropic does those as a User message and strictly
      # enforces that multiple user messages in a row are not permitted.
      {%{"role" => "user"} = item, index}, [%{"role" => "user"} = prev | rest] = _acc ->
        updated_prev = merge_user_messages(item, prev)
        # If this is the last message (index 0) and it's a user message, add cache_control
        if index == 0 && cache_messages_enabled?(cache_messages) do
          [add_cache_control_to_last_content(updated_prev, cache_messages) | rest]
        else
          [updated_prev | rest]
        end

      {%{"role" => "user"} = item, 0}, acc ->
        # This is the last message and it's a user message, add cache_control if requested
        if cache_messages_enabled?(cache_messages) do
          [add_cache_control_to_last_content(item, cache_messages) | acc]
        else
          [item | acc]
        end

      {item, _index}, acc ->
        [item | acc]
    end)
  end

  # Check if cache_messages is enabled
  defp cache_messages_enabled?(%{enabled: true}), do: true
  defp cache_messages_enabled?(%{"enabled" => true}), do: true
  defp cache_messages_enabled?(_), do: false

  # Add cache_control to the last ContentPart in a user message
  defp add_cache_control_to_last_content(
         %{"role" => "user", "content" => content} = message,
         cache_messages
       )
       when is_list(content) and length(content) > 0 do
    # Get the cache control setting
    cache_control = get_cache_control_from_setting(cache_messages)

    # Find the last text content part (skip tool_result types)
    {_content_before_last_text, last_text_index} =
      content
      |> Enum.with_index()
      |> Enum.reverse()
      |> Enum.find({nil, nil}, fn {part, _idx} ->
        part["type"] == "text"
      end)

    case last_text_index do
      nil ->
        # No text parts found, return message unchanged
        message

      idx ->
        # Update the last text part with cache_control (if it doesn't already have one)
        updated_content =
          List.update_at(content, idx, fn part ->
            if Map.has_key?(part, "cache_control") do
              # Keep existing cache_control
              part
            else
              # Add cache_control
              Map.put(part, "cache_control", cache_control)
            end
          end)

        Map.put(message, "content", updated_content)
    end
  end

  defp add_cache_control_to_last_content(message, _cache_messages), do: message

  # Convert cache_messages setting to the cache_control block format
  defp get_cache_control_from_setting(%{enabled: true} = settings) do
    case Map.get(settings, :ttl) do
      nil ->
        @default_cache_control_block

      ttl ->
        %{"type" => "ephemeral", "ttl" => ttl}
    end
  end

  defp get_cache_control_from_setting(%{"enabled" => true} = settings) do
    case Map.get(settings, "ttl") do
      nil ->
        @default_cache_control_block

      ttl ->
        %{"type" => "ephemeral", "ttl" => ttl}
    end
  end

  defp get_cache_control_from_setting(_), do: nil

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

  defp get_token_usage(usage_data) do
    # if prompt caching has been used the response will also contain
    # "cache_creation_input_tokens" and "cache_read_input_tokens"
    TokenUsage.new!(%{
      input: Map.get(usage_data, "input_tokens"),
      output: Map.get(usage_data, "output_tokens"),
      raw: usage_data
    })
  end

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
