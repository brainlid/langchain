defmodule LangChain.ChatModels.ChatOpenAI do
  @moduledoc """
  Represents the [OpenAI
  ChatModel](https://platform.openai.com/docs/api-reference/chat/create).

  Parses and validates inputs for making a requests from the OpenAI Chat API.

  Converts responses into more specialized `LangChain` data structures.

  - https://github.com/openai/openai-cookbook/blob/main/examples/How_to_call_functions_with_chat_models.ipynb

  ## OpenAI-compatible services

  This module follows OpenAI's own API, including Azure OpenAI, and adopts
  OpenAI's changes as they ship. Services that offer an OpenAI-compatible
  endpoint (Cloudflare Workers AI, Groq, OpenRouter, vLLM, SGLang, Ollama, LM
  Studio and others) often do not support those changes, and many ignore or
  silently drop what they do not understand. For example, a system message
  sent under the `developer` role, as `reasoning_mode` does, never reaches a
  model served by SGLang or vLLM when the model's chat template has no
  `developer` branch.

  Use `LangChain.ChatModels.ChatOpenAICompatible` for those services. It sends
  only the fields you set, always sends system messages as `system`, and never
  falls back to the global OpenAI API key.

  ## ContentPart Types

  OpenAI supports several types of content parts that can be combined in a single message:

  ### Text Content
  Basic text content is the default and most common type:

      Message.new_user!("Hello, how are you?")

  ### Image Content
  OpenAI supports both base64-encoded images and image URLs:

      # Using a base64 encoded image
      Message.new_user!([
        ContentPart.text!("What's in this image?"),
        ContentPart.image!("base64_encoded_image_data", media: :jpg)
      ])

      # Using an image URL
      Message.new_user!([
        ContentPart.text!("Describe this image:"),
        ContentPart.image_url!("https://example.com/image.jpg")
      ])

  For images, you can specify the detail level which affects token usage:
  - `detail: "low"` - Lower resolution, fewer tokens
  - `detail: "high"` - Higher resolution, more tokens
  - `detail: "auto"` - Let the model decide

  ### File Content
  OpenAI supports both base64-encoded files and file IDs:

      # Using a base64 encoded file
      Message.new_user!([
        ContentPart.text!("Process this file:"),
        ContentPart.file!("base64_encoded_file_data",
          type: :base64,
          filename: "document.pdf"
        )
      ])

      # Using a file ID (after uploading to OpenAI)
      Message.new_user!([
        ContentPart.text!("Process this file:"),
        ContentPart.file!("file-1234", type: :file_id)
      ])

  ## Callbacks

  See the set of available callbacks: `LangChain.Chains.ChainCallbacks`

  ### Rate Limit API Response Headers

  OpenAI returns rate limit information in the response headers. Those can be
  accessed using the LLM callback `on_llm_ratelimit_info` like this:

      handlers = %{
        on_llm_ratelimit_info: fn headers ->
          IO.inspect(headers)
        end
      }

      {:ok, chat} = ChatOpenAI.new(%{callbacks: [handlers]})

  Handlers assigned to the model are fired by the model itself and receive only
  the event's argument. Handlers assigned to an `LangChain.Chains.LLMChain`
  receive the chain as an additional first argument:

      handler = %{
        on_llm_ratelimit_info: fn _chain, headers ->
          IO.inspect(headers)
        end
      }

      %{llm: ChatOpenAI.new!(%{})}
      |> LLMChain.new!()
      |> LLMChain.add_callback(handler)

  When a request is received, something similar to the following will be output
  to the console.

      %{
        "x-ratelimit-limit-requests" => ["5000"],
        "x-ratelimit-limit-tokens" => ["160000"],
        "x-ratelimit-remaining-requests" => ["4999"],
        "x-ratelimit-remaining-tokens" => ["159973"],
        "x-ratelimit-reset-requests" => ["12ms"],
        "x-ratelimit-reset-tokens" => ["10ms"],
        "x-request-id" => ["req_1234"]
      }

  ### Token Usage

  OpenAI returns token usage information as part of the response body. The
  `LangChain.TokenUsage` is added to the `metadata` of the `LangChain.Message`
  and `LangChain.MessageDelta` structs that are processed under the `:usage`
  key.

  The OpenAI documentation instructs to provide the `stream_options` with the
  `include_usage: true` for the information to be provided.

  ```elixir
  chat = ChatOpenAI.new!(%{stream: true, stream_options: %{include_usage: true}})
  ```

  The `TokenUsage` data is accumulated for `MessageDelta` structs and the final usage information will be on the `LangChain.Message`.

  NOTE: Of special note is that the `TokenUsage` information is returned once
  for all "choices" in the response. The `LangChain.TokenUsage` data is added to
  each message, but if your usage requests multiple choices, you will see the
  same usage information for each choice but it is duplicated and only one
  response is meaningful.

  ## Tool Choice

  OpenAI's ChatGPT API supports forcing a tool to be used.
  - https://platform.openai.com/docs/api-reference/chat/create#chat-create-tool_choice

  This is supported through the `tool_choice` options. It takes a plain Elixir
  map to provide the configuration.

  By default, the LLM will choose a tool call if a tool is available and it
  determines it is needed. That's the "auto" mode.

  ## Parallel Tool Calls

  By default, OpenAI models may decide to make multiple tool calls at once,
  including calling the same tool multiple times. You can limit this behavior by
  setting the `parallel_tool_calls`
  [option](https://platform.openai.com/docs/api-reference/chat/create#chat_create-parallel_tool_calls)
  to false.

  ### Example
  For the LLM's response to make a tool call of the "get_weather" function.

      ChatOpenAI.new(%{
        model: "...",
        tool_choice: %{"type" => "function", "function" => %{"name" => "get_weather"}}
      })

  ## Log Probabilities

  OpenAI can return log probability information for output tokens, which is
  useful for evaluating model confidence, building classifiers, or debugging
  token selection.

  Enable with the `logprobs` option. Optionally set `top_logprobs` to receive
  the N most likely tokens (0-20) at each position:

      chat = ChatOpenAI.new!(%{
        model: "gpt-4o",
        logprobs: true,
        top_logprobs: 3
      })

  When enabled, the logprobs data from the API response is placed in the
  `metadata` field under the `"logprobs"` key. This works for both
  non-streaming (`LangChain.Message`) and streaming (`LangChain.MessageDelta`)
  responses:

      # Non-streaming
      {:ok, [%Message{metadata: %{"logprobs" => logprobs}}]} =
        ChatOpenAI.call(chat, [message], [])

      # Streaming - logprobs appear on each delta chunk
      chat = ChatOpenAI.new!(%{model: "gpt-4o", stream: true, logprobs: true})
      # Each MessageDelta will have metadata: %{"logprobs" => ...}

      # logprobs contains the raw OpenAI response structure:
      # %{
      #   "content" => [
      #     %{
      #       "token" => "Hello",
      #       "logprob" => -0.0002,
      #       "bytes" => [72, 101, 108, 108, 111],
      #       "top_logprobs" => [
      #         %{"token" => "Hello", "logprob" => -0.0002, ...},
      #         %{"token" => "Hi", "logprob" => -8.53, ...},
      #         ...
      #       ]
      #     },
      #     ...
      #   ]
      # }

  When `logprobs` is not enabled or the response contains no logprobs data,
  `metadata` will be `nil`.

  See the [OpenAI documentation](https://developers.openai.com/api/reference/resources/completions/methods/create)
  for full details on the response structure.

  ## Azure OpenAI Support

  To use `ChatOpenAI` with Microsoft's Azure hosted OpenAI models, the
  `endpoint` must be overridden and the API key needs to be provided in some
  way. The [MS Quickstart guide for REST
  access](https://learn.microsoft.com/en-us/azure/ai-services/openai/chatgpt-quickstart?tabs=command-line%2Cjavascript-keyless%2Ctypescript-keyless%2Cpython-new&pivots=rest-api)
  may be helpful.

  In order to use it, you must have an Azure account and from the console, a
  model must be deployed for your account. Use the Azure AI Foundry and Azure
  OpenAI Service to deploy the model you want to use. The entire URL is used as
  the `endpoint` and the provided `key` is used as the `api_key`.

  The following is an example of setting up `ChatOpenAI` for use with an Azure
  hosted model.

      endpoint = System.fetch_env!("AZURE_OPENAI_ENDPOINT")
      api_key = System.fetch_env!("AZURE_OPENAI_KEY")

      llm =
        ChatOpenAI.new!(%{
          endpoint: endpoint,
          api_key: api_key,
          seed: 0,
          temperature: 1,
          stream: false
        })

  The URL itself specifies the model to use and the `model` attribute is
  disregarded.

  A fake example URL for the endpoint value:

  `https://some-subdomain.cognitiveservices.azure.com/openai/deployments/gpt-4o-mini/chat/completions?api-version=2024-08-01-preview"`

  ## Service Tier

  The `service_tier` option asks the provider to serve the request with a
  particular processing tier, trading cost against latency:

      ChatOpenAI.new!(%{model: "gpt-5", service_tier: "priority"})

  OpenAI accepts values such as `"auto"`, `"default"`, `"flex"`, `"scale"`,
  `"priority"` and `"fast"`. OpenAI-compatible providers accept their own
  subset, so the value is passed through without validation.

  The tier that actually served the request can differ from the one requested.
  When the provider reports it, it is kept in the token usage's `raw` map:

      %Message{metadata: %{usage: %TokenUsage{raw: %{"service_tier" => tier}}}} = message

  When streaming, token usage (and the tier with it) is only reported when
  `stream_options: %{include_usage: true}` is set. Some providers don't report a
  tier at all.

  ## Token Limits

  `max_tokens` sets the upper bound on generated tokens. It is sent as
  `max_completion_tokens`, OpenAI's current name for the limit. OpenAI's
  reasoning models require that name, and the limit includes reasoning tokens.

      ChatOpenAI.new!(%{model: "gpt-5", max_tokens: 4000})

  Many OpenAI-compatible servers only accept the older `max_tokens` key and
  ignore `max_completion_tokens`. `LangChain.ChatModels.ChatOpenAICompatible`
  sends the limit as `max_tokens`.

  ## Additional Request Parameters

  `extra_body` is a map of values merged into the request body last, so they
  override anything the model computed. It reaches API parameters that
  `ChatOpenAI` has no field for:

      ChatOpenAI.new!(%{
        model: "gpt-5",
        extra_body: %{"store" => true, "metadata" => %{"run" => "eval-42"}}
      })

  - Keys may be strings or atoms. A key naming one the model already sends
    replaces that value, so `%{"n" => 2}` overrides the `n` field.
  - A `nil` value removes the key from the body. `%{"n" => nil}` stops the
    always-sent `n` from going out.
  - Nested maps are merged key by key. Any other value replaces the existing
    one.

  Overriding `stream`, `messages`, `tools` or `model` can break response
  handling. Headers and transport options belong in `req_config`, not
  `extra_body`. See `LangChain.Utils.merge_extra_body/2` for the full rules.

  ## Reasoning Model Support

  OpenAI's reasoning models (the o-series and the gpt-5 family) take a
  `reasoning_effort` setting. To send it, set `:reasoning_mode` to `true`:

      model = ChatOpenAI.new!(%{model: "gpt-5", reasoning_mode: true, reasoning_effort: "low"})

  Setting `reasoning_mode` to `true` does two things:

  - Sends `:reasoning_effort` in the request. It sets how much time, and how
    many tokens, the model spends reasoning before it answers. OpenAI rejects
    the setting on models that do not reason. The accepted values depend on
    the model; across current models they are "none", "minimal", "low",
    "medium" (the default here), "high" and "xhigh".
  - Sends system messages under the `:developer` role, OpenAI's convention for
    reasoning models. OpenAI's API also accepts `:system` on its current
    reasoning models and treats it as `:developer`.

  ### Returned Thinking

  OpenAI does not return a reasoning model's thinking on this API. Some
  OpenAI-compatible services return it in a `reasoning_content` field, which
  is parsed into a `LangChain.Message.ContentPart` of type `:thinking`. See
  `LangChain.ChatModels.ChatOpenAICompatible` for those services.

  ## Connection Retry Behavior

  The `retry_count` option controls how many times a request is retried when
  a pooled HTTP connection turns out to be stale (server closed it between
  requests). This is a transport-level issue where retrying with a fresh
  connection is the correct response.

  **Only closed-connection errors are retried.** Timeouts, rate limits (429),
  overloaded (529), authentication errors, and invalid requests all return
  immediately -- they are not problems that a simple retry will fix.

  | `retry_count` | Total HTTP requests |
  |---|---|
  | `0` | 1 (no retries) |
  | `1` | 2 (1 initial + 1 retry) |
  | `2` (default) | 3 (1 initial + 2 retries) |

  Req's built-in HTTP retry is disabled to prevent the two retry layers from
  compounding. See [GitHub issue #503](https://github.com/brainlid/langchain/issues/503).

  When running LLM calls from a background job queue (e.g., Oban) that has its
  own retry logic, set `retry_count: 0` so there are no hidden retries:

      ChatOpenAI.new!(%{model: "...", retry_count: 0})

  """
  use Ecto.Schema
  require Logger
  import Ecto.Changeset
  alias __MODULE__
  alias LangChain.Config
  alias LangChain.ChatModels.ChatModel
  alias LangChain.ChatModels.ChatCompletionsFormat
  alias LangChain.Message
  alias LangChain.Message.ContentPart
  alias LangChain.Message.ToolCall
  alias LangChain.TokenUsage
  alias LangChain.Function
  alias LangChain.LangChainError
  alias LangChain.Utils
  alias LangChain.MessageDelta
  alias LangChain.Callbacks

  @behaviour ChatModel

  @current_config_version 1

  # NOTE: As of gpt-4 and gpt-3.5, only one function_call is issued at a time
  # even when multiple requests could be issued based on the prompt.

  # allow up to 1 minute for response.
  @receive_timeout 60_000

  @primary_key false
  embedded_schema do
    field :endpoint, :string, default: "https://api.openai.com/v1/chat/completions"
    # field :model, :string, default: "gpt-4"
    field :model, :string, default: "gpt-3.5-turbo"
    # API key for OpenAI. If not set, will use global api key. Allows for usage
    # of a different API key per-call if desired. For instance, allowing a
    # customer to provide their own.
    field :api_key, :string, redact: true

    # Organization ID for OpenAI. If not set, will use global org_id. Allows for usage
    # of a different organization ID per-call if desired.
    field :org_id, :string, redact: true

    # What sampling temperature to use, between 0 and 2. Higher values like 0.8
    # will make the output more random, while lower values like 0.2 will make it
    # more focused and deterministic.
    field :temperature, :float, default: 1.0
    # Number between -2.0 and 2.0. Positive values penalize new tokens based on
    # their existing frequency in the text so far, decreasing the model's
    # likelihood to repeat the same line verbatim.
    field :frequency_penalty, :float, default: nil

    # Set when using an OpenAI reasoning model (the o-series and the gpt-5
    # family). Sends `reasoning_effort`, and sends system messages under the
    # `developer` role.
    field :reasoning_mode, :boolean, default: false

    # Sent only when `reasoning_mode` is true. Constrains how much the model
    # reasons before answering. OpenAI accepts "none", "minimal", "low",
    # "medium", "high" and "xhigh", with the supported subset depending on the
    # model. Lower effort gives faster responses and spends fewer tokens on
    # reasoning.
    field :reasoning_effort, :string, default: "medium"

    # Verbosity level for the response.
    # https://platform.openai.com/docs/api-reference/chat/create#chat-create-verbosity
    field :verbosity, :string

    # The processing tier to serve the request with. Known values include
    # "auto", "default", "flex", "scale", "priority" and "fast"; the set differs
    # between OpenAI and compatible providers, so it is not validated here. The
    # tier that actually served the request is reported in the token usage's
    # `raw` map under "service_tier".
    # https://platform.openai.com/docs/api-reference/chat/create#chat-create-service_tier
    field :service_tier, :string

    # Duration in seconds for the response to be received. When streaming a very
    # lengthy response, a longer time limit may be required. However, when it
    # goes on too long by itself, it tends to hallucinate more.
    field :receive_timeout, :integer, default: @receive_timeout
    # Seed for more deterministic output. Helpful for testing.
    # https://platform.openai.com/docs/guides/text-generation/reproducible-outputs
    field :seed, :integer
    # How many chat completion choices to generate for each input message.
    field :n, :integer, default: 1
    field :json_response, :boolean, default: false
    field :json_schema, :map, default: nil
    field :stream, :boolean, default: false
    # Upper bound on generated tokens. Sent as `max_completion_tokens`, OpenAI's
    # current name for the limit, which also counts reasoning tokens.
    field :max_tokens, :integer, default: nil
    # Options for streaming response. Only set this when you set `stream: true`
    # https://platform.openai.com/docs/api-reference/chat/create#chat-create-stream_options
    #
    # Set to `%{include_usage: true}` to have token usage returned when
    # streaming.
    field :stream_options, :map, default: nil

    # Tool choice option
    field :tool_choice, :map

    field :parallel_tool_calls, :boolean

    # When true, the API returns log probability information for each output
    # token. Ref: https://platform.openai.com/docs/api-reference/chat/create#chat-create-logprobs
    field :logprobs, :boolean

    # An integer between 0 and 20 specifying the number of most likely tokens
    # to return at each position. Requires `logprobs` to be set to `true`.
    # Ref: https://platform.openai.com/docs/api-reference/chat/create#chat-create-top_logprobs
    field :top_logprobs, :integer

    # A list of maps for callback handlers (treated as internal)
    field :callbacks, {:array, :map}, default: []

    # Can send a string user_id to help ChatGPT detect abuse by users of the
    # application.
    # https://platform.openai.com/docs/guides/safety-best-practices/end-user-ids
    field :user, :string

    # For help with debugging. It outputs the RAW Req response received and the
    # RAW Elixir map being submitted to the API.
    field :verbose_api, :boolean, default: false

    # Number of retries on closed-connection errors (stale pool). The initial
    # request always runs; this controls additional attempts only.
    field :retry_count, :integer, default: 2

    # Req options to merge into the request.
    # Refer to `https://hexdocs.pm/req/Req.html#new/1-options` for
    # `Req.new` supported set of options.
    field :req_config, :map, default: %{}

    # Provider-specific values merged into the request body last, overriding
    # anything the model computed. A `nil` value removes that key from the
    # body. See `LangChain.Utils.merge_extra_body/2` for the merge rules.
    field :extra_body, :map, default: nil
  end

  @type t :: %ChatOpenAI{}

  @create_fields [
    :endpoint,
    :model,
    :temperature,
    :frequency_penalty,
    :api_key,
    :org_id,
    :seed,
    :n,
    :stream,
    :reasoning_mode,
    :reasoning_effort,
    :verbosity,
    :service_tier,
    :receive_timeout,
    :json_response,
    :json_schema,
    :max_tokens,
    :stream_options,
    :user,
    :tool_choice,
    :parallel_tool_calls,
    :logprobs,
    :top_logprobs,
    :verbose_api,
    :retry_count,
    :req_config,
    :extra_body,
    :callbacks
  ]
  @required_fields [:endpoint, :model]

  @spec get_api_key(t()) :: String.t()
  defp get_api_key(%ChatOpenAI{api_key: api_key}) do
    # if no API key is set default to `""` which will raise a OpenAI API error
    api_key || Config.resolve(:openai_key, "")
  end

  @spec get_org_id(t()) :: String.t() | nil
  defp get_org_id(%ChatOpenAI{org_id: org_id}) when is_binary(org_id), do: org_id
  defp get_org_id(%ChatOpenAI{}), do: Config.resolve(:openai_org_id)

  @spec get_proj_id() :: String.t() | nil
  defp get_proj_id() do
    Config.resolve(:openai_proj_id)
  end

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
    |> validate_number(:top_logprobs, greater_than_or_equal_to: 0, less_than_or_equal_to: 20)
    |> validate_top_logprobs_requires_logprobs()
  end

  defp validate_top_logprobs_requires_logprobs(changeset) do
    top_logprobs = get_field(changeset, :top_logprobs)
    logprobs = get_field(changeset, :logprobs)

    if top_logprobs && !logprobs do
      add_error(changeset, :top_logprobs, "requires logprobs to be enabled")
    else
      changeset
    end
  end

  @doc """
  Return the params formatted for an API request.
  """
  @spec for_api(t | Message.t() | Function.t(), message :: [map()], ChatModel.tools()) :: %{
          atom() => any()
        }
  def for_api(%ChatOpenAI{} = openai, messages, tools) do
    %{
      model: openai.model,
      temperature: openai.temperature,
      n: openai.n,
      stream: openai.stream,
      messages: ChatCompletionsFormat.messages_for_api(messages, system_role: system_role(openai))
    }
    |> Utils.conditionally_add_to_map(:user, openai.user)
    |> Utils.conditionally_add_to_map(:frequency_penalty, openai.frequency_penalty)
    |> Utils.conditionally_add_to_map(:response_format, set_response_format(openai))
    |> Utils.conditionally_add_to_map(
      :reasoning_effort,
      if(openai.reasoning_mode, do: openai.reasoning_effort, else: nil)
    )
    |> Utils.conditionally_add_to_map(:verbosity, openai.verbosity)
    |> Utils.conditionally_add_to_map(:max_completion_tokens, openai.max_tokens)
    |> Utils.conditionally_add_to_map(:service_tier, openai.service_tier)
    |> Utils.conditionally_add_to_map(:seed, openai.seed)
    |> Utils.conditionally_add_to_map(
      :stream_options,
      get_stream_options_for_api(openai.stream_options)
    )
    |> Utils.conditionally_add_to_map(:tools, ChatCompletionsFormat.tools_for_api(tools))
    |> Utils.conditionally_add_to_map(:tool_choice, get_tool_choice(openai))
    |> Utils.conditionally_add_to_map(:parallel_tool_calls, openai.parallel_tool_calls)
    |> Utils.conditionally_add_to_map(:logprobs, openai.logprobs)
    |> Utils.conditionally_add_to_map(:top_logprobs, openai.top_logprobs)
    |> Utils.merge_extra_body(openai.extra_body)
  end

  defp get_stream_options_for_api(nil), do: nil

  defp get_stream_options_for_api(%{} = data) do
    %{"include_usage" => Map.get(data, :include_usage, Map.get(data, "include_usage"))}
  end

  defp set_response_format(%ChatOpenAI{json_response: true, json_schema: json_schema})
       when not is_nil(json_schema) do
    %{
      "type" => "json_schema",
      "json_schema" => json_schema
    }
  end

  defp set_response_format(%ChatOpenAI{json_response: true}) do
    %{"type" => "json_object"}
  end

  defp set_response_format(%ChatOpenAI{json_response: false}) do
    # NOTE: The default handling when unspecified is `%{"type" => "text"}`
    #
    # For improved compatibility with other APIs like LMStudio, this returns a
    # `nil` which has the same effect.
    nil
  end

  defp get_tool_choice(%ChatOpenAI{
         tool_choice: %{"type" => "function", "function" => %{"name" => name}} = _tool_choice
       })
       when is_binary(name) and byte_size(name) > 0,
       do: %{"type" => "function", "function" => %{"name" => name}}

  defp get_tool_choice(%ChatOpenAI{tool_choice: %{"type" => type} = _tool_choice})
       when is_binary(type) and byte_size(type) > 0,
       do: type

  defp get_tool_choice(%ChatOpenAI{}), do: nil

  @doc """
  Convert a LangChain Message-based structure to the expected map of data for
  the OpenAI API. This happens within the context of the model configuration as
  well. The additional context is needed to correctly convert a role to either
  `:system` or `:developer`.

  NOTE: The `ChatOpenAI` model's functions are reused in other modules. For this
  reason, model is more generally defined as a struct.
  """
  @spec for_api(
          struct(),
          Message.t()
          | LangChain.PromptTemplate.t()
          | ToolCall.t()
          | LangChain.Message.ToolResult.t()
          | ContentPart.t()
          | Function.t()
        ) ::
          %{String.t() => any()} | [%{String.t() => any()}]
  def for_api(%_{} = model, item) do
    ChatCompletionsFormat.item_for_api(item, system_role: system_role(model))
  end

  @doc """
  Convert a list of ContentParts to the expected map of data for the OpenAI API.

  Thinking and unsupported parts are omitted. Both are response-side artifacts
  with no request representation on this API surface. The omission is
  unconditional so a conversation presents the same prompt prefix turn after
  turn, which keeps prompt caching working.
  """
  def content_parts_for_api(%_{} = _model, content_parts) when is_list(content_parts) do
    ChatCompletionsFormat.content_parts_for_api(content_parts)
  end

  @doc """
  Convert a ContentPart to the expected map of data for the OpenAI API.
  """
  def content_part_for_api(%_{} = _model, %ContentPart{} = part) do
    ChatCompletionsFormat.content_part_for_api(part)
  end

  @doc false
  def get_parameters(%Function{} = fun), do: ChatCompletionsFormat.get_parameters(fun)

  # OpenAI's reasoning models take system messages under the `:developer`
  # role. Any other struct reusing these functions sends `:system`.
  defp system_role(%ChatOpenAI{reasoning_mode: true}), do: :developer
  defp system_role(_model), do: :system

  @doc """
  Calls the OpenAI API passing the ChatOpenAI struct with configuration, plus
  either a simple message or the list of messages to act as the prompt.

  Optionally pass in a list of tools available to the LLM for requesting
  execution in response.

  Optionally pass in a callback function that can be executed as data is
  received from the API.

  **NOTE:** This function *can* be used directly, but the primary interface
  should be through `LangChain.Chains.LLMChain`. The `ChatOpenAI` module is more
  focused on translating the `LangChain` data structures to and from the OpenAI
  API.

  Another benefit of using `LangChain.Chains.LLMChain` is that it combines the
  storage of messages, adding tools, adding custom context that should be
  passed to tools, and automatically applying `LangChain.MessageDelta`
  structs as they are are received, then converting those to the full
  `LangChain.Message` once fully complete.
  """
  @impl ChatModel
  def call(openai, prompt, tools \\ [])

  def call(%ChatOpenAI{} = openai, prompt, tools) when is_binary(prompt) do
    messages = [
      Message.new_system!(),
      Message.new_user!(prompt)
    ]

    call(openai, messages, tools)
  end

  def call(%ChatOpenAI{} = openai, messages, tools) when is_list(messages) do
    metadata = %{
      model: openai.model,
      provider: provider(),
      message_count: length(messages),
      tools_count: length(tools)
    }

    ChatModel.llm_telemetry_span(openai, metadata, fn ->
      try do
        # Track the prompt being sent
        LangChain.Telemetry.llm_prompt(
          %{system_time: System.system_time()},
          %{model: openai.model, messages: messages}
        )

        # make base api request and perform high-level success/failure checks
        case do_api_request(openai, messages, tools) do
          {:error, reason} ->
            {:error, reason}

          parsed_data ->
            # Track the response being received
            LangChain.Telemetry.llm_response(
              %{system_time: System.system_time()},
              %{model: openai.model, response: parsed_data}
            )

            {:ok, parsed_data}
        end
      rescue
        err in LangChainError ->
          {:error, err}
      end
    end)
  end

  # Make the API request from the OpenAI server.
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
  @spec do_api_request(t(), [Message.t()], ChatModel.tools(), integer() | nil) ::
          list() | struct() | {:error, LangChainError.t()}
  def do_api_request(openai, messages, tools, retry_count \\ nil)

  def do_api_request(_openai, _messages, _tools, 0) do
    raise LangChainError, "Retries exceeded. Connection failed."
  end

  def do_api_request(
        %ChatOpenAI{stream: false} = openai,
        messages,
        tools,
        retry_count
      ) do
    retry_count = retry_count || openai.retry_count + 1
    raw_data = for_api(openai, messages, tools)

    if openai.verbose_api do
      IO.inspect(raw_data, label: "RAW DATA BEING SUBMITTED")
    end

    req =
      Req.new(
        url: openai.endpoint,
        json: raw_data,
        # required for OpenAI API
        auth: {:bearer, get_api_key(openai)},
        # required for Azure OpenAI version
        headers: [
          {"api-key", get_api_key(openai)}
        ],
        receive_timeout: openai.receive_timeout,
        # Disable Req-level retry to prevent compounding with LangChain's own
        # :closed retry. See https://github.com/brainlid/langchain/issues/503
        retry: false
      )

    req
    |> maybe_add_org_id_header(openai)
    |> maybe_add_proj_id_header()
    |> Req.merge(openai.req_config |> Keyword.new())
    |> Req.post()
    # parse the body and return it as parsed structs
    |> case do
      {:ok, %Req.Response{body: data} = response} ->
        if openai.verbose_api do
          IO.inspect(response, label: "RAW REQ RESPONSE")
        end

        Callbacks.fire(openai.callbacks, :on_llm_response_headers, [response.headers])

        Callbacks.fire(openai.callbacks, :on_llm_ratelimit_info, [
          ChatCompletionsFormat.get_ratelimit_info(response.headers)
        ])

        case do_process_response(openai, data) do
          {:error, %LangChainError{} = reason} ->
            {:error, reason}

          result ->
            Callbacks.fire(openai.callbacks, :on_llm_new_message, [result])

            # Track non-streaming response completion
            LangChain.Telemetry.emit_event(
              [:langchain, :llm, :response, :non_streaming],
              %{system_time: System.system_time()},
              %{
                model: openai.model,
                response_size: byte_size(inspect(result))
              }
            )

            result
        end

      {:error, %Req.TransportError{reason: :timeout} = err} ->
        {:error,
         LangChainError.exception(type: "timeout", message: "Request timed out", original: err)}

      {:error, %Req.TransportError{reason: :closed}} ->
        # Force a retry by making a recursive call decrementing the counter
        Logger.debug(fn -> "Mint connection closed: retry count = #{inspect(retry_count)}" end)
        do_api_request(openai, messages, tools, retry_count - 1)

      other ->
        Logger.warning(fn -> "Unexpected and unhandled API response! #{inspect(other)}" end)
        other
    end
  end

  def do_api_request(
        %ChatOpenAI{stream: true} = openai,
        messages,
        tools,
        retry_count
      ) do
    retry_count = retry_count || openai.retry_count + 1
    raw_data = for_api(openai, messages, tools)

    if openai.verbose_api do
      IO.inspect(raw_data, label: "RAW DATA BEING SUBMITTED")
    end

    Req.new(
      url: openai.endpoint,
      json: raw_data,
      # required for OpenAI API
      auth: {:bearer, get_api_key(openai)},
      # required for Azure OpenAI version
      headers: [
        {"api-key", get_api_key(openai)}
      ],
      receive_timeout: openai.receive_timeout
    )
    |> maybe_add_org_id_header(openai)
    |> maybe_add_proj_id_header()
    |> Req.merge(openai.req_config |> Keyword.new())
    |> Req.post(
      into:
        Utils.handle_stream_fn(
          openai,
          &decode_stream/1,
          &do_process_response(openai, &1)
        )
    )
    |> case do
      {:ok, %Req.Response{body: data} = response} ->
        Callbacks.fire(openai.callbacks, :on_llm_response_headers, [response.headers])

        Callbacks.fire(openai.callbacks, :on_llm_ratelimit_info, [
          ChatCompletionsFormat.get_ratelimit_info(response.headers)
        ])

        data

      {:error, %LangChainError{} = error} ->
        {:error, error}

      {:error, %Req.TransportError{reason: :timeout} = err} ->
        {:error,
         LangChainError.exception(type: "timeout", message: "Request timed out", original: err)}

      {:error, %Req.TransportError{reason: :closed}} ->
        # Force a retry by making a recursive call decrementing the counter
        Logger.debug(fn -> "Connection closed: retry count = #{inspect(retry_count)}" end)
        do_api_request(openai, messages, tools, retry_count - 1)

      other ->
        Logger.warning(fn ->
          "Unhandled and unexpected response from streamed post call. #{inspect(other)}"
        end)

        {:error,
         LangChainError.exception(
           type: "unexpected_response",
           message: "Unexpected response",
           original: other
         )}
    end
  end

  @doc """
  Decode a streamed response from an OpenAI-compatible server. Parses a string
  of received content into an Elixir map data structure using string keys.

  If a partial response was received, meaning the JSON text is split across
  multiple data frames, then the incomplete portion is returned as-is in the
  buffer. The function will be successively called, receiving the incomplete
  buffer data from a previous call, and assembling it to parse.
  """
  @spec decode_stream({String.t(), String.t()}, list()) ::
          {[%{String.t() => any()}], String.t()}
  def decode_stream({raw_data, buffer}, done \\ []) do
    ChatCompletionsFormat.decode_stream({raw_data, buffer}, done)
  end

  # Parse a new message response
  @doc false
  @spec do_process_response(any(), data :: any()) ::
          :skip
          | TokenUsage.t()
          | Message.t()
          | [Message.t() | MessageDelta.t() | TokenUsage.t() | {:error, LangChainError.t()}]
          | MessageDelta.t()
          | [MessageDelta.t()]
          | ToolCall.t()
          | {:error, LangChainError.t()}
  def do_process_response(_model, data), do: ChatCompletionsFormat.process_response(data)

  defp maybe_add_org_id_header(%Req.Request{} = req, %ChatOpenAI{} = openai) do
    org_id = get_org_id(openai)

    if org_id do
      Req.Request.put_header(req, "OpenAI-Organization", org_id)
    else
      req
    end
  end

  defp maybe_add_proj_id_header(%Req.Request{} = req) do
    proj_id = get_proj_id()

    if proj_id do
      Req.Request.put_header(req, "OpenAI-Project", proj_id)
    else
      req
    end
  end

  @impl ChatModel
  def provider, do: "openai"

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
  def retry_on_fallback?(%LangChainError{type: "timeout"}), do: true
  def retry_on_fallback?(%LangChainError{type: "too_many_requests"}), do: true
  def retry_on_fallback?(_), do: false

  @doc """
  Generate a config map that can later restore the model's configuration.
  """
  @impl ChatModel
  @spec serialize_config(t()) :: %{String.t() => any()}
  def serialize_config(%ChatOpenAI{} = model) do
    Utils.to_serializable_map(
      model,
      [
        :endpoint,
        :model,
        :temperature,
        :frequency_penalty,
        :reasoning_mode,
        :reasoning_effort,
        :receive_timeout,
        :seed,
        :n,
        :json_response,
        :json_schema,
        :stream,
        :max_tokens,
        :stream_options,
        :service_tier,
        :extra_body
      ],
      @current_config_version
    )
  end

  @doc """
  Restores the model from the config.
  """
  @impl ChatModel
  def restore_from_map(%{"version" => 1} = data) do
    ChatOpenAI.new(data)
  end
end
