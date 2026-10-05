defmodule LangChain.ChatModels.ChatOpenAICompatible do
  @moduledoc """
  Chat model for services that expose an OpenAI-compatible Chat Completions
  endpoint: Cloudflare Workers AI, Groq, OpenRouter, Together, DeepInfra,
  Fireworks, vLLM, SGLang, Ollama, LM Studio and others.

  ## When to use this rather than `ChatOpenAI`

  `LangChain.ChatModels.ChatOpenAI` is the client for OpenAI's own API and
  follows OpenAI's changes, such as sending the token limit as
  `max_completion_tokens` and sending system messages under the `developer`
  role for reasoning models. Compatible services often do not adopt those
  changes, and many of them ignore or silently drop what they do not
  understand. A `developer` message sent to an SGLang or vLLM server, for
  example, reaches a model whose chat template has no `developer` branch and is
  rendered as nothing, so the model never sees its instructions.

  This module targets the part of the API that compatible services share:

  - **It sends only what you set.** Every optional field defaults to `nil` and
    is left out of the request when `nil`. A model configured with only
    `endpoint` and `model` sends exactly `model`, `stream` and `messages`.
  - **System messages are always sent as `system`.** Every compatible server
    and chat template handles that role.
  - **The token limit is sent as `max_tokens`.**
  - **`reasoning_effort` is sent whenever it is set**, independent of any other
    option. Accepted values differ between services, so it is not validated
    here.
  - **Credentials are explicit.** There is no fallback to the global OpenAI API
    key, and no OpenAI organization or project headers are sent. With
    `api_key: nil`, no `Authorization` header is sent at all, which suits local
    servers that need no key.

  ## Usage

      ChatOpenAICompatible.new!(%{
        endpoint: "https://api.groq.com/openai/v1/chat/completions",
        api_key: System.fetch_env!("GROQ_API_KEY"),
        model: "llama-3.3-70b-versatile"
      })

  `endpoint` is the full URL of the chat completions route, not a base URL.

  ## Provider-specific parameters

  Anything a service supports that has no field here goes in `extra_body`. It
  is merged into the request body last. A `nil` value removes a key. See
  `LangChain.Utils.merge_extra_body/2` for the merge rules.

      ChatOpenAICompatible.new!(%{
        endpoint: "http://localhost:8000/v1/chat/completions",
        model: "Qwen/Qwen3-8B",
        extra_body: %{"top_k" => 20, "repetition_penalty" => 1.05}
      })

  Extra headers go in `req_config`:

      req_config: %{headers: [{"x-custom-header", "value"}]}

  Overriding `stream`, `messages`, `tools` or `model` through `extra_body` can
  break response handling.

  ## Thinking

  Compatible services that serve reasoning models return the model's thinking
  in a `reasoning_content` field beside `content`, on `choices[].message` for a
  single response and on `choices[].delta` for each streamed chunk. That
  thinking becomes a `LangChain.Message.ContentPart` of type `:thinking`, held
  separately from the answer text and ordered ahead of it:

      [
        %ContentPart{type: :thinking, content: "Comparing the tenths place..."},
        %ContentPart{type: :text, content: "9.9 is larger."}
      ]

  Thinking parts are never sent back to the service. A conversation presents
  the same prompt prefix turn after turn, which keeps prompt caching working.

  ## Provider recipes

  ### Cloudflare Workers AI

      ChatOpenAICompatible.new!(%{
        endpoint:
          "https://api.cloudflare.com/client/v4/accounts/\#{account_id}/ai/v1/chat/completions",
        api_key: api_token,
        model: "@cf/zai-org/glm-5.3-flash",
        reasoning_effort: "low",
        req_config: %{
          headers: [
            {"cf-aig-gateway-id", gateway_id},
            {"x-session-affinity", session_id}
          ]
        }
      })

  - `cf-aig-gateway-id` routes the call through a named AI Gateway. On the
    Workers Free plan some models, GLM among them, are reachable only this way.
  - `x-session-affinity` sends requests that share a session id to the same
    replica, so a repeated prompt prefix is served from the prompt cache. Use
    one id for requests that share a system prompt, such as one per
    conversation. Without it, repeated prefixes are not cached.
  - `reasoning_effort` controls how long the model thinks, which affects
    latency and output tokens.
  - AI Gateway can answer a byte-identical request from its response cache.

  ### Ollama and LM Studio

  Local servers need no key:

      ChatOpenAICompatible.new!(%{
        endpoint: "http://localhost:11434/v1/chat/completions",
        model: "llama3.2"
      })

  LM Studio listens on `http://localhost:1234/v1/chat/completions` by default.

  ### vLLM and SGLang

      ChatOpenAICompatible.new!(%{
        endpoint: "http://localhost:8000/v1/chat/completions",
        model: "Qwen/Qwen3-8B"
      })

  Add `api_key` when the server was started with one.

  ### OpenRouter and Groq

      ChatOpenAICompatible.new!(%{
        endpoint: "https://openrouter.ai/api/v1/chat/completions",
        api_key: System.fetch_env!("OPENROUTER_API_KEY"),
        model: "meta-llama/llama-3.3-70b-instruct"
      })

  Groq's endpoint is `https://api.groq.com/openai/v1/chat/completions`.

  ## Token usage

  Usage is reported as a `LangChain.TokenUsage` in the `:usage` key of the
  returned message's `metadata`. When streaming, most services report usage
  only when `stream_options: %{include_usage: true}` is set.

  ## Connection Retry Behavior

  `retry_count` controls how many times a request is retried when a pooled
  HTTP connection turns out to be stale (the server closed it between
  requests). Only closed-connection errors are retried. Timeouts, rate limits,
  authentication errors and invalid requests are returned immediately.
  """
  use Ecto.Schema
  require Logger
  import Ecto.Changeset
  alias __MODULE__
  alias LangChain.ChatModels.ChatModel
  alias LangChain.ChatModels.ChatCompletionsFormat
  alias LangChain.Message
  alias LangChain.LangChainError
  alias LangChain.Utils
  alias LangChain.Callbacks

  @behaviour ChatModel

  @current_config_version 1

  # allow up to 1 minute for response.
  @receive_timeout 60_000

  @primary_key false
  embedded_schema do
    # Full URL of the service's chat completions route.
    field :endpoint, :string
    field :model, :string

    # Sent as a Bearer token. When `nil`, no `Authorization` header is sent.
    field :api_key, :string, redact: true

    field :stream, :boolean, default: false

    # Set to `%{include_usage: true}` to have token usage returned when
    # streaming.
    field :stream_options, :map, default: nil

    field :temperature, :float, default: nil
    field :top_p, :float, default: nil

    # Upper bound on generated tokens, sent as `max_tokens`.
    field :max_tokens, :integer, default: nil

    field :seed, :integer, default: nil
    field :stop, {:array, :string}, default: nil
    field :frequency_penalty, :float, default: nil
    field :presence_penalty, :float, default: nil

    # Sent whenever it is set. Accepted values differ between services, so it
    # is passed through as given.
    field :reasoning_effort, :string, default: nil

    field :json_response, :boolean, default: false
    field :json_schema, :map, default: nil

    # `%{"type" => "function", "function" => %{"name" => name}}` to force a
    # tool, or `%{"type" => "auto" | "none" | "required"}`.
    field :tool_choice, :map, default: nil
    field :parallel_tool_calls, :boolean, default: nil

    # Provider-specific values merged into the request body last. A `nil`
    # value removes that key from the body. See
    # `LangChain.Utils.merge_extra_body/2` for the merge rules.
    field :extra_body, :map, default: nil

    # Req options to merge into the request, such as extra headers. Refer to
    # `https://hexdocs.pm/req/Req.html#new/1-options`.
    field :req_config, :map, default: %{}

    # Time in milliseconds to wait for a response.
    field :receive_timeout, :integer, default: @receive_timeout

    # Number of retries on closed-connection errors (stale pool). The initial
    # request always runs; this controls additional attempts only.
    field :retry_count, :integer, default: 2

    # A list of maps for callback handlers (treated as internal)
    field :callbacks, {:array, :map}, default: []

    # Outputs the raw request data and Req response, for debugging.
    field :verbose_api, :boolean, default: false
  end

  @type t :: %ChatOpenAICompatible{}

  @create_fields [
    :endpoint,
    :model,
    :api_key,
    :stream,
    :stream_options,
    :temperature,
    :top_p,
    :max_tokens,
    :seed,
    :stop,
    :frequency_penalty,
    :presence_penalty,
    :reasoning_effort,
    :json_response,
    :json_schema,
    :tool_choice,
    :parallel_tool_calls,
    :extra_body,
    :req_config,
    :receive_timeout,
    :retry_count,
    :callbacks,
    :verbose_api
  ]
  @required_fields [:endpoint, :model]

  @doc """
  Setup a ChatOpenAICompatible client configuration.
  """
  @spec new(attrs :: map()) :: {:ok, t} | {:error, Ecto.Changeset.t()}
  def new(%{} = attrs \\ %{}) do
    %ChatOpenAICompatible{}
    |> cast(attrs, @create_fields)
    |> validate_required(@required_fields)
    |> validate_number(:receive_timeout, greater_than_or_equal_to: 0)
    |> validate_number(:retry_count, greater_than_or_equal_to: 0)
    |> validate_number(:max_tokens, greater_than: 0)
    |> apply_action(:insert)
  end

  @doc """
  Setup a ChatOpenAICompatible client configuration and return it or raise an
  error if invalid.
  """
  @spec new!(attrs :: map()) :: t() | no_return()
  def new!(attrs \\ %{}) do
    case new(attrs) do
      {:ok, model} ->
        model

      {:error, changeset} ->
        raise LangChainError, changeset
    end
  end

  @doc """
  Return the params formatted for an API request.
  """
  @spec for_api(t(), [Message.t()], ChatModel.tools()) :: %{atom() => any()}
  def for_api(%ChatOpenAICompatible{} = model, messages, tools) do
    %{
      model: model.model,
      stream: model.stream,
      messages: ChatCompletionsFormat.messages_for_api(messages, system_role: :system)
    }
    |> Utils.conditionally_add_to_map(:temperature, model.temperature)
    |> Utils.conditionally_add_to_map(:top_p, model.top_p)
    |> Utils.conditionally_add_to_map(:max_tokens, model.max_tokens)
    |> Utils.conditionally_add_to_map(:seed, model.seed)
    |> Utils.conditionally_add_to_map(:stop, model.stop)
    |> Utils.conditionally_add_to_map(:frequency_penalty, model.frequency_penalty)
    |> Utils.conditionally_add_to_map(:presence_penalty, model.presence_penalty)
    |> Utils.conditionally_add_to_map(:reasoning_effort, model.reasoning_effort)
    |> Utils.conditionally_add_to_map(:response_format, response_format(model))
    |> Utils.conditionally_add_to_map(:stream_options, stream_options_for_api(model))
    |> Utils.conditionally_add_to_map(:tools, ChatCompletionsFormat.tools_for_api(tools))
    |> Utils.conditionally_add_to_map(:tool_choice, tool_choice_for_api(model))
    |> Utils.conditionally_add_to_map(:parallel_tool_calls, model.parallel_tool_calls)
    |> Utils.merge_extra_body(model.extra_body)
  end

  defp response_format(%ChatOpenAICompatible{json_response: true, json_schema: schema})
       when not is_nil(schema) do
    %{"type" => "json_schema", "json_schema" => schema}
  end

  defp response_format(%ChatOpenAICompatible{json_response: true}), do: %{"type" => "json_object"}
  defp response_format(%ChatOpenAICompatible{}), do: nil

  defp stream_options_for_api(%ChatOpenAICompatible{stream_options: nil}), do: nil

  defp stream_options_for_api(%ChatOpenAICompatible{stream_options: %{} = data}) do
    %{"include_usage" => Map.get(data, :include_usage, Map.get(data, "include_usage"))}
  end

  defp tool_choice_for_api(%ChatOpenAICompatible{
         tool_choice: %{"type" => "function", "function" => %{"name" => name}}
       })
       when is_binary(name) and byte_size(name) > 0,
       do: %{"type" => "function", "function" => %{"name" => name}}

  defp tool_choice_for_api(%ChatOpenAICompatible{tool_choice: %{"type" => type}})
       when is_binary(type) and byte_size(type) > 0,
       do: type

  defp tool_choice_for_api(%ChatOpenAICompatible{}), do: nil

  @doc """
  Calls the service passing the ChatOpenAICompatible struct with
  configuration, plus either a simple message or the list of messages to act
  as the prompt.

  Optionally pass in a list of tools available to the LLM for requesting
  execution in response.

  **NOTE:** This function *can* be used directly, but the primary interface
  should be through `LangChain.Chains.LLMChain`.
  """
  @impl ChatModel
  def call(model, prompt, tools \\ [])

  def call(%ChatOpenAICompatible{} = model, prompt, tools) when is_binary(prompt) do
    messages = [
      Message.new_system!(),
      Message.new_user!(prompt)
    ]

    call(model, messages, tools)
  end

  def call(%ChatOpenAICompatible{} = model, messages, tools) when is_list(messages) do
    metadata = %{
      model: model.model,
      provider: provider(),
      message_count: length(messages),
      tools_count: length(tools)
    }

    ChatModel.llm_telemetry_span(model, metadata, fn ->
      try do
        LangChain.Telemetry.llm_prompt(
          %{system_time: System.system_time()},
          %{model: model.model, messages: messages}
        )

        case do_api_request(model, messages, tools) do
          {:error, reason} ->
            {:error, reason}

          parsed_data ->
            LangChain.Telemetry.llm_response(
              %{system_time: System.system_time()},
              %{model: model.model, response: parsed_data}
            )

            {:ok, parsed_data}
        end
      rescue
        err in LangChainError ->
          {:error, err}
      end
    end)
  end

  # Make the API request. Returns the parsed result, or `{:error, reason}`.
  # When streaming, each parsed chunk also fires the delta callbacks as it
  # arrives. Closed connections from a stale pool are retried up to
  # `retry_count` times.
  @doc false
  @spec do_api_request(t(), [Message.t()], ChatModel.tools(), integer() | nil) ::
          list() | struct() | {:error, LangChainError.t()}
  def do_api_request(model, messages, tools, retry_count \\ nil)

  def do_api_request(_model, _messages, _tools, 0) do
    raise LangChainError, "Retries exceeded. Connection failed."
  end

  def do_api_request(%ChatOpenAICompatible{stream: false} = model, messages, tools, retry_count) do
    retry_count = retry_count || model.retry_count + 1
    raw_data = for_api(model, messages, tools)

    if model.verbose_api do
      IO.inspect(raw_data, label: "RAW DATA BEING SUBMITTED")
    end

    model
    |> base_request(raw_data)
    # Disable Req-level retry so it does not compound with the closed
    # connection retry below.
    |> Req.merge(retry: false)
    |> Req.merge(model.req_config |> Keyword.new())
    |> Req.post()
    |> case do
      {:ok, %Req.Response{body: data} = response} ->
        if model.verbose_api do
          IO.inspect(response, label: "RAW REQ RESPONSE")
        end

        fire_response_callbacks(model, response)

        case response_with_status(do_process_response(data), response.status) do
          {:error, %LangChainError{} = reason} ->
            {:error, reason}

          result ->
            Callbacks.fire(model.callbacks, :on_llm_new_message, [result])

            LangChain.Telemetry.emit_event(
              [:langchain, :llm, :response, :non_streaming],
              %{system_time: System.system_time()},
              %{model: model.model, response_size: byte_size(inspect(result))}
            )

            result
        end

      {:error, %Req.TransportError{reason: :timeout} = err} ->
        {:error,
         LangChainError.exception(type: "timeout", message: "Request timed out", original: err)}

      {:error, %Req.TransportError{reason: :closed}} ->
        Logger.debug(fn -> "Connection closed: retry count = #{inspect(retry_count)}" end)
        do_api_request(model, messages, tools, retry_count - 1)

      other ->
        Logger.warning(fn -> "Unexpected and unhandled API response! #{inspect(other)}" end)

        {:error,
         LangChainError.exception(
           type: "unexpected_response",
           message: "Unexpected response",
           original: other
         )}
    end
  end

  def do_api_request(%ChatOpenAICompatible{stream: true} = model, messages, tools, retry_count) do
    retry_count = retry_count || model.retry_count + 1
    raw_data = for_api(model, messages, tools)

    if model.verbose_api do
      IO.inspect(raw_data, label: "RAW DATA BEING SUBMITTED")
    end

    model
    |> base_request(raw_data)
    |> Req.merge(model.req_config |> Keyword.new())
    |> Req.post(
      into:
        Utils.handle_stream_fn(
          model,
          &ChatCompletionsFormat.decode_stream/1,
          &do_process_response/1
        )
    )
    |> case do
      {:ok, %Req.Response{body: data} = response} ->
        fire_response_callbacks(model, response)
        response_with_status(data, response.status)

      {:error, %LangChainError{} = error} ->
        {:error, error}

      {:error, %Req.TransportError{reason: :timeout} = err} ->
        {:error,
         LangChainError.exception(type: "timeout", message: "Request timed out", original: err)}

      {:error, %Req.TransportError{reason: :closed}} ->
        Logger.debug(fn -> "Connection closed: retry count = #{inspect(retry_count)}" end)
        do_api_request(model, messages, tools, retry_count - 1)

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

  # Only the Bearer token is sent, and only when an `api_key` is set.
  defp base_request(%ChatOpenAICompatible{} = model, raw_data) do
    opts = [url: model.endpoint, json: raw_data, receive_timeout: model.receive_timeout]

    opts =
      case model.api_key do
        key when is_binary(key) and key != "" -> Keyword.put(opts, :auth, {:bearer, key})
        _no_key -> opts
      end

    Req.new(opts)
  end

  defp fire_response_callbacks(model, %Req.Response{headers: headers}) do
    Callbacks.fire(model.callbacks, :on_llm_response_headers, [headers])

    Callbacks.fire(model.callbacks, :on_llm_ratelimit_info, [
      ChatCompletionsFormat.get_ratelimit_info(headers)
    ])
  end

  # A rate-limited request whose body carried no error type of its own is
  # reported as `rate_limit_exceeded`, which `retry_on_fallback?/1` accepts.
  defp response_with_status({:error, %LangChainError{type: nil} = error}, 429),
    do: {:error, %LangChainError{error | type: "rate_limit_exceeded"}}

  defp response_with_status(result, _status), do: result

  # Cloudflare reports failures as `%{"errors" => [%{"message" => ...}]}`
  # rather than the `%{"error" => ...}` shape the shared parser reads.
  @doc false
  def do_process_response(%{"errors" => [%{"message" => message} | _]} = response)
      when is_binary(message) do
    {:error, LangChainError.exception(message: message, original: response)}
  end

  def do_process_response(data), do: ChatCompletionsFormat.process_response(data)

  @impl ChatModel
  def provider, do: "openai_compatible"

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

  `api_key`, `callbacks` and `req_config` are left out. `req_config` can carry
  credentials in its headers.
  """
  @impl ChatModel
  @spec serialize_config(t()) :: %{String.t() => any()}
  def serialize_config(%ChatOpenAICompatible{} = model) do
    Utils.to_serializable_map(
      model,
      [
        :endpoint,
        :model,
        :stream,
        :stream_options,
        :temperature,
        :top_p,
        :max_tokens,
        :seed,
        :stop,
        :frequency_penalty,
        :presence_penalty,
        :reasoning_effort,
        :json_response,
        :json_schema,
        :tool_choice,
        :parallel_tool_calls,
        :extra_body,
        :receive_timeout,
        :retry_count
      ],
      @current_config_version
    )
  end

  @doc """
  Restores the model from the config.
  """
  @impl ChatModel
  def restore_from_map(%{"version" => 1} = data) do
    ChatOpenAICompatible.new(data)
  end
end
