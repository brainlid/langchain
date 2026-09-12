defmodule LangChain.ChatModels.ChatFireworks do
  @moduledoc """
  Represents a chat model served by [Fireworks AI](https://fireworks.ai)
  through its OpenAI-compatible Chat Completions API.

  Fireworks accepts OpenAI-shaped requests, so this module reuses
  `LangChain.ChatModels.ChatOpenAI` for message and tool serialization, for
  stream decoding, and for parsing complete responses. It exists as a separate
  chat model because Fireworks differs from OpenAI in ways that need dedicated
  handling:

  - **Strict request schema.** Fireworks rejects fields it does not recognize,
    so the request carries only documented fields and message-level `name` is
    not sent.
  - **Reasoning controls.** `:reasoning_effort` and `:reasoning_history` are
    sent as fields, and system messages keep the `system` role. The chat
    templates of open models such as GLM have no `developer` role.
  - **Thinking in history.** With `send_reasoning_content: true`, thinking from
    earlier assistant turns is sent back as `reasoning_content`.
  - **Typed errors.** HTTP 429 rate limits and 503 load shedding become typed
    errors that `retry_on_fallback?/1` accepts, so `LangChain.Chains.LLMChain`
    fallbacks can take over.
  - **Stream positions.** Thinking merges at content position 0 and answer text
    at position 1, whether or not a chunk carries a reasoning field.

  ## Tested Models

  | Model ID | Notes |
  | -------- | ----- |
  | `accounts/fireworks/models/glm-5p3-flash` | Z.ai GLM-5.3-Flash. Always thinks. It honors `:reasoning_effort` values `"low"`, `"high"` and `"max"`, and treats any other value as `"max"`. `"high"` is a good default for agent work. Accepts images. |

  ## Authentication

  Pass `:api_key`, or set it once in application config:

      config :langchain, :fireworks_api_key, System.fetch_env!("FIREWORKS_API_KEY")

  ## Usage

      ChatFireworks.new!(%{
        model: "accounts/fireworks/models/glm-5p3-flash",
        stream: true,
        reasoning_effort: "high"
      })

  ## US-only Serverless

  Fireworks serves some models only from US data centers through a separate
  host and a router model ID:

      ChatFireworks.new!(%{
        endpoint: "https://us.api.fireworks.ai/inference/v1/chat/completions",
        model: "accounts/fireworks/routers/glm-5p3-flash-us"
      })

  See [US-only Serverless](https://docs.fireworks.ai/serverless/us-only-serverless)
  for the model list and pricing.

  ## Reasoning

  A model's thinking becomes a `LangChain.Message.ContentPart` of type
  `:thinking`, ordered ahead of the answer text. Fireworks returns it in a
  `reasoning_content` field. A `reasoning` field is read as well.

  Thinking is not sent back to the model by default. Fireworks asks multi-turn
  tool-calling agents to include it, so the model can build on its earlier
  reasoning between tool calls. Turn it on with `send_reasoning_content: true`,
  and choose how much of it the server keeps with `:reasoning_history`:

  - `"interleaved"` keeps thinking from the current tool loop and drops it from
    turns before the last user message.
  - `"preserved"` keeps all of it.
  - `"disabled"` drops all of it.

  The setting applies to every assistant turn in a request, so the prompt prefix
  stays the same from one turn to the next and prompt caching keeps working.

  ## Prompt Caching

  Fireworks caches prompt prefixes automatically. Set `:prompt_cache_key` to a
  stable value per conversation to route related requests to the same cache.

  ## Token Limits

  `:max_tokens` defaults to 32,768. Thinking counts against it, and a small limit
  can be spent entirely on thinking before any answer is written. It is sent as
  `max_tokens`. Fireworks treats `max_completion_tokens` as an alias and rejects
  a request that carries both.

  ## Provider-Specific Parameters

  Fireworks accepts parameters this module has no field for, such as `top_k`,
  `min_p` or `context_length_exceeded_behavior`. `extra_body` is a map merged
  into the request body last:

      ChatFireworks.new!(%{
        model: "accounts/fireworks/models/glm-5p3-flash",
        extra_body: %{"context_length_exceeded_behavior" => "error"}
      })

  A `nil` value removes that key from the body. Fireworks rejects fields it does
  not recognize, so only send ones it documents. See
  `LangChain.Utils.merge_extra_body/2` for the merge rules.

  ## Data Retention

  Fireworks does not store prompts or outputs of open models sent through Chat
  Completions unless the account opts in to logging. This module only calls
  Chat Completions. Fireworks' separate Responses API stores conversations by
  default.
  """
  use Ecto.Schema
  require Logger
  import Ecto.Changeset
  alias __MODULE__
  alias LangChain.Callbacks
  alias LangChain.ChatModels.ChatModel
  alias LangChain.ChatModels.ChatOpenAI
  alias LangChain.Config
  alias LangChain.LangChainError
  alias LangChain.Message
  alias LangChain.Message.ContentPart
  alias LangChain.MessageDelta
  alias LangChain.TokenUsage
  alias LangChain.Utils

  @behaviour ChatModel

  @current_config_version 1

  @default_endpoint "https://api.fireworks.ai/inference/v1/chat/completions"

  # Reasoning models can think for minutes on a hard prompt. Fireworks
  # recommends client timeouts of 5 to 30 minutes for agentic work.
  @default_receive_timeout 300_000

  # Thinking counts against the output limit, and a small limit can be spent on
  # thinking before any answer is written.
  @default_max_tokens 32_768

  @valid_reasoning_histories ~w(disabled interleaved preserved)

  @primary_key false
  embedded_schema do
    field :endpoint, :string, default: @default_endpoint
    field :model, :string

    # Falls back to the `:fireworks_api_key` application config when not set.
    field :api_key, :string, redact: true

    # Left unset by default so Fireworks applies the model's own sampling
    # defaults.
    field :temperature, :float
    field :top_p, :float
    field :max_tokens, :integer, default: @default_max_tokens
    field :stream, :boolean, default: false

    # Sent as given. Fireworks accepts "none", "low", "medium", "high", "xhigh"
    # and "max", and each model honors its own subset.
    field :reasoning_effort, :string

    # How much earlier thinking Fireworks keeps in the prompt: "disabled",
    # "interleaved" or "preserved". Unset uses the model's default.
    field :reasoning_history, :string

    # Send thinking from earlier assistant turns back as `reasoning_content`.
    field :send_reasoning_content, :boolean, default: false

    # Requests that share a key are routed to the same prompt cache.
    field :prompt_cache_key, :string
    field :user, :string

    # Tool choice option, in ChatOpenAI's shape.
    field :tool_choice, :map
    field :parallel_tool_calls, :boolean

    field :json_response, :boolean, default: false
    field :json_schema, :map

    # Provider-specific values merged into the request body last. A `nil` value
    # removes that key. See `LangChain.Utils.merge_extra_body/2`.
    field :extra_body, :map, default: nil

    field :receive_timeout, :integer, default: @default_receive_timeout

    # Number of retries on closed-connection errors (stale pool). Other errors
    # return right away so a fallback or the caller can decide what to do.
    field :retry_count, :integer, default: 2

    # A list of maps for callback handlers (treated as internal)
    field :callbacks, {:array, :map}, default: []

    # Prints the raw request and response when true.
    field :verbose_api, :boolean, default: false

    # Req options to merge into the request.
    field :req_config, :map, default: %{}
  end

  @type t :: %ChatFireworks{}

  @create_fields [
    :endpoint,
    :model,
    :api_key,
    :temperature,
    :top_p,
    :max_tokens,
    :stream,
    :reasoning_effort,
    :reasoning_history,
    :send_reasoning_content,
    :prompt_cache_key,
    :user,
    :tool_choice,
    :parallel_tool_calls,
    :json_response,
    :json_schema,
    :extra_body,
    :receive_timeout,
    :retry_count,
    :callbacks,
    :verbose_api,
    :req_config
  ]
  @required_fields [:endpoint, :model]

  @doc """
  Setup a ChatFireworks client configuration.
  """
  @spec new(attrs :: map()) :: {:ok, t()} | {:error, Ecto.Changeset.t()}
  def new(%{} = attrs \\ %{}) do
    %ChatFireworks{}
    |> cast(attrs, @create_fields)
    |> common_validation()
    |> apply_action(:insert)
  end

  @doc """
  Setup a ChatFireworks client configuration and return it or raise an error if
  invalid.
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

  defp common_validation(changeset) do
    changeset
    |> validate_required(@required_fields)
    |> validate_inclusion(:reasoning_history, @valid_reasoning_histories,
      message: "must be one of: #{Enum.join(@valid_reasoning_histories, ", ")}"
    )
    |> validate_number(:temperature, greater_than_or_equal_to: 0, less_than_or_equal_to: 2)
    |> validate_number(:top_p, greater_than_or_equal_to: 0, less_than_or_equal_to: 1)
    |> validate_number(:max_tokens, greater_than: 0)
    |> validate_number(:receive_timeout, greater_than_or_equal_to: 0)
    |> validate_number(:retry_count, greater_than_or_equal_to: 0)
  end

  @doc """
  Return the request body for the Fireworks Chat Completions API.

  Only fields Fireworks documents are included, because it rejects unknown ones.
  Messages and tools are serialized by `ChatOpenAI` and then adjusted for
  Fireworks. `extra_body` is merged in last.
  """
  @spec for_api(t(), [Message.t()], ChatModel.tools()) :: %{(atom() | String.t()) => any()}
  def for_api(%ChatFireworks{} = model, messages, tools) do
    %{
      model: model.model,
      stream: model.stream,
      messages: Enum.flat_map(messages, &message_for_api(model, &1))
    }
    |> Utils.conditionally_add_to_map(:temperature, model.temperature)
    |> Utils.conditionally_add_to_map(:top_p, model.top_p)
    |> Utils.conditionally_add_to_map(:max_tokens, model.max_tokens)
    |> Utils.conditionally_add_to_map(:reasoning_effort, model.reasoning_effort)
    |> Utils.conditionally_add_to_map(:reasoning_history, model.reasoning_history)
    |> Utils.conditionally_add_to_map(:prompt_cache_key, model.prompt_cache_key)
    |> Utils.conditionally_add_to_map(:user, model.user)
    |> Utils.conditionally_add_to_map(:response_format, response_format(model))
    |> Utils.conditionally_add_to_map(:tools, tools_for_api(model, tools))
    |> Utils.conditionally_add_to_map(:tool_choice, tool_choice_for_api(model))
    |> Utils.conditionally_add_to_map(:parallel_tool_calls, model.parallel_tool_calls)
    |> Utils.merge_extra_body(model.extra_body)
  end

  # ChatOpenAI serializes a message into one map, or into several for a message
  # carrying multiple tool results.
  defp message_for_api(%ChatFireworks{} = model, %Message{} = message) do
    model
    |> ChatOpenAI.for_api(message)
    |> List.wrap()
    |> Enum.map(&adjust_message(&1, message, model))
  end

  # Fireworks' message schema has no `name` field.
  defp adjust_message(data, %Message{role: :assistant} = message, model) do
    data
    |> Map.delete("name")
    |> normalize_empty_content()
    |> maybe_put_reasoning_content(message, model)
  end

  defp adjust_message(data, _message, _model), do: Map.delete(data, "name")

  # An assistant turn that held only thinking serializes to an empty content
  # list once the thinking is dropped. Fireworks documents `null` content for a
  # turn that calls tools, and takes a string otherwise.
  defp normalize_empty_content(%{"content" => []} = data) do
    if Map.has_key?(data, "tool_calls") do
      Map.put(data, "content", nil)
    else
      Map.put(data, "content", "")
    end
  end

  defp normalize_empty_content(data), do: data

  defp maybe_put_reasoning_content(data, message, %ChatFireworks{send_reasoning_content: true}) do
    case thinking_text(message) do
      "" -> data
      thinking -> Map.put(data, "reasoning_content", thinking)
    end
  end

  defp maybe_put_reasoning_content(data, _message, _model), do: data

  defp thinking_text(%Message{content: parts}) when is_list(parts) do
    parts
    |> Enum.filter(&(&1.type == :thinking))
    |> Enum.map_join("", &(&1.content || ""))
  end

  defp thinking_text(_message), do: ""

  defp response_format(%ChatFireworks{json_response: true, json_schema: schema})
       when not is_nil(schema) do
    %{"type" => "json_schema", "json_schema" => schema}
  end

  defp response_format(%ChatFireworks{json_response: true}), do: %{"type" => "json_object"}
  defp response_format(%ChatFireworks{}), do: nil

  defp tools_for_api(_model, nil), do: []
  defp tools_for_api(_model, []), do: []

  defp tools_for_api(%ChatFireworks{} = model, tools) do
    Enum.map(tools, fn %LangChain.Function{} = function ->
      %{"type" => "function", "function" => ChatOpenAI.for_api(model, function)}
    end)
  end

  defp tool_choice_for_api(%ChatFireworks{
         tool_choice: %{"type" => "function", "function" => %{"name" => name}}
       })
       when is_binary(name) and byte_size(name) > 0,
       do: %{"type" => "function", "function" => %{"name" => name}}

  defp tool_choice_for_api(%ChatFireworks{tool_choice: %{"type" => type}})
       when is_binary(type) and byte_size(type) > 0,
       do: type

  defp tool_choice_for_api(%ChatFireworks{}), do: nil

  @doc """
  Calls the Fireworks API passing the ChatFireworks struct with configuration,
  plus either a simple message or the list of messages to act as the prompt.

  Returns `{:ok, [%Message{}]}` for a complete response, the list of streamed
  deltas when streaming, or `{:error, %LangChainError{}}` on failure.
  """
  @impl ChatModel
  def call(model, prompt, tools \\ [])

  def call(%ChatFireworks{} = model, prompt, tools) when is_binary(prompt) do
    call(model, [Message.new_user!(prompt)], tools)
  end

  def call(%ChatFireworks{} = model, messages, tools) when is_list(messages) do
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
          {:error, %LangChainError{} = error} ->
            {:error, error}

          parsed ->
            LangChain.Telemetry.llm_response(
              %{system_time: System.system_time()},
              %{model: model.model, response: parsed}
            )

            {:ok, parsed}
        end
      rescue
        err in LangChainError ->
          {:error, err}
      end
    end)
  end

  @impl ChatModel
  def provider, do: "fireworks"

  @doc """
  Determine if an error should be retried with a fallback model. Rate limits,
  load shedding, server errors, timeouts and dropped connections are service
  problems that another model or host may not have. Request, authentication and
  not-found errors would fail the same way anywhere.
  """
  @impl ChatModel
  @spec retry_on_fallback?(LangChainError.t()) :: boolean()
  def retry_on_fallback?(%LangChainError{type: type})
      when type in ["rate_limit_exceeded", "overloaded", "server_error", "timeout", "connection"],
      do: true

  def retry_on_fallback?(_), do: false

  @doc """
  Generate a config map that can later restore the model's configuration. The
  API key is not included.
  """
  @impl ChatModel
  @spec serialize_config(t()) :: %{String.t() => any()}
  def serialize_config(%ChatFireworks{} = model) do
    Utils.to_serializable_map(
      model,
      [
        :endpoint,
        :model,
        :temperature,
        :top_p,
        :max_tokens,
        :stream,
        :reasoning_effort,
        :reasoning_history,
        :send_reasoning_content,
        :prompt_cache_key,
        :user,
        :tool_choice,
        :parallel_tool_calls,
        :json_response,
        :json_schema,
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
    ChatFireworks.new(data)
  end

  @doc false
  @spec do_api_request(t(), [Message.t()], ChatModel.tools(), integer() | nil) ::
          list() | {:error, LangChainError.t()}
  def do_api_request(model, messages, tools, retry_count \\ nil)

  def do_api_request(_model, _messages, _tools, 0) do
    {:error,
     LangChainError.exception(
       type: "connection",
       message: "Retries exceeded. Connection failed."
     )}
  end

  def do_api_request(%ChatFireworks{stream: false} = model, messages, tools, retry_count) do
    retry_count = retry_count || model.retry_count + 1
    body = for_api(model, messages, tools)

    if model.verbose_api do
      IO.inspect(body, label: "RAW DATA BEING SUBMITTED (Fireworks)")
    end

    model
    |> build_request(body)
    |> Req.post()
    |> case do
      {:ok, %Req.Response{status: status, body: data} = response} when status in 200..299 ->
        if model.verbose_api do
          IO.inspect(response, label: "RAW REQ RESPONSE (Fireworks)")
        end

        Callbacks.fire(model.callbacks, :on_llm_response_headers, [response.headers])
        do_process_response(model, data)

      {:ok, %Req.Response{status: status, body: data}} ->
        {:error, error_from_response(status, data)}

      {:error, %Req.TransportError{reason: :closed}} ->
        Logger.debug(fn -> "Connection closed: retry count = #{inspect(retry_count)}" end)
        do_api_request(model, messages, tools, retry_count - 1)

      {:error, error} ->
        {:error, transport_error(error)}
    end
  end

  def do_api_request(%ChatFireworks{stream: true} = model, messages, tools, retry_count) do
    retry_count = retry_count || model.retry_count + 1
    body = for_api(model, messages, tools)

    if model.verbose_api do
      IO.inspect(body, label: "RAW DATA BEING SUBMITTED (Fireworks stream)")
    end

    model
    |> build_request(body)
    |> Req.post(
      into:
        Utils.handle_stream_fn(
          model,
          &ChatOpenAI.decode_stream/1,
          &do_process_response(model, &1)
        )
    )
    |> case do
      {:ok, %Req.Response{status: status, body: data} = response} when status in 200..299 ->
        Callbacks.fire(model.callbacks, :on_llm_response_headers, [response.headers])
        data

      {:ok, %Req.Response{} = response} ->
        {:error, streamed_error(response)}

      {:error, %LangChainError{} = error} ->
        {:error, error}

      {:error, %Req.TransportError{reason: :closed}} ->
        Logger.debug(fn -> "Connection closed: retry count = #{inspect(retry_count)}" end)
        do_api_request(model, messages, tools, retry_count - 1)

      {:error, error} ->
        {:error, transport_error(error)}
    end
  end

  defp build_request(%ChatFireworks{} = model, body) do
    Req.new(
      url: model.endpoint,
      json: body,
      auth: {:bearer, get_api_key(model)},
      receive_timeout: model.receive_timeout,
      # LangChain decides on retries and fallbacks. Req-level retries would
      # compound with them. See https://github.com/brainlid/langchain/issues/503
      retry: false
    )
    |> Req.merge(Keyword.new(model.req_config))
  end

  defp transport_error(%Req.TransportError{reason: :timeout} = error) do
    LangChainError.exception(type: "timeout", message: "Request timed out", original: error)
  end

  defp transport_error(%Req.TransportError{reason: reason} = error) do
    LangChainError.exception(
      type: "connection",
      message: "Fireworks connection error: #{inspect(reason)}",
      original: error
    )
  end

  defp transport_error(other) do
    LangChainError.exception(
      type: "unexpected_response",
      message: "Unexpected response: #{inspect(other)}",
      original: other
    )
  end

  # A failed streamed request is decoded by `Utils.handle_stream_fn/3`. A JSON
  # error body goes through `do_process_response/2`, and anything else stays
  # buffered. Either way the HTTP status decides the error type.
  defp streamed_error(%Req.Response{
         status: status,
         body: {:error, %LangChainError{original: original}}
       }) do
    error_from_response(status, original)
  end

  defp streamed_error(%Req.Response{status: status} = response) do
    error_from_response(status, Req.Response.get_private(response, :error_buffer, response.body))
  end

  @doc false
  @spec error_from_response(integer(), any()) :: LangChainError.t()
  def error_from_response(status, body) do
    LangChainError.exception(
      type: type_for_status(status),
      message: "Fireworks returned HTTP #{status}: #{error_message(body)}",
      original: body
    )
  end

  @doc false
  @spec type_for_status(integer()) :: String.t()
  def type_for_status(status) when status in [400, 422], do: "invalid_request"
  def type_for_status(status) when status in [401, 403], do: "authentication_error"
  def type_for_status(404), do: "not_found"
  def type_for_status(status) when status in [408, 504], do: "timeout"
  def type_for_status(413), do: "request_too_large"
  def type_for_status(429), do: "rate_limit_exceeded"
  def type_for_status(status) when status in [500, 502], do: "server_error"
  def type_for_status(503), do: "overloaded"
  def type_for_status(_status), do: "api_error"

  defp error_message(%{"error" => %{"message" => message}}) when is_binary(message), do: message
  defp error_message(%{"error" => message}) when is_binary(message), do: message
  defp error_message(%{"detail" => detail}) when is_binary(detail), do: detail

  # Fireworks' validation errors list each problem with its location.
  defp error_message(%{"detail" => details}) when is_list(details) do
    Enum.map_join(details, "; ", &detail_message/1)
  end

  defp error_message(body) when is_binary(body) and body != "", do: body
  defp error_message(body), do: inspect(body)

  defp detail_message(%{"msg" => msg, "loc" => loc}) when is_list(loc) do
    "#{Enum.join(loc, ".")}: #{msg}"
  end

  defp detail_message(%{"msg" => msg}), do: msg
  defp detail_message(other), do: inspect(other)

  @doc false
  @spec do_process_response(t(), data :: any()) ::
          :skip
          | TokenUsage.t()
          | [Message.t() | MessageDelta.t() | {:error, LangChainError.t()}]
          | {:error, LangChainError.t()}
  # A streamed chunk. Each choice carries a "delta".
  def do_process_response(
        %ChatFireworks{} = model,
        %{"choices" => [%{"delta" => _} | _] = choices} = data
      ) do
    deltas = Enum.flat_map(choices, &stream_choice_deltas(model, &1))

    case {deltas, get_token_usage(data)} do
      {[], nil} -> :skip
      {[], usage} -> usage
      {deltas, nil} -> deltas
      {deltas, usage} -> set_usage_on_last(deltas, usage)
    end
  end

  # The last chunk of a streamed response, when usage is reported on its own.
  def do_process_response(_model, %{"choices" => [], "usage" => usage} = data)
      when is_map(usage) do
    get_token_usage(data)
  end

  def do_process_response(_model, %{"choices" => []}), do: :skip

  # A complete response.
  def do_process_response(%ChatFireworks{} = model, %{"choices" => choices} = data)
      when is_list(choices) do
    usage = get_token_usage(data)

    Enum.map(choices, fn choice ->
      model
      |> ChatOpenAI.do_process_response(normalize_reasoning(choice))
      |> attach_usage(usage)
    end)
  end

  def do_process_response(_model, %{"error" => _} = body) do
    {:error,
     LangChainError.exception(type: "api_error", message: error_message(body), original: body)}
  end

  def do_process_response(_model, %{"detail" => _} = body) do
    {:error,
     LangChainError.exception(
       type: "invalid_request",
       message: error_message(body),
       original: body
     )}
  end

  def do_process_response(_model, other) do
    {:error,
     LangChainError.exception(
       type: "unexpected_response",
       message: "Unexpected response: #{inspect(other)}",
       original: other
     )}
  end

  # Some OpenAI-compatible servers name the thinking field `reasoning`. It is
  # read when `reasoning_content` is absent, so ChatOpenAI's parsing picks it up.
  defp normalize_reasoning(%{"message" => %{"reasoning" => reasoning} = message} = choice)
       when is_binary(reasoning) and reasoning != "" do
    case message do
      %{"reasoning_content" => existing} when is_binary(existing) and existing != "" ->
        choice

      _no_reasoning_content ->
        message = message |> Map.delete("reasoning") |> Map.put("reasoning_content", reasoning)
        Map.put(choice, "message", message)
    end
  end

  defp normalize_reasoning(choice), do: choice

  defp attach_usage(%Message{} = message, %TokenUsage{} = usage),
    do: TokenUsage.set(message, usage)

  defp attach_usage(result, _usage), do: result

  # Streamed thinking merges at content position 0 and answer text at position
  # 1, so the assembled message reads [thinking, text]. Text is placed at
  # position 1 even when a chunk has no reasoning field, which keeps it from
  # landing on the thinking part. Tool calls merge by their own index.
  #
  # `index` is repurposed from the choice index, so asking for several choices
  # would merge them together. This module never sends `n`.
  defp stream_choice_deltas(model, %{"delta" => delta} = choice) do
    role = delta_role(delta)
    status = finish_reason_to_status(Map.get(choice, "finish_reason"))

    []
    |> maybe_thinking_delta(delta_reasoning(delta), role, status)
    |> maybe_text_delta(Map.get(delta, "content"), role, status)
    |> maybe_tool_calls_delta(model, Map.get(delta, "tool_calls"), role, status)
    |> case do
      [] ->
        # A role-only opening chunk, or a terminal chunk with no payload.
        if role == :assistant or status != :incomplete do
          [build_delta(%{role: role, status: status, index: 0})]
        else
          []
        end

      deltas ->
        deltas
    end
  end

  defp stream_choice_deltas(_model, _choice), do: []

  defp delta_reasoning(%{"reasoning_content" => text}) when is_binary(text) and text != "",
    do: text

  defp delta_reasoning(%{"reasoning" => text}) when is_binary(text) and text != "", do: text
  defp delta_reasoning(_delta), do: nil

  defp maybe_thinking_delta(acc, nil, _role, _status), do: acc

  defp maybe_thinking_delta(acc, text, role, status) do
    acc ++
      [build_delta(%{content: ContentPart.thinking!(text), role: role, status: status, index: 0})]
  end

  defp maybe_text_delta(acc, text, role, status) when is_binary(text) and text != "" do
    acc ++ [build_delta(%{content: text, role: role, status: status, index: 1})]
  end

  defp maybe_text_delta(acc, _text, _role, _status), do: acc

  defp maybe_tool_calls_delta(acc, model, [_ | _] = raw_calls, role, status) do
    tool_calls = Enum.map(raw_calls, &ChatOpenAI.do_process_response(model, &1))
    acc ++ [build_delta(%{tool_calls: tool_calls, role: role, status: status, index: 0})]
  end

  defp maybe_tool_calls_delta(acc, _model, _raw_calls, _role, _status), do: acc

  defp build_delta(attrs) do
    attrs =
      attrs
      |> Map.put_new(:role, :unknown)
      |> Map.put_new(:status, :incomplete)

    struct!(MessageDelta, attrs)
  end

  defp delta_role(%{"role" => "assistant"}), do: :assistant
  defp delta_role(_delta), do: :unknown

  defp finish_reason_to_status(nil), do: :incomplete
  defp finish_reason_to_status("length"), do: :length
  defp finish_reason_to_status("max_tokens"), do: :length
  defp finish_reason_to_status("content_filter"), do: :content_filtered
  # "stop", "tool_calls", "function_call", and any other reason a choice finished
  # with.
  defp finish_reason_to_status(_finished), do: :complete

  # Usage arrives once per response. Attach it to one delta so it is counted
  # once.
  defp set_usage_on_last(deltas, usage) do
    {earlier, [last]} = Enum.split(deltas, -1)
    earlier ++ [TokenUsage.set(last, usage)]
  end

  defp get_token_usage(%{"usage" => usage}) when is_map(usage) do
    TokenUsage.new!(%{
      input: Map.get(usage, "prompt_tokens"),
      output: Map.get(usage, "completion_tokens"),
      raw: usage
    })
  end

  defp get_token_usage(_data), do: nil

  defp get_api_key(%ChatFireworks{api_key: api_key}) when is_binary(api_key), do: api_key
  defp get_api_key(%ChatFireworks{}), do: Config.resolve(:fireworks_api_key, "")
end
