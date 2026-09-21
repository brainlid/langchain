defmodule LangChain.ChatModels.ChatLiteLLM do
  @moduledoc """
  Represents the [LiteLLM](https://docs.litellm.ai/) AI gateway.

  LiteLLM is an open source gateway that exposes a single OpenAI-compatible API
  in front of 100+ model providers (OpenAI, Anthropic, Google, Azure, Bedrock,
  Vertex, Mistral, Groq, Ollama, and more). Running it in front of your models
  gives you one endpoint, one key, and centralized cost tracking, budgets,
  rate limiting, fallbacks, and load balancing.

  Because the gateway speaks the OpenAI chat-completions wire format, this
  module follows the same shape as the other OpenAI-compatible chat models in
  this library, while defaulting to a locally running gateway and treating the
  model name as an opaque routing string.

  ## Why a dedicated module

  A LiteLLM gateway can be reached by pointing `ChatOpenAI` at a different
  `:endpoint`. This module exists for the same reason `ChatDeepSeek`,
  `ChatGrok`, `ChatPerplexity`, and `ChatOrq` do: the defaults, the
  configuration key, the model discovery, and the response quirks are
  gateway-specific, and hiding them behind an OpenAI struct makes them the
  caller's problem. Concretely, it provides:

  - a default endpoint pointing at a local gateway (`http://localhost:4000`)
  - a dedicated `:litellm_key` config key, and an **optional** API key, since a
    gateway run without a master key accepts unauthenticated local requests
  - `list_models/1` for discovering whatever the gateway is actually serving
  - tolerance for the gateway's streaming shape, where `finish_reason` and
    `content` are frequently absent on a chunk, and where the terminal
    usage-bearing chunk arrives as a populated `choices` array holding an empty
    delta rather than OpenAI's empty `choices` array

  ## Configuration

  The API key is optional. When set, it is the gateway's master key or a
  virtual key, **not** an upstream provider key: upstream credentials live in
  the gateway's own configuration, server-side.

      config :langchain, :litellm_key, System.get_env("LITELLM_API_KEY")

  Or pass it per model with `%{api_key: "sk-..."}`.

  ## Model names

  The `:model` value is passed through to the gateway untouched and can be
  anything the gateway routes, including a provider-prefixed name such as
  `"anthropic/claude-sonnet-4-5"`, a `"litellm_proxy/..."` name, or a plain
  alias configured in the gateway's `model_list`.

  ## Examples

  Basic call against a local gateway:

      {:ok, chat} = ChatLiteLLM.new(%{model: "gpt-4o-mini"})
      {:ok, updated_chain} =
        %{llm: chat}
        |> LLMChain.new!()
        |> LLMChain.add_message(Message.new_user!("Why is the sky blue?"))
        |> LLMChain.run()

  Pointing at a remote gateway with a virtual key, and streaming:

      {:ok, chat} =
        ChatLiteLLM.new(%{
          endpoint: "https://litellm.internal.example.com/v1/chat/completions",
          api_key: System.get_env("LITELLM_API_KEY"),
          model: "anthropic/claude-sonnet-4-5",
          stream: true
        })

  Discovering what the gateway serves:

      {:ok, chat} = ChatLiteLLM.new(%{model: "gpt-4o-mini"})
      {:ok, model_ids} = ChatLiteLLM.list_models(chat)
      # => {:ok, ["gpt-4o-mini", "anthropic/claude-sonnet-4-5", ...]}

  """
  use Ecto.Schema
  require Logger
  import Ecto.Changeset
  alias __MODULE__
  alias LangChain.Config
  alias LangChain.ChatModels.ChatModel
  alias LangChain.PromptTemplate
  alias LangChain.Message
  alias LangChain.Message.ContentPart
  alias LangChain.Message.ToolCall
  alias LangChain.Message.ToolResult
  alias LangChain.TokenUsage
  alias LangChain.Function
  alias LangChain.FunctionParam
  alias LangChain.LangChainError
  alias LangChain.Utils
  alias LangChain.MessageDelta
  alias LangChain.Callbacks

  @behaviour ChatModel

  @current_config_version 1

  # The default gateway endpoint. LiteLLM's `litellm --port 4000` and the
  # published docker-compose both listen here.
  @default_endpoint "http://localhost:4000/v1/chat/completions"

  @receive_timeout 60_000

  @primary_key false
  embedded_schema do
    field :endpoint, :string, default: @default_endpoint
    # The model name routed by the gateway. Opaque: it may be a plain name, a
    # provider-prefixed name, or a gateway alias.
    field :model, :string

    # The gateway master key or virtual key. Optional: a gateway started
    # without a master key serves unauthenticated requests.
    field :api_key, :string, redact: true

    field :temperature, :float, default: nil
    field :frequency_penalty, :float, default: nil
    field :presence_penalty, :float, default: nil
    field :top_p, :float, default: nil
    field :receive_timeout, :integer, default: @receive_timeout
    field :seed, :integer
    field :n, :integer, default: 1
    field :json_response, :boolean, default: false
    field :json_schema, :map, default: nil
    field :stream, :boolean, default: false
    field :max_tokens, :integer, default: nil
    field :stream_options, :map, default: nil
    field :tool_choice, :map
    field :parallel_tool_calls, :boolean
    field :user, :string

    # Extra body params passed through to the gateway untouched. Useful for
    # gateway-specific routing controls such as `fallbacks`, `metadata`, or
    # `tags` that have no dedicated field here.
    field :extra_body, :map, default: nil

    field :callbacks, {:array, :map}, default: []
    field :verbose_api, :boolean, default: false
    field :retry_count, :integer, default: 2
    field :req_config, :map, default: %{}
  end

  @type t :: %ChatLiteLLM{}

  @create_fields [
    :endpoint,
    :model,
    :api_key,
    :temperature,
    :frequency_penalty,
    :presence_penalty,
    :top_p,
    :receive_timeout,
    :seed,
    :n,
    :json_response,
    :json_schema,
    :stream,
    :max_tokens,
    :stream_options,
    :tool_choice,
    :parallel_tool_calls,
    :user,
    :extra_body,
    :verbose_api,
    :retry_count,
    :req_config,
    :callbacks
  ]
  @required_fields [:endpoint, :model]

  # The gateway key is optional, so this can legitimately return "". A gateway
  # started without a master key accepts the request anyway; one started with a
  # master key answers 401, which is surfaced as an authentication error.
  @spec get_api_key(t()) :: String.t()
  defp get_api_key(%ChatLiteLLM{api_key: api_key}) do
    api_key || Config.resolve(:litellm_key, "")
  end

  @doc """
  Setup a ChatLiteLLM client configuration.
  """
  @spec new(attrs :: map()) :: {:ok, t} | {:error, Ecto.Changeset.t()}
  def new(%{} = attrs \\ %{}) do
    %ChatLiteLLM{}
    |> cast(attrs, @create_fields)
    |> common_validation()
    |> apply_action(:insert)
  end

  @doc """
  Setup a ChatLiteLLM client configuration and return it or raise an error if
  invalid.
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
    |> validate_number(:n, greater_than_or_equal_to: 1)
    |> validate_number(:receive_timeout, greater_than_or_equal_to: 0)
    |> validate_endpoint_url()
  end

  defp validate_endpoint_url(changeset) do
    case get_field(changeset, :endpoint) do
      nil ->
        changeset

      url when is_binary(url) ->
        case URI.parse(url) do
          %URI{scheme: scheme, host: host}
          when scheme in ["http", "https"] and is_binary(host) and host != "" ->
            changeset

          _ ->
            add_error(changeset, :endpoint, "must be a valid http or https URL")
        end

      _ ->
        add_error(changeset, :endpoint, "must be a valid http or https URL")
    end
  end

  @doc """
  Return the params formatted for an API request.
  """
  @spec for_api(t | Message.t() | Function.t(), message :: [map()], ChatModel.tools()) :: %{
          atom() => any()
        }
  def for_api(%ChatLiteLLM{} = litellm, messages, tools) do
    %{
      model: litellm.model,
      n: litellm.n,
      stream: litellm.stream,
      messages:
        messages
        |> Enum.flat_map(fn m ->
          case for_api(litellm, m) do
            %{} = data ->
              [data]

            data when is_list(data) ->
              data
          end
        end)
    }
    |> Utils.conditionally_add_to_map(:temperature, litellm.temperature)
    |> Utils.conditionally_add_to_map(:top_p, litellm.top_p)
    |> Utils.conditionally_add_to_map(:user, litellm.user)
    |> Utils.conditionally_add_to_map(:frequency_penalty, litellm.frequency_penalty)
    |> Utils.conditionally_add_to_map(:presence_penalty, litellm.presence_penalty)
    |> Utils.conditionally_add_to_map(:response_format, set_response_format(litellm))
    |> Utils.conditionally_add_to_map(:max_tokens, litellm.max_tokens)
    |> Utils.conditionally_add_to_map(:seed, litellm.seed)
    |> Utils.conditionally_add_to_map(
      :stream_options,
      get_stream_options_for_api(litellm.stream_options)
    )
    |> Utils.conditionally_add_to_map(:tools, get_tools_for_api(litellm, tools))
    |> Utils.conditionally_add_to_map(:tool_choice, get_tool_choice(litellm))
    |> Utils.conditionally_add_to_map(:parallel_tool_calls, litellm.parallel_tool_calls)
    |> merge_extra_body(litellm.extra_body)
  end

  # Gateway-specific routing controls (`fallbacks`, `tags`, `metadata`, ...) are
  # merged last so they can also override a generated key when deliberately set.
  defp merge_extra_body(data, nil), do: data

  defp merge_extra_body(data, %{} = extra) when map_size(extra) == 0, do: data

  defp merge_extra_body(data, %{} = extra), do: Map.merge(data, extra)

  defp get_tools_for_api(%_{} = _model, nil), do: []

  defp get_tools_for_api(%_{} = model, tools) do
    Enum.map(tools, fn
      %Function{} = function ->
        %{"type" => "function", "function" => for_api(model, function)}
    end)
  end

  defp get_stream_options_for_api(nil), do: nil

  defp get_stream_options_for_api(%{} = data) do
    %{"include_usage" => Map.get(data, :include_usage, Map.get(data, "include_usage"))}
  end

  defp set_response_format(%ChatLiteLLM{json_response: true, json_schema: json_schema})
       when not is_nil(json_schema) do
    %{
      "type" => "json_schema",
      "json_schema" => json_schema
    }
  end

  defp set_response_format(%ChatLiteLLM{json_response: true}) do
    %{"type" => "json_object"}
  end

  defp set_response_format(%ChatLiteLLM{json_response: false}) do
    nil
  end

  defp get_tool_choice(%ChatLiteLLM{
         tool_choice: %{"type" => "function", "function" => %{"name" => name}} = _tool_choice
       })
       when is_binary(name) and byte_size(name) > 0,
       do: %{"type" => "function", "function" => %{"name" => name}}

  defp get_tool_choice(%ChatLiteLLM{tool_choice: %{"type" => type} = _tool_choice})
       when is_binary(type) and byte_size(type) > 0,
       do: type

  defp get_tool_choice(%ChatLiteLLM{}), do: nil

  @doc """
  Convert a LangChain structure to the expected map of data for the API.
  """
  @spec for_api(
          struct(),
          Message.t()
          | ContentPart.t()
          | ToolCall.t()
          | ToolResult.t()
          | Function.t()
          | PromptTemplate.t()
        ) ::
          %{String.t() => any()} | [%{String.t() => any()}]
  def for_api(%_{} = model, %Message{content: content} = msg) when is_list(content) do
    %{
      "role" => msg.role,
      "content" => ContentPart.content_to_string(content)
    }
    |> Utils.conditionally_add_to_map("name", msg.name)
    |> Utils.conditionally_add_to_map(
      "tool_calls",
      Enum.map(msg.tool_calls || [], &for_api(model, &1))
    )
  end

  def for_api(%_{} = model, %Message{role: :assistant, tool_calls: tool_calls} = msg)
      when is_list(tool_calls) do
    %{
      "role" => :assistant,
      "content" => ContentPart.content_to_string(msg.content)
    }
    |> Utils.conditionally_add_to_map("tool_calls", Enum.map(tool_calls, &for_api(model, &1)))
  end

  def for_api(%_{} = _model, %ToolResult{type: :function} = result) do
    # a ToolResult becomes a stand-alone %Message{role: :tool} response.
    %{
      "role" => :tool,
      "tool_call_id" => result.tool_call_id,
      "content" => ContentPart.content_to_string(result.content)
    }
  end

  def for_api(%_{} = _model, %Message{role: :tool, tool_results: tool_results} = _msg)
      when is_list(tool_results) do
    # A single ToolResult can expand into multiple tool messages.
    Enum.map(tool_results, fn result ->
      %{
        "role" => :tool,
        "tool_call_id" => result.tool_call_id,
        "content" => ContentPart.content_to_string(result.content)
      }
    end)
  end

  # ToolCall support
  def for_api(%_{} = _model, %ToolCall{type: :function} = fun) do
    %{
      "id" => fun.call_id,
      "type" => "function",
      "function" => %{
        "name" => fun.name,
        "arguments" => Jason.encode!(fun.arguments)
      }
    }
  end

  # Function support
  def for_api(%_{} = _model, %Function{} = fun) do
    %{
      "name" => fun.name,
      "parameters" => get_parameters(fun)
    }
    |> Utils.conditionally_add_to_map("description", fun.description)
  end

  def for_api(%_{} = _model, %PromptTemplate{} = _template) do
    raise LangChainError, "PromptTemplates must be converted to messages."
  end

  @doc false
  def get_parameters(%Function{parameters: [], parameters_schema: nil} = _fun) do
    %{
      "type" => "object",
      "properties" => %{}
    }
  end

  def get_parameters(%Function{parameters: [], parameters_schema: schema} = _fun)
      when is_map(schema) do
    schema
  end

  def get_parameters(%Function{parameters: params} = _fun) do
    FunctionParam.to_parameters_schema(params)
  end

  @doc """
  Calls the LiteLLM gateway passing the ChatLiteLLM struct with configuration,
  plus either a simple message or the list of messages to act as the prompt.

  Optionally pass in a list of tools available to the LLM for requesting
  execution in response.

  **NOTE:** This function *can* be used directly, but the primary interface
  should be through `LangChain.Chains.LLMChain`.
  """
  @impl ChatModel
  def call(litellm, prompt, tools \\ [])

  def call(%ChatLiteLLM{} = litellm, prompt, tools) when is_binary(prompt) do
    messages = [
      Message.new_system!(),
      Message.new_user!(prompt)
    ]

    call(litellm, messages, tools)
  end

  def call(%ChatLiteLLM{} = litellm, messages, tools) when is_list(messages) do
    metadata = %{
      model: litellm.model,
      provider: provider(),
      message_count: length(messages),
      tools_count: length(tools)
    }

    ChatModel.llm_telemetry_span(litellm, metadata, fn ->
      try do
        LangChain.Telemetry.llm_prompt(
          %{system_time: System.system_time()},
          %{model: litellm.model, messages: messages}
        )

        case do_api_request(litellm, messages, tools) do
          {:error, %LangChainError{} = reason} ->
            {:error, reason}

          parsed_data ->
            LangChain.Telemetry.llm_response(
              %{system_time: System.system_time()},
              %{model: litellm.model, response: parsed_data}
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
  List the models the configured gateway is currently serving.

  Calls the gateway's `/v1/models` endpoint, derived from the configured
  `:endpoint`, and returns the model ids. This is the practical benefit of
  routing through a gateway: the set of reachable models is whatever the
  operator configured, so it is discovered rather than hardcoded.

      {:ok, chat} = ChatLiteLLM.new(%{model: "gpt-4o-mini"})
      {:ok, models} = ChatLiteLLM.list_models(chat)

  """
  @spec list_models(t()) :: {:ok, [String.t()]} | {:error, LangChainError.t()}
  def list_models(%ChatLiteLLM{} = litellm) do
    url = models_url(litellm.endpoint)

    Req.new(
      url: url,
      auth: {:bearer, get_api_key(litellm)},
      receive_timeout: litellm.receive_timeout,
      retry: false
    )
    |> Req.merge(litellm.req_config |> Keyword.new())
    |> Req.get()
    |> case do
      {:ok, %Req.Response{status: status, body: %{"data" => data}}}
      when status in 200..299 and is_list(data) ->
        {:ok, data |> Enum.map(&Map.get(&1, "id")) |> Enum.reject(&is_nil/1)}

      {:ok, %Req.Response{status: status, body: body}} when status in 200..299 ->
        {:error,
         LangChainError.exception(
           type: "unexpected_response",
           message: "Unexpected model list response: #{inspect(body)}"
         )}

      {:ok, %Req.Response{status: 401}} ->
        {:error,
         LangChainError.exception(type: "authentication_error", message: "Authentication failed")}

      {:ok, %Req.Response{status: status}} ->
        {:error,
         LangChainError.exception(
           type: "unexpected_response",
           message: "Model list request failed with status #{status}"
         )}

      {:error, reason} ->
        {:error,
         LangChainError.exception(
           type: "transport_error",
           message: "Model list request failed: #{inspect(reason)}",
           original: reason
         )}
    end
  end

  # Derive the `/v1/models` URL from a chat-completions endpoint. Handles both a
  # gateway mounted at the root and one mounted under a path prefix.
  @doc false
  @spec models_url(String.t()) :: String.t()
  def models_url(endpoint) when is_binary(endpoint) do
    cond do
      String.ends_with?(endpoint, "/chat/completions") ->
        String.replace_suffix(endpoint, "/chat/completions", "/models")

      String.ends_with?(endpoint, "/") ->
        endpoint <> "models"

      true ->
        endpoint <> "/models"
    end
  end

  # Make the API request to the LiteLLM gateway.
  @doc false
  @spec do_api_request(t(), [Message.t()], ChatModel.tools(), integer() | nil) ::
          list() | struct() | {:error, LangChainError.t()}
  def do_api_request(litellm, messages, tools, retry_count \\ nil)

  def do_api_request(_litellm, _messages, _tools, 0) do
    raise LangChainError, "Retries exceeded. Connection failed."
  end

  def do_api_request(
        %ChatLiteLLM{stream: false} = litellm,
        messages,
        tools,
        retry_count
      ) do
    retry_count = retry_count || litellm.retry_count + 1
    raw_data = for_api(litellm, messages, tools)

    if litellm.verbose_api do
      IO.inspect(raw_data, label: "RAW DATA BEING SUBMITTED")
    end

    req =
      Req.new(
        url: litellm.endpoint,
        json: raw_data,
        auth: {:bearer, get_api_key(litellm)},
        headers: [
          {"Content-Type", "application/json"}
        ],
        receive_timeout: litellm.receive_timeout,
        # Disable Req-level retry to prevent compounding with LangChain's own
        # :closed retry. See https://github.com/brainlid/langchain/issues/503
        retry: false
      )

    req
    |> Req.merge(litellm.req_config |> Keyword.new())
    |> Req.post()
    |> case do
      {:ok, %Req.Response{status: status, body: data} = response} when status in 200..299 ->
        if litellm.verbose_api do
          IO.inspect(response, label: "RAW REQ RESPONSE")
        end

        Callbacks.fire(litellm.callbacks, :on_llm_response_headers, [response.headers])

        case do_process_response(litellm, data) do
          {:error, %LangChainError{} = reason} ->
            {:error, reason}

          result ->
            Callbacks.fire(litellm.callbacks, :on_llm_new_message, [result])

            LangChain.Telemetry.emit_event(
              [:langchain, :llm, :response, :non_streaming],
              %{system_time: System.system_time()},
              %{
                model: litellm.model,
                response_size: byte_size(inspect(result))
              }
            )

            result
        end

      {:ok, %Req.Response{status: 400, body: body}} ->
        {:error,
         LangChainError.exception(type: "bad_request", message: "Bad request: #{inspect(body)}")}

      {:ok, %Req.Response{status: 401}} ->
        {:error,
         LangChainError.exception(type: "authentication_error", message: "Authentication failed")}

      {:ok, %Req.Response{status: 403}} ->
        {:error,
         LangChainError.exception(type: "permission_denied", message: "Permission denied")}

      {:ok, %Req.Response{status: 404}} ->
        {:error, LangChainError.exception(type: "not_found", message: "Endpoint not found")}

      {:ok, %Req.Response{status: 429}} ->
        {:error,
         LangChainError.exception(type: "rate_limit_exceeded", message: "Rate limit exceeded")}

      {:ok, %Req.Response{status: 529}} ->
        {:error, LangChainError.exception(type: "overloaded", message: "Overloaded")}

      {:ok, %Req.Response{status: status}} when status in 500..599 ->
        {:error,
         LangChainError.exception(type: "server_error", message: "Server error: #{status}")}

      {:error, %Req.TransportError{reason: reason} = err} ->
        transport_error(litellm, messages, tools, retry_count, reason, err)

      other ->
        Logger.warning(fn -> "Unexpected and unhandled API response! #{inspect(other)}" end)
        other
    end
  end

  def do_api_request(
        %ChatLiteLLM{stream: true} = litellm,
        messages,
        tools,
        retry_count
      ) do
    retry_count = retry_count || litellm.retry_count + 1
    raw_data = for_api(litellm, messages, tools)

    if litellm.verbose_api do
      IO.inspect(raw_data, label: "RAW DATA BEING SUBMITTED")
    end

    Req.new(
      url: litellm.endpoint,
      json: raw_data,
      auth: {:bearer, get_api_key(litellm)},
      headers: [
        {"Content-Type", "application/json"}
      ],
      receive_timeout: litellm.receive_timeout,
      # Disable Req-level retry to prevent compounding with LangChain's own
      # :closed retry. See https://github.com/brainlid/langchain/issues/503
      retry: false
    )
    |> Req.merge(litellm.req_config |> Keyword.new())
    |> Req.post(
      into:
        Utils.handle_stream_fn(
          litellm,
          &decode_stream/1,
          &do_process_response(litellm, &1)
        )
    )
    |> case do
      {:ok, %Req.Response{body: data} = response} ->
        Callbacks.fire(litellm.callbacks, :on_llm_response_headers, [response.headers])

        data

      {:error, %LangChainError{} = error} ->
        {:error, error}

      {:error, %Req.TransportError{reason: reason} = err} ->
        transport_error(litellm, messages, tools, retry_count, reason, err)

      other ->
        Logger.warning(fn ->
          "Unhandled and unexpected response from streamed post call. #{inspect(other)}"
        end)

        {:error,
         LangChainError.exception(
           type: "unexpected_response",
           message: "Unexpected response: #{inspect(other)}"
         )}
    end
  end

  # Shared transport-error mapping for both the streaming and non-streaming
  # paths. A `:closed` reason forces a retry by recursing with a decremented
  # counter, matching the sibling OpenAI-compatible models.
  defp transport_error(litellm, messages, tools, retry_count, reason, err) do
    case reason do
      :timeout ->
        {:error,
         LangChainError.exception(type: "timeout", message: "Request timed out", original: err)}

      :closed ->
        Logger.debug(fn -> "Connection closed: retry count = #{inspect(retry_count)}" end)
        do_api_request(litellm, messages, tools, retry_count - 1)

      :nxdomain ->
        {:error,
         LangChainError.exception(
           type: "dns_error",
           message: "DNS resolution failed",
           original: err
         )}

      :econnrefused ->
        {:error,
         LangChainError.exception(
           type: "connection_refused",
           message:
             "Connection refused. Is the LiteLLM gateway running and reachable at the configured endpoint?",
           original: err
         )}

      :connect_timeout ->
        {:error,
         LangChainError.exception(
           type: "connection_timeout",
           message: "Connection timeout",
           original: err
         )}

      other ->
        {:error,
         LangChainError.exception(
           type: "transport_error",
           message: "Transport error: #{inspect(other)}",
           original: err
         )}
    end
  end

  @doc """
  Decode a streamed response from the gateway. The gateway emits the OpenAI
  server-sent-event format, so this matches the OpenAI implementation.
  """
  @spec decode_stream({String.t(), String.t()}, list(), non_neg_integer()) ::
          {%{String.t() => any()}} | {:error, LangChainError.t()}
  def decode_stream({raw_data, buffer}, done \\ [], depth \\ 0) do
    raw_data
    |> String.split("data: ")
    |> Enum.reduce({done, buffer}, fn str, {done, incomplete} = acc ->
      str
      |> String.trim()
      |> case do
        "" ->
          acc

        "[DONE]" ->
          acc

        json ->
          parse_combined_data(incomplete, json, done, depth)
      end
    end)
  end

  defp parse_combined_data(incomplete, json, done, depth)

  defp parse_combined_data("", json, done, _depth) do
    json
    |> Jason.decode()
    |> case do
      {:ok, parsed} ->
        {done ++ [parsed], ""}

      {:error, _reason} ->
        {done, json}
    end
  end

  defp parse_combined_data(incomplete, json, done, depth) when depth < 10 do
    starting_json = incomplete <> json

    decode_stream({starting_json, ""}, done, depth + 1)
  end

  defp parse_combined_data(_incomplete, _json, done, depth) when depth >= 10 do
    Logger.warning(fn -> "Stream parsing recursion limit exceeded: depth = #{depth}" end)
    {done, ""}
  end

  @doc false
  @spec do_process_response(
          ChatLiteLLM.t(),
          data :: %{String.t() => any()} | {:error, any()}
        ) ::
          :skip
          | Message.t()
          | [Message.t()]
          | MessageDelta.t()
          | [MessageDelta.t()]
          | TokenUsage.t()
          | {:error, LangChainError.t()}
  def do_process_response(model, %{"choices" => _choices} = data) do
    token_usage = get_token_usage(data)

    case data do
      # No choices but usage was reported: the terminal usage-only chunk.
      %{"choices" => [], "usage" => _usage} ->
        token_usage

      %{"choices" => []} ->
        :skip

      %{"choices" => choices} ->
        choices
        |> Enum.map(&do_process_response(model, &1))
        |> Enum.map(fn result ->
          result
          |> TokenUsage.set(token_usage)
          |> merge_response_metadata(data)
        end)
    end
  end

  # Full message with tool calls.
  def do_process_response(
        model,
        %{"finish_reason" => finish_reason, "message" => %{"tool_calls" => calls} = message} =
          data
      )
      when finish_reason in ["tool_calls", "stop"] do
    metadata = extract_response_metadata(data)

    case Message.new(%{
           "role" => "assistant",
           "content" => message["content"],
           "complete" => true,
           "index" => data["index"],
           "tool_calls" => Enum.map(calls || [], &do_process_response(model, &1)),
           "metadata" => metadata
         }) do
      {:ok, message} ->
        message

      {:error, %Ecto.Changeset{} = changeset} ->
        {:error, LangChainError.exception(changeset)}
    end
  end

  # Streamed delta.
  #
  # NOTE: this clause deliberately does NOT pattern match on a model name. The
  # gateway routes arbitrary model strings, so matching a literal name (as the
  # DeepSeek module does for "deepseek-chat") would silently skip this clause
  # for every other model the gateway serves.
  def do_process_response(model, %{"delta" => delta_body} = msg) do
    # The gateway frequently omits `finish_reason` on intermediate chunks.
    finish = Map.get(msg, "finish_reason", nil)
    status = finish_reason_to_status(finish)

    tool_calls =
      case delta_body do
        %{"tool_calls" => tools_data} when is_list(tools_data) ->
          Enum.map(tools_data, &do_process_response(model, &1))

        _other ->
          nil
      end

    role =
      case delta_body do
        %{"role" => role} when is_binary(role) -> role
        _other -> "unknown"
      end

    metadata = extract_response_metadata(msg)

    data =
      delta_body
      |> Map.put("role", role)
      |> Map.put("index", Map.get(msg, "index", 0))
      |> Map.put("status", status)
      |> Map.put("tool_calls", tool_calls)
      |> Map.put("metadata", metadata)
      |> put_reasoning_content(delta_body)

    case MessageDelta.new(data) do
      {:ok, message} ->
        message

      {:error, %Ecto.Changeset{} = changeset} ->
        {:error, LangChainError.exception(changeset)}
    end
  end

  # Tool call as part of a delta message.
  def do_process_response(_model, %{"function" => func_body, "index" => index} = tool_call) do
    case ToolCall.new(%{
           status: :incomplete,
           type: :function,
           call_id: tool_call["id"],
           name: Map.get(func_body, "name", nil),
           arguments: Map.get(func_body, "arguments", nil),
           index: index
         }) do
      {:ok, %ToolCall{} = call} ->
        call

      {:error, %Ecto.Changeset{} = changeset} ->
        {:error, LangChainError.exception(changeset)}
    end
  end

  # Tool call from a complete message.
  def do_process_response(_model, %{
        "function" => %{
          "arguments" => args,
          "name" => name
        },
        "id" => call_id,
        "type" => "function"
      }) do
    cond do
      is_nil(call_id) or call_id == "" ->
        {:error,
         LangChainError.exception(
           type: "invalid_tool_call",
           message: "Tool call missing required field: id"
         )}

      is_nil(name) or name == "" ->
        {:error,
         LangChainError.exception(
           type: "invalid_tool_call",
           message: "Tool call missing required field: name"
         )}

      true ->
        case ToolCall.new(%{
               type: :function,
               status: :complete,
               name: name,
               arguments: args,
               call_id: call_id
             }) do
          {:ok, %ToolCall{} = call} ->
            call

          {:error, %Ecto.Changeset{} = changeset} ->
            {:error, LangChainError.exception(changeset)}
        end
    end
  end

  # Complete (non-streamed) message.
  def do_process_response(
        _model,
        %{
          "finish_reason" => finish_reason,
          "message" => message
        } = data
      ) do
    status = finish_reason_to_status(finish_reason)
    metadata = extract_response_metadata(data)

    message =
      case message["reasoning_content"] do
        nil ->
          message

        reasoning_content ->
          content =
            [
              ContentPart.thinking!(reasoning_content),
              ContentPart.text!(message["content"] || "")
            ]

          Map.put(message, "content", content)
      end

    case Message.new(
           Map.merge(message, %{
             "status" => status,
             "index" => Map.get(data, "index", 0),
             "metadata" => metadata
           })
         ) do
      {:ok, message} ->
        message

      {:error, %Ecto.Changeset{} = changeset} ->
        {:error, LangChainError.exception(changeset)}
    end
  end

  def do_process_response(
        _model,
        %{
          "error" => %{"code" => code, "message" => reason} = _error_data
        } = response
      ) do
    type =
      case code do
        "429" -> "rate_limit_exceeded"
        _other -> nil
      end

    {:error, LangChainError.exception(type: type, message: reason, original: response)}
  end

  def do_process_response(_model, %{"error" => %{"message" => reason}} = response) do
    {:error, LangChainError.exception(message: reason, original: response)}
  end

  def do_process_response(_model, {:error, %Jason.DecodeError{} = response}) do
    error_message = "Received invalid JSON: #{inspect(response)}"

    {:error,
     LangChainError.exception(type: "invalid_json", message: error_message, original: response)}
  end

  def do_process_response(_model, other) do
    {:error, LangChainError.exception(message: "Unexpected response", original: other)}
  end

  # Reasoning models routed through the gateway report their thinking in a
  # `reasoning_content` delta field alongside (or instead of) `content`. Promote
  # it to a thinking ContentPart when present, and otherwise leave the raw
  # `content` untouched so a role-only or empty chunk stays valid.
  defp put_reasoning_content(data, delta_body) do
    case Map.get(delta_body, "reasoning_content") do
      nil ->
        data

      "" ->
        data

      reasoning_content ->
        Map.put(data, "content", ContentPart.thinking!(reasoning_content))
    end
  end

  defp merge_response_metadata(message, response_data) do
    response_metadata = extract_response_metadata(response_data)
    current_metadata = message.metadata || %{}
    %{message | metadata: Map.merge(current_metadata, response_metadata)}
  end

  defp finish_reason_to_status(nil), do: :incomplete
  defp finish_reason_to_status("stop"), do: :complete
  defp finish_reason_to_status("tool_calls"), do: :complete
  defp finish_reason_to_status("function_call"), do: :complete
  defp finish_reason_to_status("content_filter"), do: :content_filtered
  defp finish_reason_to_status("length"), do: :length
  defp finish_reason_to_status("max_tokens"), do: :length

  defp finish_reason_to_status(other) do
    Logger.warning("Unsupported finish_reason in response. Reason: #{inspect(other)}")
    nil
  end

  # Capture the gateway-reported provenance of the response. `model` matters
  # more here than with a single-provider API: the gateway may route a request
  # to a different upstream model than the one requested (fallbacks, aliases,
  # load balancing), so the responding model is recorded.
  defp extract_response_metadata(response_data) do
    metadata = %{}

    metadata =
      case Map.get(response_data, "model") do
        nil -> metadata
        model -> Map.put(metadata, :model, model)
      end

    metadata =
      case Map.get(response_data, "id") do
        nil -> metadata
        id -> Map.put(metadata, :id, id)
      end

    metadata =
      case Map.get(response_data, "system_fingerprint") do
        nil -> metadata
        fingerprint -> Map.put(metadata, :system_fingerprint, fingerprint)
      end

    case Map.get(response_data, "object") do
      nil -> metadata
      object -> Map.put(metadata, :object, object)
    end
  end

  defp get_token_usage(%{"usage" => usage} = _response_body) when is_map(usage) do
    TokenUsage.new!(%{
      input: Map.get(usage, "prompt_tokens"),
      output: Map.get(usage, "completion_tokens"),
      raw: usage
    })
  end

  defp get_token_usage(_response_body), do: nil

  @impl ChatModel
  def provider, do: "litellm"

  @doc """
  Whether a failed request should be retried against a fallback model.
  """
  @impl ChatModel
  @spec retry_on_fallback?(LangChainError.t()) :: boolean()
  def retry_on_fallback?(%LangChainError{type: "rate_limited"}), do: true
  def retry_on_fallback?(%LangChainError{type: "rate_limit_exceeded"}), do: true
  def retry_on_fallback?(%LangChainError{type: "timeout"}), do: true
  def retry_on_fallback?(%LangChainError{type: "too_many_requests"}), do: true
  def retry_on_fallback?(%LangChainError{type: "server_error"}), do: true
  def retry_on_fallback?(%LangChainError{type: "overloaded"}), do: true
  def retry_on_fallback?(_), do: false

  @doc """
  Generate a config map that can later restore the model's configuration.
  """
  @impl ChatModel
  @spec serialize_config(t()) :: %{String.t() => any()}
  def serialize_config(%ChatLiteLLM{} = model) do
    Utils.to_serializable_map(
      model,
      [
        :endpoint,
        :model,
        :temperature,
        :frequency_penalty,
        :presence_penalty,
        :top_p,
        :receive_timeout,
        :seed,
        :n,
        :json_response,
        :json_schema,
        :stream,
        :max_tokens,
        :stream_options,
        :user,
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
    ChatLiteLLM.new(data)
  end
end
