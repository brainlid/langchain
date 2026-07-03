defmodule LangChain.OpenTelemetry.Attributes do
  @moduledoc """
  Builds OpenTelemetry span attribute maps from LangChain telemetry metadata,
  following a subset of the GenAI Semantic Conventions (v1.40+).

  Attribute key constants are defined as string literals because the Hex
  `opentelemetry_semantic_conventions` package lags behind the latest spec.

  ## Coverage

  This integration emits the following semantic-convention attributes:

    * `gen_ai.operation.name`, `gen_ai.provider.name`, `gen_ai.output.type`
      (`"json"` when the model requests structured output, else `"text"`)
    * `gen_ai.request.model`, `gen_ai.response.model`
    * `server.address`, `server.port` (derived from the model's request endpoint)
    * Request parameters (when the model sets them):
      `gen_ai.request.temperature`, `gen_ai.request.max_tokens`,
      `gen_ai.request.top_p`, `gen_ai.request.top_k`,
      `gen_ai.request.frequency_penalty`, `gen_ai.request.presence_penalty`,
      `gen_ai.request.seed`, `gen_ai.request.choice.count`,
      `gen_ai.request.stream`, `gen_ai.request.stop_sequences`,
      `gen_ai.request.reasoning.level`
    * `gen_ai.usage.input_tokens`, `gen_ai.usage.output_tokens`, and — best-effort
      from the provider-specific `TokenUsage.raw` — `gen_ai.usage.cache_read.input_tokens`,
      `gen_ai.usage.cache_creation.input_tokens`, `gen_ai.usage.reasoning.output_tokens`
    * `gen_ai.response.finish_reasons` (best-effort from `Message.status`)
    * `gen_ai.input.messages`, `gen_ai.output.messages` (opt-in — see `Config`)
    * `gen_ai.tool.name`, `gen_ai.tool.call.id`, `gen_ai.tool.type`,
      `gen_ai.tool.description`,
      `gen_ai.tool.call.arguments` / `gen_ai.tool.call.result` (opt-in)
    * `gen_ai.agent.name`, `gen_ai.conversation.id` (from `custom_context`)
    * `error.type` (on failed operations)

  It does **not** currently emit `gen_ai.response.id` or a distinct
  `gen_ai.response.model` (LangChain keeps no response id and normalizes the
  response model to the requested one), nor streaming timing
  (`gen_ai.response.time_to_first_chunk`). Treat the output as a useful subset
  rather than full conformance.

  See: https://opentelemetry.io/docs/specs/semconv/gen-ai/
  """

  alias LangChain.OpenTelemetry.Config
  alias LangChain.OpenTelemetry.MessageSerializer
  alias LangChain.OpenTelemetry.ProviderMapping

  # Attribute key constants
  @operation_name "gen_ai.operation.name"
  @provider_name "gen_ai.provider.name"
  @output_type "gen_ai.output.type"
  @request_model "gen_ai.request.model"
  @request_temperature "gen_ai.request.temperature"
  @request_max_tokens "gen_ai.request.max_tokens"
  @request_top_p "gen_ai.request.top_p"
  @request_top_k "gen_ai.request.top_k"
  @request_frequency_penalty "gen_ai.request.frequency_penalty"
  @request_presence_penalty "gen_ai.request.presence_penalty"
  @request_seed "gen_ai.request.seed"
  @request_choice_count "gen_ai.request.choice.count"
  @request_stream "gen_ai.request.stream"
  @request_stop_sequences "gen_ai.request.stop_sequences"
  @request_reasoning_level "gen_ai.request.reasoning.level"
  @response_model "gen_ai.response.model"
  @response_finish_reasons "gen_ai.response.finish_reasons"
  @usage_input_tokens "gen_ai.usage.input_tokens"
  @usage_output_tokens "gen_ai.usage.output_tokens"
  @usage_cache_read_tokens "gen_ai.usage.cache_read.input_tokens"
  @usage_cache_creation_tokens "gen_ai.usage.cache_creation.input_tokens"
  @usage_reasoning_tokens "gen_ai.usage.reasoning.output_tokens"
  @input_messages "gen_ai.input.messages"
  @output_messages "gen_ai.output.messages"
  @tool_name "gen_ai.tool.name"
  @tool_call_id "gen_ai.tool.call.id"
  @tool_type "gen_ai.tool.type"
  @tool_description "gen_ai.tool.description"
  @tool_call_arguments "gen_ai.tool.call.arguments"
  @tool_call_result "gen_ai.tool.call.result"
  @agent_name "gen_ai.agent.name"
  @conversation_id "gen_ai.conversation.id"
  @server_address "server.address"
  @server_port "server.port"

  @doc """
  Returns the `gen_ai.operation.name` attribute key.
  """
  def operation_name_key, do: @operation_name

  @doc """
  Builds attributes for an LLM call start event.

  Returns operation name, output type, model, provider, and request-parameter
  attributes (`gen_ai.request.*`, sourced from `metadata[:request_options]`).
  Input message capture is handled separately by the prompt event handler in
  `SpanHandler`.
  """
  @spec llm_call_start(map()) :: [{String.t(), term()}]
  @spec llm_call_start(map(), Config.t()) :: [{String.t(), term()}]
  def llm_call_start(metadata, %Config{} = _config \\ %Config{}) do
    # All LangChain chat models are chat-completion style, so output is text unless
    # the model requested structured output. `ChatModel.output_type/1` resolves
    # that; default to "text" when the model didn't set it.
    attrs = [
      {@operation_name, "chat"},
      {@output_type, metadata[:output_type] || "text"},
      {@request_model, metadata[:model]}
    ]

    attrs =
      case metadata[:provider] do
        nil -> attrs
        provider -> [{@provider_name, ProviderMapping.to_otel(provider)} | attrs]
      end

    request_option_attributes(metadata[:request_options]) ++
      server_attributes(metadata[:endpoint]) ++ attrs
  end

  # Derives `server.address` / `server.port` from the request endpoint URL. `URI`
  # supplies the default port for known schemes (443 for https), which is accurate
  # per the semantic convention. Returns [] when there's no usable host.
  @spec server_attributes(String.t() | nil) :: [{String.t(), term()}]
  defp server_attributes(endpoint) when is_binary(endpoint) do
    case URI.parse(endpoint) do
      %URI{host: host, port: port} when is_binary(host) and host != "" ->
        attrs = [{@server_address, host}]
        if port, do: [{@server_port, port} | attrs], else: attrs

      _ ->
        []
    end
  end

  defp server_attributes(_), do: []

  # Maps the provider-neutral `:request_options` map (built by
  # `LangChain.ChatModels.ChatModel.request_options/1`) to `gen_ai.request.*`
  # semantic-convention attributes, dropping any parameter the model didn't set.
  @spec request_option_attributes(map() | nil) :: [{String.t(), term()}]
  defp request_option_attributes(opts) when is_map(opts) do
    [
      {@request_temperature, opts[:temperature]},
      {@request_max_tokens, opts[:max_tokens]},
      {@request_top_p, opts[:top_p]},
      {@request_top_k, opts[:top_k]},
      {@request_frequency_penalty, opts[:frequency_penalty]},
      {@request_presence_penalty, opts[:presence_penalty]},
      {@request_seed, opts[:seed]},
      {@request_choice_count, opts[:choice_count]},
      {@request_stream, opts[:stream]},
      {@request_stop_sequences, stop_sequences(opts[:stop_sequences])},
      {@request_reasoning_level, reasoning_level(opts[:reasoning_level])}
    ]
    |> Enum.reject(fn {_key, value} -> is_nil(value) end)
  end

  defp request_option_attributes(_), do: []

  # `gen_ai.request.stop_sequences` is a string array. LangChain models carry the
  # stop value as either a single string or a list, so normalize to a list of
  # strings (dropping non-strings). Returns nil when nothing usable remains.
  defp stop_sequences(nil), do: nil
  defp stop_sequences(seq) when is_binary(seq), do: [seq]

  defp stop_sequences(seq) when is_list(seq) do
    case Enum.filter(seq, &is_binary/1) do
      [] -> nil
      list -> list
    end
  end

  defp stop_sequences(_), do: nil

  defp reasoning_level(nil), do: nil
  defp reasoning_level(level) when is_binary(level), do: level
  defp reasoning_level(level) when is_atom(level), do: Atom.to_string(level)
  defp reasoning_level(_), do: nil

  @doc """
  Builds attributes for an LLM call stop event (token usage and response model).

  When `config.capture_output_messages` is true and `metadata[:result]` contains
  a message, serializes output messages into `gen_ai.output.messages`.
  """
  @spec llm_call_stop(map()) :: [{String.t(), term()}]
  @spec llm_call_stop(map(), Config.t()) :: [{String.t(), term()}]
  def llm_call_stop(metadata, %Config{} = config \\ %Config{}) do
    attrs =
      case metadata[:token_usage] do
        %{input: input, output: output} = usage ->
          attrs = usage_extra_attributes(usage)
          attrs = if output, do: [{@usage_output_tokens, output} | attrs], else: attrs
          if input, do: [{@usage_input_tokens, input} | attrs], else: attrs

        _ ->
          []
      end

    attrs =
      case metadata[:model] do
        nil -> attrs
        model -> [{@response_model, model} | attrs]
      end

    attrs = finish_reason_attributes(metadata[:result]) ++ attrs

    if config.capture_output_messages do
      case output_messages_from_result(metadata[:result]) do
        nil -> attrs
        serialized -> [{@output_messages, serialized} | attrs]
      end
    else
      attrs
    end
  end

  # Best-effort cache/reasoning token counts from the provider-specific
  # `TokenUsage.raw` map. Anthropic reports `cache_creation_input_tokens` /
  # `cache_read_input_tokens` at the top level; OpenAI nests `cached_tokens` under
  # `prompt_tokens_details` and `reasoning_tokens` under `completion_tokens_details`
  # (or `output_tokens_details` on the Responses API). Only positive counts are
  # emitted — providers report 0 for unused cache/reasoning on every call, which is
  # pure noise on the span.
  @spec usage_extra_attributes(map()) :: [{String.t(), term()}]
  defp usage_extra_attributes(%{raw: raw}) when is_map(raw) do
    [
      {@usage_cache_read_tokens, cache_read_tokens(raw)},
      {@usage_cache_creation_tokens, positive_int(Map.get(raw, "cache_creation_input_tokens"))},
      {@usage_reasoning_tokens, reasoning_output_tokens(raw)}
    ]
    |> Enum.reject(fn {_key, value} -> is_nil(value) end)
  end

  defp usage_extra_attributes(_), do: []

  defp cache_read_tokens(raw) do
    positive_int(Map.get(raw, "cache_read_input_tokens")) ||
      positive_int(nested(raw, "prompt_tokens_details", "cached_tokens"))
  end

  defp reasoning_output_tokens(raw) do
    positive_int(nested(raw, "completion_tokens_details", "reasoning_tokens")) ||
      positive_int(nested(raw, "output_tokens_details", "reasoning_tokens"))
  end

  defp nested(map, key1, key2) do
    case Map.get(map, key1) do
      inner when is_map(inner) -> Map.get(inner, key2)
      _ -> nil
    end
  end

  defp positive_int(n) when is_integer(n) and n > 0, do: n
  defp positive_int(_), do: nil

  # Best-effort `gen_ai.response.finish_reasons` reconstructed from the assembled
  # message's normalized `:status`. LangChain collapses a provider's raw finish
  # reason into `Message.status` (`:complete | :length | :cancelled`), so this is
  # lossy: a `:complete` message that carries tool calls maps to `"tool_calls"`,
  # otherwise `:complete` -> `"stop"`. Streaming results (a delta or list of
  # deltas) are merged to a message first. Returns [] when nothing is derivable.
  @spec finish_reason_attributes(term()) :: [{String.t(), term()}]
  defp finish_reason_attributes(result) do
    case finish_reason(result) do
      nil -> []
      reason -> [{@response_finish_reasons, [reason]}]
    end
  end

  defp finish_reason({:ok, %LangChain.Message{} = msg}), do: message_finish_reason(msg)

  defp finish_reason({:ok, [%LangChain.Message{} | _] = msgs}),
    do: msgs |> List.last() |> message_finish_reason()

  defp finish_reason({:ok, %LangChain.MessageDelta{} = delta}), do: delta_finish_reason([delta])

  defp finish_reason({:ok, [%LangChain.MessageDelta{} | _] = deltas}),
    do: delta_finish_reason(deltas)

  defp finish_reason(_), do: nil

  defp delta_finish_reason(deltas) do
    case deltas |> LangChain.MessageDelta.merge_deltas() |> LangChain.MessageDelta.to_message() do
      {:ok, %LangChain.Message{} = msg} -> message_finish_reason(msg)
      _ -> nil
    end
  end

  defp message_finish_reason(%LangChain.Message{tool_calls: [_ | _]}), do: "tool_calls"
  defp message_finish_reason(%LangChain.Message{status: :complete}), do: "stop"
  defp message_finish_reason(%LangChain.Message{status: :length}), do: "length"
  defp message_finish_reason(%LangChain.Message{status: :cancelled}), do: "cancelled"
  defp message_finish_reason(_), do: nil

  # Serializes the LLM output for `gen_ai.output.messages`. Streaming calls return
  # a list of `%MessageDelta{}` structs (or a single delta) rather than a
  # `%Message{}`; these are merged and converted to a message first so streamed
  # responses are captured the same as non-streaming ones. Returns `nil` when
  # there is nothing to capture.
  defp output_messages_from_result({:ok, %LangChain.Message{} = msg}),
    do: MessageSerializer.serialize_output(msg)

  defp output_messages_from_result({:ok, [%LangChain.Message{} | _] = msgs}),
    do: MessageSerializer.serialize_output(msgs)

  defp output_messages_from_result({:ok, %LangChain.MessageDelta{} = delta}),
    do: serialize_delta_output([delta])

  defp output_messages_from_result({:ok, [%LangChain.MessageDelta{} | _] = deltas}),
    do: serialize_delta_output(deltas)

  defp output_messages_from_result(_), do: nil

  defp serialize_delta_output(deltas) do
    case deltas |> LangChain.MessageDelta.merge_deltas() |> LangChain.MessageDelta.to_message() do
      {:ok, %LangChain.Message{} = msg} -> MessageSerializer.serialize_output(msg)
      # An incomplete stream can't convert to a message — nothing to capture.
      {:error, _reason} -> nil
    end
  end

  @doc """
  Builds attributes for a tool call start event.

  When `config.capture_tool_arguments` is true and `metadata[:arguments]` is present,
  serializes arguments into `gen_ai.tool.call.arguments`.
  """
  @spec tool_call(map()) :: [{String.t(), term()}]
  @spec tool_call(map(), Config.t()) :: [{String.t(), term()}]
  def tool_call(metadata, %Config{} = config \\ %Config{}) do
    attrs = [
      {@operation_name, "execute_tool"},
      {@tool_type, "function"}
    ]

    attrs =
      case metadata[:tool_call_id] do
        nil -> attrs
        id -> [{@tool_call_id, id} | attrs]
      end

    attrs =
      case metadata[:tool_name] do
        nil -> attrs
        name -> [{@tool_name, name} | attrs]
      end

    attrs =
      case metadata[:tool_description] do
        desc when is_binary(desc) and desc != "" -> [{@tool_description, desc} | attrs]
        _ -> attrs
      end

    if config.capture_tool_arguments do
      case metadata[:arguments] do
        nil ->
          attrs

        args when is_map(args) ->
          [{@tool_call_arguments, Jason.encode!(args)} | attrs]

        args when is_binary(args) ->
          [{@tool_call_arguments, args} | attrs]

        _ ->
          attrs
      end
    else
      attrs
    end
  end

  @doc """
  Builds attributes for a tool call stop event.

  When `config.capture_tool_results` is true and `metadata[:tool_result]` is present,
  extracts the result content into `gen_ai.tool.call.result`.
  """
  @spec tool_call_stop(map(), Config.t()) :: [{String.t(), term()}]
  def tool_call_stop(metadata, %Config{} = config) do
    if config.capture_tool_results do
      case metadata[:tool_result] do
        %{content: content} when is_binary(content) ->
          [{@tool_call_result, content}]

        %{content: content} when not is_nil(content) ->
          [{@tool_call_result, inspect(content)}]

        _ ->
          []
      end
    else
      []
    end
  end

  @doc """
  Builds attributes for a chain execution event.

  Sets `gen_ai.conversation.id` from `custom_context` (a `:conversation_id` key,
  falling back to `:langfuse_session_id` — the same session concept) and also
  extracts Langfuse-specific attributes from `custom_context` when present.
  """
  @spec chain_start(map()) :: [{String.t(), term()}]
  def chain_start(metadata) do
    attrs = [{@operation_name, "invoke_agent"}]

    attrs =
      case metadata[:chain_type] do
        nil -> attrs
        chain_type -> [{@agent_name, chain_type} | attrs]
      end

    attrs =
      case conversation_id(metadata[:custom_context]) do
        nil -> attrs
        id -> [{@conversation_id, id} | attrs]
      end

    case metadata[:custom_context] do
      nil -> attrs
      context -> custom_context_attributes(context) ++ attrs
    end
  end

  # `gen_ai.conversation.id` groups traces into a session/thread. LangChain has no
  # first-class conversation id, so it's read from `custom_context`: an explicit
  # `:conversation_id`, else the `:langfuse_session_id` a user may already set.
  defp conversation_id(%{conversation_id: id}) when not is_nil(id), do: id
  defp conversation_id(%{langfuse_session_id: id}) when not is_nil(id), do: id
  defp conversation_id(_), do: nil

  @doc """
  Builds attributes for a chain execution stop event.

  Extracts the first user message as input and the last assistant message as output
  so they appear on the trace-level span in Langfuse (and other OTEL backends).
  """
  @spec chain_stop(map(), Config.t()) :: [{String.t(), term()}]
  def chain_stop(metadata, %Config{} = config) do
    attrs = []

    # Extract input from the original messages (first user message)
    attrs =
      if config.capture_input_messages do
        case extract_user_input(metadata) do
          nil -> attrs
          input -> [{@input_messages, input} | attrs]
        end
      else
        attrs
      end

    # Extract output from last_message (the final assistant response)
    if config.capture_output_messages do
      case metadata[:last_message] do
        %LangChain.Message{role: :assistant} = msg ->
          [{@output_messages, MessageSerializer.serialize_output(msg)} | attrs]

        _ ->
          attrs
      end
    else
      attrs
    end
  end

  defp extract_user_input(metadata) do
    # The chain stop metadata inherits from start, which includes the chain's messages
    # via the result tuple. Try to get the first user message.
    #
    # A standard run terminates with `{:ok, chain}`; `:until_tool_used` terminates
    # with `{:ok, chain, tool_result}` when the target tool is found. Both carry the
    # chain's messages, so handle either shape.
    case metadata[:result] do
      {:ok, %{messages: messages}} -> first_user_input(messages)
      {:ok, %{messages: messages}, _extra} -> first_user_input(messages)
      _ -> nil
    end
  end

  defp first_user_input([_ | _] = messages) do
    messages
    |> Enum.find(fn
      %LangChain.Message{role: :user} -> true
      _ -> false
    end)
    |> case do
      nil -> nil
      first_user -> MessageSerializer.serialize_input([first_user])
    end
  end

  defp first_user_input(_), do: nil

  @doc """
  Extracts Langfuse-specific attributes from a `custom_context` map.

  Supported keys:
  - `:langfuse_trace_name` -> `langfuse.trace.name`
  - `:langfuse_user_id` -> `langfuse.user.id`
  - `:langfuse_session_id` -> `langfuse.session.id`
  - `:langfuse_tags` -> `langfuse.trace.tags`
  - `:langfuse_metadata` -> `langfuse.trace.metadata.*` (flattened)
  """
  @spec custom_context_attributes(map() | nil) :: [{String.t(), term()}]
  def custom_context_attributes(nil), do: []

  def custom_context_attributes(context) when is_map(context) do
    attrs = []

    attrs =
      case Map.get(context, :langfuse_trace_name) do
        nil -> attrs
        name -> [{"langfuse.trace.name", name} | attrs]
      end

    attrs =
      case Map.get(context, :langfuse_user_id) do
        nil -> attrs
        user_id -> [{"langfuse.user.id", user_id} | attrs]
      end

    attrs =
      case Map.get(context, :langfuse_session_id) do
        nil -> attrs
        session_id -> [{"langfuse.session.id", session_id} | attrs]
      end

    attrs =
      case Map.get(context, :langfuse_tags) do
        nil -> attrs
        tags when is_list(tags) -> [{"langfuse.trace.tags", Enum.join(tags, ",")} | attrs]
        _ -> attrs
      end

    case Map.get(context, :langfuse_metadata) do
      nil ->
        attrs

      meta when is_map(meta) ->
        flat =
          Enum.map(meta, fn {k, v} ->
            {"langfuse.trace.metadata.#{k}", stringify_metadata_value(v)}
          end)

        flat ++ attrs

      _ ->
        attrs
    end
  end

  def custom_context_attributes(_), do: []

  # Langfuse metadata values become flat span attributes, which must be strings.
  # Primitives stringify directly. Collections (maps/lists) are JSON-encoded to
  # preserve their structure: a nested map has no `String.Chars` implementation
  # and would make `to_string/1` raise — which, trapped by the span handler,
  # silently drops the entire chain span — while a list would be lossily flattened
  # as an iolist (`["x", "y"]` -> `"xy"`). JSON keeps both faithful and safe.
  defp stringify_metadata_value(value) when is_binary(value), do: value
  defp stringify_metadata_value(value) when is_number(value), do: to_string(value)
  defp stringify_metadata_value(value) when is_atom(value), do: to_string(value)

  defp stringify_metadata_value(value) do
    case Jason.encode(value) do
      {:ok, json} -> json
      {:error, _} -> inspect(value)
    end
  end
end
