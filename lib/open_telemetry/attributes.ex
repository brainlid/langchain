defmodule LangChain.OpenTelemetry.Attributes do
  @moduledoc """
  Builds OpenTelemetry span attribute maps from LangChain telemetry metadata,
  following the GenAI Semantic Conventions (v1.40+).

  Attribute key constants are defined as string literals because the Hex
  `opentelemetry_semantic_conventions` package lags behind the latest spec.

  See: https://opentelemetry.io/docs/specs/semconv/gen-ai/
  """

  alias LangChain.OpenTelemetry.Config
  alias LangChain.OpenTelemetry.MessageSerializer
  alias LangChain.OpenTelemetry.ProviderMapping

  # Attribute key constants
  @operation_name "gen_ai.operation.name"
  @provider_name "gen_ai.provider.name"
  @request_model "gen_ai.request.model"
  @response_model "gen_ai.response.model"
  @usage_input_tokens "gen_ai.usage.input_tokens"
  @usage_output_tokens "gen_ai.usage.output_tokens"
  @input_messages "gen_ai.input.messages"
  @output_messages "gen_ai.output.messages"
  @tool_name "gen_ai.tool.name"
  @tool_call_id "gen_ai.tool.call.id"
  @tool_type "gen_ai.tool.type"
  @tool_call_arguments "gen_ai.tool.call.arguments"
  @tool_call_result "gen_ai.tool.call.result"
  @agent_name "gen_ai.agent.name"

  @doc """
  Returns the `gen_ai.operation.name` attribute key.
  """
  def operation_name_key, do: @operation_name

  @doc """
  Builds attributes for an LLM call start event.

  When `config.capture_input_messages` is true and `metadata[:messages]` is present,
  serializes input messages into `gen_ai.input.messages`.
  """
  @spec llm_call_start(map()) :: [{String.t(), term()}]
  @spec llm_call_start(map(), Config.t()) :: [{String.t(), term()}]
  def llm_call_start(metadata, %Config{} = config \\ %Config{}) do
    attrs = [
      {@operation_name, "chat"},
      {@request_model, metadata[:model]}
    ]

    attrs =
      case metadata[:provider] do
        nil -> attrs
        provider -> [{@provider_name, ProviderMapping.to_otel(provider)} | attrs]
      end

    if config.capture_input_messages do
      case metadata[:messages] do
        [_ | _] = messages ->
          [{@input_messages, MessageSerializer.serialize_input(messages)} | attrs]

        _ ->
          attrs
      end
    else
      attrs
    end
  end

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
        %{input: input, output: output} ->
          attrs = []
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

    if config.capture_output_messages do
      case metadata[:result] do
        {:ok, %LangChain.Message{} = msg} ->
          [{@output_messages, MessageSerializer.serialize_output(msg)} | attrs]

        {:ok, [%LangChain.Message{} | _] = msgs} ->
          [{@output_messages, MessageSerializer.serialize_output(msgs)} | attrs]

        _ ->
          attrs
      end
    else
      attrs
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

  Also extracts Langfuse-specific attributes from `custom_context` when present.
  """
  @spec chain_start(map()) :: [{String.t(), term()}]
  def chain_start(metadata) do
    attrs = [{@operation_name, "invoke_agent"}]

    attrs =
      case metadata[:chain_type] do
        nil -> attrs
        chain_type -> [{@agent_name, chain_type} | attrs]
      end

    case metadata[:custom_context] do
      nil -> attrs
      context -> custom_context_attributes(context) ++ attrs
    end
  end

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
    case metadata[:result] do
      {:ok, %{messages: [_ | _] = messages}} ->
        user_messages =
          Enum.filter(messages, fn
            %LangChain.Message{role: :user} -> true
            _ -> false
          end)

        case user_messages do
          [first_user | _] -> MessageSerializer.serialize_input([first_user])
          [] -> nil
        end

      _ ->
        nil
    end
  end

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
            {"langfuse.trace.metadata.#{k}", to_string(v)}
          end)

        flat ++ attrs

      _ ->
        attrs
    end
  end

  def custom_context_attributes(_), do: []
end
