defmodule LangChain.OpenTelemetry.Attributes do
  @moduledoc """
  Builds OpenTelemetry span attribute maps from LangChain telemetry metadata,
  following the GenAI Semantic Conventions (v1.40+).

  Attribute key constants are defined as string literals because the Hex
  `opentelemetry_semantic_conventions` package lags behind the latest spec.

  See: https://opentelemetry.io/docs/specs/semconv/gen-ai/
  """

  alias LangChain.OpenTelemetry.ProviderMapping

  # Attribute key constants
  @operation_name "gen_ai.operation.name"
  @provider_name "gen_ai.provider.name"
  @request_model "gen_ai.request.model"
  @usage_input_tokens "gen_ai.usage.input_tokens"
  @usage_output_tokens "gen_ai.usage.output_tokens"
  @tool_name "gen_ai.tool.name"
  @tool_call_id "gen_ai.tool.call.id"
  @tool_type "gen_ai.tool.type"
  @agent_name "gen_ai.agent.name"

  @doc """
  Returns the `gen_ai.operation.name` attribute key.
  """
  def operation_name_key, do: @operation_name

  @doc """
  Builds attributes for an LLM call start event.
  """
  @spec llm_call_start(map()) :: [{String.t(), term()}]
  def llm_call_start(metadata) do
    attrs = [
      {@operation_name, "chat"},
      {@request_model, metadata[:model]}
    ]

    case metadata[:provider] do
      nil -> attrs
      provider -> [{@provider_name, ProviderMapping.to_otel(provider)} | attrs]
    end
  end

  @doc """
  Builds attributes for an LLM call stop event (token usage).

  Returns only attributes that have non-nil values.
  """
  @spec llm_call_stop(map()) :: [{String.t(), term()}]
  def llm_call_stop(metadata) do
    case metadata[:token_usage] do
      %{input: input, output: output} ->
        attrs = []
        attrs = if output, do: [{@usage_output_tokens, output} | attrs], else: attrs
        attrs = if input, do: [{@usage_input_tokens, input} | attrs], else: attrs
        attrs

      _ ->
        []
    end
  end

  @doc """
  Builds attributes for a tool call event.
  """
  @spec tool_call(map()) :: [{String.t(), term()}]
  def tool_call(metadata) do
    attrs = [
      {@operation_name, "execute_tool"},
      {@tool_type, "function"}
    ]

    attrs =
      case metadata[:tool_call_id] do
        nil -> attrs
        id -> [{@tool_call_id, id} | attrs]
      end

    case metadata[:tool_name] do
      nil -> attrs
      name -> [{@tool_name, name} | attrs]
    end
  end

  @doc """
  Builds attributes for a chain execution event.
  """
  @spec chain_start(map()) :: [{String.t(), term()}]
  def chain_start(metadata) do
    attrs = [{@operation_name, "invoke_agent"}]

    case metadata[:chain_type] do
      nil -> attrs
      chain_type -> [{@agent_name, chain_type} | attrs]
    end
  end
end
