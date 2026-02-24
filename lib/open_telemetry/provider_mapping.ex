defmodule LangChain.OpenTelemetry.ProviderMapping do
  @moduledoc """
  Maps LangChain provider strings to OpenTelemetry GenAI Semantic Convention
  `gen_ai.provider.name` values.

  See: https://opentelemetry.io/docs/specs/semconv/gen-ai/
  """

  @mapping %{
    "openai" => "openai",
    "anthropic" => "anthropic",
    "google" => "gcp.gemini",
    "vertex_ai" => "gcp.vertex_ai",
    "mistralai" => "mistral_ai",
    "deepseek" => "deepseek",
    "perplexity" => "perplexity",
    "xai" => "x_ai",
    "ollama" => "ollama"
  }

  @doc """
  Returns the OTel `gen_ai.provider.name` for a LangChain provider string.

  Unknown providers are passed through unchanged.

  ## Examples

      iex> LangChain.OpenTelemetry.ProviderMapping.to_otel("openai")
      "openai"

      iex> LangChain.OpenTelemetry.ProviderMapping.to_otel("google")
      "gcp.gemini"

      iex> LangChain.OpenTelemetry.ProviderMapping.to_otel("custom_provider")
      "custom_provider"
  """
  @spec to_otel(String.t()) :: String.t()
  def to_otel(provider) when is_binary(provider) do
    Map.get(@mapping, provider, provider)
  end
end
