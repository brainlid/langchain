defmodule LangChain.OpenTelemetry.ProviderMapping do
  @moduledoc """
  Maps LangChain provider strings to OpenTelemetry GenAI Semantic Convention
  `gen_ai.provider.name` values.

  Values are verified against the well-known `gen_ai.provider.name` set in the
  GenAI semantic conventions. Providers with no well-known value (`ollama`,
  `orq`, `bumblebee`) intentionally pass through unchanged — the spec permits
  custom identifiers for providers it doesn't enumerate.

  ## `google` → `gcp.gemini` (reconciliation note)

  The spec defines two Google values: `gcp.gemini` ("Gemini", i.e. the Gemini
  API / Google AI Studio served from `generativelanguage.googleapis.com`) and
  `gcp.gen_ai` ("any Google generative AI endpoint", a generic fallback).
  `LangChain.ChatModels.ChatGoogleAI` targets the Gemini API specifically, so we
  deliberately emit the more precise `gcp.gemini` rather than the generic
  `gcp.gen_ai`. (The sibling `req_llm` library maps its `:google` provider to the
  generic `gcp.gen_ai`; the divergence is intentional — LangChain's value is the
  more specific, spec-correct one for the Gemini endpoint.) Vertex AI is a
  separate provider (`ChatVertexAI` → `gcp.vertex_ai`).

  See: https://opentelemetry.io/docs/specs/semconv/gen-ai/
  """

  @mapping %{
    "openai" => "openai",
    "openai_responses" => "openai",
    "anthropic" => "anthropic",
    "google" => "gcp.gemini",
    "vertex_ai" => "gcp.vertex_ai",
    "mistralai" => "mistral_ai",
    "deepseek" => "deepseek",
    "perplexity" => "perplexity",
    "xai" => "x_ai",
    "ollama" => "ollama",
    "orq" => "orq",
    "bumblebee" => "bumblebee",
    "aws_mantle" => "aws.bedrock"
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
