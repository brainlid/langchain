defmodule LangChain.OpenTelemetry.ProviderMappingTest do
  use ExUnit.Case, async: true

  alias LangChain.OpenTelemetry.ProviderMapping

  alias LangChain.ChatModels.ChatAnthropic
  alias LangChain.ChatModels.ChatAwsMantle
  alias LangChain.ChatModels.ChatBumblebee
  alias LangChain.ChatModels.ChatDeepSeek
  alias LangChain.ChatModels.ChatGoogleAI
  alias LangChain.ChatModels.ChatGrok
  alias LangChain.ChatModels.ChatMistralAI
  alias LangChain.ChatModels.ChatOllamaAI
  alias LangChain.ChatModels.ChatOpenAI
  alias LangChain.ChatModels.ChatOpenAIResponses
  alias LangChain.ChatModels.ChatOrq
  alias LangChain.ChatModels.ChatPerplexity
  alias LangChain.ChatModels.ChatVertexAI

  # The documented contract: every LangChain provider string mapped to its
  # GenAI-semantic-convention `gen_ai.provider.name`. Kept independent of the
  # production `@mapping` on purpose — it encodes the intended contract so drift
  # (a changed or removed mapping) fails here rather than silently.
  @expected_otel_names %{
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

  # Every chat model that participates in telemetry via the `provider/0` callback.
  # (ChatReqLLM is intentionally absent: it's a multi-provider adapter that derives
  # `:provider` per call from its model spec rather than via `provider/0`.)
  @provider_chat_models [
    ChatOpenAI,
    ChatOpenAIResponses,
    ChatAnthropic,
    ChatGoogleAI,
    ChatVertexAI,
    ChatMistralAI,
    ChatDeepSeek,
    ChatPerplexity,
    ChatGrok,
    ChatOllamaAI,
    ChatOrq,
    ChatBumblebee,
    ChatAwsMantle
  ]

  describe "to_otel/1" do
    test "maps known LangChain providers to OTel convention names" do
      assert ProviderMapping.to_otel("openai") == "openai"
      assert ProviderMapping.to_otel("openai_responses") == "openai"
      assert ProviderMapping.to_otel("anthropic") == "anthropic"
      assert ProviderMapping.to_otel("google") == "gcp.gemini"
      assert ProviderMapping.to_otel("vertex_ai") == "gcp.vertex_ai"
      assert ProviderMapping.to_otel("mistralai") == "mistral_ai"
      assert ProviderMapping.to_otel("deepseek") == "deepseek"
      assert ProviderMapping.to_otel("perplexity") == "perplexity"
      assert ProviderMapping.to_otel("xai") == "x_ai"
      assert ProviderMapping.to_otel("ollama") == "ollama"
      assert ProviderMapping.to_otel("orq") == "orq"
      assert ProviderMapping.to_otel("bumblebee") == "bumblebee"
      assert ProviderMapping.to_otel("aws_mantle") == "aws.bedrock"
    end

    test "passes through unknown providers unchanged" do
      assert ProviderMapping.to_otel("custom_provider") == "custom_provider"
      assert ProviderMapping.to_otel("my_llm") == "my_llm"
    end

    test "google maps to the specific gcp.gemini, not the generic gcp.gen_ai" do
      # Reconciliation decision: the spec defines both `gcp.gemini` (the Gemini
      # API, which ChatGoogleAI targets) and `gcp.gen_ai` (a generic Google
      # fallback). We deliberately emit the specific value. Do not "align" this to
      # the generic `gcp.gen_ai` — see ProviderMapping's moduledoc.
      assert ProviderMapping.to_otel("google") == "gcp.gemini"
      refute ProviderMapping.to_otel("google") == "gcp.gen_ai"
    end
  end

  describe "coverage of built-in chat models" do
    test "every chat model's provider/0 string maps to a documented OTel name" do
      for module <- @provider_chat_models do
        provider = module.provider()

        assert Map.has_key?(@expected_otel_names, provider),
               "#{inspect(module)} reports provider #{inspect(provider)}, which has no " <>
                 "documented OTel mapping. Add it to ProviderMapping (and this test) so it " <>
                 "does not silently pass through unmapped."

        assert ProviderMapping.to_otel(provider) == @expected_otel_names[provider],
               "#{inspect(module)} provider #{inspect(provider)} mapped to " <>
                 "#{inspect(ProviderMapping.to_otel(provider))}, expected " <>
                 "#{inspect(@expected_otel_names[provider])}"
      end
    end
  end
end
