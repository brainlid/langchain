defmodule LangChain.OpenTelemetry.ProviderMappingTest do
  use ExUnit.Case, async: true

  alias LangChain.OpenTelemetry.ProviderMapping

  describe "to_otel/1" do
    test "maps known LangChain providers to OTel convention names" do
      assert ProviderMapping.to_otel("openai") == "openai"
      assert ProviderMapping.to_otel("anthropic") == "anthropic"
      assert ProviderMapping.to_otel("google") == "gcp.gemini"
      assert ProviderMapping.to_otel("vertex_ai") == "gcp.vertex_ai"
      assert ProviderMapping.to_otel("mistralai") == "mistral_ai"
      assert ProviderMapping.to_otel("deepseek") == "deepseek"
      assert ProviderMapping.to_otel("perplexity") == "perplexity"
      assert ProviderMapping.to_otel("xai") == "x_ai"
      assert ProviderMapping.to_otel("ollama") == "ollama"
    end

    test "passes through unknown providers unchanged" do
      assert ProviderMapping.to_otel("custom_provider") == "custom_provider"
      assert ProviderMapping.to_otel("my_llm") == "my_llm"
    end
  end
end
