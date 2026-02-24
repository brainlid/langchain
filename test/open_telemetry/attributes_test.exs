defmodule LangChain.OpenTelemetry.AttributesTest do
  use ExUnit.Case, async: true

  alias LangChain.OpenTelemetry.Attributes

  describe "llm_call_start/1" do
    test "builds basic LLM call start attributes" do
      metadata = %{model: "gpt-4o", provider: "openai"}
      attrs = Attributes.llm_call_start(metadata)

      assert {"gen_ai.operation.name", "chat"} in attrs
      assert {"gen_ai.request.model", "gpt-4o"} in attrs
      assert {"gen_ai.provider.name", "openai"} in attrs
    end

    test "maps provider through ProviderMapping" do
      metadata = %{model: "gemini-pro", provider: "google"}
      attrs = Attributes.llm_call_start(metadata)

      assert {"gen_ai.provider.name", "gcp.gemini"} in attrs
    end

    test "omits provider when nil" do
      metadata = %{model: "gpt-4o"}
      attrs = Attributes.llm_call_start(metadata)

      refute Enum.any?(attrs, fn {k, _v} -> k == "gen_ai.provider.name" end)
    end
  end

  describe "llm_call_stop/1" do
    test "extracts token usage" do
      metadata = %{token_usage: %{input: 100, output: 50}}
      attrs = Attributes.llm_call_stop(metadata)

      assert {"gen_ai.usage.input_tokens", 100} in attrs
      assert {"gen_ai.usage.output_tokens", 50} in attrs
    end

    test "returns empty list when no token usage" do
      assert Attributes.llm_call_stop(%{}) == []
      assert Attributes.llm_call_stop(%{token_usage: nil}) == []
    end

    test "omits nil input or output" do
      metadata = %{token_usage: %{input: 10, output: nil}}
      attrs = Attributes.llm_call_stop(metadata)

      assert {"gen_ai.usage.input_tokens", 10} in attrs
      refute Enum.any?(attrs, fn {k, _v} -> k == "gen_ai.usage.output_tokens" end)
    end
  end

  describe "tool_call/1" do
    test "builds tool call attributes" do
      metadata = %{tool_name: "calculator", tool_call_id: "call-123"}
      attrs = Attributes.tool_call(metadata)

      assert {"gen_ai.operation.name", "execute_tool"} in attrs
      assert {"gen_ai.tool.type", "function"} in attrs
      assert {"gen_ai.tool.name", "calculator"} in attrs
      assert {"gen_ai.tool.call.id", "call-123"} in attrs
    end

    test "omits optional fields when nil" do
      attrs = Attributes.tool_call(%{})

      assert {"gen_ai.operation.name", "execute_tool"} in attrs
      assert {"gen_ai.tool.type", "function"} in attrs
      refute Enum.any?(attrs, fn {k, _v} -> k == "gen_ai.tool.name" end)
      refute Enum.any?(attrs, fn {k, _v} -> k == "gen_ai.tool.call.id" end)
    end
  end

  describe "chain_start/1" do
    test "builds chain start attributes" do
      metadata = %{chain_type: "llm_chain"}
      attrs = Attributes.chain_start(metadata)

      assert {"gen_ai.operation.name", "invoke_agent"} in attrs
      assert {"gen_ai.agent.name", "llm_chain"} in attrs
    end

    test "omits agent name when chain_type is nil" do
      attrs = Attributes.chain_start(%{})

      assert {"gen_ai.operation.name", "invoke_agent"} in attrs
      refute Enum.any?(attrs, fn {k, _v} -> k == "gen_ai.agent.name" end)
    end
  end
end
