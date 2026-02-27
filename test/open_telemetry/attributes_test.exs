defmodule LangChain.OpenTelemetry.AttributesTest do
  use ExUnit.Case, async: true

  alias LangChain.Message
  alias LangChain.OpenTelemetry.Attributes
  alias LangChain.OpenTelemetry.Config

  describe "llm_call_start/2" do
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

    test "does not include input messages (moved to prompt event handler)" do
      config = %Config{capture_input_messages: true}
      messages = [Message.new_system!("Be helpful"), Message.new_user!("Hello")]
      metadata = %{model: "gpt-4o", provider: "openai", messages: messages}

      attrs = Attributes.llm_call_start(metadata, config)

      refute Enum.any?(attrs, fn {k, _v} -> k == "gen_ai.input.messages" end)
    end
  end

  describe "llm_call_stop/2" do
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

    test "includes gen_ai.response.model" do
      metadata = %{model: "gpt-4o-2024-05-13", token_usage: %{input: 10, output: 5}}
      attrs = Attributes.llm_call_stop(metadata)

      assert {"gen_ai.response.model", "gpt-4o-2024-05-13"} in attrs
    end

    test "omits gen_ai.response.model when nil" do
      metadata = %{token_usage: %{input: 10, output: 5}}
      attrs = Attributes.llm_call_stop(metadata)

      refute Enum.any?(attrs, fn {k, _v} -> k == "gen_ai.response.model" end)
    end

    test "includes output messages when capture_output_messages is true" do
      config = %Config{capture_output_messages: true}
      msg = Message.new_assistant!(%{content: "Hello there!"})
      metadata = %{result: {:ok, msg}}

      attrs = Attributes.llm_call_stop(metadata, config)

      assert {_key, json} =
               Enum.find(attrs, fn {k, _v} -> k == "gen_ai.output.messages" end)

      decoded = Jason.decode!(json)
      assert [%{"role" => "assistant", "content" => "Hello there!"}] = decoded
    end

    test "omits output messages when capture_output_messages is false" do
      config = %Config{capture_output_messages: false}
      msg = Message.new_assistant!(%{content: "Hello there!"})
      metadata = %{result: {:ok, msg}}

      attrs = Attributes.llm_call_stop(metadata, config)

      refute Enum.any?(attrs, fn {k, _v} -> k == "gen_ai.output.messages" end)
    end
  end

  describe "tool_call/2" do
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

    test "includes arguments when capture_tool_arguments is true" do
      config = %Config{capture_tool_arguments: true}
      args = %{"x" => 1, "y" => 2}
      metadata = %{tool_name: "add", tool_call_id: "tc-1", arguments: args}

      attrs = Attributes.tool_call(metadata, config)

      assert {_key, json} =
               Enum.find(attrs, fn {k, _v} -> k == "gen_ai.tool.call.arguments" end)

      assert Jason.decode!(json) == args
    end

    test "omits arguments when capture_tool_arguments is false" do
      config = %Config{capture_tool_arguments: false}
      metadata = %{tool_name: "add", arguments: %{"x" => 1}}

      attrs = Attributes.tool_call(metadata, config)

      refute Enum.any?(attrs, fn {k, _v} -> k == "gen_ai.tool.call.arguments" end)
    end
  end

  describe "tool_call_stop/2" do
    test "includes result when capture_tool_results is true" do
      config = %Config{capture_tool_results: true}
      metadata = %{tool_result: %{content: "42"}}

      attrs = Attributes.tool_call_stop(metadata, config)

      assert {"gen_ai.tool.call.result", "42"} in attrs
    end

    test "returns empty when capture_tool_results is false" do
      config = %Config{capture_tool_results: false}
      metadata = %{tool_result: %{content: "42"}}

      assert Attributes.tool_call_stop(metadata, config) == []
    end

    test "returns empty when no tool_result in metadata" do
      config = %Config{capture_tool_results: true}

      assert Attributes.tool_call_stop(%{}, config) == []
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

    test "includes langfuse attributes from custom_context" do
      metadata = %{
        chain_type: "llm_chain",
        custom_context: %{
          langfuse_user_id: "user-123",
          langfuse_session_id: "sess-456"
        }
      }

      attrs = Attributes.chain_start(metadata)

      assert {"langfuse.user.id", "user-123"} in attrs
      assert {"langfuse.session.id", "sess-456"} in attrs
    end
  end

  describe "custom_context_attributes/1" do
    test "returns empty list for nil" do
      assert Attributes.custom_context_attributes(nil) == []
    end

    test "extracts langfuse_trace_name" do
      attrs = Attributes.custom_context_attributes(%{langfuse_trace_name: "chat_agent"})
      assert {"langfuse.trace.name", "chat_agent"} in attrs
    end

    test "extracts langfuse_user_id" do
      attrs = Attributes.custom_context_attributes(%{langfuse_user_id: "u-1"})
      assert {"langfuse.user.id", "u-1"} in attrs
    end

    test "extracts langfuse_session_id" do
      attrs = Attributes.custom_context_attributes(%{langfuse_session_id: "s-1"})
      assert {"langfuse.session.id", "s-1"} in attrs
    end

    test "extracts langfuse_tags as comma-separated string" do
      attrs = Attributes.custom_context_attributes(%{langfuse_tags: ["prod", "v2"]})
      assert {"langfuse.trace.tags", "prod,v2"} in attrs
    end

    test "flattens langfuse_metadata into individual attributes" do
      attrs =
        Attributes.custom_context_attributes(%{
          langfuse_metadata: %{env: "production", version: "1.0"}
        })

      assert {"langfuse.trace.metadata.env", "production"} in attrs
      assert {"langfuse.trace.metadata.version", "1.0"} in attrs
    end

    test "handles empty custom_context" do
      assert Attributes.custom_context_attributes(%{}) == []
    end

    test "ignores unknown keys" do
      attrs = Attributes.custom_context_attributes(%{unknown_key: "value"})
      assert attrs == []
    end
  end
end
