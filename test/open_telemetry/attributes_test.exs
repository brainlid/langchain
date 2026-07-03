defmodule LangChain.OpenTelemetry.AttributesTest do
  use ExUnit.Case, async: true

  alias LangChain.Message
  alias LangChain.MessageDelta
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

    test "defaults gen_ai.output.type to text when metadata omits it" do
      attrs = Attributes.llm_call_start(%{model: "gpt-4o", provider: "openai"})

      assert {"gen_ai.output.type", "text"} in attrs
    end

    test "records gen_ai.output.type from metadata (json for structured output)" do
      attrs =
        Attributes.llm_call_start(%{model: "gpt-4o", provider: "openai", output_type: "json"})

      assert {"gen_ai.output.type", "json"} in attrs
      refute {"gen_ai.output.type", "text"} in attrs
    end

    test "derives server.address and server.port from the endpoint" do
      metadata = %{
        model: "gpt-4o",
        provider: "openai",
        endpoint: "https://api.openai.com/v1/chat/completions"
      }

      attrs = Attributes.llm_call_start(metadata)

      assert {"server.address", "api.openai.com"} in attrs
      # URI supplies the scheme default port; https -> 443.
      assert {"server.port", 443} in attrs
    end

    test "captures an explicit non-default endpoint port" do
      metadata = %{model: "llama", endpoint: "http://localhost:11434/api/chat"}

      attrs = Attributes.llm_call_start(metadata)

      assert {"server.address", "localhost"} in attrs
      assert {"server.port", 11434} in attrs
    end

    test "omits server.* attributes when no endpoint is present" do
      attrs = Attributes.llm_call_start(%{model: "gpt-4o", provider: "openai"})

      refute Enum.any?(attrs, fn {k, _v} -> k == "server.address" end)
      refute Enum.any?(attrs, fn {k, _v} -> k == "server.port" end)
    end

    test "maps request_options to gen_ai.request.* attributes" do
      metadata = %{
        model: "gpt-4o",
        provider: "openai",
        request_options: %{
          temperature: 0.7,
          max_tokens: 512,
          top_p: 0.9,
          top_k: 40,
          frequency_penalty: 0.1,
          presence_penalty: 0.2,
          seed: 42,
          choice_count: 2,
          stream: true,
          reasoning_level: "medium"
        }
      }

      attrs = Attributes.llm_call_start(metadata)

      assert {"gen_ai.request.temperature", 0.7} in attrs
      assert {"gen_ai.request.max_tokens", 512} in attrs
      assert {"gen_ai.request.top_p", 0.9} in attrs
      assert {"gen_ai.request.top_k", 40} in attrs
      assert {"gen_ai.request.frequency_penalty", 0.1} in attrs
      assert {"gen_ai.request.presence_penalty", 0.2} in attrs
      assert {"gen_ai.request.seed", 42} in attrs
      assert {"gen_ai.request.choice.count", 2} in attrs
      assert {"gen_ai.request.stream", true} in attrs
      assert {"gen_ai.request.reasoning.level", "medium"} in attrs
    end

    test "coerces a single stop string into gen_ai.request.stop_sequences array" do
      metadata = %{model: "m", provider: "openai", request_options: %{stop_sequences: "STOP"}}

      attrs = Attributes.llm_call_start(metadata)

      assert {"gen_ai.request.stop_sequences", ["STOP"]} in attrs
    end

    test "keeps a stop sequence list, dropping non-string entries" do
      metadata = %{
        model: "m",
        provider: "openai",
        request_options: %{stop_sequences: ["A", "B", 3]}
      }

      attrs = Attributes.llm_call_start(metadata)

      assert {"gen_ai.request.stop_sequences", ["A", "B"]} in attrs
    end

    test "stringifies an atom reasoning level" do
      # Some models carry :reasoning_effort as an atom rather than a string.
      metadata = %{model: "m", provider: "openai", request_options: %{reasoning_level: :high}}

      attrs = Attributes.llm_call_start(metadata)

      assert {"gen_ai.request.reasoning.level", "high"} in attrs
    end

    test "emits no gen_ai.request.* parameter attributes when request_options is empty/absent" do
      for metadata <- [
            %{model: "m", provider: "openai"},
            %{model: "m", provider: "openai", request_options: %{}}
          ] do
        attrs = Attributes.llm_call_start(metadata)
        params = Enum.filter(attrs, fn {k, _v} -> String.starts_with?(k, "gen_ai.request.") end)
        # Only gen_ai.request.model remains; no parameter attributes.
        assert params == [{"gen_ai.request.model", "m"}]
      end
    end

    test "drops request-option keys whose value is nil" do
      metadata = %{
        model: "m",
        provider: "openai",
        request_options: %{temperature: nil, seed: 7}
      }

      attrs = Attributes.llm_call_start(metadata)

      assert {"gen_ai.request.seed", 7} in attrs
      refute Enum.any?(attrs, fn {k, _v} -> k == "gen_ai.request.temperature" end)
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

    test "extracts Anthropic-style cache tokens from token_usage.raw" do
      metadata = %{
        token_usage: %{
          input: 100,
          output: 50,
          raw: %{"cache_creation_input_tokens" => 292, "cache_read_input_tokens" => 3604}
        }
      }

      attrs = Attributes.llm_call_stop(metadata)

      assert {"gen_ai.usage.cache_creation.input_tokens", 292} in attrs
      assert {"gen_ai.usage.cache_read.input_tokens", 3604} in attrs
    end

    test "extracts OpenAI-style nested cached and reasoning tokens from token_usage.raw" do
      metadata = %{
        token_usage: %{
          input: 100,
          output: 50,
          raw: %{
            "prompt_tokens_details" => %{"cached_tokens" => 64},
            "completion_tokens_details" => %{"reasoning_tokens" => 20}
          }
        }
      }

      attrs = Attributes.llm_call_stop(metadata)

      assert {"gen_ai.usage.cache_read.input_tokens", 64} in attrs
      assert {"gen_ai.usage.reasoning.output_tokens", 20} in attrs
    end

    test "drops zero and absent cache/reasoning counts" do
      metadata = %{
        token_usage: %{
          input: 100,
          output: 50,
          raw: %{"cache_creation_input_tokens" => 0, "cache_read_input_tokens" => 0}
        }
      }

      attrs = Attributes.llm_call_stop(metadata)

      refute Enum.any?(attrs, fn {k, _v} -> k == "gen_ai.usage.cache_creation.input_tokens" end)
      refute Enum.any?(attrs, fn {k, _v} -> k == "gen_ai.usage.cache_read.input_tokens" end)
      refute Enum.any?(attrs, fn {k, _v} -> k == "gen_ai.usage.reasoning.output_tokens" end)
    end

    test "derives gen_ai.response.finish_reasons stop from a completed message" do
      metadata = %{result: {:ok, Message.new_assistant!(%{content: "done"})}}

      attrs = Attributes.llm_call_stop(metadata)

      assert {"gen_ai.response.finish_reasons", ["stop"]} in attrs
    end

    test "derives finish_reasons length from a length-truncated message" do
      metadata = %{result: {:ok, Message.new_assistant!(%{content: "trunc", status: :length})}}

      attrs = Attributes.llm_call_stop(metadata)

      assert {"gen_ai.response.finish_reasons", ["length"]} in attrs
    end

    test "derives finish_reasons tool_calls when the message carries tool calls" do
      tool_call =
        LangChain.Message.ToolCall.new!(%{
          call_id: "call_1",
          name: "get_weather",
          arguments: %{}
        })

      msg = Message.new_assistant!(%{tool_calls: [tool_call]})
      metadata = %{result: {:ok, msg}}

      attrs = Attributes.llm_call_stop(metadata)

      assert {"gen_ai.response.finish_reasons", ["tool_calls"]} in attrs
    end

    test "omits finish_reasons when the result is not a message" do
      attrs = Attributes.llm_call_stop(%{result: {:error, "boom"}})

      refute Enum.any?(attrs, fn {k, _v} -> k == "gen_ai.response.finish_reasons" end)
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

    test "captures streamed output by merging a delta list into a message" do
      # A streaming call returns `{:ok, [%MessageDelta{} | _]}`, not a `%Message{}`.
      # Without merging, `gen_ai.output.messages` would be empty for streamed calls
      # even though input capture works — this guards that asymmetry.
      config = %Config{capture_output_messages: true}

      deltas = [
        MessageDelta.new!(%{role: :assistant, content: "Hel", status: :incomplete}),
        MessageDelta.new!(%{content: "lo", status: :incomplete}),
        MessageDelta.new!(%{content: "!", status: :complete})
      ]

      attrs = Attributes.llm_call_stop(%{result: {:ok, deltas}}, config)

      assert {_key, json} =
               Enum.find(attrs, fn {k, _v} -> k == "gen_ai.output.messages" end)

      assert [%{"role" => "assistant", "content" => "Hello!"}] = Jason.decode!(json)
    end

    test "skips output capture when a streamed delta list is incomplete" do
      # An interrupted stream can't be converted to a message; capture nothing
      # rather than raising or emitting a partial artifact.
      config = %Config{capture_output_messages: true}
      deltas = [MessageDelta.new!(%{role: :assistant, content: "Hel", status: :incomplete})]

      attrs = Attributes.llm_call_stop(%{result: {:ok, deltas}}, config)

      refute Enum.any?(attrs, fn {k, _v} -> k == "gen_ai.output.messages" end)
    end

    test "captures a single (non-list) streamed delta as output" do
      # A streaming call can surface a lone `%MessageDelta{}` rather than a list;
      # it should be captured the same as a delta list, not silently dropped.
      config = %Config{capture_output_messages: true}
      delta = MessageDelta.new!(%{role: :assistant, content: "Hi", status: :complete})

      attrs = Attributes.llm_call_stop(%{result: {:ok, delta}}, config)

      assert {_key, json} = Enum.find(attrs, fn {k, _v} -> k == "gen_ai.output.messages" end)
      assert [%{"role" => "assistant", "content" => "Hi"}] = Jason.decode!(json)
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

    test "includes gen_ai.tool.description when present" do
      metadata = %{tool_name: "get_weather", tool_description: "Look up the weather"}
      attrs = Attributes.tool_call(metadata)

      assert {"gen_ai.tool.description", "Look up the weather"} in attrs
    end

    test "omits gen_ai.tool.description when absent or blank" do
      for metadata <- [
            %{tool_name: "t"},
            %{tool_name: "t", tool_description: nil},
            %{tool_name: "t", tool_description: ""}
          ] do
        attrs = Attributes.tool_call(metadata)
        refute Enum.any?(attrs, fn {k, _v} -> k == "gen_ai.tool.description" end)
      end
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

    test "sets gen_ai.conversation.id from an explicit custom_context conversation_id" do
      metadata = %{chain_type: "llm_chain", custom_context: %{conversation_id: "conv-1"}}

      attrs = Attributes.chain_start(metadata)

      assert {"gen_ai.conversation.id", "conv-1"} in attrs
    end

    test "falls back to langfuse_session_id for gen_ai.conversation.id" do
      metadata = %{chain_type: "llm_chain", custom_context: %{langfuse_session_id: "sess-456"}}

      attrs = Attributes.chain_start(metadata)

      assert {"gen_ai.conversation.id", "sess-456"} in attrs
    end

    test "prefers an explicit conversation_id over langfuse_session_id" do
      metadata = %{
        chain_type: "llm_chain",
        custom_context: %{conversation_id: "conv-1", langfuse_session_id: "sess-456"}
      }

      attrs = Attributes.chain_start(metadata)

      assert {"gen_ai.conversation.id", "conv-1"} in attrs
    end

    test "omits gen_ai.conversation.id when no session key is present" do
      attrs = Attributes.chain_start(%{chain_type: "llm_chain", custom_context: %{foo: "bar"}})

      refute Enum.any?(attrs, fn {k, _v} -> k == "gen_ai.conversation.id" end)
    end
  end

  describe "chain_stop/2" do
    test "extracts the first user message as input for a standard {:ok, chain} result" do
      config = %Config{capture_input_messages: true}

      messages = [
        Message.new_system!("Be helpful"),
        Message.new_user!("first question"),
        Message.new_user!("second question")
      ]

      metadata = %{result: {:ok, %{messages: messages}}}

      attrs = Attributes.chain_stop(metadata, config)

      assert {"gen_ai.input.messages", json} =
               Enum.find(attrs, fn {k, _v} -> k == "gen_ai.input.messages" end)

      assert json =~ "first question"
      refute json =~ "second question"
    end

    test "extracts input from an :until_tool_used {:ok, chain, tool_result} 3-tuple result" do
      config = %Config{capture_input_messages: true}
      messages = [Message.new_user!("call the tool please")]
      # The `:until_tool_used` success path returns a 3-tuple; input capture must
      # still find the chain's messages instead of silently dropping them.
      metadata = %{result: {:ok, %{messages: messages}, :some_tool_result}}

      attrs = Attributes.chain_stop(metadata, config)

      assert {"gen_ai.input.messages", json} =
               Enum.find(attrs, fn {k, _v} -> k == "gen_ai.input.messages" end)

      assert json =~ "call the tool please"
    end

    test "captures the assistant last_message as output" do
      config = %Config{capture_output_messages: true}
      metadata = %{last_message: Message.new_assistant!("the answer is 42")}

      attrs = Attributes.chain_stop(metadata, config)

      assert {"gen_ai.output.messages", json} =
               Enum.find(attrs, fn {k, _v} -> k == "gen_ai.output.messages" end)

      assert json =~ "the answer is 42"
    end

    test "captures nothing when both capture flags are off (default)" do
      metadata = %{
        result: {:ok, %{messages: [Message.new_user!("hi")]}, :tool_result},
        last_message: Message.new_assistant!("hello")
      }

      assert Attributes.chain_stop(metadata, %Config{}) == []
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

    test "stringifies non-primitive metadata values without raising" do
      # A nested map/list value has no String.Chars implementation. If left to
      # `to_string/1` it raises, and — trapped by the span handler — silently drops
      # the whole chain span. It must be JSON-encoded instead, keeping the trace.
      attrs =
        Attributes.custom_context_attributes(%{
          langfuse_metadata: %{
            nested: %{a: 1},
            list: ["x", "y"],
            count: 3,
            flag: true
          }
        })

      assert {"langfuse.trace.metadata.nested", ~s({"a":1})} in attrs
      assert {"langfuse.trace.metadata.list", ~s(["x","y"])} in attrs
      assert {"langfuse.trace.metadata.count", "3"} in attrs
      assert {"langfuse.trace.metadata.flag", "true"} in attrs
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
