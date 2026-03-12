defmodule LangChain.ChatModels.ChatReqLLMTest do
  use LangChain.BaseCase
  use Mimic

  alias LangChain.ChatModels.ChatReqLLM
  alias LangChain.Chains.LLMChain
  alias LangChain.Message
  alias LangChain.Message.ContentPart
  alias LangChain.Message.ToolCall
  alias LangChain.Message.ToolResult
  alias LangChain.MessageDelta
  alias LangChain.TokenUsage
  alias LangChain.Function
  alias LangChain.LangChainError

  # The Anthropic-based req_llm model string for live tests
  @live_model "anthropic:claude-haiku-4-5"

  # ============================================================
  # Helpers for constructing fake ReqLLM structs in unit tests
  # ============================================================

  defp base_response_fields do
    %{
      id: "test-response-id",
      model: "claude-haiku-4-5",
      context: ReqLLM.Context.new([]),
      object: nil,
      stream?: false,
      stream: nil,
      provider_meta: %{},
      error: nil
    }
  end

  defp req_llm_text_response(text, finish_reason \\ :stop, usage \\ nil) do
    struct!(
      ReqLLM.Response,
      Map.merge(base_response_fields(), %{
        message: %ReqLLM.Message{
          role: :assistant,
          content: [ReqLLM.Message.ContentPart.text(text)],
          tool_calls: nil
        },
        finish_reason: finish_reason,
        usage: usage
      })
    )
  end

  defp req_llm_tool_call_response(tool_calls) do
    struct!(
      ReqLLM.Response,
      Map.merge(base_response_fields(), %{
        message: %ReqLLM.Message{
          role: :assistant,
          content: [],
          tool_calls: tool_calls
        },
        finish_reason: :tool_calls,
        usage: %{input_tokens: 20, output_tokens: 10, total_tokens: 30}
      })
    )
  end

  # ============================================================
  # new/1 and new!/1
  # ============================================================

  describe "new/1" do
    test "works with minimal attrs (model only)" do
      assert {:ok, %ChatReqLLM{} = model} =
               ChatReqLLM.new(%{model: "anthropic:claude-haiku-4-5"})

      assert model.model == "anthropic:claude-haiku-4-5"
      assert model.stream == false
      assert model.receive_timeout == 60_000
      assert model.provider_opts == %{}
      assert model.callbacks == []
      assert model.verbose_api == false
    end

    test "accepts string keys" do
      assert {:ok, %ChatReqLLM{} = model} =
               ChatReqLLM.new(%{"model" => "openai:gpt-4o"})

      assert model.model == "openai:gpt-4o"
    end

    test "accepts all optional fields" do
      assert {:ok, %ChatReqLLM{} = model} =
               ChatReqLLM.new(%{
                 model: "groq:llama-3.3-70b-versatile",
                 max_tokens: 1000,
                 temperature: 0.7,
                 stream: false,
                 receive_timeout: 30_000,
                 base_url: "http://localhost:11434",
                 provider_opts: %{"thinking" => %{"type" => "enabled"}},
                 verbose_api: true
               })

      assert model.max_tokens == 1000
      assert model.temperature == 0.7
      assert model.stream == false
      assert model.receive_timeout == 30_000
      assert model.base_url == "http://localhost:11434"
      assert model.provider_opts == %{"thinking" => %{"type" => "enabled"}}
      assert model.verbose_api == true
    end

    test "returns error when model is blank" do
      assert {:error, changeset} = ChatReqLLM.new(%{model: ""})
      refute changeset.valid?
      # empty string triggers required validation (can't be blank) before length check
      assert changeset.errors[:model]
    end

    test "returns error when model is nil" do
      assert {:error, changeset} = ChatReqLLM.new(%{model: nil})
      refute changeset.valid?
      assert {"can't be blank", _} = changeset.errors[:model]
    end

    test "returns error when temperature is out of range" do
      assert {:error, changeset} = ChatReqLLM.new(%{model: "anthropic:x", temperature: 3.0})
      refute changeset.valid?
      assert changeset.errors[:temperature]
    end

    test "returns error when max_tokens is zero or negative" do
      assert {:error, _} = ChatReqLLM.new(%{model: "anthropic:x", max_tokens: 0})
      assert {:error, _} = ChatReqLLM.new(%{model: "anthropic:x", max_tokens: -1})
    end

    test "new!/1 raises on invalid" do
      assert_raise LangChainError, fn ->
        ChatReqLLM.new!(%{model: nil})
      end
    end
  end

  # ============================================================
  # retry_on_fallback?/1
  # ============================================================

  describe "retry_on_fallback?/1" do
    test "returns true for rate_limit_exceeded" do
      assert ChatReqLLM.retry_on_fallback?(
               LangChainError.exception(type: "rate_limit_exceeded", message: "too fast")
             )
    end

    test "returns true for overloaded" do
      assert ChatReqLLM.retry_on_fallback?(
               LangChainError.exception(type: "overloaded", message: "busy")
             )
    end

    test "returns true for timeout" do
      assert ChatReqLLM.retry_on_fallback?(
               LangChainError.exception(type: "timeout", message: "timed out")
             )
    end

    test "returns false for authentication_error" do
      refute ChatReqLLM.retry_on_fallback?(
               LangChainError.exception(type: "authentication_error", message: "bad key")
             )
    end

    test "returns false for invalid_request_error" do
      refute ChatReqLLM.retry_on_fallback?(
               LangChainError.exception(type: "invalid_request_error", message: "bad request")
             )
    end

    test "returns false for unknown error type" do
      refute ChatReqLLM.retry_on_fallback?(
               LangChainError.exception(type: "something_else", message: "unknown")
             )
    end
  end

  # ============================================================
  # serialize_config/1 and restore_from_map/1
  # ============================================================

  describe "serialize_config/1" do
    test "serializes to a JSON-safe map with version" do
      model =
        ChatReqLLM.new!(%{
          model: "openai:gpt-4o",
          temperature: 0.5,
          max_tokens: 2000,
          stream: false
        })

      config = ChatReqLLM.serialize_config(model)
      assert is_map(config)
      assert config["version"] == 1
      assert config["model"] == "openai:gpt-4o"
      assert config["temperature"] == 0.5
      assert config["max_tokens"] == 2000
    end

    test "does not include api_key in serialized config" do
      model = ChatReqLLM.new!(%{model: "anthropic:claude-haiku-4-5", api_key: "secret-key"})
      config = ChatReqLLM.serialize_config(model)
      refute Map.has_key?(config, "api_key")
    end
  end

  describe "restore_from_map/1" do
    test "restores a model from serialized config" do
      model = ChatReqLLM.new!(%{model: "openai:gpt-4o", temperature: 0.7, max_tokens: 500})
      config = ChatReqLLM.serialize_config(model)

      assert {:ok, %ChatReqLLM{} = restored} = ChatReqLLM.restore_from_map(config)
      assert restored.model == "openai:gpt-4o"
      assert restored.temperature == 0.7
      assert restored.max_tokens == 500
    end
  end

  # ============================================================
  # translate_finish_reason/1
  # ============================================================

  describe "translate_finish_reason/1" do
    test "maps :stop to :complete" do
      assert :complete == ChatReqLLM.translate_finish_reason(:stop)
    end

    test "maps :tool_calls to :complete" do
      assert :complete == ChatReqLLM.translate_finish_reason(:tool_calls)
    end

    test "maps :length to :length" do
      assert :length == ChatReqLLM.translate_finish_reason(:length)
    end

    test "maps :content_filter to :complete" do
      assert :complete == ChatReqLLM.translate_finish_reason(:content_filter)
    end

    test "maps nil to :complete" do
      assert :complete == ChatReqLLM.translate_finish_reason(nil)
    end

    test "maps unknown to :complete with a warning" do
      assert :complete == ChatReqLLM.translate_finish_reason(:bizarre_unknown_reason)
    end
  end

  # ============================================================
  # translate_usage/1
  # ============================================================

  describe "translate_usage/1" do
    test "returns nil for nil usage" do
      assert nil == ChatReqLLM.translate_usage(nil)
    end

    test "maps atom-key usage map to TokenUsage" do
      usage = %{input_tokens: 100, output_tokens: 50, total_tokens: 150}
      result = ChatReqLLM.translate_usage(usage)
      assert %TokenUsage{input: 100, output: 50} = result
      assert result.raw == usage
    end

    test "maps usage with extra provider fields to TokenUsage" do
      usage = %{input_tokens: 80, output_tokens: 40, cached_tokens: 10}
      result = ChatReqLLM.translate_usage(usage)
      assert %TokenUsage{input: 80, output: 40} = result
      assert result.raw == usage
    end

    test "handles missing fields gracefully" do
      result = ChatReqLLM.translate_usage(%{})
      assert %TokenUsage{input: 0, output: 0} = result
    end
  end

  # ============================================================
  # content_part_to_req_llm/1  (outbound content translation)
  # ============================================================

  describe "content_part_to_req_llm/1" do
    test "translates :text ContentPart" do
      part = ContentPart.text!("Hello world")
      result = ChatReqLLM.content_part_to_req_llm(part)
      assert %ReqLLM.Message.ContentPart{type: :text, text: "Hello world"} = result
    end

    test "translates :thinking ContentPart" do
      part = ContentPart.new!(%{type: :thinking, content: "Let me think..."})
      result = ChatReqLLM.content_part_to_req_llm(part)
      assert %ReqLLM.Message.ContentPart{type: :thinking, text: "Let me think..."} = result
    end

    test "translates :image_url ContentPart" do
      part = ContentPart.new!(%{type: :image_url, content: "https://example.com/img.png"})
      result = ChatReqLLM.content_part_to_req_llm(part)

      assert %ReqLLM.Message.ContentPart{type: :image_url, url: "https://example.com/img.png"} =
               result
    end

    test "translates :image ContentPart with base64 data" do
      raw_bytes = <<1, 2, 3, 4, 5>>
      b64 = Base.encode64(raw_bytes)
      part = ContentPart.new!(%{type: :image, content: b64, options: [media: :png]})
      result = ChatReqLLM.content_part_to_req_llm(part)

      assert %ReqLLM.Message.ContentPart{type: :image, data: ^raw_bytes, media_type: "image/png"} =
               result
    end

    test "translates :file ContentPart with base64 data" do
      raw_bytes = <<10, 20, 30>>
      b64 = Base.encode64(raw_bytes)

      part =
        ContentPart.new!(%{
          type: :file,
          content: b64,
          options: [media: :pdf, filename: "doc.pdf"]
        })

      result = ChatReqLLM.content_part_to_req_llm(part)

      assert %ReqLLM.Message.ContentPart{
               type: :file,
               data: ^raw_bytes,
               media_type: "application/pdf",
               filename: "doc.pdf"
             } = result
    end

    test "translates :file_url ContentPart to text (unsupported)" do
      part = ContentPart.new!(%{type: :file_url, content: "https://example.com/doc.pdf"})
      result = ChatReqLLM.content_part_to_req_llm(part)
      assert %ReqLLM.Message.ContentPart{type: :text} = result
      assert String.contains?(result.text, "https://example.com/doc.pdf")
    end

    test "returns nil for :unsupported ContentPart" do
      part = ContentPart.new!(%{type: :unsupported, content: "some data"})
      result = ChatReqLLM.content_part_to_req_llm(part)
      assert nil == result
    end
  end

  # ============================================================
  # message_to_req_llm_messages/1  (outbound message translation)
  # ============================================================

  describe "message_to_req_llm_messages/1" do
    test "translates a system message with string content" do
      msg = Message.new_system!("You are helpful.")
      [result] = ChatReqLLM.message_to_req_llm_messages(msg)
      assert %ReqLLM.Message{role: :system} = result
      assert [%ReqLLM.Message.ContentPart{type: :text, text: "You are helpful."}] = result.content
    end

    test "translates a user message with string content" do
      msg = Message.new_user!("Hello!")
      [result] = ChatReqLLM.message_to_req_llm_messages(msg)
      assert %ReqLLM.Message{role: :user} = result
      assert [%ReqLLM.Message.ContentPart{type: :text, text: "Hello!"}] = result.content
    end

    test "translates a user message with multiple ContentParts" do
      msg =
        Message.new_user!([
          ContentPart.text!("Look at this:"),
          ContentPart.new!(%{type: :image_url, content: "https://example.com/img.png"})
        ])

      [result] = ChatReqLLM.message_to_req_llm_messages(msg)
      assert %ReqLLM.Message{role: :user} = result
      assert length(result.content) == 2
      assert Enum.at(result.content, 0).type == :text
      assert Enum.at(result.content, 1).type == :image_url
    end

    test "translates an assistant message with text content" do
      msg = Message.new_assistant!(%{content: "I can help with that."})
      [result] = ChatReqLLM.message_to_req_llm_messages(msg)
      assert %ReqLLM.Message{role: :assistant} = result

      assert [%ReqLLM.Message.ContentPart{type: :text, text: "I can help with that."}] =
               result.content
    end

    test "translates an assistant message with tool calls" do
      tool_call =
        ToolCall.new!(%{
          type: :function,
          status: :complete,
          call_id: "call_abc",
          name: "get_weather",
          arguments: %{"city" => "Paris"}
        })

      msg = Message.new_assistant!(%{tool_calls: [tool_call]})
      [result] = ChatReqLLM.message_to_req_llm_messages(msg)

      assert %ReqLLM.Message{role: :assistant} = result
      assert [%ReqLLM.ToolCall{id: "call_abc"} = req_call] = result.tool_calls
      assert req_call.function.name == "get_weather"
      assert req_call.function.arguments == ~s({"city":"Paris"})
    end

    test "expands a :tool message to one ReqLLM message per ToolResult" do
      results = [
        ToolResult.new!(%{tool_call_id: "call_1", content: "Result A"}),
        ToolResult.new!(%{tool_call_id: "call_2", content: "Result B"})
      ]

      msg = Message.new!(%{role: :tool, tool_results: results})
      req_messages = ChatReqLLM.message_to_req_llm_messages(msg)

      assert length(req_messages) == 2

      [msg_a, msg_b] = req_messages
      assert msg_a.role == :tool
      assert msg_a.tool_call_id == "call_1"
      assert [%ReqLLM.Message.ContentPart{type: :text, text: "Result A"}] = msg_a.content

      assert msg_b.role == :tool
      assert msg_b.tool_call_id == "call_2"
      assert [%ReqLLM.Message.ContentPart{type: :text, text: "Result B"}] = msg_b.content
    end
  end

  # ============================================================
  # messages_to_req_llm_context/1
  # ============================================================

  describe "messages_to_req_llm_context/1" do
    test "converts a list of messages to a ReqLLM.Context" do
      messages = [
        Message.new_system!("You are helpful."),
        Message.new_user!("Hi"),
        Message.new_assistant!(%{content: "Hello!"})
      ]

      context = ChatReqLLM.messages_to_req_llm_context(messages)
      assert %ReqLLM.Context{} = context
      assert length(context.messages) == 3
    end

    test "expands tool result messages correctly" do
      results = [
        ToolResult.new!(%{tool_call_id: "c1", content: "R1"}),
        ToolResult.new!(%{tool_call_id: "c2", content: "R2"})
      ]

      messages = [
        Message.new_user!("Use tools"),
        Message.new!(%{role: :tool, tool_results: results})
      ]

      context = ChatReqLLM.messages_to_req_llm_context(messages)
      # 1 user + 2 expanded tool results = 3 total
      assert length(context.messages) == 3
      assert Enum.at(context.messages, 1).role == :tool
      assert Enum.at(context.messages, 2).role == :tool
    end
  end

  # ============================================================
  # function_to_req_llm_tool/1
  # ============================================================

  describe "function_to_req_llm_tool/1" do
    test "creates a ReqLLM.Tool with correct name and description" do
      fun =
        Function.new!(%{
          name: "get_weather",
          description: "Get current weather",
          parameters_schema: %{
            "type" => "object",
            "properties" => %{"city" => %{"type" => "string"}},
            "required" => ["city"]
          },
          function: fn _, _ -> {:ok, "sunny"} end
        })

      tool = ChatReqLLM.function_to_req_llm_tool(fun)

      assert %ReqLLM.Tool{} = tool
      assert tool.name == "get_weather"
      assert tool.description == "Get current weather"
      assert tool.parameter_schema == fun.parameters_schema
    end

    test "stub callback returns {:ok, stub}" do
      fun =
        Function.new!(%{
          name: "noop",
          description: "Does nothing",
          function: fn _, _ -> {:ok, "real"} end
        })

      tool = ChatReqLLM.function_to_req_llm_tool(fun)
      assert {:ok, "stub"} == tool.callback.(%{})
    end

    test "functions_to_req_llm_tools returns empty list for nil" do
      assert [] == ChatReqLLM.functions_to_req_llm_tools(nil)
    end

    test "functions_to_req_llm_tools returns empty list for empty list" do
      assert [] == ChatReqLLM.functions_to_req_llm_tools([])
    end

    test "functions_to_req_llm_tools converts multiple functions" do
      funs =
        Enum.map(1..3, fn i ->
          Function.new!(%{
            name: "tool_#{i}",
            description: "Tool #{i}",
            function: fn _, _ -> {:ok, "ok"} end
          })
        end)

      tools = ChatReqLLM.functions_to_req_llm_tools(funs)
      assert length(tools) == 3
      assert Enum.all?(tools, &match?(%ReqLLM.Tool{}, &1))
    end
  end

  # ============================================================
  # do_process_response/2  (inbound response processing)
  # ============================================================

  describe "do_process_response/2" do
    setup do
      {:ok, model: ChatReqLLM.new!(%{model: "anthropic:claude-haiku-4-5"})}
    end

    test "converts a simple text response to a LangChain Message", %{model: model} do
      response = req_llm_text_response("Hello there!")

      result = ChatReqLLM.do_process_response(model, response)

      assert %Message{role: :assistant} = result
      assert [%ContentPart{type: :text, content: "Hello there!"}] = result.content
      assert result.status == :complete
      assert result.tool_calls == nil or result.tool_calls == []
    end

    test "maps finish_reason :length to status :length", %{model: model} do
      response = req_llm_text_response("Truncated...", :length)
      result = ChatReqLLM.do_process_response(model, response)
      assert result.status == :length
    end

    test "maps token usage to message metadata", %{model: model} do
      usage = %{input_tokens: 100, output_tokens: 50, total_tokens: 150}
      response = req_llm_text_response("Hello!", :stop, usage)

      result = ChatReqLLM.do_process_response(model, response)
      assert %TokenUsage{input: 100, output: 50} = result.metadata[:usage]
    end

    test "returns error when response has error field set", %{model: model} do
      response =
        struct!(
          ReqLLM.Response,
          Map.merge(base_response_fields(), %{
            message: nil,
            finish_reason: :error,
            error: %{reason: "something went wrong"},
            usage: nil
          })
        )

      assert {:error, %LangChainError{type: "api_error"}} =
               ChatReqLLM.do_process_response(model, response)
    end

    test "returns error when response has no message", %{model: model} do
      response =
        struct!(
          ReqLLM.Response,
          Map.merge(base_response_fields(), %{
            message: nil,
            finish_reason: nil,
            error: nil,
            usage: nil
          })
        )

      assert {:error, %LangChainError{type: "unexpected_response"}} =
               ChatReqLLM.do_process_response(model, response)
    end

    test "converts tool call response to LangChain Message with tool_calls", %{model: model} do
      req_tool_calls = [
        ReqLLM.ToolCall.new("call_abc", "get_weather", ~s({"city":"Paris"}))
      ]

      response = req_llm_tool_call_response(req_tool_calls)
      result = ChatReqLLM.do_process_response(model, response)

      assert %Message{role: :assistant} = result
      assert [%ToolCall{} = tc] = result.tool_calls
      assert tc.call_id == "call_abc"
      assert tc.name == "get_weather"
      assert tc.arguments == %{"city" => "Paris"}
      assert tc.status == :complete
      assert tc.type == :function
    end

    test "handles multiple tool calls", %{model: model} do
      req_tool_calls = [
        ReqLLM.ToolCall.new("c1", "search", ~s({"query":"elixir"})),
        ReqLLM.ToolCall.new("c2", "calculate", ~s({"expr":"2+2"}))
      ]

      response = req_llm_tool_call_response(req_tool_calls)
      result = ChatReqLLM.do_process_response(model, response)

      assert length(result.tool_calls) == 2
      assert Enum.at(result.tool_calls, 0).name == "search"
      assert Enum.at(result.tool_calls, 1).name == "calculate"
    end

    test "handles empty tool_calls list as nil tool_calls in output", %{model: model} do
      response =
        struct!(
          ReqLLM.Response,
          Map.merge(base_response_fields(), %{
            message: %ReqLLM.Message{
              role: :assistant,
              content: [ReqLLM.Message.ContentPart.text("No tools needed")],
              tool_calls: []
            },
            finish_reason: :stop,
            usage: nil
          })
        )

      result = ChatReqLLM.do_process_response(model, response)

      # Empty [] from ReqLLM maps to nil/[] in LangChain (our translate_response_tool_calls returns nil for [])
      assert result.tool_calls == nil or result.tool_calls == []
    end
  end

  # ============================================================
  # call/3  (mocked unit tests)
  # ============================================================

  describe "call/3 (mocked)" do
    setup do
      model = ChatReqLLM.new!(%{model: "anthropic:claude-haiku-4-5"})
      {:ok, model: model}
    end

    test "returns {:ok, message} on successful text response", %{model: model} do
      stub(ReqLLM, :generate_text, fn _model_spec, _context, _opts ->
        {:ok, req_llm_text_response("Hello from the LLM!")}
      end)

      messages = [Message.new_user!("Say hello")]
      assert {:ok, %Message{role: :assistant} = msg} = ChatReqLLM.call(model, messages, [])
      assert [%ContentPart{type: :text, content: "Hello from the LLM!"}] = msg.content
    end

    test "converts a string prompt to messages before calling", %{model: model} do
      stub(ReqLLM, :generate_text, fn _model_spec, context, _opts ->
        # Verify context has system + user messages
        assert length(context.messages) == 2
        {:ok, req_llm_text_response("Hello!")}
      end)

      assert {:ok, %Message{}} = ChatReqLLM.call(model, "Hi there", [])
    end

    test "passes tools as ReqLLM tools in opts", %{model: model} do
      fun =
        Function.new!(%{
          name: "get_weather",
          description: "Get weather",
          parameters_schema: %{"type" => "object", "properties" => %{}},
          function: fn _, _ -> {:ok, "sunny"} end
        })

      stub(ReqLLM, :generate_text, fn _model_spec, _context, opts ->
        tools = Keyword.get(opts, :tools, [])
        assert length(tools) == 1
        assert hd(tools).name == "get_weather"
        {:ok, req_llm_text_response("The weather is nice.")}
      end)

      messages = [Message.new_user!("What's the weather?")]
      assert {:ok, %Message{}} = ChatReqLLM.call(model, messages, [fun])
    end

    test "returns {:error, LangChainError} when ReqLLM returns an error", %{model: model} do
      stub(ReqLLM, :generate_text, fn _model_spec, _context, _opts ->
        {:error, %{status: 401, message: "Unauthorized"}}
      end)

      messages = [Message.new_user!("Hello")]

      assert {:error, %LangChainError{type: "authentication_error"}} =
               ChatReqLLM.call(model, messages, [])
    end

    test "passes max_tokens and temperature as opts", %{model: model} do
      model_with_opts = %{model | max_tokens: 500, temperature: 0.3}

      stub(ReqLLM, :generate_text, fn _model_spec, _context, opts ->
        assert Keyword.get(opts, :max_tokens) == 500
        assert Keyword.get(opts, :temperature) == 0.3
        {:ok, req_llm_text_response("OK")}
      end)

      assert {:ok, _} = ChatReqLLM.call(model_with_opts, "Test", [])
    end

    test "does not pass nil fields to opts", %{model: model} do
      # model has nil max_tokens and nil temperature by default
      stub(ReqLLM, :generate_text, fn _model_spec, _context, opts ->
        refute Keyword.has_key?(opts, :max_tokens)
        refute Keyword.has_key?(opts, :temperature)
        {:ok, req_llm_text_response("OK")}
      end)

      assert {:ok, _} = ChatReqLLM.call(model, "Test", [])
    end

    test "retries on connection closed error", %{model: model} do
      call_count = :counters.new(1, [])

      stub(ReqLLM, :generate_text, fn _model_spec, _context, _opts ->
        count = :counters.get(call_count, 1) + 1
        :counters.put(call_count, 1, count)

        if count < 2 do
          {:error, %Req.TransportError{reason: :closed}}
        else
          {:ok, req_llm_text_response("Recovered!")}
        end
      end)

      assert {:ok, %Message{}} = ChatReqLLM.call(model, "Test", [])
    end

    test "fires on_llm_new_message callback on success", %{model: model} do
      test_pid = self()

      model_with_cb = %{
        model
        | callbacks: [%{on_llm_new_message: fn msg -> send(test_pid, {:got_msg, msg}) end}]
      }

      stub(ReqLLM, :generate_text, fn _, _, _ ->
        {:ok, req_llm_text_response("Callback test")}
      end)

      ChatReqLLM.call(model_with_cb, "Test", [])
      assert_receive {:got_msg, %Message{role: :assistant}}, 1_000
    end
  end

  # ============================================================
  # Integration with LLMChain (mocked)
  # ============================================================

  describe "LLMChain integration (mocked)" do
    test "works end-to-end with LLMChain" do
      model = ChatReqLLM.new!(%{model: "anthropic:claude-haiku-4-5"})

      stub(ReqLLM, :generate_text, fn _, _, _ ->
        {:ok, req_llm_text_response("Two plus two is four.")}
      end)

      {:ok, chain} =
        %{llm: model}
        |> LLMChain.new!()
        |> LLMChain.add_message(Message.new_user!("What is 2+2?"))
        |> LLMChain.run()

      response = List.last(chain.messages)
      assert %Message{role: :assistant} = response
      assert is_list(response.content)
      assert hd(response.content).content =~ "four"
    end
  end

  # ============================================================
  # Streaming helpers
  # ============================================================

  # Build a fake ReqLLM.StreamResponse for unit tests.
  # All 5 enforce_keys are required; metadata_handle is a PID (won't be called).
  defp fake_stream_response(chunks) do
    %ReqLLM.StreamResponse{
      stream: chunks,
      metadata_handle: self(),
      cancel: fn -> :ok end,
      model: nil,
      context: ReqLLM.Context.new([])
    }
  end

  # ============================================================
  # translate_stream_chunk/1
  # ============================================================

  describe "translate_stream_chunk/1" do
    test "translates a content chunk to a text MessageDelta" do
      chunk = %ReqLLM.StreamChunk{type: :content, text: "Hello"}
      [delta] = ChatReqLLM.translate_stream_chunk(chunk)
      assert %MessageDelta{role: :assistant, status: :incomplete, index: 0} = delta
      assert %ContentPart{type: :text, content: "Hello"} = delta.content
    end

    test "returns [] for empty content chunk" do
      chunk = %ReqLLM.StreamChunk{type: :content, text: ""}
      assert [] = ChatReqLLM.translate_stream_chunk(chunk)
    end

    test "returns [] for nil content chunk" do
      chunk = %ReqLLM.StreamChunk{type: :content, text: nil}
      assert [] = ChatReqLLM.translate_stream_chunk(chunk)
    end

    test "translates a thinking chunk to a thinking MessageDelta" do
      chunk = %ReqLLM.StreamChunk{type: :thinking, text: "I am reasoning"}
      [delta] = ChatReqLLM.translate_stream_chunk(chunk)
      assert %MessageDelta{role: :assistant, status: :incomplete} = delta
      assert %ContentPart{type: :thinking, content: "I am reasoning"} = delta.content
    end

    test "translates a tool_call chunk to a tool_calls MessageDelta" do
      chunk = %ReqLLM.StreamChunk{
        type: :tool_call,
        name: "get_weather",
        arguments: %{"city" => "Paris"},
        metadata: %{id: "call_abc"}
      }

      [delta] = ChatReqLLM.translate_stream_chunk(chunk)
      assert %MessageDelta{role: :assistant, status: :incomplete} = delta
      [tc] = delta.tool_calls
      assert %ToolCall{name: "get_weather", call_id: "call_abc", status: :complete} = tc
      assert tc.arguments == %{"city" => "Paris"}
    end

    test "tool_call chunk generates a non-empty fallback id when metadata has no id" do
      chunk = %ReqLLM.StreamChunk{
        type: :tool_call,
        name: "search",
        arguments: %{},
        metadata: %{}
      }

      [delta] = ChatReqLLM.translate_stream_chunk(chunk)
      [tc] = delta.tool_calls
      assert is_binary(tc.call_id) && tc.call_id != ""
    end

    test "terminal meta chunk produces a complete MessageDelta" do
      chunk = %ReqLLM.StreamChunk{
        type: :meta,
        metadata: %{finish_reason: :stop, terminal?: true}
      }

      [delta] = ChatReqLLM.translate_stream_chunk(chunk)
      assert %MessageDelta{role: :assistant, status: :complete} = delta
    end

    test "terminal meta with :length finish reason produces :length status" do
      chunk = %ReqLLM.StreamChunk{
        type: :meta,
        metadata: %{finish_reason: :length, terminal?: true}
      }

      [delta] = ChatReqLLM.translate_stream_chunk(chunk)
      assert delta.status == :length
    end

    test "meta chunk with usage produces a usage MessageDelta" do
      chunk = %ReqLLM.StreamChunk{
        type: :meta,
        metadata: %{usage: %{input_tokens: 10, output_tokens: 5}}
      }

      [delta] = ChatReqLLM.translate_stream_chunk(chunk)
      assert %MessageDelta{} = delta
      assert %TokenUsage{input: 10, output: 5} = delta.metadata[:usage]
    end

    test "terminal meta with usage produces both usage and finish deltas" do
      chunk = %ReqLLM.StreamChunk{
        type: :meta,
        metadata: %{
          finish_reason: :stop,
          terminal?: true,
          usage: %{input_tokens: 20, output_tokens: 10}
        }
      }

      deltas = ChatReqLLM.translate_stream_chunk(chunk)
      assert length(deltas) == 2
      usage_delta = Enum.find(deltas, &(&1.metadata != nil))
      finish_delta = Enum.find(deltas, &(&1.status == :complete))
      assert %TokenUsage{input: 20, output: 10} = usage_delta.metadata[:usage]
      assert finish_delta.status == :complete
    end

    test "non-terminal meta chunk with no usage produces []" do
      chunk = %ReqLLM.StreamChunk{
        type: :meta,
        metadata: %{some_info: "value"}
      }

      assert [] = ChatReqLLM.translate_stream_chunk(chunk)
    end

    test "unknown chunk type produces []" do
      chunk = %ReqLLM.StreamChunk{type: :content, text: nil}
      assert [] = ChatReqLLM.translate_stream_chunk(chunk)
    end
  end

  # ============================================================
  # do_api_request streaming (mocked)
  # ============================================================

  describe "do_api_request/4 streaming (mocked)" do
    test "returns a flat list of MessageDeltas from stream chunks" do
      model = ChatReqLLM.new!(%{model: @live_model, stream: true})

      chunks = [
        %ReqLLM.StreamChunk{type: :content, text: "Hello"},
        %ReqLLM.StreamChunk{type: :content, text: " world"},
        %ReqLLM.StreamChunk{type: :meta, metadata: %{finish_reason: :stop, terminal?: true}}
      ]

      stub(ReqLLM, :stream_text, fn _model, _context, _opts ->
        {:ok, fake_stream_response(chunks)}
      end)

      result = ChatReqLLM.do_api_request(model, [Message.new_user!("hi")], [], 3)
      assert is_list(result)
      assert length(result) == 3

      text_deltas = Enum.filter(result, &(&1.content != nil))
      assert length(text_deltas) == 2

      final_delta = List.last(result)
      assert final_delta.status == :complete
    end

    test "fires on_llm_new_delta callback for each chunk" do
      test_pid = self()

      model =
        %{
          ChatReqLLM.new!(%{model: @live_model, stream: true})
          | callbacks: [%{on_llm_new_delta: fn deltas -> send(test_pid, {:delta, deltas}) end}]
        }

      chunks = [
        %ReqLLM.StreamChunk{type: :content, text: "Hi"},
        %ReqLLM.StreamChunk{type: :meta, metadata: %{finish_reason: :stop, terminal?: true}}
      ]

      stub(ReqLLM, :stream_text, fn _model, _context, _opts ->
        {:ok, fake_stream_response(chunks)}
      end)

      ChatReqLLM.do_api_request(model, [Message.new_user!("hi")], [], 3)

      assert_received {:delta, [%MessageDelta{status: :incomplete}]}
      assert_received {:delta, [%MessageDelta{status: :complete}]}
    end

    test "translates tool call chunks correctly" do
      model = ChatReqLLM.new!(%{model: @live_model, stream: true})

      chunks = [
        %ReqLLM.StreamChunk{
          type: :tool_call,
          name: "get_weather",
          arguments: %{"city" => "Paris"},
          metadata: %{id: "call_1"}
        },
        %ReqLLM.StreamChunk{type: :meta, metadata: %{finish_reason: :tool_calls, terminal?: true}}
      ]

      stub(ReqLLM, :stream_text, fn _model, _context, _opts ->
        {:ok, fake_stream_response(chunks)}
      end)

      result = ChatReqLLM.do_api_request(model, [Message.new_user!("weather?")], [], 3)

      tool_delta = Enum.find(result, &(&1.tool_calls != nil and &1.tool_calls != []))
      assert tool_delta != nil
      [tc] = tool_delta.tool_calls
      assert tc.name == "get_weather"
      assert tc.arguments == %{"city" => "Paris"}
    end

    test "returns {:error, LangChainError} when stream_text fails" do
      model = ChatReqLLM.new!(%{model: @live_model, stream: true})

      stub(ReqLLM, :stream_text, fn _model, _context, _opts ->
        {:error, %{status: 401}}
      end)

      assert {:error, %LangChainError{type: "authentication_error"}} =
               ChatReqLLM.do_api_request(model, [Message.new_user!("hi")], [], 3)
    end

    test "call/3 with stream:true returns {:ok, [MessageDelta]}" do
      model = ChatReqLLM.new!(%{model: @live_model, stream: true})

      chunks = [
        %ReqLLM.StreamChunk{type: :content, text: "Streaming!"},
        %ReqLLM.StreamChunk{type: :meta, metadata: %{finish_reason: :stop, terminal?: true}}
      ]

      stub(ReqLLM, :stream_text, fn _model, _context, _opts ->
        {:ok, fake_stream_response(chunks)}
      end)

      assert {:ok, deltas} = ChatReqLLM.call(model, [Message.new_user!("hi")], [])
      assert is_list(deltas)
      assert Enum.all?(deltas, &match?(%MessageDelta{}, &1))
    end

    test "LLMChain runs successfully with stream:true" do
      model = ChatReqLLM.new!(%{model: @live_model, stream: true})

      chunks = [
        %ReqLLM.StreamChunk{type: :content, text: "Hello "},
        %ReqLLM.StreamChunk{type: :content, text: "there!"},
        %ReqLLM.StreamChunk{type: :meta, metadata: %{finish_reason: :stop, terminal?: true}}
      ]

      stub(ReqLLM, :stream_text, fn _model, _context, _opts ->
        {:ok, fake_stream_response(chunks)}
      end)

      assert {:ok, chain} =
               %{llm: model}
               |> LLMChain.new!()
               |> LLMChain.add_message(Message.new_user!("Say hello"))
               |> LLMChain.run()

      last_msg = List.last(chain.messages)
      assert last_msg.role == :assistant

      text =
        last_msg.content
        |> Enum.filter(&(&1.type == :text))
        |> Enum.map_join("", & &1.content)

      assert text == "Hello there!"
    end
  end

  # ============================================================
  # Live API Tests (tagged :live_call — excluded by default)
  # ============================================================

  @tag :live_call
  @tag :live_anthropic
  @tag live_api: :anthropic
  test "live: simple text generation via Anthropic" do
    model = ChatReqLLM.new!(%{model: @live_model})

    messages = [Message.new_user!("Reply with exactly the text: HELLO_WORLD")]

    assert {:ok, %Message{role: :assistant} = response} =
             ChatReqLLM.call(model, messages, [])

    content_text =
      response.content
      |> Enum.filter(&(&1.type == :text))
      |> Enum.map_join("", & &1.content)

    assert String.contains?(content_text, "HELLO_WORLD")

    # Capture the raw response shape for offline test fixtures
    IO.inspect(response, label: "LIVE RESPONSE SHAPE")
  end

  @tag :live_call
  @tag :live_anthropic
  test "live: token usage is populated" do
    model = ChatReqLLM.new!(%{model: @live_model})
    messages = [Message.new_user!("Say hi")]

    assert {:ok, %Message{} = response} = ChatReqLLM.call(model, messages, [])

    assert %TokenUsage{} = response.metadata[:usage]
    assert response.metadata[:usage].input > 0
    assert response.metadata[:usage].output > 0

    IO.inspect(response.metadata[:usage], label: "LIVE TOKEN USAGE")
  end

  @tag :live_call
  @tag :live_anthropic
  test "live: tool calling round-trip" do
    weather_fn =
      Function.new!(%{
        name: "get_weather",
        description: "Get the current weather for a city",
        parameters_schema: %{
          "type" => "object",
          "properties" => %{
            "city" => %{"type" => "string", "description" => "The city name"}
          },
          "required" => ["city"]
        },
        function: fn %{"city" => city}, _ctx -> {:ok, "Sunny in #{city}, 22°C"} end
      })

    model = ChatReqLLM.new!(%{model: @live_model})

    messages = [Message.new_user!("What's the weather in Paris? Use the get_weather tool.")]

    assert {:ok, %Message{role: :assistant} = response} =
             ChatReqLLM.call(model, messages, [weather_fn])

    # Should have tool calls
    assert is_list(response.tool_calls)
    assert length(response.tool_calls) > 0

    [tc | _] = response.tool_calls
    assert tc.name == "get_weather"
    assert is_map(tc.arguments)
    assert Map.has_key?(tc.arguments, "city")

    IO.inspect(response, label: "LIVE TOOL CALL RESPONSE SHAPE")
  end

  @tag :live_call
  @tag :live_anthropic
  test "live: LLMChain with tool use completes successfully" do
    weather_fn =
      Function.new!(%{
        name: "get_weather",
        description: "Get the current weather for a city",
        parameters_schema: %{
          "type" => "object",
          "properties" => %{
            "city" => %{"type" => "string"}
          },
          "required" => ["city"]
        },
        function: fn %{"city" => city}, _ctx -> {:ok, "Sunny in #{city}, 22°C"} end
      })

    model = ChatReqLLM.new!(%{model: @live_model})

    {:ok, chain} =
      %{llm: model}
      |> LLMChain.new!()
      |> LLMChain.add_tools([weather_fn])
      |> LLMChain.add_message(Message.new_user!("What's the weather in Paris?"))
      |> LLMChain.run(mode: :while_needs_response)

    response = List.last(chain.messages)

    # Final response should be text after tool execution
    assert response.role == :assistant

    content_text =
      response.content
      |> Enum.filter(&(&1.type == :text))
      |> Enum.map_join("", & &1.content)

    assert String.contains?(content_text, "Paris") or String.contains?(content_text, "22")
    IO.inspect(response, label: "LIVE TOOL CHAIN FINAL RESPONSE")
  end

  @tag :live_call
  @tag :live_anthropic
  test "live: streaming simple text response via Anthropic" do
    test_pid = self()

    model =
      ChatReqLLM.new!(%{
        model: @live_model,
        stream: true
      })

    # Set callbacks after construction — Ecto cast strips function values
    model = %{
      model
      | callbacks: [%{on_llm_new_delta: fn deltas -> send(test_pid, {:delta, deltas}) end}]
    }

    assert {:ok, deltas} =
             ChatReqLLM.call(model, [Message.new_user!("Reply with: STREAM_OK")], [])

    assert is_list(deltas)
    assert Enum.any?(deltas, &match?(%MessageDelta{}, &1))

    # Verify callbacks fired
    assert_received {:delta, _}

    # Collect text from individual delta content parts
    text =
      deltas
      |> Enum.flat_map(fn
        %MessageDelta{content: %ContentPart{type: :text, content: c}} when is_binary(c) -> [c]
        _ -> []
      end)
      |> Enum.join("")

    assert String.contains?(text, "STREAM_OK")
  end

  @tag :live_call
  @tag :live_anthropic
  test "live: streaming LLMChain assembles complete message" do
    model = ChatReqLLM.new!(%{model: @live_model, stream: true})

    assert {:ok, chain} =
             %{llm: model}
             |> LLMChain.new!()
             |> LLMChain.add_message(Message.new_user!("Reply with: ASSEMBLED"))
             |> LLMChain.run()

    last_msg = List.last(chain.messages)
    assert last_msg.role == :assistant

    text =
      last_msg.content
      |> Enum.filter(&(&1.type == :text))
      |> Enum.map_join("", & &1.content)

    assert String.contains?(text, "ASSEMBLED")

    IO.inspect(last_msg, label: "LIVE STREAMING ASSEMBLED MESSAGE")
  end

  @tag :live_call
  @tag :live_anthropic
  test "live: streaming tool call" do
    weather_fn =
      Function.new!(%{
        name: "get_weather",
        description: "Get the current weather for a city",
        parameters_schema: %{
          "type" => "object",
          "properties" => %{"city" => %{"type" => "string"}},
          "required" => ["city"]
        },
        function: fn %{"city" => city}, _ctx -> {:ok, "Sunny in #{city}"} end
      })

    model = ChatReqLLM.new!(%{model: @live_model, stream: true})

    assert {:ok, chain} =
             %{llm: model}
             |> LLMChain.new!()
             |> LLMChain.add_tools([weather_fn])
             |> LLMChain.add_message(
               Message.new_user!("What's the weather in Paris? Use get_weather.")
             )
             |> LLMChain.run(mode: :while_needs_response)

    last_msg = List.last(chain.messages)
    assert last_msg.role == :assistant

    text =
      last_msg.content
      |> Enum.filter(&(&1.type == :text))
      |> Enum.map_join("", & &1.content)

    assert String.contains?(text, "Paris") or String.contains?(text, "Sunny")

    IO.inspect(last_msg, label: "LIVE STREAMING TOOL CHAIN FINAL MESSAGE")
  end
end
