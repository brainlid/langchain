defmodule LangChain.ChatModels.ChatOpenAICompatibleTest do
  use LangChain.BaseCase
  use Mimic

  alias LangChain.ChatModels.ChatOpenAICompatible
  alias LangChain.Chains.LLMChain
  alias LangChain.Config
  alias LangChain.Function
  alias LangChain.LangChainError
  alias LangChain.Message
  alias LangChain.Message.ContentPart
  alias LangChain.TokenUsage

  setup :verify_on_exit!

  @endpoint "https://llm.example.com/v1/chat/completions"

  defp model(attrs \\ %{}) do
    ChatOpenAICompatible.new!(Map.merge(%{endpoint: @endpoint, model: "test-model"}, attrs))
  end

  defp completion(message, extra \\ %{}) do
    Map.merge(
      %{
        "choices" => [%{"index" => 0, "finish_reason" => "stop", "message" => message}],
        "usage" => %{"prompt_tokens" => 12, "completion_tokens" => 3}
      },
      extra
    )
  end

  # Expect one non-streamed post. Sends the outgoing request to the test
  # process and answers with `status` and `body`.
  defp expect_post(status \\ 200, body) do
    test_pid = self()

    expect(Req, :post, fn req ->
      send(test_pid, {:request, req})
      {:ok, %Req.Response{status: status, headers: %{}, body: body}}
    end)
  end

  # A streamed request hands Req an `:into` collector, so a mocked post has to
  # drive that collector the way a real response would rather than handing back
  # a buffered body.
  defp expect_streamed_post(chunks) do
    expect(Req, :post, fn req, opts ->
      collector = Keyword.fetch!(opts, :into)
      start = {req, %Req.Response{status: 200, headers: %{}, body: ""}}

      {_req, response} =
        Enum.reduce(chunks, start, fn chunk, acc ->
          case collector.({:data, chunk}, acc) do
            {:cont, next} -> next
            {:halt, next} -> next
          end
        end)

      {:ok, response}
    end)
  end

  defp sse(payload), do: "data: " <> Jason.encode!(payload) <> "\n\n"

  describe "new/1" do
    test "requires endpoint and model" do
      assert {:error, changeset} = ChatOpenAICompatible.new(%{})
      assert {_, [validation: :required]} = changeset.errors[:endpoint]
      assert {_, [validation: :required]} = changeset.errors[:model]
    end

    test "has no default endpoint or model" do
      assert %ChatOpenAICompatible{endpoint: nil, model: nil} = %ChatOpenAICompatible{}
    end

    test "accepts any reasoning_effort string" do
      assert %ChatOpenAICompatible{reasoning_effort: "max"} = model(%{reasoning_effort: "max"})
    end
  end

  describe "for_api/3" do
    test "with only endpoint and model, sends exactly model, stream and messages" do
      body = ChatOpenAICompatible.for_api(model(), [Message.new_user!("Hi")], [])

      assert %{model: "test-model", stream: false, messages: [%{"role" => :user}]} = body
      assert Enum.sort(Map.keys(body)) == [:messages, :model, :stream]
    end

    test "sends a system message as system, also with reasoning_effort set" do
      messages = [Message.new_system!("Be brief."), Message.new_user!("Hi")]

      assert %{
               reasoning_effort: "low",
               messages: [%{"role" => :system}, %{"role" => :user}]
             } =
               ChatOpenAICompatible.for_api(model(%{reasoning_effort: "low"}), messages, [])
    end

    test "sends the token limit as max_tokens" do
      body = ChatOpenAICompatible.for_api(model(%{max_tokens: 200}), [], [])

      assert body.max_tokens == 200
      refute Map.has_key?(body, :max_completion_tokens)
    end

    test "sends reasoning_effort exactly when set" do
      refute Map.has_key?(ChatOpenAICompatible.for_api(model(), [], []), :reasoning_effort)

      assert %{reasoning_effort: "none"} =
               ChatOpenAICompatible.for_api(model(%{reasoning_effort: "none"}), [], [])
    end

    test "sends each optional sampling field when set" do
      body =
        ChatOpenAICompatible.for_api(
          model(%{
            temperature: 0.2,
            top_p: 0.9,
            seed: 7,
            stop: ["END"],
            frequency_penalty: 0.1,
            presence_penalty: 0.3,
            parallel_tool_calls: false
          }),
          [],
          []
        )

      assert %{
               temperature: 0.2,
               top_p: 0.9,
               seed: 7,
               stop: ["END"],
               frequency_penalty: 0.1,
               presence_penalty: 0.3,
               parallel_tool_calls: false
             } = body
    end

    test "encodes tools, tool_choice and stream_options" do
      tool =
        Function.new!(%{
          name: "lookup",
          parameters_schema: %{"type" => "object", "properties" => %{}},
          function: fn _args, _context -> {:ok, "found"} end
        })

      body =
        ChatOpenAICompatible.for_api(
          model(%{
            stream: true,
            stream_options: %{include_usage: true},
            tool_choice: %{"type" => "function", "function" => %{"name" => "lookup"}}
          }),
          [],
          [tool]
        )

      assert [%{"type" => "function", "function" => %{"name" => "lookup"}}] = body.tools
      assert body.tool_choice == %{"type" => "function", "function" => %{"name" => "lookup"}}
      assert body.stream_options == %{"include_usage" => true}

      assert %{tool_choice: "required"} =
               ChatOpenAICompatible.for_api(
                 model(%{tool_choice: %{"type" => "required"}}),
                 [],
                 []
               )
    end

    test "builds response_format from json_response and json_schema" do
      assert %{response_format: %{"type" => "json_object"}} =
               ChatOpenAICompatible.for_api(model(%{json_response: true}), [], [])

      schema = %{"name" => "answer", "schema" => %{"type" => "object"}}

      assert %{response_format: %{"type" => "json_schema", "json_schema" => ^schema}} =
               ChatOpenAICompatible.for_api(
                 model(%{json_response: true, json_schema: schema}),
                 [],
                 []
               )
    end

    test "extra_body adds keys and removes keys with a nil value" do
      body =
        ChatOpenAICompatible.for_api(
          model(%{temperature: 0.5, extra_body: %{"top_k" => 20, "temperature" => nil}}),
          [],
          []
        )

      assert body["top_k"] == 20
      refute Map.has_key?(body, :temperature)
    end
  end

  describe "request credentials and headers" do
    test "sends no authorization when api_key is nil" do
      expect_post(completion(%{"role" => "assistant", "content" => "Hi"}))

      assert {:ok, _} = ChatOpenAICompatible.call(model(), [Message.new_user!("Hi")], [])

      assert_received {:request, req}
      refute Map.has_key?(req.options, :auth)
      refute Map.has_key?(req.headers, "authorization")
    end

    test "sends the api_key as a Bearer token" do
      expect_post(completion(%{"role" => "assistant", "content" => "Hi"}))

      assert {:ok, _} =
               ChatOpenAICompatible.call(
                 model(%{api_key: "secret-key"}),
                 [Message.new_user!("Hi")],
                 []
               )

      assert_received {:request, req}
      assert req.options.auth == {:bearer, "secret-key"}
      refute Map.has_key?(req.headers, "api-key")
    end

    test "never sends the global OpenAI key, organization or project" do
      stub(Config, :resolve, fn
        :openai_key -> "global-openai-key"
        :openai_org_id -> "org-global"
        :openai_proj_id -> "proj-global"
        _other -> nil
      end)

      stub(Config, :resolve, fn
        :openai_key, _default -> "global-openai-key"
        :openai_org_id, _default -> "org-global"
        :openai_proj_id, _default -> "proj-global"
        _other, default -> default
      end)

      expect_post(completion(%{"role" => "assistant", "content" => "Hi"}))

      assert {:ok, _} = ChatOpenAICompatible.call(model(), [Message.new_user!("Hi")], [])

      assert_received {:request, req}
      refute Map.has_key?(req.options, :auth)

      for header <- ["authorization", "api-key", "openai-organization", "openai-project"] do
        refute Map.has_key?(req.headers, header), "unexpected #{header} header"
      end
    end

    test "merges req_config headers into the request" do
      expect_post(completion(%{"role" => "assistant", "content" => "Hi"}))

      model = model(%{req_config: %{headers: [{"cf-aig-gateway-id", "my-gateway"}]}})
      assert {:ok, _} = ChatOpenAICompatible.call(model, [Message.new_user!("Hi")], [])

      assert_received {:request, req}
      assert req.headers["cf-aig-gateway-id"] == ["my-gateway"]
    end
  end

  describe "call/3 responses" do
    test "returns the message with token usage" do
      expect_post(completion(%{"role" => "assistant", "content" => "Hello!"}))

      assert {:ok, [%Message{role: :assistant, status: :complete} = message]} =
               ChatOpenAICompatible.call(model(), [Message.new_user!("Hi")], [])

      assert ContentPart.parts_to_string(message.content) == "Hello!"
      assert %TokenUsage{input: 12, output: 3} = TokenUsage.get(message)
    end

    test "reports Cloudflare-shaped errors with their message" do
      expect_post(403, %{
        "errors" => [%{"message" => "Model is not available on the Workers Free plan"}],
        "success" => false
      })

      assert {:error, %LangChainError{message: "Model is not available on the Workers Free plan"}} =
               ChatOpenAICompatible.call(model(), [Message.new_user!("Hi")], [])
    end

    test "types a rate-limited error so fallbacks may retry it" do
      expect_post(429, %{"error" => %{"message" => "Slow down"}})

      assert {:error, %LangChainError{type: "rate_limit_exceeded"} = error} =
               ChatOpenAICompatible.call(model(), [Message.new_user!("Hi")], [])

      assert ChatOpenAICompatible.retry_on_fallback?(error)
    end
  end

  describe "use in LLMChain" do
    test "a non-streamed reasoning response becomes thinking and text parts" do
      expect_post(
        completion(%{
          "role" => "assistant",
          "content" => "Biscuit",
          "reasoning_content" => "The notes name the dog."
        })
      )

      {:ok, chain} =
        %{llm: model()}
        |> LLMChain.new!()
        |> LLMChain.add_message(Message.new_user!("Dog's name?"))
        |> LLMChain.run()

      assert %Message{content: [thinking, text]} = chain.last_message
      assert %ContentPart{type: :thinking, content: "The notes name the dog."} = thinking
      assert %ContentPart{type: :text, content: "Biscuit"} = text
    end

    test "streamed reasoning deltas assemble into thinking and text parts" do
      expect_streamed_post([
        sse(%{
          "choices" => [
            %{
              "index" => 0,
              "delta" => %{"role" => "assistant", "content" => nil, "reasoning_content" => "Hmm."}
            }
          ]
        }),
        sse(%{
          "choices" => [
            %{
              "index" => 0,
              "delta" => %{"content" => "Biscuit", "reasoning_content" => nil},
              "finish_reason" => "stop"
            }
          ]
        }),
        sse(%{"choices" => [], "usage" => %{"prompt_tokens" => 40, "completion_tokens" => 6}})
      ])

      {:ok, chain} =
        %{llm: model(%{stream: true, stream_options: %{include_usage: true}})}
        |> LLMChain.new!()
        |> LLMChain.add_message(Message.new_user!("Dog's name?"))
        |> LLMChain.run()

      assert %Message{status: :complete, content: [thinking, text]} = chain.last_message
      assert %ContentPart{type: :thinking, content: "Hmm."} = thinking
      assert %ContentPart{type: :text, content: "Biscuit"} = text
      assert %TokenUsage{input: 40, output: 6} = TokenUsage.get(chain.last_message)
    end
  end

  describe "serialize_config/1 and restore_from_map/1" do
    test "round-trips the configuration without credentials" do
      original =
        model(%{
          api_key: "secret-key",
          reasoning_effort: "low",
          max_tokens: 500,
          stop: ["END"],
          extra_body: %{"top_k" => 20},
          req_config: %{headers: [{"authorization", "Bearer other"}]}
        })

      serialized = ChatOpenAICompatible.serialize_config(original)

      assert serialized["module"] == "Elixir.LangChain.ChatModels.ChatOpenAICompatible"
      refute Map.has_key?(serialized, "api_key")
      refute Map.has_key?(serialized, "req_config")
      refute Map.has_key?(serialized, "callbacks")

      assert {:ok, restored} = ChatOpenAICompatible.restore_from_map(serialized)

      assert %ChatOpenAICompatible{
               endpoint: @endpoint,
               model: "test-model",
               reasoning_effort: "low",
               max_tokens: 500,
               stop: ["END"],
               extra_body: %{"top_k" => 20},
               api_key: nil,
               req_config: %{}
             } = restored
    end
  end
end
