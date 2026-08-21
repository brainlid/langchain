defmodule LangChain.ChatModels.StreamedTokenUsageTest do
  @moduledoc """
  A streamed turn must report its token usage, whatever carrier the provider
  chose for it, and must report it once.

  `ChatModel.token_usage_from_result/1` is what the `[:langchain, :llm, :call]`
  span reads on stop, so a provider whose streaming decode drops usage silently
  reports `token_usage: nil` to telemetry and to OpenTelemetry. A provider whose
  readings are added instead of combined reports a multiple of the truth. Both
  failures are invisible from inside one provider's own tests, so the invariant
  is asserted here across all of them at once.
  """
  use LangChain.BaseCase
  use Mimic

  alias LangChain.ChatModels.{
    ChatAnthropic,
    ChatAwsMantle,
    ChatDeepSeek,
    ChatGoogleAI,
    ChatMistralAI,
    ChatModel,
    ChatOpenAI,
    ChatOpenAIResponses,
    ChatOrq,
    ChatPerplexity,
    ChatVertexAI
  }

  alias LangChain.Message
  alias LangChain.MessageDelta
  alias LangChain.TokenUsage

  setup :verify_on_exit!

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

  # An OpenAI-shaped stream: content chunks carry no usage, and a final chunk
  # with empty `choices` carries usage alone.
  defp openai_shaped_chunks do
    [
      sse(%{
        "choices" => [
          %{"index" => 0, "delta" => %{"role" => "assistant", "content" => "Hel"}}
        ]
      }),
      sse(%{
        "choices" => [
          %{"index" => 0, "delta" => %{"content" => "lo!"}, "finish_reason" => "stop"}
        ]
      }),
      sse(%{
        "choices" => [],
        "usage" => %{"prompt_tokens" => 10, "completion_tokens" => 5, "total_tokens" => 15}
      })
    ]
  end

  # Mistral and Perplexity report usage on the chunk that closes the stream,
  # riding alongside the final content delta rather than on a chunk of its own.
  defp trailing_usage_chunks do
    [
      sse(%{
        "choices" => [
          %{"index" => 0, "delta" => %{"role" => "assistant", "content" => "Hel"}}
        ]
      }),
      sse(%{
        "choices" => [
          %{"index" => 0, "delta" => %{"content" => "lo!"}, "finish_reason" => "stop"}
        ],
        "usage" => %{"prompt_tokens" => 10, "completion_tokens" => 5, "total_tokens" => 15}
      })
    ]
  end

  # Gemini repeats the totals for the message so far on every chunk.
  defp gemini_chunks do
    for {text, output} <- [{"Hel", 2}, {"lo!", 5}] do
      sse(%{
        "candidates" => [
          %{"content" => %{"role" => "model", "parts" => [%{"text" => text}]}}
        ],
        "usageMetadata" => %{
          "promptTokenCount" => 10,
          "candidatesTokenCount" => output,
          "totalTokenCount" => 10 + output
        }
      })
    end
  end

  # Anthropic opens the message with the input classes and closes it with the
  # totals for the whole message.
  defp anthropic_chunks do
    [
      sse(%{
        "type" => "message_start",
        "message" => %{
          "id" => "msg_1",
          "type" => "message",
          "role" => "assistant",
          "content" => [],
          "usage" => %{
            "input_tokens" => 10,
            "cache_creation_input_tokens" => 0,
            "cache_read_input_tokens" => 0,
            "output_tokens" => 1
          }
        }
      }),
      sse(%{
        "type" => "content_block_start",
        "index" => 0,
        "content_block" => %{"type" => "text", "text" => ""}
      }),
      sse(%{
        "type" => "content_block_delta",
        "index" => 0,
        "delta" => %{"type" => "text_delta", "text" => "Hello!"}
      }),
      sse(%{
        "type" => "message_delta",
        "delta" => %{"stop_reason" => "end_turn"},
        "usage" => %{"output_tokens" => 5}
      })
    ]
  end

  defp responses_chunks do
    [
      sse(%{
        "type" => "response.output_text.delta",
        "delta" => "Hello!",
        "output_index" => 0,
        "item_id" => "msg_1"
      }),
      sse(%{
        "type" => "response.completed",
        "response" => %{
          "id" => "resp_1",
          "usage" => %{"input_tokens" => 10, "output_tokens" => 5, "total_tokens" => 15}
        }
      })
    ]
  end

  defp chunks_for(:openai_shaped), do: openai_shaped_chunks()
  defp chunks_for(:trailing_usage), do: trailing_usage_chunks()
  defp chunks_for(:gemini), do: gemini_chunks()
  defp chunks_for(:anthropic), do: anthropic_chunks()
  defp chunks_for(:responses), do: responses_chunks()

  @providers [
    {ChatOpenAI, %{model: "gpt-4o", api_key: "k"}, :openai_shaped},
    {ChatOpenAIResponses, %{model: "gpt-4o", api_key: "k"}, :responses},
    {ChatAnthropic, %{model: "claude-x", api_key: "k"}, :anthropic},
    {ChatMistralAI, %{model: "mistral-small", api_key: "k"}, :trailing_usage},
    {ChatPerplexity, %{model: "sonar-pro", api_key: "k"}, :trailing_usage},
    {ChatDeepSeek, %{model: "deepseek-chat", api_key: "k"}, :openai_shaped},
    {ChatOrq, %{model: "openai/gpt-4o", key: "k"}, :openai_shaped},
    {ChatAwsMantle, %{model: "anthropic.claude", endpoint: "https://mantle.test", api_key: "k"},
     :openai_shaped},
    {ChatGoogleAI, %{model: "gemini-2.0-flash", api_key: "k"}, :gemini},
    {ChatVertexAI, %{model: "gemini-2.0-flash", endpoint: "https://vertex.test", api_key: "k"},
     :gemini}
  ]

  for {module, attrs, shape} <- @providers do
    @module module
    @attrs attrs
    @shape shape

    test "#{inspect(module)} reports a streamed turn's usage exactly once" do
      model = @module.new!(Map.put(@attrs, :stream, true))
      expect_streamed_post(chunks_for(@shape))

      assert {:ok, items} = @module.call(model, [Message.new_user!("Hi")], []),
             "#{inspect(@module)} did not answer a streamed call"

      assert %{token_usage: %TokenUsage{input: 10, output: 5}} =
               ChatModel.token_usage_from_result({:ok, items}),
             "#{inspect(@module)} did not report input 10 / output 5 to the LLM call span"
    end
  end

  test "a streamed result nests one list per received chunk" do
    # `ChatModel.call_response/0` declares the nesting, and both
    # `LangChain.Chains.LLMChain` and `token_usage_from_result/1` flatten before
    # reading. A provider that flattened its own body would break neither, but
    # the declared type would stop describing the common case.
    model = ChatOpenAI.new!(%{model: "gpt-4o", api_key: "k", stream: true})
    expect_streamed_post(openai_shaped_chunks())

    assert {:ok, items} = ChatOpenAI.call(model, [Message.new_user!("Hi")], [])

    assert [[%MessageDelta{} | _] | _] = items
    assert Enum.any?(List.flatten(items), &match?(%TokenUsage{}, &1))
  end
end
