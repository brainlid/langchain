defmodule LangChain.ChatModels.ChatOpenAIReasoningLiveTest do
  @moduledoc """
  Live coverage for reasoning models reached through the OpenAI-compatible
  API surface, using Cloudflare Workers AI as the provider.

  A reasoning model returns its thinking separately from its answer. On this
  API surface the thinking arrives in a `reasoning_content` field that sits
  beside `content` - on `choices[].message` for a single response, and on
  `choices[].delta` for each streamed chunk.

  The "raw wire format" tests dump the unparsed provider payload so the field
  shape is visible. The "ChatOpenAI parsing" tests assert that thinking lands
  in the parsed structs as a `:thinking` `ContentPart`.

  Run with:

      mix test test/chat_models/chat_open_ai_reasoning_live_test.exs --include live_cloudflare
  """
  use LangChain.BaseCase

  alias LangChain.ChatModels.ChatOpenAI
  alias LangChain.Chains.LLMChain
  alias LangChain.Function
  alias LangChain.FunctionParam
  alias LangChain.Message
  alias LangChain.Message.ContentPart
  alias LangChain.MessageDelta

  # A reasoning model served over Cloudflare's OpenAI-compatible endpoint.
  @model "@cf/zai-org/glm-5.3-flash"

  # A prompt that reliably provokes visible deliberation.
  @prompt "Which is larger, 9.11 or 9.9? Think it through."

  defp endpoint do
    account_id = System.fetch_env!("CLOUDFLARE_ACCOUNT_ID")
    "https://api.cloudflare.com/client/v4/accounts/#{account_id}/ai/v1/chat/completions"
  end

  defp api_key, do: System.fetch_env!("CLOUDFLARE_AI_API_TOKEN")

  # Requests are routed through a named AI Gateway. The gateway is identified
  # by a header; without it the call goes straight to Workers AI.
  defp gateway_headers do
    case System.get_env("CLOUDFLARE_AI_GATEWAY_ID") do
      nil -> []
      "" -> []
      id -> [{"cf-aig-gateway-id", id}]
    end
  end

  defp chat_model(overrides) do
    ChatOpenAI.new!(
      Map.merge(
        %{
          endpoint: endpoint(),
          api_key: api_key(),
          model: @model,
          temperature: 1,
          # `:reasoning_effort` is only placed on the request when
          # `:reasoning_mode` is enabled.
          reasoning_mode: true,
          reasoning_effort: "medium",
          req_config: %{headers: gateway_headers()}
        },
        overrides
      )
    )
  end

  # Pull every `:thinking` part out of a message or delta's content.
  defp thinking_parts(%Message{content: content}), do: thinking_parts(content)
  defp thinking_parts(%MessageDelta{merged_content: content}), do: thinking_parts(content)

  defp thinking_parts(content) when is_list(content),
    do: Enum.filter(content, &(&1.type == :thinking))

  defp thinking_parts(_other), do: []

  describe "raw wire format" do
    @tag live_call: true, live_cloudflare: true
    test "streamed chunks carry thinking in a reasoning_content delta field" do
      body = %{
        model: @model,
        stream: true,
        reasoning_effort: "medium",
        messages: [%{role: "user", content: @prompt}]
      }

      IO.puts("\n=== REQUEST BODY ===")
      IO.puts(Jason.encode!(body))

      resp =
        Req.post!(endpoint(),
          json: body,
          auth: {:bearer, api_key()},
          headers: gateway_headers(),
          receive_timeout: 120_000,
          into: []
        )

      assert resp.status == 200

      raw = IO.iodata_to_binary(resp.body)

      IO.puts("\n=== RAW SSE ===")
      IO.puts(raw)

      deltas =
        raw
        |> String.split("\n")
        |> Enum.filter(&String.starts_with?(&1, "data: "))
        |> Enum.map(&String.replace_prefix(&1, "data: ", ""))
        |> Enum.reject(&(&1 == "[DONE]"))
        |> Enum.flat_map(fn line ->
          case Jason.decode(line) do
            {:ok, %{"choices" => choices}} -> Enum.map(choices, &(&1["delta"] || %{}))
            _other -> []
          end
        end)

      IO.puts("\n=== DISTINCT DELTA KEYS ===")
      deltas |> Enum.flat_map(&Map.keys/1) |> Enum.uniq() |> Enum.sort() |> IO.inspect()

      reasoning = deltas |> Enum.map(& &1["reasoning_content"]) |> Enum.reject(&is_nil/1)

      IO.puts("\n=== ASSEMBLED reasoning_content ===")
      IO.puts(Enum.join(reasoning))

      assert reasoning != [], "expected the provider to stream reasoning_content chunks"
    end

    @tag live_call: true, live_cloudflare: true
    test "a single response carries thinking in a reasoning_content message field" do
      resp =
        Req.post!(endpoint(),
          json: %{
            model: @model,
            stream: false,
            reasoning_effort: "medium",
            messages: [%{role: "user", content: @prompt}]
          },
          auth: {:bearer, api_key()},
          headers: gateway_headers(),
          receive_timeout: 120_000
        )

      assert resp.status == 200

      IO.puts("\n=== RAW RESPONSE BODY ===")
      IO.inspect(resp.body, limit: :infinity, printable_limit: :infinity)

      message = get_in(resp.body, ["choices", Access.at(0), "message"])

      assert is_binary(message["reasoning_content"]),
             "expected reasoning_content on the message, got keys: #{inspect(Map.keys(message))}"
    end
  end

  describe "ChatOpenAI parsing" do
    @tag live_call: true, live_cloudflare: true
    test "streamed reasoning becomes a thinking ContentPart" do
      test_pid = self()

      handler = %{
        on_llm_new_delta: fn _chain, deltas -> send(test_pid, {:deltas, deltas}) end,
        on_message_processed: fn _chain, message -> send(test_pid, {:processed, message}) end
      }

      {:ok, chain} =
        %{llm: chat_model(%{stream: true})}
        |> LLMChain.new!()
        |> LLMChain.add_callback(handler)
        |> LLMChain.add_message(Message.new_user!(@prompt))
        |> LLMChain.run()

      message = chain.last_message

      IO.puts("\n=== MERGED MESSAGE ===")
      IO.inspect(message, limit: :infinity, printable_limit: :infinity)

      IO.puts("\n=== CONTENT PART TYPES ===")
      IO.inspect(Enum.map(message.content, & &1.type))

      parts = thinking_parts(message)

      assert parts != [],
             "streamed reasoning was dropped; content parts were " <>
               inspect(Enum.map(message.content, & &1.type))

      [%ContentPart{type: :thinking} = thinking | _rest] = parts

      assert String.length(thinking.content) > 0

      assert Enum.any?(message.content, &(&1.type == :text)),
             "expected the answer text alongside the thinking part"
    end

    @tag live_call: true, live_cloudflare: true
    test "non-streamed reasoning becomes a thinking ContentPart" do
      {:ok, chain} =
        %{llm: chat_model(%{stream: false})}
        |> LLMChain.new!()
        |> LLMChain.add_message(Message.new_user!(@prompt))
        |> LLMChain.run()

      message = chain.last_message

      IO.puts("\n=== MESSAGE ===")
      IO.inspect(message, limit: :infinity, printable_limit: :infinity)

      IO.puts("\n=== CONTENT PART TYPES ===")
      IO.inspect(Enum.map(message.content, & &1.type))

      parts = thinking_parts(message)

      assert parts != [],
             "reasoning was dropped; content parts were " <>
               inspect(Enum.map(message.content, & &1.type))

      [%ContentPart{type: :thinking} = thinking | _rest] = parts

      assert String.length(thinking.content) > 0

      assert Enum.any?(message.content, &(&1.type == :text)),
             "expected the answer text alongside the thinking part"
    end
  end

  describe "reasoning with tool calls" do
    @tag live_call: true, live_cloudflare: true
    test "completes a tool call round trip when the response includes thinking" do
      weather =
        Function.new!(%{
          name: "get_weather",
          description: "Get the current weather in a given US location",
          parameters: [
            FunctionParam.new!(%{
              name: "city",
              type: "string",
              description: "The city name, e.g. San Francisco",
              required: true
            })
          ],
          function: fn _args, _context -> {:ok, "75 degrees and sunny"} end
        })

      {:ok, chain} =
        %{llm: chat_model(%{stream: true})}
        |> LLMChain.new!()
        |> LLMChain.add_tools([weather])
        |> LLMChain.add_message(
          Message.new_user!("What is the weather in Moab, Utah? Use the get_weather tool.")
        )
        |> LLMChain.run(mode: :while_needs_response)

      IO.puts("\n=== MESSAGE ROLES AND CONTENT TYPES ===")

      Enum.each(chain.messages, fn msg ->
        types =
          case msg.content do
            parts when is_list(parts) -> Enum.map(parts, & &1.type)
            other -> other
          end

        IO.puts("#{msg.role}: #{inspect(types)} tool_calls=#{length(msg.tool_calls || [])}")
      end)

      # The assistant asked for the tool, the tool ran, and the model answered
      # from the result. Reaching the final answer means the thinking part in
      # the tool-calling message serialized without error on the second request.
      assert Enum.any?(chain.messages, fn msg ->
               msg.role == :assistant and msg.tool_calls not in [nil, []]
             end)

      assert Enum.any?(chain.messages, &(&1.role == :tool))

      assert %Message{role: :assistant, status: :complete} = chain.last_message
      assert ContentPart.parts_to_string(chain.last_message.content) =~ "75"
    end

    @tag live_call: true, live_cloudflare: true
    test "runs a multi-turn chain where each tool call depends on the last" do
      test_pid = self()

      lookup_employee =
        Function.new!(%{
          name: "lookup_employee_id",
          description: "Look up an employee's ID number by their first name",
          parameters: [
            FunctionParam.new!(%{
              name: "name",
              type: "string",
              description: "The employee's first name",
              required: true
            })
          ],
          function: fn args, _context ->
            send(test_pid, {:tool_called, "lookup_employee_id", args})
            {:ok, "EMP-4471"}
          end
        })

      vacation_days =
        Function.new!(%{
          name: "get_vacation_days",
          description:
            "Get remaining vacation days for an employee ID. Requires the ID, not a name.",
          parameters: [
            FunctionParam.new!(%{
              name: "employee_id",
              type: "string",
              description: "The employee ID, in the form EMP-0000",
              required: true
            })
          ],
          function: fn args, _context ->
            send(test_pid, {:tool_called, "get_vacation_days", args})
            {:ok, "12 days remaining"}
          end
        })

      {:ok, chain} =
        %{llm: chat_model(%{stream: true})}
        |> LLMChain.new!()
        |> LLMChain.add_tools([lookup_employee, vacation_days])
        |> LLMChain.add_message(
          Message.new_user!(
            "How many vacation days does Marcy have left? " <>
              "Look up her employee ID first, then use that ID to check her vacation days."
          )
        )
        |> LLMChain.run(mode: :while_needs_response)

      IO.puts("\n=== CONVERSATION ===")

      Enum.each(chain.messages, fn msg ->
        types =
          case msg.content do
            parts when is_list(parts) -> Enum.map(parts, & &1.type)
            other -> inspect(other)
          end

        calls = Enum.map(msg.tool_calls || [], & &1.name)
        results = Enum.map(msg.tool_results || [], & &1.name)

        IO.puts(
          "#{msg.role}: content=#{inspect(types)} calls=#{inspect(calls)} results=#{inspect(results)}"
        )
      end)

      # Both tools ran, and the second received the ID the first returned rather
      # than the name from the prompt.
      assert_received {:tool_called, "lookup_employee_id", %{"name" => name}}
      assert String.downcase(name) =~ "marcy"
      assert_received {:tool_called, "get_vacation_days", %{"employee_id" => "EMP-4471"}}

      # Two separate assistant turns requested tools, so the chain made at least
      # three requests. Every request after the first re-serialized the assistant
      # messages already in the conversation, thinking parts included.
      tool_calling_turns =
        Enum.filter(chain.messages, fn msg ->
          msg.role == :assistant and msg.tool_calls not in [nil, []]
        end)

      assert [_first, _second | _rest] = tool_calling_turns

      thinking_turns =
        Enum.filter(chain.messages, fn msg ->
          is_list(msg.content) and Enum.any?(msg.content, &(&1.type == :thinking))
        end)

      IO.puts("\nassistant turns carrying thinking: #{length(thinking_turns)}")

      assert %Message{role: :assistant, status: :complete} = chain.last_message
      assert ContentPart.parts_to_string(chain.last_message.content) =~ "12"
    end
  end
end
