defmodule LangChain.ChatModels.ChatOpenAICompatibleLiveTest do
  @moduledoc """
  Live conformance checks for `ChatOpenAICompatible` against real
  OpenAI-compatible services.

  Every provider runs the same checks of the contract this module promises:

  - the system prompt reaches the model, streaming and non-streaming
  - streamed responses report token usage
  - a tool call round-trips through `LLMChain`
  - `max_tokens` limits the response
  - for reasoning models, thinking arrives as a `:thinking` `ContentPart`,
    including across a multi-step tool conversation

  A provider whose environment variables are absent is skipped.

  Requests are billable. Run one provider, or all of them:

      mix test test/chat_models/chat_open_ai_compatible_live_test.exs --include live_cloudflare
      mix test test/chat_models/chat_open_ai_compatible_live_test.exs --include live_openai_compatible

  Every request carries a unique nonce in its system prompt. Gateways such as
  Cloudflare's AI Gateway answer a byte-identical request from a response
  cache, which would test the cache instead of the model.
  """
  use LangChain.BaseCase

  alias LangChain.ChatModels.ChatOpenAICompatible
  alias LangChain.Chains.LLMChain
  alias LangChain.Function
  alias LangChain.FunctionParam
  alias LangChain.LangChainError
  alias LangChain.Message
  alias LangChain.Message.ContentPart
  alias LangChain.TokenUsage

  @providers [
    %{
      id: :cloudflare,
      name: "Cloudflare Workers AI",
      tag: :live_cloudflare,
      env: ["CLOUDFLARE_ACCOUNT_ID", "CLOUDFLARE_AI_API_TOKEN", "CLOUDFLARE_AI_GATEWAY_ID"],
      reasoning?: true
    },
    %{
      id: :ollama,
      name: "Ollama",
      tag: :live_ollama,
      env: ["OLLAMA_OPENAI_ENDPOINT", "OLLAMA_MODEL"],
      reasoning?: false
    },
    %{
      id: :openrouter,
      name: "OpenRouter",
      tag: :live_openrouter,
      env: ["OPENROUTER_API_KEY", "OPENROUTER_MODEL"],
      reasoning?: false
    },
    %{
      id: :groq,
      name: "Groq",
      tag: :live_groq,
      env: ["GROQ_API_KEY", "GROQ_MODEL"],
      reasoning?: false
    }
  ]

  # The fact the model can only know from the system prompt.
  @fact_question "In one word: what is the keeper's dog called?"
  @fact_answer "Biscuit"

  defp build_model(:cloudflare, overrides) do
    account_id = System.fetch_env!("CLOUDFLARE_ACCOUNT_ID")

    new_model(
      %{
        endpoint:
          "https://api.cloudflare.com/client/v4/accounts/#{account_id}/ai/v1/chat/completions",
        api_key: System.fetch_env!("CLOUDFLARE_AI_API_TOKEN"),
        model: "@cf/zai-org/glm-5.3-flash",
        reasoning_effort: "low",
        req_config: %{
          headers: [{"cf-aig-gateway-id", System.fetch_env!("CLOUDFLARE_AI_GATEWAY_ID")}]
        }
      },
      overrides
    )
  end

  defp build_model(:ollama, overrides) do
    new_model(
      %{
        endpoint: System.fetch_env!("OLLAMA_OPENAI_ENDPOINT"),
        model: System.fetch_env!("OLLAMA_MODEL")
      },
      overrides
    )
  end

  defp build_model(:openrouter, overrides) do
    new_model(
      %{
        endpoint: "https://openrouter.ai/api/v1/chat/completions",
        api_key: System.fetch_env!("OPENROUTER_API_KEY"),
        model: System.fetch_env!("OPENROUTER_MODEL")
      },
      overrides
    )
  end

  defp build_model(:groq, overrides) do
    new_model(
      %{
        endpoint: "https://api.groq.com/openai/v1/chat/completions",
        api_key: System.fetch_env!("GROQ_API_KEY"),
        model: System.fetch_env!("GROQ_MODEL")
      },
      overrides
    )
  end

  defp new_model(base, overrides) do
    base
    |> Map.merge(%{temperature: 1.0, receive_timeout: 120_000})
    |> Map.merge(overrides)
    |> ChatOpenAICompatible.new!()
  end

  defp nonce, do: "#{System.system_time(:millisecond)}-#{System.unique_integer([:positive])}"

  # A system prompt of several hundred tokens that holds the one fact the user
  # question needs.
  defp fact_system_prompt do
    notes =
      Enum.map_join(0..39, " ", fn i ->
        "Note #{i}: the lighthouse lamp was serviced in year #{1900 + i}."
      end)

    "You are a terse assistant (session #{nonce()}). Reference notes: #{notes} " <>
      "Important: the keeper's dog is named #{@fact_answer}."
  end

  # A lower bound on the system prompt's token count. Tokenizers average about
  # four characters per token on English text, so a floor of one token per
  # eight characters leaves wide margin and still sits far above what the user
  # message alone would count.
  defp token_floor(text), do: div(String.length(text), 8)

  defp nonce_system_message do
    Message.new_system!("You are a helpful assistant (session #{nonce()}).")
  end

  defp thinking_parts(%Message{content: content}) when is_list(content),
    do: Enum.filter(content, &(&1.type == :thinking))

  defp thinking_parts(_message), do: []

  defp text(%Message{content: content}), do: ContentPart.parts_to_string(content) || ""

  defp run_fact_question(model) do
    system = fact_system_prompt()

    {:ok, chain} =
      %{llm: model}
      |> LLMChain.new!()
      |> LLMChain.add_messages([Message.new_system!(system), Message.new_user!(@fact_question)])
      |> LLMChain.run()

    {system, chain.last_message}
  end

  for provider <- @providers do
    @provider provider
    @missing_env Enum.reject(provider.env, &System.get_env/1)

    describe provider.name do
      @describetag [{:live_call, true}, {:live_openai_compatible, true}, {provider.tag, true}]

      if @missing_env != [] do
        @describetag skip: "set #{Enum.join(@missing_env, ", ")} to run"
      end

      test "the system prompt reaches the model" do
        model = build_model(@provider.id, %{})
        {system, message} = run_fact_question(model)

        assert text(message) =~ ~r/#{@fact_answer}/i
        assert %TokenUsage{input: input} = TokenUsage.get(message)

        assert input > token_floor(system),
               "prompt tokens #{input} do not cover the system prompt"
      end

      test "the system prompt reaches the model when streaming, with usage reported" do
        model =
          build_model(@provider.id, %{stream: true, stream_options: %{include_usage: true}})

        {system, message} = run_fact_question(model)

        assert text(message) =~ ~r/#{@fact_answer}/i
        assert %TokenUsage{input: input} = TokenUsage.get(message)

        assert input > token_floor(system),
               "prompt tokens #{input} do not cover the system prompt"
      end

      test "a tool call round-trips" do
        weather =
          Function.new!(%{
            name: "get_weather",
            description: "Get the current weather in a given US city",
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
          %{llm: build_model(@provider.id, %{stream: true})}
          |> LLMChain.new!()
          |> LLMChain.add_tools([weather])
          |> LLMChain.add_messages([
            nonce_system_message(),
            Message.new_user!("What is the weather in Moab, Utah? Use the get_weather tool.")
          ])
          |> LLMChain.run(mode: :while_needs_response)

        assert Enum.any?(chain.messages, fn msg ->
                 msg.role == :assistant and msg.tool_calls not in [nil, []]
               end)

        assert Enum.any?(chain.messages, &(&1.role == :tool))
        assert %Message{role: :assistant, status: :complete} = chain.last_message
        assert text(chain.last_message) =~ "75"
      end

      test "max_tokens limits the response" do
        # A response cut off by the limit is reported as an error, with the
        # chain holding the truncated message.
        assert {:error, chain, %LangChainError{type: "response_truncated"}} =
                 %{llm: build_model(@provider.id, %{max_tokens: 16})}
                 |> LLMChain.new!()
                 |> LLMChain.add_messages([
                   nonce_system_message(),
                   Message.new_user!("Write a long story about a lighthouse keeper.")
                 ])
                 |> LLMChain.run()

        assert %Message{status: :length} = chain.last_message
        assert %TokenUsage{output: output} = TokenUsage.get(chain.last_message)
        # Some services count a token or two of framing around the limit.
        assert output <= 16 + 4
      end

      if @provider.reasoning? do
        test "non-streamed thinking arrives as a thinking part ahead of the answer" do
          {:ok, chain} =
            %{llm: build_model(@provider.id, %{})}
            |> LLMChain.new!()
            |> LLMChain.add_messages([
              nonce_system_message(),
              Message.new_user!("Which is larger, 9.11 or 9.9? Think it through.")
            ])
            |> LLMChain.run()

          assert [%ContentPart{type: :thinking, content: thinking} | _] =
                   chain.last_message.content

          assert String.length(thinking) > 0
          assert Enum.any?(chain.last_message.content, &(&1.type == :text))
        end

        test "streamed thinking arrives as a thinking part ahead of the answer" do
          {:ok, chain} =
            %{llm: build_model(@provider.id, %{stream: true})}
            |> LLMChain.new!()
            |> LLMChain.add_messages([
              nonce_system_message(),
              Message.new_user!("Which is larger, 9.11 or 9.9? Think it through.")
            ])
            |> LLMChain.run()

          assert [%ContentPart{type: :thinking, content: thinking} | _] =
                   chain.last_message.content

          assert String.length(thinking) > 0
          assert Enum.any?(chain.last_message.content, &(&1.type == :text))
        end

        test "a multi-step tool conversation carries thinking without breaking requests" do
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

          # At low effort a reasoning model often skips thinking on tool turns,
          # leaving nothing for this test to carry.
          {:ok, chain} =
            %{llm: build_model(@provider.id, %{stream: true, reasoning_effort: "medium"})}
            |> LLMChain.new!()
            |> LLMChain.add_tools([lookup_employee, vacation_days])
            |> LLMChain.add_messages([
              nonce_system_message(),
              Message.new_user!(
                "How many vacation days does Marcy have left? " <>
                  "Look up her employee ID first, then use that ID to check her vacation days."
              )
            ])
            |> LLMChain.run(mode: :while_needs_response)

          # The second tool received the ID the first returned, not the name.
          assert_received {:tool_called, "lookup_employee_id", %{"name" => name}}
          assert String.downcase(name) =~ "marcy"
          assert_received {:tool_called, "get_vacation_days", %{"employee_id" => "EMP-4471"}}

          # Every request after the first re-sent the assistant messages already
          # in the conversation. Reaching the final answer means messages that
          # held thinking parts were encoded without error.
          assert Enum.any?(chain.messages, &(thinking_parts(&1) != []))
          assert %Message{role: :assistant, status: :complete} = chain.last_message
          assert text(chain.last_message) =~ "12"
        end
      end
    end
  end
end
