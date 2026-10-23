defmodule LangChain.ChatModels.ChatAnthropicAdaptiveThinkingLiveTest do
  @moduledoc """
  One adaptive thinking config is accepted by both Sonnet 4.6 and Sonnet 5.5
  through a streamed tool loop.

  A budget-based `thinking` config on Sonnet 5.5 is rejected with an
  `invalid_request_error` that names the adaptive replacement.
  """
  use ExUnit.Case, async: true

  alias LangChain.ChatModels.ChatAnthropic
  alias LangChain.Chains.LLMChain
  alias LangChain.Function
  alias LangChain.LangChainError
  alias LangChain.Message

  @moduletag live_call: true, live_anthropic: true, timeout: 600_000

  for model <- ["claude-sonnet-4-6", "claude-sonnet-5-5"] do
    test "#{model}: a tool loop with adaptive thinking completes" do
      llm =
        ChatAnthropic.new!(%{
          model: unquote(model),
          stream: true,
          max_tokens: 16_000,
          thinking: %{type: "adaptive", display: "summarized"},
          output_config: %{"effort" => "medium"}
        })

      assert {:ok, chain} =
               %{llm: llm}
               |> LLMChain.new!()
               |> LLMChain.add_tools([weather_tool()])
               |> LLMChain.add_message(
                 Message.new_user!("What is the weather in Paris right now? Use the tool.")
               )
               |> LLMChain.run(mode: :while_needs_response)

      assert Enum.any?(chain.messages, &match?(%Message{role: :tool}, &1))
      assert %Message{role: :assistant, status: :complete} = chain.last_message
    end
  end

  test "claude-sonnet-5-5 rejects a budget thinking config without retrying" do
    model =
      ChatAnthropic.new!(%{
        model: "claude-sonnet-5-5",
        max_tokens: 16_000,
        thinking: %{type: "enabled", budget_tokens: 3000}
      })

    assert {:error, %LangChainError{type: "invalid_request_error", message: message}} =
             ChatAnthropic.call(model, [Message.new_user!("Say hello.")], [])

    assert message =~ "thinking.type.adaptive"
  end

  defp weather_tool do
    Function.new!(%{
      name: "get_weather",
      description: "Get the current weather for a city.",
      parameters_schema: %{
        type: "object",
        properties: %{city: %{type: "string", description: "City name"}},
        required: ["city"]
      },
      function: fn %{"city" => city}, _context -> {:ok, "Sunny and 21C in #{city}."} end
    })
  end
end
