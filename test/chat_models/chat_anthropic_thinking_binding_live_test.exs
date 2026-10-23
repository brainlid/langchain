defmodule LangChain.ChatModels.ChatAnthropicThinkingBindingLiveTest do
  @moduledoc """
  A thinking block's signature binds it to the conversation prefix that produced
  it: the top-level system prompt, the tool set, and every earlier message.
  Replaying the block after the system prompt or the tools change is rejected
  before any output. Stripping the thinking blocks lets the same edited
  conversation through.

  The replays set `prefix_mismatch_behavior: "error"`. Setting the field opts a
  request into enforcement whatever the account's age, so these tests behave the
  same on every account.

  Each test makes five requests to the model it names.
  """
  use ExUnit.Case, async: true

  alias LangChain.ChatModels.ChatAnthropic
  alias LangChain.Function
  alias LangChain.LangChainError
  alias LangChain.Message
  alias LangChain.Message.ContentPart

  @moduletag live_call: true, live_anthropic: true, timeout: 600_000

  @binding_beta "thinking-binding-controls-2026-08-01"

  @system "You are a careful arithmetic tutor. Reply with the final answer only."
  @switched_system @system <> "\n\nYou are now in the devices mode."

  # Hard enough that adaptive thinking engages on every model under test.
  @question "How many positive integers below 2000 are divisible by 3 or by 7 but not by 21, " <>
              "and have a digit sum that is even? Give the count."
  @follow_up "Now answer the same question for integers below 3000."

  for model <- ["claude-opus-5-5", "claude-sonnet-5-5"] do
    test "#{model}: a replayed thinking block is bound to the system prompt and tools" do
      model = unquote(model)
      assistant = mint(model)

      replay = [
        Message.new_user!(@question),
        assistant,
        Message.new_user!(@follow_up)
      ]

      # Unchanged prefix. Also proves the request shape is valid for this model,
      # so the rejections below come from the binding check.
      assert {:ok, %Message{}} = ask(model, @system, replay)

      # A mode switch that rewrites the system prompt.
      assert {:error, %LangChainError{message: message}} = ask(model, @switched_system, replay)
      assert message =~ "bound to a different conversation"

      # A mode switch that adds a tool.
      assert {:error, %LangChainError{message: message}} =
               ask(model, @system, replay, [device_tool()])

      assert message =~ "bound to a different conversation"

      # The same system prompt switch, with the thinking stripped from history.
      stripped = Enum.map(replay, &strip_thinking/1)
      assert {:ok, %Message{}} = ask(model, @switched_system, stripped)
    end
  end

  # Runs the first turn and returns the assistant message, which must carry a
  # signed thinking block or the replays prove nothing.
  defp mint(model) do
    {:ok, %Message{} = assistant} = ask(model, @system, [Message.new_user!(@question)])

    assert Enum.any?(assistant.content, &signed_thinking?/1),
           "#{model} answered without a signed thinking block: #{inspect(assistant.content)}"

    assistant
  end

  defp ask(model, system, messages, tools \\ []) do
    llm =
      ChatAnthropic.new!(%{
        model: model,
        stream: false,
        max_tokens: 16_000,
        receive_timeout: 300_000,
        thinking: %{
          "type" => "adaptive",
          "display" => "summarized",
          "block_binding" => %{"prefix_mismatch_behavior" => "error"}
        },
        output_config: %{"effort" => "high"},
        beta_headers: [@binding_beta]
      })

    case ChatAnthropic.call(llm, [Message.new_system!(system) | messages], tools) do
      {:ok, [%Message{} = message]} -> {:ok, message}
      other -> other
    end
  end

  defp signed_thinking?(%ContentPart{type: :thinking, options: options}),
    do: is_binary(Keyword.get(options || [], :signature))

  defp signed_thinking?(_part), do: false

  defp strip_thinking(%Message{role: :assistant, content: parts} = message) when is_list(parts) do
    %Message{message | content: Enum.reject(parts, &thinking_part?/1)}
  end

  defp strip_thinking(message), do: message

  defp thinking_part?(%ContentPart{type: :thinking}), do: true

  defp thinking_part?(%ContentPart{options: options}) when is_list(options),
    do: Keyword.get(options, :type) == "redacted_thinking"

  defp thinking_part?(_part), do: false

  defp device_tool do
    Function.new!(%{
      name: "list_devices",
      description: "List the devices in the current organization.",
      parameters_schema: %{type: "object", properties: %{}},
      function: fn _args, _context -> {:ok, "[]"} end
    })
  end
end
