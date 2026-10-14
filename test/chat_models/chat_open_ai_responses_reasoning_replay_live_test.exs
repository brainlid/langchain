defmodule LangChain.ChatModels.ChatOpenAIResponsesReasoningReplayLiveTest do
  @moduledoc """
  An encrypted reasoning item is checked for integrity but not bound to the
  conversation that produced it. The API rejects an item whose
  `encrypted_content` does not decrypt, and accepts an intact item after the
  system prompt or the tool set changes, or inside an unrelated conversation.

  None of this is a documented guarantee, so a failure here means the API's
  behavior moved. An accepted request shows the API took the item, not that the
  model used the reasoning in it.

  Each test makes six small requests.
  """
  use ExUnit.Case, async: true

  alias LangChain.ChatModels.ChatOpenAIResponses
  alias LangChain.Function
  alias LangChain.LangChainError
  alias LangChain.Message
  alias LangChain.Message.ContentPart

  @moduletag live_call: true, live_open_ai: true

  @model "gpt-5-mini"

  @system "You are a careful arithmetic tutor. Reply with the final answer only."
  @switched_system @system <> "\n\nYou are now in the devices mode."

  @question "A train leaves at 9:47 and arrives at 14:12 the same day. Travel costs " <>
              "$37.40 per hour, billed per minute. What does the trip cost, to the cent?"
  @follow_up "Now add a $12.50 booking fee. What is the total?"

  test "#{@model}: an encrypted reasoning item replays after the system prompt or tools change" do
    {:ok, %Message{} = assistant} = ask(@system, [Message.new_user!(@question)])

    assert Enum.any?(assistant.content, &encrypted_reasoning?/1),
           "#{@model} answered without an encrypted reasoning item: #{inspect(assistant.content)}"

    replay = [Message.new_user!(@question), assistant, Message.new_user!(@follow_up)]

    # The replay request carries the reasoning item, so an accepted edit below
    # is the API taking it, not the item being left out of the request.
    %{input: input} = ChatOpenAIResponses.for_api(llm(), replay, [])

    assert Enum.any?(input, &match?(%{"type" => "reasoning", "encrypted_content" => _}, &1))

    assert {:ok, %Message{}} = ask(@system, replay)
    assert {:ok, %Message{}} = ask(@switched_system, replay)
    assert {:ok, %Message{}} = ask(@system, replay, [device_tool()])

    # The API decrypts what it is given: a corrupted item is rejected.
    tampered = [Message.new_user!(@question), tamper(assistant), Message.new_user!(@follow_up)]
    assert {:error, %LangChainError{message: message}} = ask(@system, tampered)
    assert message =~ "could not be verified"

    # The same item moved into an unrelated conversation.
    transplanted = [
      Message.new_user!("What is the capital of Australia?"),
      assistant,
      Message.new_user!("And of New Zealand?")
    ]

    assert {:ok, %Message{}} = ask(@switched_system, transplanted)
  end

  defp tamper(%Message{content: parts} = message) do
    parts =
      Enum.map(parts, fn part ->
        if encrypted_reasoning?(part) do
          encrypted = Keyword.fetch!(part.options, :encrypted_content)
          middle = div(byte_size(encrypted), 2)
          <<head::binary-size(^middle), char, tail::binary>> = encrypted
          flipped = if char == ?A, do: ?B, else: ?A

          %{
            part
            | options: Keyword.put(part.options, :encrypted_content, head <> <<flipped>> <> tail)
          }
        else
          part
        end
      end)

    %Message{message | content: parts}
  end

  defp ask(system, messages, tools \\ []) do
    case ChatOpenAIResponses.call(llm(), [Message.new_system!(system) | messages], tools) do
      {:ok, [%Message{} = message]} -> {:ok, message}
      other -> other
    end
  end

  defp llm do
    ChatOpenAIResponses.new!(%{
      model: @model,
      stream: false,
      store: false,
      include: ["reasoning.encrypted_content"],
      reasoning: %{effort: :medium}
    })
  end

  defp encrypted_reasoning?(%ContentPart{options: options}) when is_list(options) do
    Keyword.get(options, :type) == "reasoning" and
      is_binary(Keyword.get(options, :encrypted_content))
  end

  defp encrypted_reasoning?(_part), do: false

  defp device_tool do
    Function.new!(%{
      name: "list_devices",
      description: "List the devices in the current organization.",
      parameters_schema: %{type: "object", properties: %{}},
      function: fn _args, _context -> {:ok, "[]"} end
    })
  end
end
