defmodule LangChain.ChatModels.ChatOpenAIResponsesWebSocketLiveTest do
  use LangChain.BaseCase

  alias LangChain.ChatModels.ChatOpenAIResponses
  alias LangChain.Message
  alias LangChain.Message.ContentPart
  alias LangChain.Chains.LLMChain
  alias LangChain.Tools.Calculator
  alias LangChain.WebSocket

  @moduletag live_call: true, live_open_ai: true
  @model "gpt-4o-mini"

  setup do
    llm =
      ChatOpenAIResponses.new!(%{model: @model, stream: false})
      |> ChatOpenAIResponses.connect_websocket!()

    on_exit(fn ->
      ChatOpenAIResponses.disconnect_websocket!(llm)
    end)

    [llm: llm]
  end

  test "simple non-streaming chat via WebSocket", %{llm: llm} do
    {:ok, message} = ChatOpenAIResponses.call(llm, [Message.new_user!("Say Hi")], [])

    assert message.role == :assistant
    assert is_list(message.content)
    assert ContentPart.parts_to_string(message.content) =~ ~r/hi/i
  end

  test "streaming chat via WebSocket", %{llm: llm} do
    llm = %{llm | stream: true}

    {:ok, deltas} =
      ChatOpenAIResponses.call(
        llm,
        [Message.new_user!("Return the exact text: Hello World")],
        []
      )

    assert is_list(deltas)
    assert length(deltas) > 0

    text_deltas =
      deltas
      |> Enum.filter(fn
        %MessageDelta{content: content} when is_binary(content) and content != "" -> true
        _ -> false
      end)

    assert length(text_deltas) > 0

    combined_text =
      text_deltas
      |> Enum.map(fn %MessageDelta{content: content} -> content end)
      |> Enum.join("")

    assert combined_text =~ ~r/Hello.*World/i
  end

  test "tool calling via WebSocket", %{llm: llm} do
    {:ok, message} =
      ChatOpenAIResponses.call(
        llm,
        [Message.new_user!("What is 100 + 300 - 200? Use the calculator tool.")],
        [Calculator.new!()]
      )

    assert message.role == :assistant
    [tool_call | _] = message.tool_calls
    assert tool_call.name == "calculator"
  end

  test "complete chain with tool execution via WebSocket", %{llm: llm} do
    {:ok, chain} =
      %{llm: llm}
      |> LLMChain.new!()
      |> LLMChain.add_messages([
        Message.new_user!("Answer using the 'calculator' tool: What is 10 * 5 + 5?")
      ])
      |> LLMChain.add_tools(Calculator.new!())
      |> LLMChain.run(mode: :while_needs_response)

    assert %Message{role: :assistant, status: :complete} = chain.last_message
    content_str = ContentPart.parts_to_string(chain.last_message.content)
    assert content_str =~ "55"
  end

  test "WebSocket connection is reused across multiple calls", %{llm: llm} do
    # First call
    {:ok, msg1} = ChatOpenAIResponses.call(llm, [Message.new_user!("Say one")], [])
    assert msg1.role == :assistant

    # Second call on same WebSocket
    {:ok, msg2} = ChatOpenAIResponses.call(llm, [Message.new_user!("Say two")], [])
    assert msg2.role == :assistant

    # WebSocket should still be alive
    assert WebSocket.connected?(llm.websocket)
  end

  test "streaming with tool calling chain via WebSocket", %{llm: llm} do
    llm = %{llm | stream: true}

    {:ok, chain} =
      %{llm: llm}
      |> LLMChain.new!()
      |> LLMChain.add_messages([
        Message.new_user!("Answer using the 'calculator' tool: What is 10 * 5 + 5?")
      ])
      |> LLMChain.add_tools(Calculator.new!())
      |> LLMChain.run(mode: :while_needs_response)

    assert %Message{role: :assistant, status: :complete} = chain.last_message
    content_str = ContentPart.parts_to_string(chain.last_message.content)
    assert content_str =~ "55"
  end
end
