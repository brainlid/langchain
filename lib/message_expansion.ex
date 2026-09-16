defmodule LangChain.MessageExpansion do
  @moduledoc """
  A tool result's request to expand into conversation messages before the model
  is next called.

  A tool result normally reaches the model as one thing: `:tool` content. That
  is the right container for an answer and the wrong one for a body of material
  the model is meant to treat as established. An expansion lets a tool say "put
  these messages into the conversation, and trim what my own result keeps once
  you have."

  One result becomes several messages, and shrinks as they land. That is the
  expansion: the conversation grows by what the result gives up.

  An expansion is applied by
  `LangChain.Chains.LLMChain.Mode.Steps.expand_tool_results/2`, which runs
  immediately before the next LLM call. The messages are in the conversation by
  the time the model is asked to continue, in the same run.

  ## Fields

  - `:messages` - the messages to insert, in order, exactly as the model will
    see them. Any number of them, at `:user` or `:assistant` roles.
  - `:result_content` - what the tool's own result keeps once the messages have
    been inserted. `nil` keeps the result's existing content.

  ## The two representations

  An expanding tool describes its output twice, because the expansion may not
  happen. Only a mode that composes the step applies one; under any other mode
  the field is inert.

  | | What the model reads |
  |---|---|
  | Expansion not applied | the tool result's content, as the tool wrote it |
  | Expansion applied | `:messages`, and the result trimmed to `:result_content` |

  `expand/3` takes both in one call so the two cannot drift, and so that the
  degraded path is something a tool author passes rather than something they
  remember:

      LangChain.MessageExpansion.expand(
        raw_text,
        [
          Message.new_assistant!(records),
          Message.new_user!("Use the records above to answer my question.")
        ],
        result_content: "Loaded 14 records."
      )

  The message list is entirely the tool author's: one message, or six, in
  whatever order and at whichever of the two roles the prompt needs.

  ## Ending on an assistant message

  Anthropic reads a trailing assistant message as a prefill to continue rather
  than a turn to answer, so an expansion whose last message is an `:assistant`
  one becomes the opening of the model's own next sentence rather than something
  it responds to. Ending the list with a short `:user` message that re-anchors
  the request avoids that, and is the same shape an application hand-builds when
  seeding a conversation before an agent starts.

  This is a prompt-shaping recommendation, not a rule. A tool that wants a
  prefill can have one; nothing is appended on the author's behalf.

  ## Roles

  Only `:user` and `:assistant` can be expanded into.

  `:system` is refused because a conversation carries at most one system
  message: `LangChain.Utils.split_system_message/2` raises on a second one, far
  from the tool that caused it. `:tool` is refused because a tool message is
  only meaningful against a tool call that asked for it.
  """
  alias __MODULE__
  alias LangChain.LangChainError
  alias LangChain.Message
  alias LangChain.Message.ContentPart
  alias LangChain.Message.ToolResult

  @type content :: String.t() | ContentPart.t() | [ContentPart.t()]

  @type t :: %MessageExpansion{
          messages: [Message.t()],
          result_content: nil | content()
        }

  defstruct messages: [], result_content: nil

  @expandable_roles [:user, :assistant]

  @default_result_content "The requested content has been added to the conversation."

  @doc """
  Build a tool result that expands into `messages`, and return the `{:ok, result}`
  tuple a tool body returns.

  `fallback_content` is the tool's whole answer in the shape a tool result can
  hold. It is what the result says until the expansion is applied, and what the
  model reads under a mode that never applies it. It is not a summary: a mode
  without the step has to leave the model with the content, in worse wording,
  rather than with a description of content it does not have.

      def load_reference(%{"name" => name}, _context) do
        {records, summary} = load(name)

        LangChain.MessageExpansion.expand(
          summary <> "\\n\\n" <> records,
          [
            Message.new_assistant!(records),
            Message.new_user!("Use the records above to answer my question about \#{name}.")
          ],
          result_content: summary
        )
      end

  ## Options

  - `:result_content` - what the result keeps once the messages are inserted.
    Defaults to a short note saying the content is in the conversation. Pass
    `nil` to leave `fallback_content` in place, which keeps the payload in two
    places and is rarely what an expansion is for.

  ## The call identity is filled in for you

  The result comes back with `tool_call_id`, `name` and `display_text` unset.
  `LangChain.Chains.LLMChain.execute_tool_call/3` assigns `tool_call_id` from
  the call it dispatched, and falls back to the `LangChain.Function`'s own
  `name` and `display_text`, which is what it does for any `ToolResult` a tool
  hands back.

  A tool that wants different UI text for a particular call sets `display_text`
  on the returned result; the fallback only applies while it is nil. A tool that
  needs the id for its own bookkeeping reads `context.tool_call_id`, which
  `execute_tool_call/3` puts there before dispatch. Setting `tool_call_id` on
  the result has no effect, because the call is the authority on it.

  Code building a result outside a tool body has to assign all three itself.
  """
  @spec expand(content(), [Message.t()], keyword()) :: {:ok, ToolResult.t()} | no_return()
  def expand(fallback_content, messages, opts \\ []) do
    validate_messages!(messages)

    {:ok,
     %ToolResult{
       content: normalize_content(fallback_content),
       message_expansion: %MessageExpansion{
         messages: messages,
         result_content: Keyword.get(opts, :result_content, @default_result_content)
       }
     }}
  end

  @doc """
  Whether a tool result carries an expansion that is ready to be applied.

  An interrupted result is excluded: its tool has not finished, and the turn is
  about to stop rather than continue.
  """
  @spec expandable?(ToolResult.t()) :: boolean()
  def expandable?(%ToolResult{message_expansion: %MessageExpansion{}, is_interrupt: false}),
    do: true

  def expandable?(%ToolResult{}), do: false

  defp validate_messages!([]) do
    raise LangChainError.exception(
            type: "empty_expansion",
            message:
              "An expansion needs at least one message. To trim a tool result " <>
                "without inserting anything, set `result_content` on a " <>
                "%LangChain.MessageExpansion{} directly."
          )
  end

  defp validate_messages!(messages) when is_list(messages) do
    Enum.each(messages, &validate_message!/1)
  end

  defp validate_messages!(other) do
    raise LangChainError.exception(
            type: "invalid_expansion",
            message: "Expected a list of %LangChain.Message{}. Received: #{inspect(other)}"
          )
  end

  defp validate_message!(%Message{role: role}) when role in @expandable_roles, do: :ok

  defp validate_message!(%Message{role: role}) do
    raise LangChainError.exception(
            type: "invalid_expansion_role",
            message:
              "Cannot expand into a message with role #{inspect(role)}. " <>
                "Supported roles: #{inspect(@expandable_roles)}."
          )
  end

  defp validate_message!(other) do
    raise LangChainError.exception(
            type: "invalid_expansion",
            message: "Expected a %LangChain.Message{}. Received: #{inspect(other)}"
          )
  end

  # Mirrors what `ToolResult.new/1` does to a tool result's content.
  defp normalize_content(content) when is_binary(content), do: [ContentPart.text!(content)]
  defp normalize_content(%ContentPart{} = part), do: [part]
  defp normalize_content(content) when is_list(content), do: content
end
