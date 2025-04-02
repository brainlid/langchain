defmodule LangChain.MessageDelta do
  @moduledoc """
  Models a "delta" message from a chat LLM. A delta is a small chunk, or piece
  of a much larger complete message. A series of deltas are used to construct
  the complete message.

  Delta messages must be applied in order for them to be valid. Delta messages
  can be combined and transformed into a `LangChain.Message` once the final
  piece is received.

  ## Roles

  * `:unknown` - The role data is missing for the delta.
  * `:assistant` - Responses coming back from the LLM.

  ## Tool Usage

  Tools can be used or called by the assistant (LLM). A tool call is also split
  across many message deltas and must be fully assembled before it can be
  executed.

  ## Metadata

  The `metadata` field is a map that can contain any additional information
  about the message delta. It is used to store token usage, model, and other LLM-specific
  information.

  ## Content

  The `content` field is has two modes of operating.

  - Mode 1: A single, small MessageDelta will receive a `LangChain.Message.ContentPart` struct for the content's value.
  - Mode 2: When merging a set of small MessageDeltas, the content can become a list of ContentParts. This supports receiving multi-modal responses like images or audio. But it can also receive `thinking` content that is separate from the final `text` content.

  """
  use Ecto.Schema
  import Ecto.Changeset
  require Logger
  alias __MODULE__
  alias LangChain.LangChainError
  alias LangChain.Message
  alias LangChain.Message.ContentPart
  alias LangChain.Message.ToolCall
  alias LangChain.Utils
  alias LangChain.TokenUsage

  @primary_key false
  embedded_schema do
    field :content, :any, virtual: true
    # Marks if the delta completes the message.
    field :status, Ecto.Enum, values: [:incomplete, :complete, :length], default: :incomplete
    # When requesting multiple choices for a response, the `index` represents
    # which choice it is. It is a 0 based list.
    field :index, :integer

    field :role, Ecto.Enum, values: [:unknown, :assistant], default: :unknown

    field :tool_calls, :any, virtual: true

    # Additional metadata about the message.
    field :metadata, :map
  end

  @type t :: %MessageDelta{}

  @create_fields [:role, :content, :index, :status, :tool_calls, :metadata]
  @required_fields []

  @doc """
  Create a new `MessageDelta` that represents a message chunk.
  """
  @spec new(attrs :: map()) :: {:ok, t()} | {:error, Ecto.Changeset.t()}
  def new(attrs \\ %{}) do
    %MessageDelta{}
    |> cast(attrs, @create_fields)

    # TODO: Change this to cast content when it's a string to a ContentPart of text, making it backward compatible for models.

    |> Utils.assign_string_value(:content, attrs)
    |> validate_required(@required_fields)
    |> apply_action(:insert)
  end

  @doc """
  Create a new `MessageDelta` that represents a message chunk and return it or
  raise an error if invalid.
  """
  @spec new!(attrs :: map()) :: t() | no_return()
  def new!(attrs \\ %{}) do
    case new(attrs) do
      {:ok, message} ->
        message

      {:error, changeset} ->
        raise LangChainError, changeset
    end
  end

  @doc """
  Merge two `MessageDelta` structs. The first `MessageDelta` is the `primary`
  one that smaller deltas are merged into.

      iex> delta_1 =
      ...>   %LangChain.MessageDelta{
      ...>     content: nil,
      ...>     index: 0,
      ...>     tool_calls: [],
      ...>     role: :assistant,
      ...>     status: :incomplete
      ...>   }
      iex> delta_2 =
      ...>   %LangChain.MessageDelta{
      ...>     content: "Hello",
      ...>     index: 0,
      ...>     tool_calls: [],
      ...>     role: :unknown,
      ...>     status: :incomplete
      ...>   }
      iex> LangChain.MessageDelta.merge_delta(delta_1, delta_2)
      %LangChain.MessageDelta{content: "Hello", status: :incomplete, index: 0, role: :assistant, tool_calls: []}

  A set of deltas can be easily merged like this:

      [first | rest] = list_of_delta_message

      Enum.reduce(rest, first, fn new_delta, acc ->
        MessageDelta.merge_delta(acc, new_delta)
      end)

  """
  @spec merge_delta(nil | t(), t()) :: t()
  def merge_delta(nil, %MessageDelta{} = delta_part), do: delta_part

  def merge_delta(%MessageDelta{role: :assistant} = primary, %MessageDelta{} = delta_part) do
    primary
    |> append_content(delta_part)
    |> merge_tool_calls(delta_part)
    |> update_index(delta_part)
    |> update_status(delta_part)
    |> accumulate_token_usage(delta_part)
  end

  # ContentPart being merged
  defp append_content(
         %MessageDelta{role: :assistant, content: []} = primary,
         %MessageDelta{content: %ContentPart{} = new_content_part}
       ) do
    %MessageDelta{primary | content: [new_content_part]}
  end

  defp append_content(
         %MessageDelta{role: :assistant, content: parts_list} = primary,
         %MessageDelta{
           content: new_delta_content,
           index: index
         }
       )
       when is_list(parts_list) and not is_nil(new_delta_content) do
    # Incoming delta has a single ContentPart, not a list
    case new_delta_content do
      %ContentPart{} = part ->
        merge_content_part_at_index(primary, part, index)
    end
  end

  defp append_content(%MessageDelta{} = primary, %MessageDelta{} = _delta_part) do
    # no content to merge
    primary
  end

  # Helper function to merge a content part at a specific index
  defp merge_content_part_at_index(
         %MessageDelta{} = primary,
         %ContentPart{} = new_content_part,
         index
       ) do
    parts_list = primary.content

    # If the index is beyond the current list length, pad with nil values
    padded_list =
      if index >= length(parts_list) do
        parts_list ++ List.duplicate(nil, index - length(parts_list) + 1)
      else
        parts_list
      end

    # Get the content part at the specified index from the primary's content list
    primary_part = Enum.at(padded_list, index)

    # Merge the parts if we have an existing part, otherwise use the new part
    merged_part =
      if primary_part do
        ContentPart.merge_part(primary_part, new_content_part)
      else
        new_content_part
      end

    # Replace the part at the specified index
    updated_list = List.replace_at(padded_list, index, merged_part)

    %MessageDelta{primary | content: updated_list}
  end

  # Insert or update a content part in the list based on its index
  defp insert_or_update_content_part(parts_list, part) when is_list(parts_list) do
    # Find the position index of the item
    idx = Enum.find_index(parts_list, fn item -> item.index == part.index end)

    case idx do
      nil ->
        # not in the list, append the part to the list
        parts_list ++ [part]

      position ->
        # Update the existing part at the position
        List.replace_at(parts_list, position, part)
    end
  end

  defp merge_tool_calls(
         %MessageDelta{} = primary,
         %MessageDelta{tool_calls: [delta_call]} = _delta_part
       ) do
    # point from the primary delta.
    primary_calls = primary.tool_calls || []

    # only the `index` can be counted on in the minimal delta_call for matching
    # against. Anthropic's index is used to differentiate calls but the count is
    # not related to the actual index in the list. For this reason, we match on
    # the index value and not the index as an offset.
    initial = Enum.find(primary_calls, &(&1.index == delta_call.index))

    # merge them and put it back in the correct spot of the list
    merged_call = ToolCall.merge(initial, delta_call)
    # if the index exists, update it, otherwise insert it

    # insert or update the merged item into the list based on the index value
    updated_calls = insert_or_update_tool_call(primary_calls, merged_call)
    # updated_calls = Utils.put_in_list(primary_calls, pos, merged_call)
    # return updated MessageDelta
    %MessageDelta{primary | tool_calls: updated_calls}
  end

  defp merge_tool_calls(%MessageDelta{} = primary, %MessageDelta{} = _delta_part) do
    # nothing to merge
    primary
  end

  defp update_index(%MessageDelta{} = primary, %MessageDelta{index: new_index})
       when is_number(new_index) do
    %MessageDelta{primary | index: new_index}
  end

  defp update_index(%MessageDelta{} = primary, %MessageDelta{} = _delta_part) do
    # no index update
    primary
  end

  defp update_status(%MessageDelta{status: :incomplete} = primary, %MessageDelta{
         status: :complete
       }) do
    %MessageDelta{primary | status: :complete}
  end

  defp update_status(%MessageDelta{status: :incomplete} = primary, %MessageDelta{
         status: :length
       }) do
    %MessageDelta{primary | status: :length}
  end

  defp update_status(%MessageDelta{} = primary, %MessageDelta{} = _delta_part) do
    # status flag not updated
    primary
  end

  # given the list of tool calls, insert or update the item into the list based
  # on the tool_call's index value.
  defp insert_or_update_tool_call(tool_calls, call) when is_list(tool_calls) do
    # find the position index of the item
    idx = Enum.find_index(tool_calls, fn item -> item.index == call.index end)

    case idx do
      nil ->
        # not in the list, append the call to the list
        tool_calls ++ [call]

      position ->
        List.replace_at(tool_calls, position, call)
    end
  end

  @doc """
  Convert the MessageDelta to a Message. Can only convert a fully complete
  MessageDelta.

  This is assumed to be the result of merging all the received `MessageDelta`s.
  An error is returned if the `status` is `:incomplete`.

  If the `MessageDelta` fails to convert to a `LangChain.Message`, an error is
  returned with the reason.
  """
  @spec to_message(t()) :: {:ok, Message.t()} | {:error, String.t()}
  def to_message(%MessageDelta{status: :incomplete} = _delta) do
    {:error, "Cannot convert incomplete message"}
  end

  def to_message(%MessageDelta{status: status} = delta) do
    msg_status =
      case status do
        :complete ->
          :complete

        :length ->
          :length

        _other ->
          nil
      end

    attrs =
      delta
      |> Map.from_struct()
      |> Map.put(:status, msg_status)

    case Message.new(attrs) do
      {:ok, message} ->
        {:ok, message}

      {:error, changeset} ->
        {:error, Utils.changeset_error_to_string(changeset)}
    end
  end

  @doc """
  Accumulates token usage from delta messages. Uses `LangChain.TokenUsage.add/2` to combine
  the usage data from both deltas.

  ## Example

      iex> alias LangChain.TokenUsage
      iex> alias LangChain.MessageDelta
      iex> delta1 = %MessageDelta{
      ...>   metadata: %{
      ...>     usage: %TokenUsage{input: 10, output: 5}
      ...>   }
      ...> }
      iex> delta2 = %MessageDelta{
      ...>   metadata: %{
      ...>     usage: %TokenUsage{input: 5, output: 15}
      ...>   }
      ...> }
      iex> result = MessageDelta.accumulate_token_usage(delta1, delta2)
      iex> result.metadata.usage.input
      15
      iex> result.metadata.usage.output
      20

  """
  @spec accumulate_token_usage(t(), t()) :: t()
  def accumulate_token_usage(
        %MessageDelta{} = primary,
        %MessageDelta{metadata: %{usage: new_usage}} = _delta_part
      )
      when not is_nil(new_usage) do
    current_usage = TokenUsage.get(primary)
    combined_usage = TokenUsage.add(current_usage, new_usage)

    %MessageDelta{primary | metadata: Map.put(primary.metadata || %{}, :usage, combined_usage)}
  end

  def accumulate_token_usage(%MessageDelta{} = primary, %MessageDelta{} = _delta_part) do
    # No usage data to accumulate
    primary
  end
end
