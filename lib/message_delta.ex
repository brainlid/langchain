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
  about the message delta. It is used to store token usage, model, and other
  LLM-specific information.

  ## Content Fields

  The `content` field may contain:
  - A string (for backward compatibility)
  - A `LangChain.Message.ContentPart` struct
  - An empty list `[]` that is received from some services like Anthropic, which
    is a signal that the content will be a list of content parts

  The module uses two content-related fields:

  * `content` - The raw content received from the LLM. This can be either a
    string (for backward compatibility), a `LangChain.Message.ContentPart`
    struct, or a `[]` indicating it will be a list of content parts. This field
    is cleared (set to `nil`) after merging into `merged_content`.

  * `merged_content` - The accumulated list of `ContentPart`s after merging
    deltas. This is the source of truth for the message content and is used when
    converting to a `LangChain.Message`. When merging deltas:
    - For string content, it's converted to a `ContentPart` of type `:text`
    - For `ContentPart` content, it's merged based on the `index` field
    - Multiple content parts can be maintained in the list to support
      multi-modal responses (text, images, audio) or separate thinking content
      from final text
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
    # The accumulated list of ContentParts after merging deltas
    field :merged_content, :any, virtual: true, default: []
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

  @create_fields [:role, :content, :index, :status, :tool_calls, :metadata, :merged_content]
  @required_fields []

  @doc """
  Create a new `MessageDelta` that represents a message chunk.
  """
  @spec new(attrs :: map()) :: {:ok, t()} | {:error, Ecto.Changeset.t()}
  def new(attrs \\ %{}) do
    %MessageDelta{}
    |> cast(attrs, @create_fields)
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

  The merging process:
  1. Migrates any string content to `ContentPart`s for backward compatibility
  2. Merges the content into `merged_content` based on the `index` field
  3. Clears the `content` field (sets to `nil`) after merging
  4. Updates other fields (tool_calls, status, etc.)

  ## Examples

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
      %LangChain.MessageDelta{
        content: nil,
        merged_content: [%LangChain.Message.ContentPart{type: :text, content: "Hello"}],
        status: :incomplete,
        index: 0,
        role: :assistant,
        tool_calls: []
      }

  A set of deltas can be easily merged like this:

        MessageDelta.merge_deltas(list_of_delta_messages)

  """
  @spec merge_delta(nil | t(), t()) :: t()
  def merge_delta(nil, %MessageDelta{} = delta_part) do
    merge_delta(%MessageDelta{role: :assistant}, delta_part)
  end

  def merge_delta(%MessageDelta{role: :assistant} = primary, %MessageDelta{} = delta_part) do
    new_delta = migrate_to_content_parts(delta_part)

    primary
    |> migrate_to_content_parts()
    |> append_to_merged_content(new_delta)
    |> merge_tool_calls(new_delta)
    |> update_index(new_delta)
    |> update_status(new_delta)
    |> accumulate_token_usage(new_delta)
    |> clear_content()
  end

  @doc """
  Merges a list of `MessageDelta`s into a single `MessageDelta`. The deltas
  are merged in order, with each delta being merged into the result of the
  previous merge.

  ## Examples

      iex> deltas = [
      ...>   %LangChain.MessageDelta{content: "Hello", role: :assistant},
      ...>   %LangChain.MessageDelta{content: " world", role: :assistant},
      ...>   %LangChain.MessageDelta{content: "!", role: :assistant, status: :complete}
      ...> ]
      iex> LangChain.MessageDelta.merge_deltas(deltas)
      %LangChain.MessageDelta{
        content: nil,
        merged_content: [%LangChain.Message.ContentPart{type: :text, content: "Hello world!"}],
        status: :complete,
        role: :assistant
      }

  """
  @spec merge_deltas([t()]) :: t()
  def merge_deltas(deltas) when is_list(deltas) do
    # we accumulate the deltas into the first argument which we call the
    # "primary". Then each successive delta is merged into the primary.
    Enum.reduce(deltas, nil, &merge_delta(&2, &1))
  end

  # Clear the content field after merging into merged_content
  defp clear_content(%MessageDelta{} = delta) do
    %MessageDelta{delta | content: nil}
  end

  # ContentPart being merged
  defp append_to_merged_content(
         %MessageDelta{role: :assistant, merged_content: %ContentPart{} = primary_part} = primary,
         %MessageDelta{content: nil, merged_content: %ContentPart{} = new_content_part}
       ) do
    # merging two deltas that have already been merged from smaller chunks.
    merged_part = ContentPart.merge_part(primary_part, new_content_part)
    %MessageDelta{primary | merged_content: [merged_part]}
  end

  defp append_to_merged_content(
         %MessageDelta{role: :assistant, merged_content: %ContentPart{} = primary_part} = primary,
         %MessageDelta{content: %ContentPart{} = new_content_part}
       ) do
    merged_part = ContentPart.merge_part(primary_part, new_content_part)
    %MessageDelta{primary | merged_content: [merged_part]}
  end

  defp append_to_merged_content(
         %MessageDelta{role: :assistant, merged_content: []} = primary,
         %MessageDelta{content: %ContentPart{} = new_content_part}
       ) do
    %MessageDelta{primary | merged_content: [new_content_part]}
  end

  defp append_to_merged_content(
         %MessageDelta{role: :assistant, merged_content: parts_list} = primary,
         %MessageDelta{
           content: new_delta_content,
           index: index
         }
       )
       when is_list(parts_list) and not is_nil(new_delta_content) do
    # Incoming delta has a single ContentPart, not a list
    case new_delta_content do
      [] ->
        # Incoming delta will be a list of ContentParts
        primary

      %ContentPart{} = part ->
        merge_content_part_at_index(primary, part, index)
    end
  end

  defp append_to_merged_content(%MessageDelta{} = primary, %MessageDelta{} = delta_part) do
    # Handle case where primary has no content and delta has string content
    case {primary.merged_content, delta_part.content} do
      {nil, content} when is_binary(content) ->
        %MessageDelta{primary | merged_content: [ContentPart.text!(content)]}

      {[], content} when is_binary(content) ->
        %MessageDelta{primary | merged_content: [ContentPart.text!(content)]}

      {%ContentPart{} = part, content} when is_binary(content) ->
        new_part = ContentPart.text!(content)
        merged_part = ContentPart.merge_part(part, new_part)
        %MessageDelta{primary | merged_content: [merged_part]}

      _ ->
        # no content to merge
        primary
    end
  end

  # Helper function to merge a content part at a specific index
  defp merge_content_part_at_index(
         %MessageDelta{} = primary,
         %ContentPart{type: :text, content: ""} = _new_content_part,
         _index
       ) do
    # Skip merging empty text content parts to avoid type conflicts
    primary
  end

  defp merge_content_part_at_index(
         %MessageDelta{} = primary,
         %ContentPart{} = new_content_part,
         index
       ) do
    parts_list = primary.merged_content

    # If index is nil, assume position 0 for backward compatibility with some chat models
    position = index || 0

    # Compute the length once to avoid multiple calculations
    list_length = length(parts_list)

    # If the index is beyond the current list length, pad with nil values
    padded_list =
      if position >= list_length do
        parts_list ++ List.duplicate(nil, position - list_length + 1)
      else
        parts_list
      end

    # Get the content part at the specified index from the primary's content list
    primary_part = Enum.at(padded_list, position)

    # Merge the parts if we have an existing part, otherwise use the new part
    merged_part =
      if primary_part do
        ContentPart.merge_part(primary_part, new_content_part)
      else
        new_content_part
      end

    # Replace the part at the specified index
    updated_list = List.replace_at(padded_list, position, merged_part)

    %MessageDelta{primary | merged_content: updated_list}
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
  Convert the MessageDelta's merged content to a string. Specify the type of
  content to convert so it can return just the text parts or thinking parts,
  etc. Defaults to `:text`.
  """
  @spec content_to_string(t(), type :: atom()) :: nil | String.t()
  def content_to_string(delta, type \\ :text)
  def content_to_string(nil, _type), do: nil

  def content_to_string(%MessageDelta{merged_content: merged_content}, type) do
    ContentPart.content_to_string(merged_content, type)
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
      |> Map.put(:content, delta.merged_content)

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

  @doc """
  Migrates a MessageDelta's string content to use `LangChain.Message.ContentPart`.
  This is for backward compatibility with models that don't yet support ContentPart streaming.

  ## Examples

      iex> delta = %LangChain.MessageDelta{content: "Hello world"}
      iex> upgraded = migrate_to_content_parts(delta)
      iex> upgraded.content
      %LangChain.Message.ContentPart{type: :text, content: "Hello world"}

  """
  @spec migrate_to_content_parts(t()) :: t()
  def migrate_to_content_parts(%MessageDelta{content: ""} = delta),
    do: %MessageDelta{delta | content: nil}

  def migrate_to_content_parts(%MessageDelta{content: content} = delta) when is_binary(content) do
    %MessageDelta{delta | content: ContentPart.text!(content)}
  end

  def migrate_to_content_parts(%MessageDelta{} = delta) do
    delta
  end
end
