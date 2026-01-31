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
  Merges a list of `MessageDelta`s into an accumulated `MessageDelta`. This is
  useful for UI applications that accumulate deltas as they are received from
  a streaming LLM response.

  The first argument is the accumulated delta (or `nil` if no deltas have been
  processed yet). The second argument is a list of new deltas to merge in.

  ## Examples

      iex> accumulated = nil
      iex> batch_1 = [
      ...>   %LangChain.MessageDelta{content: "Hello", role: :assistant},
      ...>   %LangChain.MessageDelta{content: " world", role: :assistant}
      ...> ]
      iex> accumulated = LangChain.MessageDelta.merge_deltas(accumulated, batch_1)
      iex> batch_2 = [
      ...>   %LangChain.MessageDelta{content: "!", role: :assistant, status: :complete}
      ...> ]
      iex> result = LangChain.MessageDelta.merge_deltas(accumulated, batch_2)
      iex> result
      %LangChain.MessageDelta{
        content: nil,
        merged_content: [%LangChain.Message.ContentPart{type: :text, content: "Hello world!"}],
        status: :complete,
        role: :assistant
      }

  """
  @spec merge_deltas(nil | t(), [t()]) :: t()
  def merge_deltas(accumulated_delta, deltas) when is_list(deltas) do
    deltas
    |> List.flatten()
    |> Enum.reduce(accumulated_delta, &merge_delta(&2, &1))
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
    merge_deltas(nil, deltas)
  end

  # Clear the content field after merging into merged_content
  defp clear_content(%MessageDelta{} = delta) do
    %MessageDelta{delta | content: nil}
  end

  # Empty content - nothing to merge
  defp append_to_merged_content(%MessageDelta{} = primary, %MessageDelta{content: content})
       when content in [nil, []],
       do: primary

  # String content - convert to ContentPart and merge
  defp append_to_merged_content(%MessageDelta{} = primary, %MessageDelta{content: content})
       when is_binary(content) do
    append_to_merged_content(primary, %MessageDelta{content: ContentPart.text!(content)})
  end

  # ContentPart from merged_content (already processed delta)
  defp append_to_merged_content(
         %MessageDelta{} = primary,
         %MessageDelta{content: nil, merged_content: %ContentPart{} = part}
       ) do
    merge_content_part_into(primary, part)
  end

  # Single ContentPart content
  defp append_to_merged_content(
         %MessageDelta{} = primary,
         %MessageDelta{content: %ContentPart{} = part, index: index}
       ) do
    merge_content_part_at_index(primary, part, index)
  end

  # List of content items (e.g., from Mistral with reference types)
  defp append_to_merged_content(
         %MessageDelta{merged_content: parts_list} = primary,
         %MessageDelta{content: content_list, index: index}
       )
       when is_list(parts_list) and is_list(content_list) do
    Enum.reduce(content_list, primary, &merge_content_item(&2, &1, index))
  end

  # Catch-all - no content to merge
  defp append_to_merged_content(%MessageDelta{} = primary, %MessageDelta{}), do: primary

  # Merge a ContentPart directly into merged_content (for already-merged deltas)
  @spec merge_content_part_into(t(), ContentPart.t()) :: t()
  defp merge_content_part_into(%MessageDelta{merged_content: content} = primary, part)
       when content in [nil, []] do
    %MessageDelta{primary | merged_content: [part]}
  end

  defp merge_content_part_into(
         %MessageDelta{merged_content: %ContentPart{} = existing} = primary,
         part
       ) do
    %MessageDelta{primary | merged_content: [ContentPart.merge_part(existing, part)]}
  end

  defp merge_content_part_into(%MessageDelta{merged_content: [_ | _]} = primary, part) do
    merge_content_part_at_index(primary, part, 0)
  end

  # Merge a single content item from a list into the primary delta
  @spec merge_content_item(t(), ContentPart.t() | map(), integer() | nil) :: t()
  defp merge_content_item(primary, %ContentPart{} = part, index) do
    merge_content_part_at_index(primary, part, index)
  end

  defp merge_content_item(primary, %{"type" => "text", "text" => text}, index)
       when is_binary(text) do
    merge_content_part_at_index(primary, ContentPart.text!(text), index)
  end

  defp merge_content_item(primary, _unrecognized, _index), do: primary

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

  # Merge tool call delta by matching on index value (not list position).
  # Anthropic's index differentiates calls but doesn't correspond to list offset.
  @spec merge_tool_calls(t(), t()) :: t()
  defp merge_tool_calls(%MessageDelta{tool_calls: primary_calls} = primary, %MessageDelta{
         tool_calls: [delta_call]
       }) do
    calls = primary_calls || []
    initial = Enum.find(calls, &(&1.index == delta_call.index))
    merged_call = ToolCall.merge(initial, delta_call)
    %MessageDelta{primary | tool_calls: upsert_by_index(calls, merged_call)}
  end

  defp merge_tool_calls(%MessageDelta{} = primary, %MessageDelta{}), do: primary

  @spec update_index(t(), t()) :: t()
  defp update_index(%MessageDelta{} = primary, %MessageDelta{index: idx}) when is_number(idx) do
    %MessageDelta{primary | index: idx}
  end

  defp update_index(%MessageDelta{} = primary, %MessageDelta{}), do: primary

  # Only update status from :incomplete to a terminal state
  @spec update_status(t(), t()) :: t()
  defp update_status(%MessageDelta{status: :incomplete} = primary, %MessageDelta{status: status})
       when status in [:complete, :length] do
    %MessageDelta{primary | status: status}
  end

  defp update_status(%MessageDelta{} = primary, %MessageDelta{}), do: primary

  # Insert or update tool call in list by matching on index field
  @spec upsert_by_index([ToolCall.t()], ToolCall.t()) :: [ToolCall.t()]
  defp upsert_by_index(calls, call) do
    case Enum.find_index(calls, &(&1.index == call.index)) do
      nil -> calls ++ [call]
      pos -> List.replace_at(calls, pos, call)
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

  def to_message(%MessageDelta{merged_content: merged_content} = delta) do
    content = reject_nil(merged_content)

    attrs =
      delta
      |> Map.from_struct()
      |> Map.put(:content, content)

    with :ok <- validate_not_empty(delta),
         {:ok, message} <- Message.new(attrs) do
      {:ok, message}
    else
      {:error, %Ecto.Changeset{} = changeset} ->
        {:error, Utils.changeset_error_to_string(changeset)}

      {:error, _reason} = error ->
        error
    end
  end

  # Validate that assistant message is not empty (no content and no tool_calls).
  # Empty assistant messages violate conversation flow rules in most LLM APIs.
  defp validate_not_empty(%MessageDelta{
         role: :assistant,
         merged_content: merged_content,
         tool_calls: tool_calls,
         status: :complete
       })
       when (merged_content == [] or merged_content == nil) and
              (tool_calls == [] or tool_calls == nil) do
    Logger.warning("Received empty assistant message with no content and no tool_calls")

    {:error, "Empty assistant message: no content and no tool_calls"}
  end

  defp validate_not_empty(_delta), do: :ok

  # Filter nil values from list (from index padding during delta merging)
  @spec reject_nil(list() | any()) :: list() | any()
  defp reject_nil(list) when is_list(list), do: Enum.reject(list, &is_nil/1)
  defp reject_nil(other), do: other

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

  @doc """
  Safely adds tool call display information to a MessageDelta's metadata.
  Handles nil delta by creating a new one if needed.

  ## Parameters
  - `delta` - MessageDelta struct or nil
  - `tool_info` - Map with :name, :call_id, :arguments, etc.

  ## Returns
  Updated MessageDelta with enriched metadata

  ## Example
      delta = MessageDelta.add_tool_display_info(nil, %{
        name: "write_file",
        call_id: "call_123",
        display_name: "Writing file..."
      })

      MessageDelta.get_tool_display_info(delta)
      # => [%{name: "write_file", call_id: "call_123", ...}]
  """
  @spec add_tool_display_info(nil | t(), map()) :: t()
  def add_tool_display_info(nil, tool_info) do
    add_tool_display_info(%MessageDelta{status: :incomplete}, tool_info)
  end

  def add_tool_display_info(%MessageDelta{} = delta, tool_info) do
    # Store in metadata under :streaming_tool_calls key
    existing_tools = get_in(delta.metadata || %{}, [:streaming_tool_calls]) || []

    # Add or update tool call info
    updated_tools =
      case Enum.find_index(existing_tools, &(&1.call_id == tool_info.call_id)) do
        nil ->
          # New tool call
          existing_tools ++ [tool_info]

        index ->
          # Update existing
          List.replace_at(existing_tools, index, tool_info)
      end

    metadata = delta.metadata || %{}
    %MessageDelta{delta | metadata: Map.put(metadata, :streaming_tool_calls, updated_tools)}
  end

  @doc """
  Retrieves tool call display information from MessageDelta metadata.
  Returns empty list if none present.
  """
  @spec get_tool_display_info(nil | t()) :: list(map())
  def get_tool_display_info(nil), do: []

  def get_tool_display_info(%MessageDelta{} = delta) do
    get_in(delta.metadata || %{}, [:streaming_tool_calls]) || []
  end
end
