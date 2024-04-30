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

  """
  use Ecto.Schema
  import Ecto.Changeset
  require Logger
  alias __MODULE__
  alias LangChain.LangChainError
  alias LangChain.Message
  alias LangChain.Message.ToolCall
  alias LangChain.Utils

  @primary_key false
  embedded_schema do
    field :content, :any, virtual: true
    # Marks if the delta completes the message.
    field :status, Ecto.Enum, values: [:incomplete, :complete, :length], default: :incomplete
    # When requesting multiple choices for a response, the `index` represents
    # which choice it is. It is a 0 based list.()
    field :index, :integer

    field :role, Ecto.Enum, values: [:unknown, :assistant], default: :unknown

    field :tool_calls, :any, virtual: true
  end

  @type t :: %MessageDelta{}

  @create_fields [:role, :content, :index, :status, :tool_calls]
  @required_fields []

  @doc """
  Create a new `MessageDelta` that represents a message chunk.
  """
  @spec new(attrs :: map()) :: {:ok, t()} | {:error, Ecto.Changeset.t()}
  def new(attrs \\ %{}) do
    %MessageDelta{}
    |> cast(attrs, @create_fields)
    |> assign_string_value(:content, attrs)
    |> assign_string_value(:arguments, attrs)
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
  end

  defp append_content(%MessageDelta{role: :assistant} = primary, %MessageDelta{
         content: new_content
       })
       when is_binary(new_content) do
    %MessageDelta{primary | content: (primary.content || "") <> new_content}
  end

  defp append_content(%MessageDelta{} = primary, %MessageDelta{} = _delta_part) do
    # no content to merge
    primary
  end

  defp merge_tool_calls(
         %MessageDelta{} = primary,
         %MessageDelta{tool_calls: [delta_call]} = _delta_part
       ) do
    # point from the primary delta.
    primary_calls = primary.tool_calls || []
    # get the index of the call being merged
    initial = Enum.at(primary_calls, delta_call.index)
    # merge them and put it back in the correct spot of the list
    merged_call = ToolCall.merge(initial, delta_call)
    # if the index exists, update it, otherwise insert it
    updated_calls = Utils.put_in_list(primary_calls, delta_call.index, merged_call)
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

  defp update_index(%MessageDelta{} = primary, %MessageDelta{} = _delta_par) do
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

  # The contents and arguments get streamed as a string. A delta of " " a single empty space
  # is expected. The "cast" process of the changeset turns this into `nil`
  # causing us to lose data.
  #
  # We want to take whatever we are given here.
  defp assign_string_value(changeset, field, attrs) do
    # get both possible versions of the arguments.
    val = Map.get(attrs, field) || Map.get(attrs, to_string(field))
    # if we got a string, use it as-is without casting
    if is_binary(val) do
      put_change(changeset, field, val)
    else
      changeset
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
end
