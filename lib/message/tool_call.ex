defmodule LangChain.Message.ToolCall do
  @moduledoc """
  Represents an LLM's request to use tool. It specifies the tool to execute and
  may provide arguments for the tool to use.

  ## Example

      ToolCall.new!(%{
        call_id: "call_123",
        name: "give_greeting"
        arguments: %{"name" => "Howard"}
      })

  """
  use Ecto.Schema
  import Ecto.Changeset
  require Logger
  alias __MODULE__
  alias LangChain.LangChainError

  @primary_key false
  embedded_schema do
    field :status, Ecto.Enum, values: [:incomplete, :complete], default: :incomplete
    field :type, Ecto.Enum, values: [:function], default: :function
    field :call_id, :string
    field :name, :string
    field :arguments, :any, virtual: true
    # when the tool call is incomplete, the index indicates which tool call to
    # update on a ToolCall.
    field :index, :integer
  end

  # https://cookbook.openai.com/examples/how_to_call_functions_with_chat_models
  #
  # if assistant_message.tool_calls:
  #   results = execute_function_call(assistant_message)
  #   messages.append({"role": "function", "tool_call_id": assistant_message.tool_calls[0].id, "name": assistant_message.tool_calls[0].function.name, "content": results})

  @type t :: %ToolCall{}

  @update_fields [:status, :type, :call_id, :name, :arguments, :index]
  @create_fields @update_fields

  @doc """
  Build a new ToolCall and return an `:ok`/`:error` tuple with the result.
  """
  @spec new(attrs :: map()) :: {:ok, t()} | {:error, Ecto.Changeset.t()}
  def new(attrs \\ %{}) do
    %ToolCall{}
    |> cast(attrs, @create_fields)
    |> assign_string_value(:arguments, attrs)
    |> common_validations()
    |> apply_action(:insert)
  end

  @doc """
  Build a new ToolCall and return it or raise an error if invalid.
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
  Ensure the ToolCall's status is set to `:complete`. The process of completing
  it parses the tool arguments, which may be invalid. Any problems parsing are
  returned as a changeset error.
  """
  @spec complete(t()) :: {:ok, t()} | {:error, Ecto.Changeset.t()}
  def complete(%ToolCall{status: :complete} = tool_call), do: {:ok, tool_call}

  def complete(%ToolCall{} = tool_call) do
    tool_call
    |> change()
    |> put_change(:status, :complete)
    |> common_validations()
    |> apply_action(:insert)
  end

  defp common_validations(changeset) do
    case get_field(changeset, :status) do
      nil ->
        changeset

      # when representing a "delta" tool call, we are very lax on requirements.
      :incomplete ->
        validate_required(changeset, [:status, :type])

      # when the message should be complete, we are more strict
      :complete ->
        changeset
        |> validate_required([:status, :type, :call_id, :name])
        |> validate_and_parse_arguments()
    end
  end

  defp validate_and_parse_arguments(changeset) do
    case get_field(changeset, :arguments) do
      # the "arguments" are not set
      nil ->
        changeset

      text when is_binary(text) ->
        # assume JSON and convert. If invalid, add an error
        case Jason.decode(text) do
          {:ok, json} when is_map(json) ->
            put_change(changeset, :arguments, json)

          {:ok, _json} ->
            add_error(changeset, :arguments, "a json object is expected for tool arguments")

          {:error, _reason} ->
            add_error(changeset, :arguments, "invalid json")
        end

      data when is_map(data) ->
        # return unmodified
        changeset
    end
  end

  @doc """
  The left side, or `primary`, is the ToolCall that is being accumulated. The
  `call_part` is being merged into the `primary`.

  Used to process streamed deltas where a single tool call can be split over
  many smaller parts.
  """
  @spec merge(nil | t(), t()) :: t()
  def merge(primary, call_part)
  def merge(nil, %ToolCall{} = call_part), do: call_part

  def merge(%ToolCall{index: t1}, %ToolCall{index: t2}) when t1 != t2 do
    raise LangChainError, "Can only merge tool calls with the same index"
  end

  def merge(%ToolCall{} = t1, %ToolCall{} = t2) when t1 == t2, do: t1

  def merge(%ToolCall{} = primary, %ToolCall{} = call_part) do
    # merge the "part" into the primary.
    primary
    |> append_tool_name(call_part)
    |> append_arguments(call_part)
    |> update_index(call_part)
    |> update_call_id(call_part)
    |> update_type(call_part)
    |> update_status(call_part)
  end

  defp append_tool_name(%ToolCall{name: primary_name} = primary, %ToolCall{name: new_name})
       when primary_name == new_name do
    primary
  end

  defp append_tool_name(%ToolCall{} = primary, %ToolCall{name: new_name})
       when is_binary(new_name) do
    %ToolCall{primary | name: (primary.name || "") <> new_name}
  end

  defp append_tool_name(%ToolCall{} = primary, %ToolCall{} = _delta_part) do
    # no function name to merge
    primary
  end

  defp update_index(%ToolCall{} = primary, %ToolCall{index: new_index})
       when is_number(new_index) do
    %ToolCall{primary | index: new_index}
  end

  defp update_index(%ToolCall{} = primary, %ToolCall{} = _delta_part) do
    # no index update
    primary
  end

  defp update_call_id(%ToolCall{} = primary, %ToolCall{call_id: id}) when is_binary(id) do
    %ToolCall{primary | call_id: id}
  end

  defp update_call_id(%ToolCall{} = primary, %ToolCall{} = _delta_part) do
    # no call_id update
    primary
  end

  defp update_type(%ToolCall{} = primary, %ToolCall{type: type})
       when is_atom(type) and not is_nil(type) do
    %ToolCall{primary | type: type}
  end

  defp update_type(%ToolCall{} = primary, %ToolCall{} = _delta_part) do
    # no type update
    primary
  end

  # support changing it to complete, does not go back to incomplete from there
  defp update_status(%ToolCall{status: :incomplete} = primary, %ToolCall{
         status: :complete
       }) do
    %ToolCall{primary | status: :complete}
  end

  defp update_status(%ToolCall{} = primary, %ToolCall{} = _delta_part) do
    # status flag not updated
    primary
  end

  defp append_arguments(%ToolCall{} = primary, %ToolCall{
         arguments: new_arguments
       })
       when is_binary(new_arguments) do
    %ToolCall{primary | arguments: (primary.arguments || "") <> new_arguments}
  end

  defp append_arguments(%ToolCall{} = primary, %ToolCall{} = _delta_part) do
    # no arguments to merge
    primary
  end

  # The contents and arguments get streamed as a string. A tool call may be part of a delta and it
  # might be expected to have strings made up of spaces. The "cast" process of the changeset turns
  # this into `nil` causing us to lose data.
  #
  # We want to take whatever we are given here.
  defp assign_string_value(changeset, field, attrs) do
    # get both possible versions of the arguments.
    case Map.get(attrs, field) || Map.get(attrs, to_string(field)) do
      "" ->
        changeset

      val when is_binary(val) ->
        put_change(changeset, field, val)

      _ ->
        changeset
    end
  end
end
