defmodule LangChain.Agents.Todo do
  @moduledoc """
  TODO item structure for task tracking.

  TODOs help agents break down complex tasks into manageable steps and track
  progress through multi-step workflows.

  ## Status Values

  - `:pending` - Task not yet started
  - `:in_progress` - Currently being worked on
  - `:completed` - Task finished successfully
  - `:cancelled` - Task no longer needed

  ## Usage

      # Create a new TODO
      {:ok, todo} = Todo.new(%{
        content: "Implement user authentication",
        status: :pending
      })

      # Update status
      {:ok, updated} = Todo.new(%{
        id: todo.id,
        content: todo.content,
        status: :in_progress
      })
  """

  use Ecto.Schema
  import Ecto.Changeset
  alias __MODULE__

  @primary_key false
  embedded_schema do
    field :id, :string
    field :content, :string

    field :status, Ecto.Enum,
      values: [:pending, :in_progress, :completed, :cancelled],
      default: :pending
  end

  @type t :: %Todo{
          id: String.t(),
          content: String.t(),
          status: :pending | :in_progress | :completed | :cancelled
        }

  @doc """
  Create a new TODO item with validation.

  Generates a unique ID if not provided.

  ## Examples

      {:ok, todo} = Todo.new(%{content: "Write tests"})
      {:ok, todo} = Todo.new(%{id: "custom-id", content: "Task", status: :completed})
  """
  def new(attrs \\ %{}) do
    attrs
    |> ensure_id()
    |> cast_and_validate()
  end

  @doc """
  Create a new TODO item, raising on error.

  ## Examples

      todo = Todo.new!(%{content: "Deploy to production"})
  """
  def new!(attrs \\ %{}) do
    case new(attrs) do
      {:ok, todo} -> todo
      {:error, changeset} -> raise LangChain.LangChainError, changeset
    end
  end

  @doc """
  Convert a TODO struct to a map for serialization.

  ## Examples

      todo = Todo.new!(%{content: "Task"})
      map = Todo.to_map(todo)
      # => %{"id" => "...", "content" => "Task", "status" => "pending"}
  """
  def to_map(%Todo{} = todo) do
    %{
      "id" => todo.id,
      "content" => todo.content,
      "status" => Atom.to_string(todo.status)
    }
  end

  @doc """
  Create a TODO from a map (for deserialization).

  ## Examples

      map = %{"id" => "123", "content" => "Task", "status" => "pending"}
      {:ok, todo} = Todo.from_map(map)
  """
  def from_map(map) when is_map(map) do
    attrs = %{
      id: map["id"] || map[:id],
      content: map["content"] || map[:content],
      status: parse_status(map["status"] || map[:status] || "pending")
    }

    new(attrs)
  end

  # Private functions

  defp ensure_id(%{id: id} = attrs) when is_binary(id) and byte_size(id) > 0, do: attrs

  defp ensure_id(attrs) do
    Map.put(attrs, :id, generate_id())
  end

  # Generate a simple unique ID
  defp generate_id do
    :crypto.strong_rand_bytes(16)
    |> Base.url_encode64(padding: false)
    |> String.slice(0, 22)
  end

  defp cast_and_validate(attrs) do
    %Todo{}
    |> cast(attrs, [:id, :content, :status])
    |> validate_required([:id, :content, :status])
    |> validate_length(:content, min: 1, max: 1000)
    |> validate_inclusion(:status, [:pending, :in_progress, :completed, :cancelled])
    |> apply_action(:insert)
  end

  defp parse_status(status) when is_atom(status), do: status
  defp parse_status("pending"), do: :pending
  defp parse_status("in_progress"), do: :in_progress
  defp parse_status("completed"), do: :completed
  defp parse_status("cancelled"), do: :cancelled
  defp parse_status(_), do: :pending
end
