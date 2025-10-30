defmodule LangChain.Agents.FileSystem.FileMetadata do
  @moduledoc """
  Metadata for a file entry in the virtual filesystem.

  Since all content is text (for LLM work), mime_type typically indicates
  the format: text/plain, text/markdown, application/json, etc.
  """
  use Ecto.Schema
  import Ecto.Changeset
  alias __MODULE__

  @primary_key false
  embedded_schema do
    field :size, :integer
    field :created_at, :utc_datetime_usec
    field :modified_at, :utc_datetime_usec
    field :mime_type, :string, default: "text/plain"
    field :encoding, :string, default: "utf-8"
    field :checksum, :string
    # Custom metadata as map
    field :custom, :map, default: %{}
  end

  @type t :: %FileMetadata{
          size: integer() | nil,
          created_at: DateTime.t() | nil,
          modified_at: DateTime.t() | nil,
          mime_type: String.t(),
          encoding: String.t(),
          checksum: String.t() | nil,
          custom: map()
        }

  @doc """
  Creates a changeset for file metadata.
  """
  def changeset(metadata \\ %FileMetadata{}, attrs) do
    metadata
    |> cast(attrs, [:size, :created_at, :modified_at, :mime_type, :encoding, :checksum, :custom])
    |> validate_required([:size, :mime_type, :encoding])
    |> validate_number(:size, greater_than_or_equal_to: 0)
  end

  @doc """
  Creates new metadata for file content with optional custom attributes.
  """
  def new(content, opts \\ []) do
    now = DateTime.utc_now()
    size = byte_size(content)
    mime_type = Keyword.get(opts, :mime_type, "text/markdown")
    custom = Keyword.get(opts, :custom, %{})

    attrs = %{
      size: size,
      created_at: now,
      modified_at: now,
      mime_type: mime_type,
      encoding: "utf-8",
      checksum: compute_checksum(content),
      custom: custom
    }

    case changeset(%FileMetadata{}, attrs) do
      %{valid?: true} = cs -> {:ok, Ecto.Changeset.apply_changes(cs)}
      cs -> {:error, cs}
    end
  end

  @doc """
  Updates metadata timestamps and checksum for modified content.
  """
  def update_for_modification(metadata, new_content) do
    now = DateTime.utc_now()
    size = byte_size(new_content)
    checksum = compute_checksum(new_content)

    attrs = %{
      size: size,
      modified_at: now,
      checksum: checksum
    }

    changeset(metadata, attrs)
  end

  # Private helpers

  defp compute_checksum(content) do
    :crypto.hash(:sha256, content)
    |> Base.encode16(case: :lower)
  end
end
