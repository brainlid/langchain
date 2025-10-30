defmodule LangChain.Agents.FileSystem.FileEntry do
  @moduledoc """
  Represents a file in the virtual filesystem.

  ## Field Semantics

  - `:persistence` - Storage strategy
    - `:memory` - Ephemeral, exists only in ETS (e.g., /scratch files)
    - `:persisted` - Durable, backed by storage (e.g., /Memories files)

  - `:loaded` - Content availability
    - `true` - Content is in ETS, ready to use
    - `false` - Content exists in storage but not loaded (lazy load on read)

  - `:dirty` - Sync status (only meaningful when persistence: :persisted)
    - `false` - In-memory content matches storage
    - `true` - In-memory content differs from storage (needs persist)

  ## State Examples

  Memory file (always loaded, never dirty):
    %FileEntry{persistence: :memory, loaded: true, dirty: false, content: "data"}

  Persisted file, clean, loaded:
    %FileEntry{persistence: :persisted, loaded: true, dirty: false, content: "data"}

  Persisted file, not yet loaded (lazy):
    %FileEntry{persistence: :persisted, loaded: false, dirty: false, content: nil}

  Persisted file, modified since last save:
    %FileEntry{persistence: :persisted, loaded: true, dirty: true, content: "new data"}
  """
  use Ecto.Schema
  import Ecto.Changeset

  alias __MODULE__
  alias LangChain.Agents.FileSystem.FileMetadata
  alias LangChain.Utils

  @primary_key false
  embedded_schema do
    field :path, :string
    # Content is always string for LLM text-based work
    field :content, :string
    # Where the file lives: memory-only or persisted to storage
    field :persistence, Ecto.Enum, values: [:memory, :persisted], default: :memory
    # Is content currently loaded in ETS? (false = lazy load needed)
    field :loaded, :boolean, default: true
    # Has content been modified since last storage write? (only relevant for :persisted files)
    field :dirty, :boolean, default: false
    embeds_one :metadata, FileMetadata
  end

  @type t :: %FileEntry{
          path: String.t(),
          content: String.t() | nil,
          persistence: :memory | :persisted,
          loaded: boolean(),
          dirty: boolean(),
          metadata: FileMetadata.t() | nil
        }

  @doc """
  Creates a changeset for a file entry.
  """
  def changeset(entry \\ %FileEntry{}, attrs) do
    entry
    |> cast(attrs, [:path, :content, :persistence, :loaded, :dirty])
    |> cast_embed(:metadata, with: &FileMetadata.changeset/2)
    |> validate_path()
    |> validate_required([:persistence, :loaded, :dirty])
  end

  @doc """
  Creates a new file entry for memory storage.
  """
  def new_memory_file(path, content, opts \\ []) do
    with {:ok, metadata} <- FileMetadata.new(content, opts) do
      attrs = %{
        path: path,
        content: content,
        persistence: :memory,
        loaded: true,
        dirty: false
      }

      %FileEntry{}
      |> changeset(attrs)
      |> put_embed(:metadata, metadata)
      |> apply_action(:insert)
      |> case do
        {:ok, entry} -> {:ok, entry}
        {:error, changeset} -> {:error, Utils.changeset_error_to_string(changeset)}
      end
    else
      {:error, _changeset} = error -> error
    end
  end

  @doc """
  Creates a new file entry for persisted storage. Intended for situations when
  the LLM instructs a new file to be created that will need to be persisted to
  storage.
  """
  def new_persisted_file(path, content, opts \\ []) do
    with {:ok, metadata} <- FileMetadata.new(content, opts) do
      attrs = %{
        path: path,
        content: content,
        persistence: :persisted,
        loaded: true,
        dirty: true
      }

      %FileEntry{}
      |> changeset(attrs)
      |> put_embed(:metadata, metadata)
      |> apply_action(:insert)
      |> case do
        {:ok, entry} -> {:ok, entry}
        {:error, changeset} -> {:error, Utils.changeset_error_to_string(changeset)}
      end
    else
      {:error, _changeset} = error -> error
    end
  end

  @doc """
  Creates a file entry for an indexed persisted file (not yet loaded).
  """
  def new_indexed_file(path) do
    attrs = %{
      path: path,
      content: nil,
      persistence: :persisted,
      loaded: false,
      dirty: false,
      metadata: nil
    }

    %FileEntry{}
    |> changeset(attrs)
    |> apply_action(:insert)
    |> case do
      {:ok, entry} -> {:ok, entry}
      {:error, changeset} -> {:error, Utils.changeset_error_to_string(changeset)}
    end
  end

  @doc """
  Marks a file as loaded with the given content.
  """
  def mark_loaded(entry, content) do
    case FileMetadata.new(content, []) do
      {:ok, metadata} ->
        {:ok, %{entry | content: content, loaded: true, metadata: metadata}}

      error ->
        error
    end
  end

  @doc """
  Marks a persisted file as clean (synced with storage).
  """
  def mark_clean(entry) do
    %{entry | dirty: false}
  end

  @doc """
  Updates file content and marks as dirty if persisted.
  """
  def update_content(entry, new_content, opts \\ []) do
    case FileMetadata.new(new_content, opts) do
      {:ok, new_metadata} ->
        dirty = entry.persistence == :persisted
        {:ok, %{entry | content: new_content, loaded: true, dirty: dirty, metadata: new_metadata}}

      error ->
        error
    end
  end

  # Private validation helpers

  defp validate_path(changeset) do
    changeset
    |> validate_required([:path])
    |> validate_format(:path, ~r{^/}, message: "must start with /")
    |> validate_no_double_dots()
  end

  defp validate_no_double_dots(changeset) do
    path = Ecto.Changeset.get_field(changeset, :path)

    if path && String.contains?(path, "..") do
      Ecto.Changeset.add_error(changeset, :path, "cannot contain ..")
    else
      changeset
    end
  end
end
