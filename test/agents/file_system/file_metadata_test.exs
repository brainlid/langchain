defmodule LangChain.Agents.FileSystem.FileMetadataTest do
  use ExUnit.Case, async: true

  alias LangChain.Agents.FileSystem.FileMetadata

  describe "new/2" do
    test "creates metadata for content with defaults" do
      content = "Hello, world!"
      assert {:ok, metadata} = FileMetadata.new(content)

      assert metadata.size == byte_size(content)
      assert metadata.mime_type == "text/markdown"
      assert metadata.encoding == "utf-8"
      assert is_binary(metadata.checksum)
      assert %DateTime{} = metadata.created_at
      assert %DateTime{} = metadata.modified_at
      assert metadata.custom == %{}
    end

    test "creates metadata with custom mime_type" do
      content = "Plain text"
      assert {:ok, metadata} = FileMetadata.new(content, mime_type: "text/plain")

      assert metadata.mime_type == "text/plain"
    end

    test "creates metadata with custom fields" do
      content = "data"
      custom = %{"tags" => ["important"], "source" => "user"}
      assert {:ok, metadata} = FileMetadata.new(content, custom: custom)

      assert metadata.custom == custom
    end

    test "computes checksum correctly" do
      content = "test content"
      assert {:ok, metadata1} = FileMetadata.new(content)
      assert {:ok, metadata2} = FileMetadata.new(content)

      # Same content should produce same checksum
      assert metadata1.checksum == metadata2.checksum

      # Different content should produce different checksum
      assert {:ok, metadata3} = FileMetadata.new("different content")
      assert metadata1.checksum != metadata3.checksum
    end

    test "timestamps are set on creation" do
      content = "data"
      before_time = DateTime.utc_now()
      assert {:ok, metadata} = FileMetadata.new(content)
      after_time = DateTime.utc_now()

      assert DateTime.compare(metadata.created_at, before_time) in [:gt, :eq]
      assert DateTime.compare(metadata.created_at, after_time) in [:lt, :eq]
      assert metadata.created_at == metadata.modified_at
    end
  end

  describe "update_for_modification/2" do
    test "updates timestamps and checksum for new content" do
      content = "original"
      assert {:ok, metadata} = FileMetadata.new(content)
      original_created_at = metadata.created_at
      original_checksum = metadata.checksum

      # Wait a tiny bit to ensure timestamp changes
      Process.sleep(10)

      new_content = "modified"
      changeset = FileMetadata.update_for_modification(metadata, new_content)
      assert changeset.valid?

      updated_metadata = Ecto.Changeset.apply_changes(changeset)

      # created_at should remain unchanged
      assert updated_metadata.created_at == original_created_at

      # modified_at should be updated
      assert DateTime.compare(updated_metadata.modified_at, original_created_at) == :gt

      # Size and checksum should be updated
      assert updated_metadata.size == byte_size(new_content)
      assert updated_metadata.checksum != original_checksum
    end
  end

  describe "changeset/2" do
    test "validates required fields" do
      changeset = FileMetadata.changeset(%FileMetadata{}, %{})
      refute changeset.valid?
      assert changeset.errors[:size]
      # mime_type and encoding have defaults, so they won't error
    end

    test "validates size is non-negative" do
      attrs = %{
        size: -1,
        mime_type: "text/plain",
        encoding: "utf-8",
        created_at: DateTime.utc_now(),
        modified_at: DateTime.utc_now()
      }

      changeset = FileMetadata.changeset(%FileMetadata{}, attrs)
      refute changeset.valid?
      assert changeset.errors[:size]
    end

    test "accepts valid attributes" do
      now = DateTime.utc_now()

      attrs = %{
        size: 100,
        created_at: now,
        modified_at: now,
        mime_type: "application/json",
        encoding: "utf-8",
        checksum: "abc123",
        custom: %{"key" => "value"}
      }

      changeset = FileMetadata.changeset(%FileMetadata{}, attrs)
      assert changeset.valid?

      metadata = Ecto.Changeset.apply_changes(changeset)
      assert metadata.size == 100
      assert metadata.mime_type == "application/json"
      assert metadata.custom == %{"key" => "value"}
    end
  end
end
