defmodule LangChain.Agents.FileSystem.Persistence.DiskTest do
  use ExUnit.Case, async: true

  alias LangChain.Agents.FileSystem.Persistence.Disk
  alias LangChain.Agents.FileSystem.FileEntry

  @moduletag :tmp_dir

  setup %{tmp_dir: tmp_dir} do
    agent_id = "test_agent_#{System.unique_integer([:positive])}"

    opts = [
      agent_id: agent_id,
      path: tmp_dir,
      memories_directory: "Memories"
    ]

    %{agent_id: agent_id, opts: opts, tmp_dir: tmp_dir}
  end

  describe "write_to_storage/2" do
    test "writes file to disk", %{opts: opts, tmp_dir: tmp_dir, agent_id: agent_id} do
      {:ok, entry} = FileEntry.new_persisted_file("/Memories/test.txt", "test content")

      assert :ok = Disk.write_to_storage(entry, opts)

      # Verify file exists on disk
      expected_path = Path.join([tmp_dir, agent_id, "Memories", "test.txt"])
      assert File.exists?(expected_path)
      assert File.read!(expected_path) == "test content"
    end

    test "creates nested directories", %{opts: opts, tmp_dir: tmp_dir, agent_id: agent_id} do
      {:ok, entry} =
        FileEntry.new_persisted_file("/Memories/deep/nested/file.txt", "nested content")

      assert :ok = Disk.write_to_storage(entry, opts)

      expected_path = Path.join([tmp_dir, agent_id, "Memories", "deep", "nested", "file.txt"])
      assert File.exists?(expected_path)
      assert File.read!(expected_path) == "nested content"
    end

    test "overwrites existing file", %{opts: opts} do
      {:ok, entry} = FileEntry.new_persisted_file("/Memories/test.txt", "original")

      assert :ok = Disk.write_to_storage(entry, opts)

      # Overwrite with new content
      {:ok, updated_entry} = FileEntry.new_persisted_file("/Memories/test.txt", "updated")

      assert :ok = Disk.write_to_storage(updated_entry, opts)

      # Verify updated content
      {:ok, content} = Disk.load_from_storage(entry, opts)
      assert content == "updated"
    end

    test "handles unicode content", %{opts: opts} do
      unicode_content = "Hello 世界 🌍"
      {:ok, entry} = FileEntry.new_persisted_file("/Memories/unicode.txt", unicode_content)

      assert :ok = Disk.write_to_storage(entry, opts)

      {:ok, loaded_content} = Disk.load_from_storage(entry, opts)
      assert loaded_content == unicode_content
    end

    test "handles large content", %{opts: opts} do
      # Generate 1MB of content
      large_content = String.duplicate("a", 1_000_000)
      {:ok, entry} = FileEntry.new_persisted_file("/Memories/large.txt", large_content)

      assert :ok = Disk.write_to_storage(entry, opts)

      {:ok, loaded_content} = Disk.load_from_storage(entry, opts)
      assert byte_size(loaded_content) == 1_000_000
    end
  end

  describe "load_from_storage/2" do
    test "loads existing file", %{opts: opts} do
      content = "file content"
      {:ok, entry} = FileEntry.new_persisted_file("/Memories/load.txt", content)

      # Write first
      :ok = Disk.write_to_storage(entry, opts)

      # Then load
      assert {:ok, ^content} = Disk.load_from_storage(entry, opts)
    end

    test "returns error for non-existent file", %{opts: opts} do
      {:ok, entry} = FileEntry.new_indexed_file("/Memories/nonexistent.txt")

      assert {:error, :enoent} = Disk.load_from_storage(entry, opts)
    end

    test "loads file from nested directory", %{opts: opts} do
      content = "nested file content"
      {:ok, entry} = FileEntry.new_persisted_file("/Memories/a/b/c/file.txt", content)

      :ok = Disk.write_to_storage(entry, opts)

      assert {:ok, ^content} = Disk.load_from_storage(entry, opts)
    end
  end

  describe "delete_from_storage/2" do
    test "deletes existing file", %{opts: opts, tmp_dir: tmp_dir, agent_id: agent_id} do
      {:ok, entry} = FileEntry.new_persisted_file("/Memories/delete.txt", "content")

      # Write file
      :ok = Disk.write_to_storage(entry, opts)

      expected_path = Path.join([tmp_dir, agent_id, "Memories", "delete.txt"])
      assert File.exists?(expected_path)

      # Delete file
      assert :ok = Disk.delete_from_storage(entry, opts)

      refute File.exists?(expected_path)
    end

    test "returns ok for non-existent file", %{opts: opts} do
      {:ok, entry} = FileEntry.new_indexed_file("/Memories/nonexistent.txt")

      assert :ok = Disk.delete_from_storage(entry, opts)
    end

    test "deletes file from nested directory", %{opts: opts, tmp_dir: tmp_dir, agent_id: agent_id} do
      {:ok, entry} = FileEntry.new_persisted_file("/Memories/a/b/file.txt", "content")

      :ok = Disk.write_to_storage(entry, opts)

      expected_path = Path.join([tmp_dir, agent_id, "Memories", "a", "b", "file.txt"])
      assert File.exists?(expected_path)

      assert :ok = Disk.delete_from_storage(entry, opts)

      refute File.exists?(expected_path)
      # Note: Empty parent directories remain (by design)
    end
  end

  describe "list_persisted_files/2" do
    test "returns empty list when no files exist", %{agent_id: agent_id, opts: opts} do
      assert {:ok, []} = Disk.list_persisted_files(agent_id, opts)
    end

    test "lists single file", %{agent_id: agent_id, opts: opts} do
      {:ok, entry} = FileEntry.new_persisted_file("/Memories/file1.txt", "content")
      :ok = Disk.write_to_storage(entry, opts)

      assert {:ok, ["/Memories/file1.txt"]} = Disk.list_persisted_files(agent_id, opts)
    end

    test "lists multiple files", %{agent_id: agent_id, opts: opts} do
      files = [
        {"/Memories/file1.txt", "content1"},
        {"/Memories/file2.txt", "content2"},
        {"/Memories/file3.txt", "content3"}
      ]

      for {path, content} <- files do
        {:ok, entry} = FileEntry.new_persisted_file(path, content)
        :ok = Disk.write_to_storage(entry, opts)
      end

      {:ok, listed_files} = Disk.list_persisted_files(agent_id, opts)
      assert length(listed_files) == 3
      assert "/Memories/file1.txt" in listed_files
      assert "/Memories/file2.txt" in listed_files
      assert "/Memories/file3.txt" in listed_files
    end

    test "lists files in nested directories", %{agent_id: agent_id, opts: opts} do
      files = [
        {"/Memories/root.txt", "root"},
        {"/Memories/dir1/file1.txt", "file1"},
        {"/Memories/dir1/file2.txt", "file2"},
        {"/Memories/dir2/subdir/file3.txt", "file3"}
      ]

      for {path, content} <- files do
        {:ok, entry} = FileEntry.new_persisted_file(path, content)
        :ok = Disk.write_to_storage(entry, opts)
      end

      {:ok, listed_files} = Disk.list_persisted_files(agent_id, opts)
      assert length(listed_files) == 4

      expected_paths = [
        "/Memories/root.txt",
        "/Memories/dir1/file1.txt",
        "/Memories/dir1/file2.txt",
        "/Memories/dir2/subdir/file3.txt"
      ]

      for path <- expected_paths do
        assert path in listed_files, "Expected #{path} to be in listed files"
      end
    end

    test "returns files with correct path format", %{agent_id: agent_id, opts: opts} do
      {:ok, entry} = FileEntry.new_persisted_file("/Memories/test.txt", "content")
      :ok = Disk.write_to_storage(entry, opts)

      {:ok, [path]} = Disk.list_persisted_files(agent_id, opts)

      # Path should start with /
      assert String.starts_with?(path, "/")
      # Path should match virtual filesystem format
      assert path == "/Memories/test.txt"
    end
  end

  describe "default_base_path" do
    test "uses system temp directory by default" do
      agent_id = "test_agent"
      opts = [agent_id: agent_id]

      {:ok, entry} = FileEntry.new_persisted_file("/Memories/temp.txt", "content")
      assert :ok = Disk.write_to_storage(entry, opts)

      # Should write to system temp directory
      temp_base = System.tmp_dir!()
      expected_path = Path.join([temp_base, "langchain_agents", agent_id, "Memories", "temp.txt"])

      assert File.exists?(expected_path)

      # Cleanup
      File.rm_rf!(Path.join([temp_base, "langchain_agents", agent_id]))
    end
  end

  describe "custom memories_directory" do
    test "works with different memories directory", %{tmp_dir: tmp_dir, agent_id: agent_id} do
      opts = [
        agent_id: agent_id,
        path: tmp_dir,
        memories_directory: "persistent"
      ]

      {:ok, entry} = FileEntry.new_persisted_file("/persistent/data.txt", "data")
      assert :ok = Disk.write_to_storage(entry, opts)

      expected_path = Path.join([tmp_dir, agent_id, "persistent", "data.txt"])
      assert File.exists?(expected_path)

      {:ok, [path]} = Disk.list_persisted_files(agent_id, opts)
      assert path == "/persistent/data.txt"
    end
  end

  describe "error handling" do
    @tag :skip
    test "handles write permission errors gracefully" do
      # Note: This test is skipped because it's difficult to reliably create
      # write permission errors in a cross-platform way. The behaviour should
      # return {:error, reason} on File.write failures, which is covered by
      # the File.write/2 documentation.

      # Create a read-only directory
      readonly_dir =
        Path.join(System.tmp_dir!(), "readonly_#{System.unique_integer([:positive])}")

      File.mkdir_p!(readonly_dir)

      agent_dir = Path.join(readonly_dir, "agent")
      File.mkdir_p!(agent_dir)

      # Make it read-only
      File.chmod!(agent_dir, 0o444)

      opts = [agent_id: "test", path: readonly_dir]

      {:ok, entry} = FileEntry.new_persisted_file("/Memories/test.txt", "content")

      # Should return error (not raise)
      assert {:error, _reason} = Disk.write_to_storage(entry, opts)

      # Cleanup
      File.chmod!(agent_dir, 0o755)
      File.rm_rf!(readonly_dir)
    end
  end

  describe "concurrent operations" do
    test "handles concurrent writes to different files", %{opts: opts, agent_id: agent_id} do
      tasks =
        for i <- 1..10 do
          Task.async(fn ->
            {:ok, entry} =
              FileEntry.new_persisted_file("/Memories/file#{i}.txt", "content#{i}")

            Disk.write_to_storage(entry, opts)
          end)
        end

      results = Task.await_many(tasks)
      assert Enum.all?(results, &(&1 == :ok))

      # Verify all files exist
      {:ok, files} = Disk.list_persisted_files(agent_id, opts)
      assert length(files) == 10
    end

    test "handles concurrent writes to same file (last write wins)", %{opts: opts} do
      path = "/Memories/concurrent.txt"

      tasks =
        for i <- 1..5 do
          Task.async(fn ->
            {:ok, entry} = FileEntry.new_persisted_file(path, "content#{i}")
            Disk.write_to_storage(entry, opts)
          end)
        end

      results = Task.await_many(tasks)
      assert Enum.all?(results, &(&1 == :ok))

      # One of the writes should have succeeded
      {:ok, entry} = FileEntry.new_indexed_file(path)
      {:ok, content} = Disk.load_from_storage(entry, opts)
      assert String.starts_with?(content, "content")
    end
  end
end
