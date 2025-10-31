defmodule LangChain.Agents.FileSystem.Persistence.Disk do
  @moduledoc """
  Default disk-based persistence implementation.

  Stores files on the local filesystem. The storage location is specified
  via the `:path` option in `storage_opts` of your `FileSystemConfig`.

  ## Configuration

      alias LangChain.Agents.FileSystem.{FileSystemServer, FileSystemConfig}
      alias LangChain.Agents.FileSystem.Persistence.Disk

      {:ok, config} = FileSystemConfig.new(%{
        base_directory: "user_files",
        persistence_module: Disk,
        debounce_ms: 5000,
        storage_opts: [path: "/var/lib/langchain/agents"]
      })

      FileSystemServer.start_link(
        agent_id: "agent-123",
        persistence_configs: [config]
      )

  ## Multiple Directories Example

      # Database storage for user files
      {:ok, user_config} = FileSystemConfig.new(%{
        base_directory: "user_files",
        persistence_module: Disk,
        storage_opts: [path: "/var/data/users"]
      })

      # Read-only system files
      {:ok, system_config} = FileSystemConfig.new(%{
        base_directory: "system",
        persistence_module: Disk,
        readonly: true,
        storage_opts: [path: "/var/data/system"]
      })

      FileSystemServer.start_link(
        agent_id: "agent-123",
        persistence_configs: [user_config, system_config]
      )

  ## Storage Options

  - `:path` - Base directory for file storage (required)
  - `:base_directory` - Virtual directory name (automatically added by FileSystemConfig)

  ## File Organization

  Files are stored by stripping the base_directory from the virtual path and
  saving to the storage path:

      <storage_path>/<file_path - base_directory>

  Example with base_directory "user_files":
      Virtual path: "/user_files/notes.txt"
      Storage path: "/var/data/agents"
      Disk path: "/var/data/agents/notes.txt"

  Example with base_directory "system":
      Virtual path: "/system/config.json"
      Storage path: "/var/data/system"
      Disk path: "/var/data/system/config.json"
  """

  @behaviour LangChain.Agents.FileSystem.Persistence

  alias LangChain.Agents.FileSystem.FileEntry
  alias LangChain.Agents.FileSystem.FileMetadata

  @impl true
  def write_to_storage(%FileEntry{path: path, content: content} = entry, opts) do
    full_path = build_file_path(path, opts)

    with :ok <- File.mkdir_p(Path.dirname(full_path)),
         :ok <- File.write(full_path, content),
         {:ok, stat} <- File.stat(full_path) do
      # Create updated metadata with actual file system information
      {:ok, metadata} =
        FileMetadata.new(content,
          mime_type: entry.metadata.mime_type,
          custom: entry.metadata.custom
        )

      # Update metadata with actual file system timestamps
      metadata = %{metadata | created_at: stat.ctime, modified_at: stat.mtime}

      # Return FileEntry with updated metadata and marked as clean
      updated_entry = %{entry | metadata: metadata, dirty: false, loaded: true}
      {:ok, updated_entry}
    end
  end

  @impl true
  def load_from_storage(%FileEntry{path: path} = entry, opts) do
    full_path = build_file_path(path, opts)

    with {:ok, content} <- File.read(full_path),
         {:ok, stat} <- File.stat(full_path) do
      # Detect MIME type from file extension
      mime_type = MIME.from_path(path)

      # Create metadata from the loaded content
      {:ok, metadata} =
        FileMetadata.new(content,
          mime_type: mime_type,
          custom: %{}
        )

      # Update metadata with actual file system timestamps
      metadata = %{metadata | created_at: stat.ctime, modified_at: stat.mtime}

      # Return FileEntry with content and metadata
      loaded_entry = %{entry | content: content, metadata: metadata, loaded: true, dirty: false}
      {:ok, loaded_entry}
    end
  end

  @impl true
  def delete_from_storage(%FileEntry{path: path}, opts) do
    full_path = build_file_path(path, opts)

    case File.rm(full_path) do
      :ok -> :ok
      {:error, :enoent} -> :ok
      error -> error
    end
  end

  @impl true
  def list_persisted_files(_agent_id, opts) do
    storage_path = Keyword.fetch!(opts, :path)
    base_directory = Keyword.get(opts, :base_directory, "")

    case File.exists?(storage_path) do
      true ->
        files = scan_directory(storage_path, storage_path, base_directory)
        {:ok, files}

      false ->
        {:ok, []}
    end
  end

  # Private helpers

  defp build_file_path(file_path, opts) do
    # Get the required storage path from opts
    storage_path = Keyword.fetch!(opts, :path)

    # Get the base_directory to strip from the virtual path
    base_directory = Keyword.get(opts, :base_directory, "")

    # Strip the base_directory prefix from the virtual file path
    # e.g., "/user_files/notes.txt" -> "/notes.txt" when base_directory is "user_files"
    relative_path =
      if base_directory != "" do
        String.replace_prefix(file_path, "/#{base_directory}", "")
      else
        file_path
      end

    # Join storage path with the relative path
    Path.join(storage_path, String.trim_leading(relative_path, "/"))
  end

  defp scan_directory(dir, storage_path, base_directory) do
    case File.ls(dir) do
      {:ok, entries} ->
        Enum.flat_map(entries, fn entry ->
          full_path = Path.join(dir, entry)

          cond do
            File.dir?(full_path) ->
              # Recurse into subdirectory
              scan_directory(full_path, storage_path, base_directory)

            File.regular?(full_path) ->
              # Get path relative to storage root
              relative_path = Path.relative_to(full_path, storage_path)

              # Prepend base_directory to create virtual path
              virtual_path =
                if base_directory != "" do
                  "/#{base_directory}/#{relative_path}"
                else
                  "/#{relative_path}"
                end

              [virtual_path]

            true ->
              []
          end
        end)

      {:error, _} ->
        []
    end
  end
end
