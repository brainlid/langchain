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

  - `:path` - Base directory for file storage (default: system temp directory)
  - `:agent_id` - Agent identifier (automatically added by FileSystemConfig)
  - `:base_directory` - Virtual directory name (automatically added by FileSystemConfig)

  ## File Organization

  Files are stored with their full path under the agent's base directory:

      <path>/<agent_id>/<file_path>

  Example with base_directory "user_files":
      /var/lib/langchain/agents/agent-123/user_files/notes.txt

  Example with base_directory "system":
      /var/lib/langchain/agents/agent-123/system/config.json
  """

  @behaviour LangChain.Agents.FileSystem.Persistence

  alias LangChain.Agents.FileSystem.FileEntry

  @impl true
  def write_to_storage(%FileEntry{path: path, content: content}, opts) do
    full_path = build_file_path(path, opts)

    with :ok <- File.mkdir_p(Path.dirname(full_path)),
         :ok <- File.write(full_path, content) do
      :ok
    end
  end

  @impl true
  def load_from_storage(%FileEntry{path: path}, opts) do
    full_path = build_file_path(path, opts)
    File.read(full_path)
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
  def list_persisted_files(agent_id, opts) do
    base_path = build_agent_base_path(agent_id, opts)

    case File.exists?(base_path) do
      true ->
        files = scan_directory(base_path, base_path)
        {:ok, files}

      false ->
        {:ok, []}
    end
  end

  # Private helpers

  defp build_file_path(file_path, opts) do
    # File paths from the virtual filesystem include agent context
    # Extract agent_id from opts if provided, otherwise build from path only
    agent_id = Keyword.get(opts, :agent_id)

    if agent_id do
      base = build_agent_base_path(agent_id, opts)
      # Combine base with the virtual file path
      Path.join(base, String.trim_leading(file_path, "/"))
    else
      # Fallback: use path option directly
      base = Keyword.get(opts, :path, default_base_path())
      Path.join(base, String.trim_leading(file_path, "/"))
    end
  end

  defp build_agent_base_path(agent_id, opts) do
    base = Keyword.get(opts, :path, default_base_path())
    Path.join(base, agent_id)
  end

  defp default_base_path do
    System.tmp_dir!()
    |> Path.join("langchain_agents")
  end

  defp scan_directory(dir, base_path) do
    case File.ls(dir) do
      {:ok, entries} ->
        Enum.flat_map(entries, fn entry ->
          full_path = Path.join(dir, entry)

          cond do
            File.dir?(full_path) ->
              # Recurse into subdirectory
              scan_directory(full_path, base_path)

            File.regular?(full_path) ->
              # Return path relative to base, with leading slash
              relative_path = Path.relative_to(full_path, base_path)
              ["/" <> relative_path]

            true ->
              []
          end
        end)

      {:error, _} ->
        []
    end
  end
end
