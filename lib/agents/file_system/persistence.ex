defmodule LangChain.Agents.FileSystem.Persistence do
  @moduledoc """
  Behaviour for persisting files to storage.

  Custom persistence implementations must implement all four callbacks.
  The default implementation writes to the local filesystem.

  ## Usage

  Create a `FileSystemConfig` with your persistence module:

      alias LangChain.Agents.FileSystem.{FileSystemServer, FileSystemConfig}

      {:ok, config} = FileSystemConfig.new(%{
        base_directory: "user_files",
        persistence_module: MyApp.DBPersistence,
        storage_opts: [path: "/data/agents"]
      })

      FileSystemServer.start_link(
        agent_id: "agent-123",
        persistence_configs: [config]
      )

  ## Storage Options

  The `:storage_opts` from your `FileSystemConfig` are passed to your persistence
  module's callbacks via the `opts` parameter. Use it to configure storage location,
  DB connection, credentials, etc.

  The `FileSystemConfig` also automatically adds `:agent_id` and `:base_directory`
  to the opts for convenience.

  ## Callbacks

  All callbacks receive `opts` as their second parameter, which includes:
  - `:agent_id` - The agent's unique identifier
  - `:base_directory` - The virtual directory (from `FileSystemConfig`)
  - All custom options from `FileSystemConfig.storage_opts`

  ### write_to_storage/2

  Write a file entry to persistent storage. Called after the debounce timer fires.

  ### load_from_storage/2

  Load a file's content from persistent storage. Called during lazy loading when a
  file is read but content is not yet in memory.

  ### delete_from_storage/2

  Delete a file from persistent storage. Called immediately when a persisted file
  is deleted (no debounce).

  ### list_persisted_files/2

  List all persisted file paths for an agent. Called during agent initialization
  to index existing files without loading content.
  """

  alias LangChain.Agents.FileSystem.FileEntry

  @doc """
  Write a file entry to persistent storage.

  ## Parameters

  - `file_entry` - The FileEntry to persist (includes path, content, metadata)
  - `opts` - Storage configuration options (e.g., [path: "/data/agents", agent_id: "agent-123"])

  ## Returns

  - `:ok` on success
  - `{:error, reason}` on failure
  """
  @callback write_to_storage(file_entry :: FileEntry.t(), opts :: keyword()) ::
              :ok | {:error, term()}

  @doc """
  Load a file's content from persistent storage.

  ## Parameters

  - `file_entry` - The FileEntry with path to load (content may be nil)
  - `opts` - Storage configuration options

  ## Returns

  - `{:ok, content}` where content is a string
  - `{:error, :enoent}` if file doesn't exist
  - `{:error, reason}` on other failures
  """
  @callback load_from_storage(file_entry :: FileEntry.t(), opts :: keyword()) ::
              {:ok, String.t()} | {:error, term()}

  @doc """
  Delete a file from persistent storage.

  ## Parameters

  - `file_entry` - The FileEntry to delete (uses path field)
  - `opts` - Storage configuration options

  ## Returns

  - `:ok` on success (even if file doesn't exist)
  - `{:error, reason}` on failure
  """
  @callback delete_from_storage(file_entry :: FileEntry.t(), opts :: keyword()) ::
              :ok | {:error, term()}

  @doc """
  List all persisted file paths for an agent.

  Used during agent initialization to index existing files without loading content.

  ## Parameters

  - `agent_id` - The agent's unique identifier
  - `opts` - Storage configuration options

  ## Returns

  - `{:ok, paths}` where paths is a list of file path strings
  - `{:error, reason}` on failure

  ## Example

      {:ok, ["/Memories/chat_log.txt", "/Memories/notes.md"]}
  """
  @callback list_persisted_files(agent_id :: String.t(), opts :: keyword()) ::
              {:ok, [String.t()]} | {:error, term()}
end
