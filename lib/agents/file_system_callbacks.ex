defmodule LangChain.Agents.FileSystemCallbacks do
  @moduledoc """
  Behavior for file system persistence callbacks.

  Implement this behavior to provide custom persistence for filesystem operations.
  All callbacks are optional - implement only what you need.

  ## Example Implementation

      defmodule MyApp.FilesystemPersistence do
        @behaviour LangChain.Agents.FilesystemCallbacks

        @impl true
        def on_write(file_path, content, context) do
          # Save to database
          case MyApp.Files.create_or_update(context.user_id, file_path, content) do
            {:ok, file} -> {:ok, %{id: file.id}}
            {:error, reason} -> {:error, reason}
          end
        end

        @impl true
        def on_read(file_path, context) do
          case MyApp.Files.get(context.user_id, file_path) do
            nil -> {:error, :not_found}
            file -> {:ok, file.content}
          end
        end

        @impl true
        def on_delete(file_path, context) do
          case MyApp.Files.delete(context.user_id, file_path) do
            {:ok, _} -> {:ok, %{}}
            {:error, reason} -> {:error, reason}
          end
        end

        @impl true
        def on_list(context) do
          files = MyApp.Files.list_for_user(context.user_id)
          {:ok, Enum.map(files, & &1.path)}
        end
      end

  ## Configuration

      {:ok, agent} = Agents.new(
        model: model,
        filesystem_opts: [
          persistence: MyApp.FilesystemPersistence,
          context: %{user_id: user_id, session_id: session_id}
        ]
      )

  ## Alternative: Function-based Callbacks

      {:ok, agent} = Agents.new(
        model: model,
        filesystem_opts: [
          on_write: fn file_path, content, context ->
            MyApp.Files.save(context.user_id, file_path, content)
          end,
          on_read: fn file_path, context ->
            case MyApp.Files.get(context.user_id, file_path) do
              nil -> {:error, :not_found}
              file -> {:ok, file.content}
            end
          end,
          context: %{user_id: user_id}
        ]
      )
  """

  @type file_path :: String.t()
  @type content :: String.t()
  @type context :: map()
  @type result :: {:ok, term()} | {:error, term()}

  @doc """
  Called when a file is created or overwritten.

  ## Parameters
  - `file_path` - Path of the file being written
  - `content` - Full content of the file
  - `context` - Additional context (user_id, session_id, etc.)

  ## Returns
  - `{:ok, metadata}` - Success, optionally with metadata
  - `{:error, reason}` - Failure reason
  """
  @callback on_write(file_path, content, context) :: result()

  @doc """
  Called when a file is read.

  If this callback is implemented, it will be called BEFORE checking the
  in-memory state. This enables lazy-loading from persistent storage.

  ## Parameters
  - `file_path` - Path of the file being read
  - `context` - Additional context (user_id, session_id, etc.)

  ## Returns
  - `{:ok, content}` - File content from storage
  - `{:error, :not_found}` - File doesn't exist in storage
  - `{:error, reason}` - Other error
  """
  @callback on_read(file_path, context) :: {:ok, content} | {:error, term()}

  @doc """
  Called when a file is deleted.

  ## Parameters
  - `file_path` - Path of the file being deleted
  - `context` - Additional context (user_id, session_id, etc.)

  ## Returns
  - `{:ok, metadata}` - Success
  - `{:error, reason}` - Failure reason
  """
  @callback on_delete(file_path, context) :: result()

  @doc """
  Called to list all files in persistent storage.

  Optional - if not implemented, only in-memory files are listed.

  ## Parameters
  - `context` - Additional context (user_id, session_id, etc.)

  ## Returns
  - `{:ok, [file_path]}` - List of file paths
  - `{:error, reason}` - Failure reason
  """
  @callback on_list(context) :: {:ok, [file_path]} | {:error, term()}

  @optional_callbacks [on_write: 3, on_read: 2, on_delete: 2, on_list: 1]
end
