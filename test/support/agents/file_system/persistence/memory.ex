defmodule LangChain.Agents.FileSystem.Persistence.Memory do
  @moduledoc """
  Test-only in-memory persistence backend.

  This module provides a no-op implementation of the Persistence behaviour
  that doesn't actually persist anything. It's located in `test/support/`
  and is only available during tests, not in production code.

  ## Purpose

  - Allows testing filesystem functionality without actual disk I/O
  - Provides minimal implementation that satisfies the Persistence behaviour
  - No data is actually stored or retrieved

  ## Usage

      config = FileSystemConfig.new(%{
        scope_key: {:test, 123},
        base_directory: "TestDir",
        persistence_module: LangChain.Agents.FileSystem.Persistence.Memory,
        storage_opts: []
      })
  """

  @behaviour LangChain.Agents.FileSystem.Persistence

  @impl true
  def write_to_storage(_entry, _opts), do: {:ok, nil}

  @impl true
  def load_from_storage(_entry, _opts), do: {:error, :not_implemented}

  @impl true
  def delete_from_storage(_entry, _opts), do: {:ok, nil}

  @impl true
  def list_persisted_files(_agent_id, _opts), do: {:ok, []}
end
