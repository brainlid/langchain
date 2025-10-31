defmodule LangChain.Agents.FileSystem.FileSystemConfigTest do
  use ExUnit.Case, async: true

  alias LangChain.Agents.FileSystem.FileSystemConfig
  alias LangChain.Agents.FileSystem.Persistence.Disk

  describe "new/1" do
    test "creates config with required fields" do
      attrs = %{
        base_directory: "user_files",
        persistence_module: Disk
      }

      assert {:ok, config} = FileSystemConfig.new(attrs)
      assert config.base_directory == "user_files"
      assert config.persistence_module == Disk
      assert config.debounce_ms == 5000
      assert config.readonly == false
      assert config.storage_opts == []
    end

    test "creates config with all fields" do
      attrs = %{
        base_directory: "account_files",
        persistence_module: Disk,
        debounce_ms: 10000,
        readonly: true,
        storage_opts: [path: "/data/accounts"]
      }

      assert {:ok, config} = FileSystemConfig.new(attrs)
      assert config.base_directory == "account_files"
      assert config.persistence_module == Disk
      assert config.debounce_ms == 10000
      assert config.readonly == true
      assert config.storage_opts == [path: "/data/accounts"]
    end

    test "accepts keyword list" do
      attrs = [
        base_directory: "S3",
        persistence_module: Disk,
        readonly: true
      ]

      assert {:ok, config} = FileSystemConfig.new(attrs)
      assert config.base_directory == "S3"
      assert config.readonly == true
    end

    test "requires base_directory" do
      attrs = %{persistence_module: Disk}

      assert {:error, changeset} = FileSystemConfig.new(attrs)
      assert %{base_directory: ["can't be blank"]} = errors_on(changeset)
    end

    test "requires persistence_module" do
      attrs = %{base_directory: "user_files"}

      assert {:error, changeset} = FileSystemConfig.new(attrs)
      assert %{persistence_module: ["can't be blank"]} = errors_on(changeset)
    end

    test "rejects base_directory starting with /" do
      attrs = %{
        base_directory: "/user_files",
        persistence_module: Disk
      }

      assert {:error, changeset} = FileSystemConfig.new(attrs)
      assert %{base_directory: [_]} = errors_on(changeset)
    end

    test "rejects base_directory ending with /" do
      attrs = %{
        base_directory: "user_files/",
        persistence_module: Disk
      }

      assert {:error, changeset} = FileSystemConfig.new(attrs)
      assert %{base_directory: [_]} = errors_on(changeset)
    end

    test "rejects base_directory containing dots" do
      attrs = %{
        base_directory: "user.files",
        persistence_module: Disk
      }

      assert {:error, changeset} = FileSystemConfig.new(attrs)
      assert %{base_directory: [_]} = errors_on(changeset)
    end

    test "rejects negative debounce_ms" do
      attrs = %{
        base_directory: "user_files",
        persistence_module: Disk,
        debounce_ms: -100
      }

      assert {:error, changeset} = FileSystemConfig.new(attrs)
      assert %{debounce_ms: [_]} = errors_on(changeset)
    end

    test "accepts any module atom" do
      # Note: Runtime will check if module implements behaviour
      # Validation only checks it's a valid atom
      defmodule NotAPersistenceModule do
        def some_function(), do: :ok
      end

      attrs = %{
        base_directory: "user_files",
        persistence_module: NotAPersistenceModule
      }

      assert {:ok, config} = FileSystemConfig.new(attrs)
      assert config.persistence_module == NotAPersistenceModule
    end
  end

  describe "new!/1" do
    test "creates config on success" do
      attrs = %{
        base_directory: "user_files",
        persistence_module: Disk
      }

      assert %FileSystemConfig{} = config = FileSystemConfig.new!(attrs)
      assert config.base_directory == "user_files"
    end

    test "raises on error" do
      attrs = %{base_directory: "user_files"}

      assert_raise ArgumentError, fn ->
        FileSystemConfig.new!(attrs)
      end
    end
  end

  describe "matches_path?/2" do
    setup do
      {:ok, config} =
        FileSystemConfig.new(%{
          base_directory: "user_files",
          persistence_module: Disk
        })

      %{config: config}
    end

    test "matches paths under base directory", %{config: config} do
      assert FileSystemConfig.matches_path?(config, "/user_files/data.txt")
      assert FileSystemConfig.matches_path?(config, "/user_files/deep/nested/file.json")
    end

    test "does not match paths in other directories", %{config: config} do
      refute FileSystemConfig.matches_path?(config, "/other/file.txt")
      refute FileSystemConfig.matches_path?(config, "/user_files_extra/file.txt")
      refute FileSystemConfig.matches_path?(config, "/scratch/temp.txt")
    end

    test "does not match partial directory names", %{config: config} do
      refute FileSystemConfig.matches_path?(config, "/user_files_v2/file.txt")
    end
  end

  describe "build_storage_opts/2" do
    test "adds agent_id and base_directory to storage_opts" do
      {:ok, config} =
        FileSystemConfig.new(%{
          base_directory: "user_files",
          persistence_module: Disk,
          storage_opts: [path: "/data", custom: "value"]
        })

      opts = FileSystemConfig.build_storage_opts(config, "agent-123")

      assert Keyword.get(opts, :path) == "/data"
      assert Keyword.get(opts, :custom) == "value"
      assert Keyword.get(opts, :agent_id) == "agent-123"
      assert Keyword.get(opts, :base_directory) == "user_files"
    end

    test "works with empty storage_opts" do
      {:ok, config} =
        FileSystemConfig.new(%{
          base_directory: "user_files",
          persistence_module: Disk
        })

      opts = FileSystemConfig.build_storage_opts(config, "agent-456")

      assert Keyword.get(opts, :agent_id) == "agent-456"
      assert Keyword.get(opts, :base_directory) == "user_files"
    end
  end

  # Helper to extract errors from changeset
  defp errors_on(changeset) do
    Ecto.Changeset.traverse_errors(changeset, fn {msg, opts} ->
      Enum.reduce(opts, msg, fn {key, value}, acc ->
        String.replace(acc, "%{#{key}}", to_string(value))
      end)
    end)
  end
end
