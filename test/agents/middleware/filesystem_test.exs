defmodule LangChain.Agents.Middleware.FileSystemTest do
  use ExUnit.Case, async: false

  alias LangChain.Agents.Middleware.FileSystem
  alias LangChain.Agents.FileSystemServer
  alias LangChain.Agents.State

  setup_all do
    # Start registry once for all tests
    {:ok, _registry} =
      start_supervised({Registry, keys: :unique, name: LangChain.Agents.Registry})

    :ok
  end

  setup do
    # Generate unique agent ID for each test
    agent_id = "test_agent_#{System.unique_integer([:positive])}"
    {:ok, _pid} = start_supervised({FileSystemServer, agent_id: agent_id})

    %{agent_id: agent_id}
  end

  describe "init/1" do
    test "initializes with agent_id", %{agent_id: agent_id} do
      assert {:ok, config} = FileSystem.init(agent_id: agent_id)
      assert config.agent_id == agent_id
      assert config.enabled_tools == ["ls", "read_file", "write_file", "edit_file"]
      assert config.custom_tool_descriptions == %{}
    end

    test "initializes with custom enabled_tools", %{agent_id: agent_id} do
      assert {:ok, config} =
               FileSystem.init(agent_id: agent_id, enabled_tools: ["ls", "read_file"])

      assert config.enabled_tools == ["ls", "read_file"]
    end

    test "initializes with custom tool descriptions", %{agent_id: agent_id} do
      custom = %{"ls" => "Custom ls description"}
      assert {:ok, config} = FileSystem.init(agent_id: agent_id, custom_tool_descriptions: custom)
      assert config.custom_tool_descriptions == custom
    end

    test "requires agent_id" do
      assert_raise KeyError, fn ->
        FileSystem.init([])
      end
    end
  end

  describe "system_prompt/1" do
    test "returns filesystem tools prompt" do
      config = %{agent_id: "test"}
      prompt = FileSystem.system_prompt(config)

      assert prompt =~ "Filesystem Tools"
      assert prompt =~ "ls"
      assert prompt =~ "read_file"
      assert prompt =~ "write_file"
      assert prompt =~ "edit_file"
      assert prompt =~ "must start with a forward slash"
    end
  end

  describe "tools/1" do
    test "returns four filesystem tools by default", %{agent_id: agent_id} do
      tools =
        FileSystem.tools(%{
          agent_id: agent_id,
          enabled_tools: ["ls", "read_file", "write_file", "edit_file"]
        })

      assert length(tools) == 4
      tool_names = Enum.map(tools, & &1.name)
      assert "ls" in tool_names
      assert "read_file" in tool_names
      assert "write_file" in tool_names
      assert "edit_file" in tool_names
    end

    test "returns only enabled tools", %{agent_id: agent_id} do
      tools = FileSystem.tools(%{agent_id: agent_id, enabled_tools: ["ls", "read_file"]})

      assert length(tools) == 2
      tool_names = Enum.map(tools, & &1.name)
      assert "ls" in tool_names
      assert "read_file" in tool_names
      refute "write_file" in tool_names
      refute "edit_file" in tool_names
    end
  end

  describe "ls tool" do
    test "lists files in filesystem", %{agent_id: agent_id} do
      # Write some files
      FileSystemServer.write_file(agent_id, "/file1.txt", "content1")
      FileSystemServer.write_file(agent_id, "/file2.txt", "content2")

      [ls_tool | _] =
        FileSystem.tools(%{
          agent_id: agent_id,
          enabled_tools: ["ls", "read_file", "write_file", "edit_file"]
        })

      assert {:ok, result} = ls_tool.function.(%{}, %{state: State.new!()})
      assert result =~ "/file1.txt"
      assert result =~ "/file2.txt"
    end

    test "reports empty filesystem", %{agent_id: agent_id} do
      [ls_tool | _] =
        FileSystem.tools(%{
          agent_id: agent_id,
          enabled_tools: ["ls", "read_file", "write_file", "edit_file"]
        })

      assert {:ok, result} = ls_tool.function.(%{}, %{state: State.new!()})
      assert result =~ "No files"
    end

    test "filters by pattern", %{agent_id: agent_id} do
      FileSystemServer.write_file(agent_id, "/test.txt", "content")
      FileSystemServer.write_file(agent_id, "/test.md", "content")
      FileSystemServer.write_file(agent_id, "/other.txt", "content")

      [ls_tool | _] =
        FileSystem.tools(%{
          agent_id: agent_id,
          enabled_tools: ["ls", "read_file", "write_file", "edit_file"]
        })

      assert {:ok, result} = ls_tool.function.(%{"pattern" => "*test*"}, %{state: State.new!()})
      assert result =~ "/test.txt"
      assert result =~ "/test.md"
      refute result =~ "/other.txt"
    end
  end

  describe "read_file tool" do
    setup %{agent_id: agent_id} do
      content = """
      line 1
      line 2
      line 3
      line 4
      line 5
      """

      FileSystemServer.write_file(agent_id, "/test.txt", String.trim(content))

      [_, read_file_tool | _] =
        FileSystem.tools(%{
          agent_id: agent_id,
          enabled_tools: ["ls", "read_file", "write_file", "edit_file"]
        })

      %{tool: read_file_tool}
    end

    test "reads entire file with line numbers", %{tool: tool} do
      args = %{"file_path" => "/test.txt"}

      assert {:ok, result} = tool.function.(args, %{state: State.new!()})
      assert result =~ "1\tline 1"
      assert result =~ "2\tline 2"
      assert result =~ "5\tline 5"
    end

    test "reads file with offset", %{tool: tool} do
      args = %{"file_path" => "/test.txt", "offset" => 2}

      assert {:ok, result} = tool.function.(args, %{state: State.new!()})
      assert result =~ "3\tline 3"
      assert result =~ "4\tline 4"
      refute result =~ "1\tline 1"
    end

    test "reads file with limit", %{tool: tool} do
      args = %{"file_path" => "/test.txt", "limit" => 2}

      assert {:ok, result} = tool.function.(args, %{state: State.new!()})
      assert result =~ "1\tline 1"
      assert result =~ "2\tline 2"
      refute result =~ "3\tline 3"
    end

    test "reads file with offset and limit", %{tool: tool} do
      args = %{"file_path" => "/test.txt", "offset" => 1, "limit" => 2}

      assert {:ok, result} = tool.function.(args, %{state: State.new!()})
      assert result =~ "2\tline 2"
      assert result =~ "3\tline 3"
      refute result =~ "1\tline 1"
      refute result =~ "4\tline 4"
    end

    test "returns error for non-existent file", %{tool: tool} do
      args = %{"file_path" => "/missing.txt"}

      assert {:error, message} = tool.function.(args, %{state: State.new!()})
      assert message =~ "not found"
    end

    test "rejects paths without leading slash", %{tool: tool} do
      args = %{"file_path" => "no-slash.txt"}

      assert {:error, message} = tool.function.(args, %{state: State.new!()})
      assert message =~ "must start with"
    end
  end

  describe "write_file tool" do
    setup %{agent_id: agent_id} do
      [_, _, write_file_tool | _] =
        FileSystem.tools(%{
          agent_id: agent_id,
          enabled_tools: ["ls", "read_file", "write_file", "edit_file"]
        })

      %{tool: write_file_tool}
    end

    test "creates new file", %{agent_id: agent_id, tool: tool} do
      args = %{"file_path" => "/new.txt", "content" => "Hello, World!"}

      assert {:ok, message} = tool.function.(args, %{state: State.new!()})
      assert message =~ "created successfully"

      # Verify file was created in FileSystemServer
      {:ok, content} = FileSystemServer.read_file(agent_id, "/new.txt")
      assert content == "Hello, World!"
    end

    test "rejects overwriting existing file", %{agent_id: agent_id, tool: tool} do
      # Create a file first
      FileSystemServer.write_file(agent_id, "/existing.txt", "original")

      args = %{"file_path" => "/existing.txt", "content" => "new content"}

      assert {:error, message} = tool.function.(args, %{state: State.new!()})
      assert message =~ "already exists"
    end

    test "rejects paths without leading slash", %{tool: tool} do
      args = %{"file_path" => "no-slash.txt", "content" => "content"}

      assert {:error, message} = tool.function.(args, %{state: State.new!()})
      assert message =~ "must start with"
    end

    test "rejects path traversal attempts", %{tool: tool} do
      args = %{"file_path" => "/../etc/passwd", "content" => "bad"}

      assert {:error, message} = tool.function.(args, %{state: State.new!()})
      assert message =~ "not allowed"
    end
  end

  describe "edit_file tool" do
    setup %{agent_id: agent_id} do
      FileSystemServer.write_file(agent_id, "/edit.txt", "Hello World")

      [_, _, _, edit_file_tool] =
        FileSystem.tools(%{
          agent_id: agent_id,
          enabled_tools: ["ls", "read_file", "write_file", "edit_file"]
        })

      %{tool: edit_file_tool}
    end

    test "edits file with single occurrence", %{agent_id: agent_id, tool: tool} do
      args = %{
        "file_path" => "/edit.txt",
        "old_string" => "World",
        "new_string" => "Elixir"
      }

      assert {:ok, message} = tool.function.(args, %{state: State.new!()})
      assert message =~ "edited successfully"

      # Verify content changed
      {:ok, content} = FileSystemServer.read_file(agent_id, "/edit.txt")
      assert content == "Hello Elixir"
    end

    test "errors on non-existent string", %{tool: tool} do
      args = %{
        "file_path" => "/edit.txt",
        "old_string" => "NotFound",
        "new_string" => "Something"
      }

      assert {:error, message} = tool.function.(args, %{state: State.new!()})
      assert message =~ "not found"
    end

    test "errors on multiple occurrences without replace_all", %{agent_id: agent_id, tool: tool} do
      FileSystemServer.write_file(agent_id, "/multi.txt", "test test test")

      args = %{
        "file_path" => "/multi.txt",
        "old_string" => "test",
        "new_string" => "foo"
      }

      assert {:error, message} = tool.function.(args, %{state: State.new!()})
      assert message =~ "appears"
      assert message =~ "times"
    end

    test "replaces all occurrences with replace_all: true", %{agent_id: agent_id, tool: tool} do
      FileSystemServer.write_file(agent_id, "/multi.txt", "test test test")

      args = %{
        "file_path" => "/multi.txt",
        "old_string" => "test",
        "new_string" => "foo",
        "replace_all" => true
      }

      assert {:ok, message} = tool.function.(args, %{state: State.new!()})
      assert message =~ "edited successfully"
      assert message =~ "3 replacements"

      {:ok, content} = FileSystemServer.read_file(agent_id, "/multi.txt")
      assert content == "foo foo foo"
    end

    test "returns error for non-existent file", %{tool: tool} do
      args = %{
        "file_path" => "/missing.txt",
        "old_string" => "old",
        "new_string" => "new"
      }

      assert {:error, message} = tool.function.(args, %{state: State.new!()})
      assert message =~ "not found"
    end
  end

  describe "path validation" do
    test "validate_path/1 accepts valid paths" do
      assert {:ok, "/file.txt"} = FileSystem.validate_path("/file.txt")
      assert {:ok, "/dir/file.txt"} = FileSystem.validate_path("/dir/file.txt")
      assert {:ok, "/path/to/file.txt"} = FileSystem.validate_path("/path/to/file.txt")
    end

    test "validate_path/1 rejects paths without leading slash" do
      assert {:error, message} = FileSystem.validate_path("file.txt")
      assert message =~ "must start with"
    end

    test "validate_path/1 rejects path traversal" do
      assert {:error, message} = FileSystem.validate_path("/dir/../etc/passwd")
      assert message =~ "not allowed"
    end

    test "validate_path/1 rejects home directory paths" do
      assert {:error, message} = FileSystem.validate_path("~/file.txt")
      # This path fails the "must start with /" check first
      assert message =~ "must start with"
    end

    test "validate_path/1 rejects empty paths" do
      assert {:error, message} = FileSystem.validate_path("")
      assert message =~ "empty"
    end

    test "normalize_path/1 normalizes slashes" do
      assert "/path/to/file.txt" = FileSystem.normalize_path("/path//to///file.txt")
      assert "/path/to/file.txt" = FileSystem.normalize_path("/path\\to\\file.txt")
    end
  end
end
