defmodule LangChain.Agents.Middleware.FileSystemTest do
  use ExUnit.Case, async: false

  alias LangChain.Agents.Middleware.FileSystem
  alias LangChain.Agents.FileSystemServer
  alias LangChain.Agents.State

  setup_all do
    # Note: Registry is started globally in test_helper.exs
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

      assert config.enabled_tools == [
               "ls",
               "read_file",
               "write_file",
               "edit_file",
               "search_text",
               "delete_file"
             ]

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

  defp get_search_text_tool(tools) when is_list(tools) do
    Enum.find(tools, fn tool -> tool.name == "search_text" end)
  end

  describe "search_text tool - single file search" do
    test "search_text tool is enabled in FileSystem", %{agent_id: agent_id} do
      # Initialize middleware
      {:ok, config} = FileSystem.init(agent_id: agent_id)

      # Find the search_text tool
      search_tool = config |> FileSystem.tools() |> get_search_text_tool()
      assert search_tool != nil
    end

    test "finds matches in a single file with line numbers", %{agent_id: agent_id} do
      # Setup: create a file with searchable content
      FileSystemServer.write_file(agent_id, "/test.txt", """
      Hello World
      This is a test
      TODO: Add feature
      Another line
      TODO: Fix bug
      End of file
      """)

      # Initialize middleware
      {:ok, config} = FileSystem.init(agent_id: agent_id)
      search_tool = config |> FileSystem.tools() |> get_search_text_tool()

      # Execute search
      args = %{"pattern" => "TODO", "file_path" => "/test.txt"}
      {:ok, result} = search_tool.function.(args, %{})

      IO.inspect result

      # Verify results
      assert result =~ "File: /test.txt"
      assert result =~ "Line 3:"
      assert result =~ "TODO: Add feature"
      assert result =~ "Line 5:"
      assert result =~ "TODO: Fix bug"
    end

    test "returns no matches when pattern not found", %{agent_id: agent_id} do
      FileSystemServer.write_file(agent_id, "/test.txt", """
      Hello World
      This is a test
      """)

      {:ok, config} = FileSystem.init(agent_id: agent_id)
      search_tool = config |> FileSystem.tools() |> get_search_text_tool()

      args = %{"pattern" => "NOTFOUND"}
      {:ok, result} = search_tool.function.(args, %{})

      assert result == "No matches found"
    end

    test "handles case-insensitive search", %{agent_id: agent_id} do
      FileSystemServer.write_file(agent_id, "/test.txt", """
      Hello World
      hello world
      HELLO WORLD
      """)

      {:ok, config} = FileSystem.init(agent_id: agent_id)
      search_tool = config |> FileSystem.tools() |> get_search_text_tool()

      args = %{"pattern" => "hello", "file_path" => "/test.txt", "case_sensitive" => false}
      {:ok, result} = search_tool.function.(args, %{})

      # Should match all three lines
      assert result =~ "Line 1:"
      assert result =~ "Line 2:"
      assert result =~ "Line 3:"
    end

    test "handles case-sensitive search", %{agent_id: agent_id} do
      FileSystemServer.write_file(agent_id, "/test.txt", """
      Hello World
      hello world
      HELLO WORLD
      """)

      {:ok, config} = FileSystem.init(agent_id: agent_id)
      search_tool = config |> FileSystem.tools() |> get_search_text_tool()

      args = %{"pattern" => "hello", "file_path" => "/test.txt", "case_sensitive" => true}
      {:ok, result} = search_tool.function.(args, %{})

      # Should only match line 2
      refute result =~ "Line 1:"
      assert result =~ "Line 2:"
      refute result =~ "Line 3:"
    end

    test "supports regex patterns", %{agent_id: agent_id} do
      FileSystemServer.write_file(agent_id, "/test.txt", """
      error123
      warning456
      error789
      info
      """)

      {:ok, config} = FileSystem.init(agent_id: agent_id)
      search_tool = config |> FileSystem.tools() |> get_search_text_tool()

      # Search for lines starting with "error" followed by digits
      args = %{"pattern" => "error\\d+", "file_path" => "/test.txt"}
      {:ok, result} = search_tool.function.(args, %{})

      assert result =~ "Line 1:"
      assert result =~ "error123"
      assert result =~ "Line 3:"
      assert result =~ "error789"
      refute result =~ "Line 2:"
      refute result =~ "warning"
    end

    test "returns error for invalid regex pattern", %{agent_id: agent_id} do
      FileSystemServer.write_file(agent_id, "/test.txt", "content")

      {:ok, config} = FileSystem.init(agent_id: agent_id)
      search_tool = config |> FileSystem.tools() |> get_search_text_tool()

      args = %{"pattern" => "[invalid(regex", "file_path" => "/test.txt"}
      {:error, error_msg} = search_tool.function.(args, %{})

      assert error_msg =~ "Invalid regex pattern"
    end

    test "returns error for non-existent file", %{agent_id: agent_id} do
      {:ok, config} = FileSystem.init(agent_id: agent_id)
      search_tool = config |> FileSystem.tools() |> get_search_text_tool()

      args = %{"pattern" => "test", "file_path" => "/nonexistent.txt"}
      {:error, error_msg} = search_tool.function.(args, %{})

      assert error_msg =~ "File not found"
    end
  end

  describe "search_text tool - context lines" do
    test "includes context lines before and after matches", %{agent_id: agent_id} do
      FileSystemServer.write_file(agent_id, "/test.txt", """
      Line 1: Before context 1
      Line 2: Before context 2
      Line 3: MATCH HERE
      Line 4: After context 1
      Line 5: After context 2
      """)

      {:ok, config} = FileSystem.init(agent_id: agent_id)
      search_tool = config |> FileSystem.tools() |> get_search_text_tool()

      args = %{"pattern" => "MATCH", "file_path" => "/test.txt", "context_lines" => 2}
      {:ok, result} = search_tool.function.(args, %{})

      # Verify main match
      assert result =~ "Line 3: MATCH HERE"

      # Verify context lines (marked with |)
      assert result =~ "| Line 1: Before context 1"
      assert result =~ "| Line 2: Before context 2"
      assert result =~ "| Line 4: After context 1"
      assert result =~ "| Line 5: After context 2"
    end

    test "handles context at file boundaries", %{agent_id: agent_id} do
      FileSystemServer.write_file(agent_id, "/test.txt", """
      Line 1: MATCH at start
      Line 2: After
      """)

      {:ok, config} = FileSystem.init(agent_id: agent_id)
      search_tool = config |> FileSystem.tools() |> get_search_text_tool()

      # Request 2 context lines, but match is at line 1 (no lines before)
      args = %{"pattern" => "MATCH", "file_path" => "/test.txt", "context_lines" => 2}
      {:ok, result} = search_tool.function.(args, %{})

      assert result =~ "Line 1: MATCH at start"
      assert result =~ "| Line 2: After"
    end
  end

  describe "search_text tool - multi-file search" do
    test "searches across all files when no file_path specified", %{agent_id: agent_id} do
      # Create multiple files
      FileSystemServer.write_file(agent_id, "/file1.txt", """
      Line 1: TODO in file 1
      Line 2: Normal line
      """)

      FileSystemServer.write_file(agent_id, "/file2.txt", """
      Line 1: Normal line
      Line 2: TODO in file 2
      """)

      FileSystemServer.write_file(agent_id, "/file3.txt", """
      Line 1: No match here
      """)

      {:ok, config} = FileSystem.init(agent_id: agent_id)
      search_tool = config |> FileSystem.tools() |> get_search_text_tool()

      # Search all files without specifying file_path
      args = %{"pattern" => "TODO"}
      {:ok, result} = search_tool.function.(args, %{})

      # Should find matches in both file1 and file2
      assert result =~ "File: /file1.txt"
      assert result =~ "TODO in file 1"
      assert result =~ "File: /file2.txt"
      assert result =~ "TODO in file 2"
      refute result =~ "file3.txt"
    end

    test "returns no matches when pattern not in any file", %{agent_id: agent_id} do
      FileSystemServer.write_file(agent_id, "/file1.txt", "content 1")
      FileSystemServer.write_file(agent_id, "/file2.txt", "content 2")

      {:ok, config} = FileSystem.init(agent_id: agent_id)
      search_tool = config |> FileSystem.tools() |> get_search_text_tool()

      args = %{"pattern" => "NOTFOUND"}
      {:ok, result} = search_tool.function.(args, %{})

      assert result == "No matches found"
    end
  end

  describe "search_text tool - max_results limiting" do
    test "limits results to max_results parameter", %{agent_id: agent_id} do
      # Create a file with many matches
      content =
        Enum.map(1..100, fn i -> "Line #{i}: MATCH #{i}" end)
        |> Enum.join("\n")

      FileSystemServer.write_file(agent_id, "/test.txt", content)

      {:ok, config} = FileSystem.init(agent_id: agent_id)
      search_tool = config |> FileSystem.tools() |> get_search_text_tool()

      # Limit to 10 results
      args = %{"pattern" => "MATCH", "file_path" => "/test.txt", "max_results" => 10}
      {:ok, result} = search_tool.function.(args, %{})

      # Count how many "Line X:" appear in results
      line_count =
        result
        |> String.split("\n")
        |> Enum.count(fn line -> String.contains?(line, "Line ") end)

      # Should have at most 10 matches
      assert line_count <= 10
    end

    test "shows truncation notice when results exceed max_results", %{agent_id: agent_id} do
      # Create a file with many matches
      content =
        Enum.map(1..60, fn i -> "Line #{i}: MATCH #{i}" end)
        |> Enum.join("\n")

      FileSystemServer.write_file(agent_id, "/test.txt", content)

      {:ok, config} = FileSystem.init(agent_id: agent_id)
      search_tool = config |> FileSystem.tools() |> get_search_text_tool()

      # Use default max_results (50)
      args = %{"pattern" => "MATCH", "file_path" => "/test.txt"}
      {:ok, result} = search_tool.function.(args, %{})

      # Should show truncation notice
      assert result =~ "Results truncated"
    end
  end

  describe "search_text tool - parameter validation" do
    test "returns error when pattern is missing", %{agent_id: agent_id} do
      {:ok, config} = FileSystem.init(agent_id: agent_id)
      search_tool = config |> FileSystem.tools() |> get_search_text_tool()

      args = %{}
      {:error, error_msg} = search_tool.function.(args, %{})

      assert error_msg =~ "pattern is required"
    end

    test "handles invalid path", %{agent_id: agent_id} do
      {:ok, config} = FileSystem.init(agent_id: agent_id)
      search_tool = config |> FileSystem.tools() |> get_search_text_tool()

      # Path without leading slash
      args = %{"pattern" => "test", "file_path" => "invalid/path.txt"}
      {:error, error_msg} = search_tool.function.(args, %{})

      assert error_msg =~ "Path must start with '/'"
    end
  end

  describe "search_text tool - tool registration" do
    test "search_text tool is included in default enabled tools", %{agent_id: agent_id} do
      {:ok, config} = FileSystem.init(agent_id: agent_id)

      assert "search_text" in config.enabled_tools
    end

    test "search_text tool can be disabled via config", %{agent_id: agent_id} do
      {:ok, config} =
        FileSystem.init(
          agent_id: agent_id,
          enabled_tools: ["ls", "read_file", "write_file"]
        )

      search_tool = config |> FileSystem.tools() |> get_search_text_tool()

      assert search_tool == nil
    end

    test "search_text tool has correct schema", %{agent_id: agent_id} do
      {:ok, config} = FileSystem.init(agent_id: agent_id)
      search_tool = config |> FileSystem.tools() |> get_search_text_tool()

      assert search_tool.name == "search_text"
      assert is_binary(search_tool.description)

      # Verify parameters schema
      schema = search_tool.parameters_schema
      assert schema["type"] == "object"
      assert "pattern" in schema["required"]

      properties = schema["properties"]
      assert Map.has_key?(properties, "pattern")
      assert Map.has_key?(properties, "file_path")
      assert Map.has_key?(properties, "case_sensitive")
      assert Map.has_key?(properties, "context_lines")
      assert Map.has_key?(properties, "max_results")
    end
  end

  describe "search_text tool - integration scenarios" do
    test "search then read workflow", %{agent_id: agent_id} do
      # Create a file with searchable content
      FileSystemServer.write_file(agent_id, "/project.md", """
      # Project Documentation

      ## Section 1
      Some content here.

      ## Section 2
      TODO: Complete this section

      ## Section 3
      More content.

      ## Section 4
      TODO: Add examples
      """)

      {:ok, config} = FileSystem.init(agent_id: agent_id)
      tools = FileSystem.tools(config)

      # First, search for TODOs
      search_tool = Enum.find(tools, fn tool -> tool.name == "search_text" end)
      args = %{"pattern" => "TODO", "file_path" => "/project.md"}
      {:ok, search_result} = search_tool.function.(args, %{})

      # Verify we found the TODOs
      assert search_result =~ "Line 7:"
      assert search_result =~ "TODO: Complete this section"
      assert search_result =~ "Line 13:"
      assert search_result =~ "TODO: Add examples"

      # Now read the file to get more context
      read_tool = Enum.find(tools, fn tool -> tool.name == "read_file" end)
      read_args = %{"file_path" => "/project.md", "offset" => 6, "limit" => 3}
      {:ok, read_result} = read_tool.function.(read_args, %{})

      # Should see the TODO line in context
      assert read_result =~ "TODO: Complete this section"
    end

    test "search across multiple files with different patterns", %{agent_id: agent_id} do
      FileSystemServer.write_file(agent_id, "/config.json", """
      {
        "error_log": true,
        "debug": false
      }
      """)

      FileSystemServer.write_file(agent_id, "/app.log", """
      2024-01-01 10:00:00 INFO: Application started
      2024-01-01 10:05:00 ERROR: Connection failed
      2024-01-01 10:10:00 INFO: Retrying...
      2024-01-01 10:15:00 ERROR: Timeout occurred
      """)

      {:ok, config} = FileSystem.init(agent_id: agent_id)
      search_tool = config |> FileSystem.tools() |> get_search_text_tool()

      # Search for "error" (case-insensitive) across all files
      args = %{"pattern" => "error", "case_sensitive" => false}
      {:ok, result} = search_tool.function.(args, %{})

      # Should find matches in both files
      assert result =~ "/config.json"
      assert result =~ "error_log"
      assert result =~ "/app.log"
      assert result =~ "ERROR: Connection failed"
      assert result =~ "ERROR: Timeout occurred"
    end
  end
end
