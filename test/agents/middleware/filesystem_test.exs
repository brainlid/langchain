defmodule LangChain.Agents.Middleware.FileSystemTest do
  use ExUnit.Case, async: true

  alias LangChain.Agents.Middleware.FileSystem
  alias LangChain.Agents.State

  describe "init/1" do
    test "initializes with default config" do
      assert {:ok, config} = FileSystem.init([])
      assert config.long_term_memory == false
      assert config.memories_prefix == "memories"
    end

    test "initializes with long_term_memory option" do
      assert {:ok, config} = FileSystem.init(long_term_memory: true)
      assert config.long_term_memory == true
    end

    test "initializes with custom memories_prefix" do
      assert {:ok, config} = FileSystem.init(memories_prefix: ".my_memories")
      assert config.memories_prefix == ".my_memories"
    end
  end

  describe "system_prompt/1" do
    test "returns base system prompt by default" do
      config = %{long_term_memory: false}
      prompt = FileSystem.system_prompt(config)

      assert prompt =~ "Filesystem Tools"
      assert prompt =~ "ls"
      assert prompt =~ "read_file"
      assert prompt =~ "write_file"
      assert prompt =~ "edit_file"
    end

    test "includes long-term memory supplement when enabled" do
      config = %{long_term_memory: true}
      prompt = FileSystem.system_prompt(config)

      assert prompt =~ "persist across conversations"
    end
  end

  describe "tools/1" do
    test "returns four filesystem tools" do
      tools = FileSystem.tools(%{long_term_memory: false})

      assert length(tools) == 4
      tool_names = Enum.map(tools, & &1.name)
      assert "ls" in tool_names
      assert "read_file" in tool_names
      assert "write_file" in tool_names
      assert "edit_file" in tool_names
    end

    test "all tools have required fields" do
      tools = FileSystem.tools(%{long_term_memory: false})

      for tool <- tools do
        assert is_binary(tool.name)
        assert is_binary(tool.description)
        assert is_map(tool.parameters_schema)
        assert is_function(tool.function, 2)
      end
    end

    test "ls tool description includes memories_prefix when long_term_memory is enabled" do
      config = %{long_term_memory: true, memories_prefix: "memories"}
      [ls_tool | _] = FileSystem.tools(config)

      assert ls_tool.name == "ls"
      assert ls_tool.description =~ "memories"
      assert ls_tool.description =~ "longterm filesystem"
    end

    test "ls tool description does not include memories_prefix when long_term_memory is disabled" do
      config = %{long_term_memory: false}
      [ls_tool | _] = FileSystem.tools(config)

      assert ls_tool.name == "ls"
      refute ls_tool.description =~ "longterm filesystem"
    end
  end

  describe "ls tool" do
    test "lists files when filesystem has files" do
      state = State.new!(%{files: %{"file1.txt" => "content1", "file2.txt" => "content2"}})
      [ls_tool | _] = FileSystem.tools(%{})

      assert {:ok, result} = ls_tool.function.(%{}, %{state: state})
      assert result =~ "file1.txt"
      assert result =~ "file2.txt"
    end

    test "reports empty filesystem" do
      state = State.new!(%{files: %{}})
      [ls_tool | _] = FileSystem.tools(%{})

      assert {:ok, result} = ls_tool.function.(%{}, %{state: state})
      assert result =~ "No files"
    end
  end

  describe "read_file tool" do
    setup do
      content = """
      line 1
      line 2
      line 3
      line 4
      line 5
      """

      # Use State.put_file to create FileData structure
      initial_state = State.new!(%{files: %{}})
      state = State.put_file(initial_state, "test.txt", String.trim(content))
      [_, read_file_tool | _] = FileSystem.tools(%{})

      %{state: state, tool: read_file_tool}
    end

    test "reads entire file with line numbers", %{state: state, tool: tool} do
      args = %{"file_path" => "test.txt"}

      assert {:ok, result} = tool.function.(args, %{state: state})
      # New format uses fixed-width line numbers (padded with spaces) with tab separator
      assert result =~ "1\tline 1"
      assert result =~ "2\tline 2"
      assert result =~ "5\tline 5"
    end

    test "reads file with offset", %{state: state, tool: tool} do
      args = %{"file_path" => "test.txt", "offset" => 2}

      assert {:ok, result} = tool.function.(args, %{state: state})
      # New format uses fixed-width line numbers (padded with spaces) with tab separator
      assert result =~ "3\tline 3"
      assert result =~ "4\tline 4"
      refute result =~ "1\tline 1"
    end

    test "reads file with limit", %{state: state, tool: tool} do
      args = %{"file_path" => "test.txt", "limit" => 2}

      assert {:ok, result} = tool.function.(args, %{state: state})
      # New format uses fixed-width line numbers (padded with spaces) with tab separator
      assert result =~ "1\tline 1"
      assert result =~ "2\tline 2"
      refute result =~ "3\tline 3"
    end

    test "reads file with offset and limit", %{state: state, tool: tool} do
      args = %{"file_path" => "test.txt", "offset" => 1, "limit" => 2}

      assert {:ok, result} = tool.function.(args, %{state: state})
      # New format uses fixed-width line numbers (padded with spaces) with tab separator
      assert result =~ "2\tline 2"
      assert result =~ "3\tline 3"
      refute result =~ "1\tline 1"
      refute result =~ "4\tline 4"
    end

    test "returns error for non-existent file", %{state: state, tool: tool} do
      args = %{"file_path" => "missing.txt"}

      assert {:error, message} = tool.function.(args, %{state: state})
      assert message =~ "not found"
    end
  end

  describe "write_file tool" do
    setup do
      state = State.new!(%{files: %{}})
      [_, _, write_file_tool | _] = FileSystem.tools(%{})

      %{state: state, tool: write_file_tool}
    end

    test "creates new file", %{state: state, tool: tool} do
      args = %{"file_path" => "new.txt", "content" => "Hello, World!"}

      assert {:ok, message, new_state} = tool.function.(args, %{state: state})
      assert message =~ "created successfully"
      # Files now use FileData structure, use State.get_file to extract content
      assert State.get_file(new_state, "new.txt") == "Hello, World!"
    end

    test "rejects overwriting existing file (overwrite protection)", %{tool: tool} do
      # Create initial state with FileData structure
      initial_state = State.new!(%{files: %{}})
      state_with_file = State.put_file(initial_state, "existing.txt", "old content")

      args = %{"file_path" => "existing.txt", "content" => "new content"}

      # Should return error, not overwrite
      assert {:error, message} = tool.function.(args, %{state: state_with_file})
      assert message =~ "already exists"
      assert message =~ "edit_file"
    end

    test "returns error when file_path is missing", %{state: state, tool: tool} do
      args = %{"content" => "content"}

      assert {:error, message} = tool.function.(args, %{state: state})
      assert message =~ "required"
    end

    test "returns error when content is missing", %{state: state, tool: tool} do
      args = %{"file_path" => "test.txt"}

      assert {:error, message} = tool.function.(args, %{state: state})
      assert message =~ "required"
    end
  end

  describe "edit_file tool" do
    setup do
      content = """
      def hello do
        IO.puts("Hello")
      end
      """

      # Use State.put_file to create FileData structure
      initial_state = State.new!(%{files: %{}})
      state = State.put_file(initial_state, "test.ex", String.trim(content))
      [_, _, _, edit_file_tool] = FileSystem.tools(%{})

      %{state: state, tool: edit_file_tool}
    end

    test "edits file with single occurrence", %{state: state, tool: tool} do
      args = %{
        "file_path" => "test.ex",
        "old_string" => "Hello",
        "new_string" => "Goodbye"
      }

      assert {:ok, message, new_state} = tool.function.(args, %{state: state})
      assert message =~ "edited successfully"
      # Use State.get_file to extract content from FileData
      content = State.get_file(new_state, "test.ex")
      assert content =~ "Goodbye"
      refute content =~ "Hello"
    end

    test "returns error when old_string not found", %{state: state, tool: tool} do
      args = %{
        "file_path" => "test.ex",
        "old_string" => "NonExistent",
        "new_string" => "Something"
      }

      assert {:error, message} = tool.function.(args, %{state: state})
      assert message =~ "not found"
    end

    test "returns error for multiple occurrences without replace_all", %{tool: tool} do
      content = "hello hello hello"
      initial_state = State.new!(%{files: %{}})
      state = State.put_file(initial_state, "multi.txt", content)

      args = %{
        "file_path" => "multi.txt",
        "old_string" => "hello",
        "new_string" => "goodbye"
      }

      assert {:error, message} = tool.function.(args, %{state: state})
      assert message =~ "appears 3 times"
      assert message =~ "replace_all"
    end

    test "replaces all occurrences with replace_all true", %{tool: tool} do
      content = "hello hello hello"
      initial_state = State.new!(%{files: %{}})
      state = State.put_file(initial_state, "multi.txt", content)

      args = %{
        "file_path" => "multi.txt",
        "old_string" => "hello",
        "new_string" => "goodbye",
        "replace_all" => true
      }

      assert {:ok, message, new_state} = tool.function.(args, %{state: state})
      assert message =~ "3 replacements"
      # Use State.get_file to extract content from FileData
      assert State.get_file(new_state, "multi.txt") == "goodbye goodbye goodbye"
    end

    test "returns error for non-existent file", %{state: state, tool: tool} do
      args = %{
        "file_path" => "missing.ex",
        "old_string" => "old",
        "new_string" => "new"
      }

      assert {:error, message} = tool.function.(args, %{state: state})
      assert message =~ "not found"
    end

    test "returns error when required args are missing", %{state: state, tool: tool} do
      args = %{"file_path" => "test.ex"}

      assert {:error, message} = tool.function.(args, %{state: state})
      assert message =~ "required"
    end
  end

  describe "integration scenarios" do
    test "complete workflow: write, read, edit" do
      [ls_tool, read_tool, write_tool, edit_tool] = FileSystem.tools(%{})

      # Start with empty filesystem
      state = State.new!(%{files: %{}})

      # 1. List files (should be empty)
      {:ok, ls_result} = ls_tool.function.(%{}, %{state: state})
      assert ls_result =~ "No files"

      # 2. Write a file
      {:ok, _, state} =
        write_tool.function.(
          %{"file_path" => "code.ex", "content" => "def test, do: :ok"},
          %{state: state}
        )

      # 3. Read the file
      {:ok, read_result} = read_tool.function.(%{"file_path" => "code.ex"}, %{state: state})
      assert read_result =~ "def test"

      # 4. Edit the file
      {:ok, _, state} =
        edit_tool.function.(
          %{
            "file_path" => "code.ex",
            "old_string" => ":ok",
            "new_string" => ":edited"
          },
          %{state: state}
        )

      # 5. Read again to verify edit
      {:ok, final_result} = read_tool.function.(%{"file_path" => "code.ex"}, %{state: state})
      assert final_result =~ ":edited"
      refute final_result =~ ":ok"

      # 6. List files (should show our file)
      {:ok, final_ls} = ls_tool.function.(%{}, %{state: state})
      assert final_ls =~ "code.ex"
    end

    test "multiple files can coexist" do
      [_, _, write_tool, _] = FileSystem.tools(%{})
      state = State.new!(%{files: %{}})

      # Write multiple files - tools return state deltas that need to be merged
      {:ok, _, delta1} =
        write_tool.function.(
          %{"file_path" => "file1.txt", "content" => "content1"},
          %{state: state}
        )

      state = State.merge_states(state, delta1)

      {:ok, _, delta2} =
        write_tool.function.(
          %{"file_path" => "file2.txt", "content" => "content2"},
          %{state: state}
        )

      state = State.merge_states(state, delta2)

      {:ok, _, delta3} =
        write_tool.function.(
          %{"file_path" => "file3.txt", "content" => "content3"},
          %{state: state}
        )

      state = State.merge_states(state, delta3)

      assert map_size(state.files) == 3
      # Use State.get_file to extract content from FileData
      assert State.get_file(state, "file1.txt") == "content1"
      assert State.get_file(state, "file2.txt") == "content2"
      assert State.get_file(state, "file3.txt") == "content3"
    end
  end
end
