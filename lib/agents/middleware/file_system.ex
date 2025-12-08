defmodule LangChain.Agents.Middleware.FileSystem do
  @moduledoc """
  Middleware that adds mock filesystem capabilities to agents.

  Provides tools for file operations in an isolated, in-memory filesystem:
  - `ls`: List all files (with optional pattern filtering)
  - `read_file`: Read file contents with line numbers and pagination
  - `write_file`: Create or overwrite files
  - `edit_file`: Make targeted edits with string replacement
  - `search_text`: Search for text patterns within files or across all files
  - `edit_lines`: Replace a range of lines by line number
  - `delete_file`: Delete files from the filesystem

  ## Usage

      {:ok, agent} = Agent.new(
        model: model,
        middleware: [Filesystem]
      )

      # Agent can now use filesystem tools

  ## Configuration

  ### Basic Configuration

  - `:long_term_memory` - Enable persistence (default: false)
  - `:memories_prefix` - Path prefix for long-term persisted files (default: "memories")
  - `:enabled_tools` - List of tool names to enable (default: all tools)
  - `:custom_tool_descriptions` - Map of custom descriptions per tool

  ### Selective Tool Enabling

  Enable only specific tools (e.g., read-only access):

      {:ok, agent} = Agent.new(
        model: model,
        filesystem_opts: [
          enabled_tools: ["ls", "read_file"]  # Read-only, no write/edit
        ]
      )

  Available tools: `"ls"`, `"read_file"`, `"write_file"`, `"edit_file"`, `"search_text"`, `"edit_lines"`, `"delete_file"`

  ### Custom Tool Descriptions

  Override default tool descriptions:

      {:ok, agent} = Agent.new(
        model: model,
        filesystem_opts: [
          custom_tool_descriptions: %{
            "read_file" => "Custom description for reading files...",
            "write_file" => "Custom description for writing files..."
          }
        ]
      )

  ### Persistence Configuration

  Provide persistence callbacks to save/load files from external storage:

      # Module-based (implement LangChain.Agents.FilesystemCallbacks)
      {:ok, agent} = Agent.new(
        model: model,
        filesystem_opts: [
          persistence: MyApp.FilesystemPersistence,
          context: %{user_id: user_id}
        ]
      )

      # Function-based (inline callbacks)
      {:ok, agent} = Agent.new(
        model: model,
        filesystem_opts: [
          on_write: fn file_path, content, ctx ->
            MyApp.Files.save(ctx.user_id, file_path, content)
          end,
          on_read: fn file_path, ctx ->
            MyApp.Files.get(ctx.user_id, file_path)
          end,
          context: %{user_id: user_id}
        ]
      )

  ### Persistence Options

  - `:persistence` - Module implementing FilesystemCallbacks behavior
  - `:on_write` - Function for write operations `fn(path, content, ctx) -> result`
  - `:on_read` - Function for read operations `fn(path, ctx) -> {:ok, content} | {:error, reason}`
  - `:on_delete` - Function for delete operations `fn(path, ctx) -> result`
  - `:on_list` - Function for list operations `fn(ctx) -> {:ok, [paths]} | {:error, reason}`
  - `:context` - Map passed to all callbacks (user_id, session_id, etc.)
  - `:cache_reads` - Cache reads in memory (default: true)
  - `:fail_on_persistence_error` - Fail operations if persistence fails (default: false)

  ## File Organization

  Use hierarchical paths to organize files:

      write_file(file_path: "Chapter 1/summary.md", content: "...")
      write_file(file_path: "images/diagram.png", content: "...")
      ls(pattern: "Chapter 1/*")

  ## Pattern Filtering

  The `ls` tool supports wildcard patterns:
  - `*` matches any characters
  - Examples: `*summary*`, `*.md`, `Chapter 1/*`

  ## Path Security

  Paths are validated to prevent security issues:
  - Paths starting with "/" are rejected (system path protection)
  - Path traversal attempts ("..") are rejected
  - Home directory shortcuts ("~") are rejected
  """

  @behaviour LangChain.Agents.Middleware

  require Logger
  alias LangChain.Function
  alias LangChain.Agents.FileSystemServer

  @system_prompt """
  ## Filesystem Tools

  You have access to a virtual filesystem with these tools:
  - `ls`: List files (optionally filter with patterns)
  - `read_file`: Read file contents with line numbers and pagination
  - `write_file`: Create new files (cannot overwrite existing files)
  - `edit_file`: Modify existing files with string replacement
  - `search_text`: Search for text patterns in specific files or across all files
  - `edit_lines`: Replace a block of lines by line number range
  - `delete_file`: Delete files from the filesystem

  ## File Organization

  Use hierarchical paths to organize files logically:
  - All paths must start with a forward slash "/"
  - Examples: "/notes.txt", "/Chapter1/summary.md", "/data/results.csv"
  - No path traversal ("..") or home directory ("~") allowed

  ## Pattern Filtering

  The `ls` tool supports wildcard patterns:
  - `*` matches any characters
  - Examples: `*summary*`, `*.md`, `/Chapter1/*`

  ## Best Practices

  - Always use `ls` first to see available files
  - Read files before editing to understand content
  - Use `edit_file` for small, targeted edits with string replacement
  - Use `edit_lines` for large block replacements (more efficient with tokens)
  - Use `write_file` only for new files
  - Provide sufficient context in `old_string` to ensure unique matches
  - Group related files in the same directory
  - Never `delete_file` without first using `ls` to locate it

  ## Persistence Behavior

  - Different directories may have different persistence settings
  - Some directories may be read-only (you can read but not write)
  - Large or archived files may load slowly on first access
  """

  @impl true
  def init(opts) do
    # Require agent_id to link to FileSystemServer
    agent_id = Keyword.fetch!(opts, :agent_id)

    config = %{
      agent_id: agent_id,
      # Tool configuration
      enabled_tools:
        Keyword.get(opts, :enabled_tools, [
          "ls",
          "read_file",
          "write_file",
          "edit_file",
          "search_text",
          "edit_lines",
          "delete_file"
        ]),
      custom_tool_descriptions: Keyword.get(opts, :custom_tool_descriptions, %{})
    }

    {:ok, config}
  end

  @impl true
  def system_prompt(_config) do
    @system_prompt
  end

  @impl true
  def tools(config) do
    all_tools = %{
      "ls" => build_ls_tool(config),
      "read_file" => build_read_file_tool(config),
      "write_file" => build_write_file_tool(config),
      "edit_file" => build_edit_file_tool(config),
      "search_text" => build_search_text_tool(config),
      "edit_lines" => build_edit_lines_tool(config),
      "delete_file" => build_delete_file_tool(config)
    }

    enabled_tools =
      Map.get(config, :enabled_tools, [
        "ls",
        "read_file",
        "write_file",
        "edit_file",
        "search_text",
        "edit_lines",
        "delete_file"
      ])

    enabled_tools
    |> Enum.map(fn tool_name -> Map.get(all_tools, tool_name) end)
    |> Enum.reject(&is_nil/1)
  end

  @impl true
  def state_schema do
    # Files are stored in state.files as %{file_path => content}
    []
  end

  # Tool builders

  defp build_ls_tool(config) do
    default_description = """
    Lists files in the filesystem, optionally filtering by pattern or directory.

    Usage:
    - The list_files tool will return a list of the files in the filesystem.
    - You can optionally provide a path parameter to list files in a specific directory.
    - You can optionally provide a pattern parameter to filter the files by pattern.
    - This is very useful for exploring the file system and finding the right file to read or edit.
    - You should almost ALWAYS use this tool before using the read or edit tools.

    Optionally filter by pattern using wildcards:
    - Use '*' to match any characters
    - Examples: '*summary*', '*.md', '/Chapter 1/*'

    Without a pattern, lists all files.
    """

    description = get_custom_description(config, "ls", default_description)

    Function.new!(%{
      name: "ls",
      description: description,
      parameters_schema: %{
        type: "object",
        properties: %{
          pattern: %{
            type: "string",
            description:
              "Optional wildcard pattern to filter files (e.g., '*summary*', '*.md', '/Chapter 1/*')"
          }
        }
      },
      function: fn args, context -> execute_ls_tool(args, context, config) end
    })
  end

  defp build_read_file_tool(config) do
    default_description = """
    Read a file's contents with line numbers.

    Supports pagination with offset and limit parameters.
    Returns the file content with line numbers for easy reference.
    """

    description = get_custom_description(config, "read_file", default_description)

    Function.new!(%{
      name: "read_file",
      description: description,
      parameters_schema: %{
        type: "object",
        properties: %{
          file_path: %{
            type: "string",
            description: "Path to the file to read"
          },
          offset: %{
            type: "integer",
            description: "Line number to start reading from (0-based)",
            default: 0
          },
          limit: %{
            type: "integer",
            description: "Maximum number of lines to read",
            default: 2000
          }
        },
        required: ["file_path"]
      },
      function: fn args, context -> execute_read_file_tool(args, context, config) end
    })
  end

  defp build_write_file_tool(config) do
    default_description = """
    Write content to a file (creates new files only).

    This tool creates new files. If the file already exists, an error will be returned.
    Use edit_file to modify existing files.
    """

    description = get_custom_description(config, "write_file", default_description)

    Function.new!(%{
      name: "write_file",
      description: description,
      parameters_schema: %{
        type: "object",
        properties: %{
          file_path: %{
            type: "string",
            description: "Path where the file should be written"
          },
          content: %{
            type: "string",
            description: "Content to write to the file"
          }
        },
        required: ["file_path", "content"]
      },
      function: fn args, context -> execute_write_file_tool(args, context, config) end
    })
  end

  defp build_edit_file_tool(config) do
    default_description = """
    Edit a file by replacing old_string with new_string.

    Performs string replacement within the file. By default, requires
    old_string to appear exactly once (for safety). Use replace_all: true
    to replace multiple occurrences.
    """

    description = get_custom_description(config, "edit_file", default_description)

    Function.new!(%{
      name: "edit_file",
      description: description,
      parameters_schema: %{
        type: "object",
        properties: %{
          file_path: %{
            type: "string",
            description: "Path to the file to edit"
          },
          old_string: %{
            type: "string",
            description: "String to find and replace (must be unique unless replace_all is true)"
          },
          new_string: %{
            type: "string",
            description: "String to replace old_string with"
          },
          replace_all: %{
            type: "boolean",
            description: "If true, replace all occurrences. If false, require exactly one match.",
            default: false
          }
        },
        required: ["file_path", "old_string", "new_string"]
      },
      function: fn args, context -> execute_edit_file_tool(args, context, config) end
    })
  end

  defp build_delete_file_tool(config) do
    default_description = """
    Delete a file from the filesystem.

    Removes the file from both memory and persistence (if applicable).
    Files in read-only directories cannot be deleted.
    """

    description = get_custom_description(config, "delete_file", default_description)

    Function.new!(%{
      name: "delete_file",
      description: description,
      parameters_schema: %{
        type: "object",
        properties: %{
          file_path: %{
            type: "string",
            description: "Path to the file to delete"
          }
        },
        required: ["file_path"]
      },
      function: fn args, context -> execute_delete_file_tool(args, context, config) end
    })
  end

  defp build_search_text_tool(config) do
    default_description = """
    Search for text patterns within files.

    Can search within a specific file or across all loaded files.
    Returns matches with line numbers and optional context lines.

    Usage:
    - Provide pattern (required) - text or regex to search for
    - Provide file_path (optional) - if omitted, searches all files
    - Set case_sensitive (optional, default: true)
    - Set context_lines (optional, default: 0) - lines before/after to show
    - Set max_results (optional, default: 50) - limit number of results

    Examples:
    - Search single file: search_text(pattern: "TODO", file_path: "/notes.txt")
    - Search all files: search_text(pattern: "important", case_sensitive: false)
    - With context: search_text(pattern: "error", context_lines: 2)
    """

    description = get_custom_description(config, "search_text", default_description)

    Function.new!(%{
      name: "search_text",
      description: description,
      parameters_schema: %{
        "type" => "object",
        "properties" => %{
          "pattern" => %{
            "type" => "string",
            "description" => "Text or regex pattern to search for"
          },
          "file_path" => %{
            "type" => "string",
            "description" =>
              "Optional: specific file to search. If omitted, searches all loaded files."
          },
          "case_sensitive" => %{
            "type" => "boolean",
            "description" => "Whether the search should be case-sensitive",
            "default" => true
          },
          "context_lines" => %{
            "type" => "integer",
            "description" => "Number of lines before and after each match to show",
            "default" => 0
          },
          "max_results" => %{
            "type" => "integer",
            "description" => "Maximum number of matches to return",
            "default" => 50
          }
        },
        "required" => ["pattern"]
      },
      function: fn args, context -> execute_search_text_tool(args, context, config) end
    })
  end

  defp build_edit_lines_tool(config) do
    default_description = """
    Edit a file by replacing a range of lines with new content.

    This tool is more efficient than edit_file for replacing large blocks of text,
    such as rewriting multiple paragraphs or sections. It uses line numbers instead
    of string matching, making it more reliable for large edits.

    Line numbers are 1-based (matching read_file output) and the range is inclusive
    (both start_line and end_line are replaced).

    Usage:
    - First use read_file to see the file with line numbers
    - Identify the start_line and end_line you want to replace
    - Provide new_content to replace those lines
    - The tool will replace lines [start_line, end_line] inclusive

    Examples:
    - Replace lines 10-15: edit_lines(file_path: "/doc.txt", start_line: 10, end_line: 15, new_content: "new text")
    - Replace single line: edit_lines(file_path: "/doc.txt", start_line: 42, end_line: 42, new_content: "new line")
    - Replace large block: edit_lines(file_path: "/story.txt", start_line: 120, end_line: 135, new_content: "...")

    Best practices:
    - Use read_file first to verify line numbers
    - For small, targeted edits, use edit_file instead
    - For large block replacements, this tool is more efficient
    """

    description = get_custom_description(config, "edit_lines", default_description)

    Function.new!(%{
      name: "edit_lines",
      description: description,
      parameters_schema: %{
        type: "object",
        properties: %{
          file_path: %{
            type: "string",
            description: "Path to the file to edit"
          },
          start_line: %{
            type: "integer",
            description: "Starting line number (1-based, inclusive)"
          },
          end_line: %{
            type: "integer",
            description: "Ending line number (1-based, inclusive)"
          },
          new_content: %{
            type: "string",
            description: "New content to replace the line range. Can be multi-line."
          }
        },
        required: ["file_path", "start_line", "end_line", "new_content"]
      },
      function: fn args, context -> execute_edit_lines_tool(args, context, config) end
    })
  end

  # Tool execution functions

  defp execute_ls_tool(args, _context, config) do
    pattern = get_arg(args, "pattern")

    # List all files using FileSystemServer
    all_files = FileSystemServer.list_files(config.agent_id)

    # Apply pattern filtering
    filtered_files = filter_by_pattern(all_files, pattern)

    message =
      if Enum.empty?(filtered_files) do
        if pattern do
          "No files match pattern: #{pattern}"
        else
          "No files in filesystem"
        end
      else
        header = if pattern, do: "Files matching '#{pattern}':\n", else: "Files:\n"
        header <> Enum.join(filtered_files, "\n")
      end

    {:ok, message}
  rescue
    e ->
      {:error, "Filesystem not available: #{Exception.message(e)}"}
  end

  defp execute_read_file_tool(args, _context, config) do
    file_path = get_arg(args, "file_path")
    offset = get_arg(args, "offset") || 0
    limit = get_arg(args, "limit") || 2000

    # Validate path
    with {:ok, normalized_path} <- validate_path(file_path) do
      # Read file using FileSystemServer (handles lazy loading automatically)
      case FileSystemServer.read_file(config.agent_id, normalized_path) do
        {:ok, content} ->
          format_file_content(content, normalized_path, offset, limit)

        {:error, :enoent} ->
          {:error, "File not found: #{normalized_path}"}

        {:error, reason} ->
          {:error, "Failed to read file: #{inspect(reason)}"}
      end
    else
      {:error, reason} -> {:error, reason}
    end
  rescue
    e ->
      {:error, "Filesystem not available: #{Exception.message(e)}"}
  end

  defp format_file_content(content, _file_path, offset, limit) do
    lines = String.split(content, "\n")
    total_lines = length(lines)

    # Apply offset and limit
    selected_lines =
      lines
      |> Enum.slice(offset, limit)
      |> Enum.with_index(offset)
      |> Enum.map(fn {line, idx} ->
        # Format with fixed-width line numbers (1-based for display), truncate long lines
        line_num = String.pad_leading(Integer.to_string(idx + 1), 6)

        truncated_line =
          if String.length(line) > 2000 do
            String.slice(line, 0, 2000) <> "... (line truncated)"
          else
            line
          end

        "#{line_num}\t#{truncated_line}"
      end)

    result =
      if Enum.empty?(selected_lines) do
        "File is empty or offset is beyond file length."
      else
        header =
          if offset > 0 or offset + limit < total_lines do
            showing_end = min(offset + limit, total_lines)
            "Showing lines #{offset + 1} to #{showing_end} of #{total_lines}:\n"
          else
            ""
          end

        header <> Enum.join(selected_lines, "\n")
      end

    {:ok, result}
  end

  defp execute_write_file_tool(args, _context, config) do
    file_path = get_arg(args, "file_path")
    content = get_arg(args, "content")

    cond do
      is_nil(file_path) or is_nil(content) ->
        {:error, "Both file_path and content are required"}

      true ->
        # Validate path
        with {:ok, normalized_path} <- validate_path(file_path) do
          # Check if file already exists (overwrite protection)
          if FileSystemServer.file_exists?(config.agent_id, normalized_path) do
            {:error,
             "File already exists: #{normalized_path}. Use edit_file to modify existing files."}
          else
            # Write file using FileSystemServer
            case FileSystemServer.write_file(
                   config.agent_id,
                   normalized_path,
                   content
                 ) do
              :ok ->
                {:ok, "File created successfully: #{normalized_path}"}

              {:error, reason} ->
                {:error, "Failed to create file: #{inspect(reason)}"}
            end
          end
        else
          {:error, reason} -> {:error, reason}
        end
    end
  rescue
    e ->
      {:error, "Filesystem not available: #{Exception.message(e)}"}
  end

  defp execute_edit_file_tool(args, _context, config) do
    file_path = get_arg(args, "file_path")
    old_string = get_arg(args, "old_string")
    new_string = get_arg(args, "new_string")
    replace_all = get_boolean_arg(args, "replace_all", false)

    cond do
      is_nil(file_path) or is_nil(old_string) or is_nil(new_string) ->
        {:error, "file_path, old_string, and new_string are required"}

      true ->
        # Validate path
        with {:ok, normalized_path} <- validate_path(file_path) do
          # Read current content using FileSystemServer
          case FileSystemServer.read_file(config.agent_id, normalized_path) do
            {:ok, content} ->
              perform_edit(
                config.agent_id,
                normalized_path,
                content,
                old_string,
                new_string,
                replace_all
              )

            {:error, :enoent} ->
              {:error, "File not found: #{normalized_path}"}

            {:error, reason} ->
              {:error, "Failed to read file: #{inspect(reason)}"}
          end
        else
          {:error, reason} -> {:error, reason}
        end
    end
  rescue
    e ->
      {:error, "Filesystem not available: #{Exception.message(e)}"}
  end

  defp execute_delete_file_tool(args, _context, config) do
    file_path = get_arg(args, "file_path")

    cond do
      is_nil(file_path) ->
        {:error, "file_path is required"}

      true ->
        # Validate path
        with {:ok, normalized_path} <- validate_path(file_path) do
          # Delete file using FileSystemServer
          case FileSystemServer.delete_file(config.agent_id, normalized_path) do
            :ok ->
              {:ok, "File deleted successfully: #{normalized_path}"}

            {:error, reason} ->
              {:error, "Failed to delete file: #{inspect(reason)}"}
          end
        else
          {:error, reason} -> {:error, reason}
        end
    end
  rescue
    e ->
      {:error, "Filesystem not available: #{Exception.message(e)}"}
  end

  defp execute_search_text_tool(args, _context, config) do
    pattern = get_arg(args, "pattern")
    file_path = get_arg(args, "file_path")
    case_sensitive = get_boolean_arg(args, "case_sensitive", true)
    context_lines = get_integer_arg(args, "context_lines", 0)
    max_results = get_integer_arg(args, "max_results", 50)

    cond do
      is_nil(pattern) ->
        {:error, "pattern is required"}

      true ->
        # Compile regex pattern with inline flags for case-insensitive search
        pattern_with_flags = if case_sensitive, do: pattern, else: "(?i)#{pattern}"

        case Regex.compile(pattern_with_flags) do
          {:ok, regex} ->
            if file_path do
              # Search single file
              search_single_file(config.agent_id, file_path, regex, context_lines, max_results)
            else
              # Search all files
              search_all_files(config.agent_id, regex, context_lines, max_results)
            end

          {:error, _reason} ->
            {:error, "Invalid regex pattern: #{pattern}"}
        end
    end
  rescue
    e ->
      {:error, "Search failed: #{Exception.message(e)}"}
  end

  defp execute_edit_lines_tool(args, _context, config) do
    file_path = get_arg(args, "file_path")
    start_line = get_integer_arg(args, "start_line", nil)
    end_line = get_integer_arg(args, "end_line", nil)
    new_content = get_arg(args, "new_content")

    cond do
      is_nil(file_path) ->
        {:error, "file_path is required"}

      is_nil(start_line) or is_nil(end_line) ->
        {:error, "start_line and end_line are required"}

      is_nil(new_content) ->
        {:error, "new_content is required"}

      start_line < 1 ->
        {:error, "start_line must be >= 1 (line numbers are 1-based)"}

      end_line < start_line ->
        {:error, "end_line must be >= start_line"}

      true ->
        with {:ok, normalized_path} <- validate_path(file_path),
             {:ok, content} <- FileSystemServer.read_file(config.agent_id, normalized_path) do
          perform_line_edit(
            config.agent_id,
            normalized_path,
            content,
            start_line,
            end_line,
            new_content
          )
        else
          {:error, :enoent} ->
            {:error, "File not found: #{file_path}"}

          {:error, reason} ->
            {:error, "Failed to read file: #{inspect(reason)}"}
        end
    end
  rescue
    e ->
      {:error, "Edit failed: #{Exception.message(e)}"}
  end

  defp search_single_file(agent_id, file_path, regex, context_lines, max_results) do
    with {:ok, normalized_path} <- validate_path(file_path),
         {:ok, content} <- FileSystemServer.read_file(agent_id, normalized_path) do
      {matches, truncated} = find_matches_in_content(content, regex, context_lines, max_results)
      format_search_results([{normalized_path, matches}], max_results, truncated)
    else
      {:error, :enoent} ->
        {:error, "File not found: #{file_path}"}

      {:error, reason} ->
        {:error, "Failed to search file: #{inspect(reason)}"}
    end
  end

  defp search_all_files(agent_id, regex, context_lines, max_results) do
    all_files = FileSystemServer.list_files(agent_id)

    # Search each file and collect matches, tracking total matches
    {results, _total_matches, any_truncated} =
      all_files
      |> Enum.reduce({[], 0, false}, fn file_path, {acc, match_count, truncated} ->
        # Calculate remaining limit for this file
        remaining = max_results - match_count

        if remaining <= 0 do
          # Already hit limit, stop collecting
          {acc, match_count, truncated}
        else
          case FileSystemServer.read_file(agent_id, file_path) do
            {:ok, content} ->
              {matches, file_truncated} =
                find_matches_in_content(content, regex, context_lines, remaining)

              if Enum.empty?(matches) do
                {acc, match_count, truncated}
              else
                {[{file_path, matches} | acc], match_count + length(matches),
                 truncated or file_truncated}
              end

            {:error, _} ->
              {acc, match_count, truncated}
          end
        end
      end)

    results = Enum.reverse(results)
    format_search_results(results, max_results, any_truncated)
  end

  defp find_matches_in_content(content, regex, context_lines, max_results) do
    lines = String.split(content, "\n")

    # Collect up to max_results + 1 to detect truncation
    matches =
      lines
      |> Enum.with_index(1)
      |> Enum.reduce([], fn {line, line_num}, acc ->
        if length(acc) > max_results do
          acc
        else
          if Regex.match?(regex, line) do
            match_info = %{
              line_number: line_num,
              line: line,
              context: extract_context(lines, line_num - 1, context_lines)
            }

            [match_info | acc]
          else
            acc
          end
        end
      end)
      |> Enum.reverse()

    # Return matches and truncation flag
    if length(matches) > max_results do
      {Enum.take(matches, max_results), true}
    else
      {matches, false}
    end
  end

  defp extract_context(lines, zero_based_line_num, context_lines) do
    if context_lines > 0 do
      start_idx = max(0, zero_based_line_num - context_lines)
      end_idx = min(length(lines) - 1, zero_based_line_num + context_lines)

      %{
        before: Enum.slice(lines, start_idx, zero_based_line_num - start_idx),
        after: Enum.slice(lines, zero_based_line_num + 1, end_idx - zero_based_line_num)
      }
    else
      nil
    end
  end

  defp format_search_results(results, max_results, truncated) do
    if Enum.empty?(results) or Enum.all?(results, fn {_, matches} -> Enum.empty?(matches) end) do
      {:ok, "No matches found"}
    else
      formatted =
        results
        |> Enum.flat_map(fn {file_path, matches} ->
          file_header = ["", "File: #{file_path}"]
          match_lines = Enum.map(matches, &format_match/1)
          file_header ++ match_lines
        end)
        |> Enum.join("\n")

      footer = if truncated, do: "\n\n(Results truncated at #{max_results} matches)", else: ""

      {:ok, formatted <> footer}
    end
  end

  defp format_match(%{line_number: line_num, line: line, context: nil}) do
    # Format line number the same way as format_file_content (6 chars padded, tab separator)
    formatted_line_num = String.pad_leading(Integer.to_string(line_num), 6)
    "#{formatted_line_num}\t#{line}"
  end

  defp format_match(%{line_number: line_num, line: line, context: context}) do
    # Format context lines with line numbers
    before =
      if context.before do
        context.before
        |> Enum.with_index()
        |> Enum.map(fn {ctx_line, idx} ->
          # Calculate line number for context line
          ctx_line_num = line_num - length(context.before) + idx
          formatted_num = String.pad_leading(Integer.to_string(ctx_line_num), 6)
          "#{formatted_num} |\t#{ctx_line}"
        end)
        |> Enum.join("\n")
      else
        ""
      end

    after_ctx =
      if context.after do
        context.after
        |> Enum.with_index()
        |> Enum.map(fn {ctx_line, idx} ->
          # Calculate line number for context line
          ctx_line_num = line_num + idx + 1
          formatted_num = String.pad_leading(Integer.to_string(ctx_line_num), 6)
          "#{formatted_num} |\t#{ctx_line}"
        end)
        |> Enum.join("\n")
      else
        ""
      end

    # Format the matching line
    formatted_line_num = String.pad_leading(Integer.to_string(line_num), 6)
    match_line = "#{formatted_line_num}\t#{line}"

    lines =
      [
        before,
        match_line,
        after_ctx
      ]
      |> Enum.reject(&(&1 == ""))
      |> Enum.join("\n")

    lines
  end

  defp perform_edit(agent_id, file_path, content, old_string, new_string, replace_all) do
    # Split to count occurrences
    parts = String.split(content, old_string, parts: :infinity)
    occurrence_count = length(parts) - 1

    cond do
      occurrence_count == 0 ->
        {:error, "String not found in file: '#{old_string}'"}

      occurrence_count == 1 ->
        # Single occurrence, safe to replace
        updated_content = String.replace(content, old_string, new_string, global: false)
        write_edit(agent_id, file_path, updated_content, "File edited successfully: #{file_path}")

      occurrence_count > 1 and not replace_all ->
        {:error,
         "String appears #{occurrence_count} times in file. Use replace_all: true or provide more context in old_string."}

      occurrence_count > 1 and replace_all ->
        # Replace all occurrences
        updated_content = String.replace(content, old_string, new_string, global: true)

        write_edit(
          agent_id,
          file_path,
          updated_content,
          "File edited successfully: #{file_path} (#{occurrence_count} replacements)"
        )
    end
  end

  defp write_edit(agent_id, file_path, updated_content, success_message) do
    case FileSystemServer.write_file(agent_id, file_path, updated_content) do
      :ok ->
        {:ok, success_message}

      {:error, reason} ->
        {:error, "Failed to save edit: #{inspect(reason)}"}
    end
  end

  defp perform_line_edit(agent_id, file_path, content, start_line, end_line, new_content) do
    lines = String.split(content, "\n")
    total_lines = length(lines)

    # Convert to 0-based for Enum operations
    start_idx = start_line - 1
    end_idx = end_line - 1

    cond do
      start_idx >= total_lines ->
        {:error, "start_line #{start_line} is beyond file length (#{total_lines} lines)"}

      end_idx >= total_lines ->
        {:error, "end_line #{end_line} is beyond file length (#{total_lines} lines)"}

      true ->
        # Extract the lines being replaced (for confirmation message)
        replaced_lines = Enum.slice(lines, start_idx, end_idx - start_idx + 1)
        lines_replaced_count = length(replaced_lines)

        # Build the new file content
        before = Enum.slice(lines, 0, start_idx)
        after_lines = Enum.slice(lines, end_idx + 1, total_lines - end_idx - 1)

        # Split new_content into lines (preserving the newlines)
        new_lines = String.split(new_content, "\n")

        updated_lines = before ++ new_lines ++ after_lines
        updated_content = Enum.join(updated_lines, "\n")

        # Write the updated content
        case FileSystemServer.write_file(agent_id, file_path, updated_content) do
          :ok ->
            {:ok,
             "File edited successfully: #{file_path}\nReplaced #{lines_replaced_count} lines (#{start_line}-#{end_line})"}

          {:error, reason} ->
            {:error, "Failed to save edit: #{inspect(reason)}"}
        end
    end
  end

  defp get_arg(nil = _args, _key), do: nil

  defp get_arg(args, key) when is_map(args) do
    # Use Map.get to avoid issues with false values
    case Map.get(args, key) do
      nil -> Map.get(args, String.to_atom(key))
      value -> value
    end
  end

  defp get_boolean_arg(args, key, default) when is_map(args) do
    case get_arg(args, key) do
      nil -> default
      val when is_boolean(val) -> val
      "true" -> true
      "false" -> false
      _ -> default
    end
  end

  defp get_integer_arg(args, key, default) when is_map(args) do
    case get_arg(args, key) do
      nil -> default
      val when is_integer(val) -> val
      val when is_binary(val) -> String.to_integer(val)
      _ -> default
    end
  end

  defp filter_by_pattern(files, nil), do: files

  defp filter_by_pattern(files, pattern) do
    # Convert wildcard pattern to regex
    # "*summary*" -> ~r/.*summary.*/
    # "Chapter 1/*" -> ~r/Chapter 1\/.*/
    # "*.md" -> ~r/.*\.md/
    regex_pattern =
      pattern
      |> String.replace(".", "\\.")
      |> String.replace("*", ".*")
      |> then(&Regex.compile!(&1))

    Enum.filter(files, &Regex.match?(regex_pattern, &1))
  end

  # Path validation and security

  @doc false
  def validate_path(path) when is_binary(path) do
    cond do
      String.trim(path) == "" ->
        {:error, "Path cannot be empty"}

      not String.starts_with?(path, "/") ->
        {:error, "Path must start with '/' (e.g., '/notes.txt', '/data/file.csv')"}

      String.contains?(path, "..") ->
        {:error, "Path traversal with '..' is not allowed"}

      String.starts_with?(path, "~") ->
        {:error, "Home directory paths starting with '~' are not allowed"}

      true ->
        {:ok, normalize_path(path)}
    end
  end

  def validate_path(_path), do: {:error, "Path must be a string"}

  @doc false
  def normalize_path(path) when is_binary(path) do
    path
    |> String.replace("\\", "/")
    |> String.replace(~r|/+|, "/")
  end

  @doc false
  def get_custom_description(config, tool_name, default_description) do
    custom_descriptions = Map.get(config, :custom_tool_descriptions, %{})
    Map.get(custom_descriptions, tool_name, default_description)
  end
end
