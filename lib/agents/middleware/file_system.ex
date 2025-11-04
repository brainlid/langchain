defmodule LangChain.Agents.Middleware.FileSystem do
  @moduledoc """
  Middleware that adds mock filesystem capabilities to agents.

  Provides tools for file operations in an isolated, in-memory filesystem:
  - `ls`: List all files (with optional pattern filtering)
  - `read_file`: Read file contents with line numbers and pagination
  - `write_file`: Create or overwrite files
  - `edit_file`: Make targeted edits with string replacement

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

  Available tools: `"ls"`, `"read_file"`, `"write_file"`, `"edit_file"`

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

  @system_prompt """
  ## Filesystem Tools

  You have access to a virtual filesystem with these tools:
  - `ls`: List files (optionally filter with patterns)
  - `read_file`: Read file contents with line numbers and pagination
  - `write_file`: Create new files (cannot overwrite existing files)
  - `edit_file`: Modify existing files with string replacement

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
  - Use `edit_file` for modifications, `write_file` only for new files
  - Provide sufficient context in `old_string` to ensure unique matches
  - Group related files in the same directory

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
        Keyword.get(opts, :enabled_tools, ["ls", "read_file", "write_file", "edit_file"]),
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
      "edit_file" => build_edit_file_tool(config)
    }

    enabled_tools =
      Map.get(config, :enabled_tools, ["ls", "read_file", "write_file", "edit_file"])

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

  # Tool execution functions

  defp execute_ls_tool(args, _context, config) do
    pattern = get_arg(args, "pattern")

    # List all files using FileSystemServer (reads directly from ETS)
    all_files = LangChain.Agents.FileSystemServer.list_files(config.agent_id)

    # Apply pattern filtering
    filtered_files = filter_by_pattern(all_files, pattern)

    # Sort for consistency (already sorted by FileSystemServer, but ensure it)
    sorted_files = Enum.sort(filtered_files)

    message =
      if Enum.empty?(sorted_files) do
        if pattern do
          "No files match pattern: #{pattern}"
        else
          "No files in filesystem"
        end
      else
        header = if pattern, do: "Files matching '#{pattern}':\n", else: "Files:\n"
        header <> Enum.join(sorted_files, "\n")
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
      case LangChain.Agents.FileSystemServer.read_file(config.agent_id, normalized_path) do
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
          if LangChain.Agents.FileSystemServer.file_exists?(config.agent_id, normalized_path) do
            {:error,
             "File already exists: #{normalized_path}. Use edit_file to modify existing files."}
          else
            # Write file using FileSystemServer
            case LangChain.Agents.FileSystemServer.write_file(
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
          case LangChain.Agents.FileSystemServer.read_file(config.agent_id, normalized_path) do
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
    case LangChain.Agents.FileSystemServer.write_file(agent_id, file_path, updated_content) do
      :ok ->
        {:ok, success_message}

      {:error, reason} ->
        {:error, "Failed to save edit: #{inspect(reason)}"}
    end
  end

  defp get_arg(args, key) when is_map(args) do
    args[key] || args[String.to_atom(key)]
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
