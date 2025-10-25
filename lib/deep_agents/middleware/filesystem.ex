defmodule LangChain.DeepAgents.Middleware.Filesystem do
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

      # Module-based (implement LangChain.DeepAgents.FilesystemCallbacks)
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

  @behaviour LangChain.DeepAgents.Middleware

  require Logger
  alias LangChain.Function

  @system_prompt """
  ## Filesystem Tools `ls`, `read_file`, `write_file`, `edit_file`

  You have access to a filesystem which you can interact with using these tools.
  All file paths must start with a /.

  - ls: list files in the filesystem (optionally filter with patterns)
  - read_file: read a file from the filesystem
  - write_file: write to a file in the filesystem
  - edit_file: edit a file in the filesystem

  ## File Organization

  Use hierarchical paths to organize files logically:
  - Use forward slashes (/) to access logical groupings
  - Examples: "data.csv", "/Chapter 1/notes.txt", "/images/diagram.png", "/data/results.csv"

  # ## Using the ls Tool

  # List files with optional pattern filtering:
  # - `ls()` - List all files
  # - `ls(pattern: "/Chapter 1/*")` - List files in "Chapter 1"
  # - `ls(pattern: "*summary*")` - Find files containing "summary"
  # - `ls(pattern: "*.md")` - Find all markdown files

  ## Best Practices

  - Always use `ls` before reading or editing to see available files
  - Read files before editing them to understand their content
  - Use `edit_file` for modifications, `write_file` for new files or complete rewrites
  - Provide sufficient context in `old_string` to ensure unique matches
  - Use hierarchical paths to organize related files together

  [PLACEHOLDER: Additional filesystem guidance will be added]
  """

  @impl true
  def init(opts) do
    config = %{
      long_term_memory: Keyword.get(opts, :long_term_memory, false),
      memories_prefix: Keyword.get(opts, :memories_prefix, "memories"),
      # Tool configuration
      enabled_tools:
        Keyword.get(opts, :enabled_tools, ["ls", "read_file", "write_file", "edit_file"]),
      custom_tool_descriptions: Keyword.get(opts, :custom_tool_descriptions, %{}),
      # Persistence configuration
      persistence: Keyword.get(opts, :persistence),
      on_write: Keyword.get(opts, :on_write),
      on_read: Keyword.get(opts, :on_read),
      on_delete: Keyword.get(opts, :on_delete),
      on_list: Keyword.get(opts, :on_list),
      context: Keyword.get(opts, :context, %{}),
      # Behavior options
      cache_reads: Keyword.get(opts, :cache_reads, true),
      fail_on_persistence_error: Keyword.get(opts, :fail_on_persistence_error, false)
    }

    {:ok, config}
  end

  @impl true
  def system_prompt(config) do
    supplement =
      if Map.get(config, :long_term_memory, false) do
        "\n\nNote: Files persist across conversations with long-term memory enabled."
      else
        ""
      end

    @system_prompt <> supplement
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

    # Add longterm memory supplement if persistence is enabled
    longterm_supplement =
      if Map.get(config, :long_term_memory, false) do
        memories_prefix = Map.get(config, :memories_prefix, "memories")

        "\n- Files from the longterm filesystem will be prefixed with the #{memories_prefix} path."
      else
        ""
      end

    description = get_custom_description(config, "ls", default_description <> longterm_supplement)

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

  defp execute_ls_tool(args, context, config) do
    pattern = get_arg(args, "pattern")

    # Get in-memory files
    memory_files = Map.keys(context.state.files)

    # Attempt to get persisted files
    all_files =
      case call_persistence_list(config) do
        {:ok, persisted_files} ->
          # Merge and deduplicate
          (memory_files ++ persisted_files) |> Enum.uniq()

        {:error, _reason} ->
          # Just use memory files
          memory_files
      end

    # Apply pattern filtering
    filtered_files = filter_by_pattern(all_files, pattern)

    # Sort for consistency
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
  end

  defp execute_read_file_tool(args, context, config) do
    file_path = get_arg(args, "file_path")

    # Validate path
    with {:ok, normalized_path} <- validate_path(file_path) do
      offset = get_arg(args, "offset") || 0
      limit = get_arg(args, "limit") || 2000

      # Try to get content from memory first (using State.get_file handles FileData extraction)
      content =
        case LangChain.DeepAgents.State.get_file(context.state, normalized_path) do
          nil ->
            # Not in memory, try persistence
            case call_persistence_read(config, normalized_path) do
              {:ok, persisted_content} ->
                persisted_content

              {:error, _reason} ->
                nil
            end

          memory_content ->
            memory_content
        end

      execute_read_with_content(content, normalized_path, offset, limit, context, config)
    else
      {:error, reason} -> {:error, reason}
    end
  end

  defp execute_read_with_content(nil, file_path, _offset, _limit, _context, _config) do
    {:error, "File not found: #{file_path}"}
  end

  defp execute_read_with_content(content, file_path, offset, limit, context, config) do
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

    # If we loaded from persistence and caching is enabled, return state delta
    state_delta =
      if get_config_value(config, :cache_reads, true) and
           not Map.has_key?(context.state.files, file_path) do
        # Return only the file change as a state delta
        file_data = LangChain.DeepAgents.State.create_file_data(content)
        %LangChain.DeepAgents.State{files: %{file_path => file_data}}
      else
        nil
      end

    if state_delta do
      {:ok, result, state_delta}
    else
      {:ok, result}
    end
  end

  defp execute_write_file_tool(args, context, config) do
    file_path = get_arg(args, "file_path")
    content = get_arg(args, "content")

    cond do
      is_nil(file_path) or is_nil(content) ->
        {:error, "Both file_path and content are required"}

      true ->
        # Validate path
        with {:ok, normalized_path} <- validate_path(file_path) do
          # Check if file already exists (overwrite protection)
          file_data = LangChain.DeepAgents.State.get_file_data(context.state, normalized_path)

          if file_data do
            {:error,
             "File already exists: #{normalized_path}. Use edit_file to modify existing files, or delete it first."}
          else
            # Create file data
            file_data = LangChain.DeepAgents.State.create_file_data(content)
            # Return state delta with only the new file
            state_delta = %LangChain.DeepAgents.State{files: %{normalized_path => file_data}}

            # Attempt persistence if configured
            case call_persistence_write(config, normalized_path, content) do
              {:ok, _metadata} ->
                {:ok, "File created successfully: #{normalized_path}", state_delta}

              {:error, :not_configured} ->
                # No persistence configured, that's fine
                {:ok, "File created successfully: #{normalized_path}", state_delta}

              {:error, reason} ->
                if get_config_value(config, :fail_on_persistence_error, false) do
                  {:error, "Failed to persist file: #{reason}"}
                else
                  Logger.warning("File persistence failed for #{normalized_path}: #{reason}")

                  {:ok, "File created in memory: #{normalized_path} (persistence warning)",
                   state_delta}
                end
            end
          end
        else
          {:error, reason} -> {:error, reason}
        end
    end
  end

  defp execute_edit_file_tool(args, context, config) do
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
          # Get file content using State.get_file (handles FileData extraction)
          case LangChain.DeepAgents.State.get_file(context.state, normalized_path) do
            nil ->
              {:error, "File not found: #{normalized_path}"}

            content ->
              perform_edit(
                context.state,
                normalized_path,
                content,
                old_string,
                new_string,
                replace_all,
                config
              )
          end
        else
          {:error, reason} -> {:error, reason}
        end
    end
  end

  defp perform_edit(state, file_path, content, old_string, new_string, replace_all, config) do
    # Split to count occurrences
    parts = String.split(content, old_string, parts: :infinity)
    occurrence_count = length(parts) - 1

    cond do
      occurrence_count == 0 ->
        {:error, "String not found in file: '#{old_string}'"}

      occurrence_count == 1 ->
        # Single occurrence, safe to replace
        updated_content = String.replace(content, old_string, new_string, global: false)

        persist_edit(
          state,
          file_path,
          updated_content,
          "File edited successfully: #{file_path}",
          config
        )

      occurrence_count > 1 and not replace_all ->
        {:error,
         "String appears #{occurrence_count} times in file. Use replace_all: true or provide more context in old_string."}

      occurrence_count > 1 and replace_all ->
        # Replace all occurrences
        updated_content = String.replace(content, old_string, new_string, global: true)

        persist_edit(
          state,
          file_path,
          updated_content,
          "File edited successfully: #{file_path} (#{occurrence_count} replacements)",
          config
        )
    end
  end

  defp persist_edit(state, file_path, updated_content, success_message, config) do
    # Get existing file data to preserve created_at
    existing_file_data = LangChain.DeepAgents.State.get_file_data(state, file_path)
    # Update file data with new content
    file_data = LangChain.DeepAgents.State.update_file_data(updated_content, existing_file_data)
    # Return state delta with only the updated file
    state_delta = %LangChain.DeepAgents.State{files: %{file_path => file_data}}

    # Attempt persistence if configured
    case call_persistence_write(config, file_path, updated_content) do
      {:ok, _metadata} ->
        {:ok, success_message, state_delta}

      {:error, :not_configured} ->
        # No persistence configured, that's fine
        {:ok, success_message, state_delta}

      {:error, reason} ->
        if get_config_value(config, :fail_on_persistence_error, false) do
          {:error, "Failed to persist edit: #{reason}"}
        else
          Logger.warning("File edit persistence failed for #{file_path}: #{reason}")
          {:ok, success_message <> " (persistence warning)", state_delta}
        end
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

  defp get_config_value(config, key, default) when is_map(config) do
    Map.get(config, key, default)
  end

  defp get_config_value(_config, _key, default), do: default

  # Persistence helper functions

  defp call_persistence_write(config, file_path, content) do
    callback = get_callback(config, :on_write)

    case callback do
      nil ->
        {:error, :not_configured}

      fun when is_function(fun, 3) ->
        fun.(file_path, content, get_config_value(config, :context, %{}))

      module when is_atom(module) ->
        try do
          if function_exported?(module, :on_write, 3) do
            module.on_write(file_path, content, get_config_value(config, :context, %{}))
          else
            {:error, :not_configured}
          end
        rescue
          e ->
            Logger.warning("Persistence write error: #{Exception.message(e)}")
            {:error, Exception.message(e)}
        end
    end
  rescue
    e ->
      Logger.warning("Persistence write error: #{Exception.message(e)}")
      {:error, Exception.message(e)}
  end

  defp call_persistence_read(config, file_path) do
    callback = get_callback(config, :on_read)

    case callback do
      nil ->
        {:error, :not_configured}

      fun when is_function(fun, 2) ->
        fun.(file_path, get_config_value(config, :context, %{}))

      module when is_atom(module) ->
        try do
          if function_exported?(module, :on_read, 2) do
            module.on_read(file_path, get_config_value(config, :context, %{}))
          else
            {:error, :not_configured}
          end
        rescue
          e ->
            Logger.warning("Persistence read error: #{Exception.message(e)}")
            {:error, Exception.message(e)}
        end
    end
  rescue
    e ->
      Logger.warning("Persistence read error: #{Exception.message(e)}")
      {:error, Exception.message(e)}
  end

  defp call_persistence_list(config) do
    callback = get_callback(config, :on_list)

    case callback do
      nil ->
        {:error, :not_configured}

      fun when is_function(fun, 1) ->
        fun.(get_config_value(config, :context, %{}))

      module when is_atom(module) ->
        try do
          if function_exported?(module, :on_list, 1) do
            module.on_list(get_config_value(config, :context, %{}))
          else
            {:error, :not_configured}
          end
        rescue
          e ->
            Logger.warning("Persistence list error: #{Exception.message(e)}")
            {:error, Exception.message(e)}
        end
    end
  rescue
    e ->
      Logger.warning("Persistence list error: #{Exception.message(e)}")
      {:error, :not_configured}
  end

  defp get_callback(config, callback_name) when is_map(config) do
    # Check for explicit callback function first
    Map.get(config, callback_name) || Map.get(config, :persistence)
  end

  defp get_callback(_config, _callback_name), do: nil

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

      String.starts_with?(path, "/") ->
        {:error,
         "Paths starting with '/' are not allowed. Use relative paths like 'file.txt' or 'dir/file.txt'"}

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
    |> String.trim("/")
  end

  @doc false
  def get_custom_description(config, tool_name, default_description) do
    custom_descriptions = Map.get(config, :custom_tool_descriptions, %{})
    Map.get(custom_descriptions, tool_name, default_description)
  end
end
