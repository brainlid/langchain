defmodule LangChain.Agents.MiddlewareEntry do
  @moduledoc """
  Represents a registered middleware instance with its configuration.

  Each middleware entry contains:
  - `id`: Unique identifier (module name by default, or custom string)
  - `module`: The middleware module implementing the Middleware behavior
  - `config`: Configuration map returned from middleware's init/1 callback

  ## ID Determination

  The middleware ID is used for message routing and uniquely identifies
  each middleware instance:

  - **Default**: Module name (e.g., `LangChain.Middleware.TodoList`)
  - **Custom**: Specified via `:id` option for multiple instances of the same middleware

  ## Examples

      # Single instance with default ID (module name)
      %MiddlewareEntry{
        id: LangChain.Middleware.TodoList,
        module: LangChain.Middleware.TodoList,
        config: %{middleware_id: LangChain.Middleware.TodoList}
      }

      # Multiple instances with custom IDs
      %MiddlewareEntry{
        id: "title_generator_en",
        module: LangChain.Middleware.ConversationTitle,
        config: %{middleware_id: "title_generator_en", language: "en"}
      }
  """

  @type t :: %__MODULE__{
          id: atom() | String.t(),
          module: module(),
          config: map()
        }

  defstruct [:id, :module, :config]

  @doc """
  Convert a MiddlewareEntry struct back to raw middleware specification.

  This is useful when middleware needs to be re-initialized in a new context,
  such as when creating a SubAgent that inherits parent middleware.

  ## Parameters

  - `entry` - A MiddlewareEntry struct or raw middleware spec

  ## Returns

  - Module atom if the middleware has no configuration options
  - `{module, opts}` tuple if the middleware has configuration options
  - Passes through non-MiddlewareEntry values unchanged

  ## Examples

      # Entry with no configuration
      entry = %MiddlewareEntry{
        module: MyMiddleware,
        config: %{id: MyMiddleware}
      }
      to_raw_spec(entry)
      # => MyMiddleware

      # Entry with configuration
      entry = %MiddlewareEntry{
        module: MyMiddleware,
        config: %{id: MyMiddleware, max_items: 100, enabled: true}
      }
      to_raw_spec(entry)
      # => {MyMiddleware, [max_items: 100, enabled: true]}

      # Pass through raw specs unchanged
      to_raw_spec(MyMiddleware)
      # => MyMiddleware

      to_raw_spec({MyMiddleware, [opt: "value"]})
      # => {MyMiddleware, [opt: "value"]}
  """
  @spec to_raw_spec(t() | module() | {module(), keyword()}) :: module() | {module(), keyword()}
  def to_raw_spec(%__MODULE__{module: module, config: config}) when is_map(config) do
    # Remove internal keys that are added during initialization
    # and shouldn't be passed back to init/1
    opts =
      config
      |> Map.drop([:id, :middleware_id])
      |> Map.to_list()

    # Return just the module if no options remain
    if opts == [] do
      module
    else
      {module, opts}
    end
  end

  # Pass through raw specs that are already in the correct format
  def to_raw_spec(module) when is_atom(module), do: module
  def to_raw_spec({module, opts} = spec) when is_atom(module) and is_list(opts), do: spec

  @doc """
  Convert a list of MiddlewareEntry structs to raw middleware specifications.

  Convenience function for converting entire middleware lists.

  ## Parameters

  - `entries` - List of MiddlewareEntry structs and/or raw middleware specs

  ## Returns

  - List of raw middleware specifications (module atoms or {module, opts} tuples)

  ## Examples

      entries = [
        %MiddlewareEntry{module: Middleware1, config: %{id: Middleware1}},
        %MiddlewareEntry{module: Middleware2, config: %{id: Middleware2, opt: "value"}},
        Middleware3
      ]
      to_raw_specs(entries)
      # => [Middleware1, {Middleware2, [opt: "value"]}, Middleware3]
  """
  @spec to_raw_specs([t() | module() | {module(), keyword()}]) :: [
          module() | {module(), keyword()}
        ]
  def to_raw_specs(entries) when is_list(entries) do
    Enum.map(entries, &to_raw_spec/1)
  end
end
