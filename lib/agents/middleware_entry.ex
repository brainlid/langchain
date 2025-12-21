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
end
