defmodule LangChain.Persistence.Hooks do
  @moduledoc """
  Optional behavior for hooking into agent state persistence events.

  Implement this behavior to automatically save state at key points
  in the agent lifecycle without explicit calls.

  ## Usage

  Define a module implementing this behavior:

      defmodule MyApp.Conversations.Hooks do
        @behaviour LangChain.Persistence.Hooks

        @impl true
        def after_message_received(conversation_id, message) do
          MyApp.Conversations.append_display_message(conversation_id, message)
        end

        @impl true
        def after_agent_response(conversation_id, agent_state) do
          MyApp.Conversations.save_agent_state(conversation_id, agent_state)
        end

        @impl true
        def after_status_change(conversation_id, status, data) do
          # Optional: track status changes
          :ok
        end
      end

  Then configure it in your application:

      # config/config.exs
      config :langchain, :persistence,
        adapter: MyApp.Conversations,
        hooks: MyApp.Conversations.Hooks,
        auto_save: true

  ## Hook Callbacks

  All callbacks are optional. If a callback is not defined, the hook
  will be skipped silently.

  ## Error Handling

  Hook callbacks should return `:ok` or `{:ok, term()}` on success.
  Errors will be logged but won't interrupt the agent execution.
  """

  @doc """
  Called after a user message is received.

  This is typically used to save the user's message to the display
  messages table.

  ## Parameters

  - `conversation_id` - The conversation identifier
  - `message` - The message struct

  ## Returns

  `:ok` or `{:ok, term()}` on success. Errors are logged but don't interrupt execution.
  """
  @callback after_message_received(conversation_id :: binary(), message :: term()) ::
              :ok | {:ok, term()} | {:error, term()}

  @doc """
  Called after the agent produces a response.

  This is typically used to:
  - Save the agent's response message to display messages
  - Save the updated agent state

  ## Parameters

  - `conversation_id` - The conversation identifier
  - `agent_state` - The complete agent state (map with string keys from export_state/1)

  ## Returns

  `:ok` or `{:ok, term()}` on success. Errors are logged but don't interrupt execution.
  """
  @callback after_agent_response(conversation_id :: binary(), agent_state :: map()) ::
              :ok | {:ok, term()} | {:error, term()}

  @doc """
  Called when the agent status changes.

  This can be used to track agent lifecycle events like:
  - `:idle` - Ready for work
  - `:running` - Executing
  - `:interrupted` - Awaiting human decision
  - `:completed` - Execution finished
  - `:error` - Execution failed

  ## Parameters

  - `conversation_id` - The conversation identifier
  - `status` - The new status (`:idle`, `:running`, `:interrupted`, `:completed`, `:error`)
  - `data` - Additional data (interrupt_data for `:interrupted`, error for `:error`, state for `:completed`)

  ## Returns

  `:ok` or `{:ok, term()}` on success. Errors are logged but don't interrupt execution.
  """
  @callback after_status_change(
              conversation_id :: binary(),
              status :: atom(),
              data :: term()
            ) ::
              :ok | {:ok, term()} | {:error, term()}

  @optional_callbacks [after_message_received: 2, after_agent_response: 2, after_status_change: 3]

  @doc """
  Invoke a hook callback if hooks are configured.

  This is a convenience function for the framework to call hooks
  without requiring the hooks module to be defined.

  ## Examples

      # In AgentServer
      LangChain.Persistence.Hooks.invoke(:after_message_received, [conversation_id, message])
  """
  def invoke(callback_name, args) when is_atom(callback_name) and is_list(args) do
    case get_hooks_module() do
      nil ->
        :ok

      hooks_module ->
        if function_exported?(hooks_module, callback_name, length(args)) do
          try do
            apply(hooks_module, callback_name, args)
          rescue
            error ->
              require Logger

              Logger.error(
                "Hook callback #{inspect(hooks_module)}.#{callback_name} failed: #{inspect(error)}"
              )

              {:error, error}
          end
        else
          :ok
        end
    end
  end

  @doc """
  Check if hooks are enabled.

  Returns `true` if a hooks module is configured, `false` otherwise.
  """
  def enabled? do
    !is_nil(get_hooks_module())
  end

  @doc """
  Get the configured hooks module.

  Returns the module or `nil` if not configured.
  """
  def get_hooks_module do
    case Application.get_env(:langchain, :persistence, []) do
      config when is_list(config) ->
        Keyword.get(config, :hooks)

      _ ->
        nil
    end
  end
end
