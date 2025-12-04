defmodule LangChain.Persistence do
  @moduledoc """
  Behavior for persisting LangChain conversations with scope-based access control.

  This behavior follows Phoenix 1.8's Scopes pattern to ensure secure data access
  by default. All conversation-related operations require a scope that defines
  the access control context (user, organization, team, etc.).

  ## Scopes

  A scope is a struct that contains the access control context for the current
  request or operation. The scope is passed as the first argument to all
  conversation-related functions.

  Example scope:

      defmodule MyApp.Accounts.Scope do
        defstruct [:current_user_id, :current_user]
      end

  ## Implementation

  Use `mix langchain.gen.persistence` to generate a starting implementation
  with a default scope and context module.

  ## Security

  All conversation CRUD operations MUST be scoped to prevent broken access control,
  which is listed as a top-10 security risk by OWASP. The scope pattern ensures
  that queries are automatically filtered to the appropriate user, organization,
  or team context.
  """

  @doc """
  Creates a new conversation within the given scope.

  The conversation will be automatically associated with the scope's owner
  (e.g., current_user_id, organization_id, team_id).
  """
  @callback create_conversation(scope :: term(), attrs :: map()) ::
              {:ok, conversation :: term()} | {:error, Ecto.Changeset.t()}

  @doc """
  Gets a conversation by ID, scoped to the given context.

  Raises if the conversation doesn't exist or doesn't belong to the scope.
  """
  @callback get_conversation!(scope :: term(), id :: binary()) ::
              term() | no_return()

  @doc """
  Lists all conversations accessible within the given scope.

  For single-user scopes, returns only that user's conversations.
  For organization scopes, may return all conversations in that organization.
  """
  @callback list_conversations(scope :: term(), opts :: keyword()) :: [term()]

  @doc """
  Updates a conversation. Verifies it belongs to the given scope.

  Raises if the conversation doesn't exist or doesn't belong to the scope.
  """
  @callback update_conversation(scope :: term(), id :: binary(), attrs :: map()) ::
              {:ok, term()} | {:error, Ecto.Changeset.t()}

  @doc """
  Deletes a conversation. Verifies it belongs to the given scope.

  Raises if the conversation doesn't exist or doesn't belong to the scope.
  """
  @callback delete_conversation(scope :: term(), id :: binary()) ::
              {:ok, term()} | {:error, Ecto.Changeset.t()}

  @doc """
  Saves agent state for a conversation.

  The conversation_id is sufficient here as agent state doesn't have
  separate ownership - it belongs to the conversation.
  """
  @callback save_agent_state(conversation_id :: binary(), state :: map()) ::
              {:ok, term()} | {:error, Ecto.Changeset.t()}

  @doc """
  Loads the current agent state for a conversation.

  Returns `{:error, :not_found}` if no state exists.
  """
  @callback load_agent_state(conversation_id :: binary()) ::
              {:ok, map()} | {:error, :not_found}

  @doc """
  Appends a display message to the conversation.

  Messages belong to the conversation and don't need separate scoping.
  """
  @callback append_display_message(conversation_id :: binary(), message :: map()) ::
              {:ok, term()} | {:error, Ecto.Changeset.t()}

  @doc """
  Loads all display messages for a conversation.

  Options may include :limit, :offset for pagination.
  """
  @callback load_display_messages(conversation_id :: binary(), opts :: keyword()) ::
              [term()]

  @doc """
  Serializes agent state to a persistable format.

  Optional callback - defaults to LangChain.Persistence.StateSerializer
  """
  @callback serialize_agent_state(agent_state :: term()) :: map()

  @doc """
  Deserializes agent state from persisted format.

  Optional callback - defaults to LangChain.Persistence.StateSerializer
  """
  @callback deserialize_agent_state(data :: map()) :: term()

  @optional_callbacks [serialize_agent_state: 1, deserialize_agent_state: 1]

  @doc """
  Check if persistence is enabled.

  Returns `true` if a persistence adapter is configured, `false` otherwise.

  ## Examples

      if LangChain.Persistence.enabled?() do
        # Save state
      end
  """
  def enabled? do
    !is_nil(get_adapter())
  end

  @doc """
  Get the configured persistence adapter module.

  Returns the adapter module or `nil` if not configured.

  ## Examples

      adapter = LangChain.Persistence.get_adapter()
      adapter.save_agent_state(conversation_id, state)
  """
  def get_adapter do
    case Application.get_env(:langchain, :persistence, []) do
      config when is_list(config) ->
        Keyword.get(config, :adapter)

      _ ->
        nil
    end
  end

  @doc """
  Check if auto-save is enabled.

  Returns `true` if auto_save is configured to `true`, `false` otherwise.

  ## Examples

      if LangChain.Persistence.auto_save?() do
        # Automatically save after each message
      end
  """
  def auto_save? do
    case Application.get_env(:langchain, :persistence, []) do
      config when is_list(config) ->
        Keyword.get(config, :auto_save, false)

      _ ->
        false
    end
  end

  @doc """
  Get the configured agent snapshot frequency.

  Returns the frequency (number of messages) or `nil` if not configured.

  The snapshot frequency determines how often full agent state snapshots
  are saved. For example, if set to 10, a full snapshot will be saved
  every 10 messages.

  ## Examples

      frequency = LangChain.Persistence.agent_snapshot_frequency()
      # => 10
  """
  def agent_snapshot_frequency do
    case Application.get_env(:langchain, :persistence, []) do
      config when is_list(config) ->
        Keyword.get(config, :agent_snapshot_frequency)

      _ ->
        nil
    end
  end
end
