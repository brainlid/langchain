defmodule LangChain.Presence do
  @moduledoc """
  Convenience wrappers for Phoenix.Presence operations.

  This module provides thin wrappers around Phoenix.Presence to make presence tracking
  more convenient in agent-based applications. Phoenix.Presence automatically cleans up
  presence entries when the tracked process terminates, so no manual cleanup is needed.

  ## Examples

      # Track presence in a LiveView mount
      if connected?(socket) do
        {:ok, ref} = LangChain.Presence.track(
          MyApp.Presence,
          "conversation:123",
          "user-1",
          self(),
          %{name: "Alice"}
        )
      end

      # Track presence in multiple topics from the same process
      LangChain.Presence.track(MyApp.Presence, "conversation:123", "user-1", self())
      LangChain.Presence.track(MyApp.Presence, "conversation:456", "user-1", self())
  """

  @doc """
  Track presence for the given topic and identifier.

  Phoenix.Presence automatically removes the entry when the tracked process terminates,
  so manual cleanup is typically not needed.

  ## Parameters

    - `presence_module` - The Presence module (e.g., MyApp.Presence)
    - `topic` - The topic string for presence tracking
    - `id` - Unique identifier for this presence entry (e.g., user_id)
    - `pid` - The process to track (typically self())
    - `metadata` - Optional metadata map (default: empty map)

  ## Returns

    - `{:ok, ref}` - Presence tracked successfully
    - `{:error, reason}` - Failed to track presence

  ## Examples

      # In a LiveView after connected
      {:ok, ref} = LangChain.Presence.track(
        MyApp.Presence,
        "conversation:123",
        "user-1",
        self(),
        %{joined_at: System.system_time(:second)}
      )

      # Track in multiple topics
      {:ok, _} = LangChain.Presence.track(MyApp.Presence, "topic:123", "user-1", self())
      {:ok, _} = LangChain.Presence.track(MyApp.Presence, "topic:456", "user-1", self())
  """
  def track(presence_module, topic, id, pid, metadata \\ %{}) do
    presence_module.track(pid, topic, id, metadata)
  end

  @doc """
  Untrack presence for the given topic and identifier.

  Note: This is rarely needed since Phoenix.Presence automatically cleans up
  when the tracked process terminates. Only use this for explicit early cleanup.
  """
  def untrack(presence_module, topic, id, pid) do
    presence_module.untrack(pid, topic, id)
  end

  @doc """
  List all presence entries for a topic.

  This is a convenience wrapper around the presence module's list function.
  """
  def list(presence_module, topic) do
    presence_module.list(topic)
  end
end
