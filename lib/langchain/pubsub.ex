defmodule LangChain.PubSub do
  @moduledoc """
  Helper functions for idempotent PubSub subscriptions.

  This module provides utilities for managing PubSub subscriptions with automatic
  deduplication using the Process dictionary. Subscriptions are tracked per-process,
  supporting multiple subscriptions to different topics from a single process.

  ## Examples

      # Subscribe to a topic (idempotent)
      LangChain.PubSub.ensure_subscribe(Phoenix.PubSub, MyApp.PubSub, "topic:123")
      LangChain.PubSub.ensure_subscribe(Phoenix.PubSub, MyApp.PubSub, "topic:123")  # No-op

      # Subscribe to multiple topics from same process
      LangChain.PubSub.ensure_subscribe(Phoenix.PubSub, MyApp.PubSub, "topic:123")
      LangChain.PubSub.ensure_subscribe(Phoenix.PubSub, MyApp.PubSub, "topic:456")

      # Unsubscribe
      LangChain.PubSub.ensure_unsubscribe(Phoenix.PubSub, MyApp.PubSub, "topic:123")
  """

  @doc """
  Ensure the current process is subscribed to a PubSub topic.

  This function is idempotent - safe to call multiple times with the same topic.
  It uses the Process dictionary to track subscriptions and prevent duplicates.

  Multiple topics can be subscribed to from the same process - each topic is
  tracked independently.

  ## Parameters

    - `pubsub_module` - The PubSub module (e.g., Phoenix.PubSub)
    - `pubsub_name` - The PubSub server name (e.g., MyApp.PubSub)
    - `topic` - The topic string to subscribe to

  ## Returns

    - `:ok` on success (whether newly subscribed or already subscribed)

  ## Examples

      # Single subscription
      :ok = LangChain.PubSub.ensure_subscribe(Phoenix.PubSub, MyApp.PubSub, "agent:123")

      # Multiple calls - no duplicate subscription
      :ok = LangChain.PubSub.ensure_subscribe(Phoenix.PubSub, MyApp.PubSub, "agent:123")

      # Multiple topics from same process
      :ok = LangChain.PubSub.ensure_subscribe(Phoenix.PubSub, MyApp.PubSub, "agent:123")
      :ok = LangChain.PubSub.ensure_subscribe(Phoenix.PubSub, MyApp.PubSub, "agent:456")
  """
  def ensure_subscribe(pubsub_module, pubsub_name, topic) do
    subscription_key = {:pubsub_subscription, topic}

    unless Process.get(subscription_key) do
      :ok = pubsub_module.subscribe(pubsub_name, topic)
      Process.put(subscription_key, true)
    end

    :ok
  end

  @doc """
  Subscribe to a PubSub topic without deduplication.

  Use `ensure_subscribe/3` instead for automatic deduplication.
  """
  def subscribe(pubsub_module, pubsub_name, topic) do
    pubsub_module.subscribe(pubsub_name, topic)
  end

  @doc """
  Ensure the current process is unsubscribed from a PubSub topic.

  Clears the subscription tracking in the Process dictionary.
  """
  def ensure_unsubscribe(pubsub_module, pubsub_name, topic) do
    :ok = pubsub_module.unsubscribe(pubsub_name, topic)

    subscription_key = {:pubsub_subscription, topic}
    Process.delete(subscription_key)

    :ok
  end

  @doc """
  Unsubscribe from a PubSub topic without clearing Process dictionary tracking.

  Use `ensure_unsubscribe/3` instead for proper cleanup.
  """
  def unsubscribe(pubsub_module, pubsub_name, topic) do
    pubsub_module.unsubscribe(pubsub_name, topic)
  end

  @doc """
  Check if the current process is subscribed to a topic (according to Process dictionary).

  Note: This only checks the Process dictionary tracking, not the actual PubSub state.
  """
  def subscribed?(topic) do
    subscription_key = {:pubsub_subscription, topic}
    Process.get(subscription_key) != nil
  end
end
