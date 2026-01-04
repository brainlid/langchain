defmodule LangChain.PubSub do
  @moduledoc """
  Helper functions for idempotent PubSub subscriptions.

  This module provides utilities for managing PubSub subscriptions with automatic
  deduplication using the Process dictionary. Subscriptions are tracked per-process,
  supporting multiple subscriptions to different topics from a single process.

  ## Examples

      # Subscribe to a topic (idempotent)
      LangChain.PubSub.subscribe(Phoenix.PubSub, MyApp.PubSub, "topic:123")
      LangChain.PubSub.subscribe(Phoenix.PubSub, MyApp.PubSub, "topic:123")  # No-op

      # Subscribe to multiple topics from same process
      LangChain.PubSub.subscribe(Phoenix.PubSub, MyApp.PubSub, "topic:123")
      LangChain.PubSub.subscribe(Phoenix.PubSub, MyApp.PubSub, "topic:456")

      # Unsubscribe
      LangChain.PubSub.unsubscribe(Phoenix.PubSub, MyApp.PubSub, "topic:123")
  """

  @doc """
  Subscribe the current process to a PubSub topic with automatic deduplication.

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
      :ok = LangChain.PubSub.subscribe(Phoenix.PubSub, MyApp.PubSub, "agent:123")

      # Multiple calls - no duplicate subscription
      :ok = LangChain.PubSub.subscribe(Phoenix.PubSub, MyApp.PubSub, "agent:123")

      # Multiple topics from same process
      :ok = LangChain.PubSub.subscribe(Phoenix.PubSub, MyApp.PubSub, "agent:123")
      :ok = LangChain.PubSub.subscribe(Phoenix.PubSub, MyApp.PubSub, "agent:456")
  """
  def subscribe(pubsub_module, pubsub_name, topic) do
    subscription_key = {:pubsub_subscription, topic}

    unless Process.get(subscription_key) do
      :ok = pubsub_module.subscribe(pubsub_name, topic)
      Process.put(subscription_key, true)
    end

    :ok
  end

  @doc """
  Subscribe to a PubSub topic without deduplication.

  This performs a raw subscription without tracking in the Process dictionary.
  Most code should use `subscribe/3` instead for automatic deduplication.

  Only use this function when you explicitly need to bypass deduplication,
  such as testing or special cases where multiple subscriptions are acceptable.
  """
  def raw_subscribe(pubsub_module, pubsub_name, topic) do
    pubsub_module.subscribe(pubsub_name, topic)
  end

  @doc """
  Unsubscribe the current process from a PubSub topic with proper cleanup.

  This clears the subscription tracking in the Process dictionary and performs
  the actual unsubscription. This is the standard way to unsubscribe.

  ## Parameters

    - `pubsub_module` - The PubSub module (e.g., Phoenix.PubSub)
    - `pubsub_name` - The PubSub server name (e.g., MyApp.PubSub)
    - `topic` - The topic string to unsubscribe from

  ## Returns

    - `:ok` on success
  """
  def unsubscribe(pubsub_module, pubsub_name, topic) do
    :ok = pubsub_module.unsubscribe(pubsub_name, topic)

    subscription_key = {:pubsub_subscription, topic}
    Process.delete(subscription_key)

    :ok
  end

  @doc """
  Unsubscribe from a PubSub topic without clearing Process dictionary tracking.

  This performs a raw unsubscription without clearing the tracking state.
  Most code should use `unsubscribe/3` instead for proper cleanup.

  Only use this function when you explicitly need to bypass cleanup,
  such as testing or special cases.
  """
  def raw_unsubscribe(pubsub_module, pubsub_name, topic) do
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
