defmodule LangChain.PubSubTest do
  use ExUnit.Case, async: true

  alias LangChain.PubSub

  # Mock PubSub module for testing
  defmodule MockPubSub do
    def subscribe(_pubsub_name, topic) do
      send(self(), {:subscribed, topic})
      :ok
    end

    def unsubscribe(_pubsub_name, topic) do
      send(self(), {:unsubscribed, topic})
      :ok
    end
  end

  setup do
    # Clear Process dictionary before each test
    Process.get_keys()
    |> Enum.filter(&match?({:pubsub_subscription, _}, &1))
    |> Enum.each(&Process.delete/1)

    :ok
  end

  describe "ensure_subscribe/3" do
    test "subscribes to topic on first call" do
      topic = "test:topic1"

      assert :ok = PubSub.ensure_subscribe(MockPubSub, :test_pubsub, topic)
      assert_received {:subscribed, ^topic}
    end

    test "is idempotent - does not subscribe twice" do
      topic = "test:topic2"

      # First call subscribes
      assert :ok = PubSub.ensure_subscribe(MockPubSub, :test_pubsub, topic)
      assert_received {:subscribed, ^topic}

      # Second call is a no-op
      assert :ok = PubSub.ensure_subscribe(MockPubSub, :test_pubsub, topic)
      refute_received {:subscribed, ^topic}
    end

    test "tracks subscription in Process dictionary" do
      topic = "test:topic3"

      refute PubSub.subscribed?(topic)

      PubSub.ensure_subscribe(MockPubSub, :test_pubsub, topic)

      assert PubSub.subscribed?(topic)
    end

    test "allows subscribing to multiple topics" do
      topic1 = "test:topic4"
      topic2 = "test:topic5"

      # Subscribe to first topic
      assert :ok = PubSub.ensure_subscribe(MockPubSub, :test_pubsub, topic1)
      assert_received {:subscribed, ^topic1}
      assert PubSub.subscribed?(topic1)

      # Subscribe to second topic
      assert :ok = PubSub.ensure_subscribe(MockPubSub, :test_pubsub, topic2)
      assert_received {:subscribed, ^topic2}
      assert PubSub.subscribed?(topic2)

      # Both subscriptions are tracked independently
      assert PubSub.subscribed?(topic1)
      assert PubSub.subscribed?(topic2)
    end

    test "multiple calls to same topic only subscribe once" do
      topic = "test:topic6"

      # Call three times
      assert :ok = PubSub.ensure_subscribe(MockPubSub, :test_pubsub, topic)
      assert :ok = PubSub.ensure_subscribe(MockPubSub, :test_pubsub, topic)
      assert :ok = PubSub.ensure_subscribe(MockPubSub, :test_pubsub, topic)

      # Should only receive one subscription message
      assert_received {:subscribed, ^topic}
      refute_received {:subscribed, ^topic}
      refute_received {:subscribed, ^topic}
    end
  end

  describe "subscribe/3" do
    test "subscribes without tracking in Process dictionary" do
      topic = "test:topic7"

      assert :ok = PubSub.subscribe(MockPubSub, :test_pubsub, topic)
      assert_received {:subscribed, ^topic}

      # Should not be tracked
      refute PubSub.subscribed?(topic)
    end
  end

  describe "ensure_unsubscribe/3" do
    test "unsubscribes and clears Process dictionary tracking" do
      topic = "test:topic8"

      # First subscribe
      PubSub.ensure_subscribe(MockPubSub, :test_pubsub, topic)
      assert PubSub.subscribed?(topic)

      # Then unsubscribe
      assert :ok = PubSub.ensure_unsubscribe(MockPubSub, :test_pubsub, topic)
      assert_received {:unsubscribed, ^topic}

      # Should no longer be tracked
      refute PubSub.subscribed?(topic)
    end

    test "can resubscribe after unsubscribing" do
      topic = "test:topic9"

      # Subscribe
      PubSub.ensure_subscribe(MockPubSub, :test_pubsub, topic)
      assert_received {:subscribed, ^topic}

      # Unsubscribe
      PubSub.ensure_unsubscribe(MockPubSub, :test_pubsub, topic)
      assert_received {:unsubscribed, ^topic}

      # Subscribe again
      PubSub.ensure_subscribe(MockPubSub, :test_pubsub, topic)
      assert_received {:subscribed, ^topic}

      assert PubSub.subscribed?(topic)
    end
  end

  describe "subscribed?/1" do
    test "returns false when not subscribed" do
      refute PubSub.subscribed?("test:topic10")
    end

    test "returns true when subscribed via ensure_subscribe" do
      topic = "test:topic11"

      PubSub.ensure_subscribe(MockPubSub, :test_pubsub, topic)

      assert PubSub.subscribed?(topic)
    end

    test "returns false when subscribed via plain subscribe" do
      topic = "test:topic12"

      PubSub.subscribe(MockPubSub, :test_pubsub, topic)

      # subscribe() doesn't track, so subscribed? returns false
      refute PubSub.subscribed?(topic)
    end
  end
end
