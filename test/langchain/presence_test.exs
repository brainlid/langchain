defmodule LangChain.PresenceTest do
  use ExUnit.Case, async: true

  alias LangChain.Presence

  # Mock Presence module for testing
  defmodule MockPresence do
    def track(pid, topic, id, metadata) do
      send(pid, {:tracked, topic, id, metadata})
      {:ok, make_ref()}
    end

    def untrack(pid, topic, id) do
      send(pid, {:untracked, topic, id})
      :ok
    end

    def list(_topic) do
      # Return a simple map for testing
      %{
        "user-1" => %{metas: [%{joined_at: System.system_time(:second)}]},
        "user-2" => %{metas: [%{joined_at: System.system_time(:second)}]}
      }
    end
  end

  describe "track/5" do
    test "tracks presence" do
      topic = "conversation:1"
      id = "user-1"
      metadata = %{name: "Alice"}

      assert {:ok, ref} = Presence.track(MockPresence, topic, id, self(), metadata)
      assert is_reference(ref)
      assert_received {:tracked, ^topic, ^id, ^metadata}
    end

    test "allows tracking multiple presences with different topics" do
      topic1 = "conversation:1"
      topic2 = "conversation:2"
      id = "user-1"

      # Track in first topic
      assert {:ok, _ref1} = Presence.track(MockPresence, topic1, id, self())
      assert_received {:tracked, ^topic1, ^id, _}

      # Track in second topic
      assert {:ok, _ref2} = Presence.track(MockPresence, topic2, id, self())
      assert_received {:tracked, ^topic2, ^id, _}
    end

    test "allows tracking multiple presences with different ids in same topic" do
      topic = "conversation:1"
      id1 = "user-1"
      id2 = "user-2"

      # Track first user
      assert {:ok, _ref1} = Presence.track(MockPresence, topic, id1, self())
      assert_received {:tracked, ^topic, ^id1, _}

      # Track second user
      assert {:ok, _ref2} = Presence.track(MockPresence, topic, id2, self())
      assert_received {:tracked, ^topic, ^id2, _}
    end

    test "uses default empty metadata if not provided" do
      topic = "conversation:1"
      id = "user-1"

      assert {:ok, _ref} = Presence.track(MockPresence, topic, id, self())
      assert_received {:tracked, ^topic, ^id, metadata}
      assert metadata == %{}
    end

    test "accepts custom metadata" do
      topic = "conversation:1"
      id = "user-1"
      metadata = %{name: "Charlie", role: "admin"}

      assert {:ok, _ref} = Presence.track(MockPresence, topic, id, self(), metadata)
      assert_received {:tracked, ^topic, ^id, ^metadata}
    end
  end

  describe "untrack/4" do
    test "untracks presence" do
      topic = "conversation:1"
      id = "user-1"

      # First track
      Presence.track(MockPresence, topic, id, self())
      assert_received {:tracked, ^topic, ^id, _}

      # Then untrack
      assert :ok = Presence.untrack(MockPresence, topic, id, self())
      assert_received {:untracked, ^topic, ^id}
    end

    test "can retrack after untracking" do
      topic = "conversation:1"
      id = "user-1"

      # Track
      Presence.track(MockPresence, topic, id, self())
      assert_received {:tracked, ^topic, ^id, _}

      # Untrack
      Presence.untrack(MockPresence, topic, id, self())
      assert_received {:untracked, ^topic, ^id}

      # Track again
      Presence.track(MockPresence, topic, id, self())
      assert_received {:tracked, ^topic, ^id, _}
    end
  end

  describe "list/2" do
    test "returns presence list from presence module" do
      topic = "conversation:1"

      result = Presence.list(MockPresence, topic)

      assert is_map(result)
      assert Map.has_key?(result, "user-1")
      assert Map.has_key?(result, "user-2")
    end
  end
end
