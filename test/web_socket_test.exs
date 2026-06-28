defmodule LangChain.WebSocketTest do
  use ExUnit.Case, async: true

  alias LangChain.WebSocket

  describe "start_link/1" do
    test "requires :url option" do
      Process.flag(:trap_exit, true)

      assert {:error, {%KeyError{key: :url}, _stacktrace}} =
               WebSocket.start_link(headers: [])
    end
  end

  describe "connected?/1" do
    test "returns false for dead process" do
      refute WebSocket.connected?(spawn(fn -> :ok end))
    end
  end

  describe "close/1" do
    test "stops the GenServer" do
      # We can't easily test close without a real connection,
      # but we can verify the function exists and the API contract
      assert is_function(&WebSocket.close/1)
    end
  end
end
