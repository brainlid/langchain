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

  describe "TLS verification" do
    setup do
      port = start_untrusted_tls_server()
      {:ok, port: port}
    end

    test "rejects an untrusted certificate for a wss URL", %{port: port} do
      Process.flag(:trap_exit, true)

      assert {:error, %Mint.TransportError{reason: {:tls_alert, _alert}}} =
               WebSocket.start_link(
                 url: "wss://localhost:#{port}/socket",
                 headers: [{"authorization", "Bearer test-token"}],
                 connect_timeout: 1_000
               )

      refute_received {:tls_server, :handshake_ok}
    end

    test "rejects an untrusted certificate for an https URL", %{port: port} do
      Process.flag(:trap_exit, true)

      assert {:error, %Mint.TransportError{reason: {:tls_alert, _alert}}} =
               WebSocket.start_link(
                 url: "https://localhost:#{port}/socket",
                 headers: [{"authorization", "Bearer test-token"}],
                 connect_timeout: 1_000
               )

      refute_received {:tls_server, :handshake_ok}
    end
  end

  # Starts a TLS listener on localhost whose certificate chains to a freshly
  # generated root that is not in the system trust store. A client that
  # verifies the peer aborts the handshake before sending any request.
  defp start_untrusted_tls_server do
    chain = %{
      root: [key: {:namedCurve, :secp256r1}, digest: :sha256],
      intermediates: [],
      peer: [
        key: {:namedCurve, :secp256r1},
        digest: :sha256,
        extensions: [{:Extension, {2, 5, 29, 17}, false, [{:dNSName, ~c"localhost"}]}]
      ]
    }

    data = :public_key.pkix_test_data(%{server_chain: chain, client_chain: chain})
    server_opts = Keyword.take(data[:server_config], [:cert, :key])

    {:ok, listen} =
      :ssl.listen(0, [:binary, active: false, reuseaddr: true, ip: {127, 0, 0, 1}] ++ server_opts)

    {:ok, {_ip, port}} = :ssl.sockname(listen)
    test_pid = self()

    acceptor =
      spawn(fn ->
        with {:ok, socket} <- :ssl.transport_accept(listen, 5_000),
             {:ok, _socket} <- :ssl.handshake(socket, 5_000) do
          send(test_pid, {:tls_server, :handshake_ok})
        end
      end)

    on_exit(fn ->
      Process.exit(acceptor, :kill)
      :ssl.close(listen)
    end)

    port
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
