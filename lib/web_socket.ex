defmodule LangChain.WebSocket do
  @moduledoc """
  A generic WebSocket client GenServer built on `Mint.WebSocket`.

  Provides a persistent WebSocket connection that can send text frames and
  collect responses. This module is provider-agnostic — it handles connection
  lifecycle, frame encoding/decoding, and ping/pong, but has no knowledge of
  any specific API protocol.

  ## Usage

      {:ok, ws} = LangChain.WebSocket.start_link(
        url: "wss://api.openai.com/v1/responses",
        headers: [{"authorization", "Bearer sk-..."}]
      )

      # Send a request and collect events until done_fn returns true
      done_fn = fn event -> event["type"] == "response.completed" end
      {:ok, events} = LangChain.WebSocket.send_and_collect(ws, payload, done_fn)

      # When finished
      LangChain.WebSocket.close(ws)

  ## Options for `start_link/1`

  - `:url` (required) — WebSocket URL (e.g. `"wss://example.com/ws"`)
  - `:headers` — additional HTTP headers for the upgrade request (default: `[]`)
  - `:receive_timeout` — timeout in ms for receiving responses (default: `60_000`)
  - `:connect_timeout` — timeout in ms for initial connection (default: `10_000`)
  """

  use GenServer
  require Logger

  @receive_timeout 60_000
  @connect_timeout 10_000

  defstruct [
    :conn,
    :websocket,
    :ref,
    :url,
    :headers,
    :caller,
    :receive_timeout,
    :connect_timeout,
    status: :disconnected,
    buffer: ""
  ]

  # -- Public API --

  @doc """
  Start a WebSocket connection.

  ## Options

  - `:url` (required) — WebSocket URL
  - `:headers` — HTTP headers for the upgrade request (default: `[]`)
  - `:receive_timeout` — timeout for collecting responses in ms (default: `60_000`)
  - `:connect_timeout` — timeout for initial connection in ms (default: `10_000`)

  Any other options are passed through to `GenServer.start_link/3` (e.g. `:name`).
  """
  @spec start_link(keyword()) :: GenServer.on_start()
  def start_link(opts) do
    {ws_opts, genserver_opts} =
      Keyword.split(opts, [:url, :headers, :receive_timeout, :connect_timeout])

    GenServer.start_link(__MODULE__, ws_opts, genserver_opts)
  end

  @doc """
  Send a text frame and collect all decoded JSON events until `done_fn` returns true.

  Returns `{:ok, [decoded_events]}` on success.

  ## Options

  - `:timeout` — GenServer call timeout in ms (default: the configured `receive_timeout`)
  """
  @spec send_and_collect(GenServer.server(), binary(), (map() -> boolean()), keyword()) ::
          {:ok, [map()]} | {:error, term()}
  def send_and_collect(pid, payload, done_fn, opts \\ []) do
    timeout = Keyword.get(opts, :timeout, @receive_timeout)
    GenServer.call(pid, {:send_and_collect, payload, done_fn}, timeout)
  end

  @doc """
  Send a text frame and stream each decoded JSON event to `callback_fn` until
  `done_fn` returns true.

  Returns `{:ok, [callback_results]}` with the return values from each
  `callback_fn` invocation.

  ## Options

  - `:timeout` — GenServer call timeout in ms (default: the configured `receive_timeout`)
  """
  @spec send_and_stream(
          GenServer.server(),
          binary(),
          (map() -> term()),
          (map() -> boolean()),
          keyword()
        ) ::
          {:ok, [term()]} | {:error, term()}
  def send_and_stream(pid, payload, callback_fn, done_fn, opts \\ []) do
    timeout = Keyword.get(opts, :timeout, @receive_timeout)
    GenServer.call(pid, {:send_and_stream, payload, callback_fn, done_fn}, timeout)
  end

  @doc """
  Close the WebSocket connection and stop the GenServer.
  """
  @spec close(GenServer.server()) :: :ok
  def close(pid) do
    GenServer.stop(pid, :normal)
  end

  @doc """
  Check if the WebSocket connection is alive and connected.
  """
  @spec connected?(GenServer.server()) :: boolean()
  def connected?(pid) do
    GenServer.call(pid, :connected?)
  catch
    :exit, _ -> false
  end

  # -- GenServer Callbacks --

  @impl true
  def init(opts) do
    url = Keyword.fetch!(opts, :url)
    headers = Keyword.get(opts, :headers, [])
    receive_timeout = Keyword.get(opts, :receive_timeout, @receive_timeout)
    connect_timeout = Keyword.get(opts, :connect_timeout, @connect_timeout)

    uri = URI.parse(url)

    state = %__MODULE__{
      url: uri,
      headers: headers,
      receive_timeout: receive_timeout,
      connect_timeout: connect_timeout
    }

    case do_connect(state) do
      {:ok, state} ->
        {:ok, state}

      {:error, reason} ->
        {:stop, reason}
    end
  end

  @impl true
  def handle_call(:connected?, _from, state) do
    {:reply, state.status == :connected, state}
  end

  def handle_call({:send_and_collect, payload, done_fn}, from, %{status: :connected} = state) do
    case send_text_frame(state, payload) do
      {:ok, state} ->
        caller = %{from: from, done_fn: done_fn, callback_fn: nil, acc: []}
        {:noreply, %{state | caller: caller}}

      {:error, reason} ->
        {:reply, {:error, reason}, state}
    end
  end

  def handle_call(
        {:send_and_stream, payload, callback_fn, done_fn},
        from,
        %{status: :connected} = state
      ) do
    case send_text_frame(state, payload) do
      {:ok, state} ->
        caller = %{from: from, done_fn: done_fn, callback_fn: callback_fn, acc: []}
        {:noreply, %{state | caller: caller}}

      {:error, reason} ->
        {:reply, {:error, reason}, state}
    end
  end

  def handle_call({tag, _payload, _done_fn}, _from, state)
      when tag in [:send_and_collect, :send_and_stream] do
    {:reply, {:error, :not_connected}, state}
  end

  def handle_call({tag, _payload, _callback_fn, _done_fn}, _from, state)
      when tag in [:send_and_stream] do
    {:reply, {:error, :not_connected}, state}
  end

  @impl true
  def handle_info(message, state) do
    case Mint.WebSocket.stream(state.conn, message) do
      {:ok, conn, responses} ->
        state = %{state | conn: conn}
        handle_responses(state, responses)

      {:error, conn, reason, _responses} ->
        Logger.error("WebSocket stream error: #{inspect(reason)}")
        state = %{state | conn: conn, status: :disconnected}
        maybe_reply_error(state, {:error, reason})

      :unknown ->
        {:noreply, state}
    end
  end

  @impl true
  def terminate(_reason, %{conn: conn, websocket: ws, ref: ref} = _state)
      when not is_nil(ws) do
    # Try to send a close frame gracefully
    with {:ok, _ws, data} <- Mint.WebSocket.encode(ws, :close),
         {:ok, conn} <- Mint.WebSocket.stream_request_body(conn, ref, data) do
      Mint.HTTP.close(conn)
    else
      _ -> Mint.HTTP.close(conn)
    end

    :ok
  end

  def terminate(_reason, %{conn: conn}) do
    if conn, do: Mint.HTTP.close(conn)
    :ok
  end

  # -- Private: Connection --

  defp do_connect(%{url: uri} = state) do
    scheme = ws_to_http_scheme(uri.scheme)
    ws_scheme = http_to_ws_scheme(uri.scheme)
    port = uri.port || default_port(scheme)
    path = (uri.path || "/") <> if(uri.query, do: "?#{uri.query}", else: "")

    with {:ok, conn} <-
           Mint.HTTP.connect(scheme, uri.host, port,
             protocols: [:http1],
             transport_opts: [timeout: state.connect_timeout]
           ),
         {:ok, conn, ref} <-
           Mint.WebSocket.upgrade(ws_scheme, conn, path, state.headers) do
      # Wait for the upgrade response
      await_upgrade(%{state | conn: conn, ref: ref})
    else
      {:error, reason} -> {:error, reason}
      {:error, _conn, reason} -> {:error, reason}
    end
  end

  defp await_upgrade(state) do
    receive do
      message ->
        case Mint.WebSocket.stream(state.conn, message) do
          {:ok, conn, responses} ->
            state = %{state | conn: conn}
            process_upgrade_responses(state, responses)

          {:error, _conn, reason, _responses} ->
            {:error, reason}

          :unknown ->
            await_upgrade(state)
        end
    after
      state.connect_timeout ->
        {:error, :connect_timeout}
    end
  end

  defp process_upgrade_responses(state, responses) do
    {status, headers} =
      Enum.reduce(responses, {nil, []}, fn
        {:status, _ref, status}, {_s, h} -> {status, h}
        {:headers, _ref, headers}, {s, _h} -> {s, headers}
        {:done, _ref}, acc -> acc
        _other, acc -> acc
      end)

    case Mint.WebSocket.new(state.conn, state.ref, status, headers) do
      {:ok, conn, websocket} ->
        {:ok, %{state | conn: conn, websocket: websocket, status: :connected}}

      {:error, _conn, reason} ->
        {:error, reason}
    end
  end

  # -- Private: Frame Handling --

  defp send_text_frame(state, payload) do
    with {:ok, websocket, data} <- Mint.WebSocket.encode(state.websocket, {:text, payload}),
         {:ok, conn} <- Mint.WebSocket.stream_request_body(state.conn, state.ref, data) do
      {:ok, %{state | conn: conn, websocket: websocket}}
    else
      {:error, reason} -> {:error, reason}
      {:error, _ws_or_conn, reason} -> {:error, reason}
    end
  end

  defp handle_responses(state, responses) do
    Enum.reduce(responses, {:noreply, state}, fn
      {:data, _ref, data}, {_action, state} ->
        handle_data(state, data)

      _other, acc ->
        acc
    end)
  end

  defp handle_data(state, data) do
    combined = state.buffer <> data

    case Mint.WebSocket.decode(state.websocket, combined) do
      {:ok, websocket, frames} ->
        state = %{state | websocket: websocket, buffer: ""}
        process_frames(state, frames)

      {:error, websocket, reason} ->
        Logger.error("WebSocket decode error: #{inspect(reason)}")
        state = %{state | websocket: websocket, buffer: ""}
        maybe_reply_error(state, {:error, reason})
    end
  end

  defp process_frames(state, frames) do
    Enum.reduce(frames, {:noreply, state}, fn
      {:text, text}, {_action, state} ->
        handle_text_frame(state, text)

      {:binary, data}, {_action, state} ->
        handle_text_frame(state, data)

      {:ping, data}, {_action, state} ->
        case send_pong(state, data) do
          {:ok, state} -> {:noreply, state}
          {:error, _reason} -> {:noreply, state}
        end

      {:close, code, reason}, {_action, state} ->
        Logger.info("WebSocket closed by server: code=#{code} reason=#{reason}")
        state = %{state | status: :disconnected}

        # Normal close (1000) after done_fn matched means the caller was already
        # replied to. Only send error if caller is still waiting.
        if state.caller do
          maybe_reply_error(state, {:error, {:closed, code, reason}})
        else
          {:noreply, state}
        end

      _other, acc ->
        acc
    end)
  end

  defp handle_text_frame(%{caller: nil} = state, _text) do
    # No caller waiting, discard
    {:noreply, state}
  end

  defp handle_text_frame(%{caller: caller} = state, text) do
    case Jason.decode(text) do
      {:ok, event} ->
        caller =
          if caller.callback_fn do
            result = caller.callback_fn.(event)
            %{caller | acc: [result | caller.acc]}
          else
            %{caller | acc: [event | caller.acc]}
          end

        if caller.done_fn.(event) do
          GenServer.reply(caller.from, {:ok, Enum.reverse(caller.acc)})
          {:noreply, %{state | caller: nil}}
        else
          {:noreply, %{state | caller: caller}}
        end

      {:error, _reason} ->
        # Non-JSON text frame, skip
        {:noreply, state}
    end
  end

  defp send_pong(state, data) do
    with {:ok, websocket, frame_data} <- Mint.WebSocket.encode(state.websocket, {:pong, data}),
         {:ok, conn} <- Mint.WebSocket.stream_request_body(state.conn, state.ref, frame_data) do
      {:ok, %{state | conn: conn, websocket: websocket}}
    end
  end

  defp maybe_reply_error(%{caller: nil} = state, _error) do
    {:noreply, state}
  end

  defp maybe_reply_error(%{caller: caller} = state, error) do
    GenServer.reply(caller.from, error)
    {:noreply, %{state | caller: nil}}
  end

  # -- Private: URI Helpers --

  defp ws_to_http_scheme("ws"), do: :http
  defp ws_to_http_scheme("wss"), do: :https
  defp ws_to_http_scheme(:ws), do: :http
  defp ws_to_http_scheme(:wss), do: :https
  # Allow http/https schemes to pass through
  defp ws_to_http_scheme("http"), do: :http
  defp ws_to_http_scheme("https"), do: :https

  defp http_to_ws_scheme("ws"), do: :ws
  defp http_to_ws_scheme("wss"), do: :wss
  defp http_to_ws_scheme(:ws), do: :ws
  defp http_to_ws_scheme(:wss), do: :wss
  defp http_to_ws_scheme("http"), do: :ws
  defp http_to_ws_scheme("https"), do: :wss

  defp default_port(:http), do: 80
  defp default_port(:https), do: 443
end
