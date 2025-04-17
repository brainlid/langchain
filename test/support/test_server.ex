defmodule LangChain.TestServer do
  @moduledoc """
  A mock server for testing LangChain API integrations.
  
  This module provides a simple way to mock external API responses,
  allowing tests to run without making actual API calls. It supports
  predefined responses, error simulation, and response verification.
  
  ## Example Usage
  
  ```elixir
  # Create a server with predetermined responses
  server = TestServer.new(
    responses: %{
      "chat.completions" => fn _req ->
        %{
          "id" => "chatcmpl-mock-id",
          "choices" => [
            %{
              "message" => %{
                "role" => "assistant",
                "content" => "This is a mock response"
              },
              "finish_reason" => "stop",
              "index" => 0
            }
          ],
          "usage" => %{
            "prompt_tokens" => 10,
            "completion_tokens" => 20,
            "total_tokens" => 30
          }
        }
      end
    }
  )
  
  # Use the server in tests
  test "it calls the API correctly", %{conn: conn} do
    # Start the server
    {:ok, server_pid} = TestServer.start_link(server)
    
    # Make requests (they'll be routed to the mock server)
    {:ok, result} = ChatGroq.call(%ChatGroq{endpoint: TestServer.endpoint()}, "Hello", [])
    
    # Verify requests were made correctly
    assert TestServer.received_request(server_pid, "chat.completions")
    
    # Verify response handling
    assert [%Message{content: "This is a mock response"}] = result
  end
  ```
  """
  
  use GenServer
  
  require Logger
  
  # Default port to use for the mock server
  @default_port 9876
  
  # ----------------
  # Client API
  # ----------------
  
  @doc """
  Creates a new test server configuration.
  
  ## Options
  
  * `:responses` - Map of endpoint paths to response functions
  * `:status_codes` - Map of endpoint paths to status codes
  * `:delay` - Artificial delay in milliseconds for all responses
  * `:error_rate` - Probability of simulated errors (0.0-1.0)
  
  ## Response Functions
  
  The response functions should take a request map and return the response body.
  The request map contains `:body`, `:headers`, and `:params` keys.
  
  ## Examples
  
  ```elixir
  TestServer.new(
    responses: %{
      "chat.completions" => fn _req -> %{"choices" => [%{"text" => "Hello"}]} end
    },
    status_codes: %{
      "chat.completions" => 200
    },
    delay: 50, # ms
    error_rate: 0.1 # 10% chance of error
  )
  ```
  """
  @spec new(keyword) :: map
  def new(opts \\ []) do
    %{
      responses: Keyword.get(opts, :responses, %{}),
      status_codes: Keyword.get(opts, :status_codes, %{}),
      delay: Keyword.get(opts, :delay, 0),
      error_rate: Keyword.get(opts, :error_rate, 0.0),
      requests: [],
      port: Keyword.get(opts, :port, @default_port)
    }
  end
  
  @doc """
  Starts the test server process.
  
  ## Options
  
  All options from `new/1` are supported, plus:
  
  * `:name` - Name to register the server process under
  
  ## Returns
  
  * `{:ok, pid}` - The server process ID
  * `{:error, reason}` - If server fails to start
  
  ## Examples
  
  ```elixir
  {:ok, server_pid} = TestServer.start_link(
    responses: %{"chat.completions" => fn _req -> %{} end}
  )
  ```
  """
  @spec start_link(map | keyword) :: {:ok, pid} | {:error, term}
  def start_link(server) when is_map(server) do
    GenServer.start_link(__MODULE__, server)
  end
  
  def start_link(opts) when is_list(opts) do
    name = Keyword.get(opts, :name)
    server = new(opts)
    
    if name do
      GenServer.start_link(__MODULE__, server, name: name)
    else
      GenServer.start_link(__MODULE__, server)
    end
  end
  
  @doc """
  Returns the base endpoint URL for the test server.
  
  ## Examples
  
  ```elixir
  TestServer.endpoint() #=> "http://localhost:9876"
  TestServer.endpoint(1234) #=> "http://localhost:1234"
  ```
  """
  @spec endpoint(integer | nil) :: String.t()
  def endpoint(port \\ nil) do
    "http://localhost:#{port || @default_port}"
  end
  
  @doc """
  Stops the test server.
  
  ## Examples
  
  ```elixir
  TestServer.stop(server_pid)
  ```
  """
  @spec stop(pid) :: :ok
  def stop(server) do
    GenServer.stop(server)
  end
  
  @doc """
  Sets a response for a specific endpoint.
  
  ## Examples
  
  ```elixir
  TestServer.set_response(server_pid, "chat.completions", fn _req -> 
    %{"choices" => [%{"text" => "Updated response"}]} 
  end)
  ```
  """
  @spec set_response(pid, String.t(), function) :: :ok
  def set_response(server, path, response_fn) when is_function(response_fn, 1) do
    GenServer.call(server, {:set_response, path, response_fn})
  end
  
  @doc """
  Sets a status code for a specific endpoint.
  
  ## Examples
  
  ```elixir
  TestServer.set_status_code(server_pid, "chat.completions", 429)
  ```
  """
  @spec set_status_code(pid, String.t(), integer) :: :ok
  def set_status_code(server, path, status_code) when is_integer(status_code) do
    GenServer.call(server, {:set_status_code, path, status_code})
  end
  
  @doc """
  Clears all recorded requests from the server.
  
  ## Examples
  
  ```elixir
  TestServer.clear_requests(server_pid)
  ```
  """
  @spec clear_requests(pid) :: :ok
  def clear_requests(server) do
    GenServer.call(server, :clear_requests)
  end
  
  @doc """
  Checks if the server received a request to a specific endpoint.
  
  ## Examples
  
  ```elixir
  assert TestServer.received_request(server_pid, "chat.completions")
  ```
  """
  @spec received_request(pid, String.t()) :: boolean
  def received_request(server, path) do
    GenServer.call(server, {:received_request, path})
  end
  
  @doc """
  Gets all requests received by the server.
  
  ## Examples
  
  ```elixir
  requests = TestServer.get_requests(server_pid)
  ```
  """
  @spec get_requests(pid) :: list
  def get_requests(server) do
    GenServer.call(server, :get_requests)
  end
  
  @doc """
  Gets requests to a specific endpoint.
  
  ## Examples
  
  ```elixir
  chat_requests = TestServer.get_requests(server_pid, "chat.completions")
  ```
  """
  @spec get_requests(pid, String.t()) :: list
  def get_requests(server, path) do
    GenServer.call(server, {:get_requests, path})
  end
  
  @doc """
  Gets the last request received by the server.
  
  ## Examples
  
  ```elixir
  last_request = TestServer.last_request(server_pid)
  ```
  """
  @spec last_request(pid) :: map | nil
  def last_request(server) do
    GenServer.call(server, :last_request)
  end
  
  @doc """
  Gets the last request to a specific endpoint.
  
  ## Examples
  
  ```elixir
  last_chat_request = TestServer.last_request(server_pid, "chat.completions")
  ```
  """
  @spec last_request(pid, String.t()) :: map | nil
  def last_request(server, path) do
    GenServer.call(server, {:last_request, path})
  end
  
  @doc """
  Sends a request to the mock server and returns the response.
  This can be used for testing the server itself or as a
  Req-compatible client function.
  
  ## Examples
  
  ```elixir
  {:ok, response} = TestServer.request(server_pid, "POST", "/v1/chat/completions", 
    body: %{messages: [%{role: "user", content: "Hello"}]},
    headers: [{"Authorization", "Bearer sk-123"}]
  )
  ```
  """
  @spec request(pid, String.t(), String.t(), keyword) :: {:ok, map} | {:error, term}
  def request(server, method, path, opts \\ []) do
    body = Keyword.get(opts, :body, nil)
    headers = Keyword.get(opts, :headers, [])
    params = Keyword.get(opts, :params, %{})
    
    GenServer.call(server, {:request, method, path, body, headers, params})
  end
  
  # ----------------
  # Server Callbacks
  # ----------------
  
  @impl true
  def init(server) do
    {:ok, server}
  end
  
  @impl true
  def handle_call({:set_response, path, response_fn}, _from, state) do
    new_responses = Map.put(state.responses, path, response_fn)
    {:reply, :ok, %{state | responses: new_responses}}
  end
  
  @impl true
  def handle_call({:set_status_code, path, status_code}, _from, state) do
    new_status_codes = Map.put(state.status_codes, path, status_code)
    {:reply, :ok, %{state | status_codes: new_status_codes}}
  end
  
  @impl true
  def handle_call(:clear_requests, _from, state) do
    {:reply, :ok, %{state | requests: []}}
  end
  
  @impl true
  def handle_call({:received_request, path}, _from, state) do
    received = Enum.any?(state.requests, fn req -> req.path == path end)
    {:reply, received, state}
  end
  
  @impl true
  def handle_call(:get_requests, _from, state) do
    {:reply, state.requests, state}
  end
  
  @impl true
  def handle_call({:get_requests, path}, _from, state) do
    requests = Enum.filter(state.requests, fn req -> req.path == path end)
    {:reply, requests, state}
  end
  
  @impl true
  def handle_call(:last_request, _from, %{requests: []} = state) do
    {:reply, nil, state}
  end
  
  @impl true
  def handle_call(:last_request, _from, state) do
    [last | _] = state.requests
    {:reply, last, state}
  end
  
  @impl true
  def handle_call({:last_request, path}, _from, state) do
    last = Enum.find(state.requests, fn req -> req.path == path end)
    {:reply, last, state}
  end
  
  @impl true
  def handle_call({:request, method, path, body, headers, params}, _from, state) do
    # Record the request
    request = %{
      method: method,
      path: normalize_path(path),
      body: body,
      headers: headers,
      params: params,
      timestamp: DateTime.utc_now()
    }
    
    # Add to the list of requests (most recent first)
    updated_state = %{state | requests: [request | state.requests]}
    
    # Apply artificial delay if configured
    if state.delay > 0 do
      Process.sleep(state.delay)
    end
    
    # Check if we should return an error based on error_rate
    if :rand.uniform() < state.error_rate do
      error_response = %{
        status: 500,
        body: %{
          "error" => %{
            "message" => "Simulated server error",
            "type" => "server_error",
            "code" => "mock_server_error"
          }
        },
        headers: [{"content-type", "application/json"}]
      }
      
      {:reply, {:ok, error_response}, updated_state}
    else
      # Get appropriate handler for this path
      normalized_path = normalize_path(path)
      response_fn = Map.get(state.responses, normalized_path)
      
      if response_fn do
        # Get response from the handler
        response_body = response_fn.(request)
        
        # Get status code (default to 200)
        status_code = Map.get(state.status_codes, normalized_path, 200)
        
        response = %{
          status: status_code,
          body: response_body,
          headers: [{"content-type", "application/json"}]
        }
        
        {:reply, {:ok, response}, updated_state}
      else
        # No handler found for this path
        not_found_response = %{
          status: 404,
          body: %{
            "error" => %{
              "message" => "Not found",
              "type" => "invalid_request_error",
              "code" => "resource_not_found"
            }
          },
          headers: [{"content-type", "application/json"}]
        }
        
        {:reply, {:ok, not_found_response}, updated_state}
      end
    end
  end
  
  # Helper function to normalize path
  # Removes leading/trailing slashes and extracts the main component
  defp normalize_path(path) do
    path
    |> String.trim("/")
    |> String.split("/")
    |> Enum.drop(1) # Drop v1 or openai/v1 prefix
    |> Enum.join("/")
  end
end