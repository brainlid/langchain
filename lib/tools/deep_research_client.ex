defmodule LangChain.Tools.DeepResearchClient do
  @moduledoc """
  HTTP client for OpenAI Deep Research API.

  This module handles the three main operations for OpenAI Deep Research:
  1. Creating a research request
  2. Checking request status
  3. Retrieving completed results

  Uses the same authentication and HTTP client patterns as other OpenAI integrations
  in LangChain, reusing the existing `:openai_key` configuration.
  """
  require Logger
  alias LangChain.Config

  @base_url "https://api.openai.com/v1/responses"

  @doc """
  Creates a new deep research request.

  ## Parameters
  - `query`: The research question or topic
  - `options`: Optional parameters including:
    - `:model` - The model to use (defaults to "o3-deep-research-2025-06-26")
    - `:system_message` - Optional guidance for research approach
    - `:max_tool_calls` - Maximum number of tool calls to make

  ## Returns
  - `{:ok, request_id}` on success
  - `{:error, reason}` on failure
  """
  @spec create_research(String.t(), map()) :: {:ok, String.t()} | {:error, String.t()}
  def create_research(query, options \\ %{}) do
    model = Map.get(options, :model, "o3-deep-research-2025-06-26")
    system_message = Map.get(options, :system_message)
    max_tool_calls = Map.get(options, :max_tool_calls)
    # Build the request body according to OpenAI Deep Research API
    request_body = build_request_body(query, model, system_message, max_tool_calls)

    Logger.debug("Creating deep research request with body: #{inspect(request_body)}")

    case make_request(:post, @base_url, request_body) do
      {:ok, %{"id" => request_id}} ->
        {:ok, request_id}

      {:ok, response} ->
        Logger.error("Unexpected response format: #{inspect(response)}")
        {:error, "Unexpected response format from OpenAI API"}

      {:error, reason} ->
        {:error, reason}
    end
  end

  @doc """
  Checks the status of a deep research request.

  ## Parameters
  - `request_id`: The ID returned from create_research/2

  ## Returns
  - `{:ok, status_map}` containing status information
  - `{:error, reason}` on failure
  """
  @spec check_status(String.t()) :: {:ok, map()} | {:error, String.t()}
  def check_status(request_id) do
    url = "#{@base_url}/#{request_id}"

    case make_request(:get, url) do
      {:ok, response} ->
        status_map = %{
          status: Map.get(response, "status"),
          error: get_in(response, ["error", "message"])
        }

        {:ok, status_map}

      {:error, reason} ->
        {:error, reason}
    end
  end

  @doc """
  Retrieves the results of a completed deep research request.

  ## Parameters
  - `request_id`: The ID of the completed research request

  ## Returns
  - `{:ok, result_map}` containing the research findings and metadata
  - `{:error, reason}` on failure
  """
  @spec get_results(String.t()) :: {:ok, map()} | {:error, String.t()}
  def get_results(request_id) do
    url = "#{@base_url}/#{request_id}"

    case make_request(:get, url) do
      {:ok, response} ->
        case Map.get(response, "status") do
          "completed" ->
            result = extract_research_result(response)
            {:ok, result}

          other_status ->
            {:error, "Research not completed, status: #{other_status}"}
        end

      {:error, reason} ->
        {:error, reason}
    end
  end

  # Private functions

  @spec build_request_body(String.t(), String.t(), String.t() | nil, integer() | nil) :: map()
  defp build_request_body(query, model, system_message, max_tool_calls) do
    base_body = %{
      model: model,
      input: query,
      background: true,
      tools: [
        %{type: "web_search_preview"}
      ]
    }

    # Add optional parameters if provided
    base_body
    |> maybe_add_field(:instructions, system_message)
    |> maybe_add_field(:max_tool_calls, max_tool_calls)
  end

  @spec maybe_add_field(map(), atom(), any()) :: map()
  defp maybe_add_field(map, _key, nil), do: map
  defp maybe_add_field(map, key, value), do: Map.put(map, key, value)

  @spec extract_research_result(map()) :: map()
  defp extract_research_result(response) do
    # Extract the main research text from the output array
    output_text = extract_output_text(response)

    # Extract source information from annotations or web search calls
    sources = extract_sources(response)

    %{
      output_text: output_text,
      sources: sources,
      usage: Map.get(response, "usage"),
      model: Map.get(response, "model")
    }
  end

  @spec extract_output_text(map()) :: String.t()
  defp extract_output_text(response) do
    response
    |> get_in(["output"])
    |> case do
      nil ->
        "No output available"

      output when is_list(output) ->
        output
        |> Enum.find(&(Map.get(&1, "type") == "message"))
        |> case do
          nil ->
            "No message output found"

          message ->
            message
            |> get_in(["content"])
            |> case do
              nil ->
                "No content found"

              content when is_list(content) ->
                content
                |> Enum.find(&(Map.get(&1, "type") == "output_text"))
                |> case do
                  nil -> "No text content found"
                  text_content -> Map.get(text_content, "text", "No text available")
                end

              _ ->
                "Invalid content format"
            end
        end

      _ ->
        "Invalid output format"
    end
  end

  @spec extract_sources(map()) :: list()
  defp extract_sources(response) do
    # Look for sources in multiple places: annotations and web search calls
    annotation_sources = extract_annotation_sources(response)
    web_search_sources = extract_web_search_sources(response)

    (annotation_sources ++ web_search_sources)
    |> Enum.uniq_by(&Map.get(&1, "url"))
  end

  @spec extract_annotation_sources(map()) :: list()
  defp extract_annotation_sources(response) do
    response
    |> get_in(["output"])
    |> case do
      output when is_list(output) ->
        output
        |> Enum.flat_map(fn item ->
          item
          |> get_in(["content"])
          |> case do
            content when is_list(content) ->
              content
              |> Enum.flat_map(fn content_item ->
                Map.get(content_item, "annotations", [])
              end)

            _ ->
              []
          end
        end)

      _ ->
        []
    end
  end

  @spec extract_web_search_sources(map()) :: list()
  defp extract_web_search_sources(response) do
    response
    |> get_in(["output"])
    |> case do
      output when is_list(output) ->
        output
        |> Enum.filter(&(Map.get(&1, "type") == "web_search_call"))
        |> Enum.flat_map(&extract_sources_from_search_call/1)

      _ ->
        []
    end
  end

  @spec extract_sources_from_search_call(map()) :: list()
  defp extract_sources_from_search_call(_search_call) do
    # This would need to be refined based on actual API response structure
    # For now, return empty list as we'll primarily rely on annotations
    []
  end

  @spec make_request(atom(), String.t(), map() | nil) :: {:ok, map()} | {:error, String.t()}
  defp make_request(method, url, body \\ nil) do
    headers = build_headers()

    request_options = [
      method: method,
      url: url,
      headers: headers,
      # 30 seconds for API calls
      receive_timeout: 30_000,
      retry: :transient
    ]

    request_options =
      if body do
        Keyword.put(request_options, :json, body)
      else
        request_options
      end

    Logger.debug("Making #{method} request to #{url}")

    case Req.request(request_options) do
      {:ok, %Req.Response{status: status, body: response_body}} when status in 200..299 ->
        {:ok, response_body}

      {:ok, %Req.Response{status: status, body: error_body}} ->
        error_message = extract_error_message(error_body, status)
        Logger.error("API request failed with status #{status}: #{error_message}")
        {:error, error_message}

      {:error, %Req.TransportError{reason: reason}} ->
        Logger.error("Transport error in API request: #{inspect(reason)}")
        {:error, "Network error: #{inspect(reason)}"}

      {:error, reason} ->
        Logger.error("Unexpected error in API request: #{inspect(reason)}")
        {:error, "Request failed: #{inspect(reason)}"}
    end
  end

  @spec build_headers() :: list()
  defp build_headers() do
    api_key = get_api_key()

    [
      {"authorization", "Bearer #{api_key}"},
      {"content-type", "application/json"},
      {"user-agent", "LangChain-Elixir/1.0"}
    ]
  end

  @spec get_api_key() :: String.t()
  defp get_api_key() do
    # Reuse the same OpenAI API key resolution as ChatOpenAI
    Config.resolve(:openai_key, "")
  end

  @spec extract_error_message(map() | String.t(), integer()) :: String.t()
  defp extract_error_message(error_body, status) when is_map(error_body) do
    case get_in(error_body, ["error", "message"]) do
      nil -> "HTTP #{status}: #{inspect(error_body)}"
      message -> "HTTP #{status}: #{message}"
    end
  end

  defp extract_error_message(error_body, status) do
    "HTTP #{status}: #{inspect(error_body)}"
  end
end
