defmodule LangChain.Tools.DeepResearch do
  @moduledoc """
  Defines an OpenAI Deep Research tool for conducting comprehensive research on complex topics.

  This tool leverages OpenAI's o3-deep-research and o4-mini-deep-research models to perform
  multi-step research analysis that can take 5-30 minutes to complete. The models can find,
  analyze, and synthesize hundreds of sources to create comprehensive reports at the level
  of a research analyst.

  The Deep Research tool is designed for complex analysis and research tasks such as:
  * Legal or scientific research
  * Market analysis
  * Reporting on large bodies of data

  ## Important Notes

  * Deep Research requests are **long-running operations** (5-30 minutes)
  * The tool will return a request ID immediately, then poll for completion
  * Research results include inline citations and source metadata
  * Requires an OpenAI API key with access to Deep Research models

  ## Timeout Configuration

  Deep Research runs as an async tool with a default timeout of 2 minutes, which is insufficient
  for Deep Research operations. You must configure the `async_tool_timeout` when using this tool:

      {:ok, chain} =
        %{
          llm: model,
          verbose: true,
          async_tool_timeout: 35 * 60 * 1000  # 35 minutes in milliseconds
        }
        |> LLMChain.new!()
        |> LLMChain.add_tools(DeepResearch.new!())

  Without this configuration, you may encounter timeout errors after 2 minutes.

  ## Example

  The following example shows how to use the Deep Research tool in a chain:

      {:ok, updated_chain, %Message{} = message} =
        %{
          llm: ChatOpenAI.new!(%{temperature: 0}), 
          verbose: true,
          async_tool_timeout: 35 * 60 * 1000  # 35 minutes for Deep Research
        }
        |> LLMChain.new!()
        |> LLMChain.add_message(
          Message.new_user!("Research the economic impact of renewable energy adoption on job markets.")
        )
        |> LLMChain.add_functions(DeepResearch.new!())
        |> LLMChain.run(mode: :until_success)

  The tool will initiate a research request and return comprehensive findings with citations.
  """
  require Logger
  alias LangChain.Function
  alias LangChain.Tools.DeepResearchClient

  @doc """
  Define the "deep_research" function. Returns a success/failure response.
  """
  @spec new() :: {:ok, Function.t()} | {:error, Ecto.Changeset.t()}
  def new() do
    Function.new(%{
      name: "deep_research",
      description:
        "Perform comprehensive research on complex topics using OpenAI Deep Research models. This is a long-running operation (5-30 minutes) that provides detailed analysis with citations.",
      parameters_schema: %{
        type: "object",
        properties: %{
          query: %{
            type: "string",
            description:
              "The research question or topic to investigate. Be specific and detailed for best results."
          },
          model: %{
            type: "string",
            enum: ["o3-deep-research-2025-06-26", "o4-mini-deep-research-2025-06-26"],
            description:
              "The deep research model to use. o3-deep-research provides highest quality (5-30 min), o4-mini-deep-research is faster (shorter time).",
            default: "o3-deep-research-2025-06-26"
          },
          system_message: %{
            type: "string",
            description:
              "Optional guidance for the research approach, methodology, or specific requirements."
          },
          max_tool_calls: %{
            type: "integer",
            description:
              "Maximum number of tool calls (web searches, etc.) to make. Controls cost and latency.",
            minimum: 1,
            maximum: 100
          },
          summary: %{
            type: "string",
            enum: ["auto", "detailed"],
            description:
              "Summary mode for the research report. 'auto' provides the best possible summary, 'detailed' provides more comprehensive results.",
            default: "auto"
          },
          include_code_interpreter: %{
            type: "boolean",
            description:
              "Include code interpreter tool for data analysis and visualization capabilities.",
            default: true
          }
        },
        required: ["query"]
      },
      function: &execute/2,
      async: true
    })
  end

  @doc """
  Define the "deep_research" function. Raises an exception if function creation fails.
  """
  @spec new!() :: Function.t() | no_return()
  def new!() do
    case new() do
      {:ok, function} ->
        function

      {:error, changeset} ->
        raise LangChain.LangChainError, changeset
    end
  end

  @doc """
  Executes the deep research request. This function handles the long-running nature
  of deep research by creating a request and polling for completion.

  Returns the research findings with inline citations and source metadata.
  """
  @spec execute(args :: %{String.t() => any()}, context :: map()) ::
          {:ok, String.t()} | {:error, String.t()}
  def execute(%{"query" => query} = args, _context) do
    try do
      # Extract optional parameters with defaults
      model = Map.get(args, "model", "o3-deep-research-2025-06-26")
      system_message = Map.get(args, "system_message")
      max_tool_calls = Map.get(args, "max_tool_calls")
      summary = Map.get(args, "summary", "auto")
      include_code_interpreter = Map.get(args, "include_code_interpreter", true)

      Logger.info("Starting deep research request for query: #{inspect(query)}")

      # Create the research request
      case DeepResearchClient.create_research(query, %{
             model: model,
             system_message: system_message,
             max_tool_calls: max_tool_calls,
             summary: summary,
             include_code_interpreter: include_code_interpreter
           }) do
        {:ok, request_id} ->
          Logger.info("Deep research request created with ID: #{request_id}")

          # Poll for completion
          case wait_for_completion(request_id) do
            {:ok, result} ->
              Logger.info("Deep research completed successfully")
              {:ok, format_research_result(result)}

            {:error, reason} ->
              {:error, "Deep research failed: #{reason}"}
          end

        {:error, reason} ->
          {:error, "Failed to start research: #{reason}"}
      end
    rescue
      err ->
        {:error, "Deep research tool error: #{Exception.message(err)}"}
    end
  end

  # Handle missing required parameter
  def execute(_args, _context) do
    {:error, "ERROR: 'query' parameter is required for deep research"}
  end

  # Waits for a deep research request to complete by polling the status endpoint.
  # Implements exponential backoff for efficient polling.
  @spec wait_for_completion(String.t()) :: {:ok, map()} | {:error, String.t()}
  defp wait_for_completion(request_id) do
    # Start with 10 second polling interval, max 60 seconds
    poll_with_backoff(request_id, 10_000, 60_000, 0)
  end

  @spec poll_with_backoff(String.t(), integer(), integer(), integer()) ::
          {:ok, map()} | {:error, String.t()}
  defp poll_with_backoff(request_id, interval, max_interval, attempts) do
    case DeepResearchClient.check_status(request_id) do
      {:ok, %{status: "completed"}} ->
        DeepResearchClient.get_results(request_id)

      {:ok, %{status: "failed", error: error}} ->
        {:error, "Research failed: #{error}"}

      {:ok, %{status: "cancelled"}} ->
        {:error, "Research was cancelled"}

      {:ok, %{status: status}} when status in ["in_progress", "queued"] ->
        Logger.info("Deep research #{request_id} still #{status}, waiting #{interval}ms...")
        Process.sleep(interval)

        # Exponential backoff with jitter, but cap at max_interval
        next_interval = min(trunc(interval * 1.5 + :rand.uniform(5000)), max_interval)
        poll_with_backoff(request_id, next_interval, max_interval, attempts + 1)

      {:ok, %{status: unknown_status}} ->
        {:error, "Unknown research status: #{unknown_status}"}

      {:error, reason} ->
        if attempts < 3 do
          Logger.warning(
            "Status check failed (attempt #{attempts + 1}), retrying: #{inspect(reason)}"
          )

          Process.sleep(5000)
          poll_with_backoff(request_id, interval, max_interval, attempts + 1)
        else
          {:error, "Failed to check research status after #{attempts + 1} attempts: #{reason}"}
        end
    end
  end

  # Formats the research result for LLM consumption, including the main findings
  # and a summary of sources used.
  @spec format_research_result(map()) :: String.t()
  defp format_research_result(%{output_text: text, sources: sources}) when is_list(sources) do
    source_summary = format_sources(sources)

    """
    ## Research Findings

    #{text}

    ## Sources Consulted

    #{source_summary}
    """
  end

  defp format_research_result(%{output_text: text}) do
    text
  end

  defp format_research_result(result) do
    "Research completed. Result: #{inspect(result)}"
  end

  @spec format_sources(list()) :: String.t()
  defp format_sources(sources) when is_list(sources) do
    sources
    |> Enum.with_index(1)
    |> Enum.map(fn {source, index} ->
      title = Map.get(source, "title", "Untitled")
      url = Map.get(source, "url", "No URL")
      "#{index}. #{title} - #{url}"
    end)
    |> Enum.join("\n")
  end

  defp format_sources(_), do: "No source information available."
end
