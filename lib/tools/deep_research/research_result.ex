defmodule LangChain.Tools.DeepResearch.ResearchResult do
  @moduledoc """
  Represents the final result of a completed Deep Research request.

  This schema captures the research findings, citations, usage statistics,
  and other metadata from a successful research operation.
  """
  use Ecto.Schema
  import Ecto.Changeset

  @type t() :: %__MODULE__{
          id: String.t(),
          output_text: String.t(),
          model: String.t() | nil,
          created_at: integer() | nil,
          completion_time: integer() | nil,
          sources: [any()],
          usage: any() | nil,
          tool_calls: [any()]
        }

  @primary_key false
  embedded_schema do
    field :id, :string
    field :output_text, :string
    field :model, :string
    field :created_at, :integer
    field :completion_time, :integer

    embeds_many :sources, Source do
      field :title, :string
      field :url, :string
      field :start_index, :integer
      field :end_index, :integer
      field :snippet, :string
    end

    embeds_one :usage, Usage do
      field :input_tokens, :integer
      field :output_tokens, :integer
      field :total_tokens, :integer
      field :reasoning_tokens, :integer
    end

    embeds_many :tool_calls, ToolCall do
      field :type, :string
      field :status, :string
      field :action, :map
      field :result, :map
    end
  end

  @doc """
  Creates a changeset for research result.
  """
  @spec changeset(__MODULE__.t(), map()) :: Ecto.Changeset.t()
  def changeset(result \\ %__MODULE__{}, attrs) do
    result
    |> cast(attrs, [:id, :output_text, :model, :created_at, :completion_time])
    |> cast_embed(:sources, with: &source_changeset/2)
    |> cast_embed(:usage, with: &usage_changeset/2)
    |> cast_embed(:tool_calls, with: &tool_call_changeset/2)
    |> validate_required([:id, :output_text])
  end

  @doc """
  Creates a ResearchResult from an OpenAI API response.
  """
  @spec from_api_response(map()) :: {:ok, __MODULE__.t()} | {:error, Ecto.Changeset.t()}
  def from_api_response(response) do
    attrs = %{
      id: Map.get(response, "id"),
      output_text: extract_output_text(response),
      model: Map.get(response, "model"),
      created_at: Map.get(response, "created_at"),
      completion_time: calculate_completion_time(response),
      sources: extract_sources(response),
      usage: Map.get(response, "usage"),
      tool_calls: extract_tool_calls(response)
    }

    changeset = changeset(%__MODULE__{}, attrs)

    if changeset.valid? do
      {:ok, apply_changes(changeset)}
    else
      {:error, changeset}
    end
  end

  @doc """
  Gets the total number of sources cited in the research.
  """
  @spec source_count(__MODULE__.t()) :: integer()
  def source_count(%__MODULE__{sources: sources}), do: length(sources)

  @doc """
  Gets the total number of tool calls made during research.
  """
  @spec tool_call_count(__MODULE__.t()) :: integer()
  def tool_call_count(%__MODULE__{tool_calls: tool_calls}), do: length(tool_calls)

  @doc """
  Formats the result for display, including the main text and source summary.
  """
  @spec format_for_display(__MODULE__.t()) :: String.t()
  def format_for_display(%__MODULE__{} = result) do
    source_summary = format_sources(result.sources)

    output = """
    ## Research Findings

    #{result.output_text}
    """

    if source_summary != "" do
      output <> "\n\n## Sources\n\n#{source_summary}"
    else
      output
    end
  end

  @doc """
  Extracts just the URLs from the sources for easy reference.
  """
  @spec source_urls(__MODULE__.t()) :: [String.t()]
  def source_urls(%__MODULE__{sources: sources}) do
    Enum.map(sources, & &1.url)
  end

  # Private functions

  @spec source_changeset(map(), map()) :: Ecto.Changeset.t()
  defp source_changeset(source, attrs) do
    source
    |> cast(attrs, [:title, :url, :start_index, :end_index, :snippet])
    |> validate_required([:url])
  end

  @spec usage_changeset(map(), map()) :: Ecto.Changeset.t()
  defp usage_changeset(usage, attrs) do
    usage
    |> cast(attrs, [:input_tokens, :output_tokens, :total_tokens, :reasoning_tokens])
    |> validate_number(:input_tokens, greater_than_or_equal_to: 0)
    |> validate_number(:output_tokens, greater_than_or_equal_to: 0)
    |> validate_number(:total_tokens, greater_than_or_equal_to: 0)
    |> validate_number(:reasoning_tokens, greater_than_or_equal_to: 0)
  end

  @spec tool_call_changeset(map(), map()) :: Ecto.Changeset.t()
  defp tool_call_changeset(tool_call, attrs) do
    tool_call
    |> cast(attrs, [:type, :status, :action, :result])
    |> validate_required([:type])
    |> validate_inclusion(:type, ["web_search_call", "code_interpreter_call", "mcp_tool_call"])
  end

  @spec extract_output_text(map()) :: String.t()
  defp extract_output_text(response) do
    response
    |> get_in(["output"])
    |> case do
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
        "No output available"
    end
  end

  @spec extract_sources(map()) :: [map()]
  defp extract_sources(response) do
    # Extract sources from annotations in the output
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
        |> Enum.map(&normalize_source/1)

      _ ->
        []
    end
  end

  @spec normalize_source(map()) :: map()
  defp normalize_source(annotation) do
    %{
      title: Map.get(annotation, "title"),
      url: Map.get(annotation, "url"),
      start_index: Map.get(annotation, "start_index"),
      end_index: Map.get(annotation, "end_index"),
      snippet: Map.get(annotation, "snippet")
    }
  end

  @spec extract_tool_calls(map()) :: [map()]
  defp extract_tool_calls(response) do
    response
    |> get_in(["output"])
    |> case do
      output when is_list(output) ->
        output
        |> Enum.filter(
          &(Map.get(&1, "type") in ["web_search_call", "code_interpreter_call", "mcp_tool_call"])
        )
        |> Enum.map(&normalize_tool_call/1)

      _ ->
        []
    end
  end

  @spec normalize_tool_call(map()) :: map()
  defp normalize_tool_call(tool_call) do
    %{
      type: Map.get(tool_call, "type"),
      status: Map.get(tool_call, "status"),
      action: Map.get(tool_call, "action"),
      result: Map.get(tool_call, "result")
    }
  end

  @spec calculate_completion_time(map()) :: integer() | nil
  defp calculate_completion_time(_response) do
    # If we have completion timestamp info, calculate duration
    # For now, return nil as this would need to be tracked externally
    nil
  end

  @spec format_sources([map()]) :: String.t()
  defp format_sources(sources) when is_list(sources) do
    sources
    |> Enum.with_index(1)
    |> Enum.map(fn {source, index} ->
      title = source.title || "Untitled"
      url = source.url || "No URL"
      "#{index}. #{title} - #{url}"
    end)
    |> Enum.join("\n")
  end

  defp format_sources(_), do: ""
end
