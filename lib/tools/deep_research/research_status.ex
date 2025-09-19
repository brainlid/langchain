defmodule LangChain.Tools.DeepResearch.ResearchStatus do
  @moduledoc """
  Represents the status of a Deep Research request.

  This schema captures the current state of a research request, including
  progress information and any errors that may have occurred.
  """
  use Ecto.Schema
  import Ecto.Changeset

  @type t() :: %__MODULE__{
          id: String.t(),
          status: String.t(),
          created_at: integer() | nil,
          error_message: String.t() | nil,
          error_code: String.t() | nil,
          progress_info: map() | nil
        }

  @primary_key false
  embedded_schema do
    field :id, :string
    field :status, :string
    field :created_at, :integer
    field :error_message, :string
    field :error_code, :string
    field :progress_info, :map
  end

  @valid_statuses ~w(queued in_progress completed failed cancelled incomplete)

  @doc """
  Creates a changeset for research status.
  """
  @spec changeset(__MODULE__.t(), map()) :: Ecto.Changeset.t()
  def changeset(status \\ %__MODULE__{}, attrs) do
    status
    |> cast(attrs, [:id, :status, :created_at, :error_message, :error_code, :progress_info])
    |> validate_required([:id, :status])
    |> validate_inclusion(:status, @valid_statuses)
  end

  @doc """
  Creates a ResearchStatus from an OpenAI API response.
  """
  @spec from_api_response(map()) :: {:ok, __MODULE__.t()} | {:error, Ecto.Changeset.t()}
  def from_api_response(response) do
    attrs = %{
      id: Map.get(response, "id"),
      status: Map.get(response, "status"),
      created_at: Map.get(response, "created_at"),
      error_message: get_in(response, ["error", "message"]),
      error_code: get_in(response, ["error", "code"]),
      progress_info: extract_progress_info(response)
    }

    changeset = changeset(%__MODULE__{}, attrs)

    if changeset.valid? do
      {:ok, apply_changes(changeset)}
    else
      {:error, changeset}
    end
  end

  @doc """
  Checks if the research is complete (either successfully or with failure).
  """
  @spec complete?(__MODULE__.t()) :: boolean()
  def complete?(%__MODULE__{status: status})
      when status in ["completed", "failed", "cancelled"] do
    true
  end

  def complete?(_), do: false

  @doc """
  Checks if the research completed successfully.
  """
  @spec successful?(__MODULE__.t()) :: boolean()
  def successful?(%__MODULE__{status: "completed"}), do: true
  def successful?(_), do: false

  @doc """
  Checks if the research failed.
  """
  @spec failed?(__MODULE__.t()) :: boolean()
  def failed?(%__MODULE__{status: status}) when status in ["failed", "cancelled"] do
    true
  end

  def failed?(_), do: false

  @doc """
  Gets a human-readable description of the current status.
  """
  @spec status_description(__MODULE__.t()) :: String.t()
  def status_description(%__MODULE__{status: "queued"}),
    do: "Research request is queued for processing"

  def status_description(%__MODULE__{status: "in_progress"}),
    do: "Research is currently in progress"

  def status_description(%__MODULE__{status: "completed"}), do: "Research completed successfully"

  def status_description(%__MODULE__{status: "failed", error_message: msg}) when is_binary(msg),
    do: "Research failed: #{msg}"

  def status_description(%__MODULE__{status: "failed"}), do: "Research failed"
  def status_description(%__MODULE__{status: "cancelled"}), do: "Research was cancelled"

  def status_description(%__MODULE__{status: "incomplete"}),
    do: "Research completed but may be incomplete"

  def status_description(%__MODULE__{status: status}), do: "Unknown status: #{status}"

  @spec extract_progress_info(map()) :: map() | nil
  defp extract_progress_info(response) do
    # Extract any progress-related information from the response
    # This could include tool call counts, current step, etc.
    case response do
      %{"output" => output} when is_list(output) ->
        tool_calls =
          Enum.count(
            output,
            &(Map.get(&1, "type") in ["web_search_call", "code_interpreter_call", "mcp_tool_call"])
          )

        if tool_calls > 0 do
          %{tool_calls_made: tool_calls}
        else
          nil
        end

      _ ->
        nil
    end
  end
end
