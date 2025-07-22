defmodule LangChain.Tools.DeepResearch.ResearchRequest do
  @moduledoc """
  Represents a Deep Research request sent to the OpenAI API.

  This schema defines the structure of a research request including the query,
  model selection, and various configuration options.
  """
  use Ecto.Schema
  import Ecto.Changeset

  @type t() :: %__MODULE__{
          query: String.t(),
          model: String.t(),
          system_message: String.t() | nil,
          max_tool_calls: integer() | nil,
          background: boolean(),
          temperature: float(),
          max_output_tokens: integer() | nil,
          summary: String.t()
        }

  @primary_key false
  embedded_schema do
    field :query, :string
    field :model, :string, default: "o3-deep-research-2025-06-26"
    field :system_message, :string
    field :max_tool_calls, :integer
    field :background, :boolean, default: true
    field :temperature, :float, default: 1.0
    field :max_output_tokens, :integer
    field :summary, :string, default: "auto"
  end

  @doc """
  Creates a changeset for a research request.
  """
  @spec changeset(__MODULE__.t(), map()) :: Ecto.Changeset.t()
  def changeset(request \\ %__MODULE__{}, attrs) do
    request
    |> cast(attrs, [
      :query,
      :model,
      :system_message,
      :max_tool_calls,
      :background,
      :temperature,
      :max_output_tokens,
      :summary
    ])
    |> validate_required([:query])
    |> validate_length(:query, min: 1, max: 10_000)
    |> validate_inclusion(:model, [
      "o3-deep-research-2025-06-26",
      "o4-mini-deep-research-2025-06-26"
    ])
    |> validate_inclusion(:summary, ["auto", "detailed"])
    |> validate_number(:max_tool_calls, greater_than: 0, less_than_or_equal_to: 100)
    |> validate_number(:temperature, greater_than_or_equal_to: 0.0, less_than_or_equal_to: 2.0)
    |> validate_number(:max_output_tokens, greater_than: 0)
  end

  @doc """
  Converts the research request to the format expected by the OpenAI API.
  """
  @spec to_api_format(__MODULE__.t()) :: map()
  def to_api_format(%__MODULE__{} = request) do
    %{
      model: request.model,
      input: request.query,
      background: request.background,
      tools: [%{type: "web_search_preview"}],
      reasoning: %{summary: request.summary}
    }
    |> maybe_add_field(:instructions, request.system_message)
    |> maybe_add_field(:max_tool_calls, request.max_tool_calls)
    |> maybe_add_field(:temperature, request.temperature)
    |> maybe_add_field(:max_output_tokens, request.max_output_tokens)
  end

  @spec maybe_add_field(map(), atom(), any()) :: map()
  defp maybe_add_field(map, _key, nil), do: map
  defp maybe_add_field(map, key, value), do: Map.put(map, key, value)
end
