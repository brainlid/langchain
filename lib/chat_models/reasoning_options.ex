defmodule LangChain.ChatModels.ReasoningOptions do
  @moduledoc """
  Embedded schema for OpenAI reasoning configuration options.

  Used with gpt-5 and o-series models only.

  ## Fields

  - `effort` - Constrains effort on reasoning. Supported values: :minimal, :low, :medium, :high
  - `generate_summary` - (Deprecated) A summary of the reasoning performed. Use `summary` instead.
  - `summary` - A summary of the reasoning performed. Supported values: :auto, :concise, :detailed
  """
  use Ecto.Schema
  import Ecto.Changeset
  alias LangChain.Utils

  @primary_key false

  embedded_schema do
    field(:effort, Ecto.Enum, values: [:minimal, :low, :medium, :high], default: nil)
    field(:generate_summary, Ecto.Enum, values: [:auto, :concise, :detailed], default: nil)
    field(:summary, Ecto.Enum, values: [:auto, :concise, :detailed], default: nil)
  end

  @create_fields [:effort, :generate_summary, :summary]

  @type t :: %__MODULE__{}

  @doc """
  Creates a changeset for ReasoningOptions.
  """
  @spec changeset(t(), map()) :: Ecto.Changeset.t()
  def changeset(%__MODULE__{} = reasoning, attrs) do
    reasoning
    |> cast(attrs, @create_fields)
  end

  @doc """
  Creates a new ReasoningOptions struct.
  """
  @spec new(map()) :: {:ok, t()} | {:error, Ecto.Changeset.t()}
  def new(attrs \\ %{}) do
    %__MODULE__{}
    |> changeset(attrs)
    |> apply_action(:insert)
  end

  @doc """
  Creates a new ReasoningOptions struct, raising on error.
  """
  @spec new!(map()) :: t() | no_return()
  def new!(attrs \\ %{}) do
    case new(attrs) do
      {:ok, reasoning} ->
        reasoning

      {:error, changeset} ->
        raise ArgumentError, "Invalid reasoning options: #{inspect(changeset.errors)}"
    end
  end

  @doc """
  Converts the ReasoningOptions to a map suitable for API requests.
  Returns nil if no options are set.
  """
  @spec to_api_map(t() | nil) :: map() | nil
  def to_api_map(nil), do: nil

  def to_api_map(%__MODULE__{} = reasoning) do
    %{}
    |> Utils.conditionally_add_to_map("effort", atom_to_string(reasoning.effort))
    |> Utils.conditionally_add_to_map(
      "generate_summary",
      atom_to_string(reasoning.generate_summary)
    )
    |> Utils.conditionally_add_to_map("summary", atom_to_string(reasoning.summary))
    |> case do
      empty when map_size(empty) == 0 -> nil
      options -> options
    end
  end

  defp atom_to_string(nil), do: nil
  defp atom_to_string(atom) when is_atom(atom), do: Atom.to_string(atom)

  @doc """
  Returns the list of valid effort values.
  """
  def valid_efforts, do: [:minimal, :low, :medium, :high]

  @doc """
  Returns the list of valid summary values.
  """
  def valid_summaries, do: [:auto, :concise, :detailed]
end
