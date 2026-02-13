defmodule LangChain.Message.Citation do
  @moduledoc """
  Represents a citation linking a span of response text to a source.

  All four major providers (Anthropic, OpenAI, Google, Perplexity) share the
  pattern of linking response text to sources. This struct normalizes those
  provider-specific formats into a common shape.

  ## Fields

  - `:cited_text` - The actual text cited from the source (when available)
  - `:source` - A `CitationSource` identifying where the citation came from
  - `:start_index` - Character offset into the ContentPart's text where this
    citation starts
  - `:end_index` - Character offset where this citation ends
  - `:confidence` - Confidence score (0.0-1.0) when provided (e.g., Gemini)
  - `:metadata` - Provider-specific data for round-tripping. String keys for
    JSON serialization.
  """
  use Ecto.Schema
  import Ecto.Changeset
  alias LangChain.LangChainError
  alias LangChain.Message.CitationSource
  alias __MODULE__

  @primary_key false
  embedded_schema do
    field :cited_text, :string
    embeds_one :source, CitationSource
    field :start_index, :integer
    field :end_index, :integer
    field :confidence, :float

    # Provider-specific data. String keys for JSON serialization.
    field :metadata, :map, default: %{}
  end

  @type t :: %Citation{}

  @create_fields [:cited_text, :start_index, :end_index, :confidence, :metadata]

  @doc false
  def changeset(citation, attrs) do
    citation
    |> cast(attrs, @create_fields)
    |> cast_embed(:source)
  end

  @spec new(map()) :: {:ok, t()} | {:error, Ecto.Changeset.t()}
  def new(attrs \\ %{}) do
    %Citation{}
    |> changeset(attrs)
    |> apply_action(:insert)
  end

  @spec new!(map()) :: t() | no_return()
  def new!(attrs \\ %{}) do
    case new(attrs) do
      {:ok, struct} -> struct
      {:error, changeset} -> raise LangChainError, changeset
    end
  end

  @doc """
  Returns all unique source URLs from a list of citations.
  """
  @spec source_urls([t()]) :: [String.t()]
  def source_urls(citations) when is_list(citations) do
    citations
    |> Enum.map(& &1.source.url)
    |> Enum.reject(&is_nil/1)
    |> Enum.uniq()
  end

  @doc """
  Filters citations by source type.
  """
  @spec filter_by_source_type([t()], atom()) :: [t()]
  def filter_by_source_type(citations, type) when is_list(citations) do
    Enum.filter(citations, &(&1.source && &1.source.type == type))
  end
end
