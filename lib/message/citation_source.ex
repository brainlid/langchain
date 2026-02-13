defmodule LangChain.Message.CitationSource do
  @moduledoc """
  Represents the source of a citation - where the cited information came from.

  ## Source Types

  - `:web` - A URL-based web source (search results, web pages)
  - `:document` - A document provided in the request (Anthropic documents,
    OpenAI files)
  - `:place` - A geographic location (Google Maps)
  """
  use Ecto.Schema
  import Ecto.Changeset
  alias LangChain.LangChainError
  alias __MODULE__

  @primary_key false
  embedded_schema do
    field :type, Ecto.Enum, values: [:web, :document, :place]
    field :title, :string
    field :url, :string
    field :document_id, :string

    # Provider-specific data. String keys for JSON serialization.
    field :metadata, :map, default: %{}
  end

  @type t :: %CitationSource{}

  @create_fields [:type, :title, :url, :document_id, :metadata]

  @doc false
  def changeset(source, attrs) do
    source
    |> cast(attrs, @create_fields)
    |> validate_required([:type])
  end

  @spec new(map()) :: {:ok, t()} | {:error, Ecto.Changeset.t()}
  def new(attrs \\ %{}) do
    %CitationSource{}
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
end
