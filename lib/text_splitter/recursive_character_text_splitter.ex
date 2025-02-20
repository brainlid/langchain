defmodule LangChain.TextSplitter.RecursiveCharacterTextSplitter do
  use Ecto.Schema
  import Ecto.Changeset
  alias LangChain.LangChainError
  alias LangChain.TextSplitter
  alias __MODULE__

  @primary_key false
  embedded_schema do
    field :separators, {:array, :string}, default: ["\n\n", "\n", " ", ""]
    field :chunk_size, :integer
    field :chunk_overlap, :integer
    field :keep_separator, Ecto.Enum, values: [:start, :end]
    field :is_separator_regex, :boolean, default: false
  end

  @type t :: %RecursiveCharacterTextSplitter{}

  @update_fields [
    :separators,
    :chunk_size,
    :chunk_overlap,
    :keep_separator,
    :is_separator_regex
  ]
  @create_fields @update_fields

  @doc """
  Build a new RecursiveCharcterTextSplitter and return an `:ok`/`:error` tuple with the result.
  """
  def new(attrs \\ %{}) do
    %TextSplitter.RecursiveCharacterTextSplitter{}
    |> cast(attrs, @create_fields)
    |> apply_action(:insert)
  end

  @doc """
  Build a new RecursiveCharacterTextSplitter and return it or raise an error if invalid.
  """
  def new!(attrs \\ %{}) do
    case new(attrs) do
      {:ok, character_text_spliiter} ->
        character_text_spliiter

      {:error, changeset} ->
        raise LangChainError, changeset
    end
  end
  
end
