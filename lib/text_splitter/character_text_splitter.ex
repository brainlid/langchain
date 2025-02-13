defmodule LangChain.TextSplitter.CharacterTextSplitter do
  use Ecto.Schema
  import Ecto.Changeset
  alias LangChain.LangChainError
  alias LangChain.TextSplitter
  alias __MODULE__

  @primary_key false
  embedded_schema do
    field :separator, :string, default: " "
    field :chunk_size, :integer
    field :chunk_overlap, :integer
    field :keep_separator, Ecto.Enum, values: [:start, :end]
    field :is_separator_regex, :boolean, default: false
  end

  @type t :: %CharacterTextSplitter{}

  @update_fields [
    :separator,
    :chunk_size,
    :chunk_overlap,
    :keep_separator,
    :is_separator_regex
  ]
  @create_fields @update_fields

  def new(attrs \\ %{}) do
    %TextSplitter.CharacterTextSplitter{} 
    |> cast(attrs, @create_fields)
    |> apply_action(:insert)
  end

  def new!(attrs \\ %{}) do
    case new(attrs) do
      {:ok, character_text_spliiter} ->
        character_text_spliiter

      {:error, changeset} ->
        raise LangChainError, changeset
    end
  end

  def split_text(%CharacterTextSplitter{} = text_splitter, text) do
    text
    |> split_text_with_regex(text_splitter)
    |> TextSplitter.merge_splits(text_splitter)
  end

  defp split_text_with_regex(
         text,
         %CharacterTextSplitter{} = text_splitter
       ) do
    {:ok, separator} =
      if text_splitter.is_separator_regex do
        text_splitter.separator |> Regex.compile()
      else
        text_splitter.separator
        |> Regex.escape()
        |> Regex.compile()
      end

    chunk_and_join = fn x ->
      x
      |> Enum.chunk_every(2)
      |> Enum.map(&Enum.join(&1, ""))
    end

    if Enum.any?(
         [:end, :start],
         fn x -> x == text_splitter.keep_separator end
       ) do
      splits =
        separator
        |> Regex.split(text, include_captures: true)

      case text_splitter.keep_separator do
        :start ->
          [
            splits |> List.first()
            | splits
              |> Enum.drop(1)
              |> chunk_and_join.()
          ]

        :end ->
          splits
          |> chunk_and_join.()
      end
    else
      separator
      |> Regex.split(text)
    end
    |> Enum.filter(fn x -> x != "" end)
  end
end
