defmodule LangChain.TextSplitter.RecursiveCharacterTextSplitter do
  use Ecto.Schema
  import Ecto.Changeset
  alias LangChain.LangChainError
  alias LangChain.TextSplitter
  alias LangChain.TextSplitter.CharacterTextSplitter
  alias __MODULE__

  @primary_key false
  embedded_schema do
    field :separators, {:array, :string}, default: ["\n\n", "\n", " ", ""]
    field :chunk_size, :integer
    field :chunk_overlap, :integer
    field :keep_separator, Ecto.Enum,
          values: [:discard_separator, :start, :end],
          default: :start
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
    |> cast(attrs, @create_fields, empty_values: [nil])
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

  def split_text(%RecursiveCharacterTextSplitter{} = text_splitter, text) do
    new_separators =
      text_splitter.separators
      |> Enum.map(fn s ->
        if text_splitter.is_separator_regex,
          do: s,
          else: s |> Regex.escape()
      end)
      |> Enum.drop_while(fn s -> not (s |> Regex.compile!() |> Regex.match?(text)) end)

    separator =
      if Enum.count(new_separators) > 0,
        do: List.first(new_separators),
        else:
          text_splitter.separators
          |> List.last()
          |> Regex.escape()

    character_text_splitter =
      CharacterTextSplitter.new!(
        text_splitter
        |> Map.from_struct()
        |> Map.delete(:separators)
        |> Map.put(:is_separator_regex, true)
        |> Map.put_new(:separator, separator)
      )

    acc = %{good_splits: [], final_chunks: []}

    splits =
      text
      |> CharacterTextSplitter.split_text_with_regex(character_text_splitter)

    merge_separator =
      if not (text_splitter.keep_separator == :discard_separator),
        do: "",
        else: character_text_splitter.separator

    recursive_splits =
      splits
      |> Enum.reduce(acc, fn split, acc ->
        if String.length(split) < text_splitter.chunk_size do
          %{acc | good_splits: acc.good_splits ++ [split]}
        else
          acc =
            if Enum.count(acc.good_splits) > 0 do
              merged_text =
                TextSplitter.merge_splits(
                  acc.good_splits,
                  character_text_splitter,
                  merge_separator
                )

              %{good_splits: [], final_chunks: acc.final_chunks ++ merged_text}
            else
              acc
            end

          if Enum.count(new_separators) <= 1 do
            %{acc | final_chunks: acc.final_chunks ++ [split]}
          else
            new_recursive_splitter =
              %{
                (text_splitter
                 |> Map.from_struct())
                | separators: new_separators |> Enum.drop(1),
                  is_separator_regex: true
              }
              |> RecursiveCharacterTextSplitter.new!()

            other_info =
              new_recursive_splitter
              |> RecursiveCharacterTextSplitter.split_text(split)

            %{acc | final_chunks: acc.final_chunks ++ other_info}
          end
        end
      end)

    if Enum.count(recursive_splits.good_splits) > 0 do
      merged_text =
        recursive_splits.good_splits
        |> TextSplitter.merge_splits(character_text_splitter, merge_separator)

      recursive_splits.final_chunks ++ merged_text
    else
      recursive_splits.final_chunks
    end
    |> Enum.filter(fn x -> not (x |> is_nil()) end)
  end
end
