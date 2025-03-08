defmodule LangChain.TextSplitter.RecursiveCharacterTextSplitter do
  @moduledoc """
  The `RecursiveCharacterTextSplitter` is the recommended spliltter for generic text.
  It splits the text based on a list of characters.
  It uses each of these characters sequentially, until the text is split
  into small enough chunks. The default list is `["\n\n", "\n", " ", ""]`.

  The purpose is to prepare text for processing
  by large language models with limited context windows,
  or where a shorter context window is desired.

  The main characterstinc of this splitter is that tries to keep
  paragraphs, sentences or code functions together as long as possible.

  `LangChain.TextSplitter.LanguageSeparators` provide separator lists for some programming and markup languages.
  To use these Separators, it's recommended to set the `is_separator_regex` option to `true`.

  How it works:
  - It splits the text at the first specified `separator` characters
    from the given `separators` list.
    It uses `LangChain.TextSplitter.CharacterTextSplitter` to do so.
  - For each of the above splits, it calls itself recursively
    using the tail of the `separators` list.

  A `RecursiveCharacterTextSplitter` is defined using a schema.
  * `separators` - List of string that split a given text.
    The default list is `["\n\n", "\n", " ", ""]`.
  * `chunk_size` - Integer number of characters that a chunk should have.
  * `chunk_overlap` - Integer number of characters that two consecutive chunks should share.
  * `keep_separator` - Either `:discard_separator`, `:start` or `:end`. If `nil`, the separator is discarded from the output chunks. `:start` and `:end` keep the separator at the start or end of the output chunks. Defaults to `start`.
  * `is_separator_regex` - Boolean defaulting to `false`. If `true`, the `separator` string is not escaped. Defaults to `false`
  """
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

  @doc """
  Splits text recursively based on a list of characters.
  By default, the `separators` characters are kept at the start
      iex> split_tags = [",", "."]
      iex> base_params = %{chunk_size: 10, chunk_overlap: 0, separators: split_tags}
      iex> query = "Apple,banana,orange and tomato."
      iex> splitter = RecursiveCharacterTextSplitter.new!(base_params)    
      iex> splitter |> RecursiveCharacterTextSplitter.split_text(query)
      ["Apple", ",banana", ",orange and tomato", "."]

  We can keep the separator at the end of a chunk, providing the
  `keep_separator: :end` option:
      iex> split_tags = [",", "."]
      iex> base_params = %{chunk_size: 10, chunk_overlap: 0, separators: split_tags, keep_separator: :end}
      iex> query = "Apple,banana,orange and tomato."
      iex> splitter = RecursiveCharacterTextSplitter.new!(base_params)    
      iex> splitter |> RecursiveCharacterTextSplitter.split_text(query)
      ["Apple,", "banana,", "orange and tomato."]

  See `LangChain.TextSplitter.CharacterTextSplitter` for the usage of the different options.

  `LanguageSeparators` provides `separators` for multiple
  programming and markdown languages.
  To use these Separators, it's recommended to set the `is_separator_regex` option to `true`.
  To split Python code:
      iex> python_code = "
      ...>def hello_world():
      ...>  print('Hello, World')
      ...>
      ...>            
      ...># Call the function
      ...>hello_world()"
      iex> splitter =
      ...>  RecursiveCharacterTextSplitter.new!(%{
      ...>    separators: LanguageSeparators.python(),
      ...>    keep_separator: :start,
      ...>    is_separator_regex: :true,
      ...>    chunk_size: 16,
      ...>    chunk_overlap: 0})
      iex> splitter |> RecursiveCharacterTextSplitter.split_text(python_code)
      ["def", "hello_world():", "print('Hello,", "World')", "# Call the", "function", "hello_world()"]
  """
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
