defmodule LangChain.TextSplitter.CharacterTextSplitter do
  @moduledoc """
  The `CharacterTextSplitter` is a length based text splitter
  that divides text based on specified characters.
  This splitter provides consistent chunk sizes.
  It operates as follows:

  - It splits the text at specified `separator` characters.  
  - It takes a `chunk_size` parameter that determines the maximum number of characters
    in each chunk.
  - If no separator is found within the `chunk_size`,
    it will create a chunk larger than the specified size.

  The purpose is to prepare text for processing
  by large language models with limited context windows,
  or where a shorter context window is desired.

  A `CharacterTextSplitter` is defined using a schema.
  * `separator` - String that splits a given text.
  * `chunk_size` - Integer number of characters that a chunk should have.
  * `chunk_overlap` - Integer number of characters that two consecutive chunks should share.
  * `keep_separator` - Either `:discard_separator`, `:start` or `:end`. If `:discard_separator`, the separator is discarded from the output chunks. `:start` and `:end` keep the separator at the start or end of the output chunks. Defaults to `:discard_separator`.
  * `is_separator_regex` - Boolean defaulting to `false`. If `true`, the `separator` string is not escaped. Defaults to `false`
  """
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
    field :keep_separator, Ecto.Enum,
          values: [:discard_separator, :start, :end],
          default: :discard_separator
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

  @doc """
  Build a new CharacterTextSplitter and return an `:ok`/`:error` tuple with the result.
  """
  def new(attrs \\ %{}) do
    %TextSplitter.CharacterTextSplitter{}
    |> cast(attrs, @create_fields, empty_values: [nil])
    |> apply_action(:insert)
  end

  @doc """
  Build a new CharacterTextSplitter and return it or raise an error if invalid.
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
  Splits text based on a given character.
  By default, the `separator` character is discarded
      iex> text_splitter = CharacterTextSplitter.new!(%{separator: " ", chunk_size: 3, chunk_overlap: 0})
      iex> text = "foo bar baz"
      iex> CharacterTextSplitter.split_text(text_splitter, text)
      ["foo", "bar", "baz"]

  We can keep the separator at the end of a chunk, providing the
  `keep_separator: :end` option:
      iex> text_splitter = CharacterTextSplitter.new!(%{separator: ".", chunk_size: 3, chunk_overlap: 0, keep_separator: :end})
      iex> text = "foo.bar.baz"
      iex> CharacterTextSplitter.split_text(text_splitter, text)
      ["foo.", "bar.", "baz"]

  In order to keep the separator at the beginning of a chunk, provide the
  `keep_separator: :start` option:
      iex> text_splitter = CharacterTextSplitter.new!(%{separator: ".", chunk_size: 3, chunk_overlap: 0, keep_separator: :start})
      iex> text = "foo.bar.baz"
      iex> CharacterTextSplitter.split_text(text_splitter, text)
      ["foo", ".bar", ".baz"]

  The last two examples used a regex special character as a `separator`.
  Plain strings are escaped and parsed as regex before splitting.
  If you want to use a complex regex as `separator` you can,
  but make sure to pass the `is_separator_regex: true` option:
      iex> text_splitter = CharacterTextSplitter.new!(%{separator: Regex.escape("."), chunk_size: 3, chunk_overlap: 0, keep_separator: :start, is_separator_regex: true})
      iex> text = "foo.bar.baz"
      iex> CharacterTextSplitter.split_text(text_splitter, text)
      ["foo", ".bar", ".baz"]

  You can control the overlap of chunks trhough the `chunk_overlap` parameter:
      iex> text_splitter = CharacterTextSplitter.new!(%{separator: " ", chunk_size: 7, chunk_overlap: 3})
      iex> text = "foo bar baz"
      iex> CharacterTextSplitter.split_text(text_splitter, text)
      ["foo bar", "bar baz"]
  """
  def split_text(%CharacterTextSplitter{} = text_splitter, text) do
    text
    |> split_text_with_regex(text_splitter)
    |> TextSplitter.merge_splits(text_splitter, text_splitter.separator)
  end

  @doc false
  def split_text_with_regex(
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
