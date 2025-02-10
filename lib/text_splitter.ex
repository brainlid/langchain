defmodule LangChain.TextSplitter do
  use Ecto.Schema
  import Ecto.Changeset
  alias __MODULE__

  @primary_key false
  embedded_schema do
    field :separator, :string, default: " "
    field :chunk_size, :integer
    field :chunk_overlap, :integer
    field :keep_separator, Ecto.Enum, values: [:start, :end]
    field :is_separator_regex, :boolean, default: false
  end

  @type t :: %TextSplitter{}

  @update_fields [
    :separator,
    :chunk_size,
    :chunk_overlap,
    :keep_separator,
    :is_separator_regex
  ]
  @create_fields @update_fields

  def new(attrs \\ %{}) do
    %TextSplitter{}
    |> cast(attrs, @create_fields)
    |> apply_action(:insert)
  end

  def split_text(%TextSplitter{} = text_splitter, text) do
    text
    |> split_text_with_regex(text_splitter)
    |> merge_splits(text_splitter)
  end

  defp split_text_with_regex(
         text,
         %TextSplitter{} = text_splitter
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

  defp join_docs(docs, separator), do: Enum.join(docs, separator)

  defp merge_split_helper(d, acc, text_splitter) do
    separator_len = String.length(text_splitter.separator)
    len = String.length(d)

    test_separator_length =
      if Enum.count(acc.current_doc) > 0, do: separator_len, else: 0

    if not (acc.total > text_splitter.chunk_overlap or
              (acc.total + len + test_separator_length >
                 text_splitter.chunk_size and
                 acc.total > 0)) do
      acc
    else
      separator_length =
        if Enum.count(acc.current_doc) > 1, do: separator_len, else: 0

      new_total =
        acc.total -
          (acc.current_doc
           |> Enum.at(0, "")
           |> String.length()) - separator_length

      new_current_doc = acc.current_doc |> Enum.drop(1)

      merge_split_helper(
        d,
        %{acc | total: new_total, current_doc: new_current_doc},
        text_splitter
      )
    end
  end

  defp merge_splits(splits, %TextSplitter{} = text_splitter) do
    acc = %{current_doc: [], docs: [], total: 0}

    output_acc =
      splits
      |> Enum.reduce(
        acc,
        fn d, acc ->
          len = String.length(d)

          separator_length =
            if Enum.count(acc.current_doc) > 0,
              do: String.length(text_splitter.separator),
              else: 0

          acc =
            if acc.total + len + separator_length >
                 text_splitter.chunk_size do
              if Enum.count(acc.current_doc) > 0 do
                doc = join_docs(acc.current_doc, text_splitter.separator)
                acc = %{acc | docs: acc.docs ++ [doc]}
                merge_split_helper(d, acc, text_splitter)
              else
                acc
              end
            else
              acc
            end

          acc = %{acc | current_doc: acc.current_doc ++ [d]}

          separator_length =
            if Enum.count(acc.current_doc) > 1,
              do: String.length(text_splitter.separator),
              else: 0

          %{acc | total: acc.total + separator_length + len}
        end
      )

    output_acc.docs ++ [join_docs(output_acc.current_doc, text_splitter.separator)]
  end
end
