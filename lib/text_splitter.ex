defmodule LangChain.TextSplitter do
  @moduledoc false
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

  @doc false
  def merge_splits(splits, text_splitter) do
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
