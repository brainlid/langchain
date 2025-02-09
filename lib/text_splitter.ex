defmodule LangChain.TextSplitter do
  @separator " "
  @chunk_size 7
  @chunk_overlap 3
  
  def split_text(text) do
    text
    |> split_text_with_regex
    |> merge_splits
  end

  defp split_text_with_regex(text) do
    String.split(text, @separator)
  end

  defp join_docs(docs), do: Enum.join(docs, @separator)

  defp merge_split_helper(d, acc) do
    IO.puts("RECURSION!!!!!!!!!!!!!!!!!!!")
    IO.inspect(acc)
    total = acc.total
    len = String.length(d)
    if (total <= @chunk_overlap) or
         (total + len + String.length(@separator) <= @chunk_size) and
         (total <= 0) do
      acc
    else
      new_total = acc.total - (acc.current_doc
      |> Enum.at(0, "")
      |> String.length) - String.length(@separator)

      new_current_doc =
        acc.current_doc
        |> Enum.drop(1)
      merge_split_helper(
        d,
        %{acc | total: new_total, current_doc: new_current_doc})
    end
  end

  defp merge_splits(splits) do
    separator_len = String.length(@separator)
    first_split = List.first(splits)
    acc = %{current_doc: [first_split], docs: [], total: String.length(first_split)}
    output_acc = splits
    |> Enum.drop(1)
    |> Enum.reduce(
      acc,
      fn d, acc ->
        IO.puts("REDUCTION!!!!!")
        IO.inspect(acc)
        len = String.length(d)
        separator_length =
          if Enum.count(acc.current_doc) > 0,
             do: String.length(@separator), else: 0
        
        acc =
          if (acc.total + len + separator_length) > @chunk_size do
                if Enum.count(acc.current_doc) do
                  doc = join_docs(acc.current_doc)
                  acc = %{acc | docs: acc.docs ++ [doc]}
                  merge_split_helper(d, acc)
                else
                  acc
                end
              else
                acc
        end
        %{acc |
          current_doc: acc.current_doc ++ [d],
          total: acc.total + separator_len + len
        }
      end
    )
    output_acc.docs ++ [join_docs(output_acc.current_doc)]
  end
end
