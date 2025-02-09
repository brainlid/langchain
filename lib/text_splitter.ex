defmodule LangChain.TextSplitter do
  use Ecto.Schema
  import Ecto.Changeset
  alias __MODULE__

  @primary_key false
  embedded_schema do
    field :separator, :string, default: " "
    field :chunk_size, :integer
    field :chunk_overlap, :integer
  end

  @type t :: %TextSplitter{}

  @update_fields [
   :separator,
   :chunk_size,
   :chunk_overlap
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
         %TextSplitter{} = text_splitter) do
    text
    |> String.split(text_splitter.separator)
    |> Enum.filter(fn x -> x != "" end)
  end

  defp join_docs(docs, separator), do: Enum.join(docs, separator)

  defp merge_split_helper(d, acc, text_splitter) do
    total = acc.total
    len = String.length(d)
    if (total <= text_splitter.chunk_overlap) or
         (total + len + String.length(text_splitter.separator)
         <= text_splitter.chunk_size) and
         (total <= 0) do
      acc
    else
      new_total = acc.total - (acc.current_doc
      |> Enum.at(0, "")
      |> String.length) - String.length(text_splitter.separator)

      new_current_doc =
        acc.current_doc
        |> Enum.drop(1)
      merge_split_helper(
        d,
        %{acc | total: new_total, current_doc: new_current_doc},
        text_splitter)
    end
  end

  defp merge_splits(splits, %TextSplitter{} = text_splitter) do
    separator_len = String.length(text_splitter.separator)
    first_split = List.first(splits)
    acc = %{current_doc: [first_split], docs: [], total: String.length(first_split)}
    output_acc = splits
    |> Enum.drop(1)
    |> Enum.reduce(
      acc,
      fn d, acc ->
        len = String.length(d)
        separator_length =
          if Enum.count(acc.current_doc) > 0,
             do: String.length(text_splitter.separator), else: 0
        
        acc =
          if (acc.total + len + separator_length) >
               text_splitter.chunk_size do
                if Enum.count(acc.current_doc) do
                  doc = join_docs(acc.current_doc, text_splitter.separator)
                  acc = %{acc | docs: acc.docs ++ [doc]}
                  merge_split_helper(d, acc, text_splitter)
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
    output_acc.docs ++ [join_docs(output_acc.current_doc, text_splitter.separator)]
  end
end
