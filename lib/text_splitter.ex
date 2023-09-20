defprotocol Langchain.TextSplitter do
  def split_strings(splitter, documents)
  def split_text(splitter, text)
end

