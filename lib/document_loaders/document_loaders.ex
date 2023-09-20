defmodule Langchain.DocumentLoaders do
  alias Langchain.Document

  @callback load(options :: keyword()) :: %Document{}
  @callback load_and_split(splitter :: Langchain.TextSplitter, options :: keyword()) :: %Document{}
end