defmodule Langchain.DocumentLoaders.Web.Crawly do
  alias Crawly
  alias Langchain.Document
  @behaviour Langchain.DocumentLoaders

  @impl Langchain.DocumentLoaders
  def load(opts) do
      response = Crawly.fetch(Keyword.get(opts, :url))

      document = Document.new(%{metadata: %{source: Keyword.get(opts, :url)}, page_content: response.body})
      IO.inspect(document, label: "Crawly response")
  end

  @impl Langchain.DocumentLoaders
  def load_and_split(splitter \\ Langchain.TextSplitter, options) do

  end
end