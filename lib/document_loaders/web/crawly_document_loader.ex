defmodule LangChain.DocumentLoaders.Web.Crawly do
  alias Crawly
  alias LangChain.Document
  alias LangChain.DocumentLoaders.Web.Crawly.Spiders.BasicSpider

  @behaviour LangChain.DocumentLoaders
  def load(opts) do
    with response <-
           Crawly.fetch(Keyword.get(opts, :url)) |> IO.inspect(label: "value from fetch") do
      Document.new(%{metadata: %{source: Keyword.get(opts, :url)}, page_content: response.body})
    else
      {:error, err} -> {:error, err}
    end
  end
end
