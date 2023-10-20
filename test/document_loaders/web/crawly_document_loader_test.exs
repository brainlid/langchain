defmodule LangChain.DocumentLoaders.Web.CrawlyTest do
  use LangChain.BaseCase

  alias LangChain.DocumentLoaders.Web.Crawly

  describe "load/0" do
    test "fetches a single web document correctly" do
      opts = [url: "https://example.com"]
      {:ok, document} = Crawly.load(opts)
      assert document.metadata.source == "https://example.com"

      assert String.contains?(
               document.page_content,
               "This domain is for use in illustrative examples in documents."
             )
    end
  end
end
