defmodule LangChain.Message.CitationTest do
  use ExUnit.Case
  alias LangChain.Message.Citation
  alias LangChain.Message.CitationSource

  defp web_citation(url, opts \\ []) do
    Citation.new!(%{
      cited_text: Keyword.get(opts, :cited_text, "some text"),
      source: %{type: :web, title: "Page", url: url},
      start_index: Keyword.get(opts, :start_index, 0),
      end_index: Keyword.get(opts, :end_index, 9)
    })
  end

  defp doc_citation(doc_id, opts \\ []) do
    Citation.new!(%{
      cited_text: Keyword.get(opts, :cited_text, "doc text"),
      source: %{type: :document, document_id: doc_id, title: "Doc"},
      start_index: 0,
      end_index: 8
    })
  end

  describe "new/1" do
    test "creates valid citation with embedded source" do
      assert {:ok, %Citation{} = citation} =
               Citation.new(%{
                 cited_text: "The sky is blue.",
                 source: %{
                   type: :web,
                   title: "Science Facts",
                   url: "https://example.com/science"
                 },
                 start_index: 10,
                 end_index: 26,
                 confidence: 0.95,
                 metadata: %{"provider_type" => "url_citation"}
               })

      assert citation.cited_text == "The sky is blue."
      assert citation.start_index == 10
      assert citation.end_index == 26
      assert citation.confidence == 0.95
      assert citation.metadata == %{"provider_type" => "url_citation"}

      assert %CitationSource{} = citation.source
      assert citation.source.type == :web
      assert citation.source.title == "Science Facts"
      assert citation.source.url == "https://example.com/science"
    end

    test "creates citation without source" do
      assert {:ok, %Citation{source: nil}} = Citation.new(%{cited_text: "text"})
    end

    test "creates citation with minimal fields" do
      assert {:ok, %Citation{}} = Citation.new(%{})
    end

    test "defaults metadata to empty map" do
      assert {:ok, %Citation{metadata: %{}}} = Citation.new(%{})
    end
  end

  describe "new!/1" do
    test "returns struct on success" do
      assert %Citation{cited_text: "hello"} = Citation.new!(%{cited_text: "hello"})
    end

    test "raises on invalid source" do
      assert_raise LangChain.LangChainError, fn ->
        Citation.new!(%{source: %{type: :invalid_type}})
      end
    end
  end

  describe "source_urls/1" do
    test "extracts unique URLs from citations" do
      citations = [
        web_citation("https://a.com"),
        web_citation("https://b.com"),
        web_citation("https://a.com")
      ]

      assert Citation.source_urls(citations) == ["https://a.com", "https://b.com"]
    end

    test "filters out nil URLs" do
      citations = [
        web_citation("https://a.com"),
        doc_citation("doc_1")
      ]

      assert Citation.source_urls(citations) == ["https://a.com"]
    end

    test "returns empty list for empty input" do
      assert Citation.source_urls([]) == []
    end
  end

  describe "filter_by_source_type/2" do
    test "filters citations by source type" do
      citations = [
        web_citation("https://a.com"),
        doc_citation("doc_1"),
        web_citation("https://b.com")
      ]

      web_only = Citation.filter_by_source_type(citations, :web)
      assert length(web_only) == 2
      assert Enum.all?(web_only, &(&1.source.type == :web))

      doc_only = Citation.filter_by_source_type(citations, :document)
      assert length(doc_only) == 1
      assert hd(doc_only).source.type == :document
    end

    test "returns empty list when no match" do
      citations = [web_citation("https://a.com")]
      assert Citation.filter_by_source_type(citations, :place) == []
    end
  end
end
