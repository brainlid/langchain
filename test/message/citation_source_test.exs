defmodule LangChain.Message.CitationSourceTest do
  use ExUnit.Case
  alias LangChain.Message.CitationSource

  describe "new/1" do
    test "creates valid source with all fields" do
      assert {:ok, %CitationSource{} = source} =
               CitationSource.new(%{
                 type: :web,
                 title: "Example Page",
                 url: "https://example.com",
                 document_id: "doc_123",
                 metadata: %{"extra" => "data"}
               })

      assert source.type == :web
      assert source.title == "Example Page"
      assert source.url == "https://example.com"
      assert source.document_id == "doc_123"
      assert source.metadata == %{"extra" => "data"}
    end

    test "requires type" do
      assert {:error, changeset} = CitationSource.new(%{title: "No Type"})
      refute changeset.valid?
      assert {"can't be blank", _} = changeset.errors[:type]
    end

    test "accepts all valid source types" do
      for type <- [:web, :document, :place] do
        assert {:ok, %CitationSource{type: ^type}} = CitationSource.new(%{type: type})
      end
    end

    test "defaults metadata to empty map" do
      assert {:ok, %CitationSource{metadata: %{}}} = CitationSource.new(%{type: :web})
    end

    test "accepts string-keyed metadata map" do
      assert {:ok, %CitationSource{metadata: meta}} =
               CitationSource.new(%{
                 type: :document,
                 metadata: %{"provider_type" => "char_location", "start" => 0}
               })

      assert meta == %{"provider_type" => "char_location", "start" => 0}
    end
  end

  describe "new!/1" do
    test "returns struct on success" do
      assert %CitationSource{type: :web} = CitationSource.new!(%{type: :web})
    end

    test "raises on failure" do
      assert_raise LangChain.LangChainError, fn ->
        CitationSource.new!(%{})
      end
    end
  end
end
