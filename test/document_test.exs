defmodule LangChain.DocumentTest do
  use ExUnit.Case
  doctest LangChain.Document
  alias LangChain.Document

  describe "new/1" do
    test "works with minimal attrs" do
      assert {:ok, %Document{} = document} =
               Document.new(%{
                 "page_content" => "Here's some content",
                 "metadata" => %{"source" => "https://example.com"}
               })

      assert document.page_content == "Here's some content"
      assert document.metadata["source"] == "https://example.com"
    end

    test "returns error when invalid" do
      assert {:error, changeset} = Document.new(%{"page_content" => nil})
      refute changeset.valid?
      assert {"can't be blank", _} = changeset.errors[:page_content]
    end
  end
end
