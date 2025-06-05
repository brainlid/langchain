defmodule LangChain.Document.TextDocuemntTest do
  use ExUnit.Case
  doctest LangChain.Document
  alias LangChain.Document
  alias LangChain.LangChainError

  describe "new/1" do
    test "works with basic attrs" do
      assert {:ok, %Document{} = document} =
               Document.new(%{
                 "content" => "Here's some content",
                 "type" => "plain_text",
                 "metadata" => %{"source" => "https://example.com"}
               })

      assert document.type == "plain_text"
      assert document.content == "Here's some content"
      assert document.metadata["source"] == "https://example.com"
    end

    test "returns error when invalid" do
      assert {:error, changeset} = Document.new(%{"content" => nil})
      refute changeset.valid?
      assert {"can't be blank", _} = changeset.errors[:content]
      assert {"can't be blank", _} = changeset.errors[:type]
    end
  end

  describe "new!/1" do
    test "throws when invalid" do
      assert_raise LangChainError, fn ->
        Document.new!(%{"content" => nil})
      end
    end
  end
end
