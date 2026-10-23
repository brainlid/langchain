defmodule LangChain.NativeToolTest do
  use ExUnit.Case

  alias LangChain.LangChainError
  alias LangChain.NativeTool

  describe "tool_search/1" do
    test "defaults to the bm25 variant" do
      assert %NativeTool{name: "tool_search", configuration: %{"variant" => "bm25"}} =
               NativeTool.tool_search()
    end

    test "accepts the regex variant" do
      assert %NativeTool{configuration: %{"variant" => "regex"}} =
               NativeTool.tool_search(variant: :regex)
    end

    test "rejects an unknown variant" do
      assert_raise LangChainError, ~r/Unsupported tool search variant/, fn ->
        NativeTool.tool_search(variant: :semantic)
      end
    end
  end

  describe "tool_search?/1" do
    test "recognizes the search tool" do
      assert NativeTool.tool_search?(NativeTool.tool_search())
    end

    test "is false for other tools" do
      refute NativeTool.tool_search?(
               NativeTool.new!(%{name: "google_search", configuration: %{}})
             )

      refute NativeTool.tool_search?(nil)
    end
  end
end
