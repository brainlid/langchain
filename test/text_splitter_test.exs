defmodule TextSplitterTest do
  use ExUnit.Case

  alias LangChain.TextSplitter

  describe "CharacterTextSplitter" do
    test "New TextSplitter" do
      expected_splitter = %TextSplitter{
        separator: " ",
        chunk_overlap: 0,
        chunk_size: 2,
      }
      assert {:ok, %TextSplitter{} = output_splitter} =
        %{separator: " ", chunk_overlap: 0, chunk_size: 2,}
        |> TextSplitter.new()
      assert expected_splitter == output_splitter
    end
    
    test "Splitting by character count" do
      text = "foo bar baz 123"
      expected_output = ["foo bar", "bar baz", "baz 123"]
      {:ok, character_splitter} =
        TextSplitter.new(
           %{separator: " ", chunk_size: 7, chunk_overlap: 3})
      output =
        character_splitter
        |> TextSplitter.split_text(text)
      assert expected_output == output
    end

    test "Splitting character by count doesn't create empty documents" do
      text = "foo  bar"
      expected_output = ["foo", "bar"]
      {:ok, character_splitter} =
        TextSplitter.new(
           %{separator: " ", chunk_size: 2, chunk_overlap: 0})
      output =
        character_splitter
        |> TextSplitter.split_text(text)
      assert expected_output == output
    end
  end
end
