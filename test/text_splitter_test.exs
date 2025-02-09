defmodule TextSplitterTest do
  use ExUnit.Case

  alias LangChain.TextSplitter

  describe "CharacterTextSplitter" do
    @tag :wip
    test "Splitting by character count" do
      text = "foo bar baz 123"
      expected_output = ["foo bar", "bar baz", "baz 123"]
      output = TextSplitter.split_text(text)
      assert expected_output == output
    end

    @tag :wip
    test "Splitting character by count doesn't create empty documents" do
      text = "foo  bar"
      expected_output = ["foo", "bar"]
      output = TextSplitter.split_text(text)
      assert expected_output == output
    end
  end
end
