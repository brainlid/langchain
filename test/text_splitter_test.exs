defmodule TextSplitterTest do
  use ExUnit.Case

  alias LangChain.TextSplitter

  describe "CharacterTextSplitter" do
    @tag :wip
    test "text splitting by character count" do
      text = "foo bar baz 123"
      expected_output = ["foo bar", "bar baz", "baz 123"]
      output = TextSplitter.split_text(text)
      assert output == expected_output
    end
  end
  
end
