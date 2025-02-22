defmodule TextSplitterTest do
  use ExUnit.Case
  alias LangChain.TextSplitter.CharacterTextSplitter
  alias LangChain.TextSplitter.RecursiveCharacterTextSplitter
  doctest CharacterTextSplitter

  describe "CharacterTextSplitter" do
    test "New TextSplitter" do
      expected_splitter = %CharacterTextSplitter{
        separator: " ",
        chunk_overlap: 0,
        chunk_size: 2
      }

      assert {:ok, %CharacterTextSplitter{} = output_splitter} =
               %{separator: " ", chunk_overlap: 0, chunk_size: 2}
               |> CharacterTextSplitter.new()

      assert expected_splitter == output_splitter
    end

    test "New TextSplitter with keep_separator" do
      expected_splitter = %CharacterTextSplitter{
        separator: " ",
        chunk_overlap: 0,
        chunk_size: 2,
        keep_separator: :start
      }

      assert {:ok, %CharacterTextSplitter{} = output_splitter} =
               %{separator: " ", chunk_overlap: 0, chunk_size: 2, keep_separator: :start}
               |> CharacterTextSplitter.new()

      assert expected_splitter == output_splitter
    end

    test "Splitting by character count" do
      text = "foo bar baz 123"
      expected_output = ["foo bar", "bar baz", "baz 123"]

      character_splitter =
        CharacterTextSplitter.new!(%{separator: " ", chunk_size: 7, chunk_overlap: 3})

      output =
        character_splitter
        |> CharacterTextSplitter.split_text(text)

      assert expected_output == output
    end

    test "Splitting character by count doesn't create empty documents" do
      text = "foo  bar"
      expected_output = ["foo", "bar"]

      character_splitter =
        CharacterTextSplitter.new!(%{separator: " ", chunk_size: 2, chunk_overlap: 0})

      output =
        character_splitter
        |> CharacterTextSplitter.split_text(text)

      assert expected_output == output
    end

    test "Edge cases are separators" do
      text = "f b"
      expected_output = ["f", "b"]

      character_splitter =
        CharacterTextSplitter.new!(%{separator: " ", chunk_size: 2, chunk_overlap: 0})

      output =
        character_splitter
        |> CharacterTextSplitter.split_text(text)

      assert expected_output == output
    end

    test "Splitting by character count on long words" do
      text = "foo bar baz a a"
      expected_output = ["foo", "bar", "baz", "a a"]

      character_splitter =
        CharacterTextSplitter.new!(%{separator: " ", chunk_size: 3, chunk_overlap: 1})

      output =
        character_splitter
        |> CharacterTextSplitter.split_text(text)

      assert expected_output == output
    end

    test "Splitting by character count when shorter words are first" do
      text = "a a foo bar baz"
      expected_output = ["a a", "foo", "bar", "baz"]

      character_splitter =
        CharacterTextSplitter.new!(%{separator: " ", chunk_size: 3, chunk_overlap: 1})

      output =
        character_splitter
        |> CharacterTextSplitter.split_text(text)

      assert expected_output == output
    end

    test "Splitting by characters when splits not found easily" do
      text = "foo bar baz 123"
      expected_output = ["foo", "bar", "baz", "123"]

      character_splitter =
        CharacterTextSplitter.new!(%{separator: " ", chunk_size: 3, chunk_overlap: 1})

      output =
        character_splitter
        |> CharacterTextSplitter.split_text(text)

      assert expected_output == output
    end

    test "Splitting by characters and keeping at start separator that is a regex special char" do
      text = "foo.bar.baz.123"
      expected_output = ["foo", ".bar", ".baz", ".123"]

      base_params = %{
        chunk_size: 1,
        chunk_overlap: 0,
        keep_separator: :start
      }

      test_params = [
        %{separator: ".", is_separator_regex: false},
        %{separator: Regex.escape("."), is_separator_regex: true}
      ]

      for tt <- test_params do
        character_splitter =
          CharacterTextSplitter.new!(Map.merge(base_params, tt))

        output =
          character_splitter
          |> CharacterTextSplitter.split_text(text)

        assert expected_output == output
      end
    end

    test "Splitting by characters and keeping at end separator that is a regex special char" do
      text = "foo.bar.baz.123"
      expected_output = ["foo.", "bar.", "baz.", "123"]

      base_params = %{
        chunk_size: 1,
        chunk_overlap: 0,
        keep_separator: :end
      }

      test_params = [
        %{separator: ".", is_separator_regex: false},
        %{separator: Regex.escape("."), is_separator_regex: true}
      ]

      for tt <- test_params do
        character_splitter =
          CharacterTextSplitter.new!(Map.merge(base_params, tt))

        output =
          character_splitter
          |> CharacterTextSplitter.split_text(text)

        assert expected_output == output
      end
    end

    test "Splitting by characters and discard separator that is a regex special char" do
      text = "foo.bar.baz.123"
      expected_output = ["foo", "bar", "baz", "123"]

      base_params = %{
        chunk_size: 1,
        chunk_overlap: 0
      }

      test_params = [
        %{separator: ".", is_separator_regex: false},
        %{separator: Regex.escape("."), is_separator_regex: true}
      ]

      for tt <- test_params do
        character_splitter =
          CharacterTextSplitter.new!(Map.merge(base_params, tt))

        output =
          character_splitter
          |> CharacterTextSplitter.split_text(text)

        assert expected_output == output
      end
    end
  end

  describe "RecursiveCharacterTextSplitter" do
    test "recursive_character_text_splitter" do
      split_tags = [",", "."]
      query = "Apple,banana,orange and tomato."
      expected_output_1 = ["Apple,", "banana,", "orange and tomato."]
      expected_output_2 = ["Apple", ",banana", ",orange and tomato", "."]
      base_params = %{chunk_size: 10,
                      chunk_overlap: 0,
                      separators: split_tags}
      test_data = [
        %{expected: expected_output_1, params: %{keep_separator: :end}},
        %{expected: expected_output_2, params: %{keep_separator: :start}}
      ]
      for tt <- test_data do
        splitter =
          RecursiveCharacterTextSplitter.new!(
            Map.merge(base_params, tt.params))
        output = splitter |> RecursiveCharacterTextSplitter.split_text(query)
        assert tt.expected == output        
      end
    end

    @tag :wip
    test "Iterative splitter discard separator" do
      text = "....5X..3Y...4X....5Y..."
      base_params = %{
        chunk_overlap: 0,
        is_separator_regex: false,
        separators: ["X", "Y"],
      }
      expected_output_1 = [
        "....5",
        "..3",
        "...4",
        "....5",
        "..."]
      expected_output_2 = [
        "....5",
        "X..3",
        "Y...4",
        "X....5",
        "Y..."]      
      test_data = [
        %{expected: expected_output_1,
          params: %{chunk_size: 5}},
        %{expected: expected_output_2,
          params: %{chunk_size: 6, keep_separator: :start}},        
      ]
      for tt <- test_data do
        splitter =
          RecursiveCharacterTextSplitter.new!(
            Map.merge(base_params, tt.params))
        output = splitter |> RecursiveCharacterTextSplitter.split_text(text)
        assert tt.expected == output
      end
    end
  end
end
