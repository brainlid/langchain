defmodule TextSplitterTest do
  use ExUnit.Case
  alias LangChain.TextSplitter.CharacterTextSplitter
  alias LangChain.TextSplitter.RecursiveCharacterTextSplitter
  alias LangChain.TextSplitter.LanguageSeparators
  doctest CharacterTextSplitter

  @chunk_size 16

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

      assert output_splitter == expected_splitter
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

      assert output_splitter == expected_splitter
    end

    test "Splitting by character count" do
      text = "foo bar baz 123"
      expected_output = ["foo bar", "bar baz", "baz 123"]

      character_splitter =
        CharacterTextSplitter.new!(%{separator: " ", chunk_size: 7, chunk_overlap: 3})

      output =
        character_splitter
        |> CharacterTextSplitter.split_text(text)

      assert output == expected_output
    end

    test "Splitting character by count doesn't create empty documents" do
      text = "foo  bar"
      expected_output = ["foo", "bar"]

      character_splitter =
        CharacterTextSplitter.new!(%{separator: " ", chunk_size: 2, chunk_overlap: 0})

      output =
        character_splitter
        |> CharacterTextSplitter.split_text(text)

      assert output == expected_output
    end

    test "Edge cases are separators" do
      text = "f b"
      expected_output = ["f", "b"]

      character_splitter =
        CharacterTextSplitter.new!(%{separator: " ", chunk_size: 2, chunk_overlap: 0})

      output =
        character_splitter
        |> CharacterTextSplitter.split_text(text)

      assert output == expected_output
    end

    test "Splitting by character count on long words" do
      text = "foo bar baz a a"
      expected_output = ["foo", "bar", "baz", "a a"]

      character_splitter =
        CharacterTextSplitter.new!(%{separator: " ", chunk_size: 3, chunk_overlap: 1})

      output =
        character_splitter
        |> CharacterTextSplitter.split_text(text)

      assert output == expected_output
    end

    test "Splitting by character count when shorter words are first" do
      text = "a a foo bar baz"
      expected_output = ["a a", "foo", "bar", "baz"]

      character_splitter =
        CharacterTextSplitter.new!(%{separator: " ", chunk_size: 3, chunk_overlap: 1})

      output =
        character_splitter
        |> CharacterTextSplitter.split_text(text)

      assert output == expected_output
    end

    test "Splitting by characters when splits not found easily" do
      text = "foo bar baz 123"
      expected_output = ["foo", "bar", "baz", "123"]

      character_splitter =
        CharacterTextSplitter.new!(%{separator: " ", chunk_size: 3, chunk_overlap: 1})

      output =
        character_splitter
        |> CharacterTextSplitter.split_text(text)

      assert output == expected_output
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

        assert output == expected_output
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

        assert output == expected_output
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

        assert output == expected_output
      end
    end
  end

  describe "RecursiveCharacterTextSplitter" do
    test "recursive_character_text_splitter" do
      split_tags = [",", "."]
      query = "Apple,banana,orange and tomato."
      expected_output_1 = ["Apple,", "banana,", "orange and tomato."]
      expected_output_2 = ["Apple", ",banana", ",orange and tomato", "."]
      base_params = %{chunk_size: 10, chunk_overlap: 0, separators: split_tags}

      test_data = [
        %{expected: expected_output_1, params: %{keep_separator: :end}},
        %{expected: expected_output_2, params: %{keep_separator: :start}}
      ]

      for tt <- test_data do
        splitter =
          RecursiveCharacterTextSplitter.new!(Map.merge(base_params, tt.params))

        output = splitter |> RecursiveCharacterTextSplitter.split_text(query)
        assert tt.expected == output
      end
    end

    test "Iterative splitter discard separator" do
      text = "....5X..3Y...4X....5Y..."

      base_params = %{
        chunk_overlap: 0,
        is_separator_regex: false,
        separators: ["X", "Y"]
      }

      expected_output_1 = [
        "....5",
        "..3",
        "...4",
        "....5",
        "..."
      ]

      expected_output_2 = [
        "....5",
        "X..3",
        "Y...4",
        "X....5",
        "Y..."
      ]

      test_data = [
        %{expected: expected_output_1, params: %{chunk_size: 5}},
        %{expected: expected_output_2, params: %{chunk_size: 6, keep_separator: :start}}
      ]

      for tt <- test_data do
        splitter =
          RecursiveCharacterTextSplitter.new!(Map.merge(base_params, tt.params))

        output = splitter |> RecursiveCharacterTextSplitter.split_text(text)
        assert tt.expected == output
      end
    end

    test "Iterative text splitter" do
      text = "Hi.\n\nI'm Iglesias.\n\nHow? Are? You?\nOkay then f f f f.
This is a weird text to write, but gotta test the splittingggg some how.

Bye!\n\n-I."

      expected_output = [
        "Hi.",
        "I'm",
        "Iglesias.",
        "How? Are?",
        "You?",
        "Okay then",
        "f f f f.",
        "This is a",
        "weird",
        "text to",
        "write,",
        "but gotta",
        "test the",
        "splitting",
        "gggg",
        "some how.",
        "Bye!",
        "-I."
      ]

      splitter =
        RecursiveCharacterTextSplitter.new!(%{
          keep_separator: :start,
          chunk_size: 10,
          chunk_overlap: 1
        })

      output = splitter |> RecursiveCharacterTextSplitter.split_text(text)
      assert output == expected_output
    end
  end

  describe "Programming languages splitters" do
    test "Python test splitter" do
      fake_python_text = """
      class Foo:

          def bar():


      def foo():

      def testing_func():

      def bar():
      """

      split_0 = "class Foo:\n\n    def bar():"
      split_1 = "def foo():"
      split_2 = "def testing_func():"
      split_3 = "def bar():"
      expected_splits = [split_0, split_1, split_2, split_3]

      splitter =
        RecursiveCharacterTextSplitter.new!(%{
          separators: LanguageSeparators.python(),
          keep_separator: :start,
          chunk_size: 30,
          chunk_overlap: 0
        })

      splits =
        splitter
        |> RecursiveCharacterTextSplitter.split_text(fake_python_text)

      assert splits == expected_splits
    end

    test "More python splitting" do
      code = "
def hello_world():
    print(\"Hello, World!\")

# Call the function
hello_world()
    "

      expected_splits = [
        "def",
        "hello_world():",
        "print(\"Hello,",
        "World!\")",
        "# Call the",
        "function",
        "hello_world()"
      ]

      splitter =
        RecursiveCharacterTextSplitter.new!(%{
          separators: LanguageSeparators.python(),
          keep_separator: :start,
          chunk_size: @chunk_size,
          chunk_overlap: 0
        })

      splits =
        splitter
        |> RecursiveCharacterTextSplitter.split_text(code)

      assert splits == expected_splits
    end

    test "Golang splitting" do
      code = "
package main

import \"fmt\"

func helloWorld() {
    fmt.Println(\"Hello, World!\")
}

func main() {
    helloWorld()
}
    "

      expected_splits = [
        "package main",
        "import \"fmt\"",
        "func",
        "helloWorld() {",
        "fmt.Println(\"He",
        "llo,",
        "World!\")",
        "}",
        "func main() {",
        "helloWorld()",
        "}"
      ]

      splitter =
        RecursiveCharacterTextSplitter.new!(%{
          separators: LanguageSeparators.go(),
          keep_separator: :start,
          chunk_size: @chunk_size,
          chunk_overlap: 0
        })

      splits =
        splitter
        |> RecursiveCharacterTextSplitter.split_text(code)

      assert splits == expected_splits
    end

    @tag :wip
    test "Rst splitting" do
      code = "
Sample Document
===============

Section
-------

This is the content of the section.

Lists
-------

- Item 1
- Item 2
- Item 3

Comment
*******
Not a comment

.. This is a comment
    "

      expected_splits = [
        "Sample Document",
        "===============",
        "Section",
        "-------",
        "This is the",
        "content of the",
        "section.",
        "Lists",
        "-------",
        "- Item 1",
        "- Item 2",
        "- Item 3",
        "Comment",
        "*******",
        "Not a comment",
        ".. This is a",
        "comment"
      ]

      splitter =
        RecursiveCharacterTextSplitter.new!(%{
          separators: LanguageSeparators.rst(),
          keep_separator: :start,
          chunk_size: @chunk_size,
          chunk_overlap: 0,
          is_separator_regex: true
        })

      splits =
        splitter
        |> RecursiveCharacterTextSplitter.split_text(code)

      assert splits == expected_splits
      code = "harry\n***\nbabylon is"

      chunks =
        splitter
        |> RecursiveCharacterTextSplitter.split_text(code)

      assert chunks == ["harry", "***\nbabylon is"]
    end

    test "Proto splitting" do
      code = "
syntax = \"proto3\";

package example;

message Person {
    string name = 1;
    int32 age = 2;
    repeated string hobbies = 3;
}      
    "

      expected_splits = [
        "syntax =",
        "\"proto3\";",
        "package",
        "example;",
        "message Person",
        "{",
        "string name",
        "= 1;",
        "int32 age =",
        "2;",
        "repeated",
        "string hobbies",
        "= 3;",
        "}"
      ]

      splitter =
        RecursiveCharacterTextSplitter.new!(%{
          separators: LanguageSeparators.proto(),
          keep_separator: :start,
          chunk_size: @chunk_size,
          chunk_overlap: 0
        })

      splits =
        splitter
        |> RecursiveCharacterTextSplitter.split_text(code)

      assert splits == expected_splits
    end

    test "Javscript splitting" do
      code = "
function helloWorld() {
  console.log(\"Hello, World!\");
}

// Call the function
helloWorld();      
    "

      expected_splits = [
        "function",
        "helloWorld() {",
        "console.log(\"He",
        "llo,",
        "World!\");",
        "}",
        "// Call the",
        "function",
        "helloWorld();"
      ]

      splitter =
        RecursiveCharacterTextSplitter.new!(%{
          separators: LanguageSeparators.js(),
          keep_separator: :start,
          chunk_size: @chunk_size,
          chunk_overlap: 0
        })

      splits =
        splitter
        |> RecursiveCharacterTextSplitter.split_text(code)

      assert splits == expected_splits
    end

    test "Cobol splitting" do
      code = "
IDENTIFICATION DIVISION.
PROGRAM-ID. HelloWorld.
DATA DIVISION.
WORKING-STORAGE SECTION.
01 GREETING           PIC X(12)   VALUE 'Hello, World!'.
PROCEDURE DIVISION.
DISPLAY GREETING.
STOP RUN.
    "

      expected_splits = [
        "IDENTIFICATION",
        "DIVISION.",
        "PROGRAM-ID.",
        "HelloWorld.",
        "DATA DIVISION.",
        "WORKING-STORAGE",
        "SECTION.",
        "01 GREETING",
        "PIC X(12)",
        "VALUE 'Hello,",
        "World!'.",
        "PROCEDURE",
        "DIVISION.",
        "DISPLAY",
        "GREETING.",
        "STOP RUN."
      ]

      splitter =
        RecursiveCharacterTextSplitter.new!(%{
          separators: LanguageSeparators.cobol(),
          keep_separator: :start,
          chunk_size: @chunk_size,
          chunk_overlap: 0
        })

      splits =
        splitter
        |> RecursiveCharacterTextSplitter.split_text(code)

      assert splits == expected_splits
    end

    test "Typescript splitting" do
      code = "
function helloWorld(): void {
  console.log(\"Hello, World!\");
}

// Call the function
helloWorld();
    "

      expected_splits = [
        "function",
        "helloWorld():",
        "void {",
        "console.log(\"He",
        "llo,",
        "World!\");",
        "}",
        "// Call the",
        "function",
        "helloWorld();"
      ]

      splitter =
        RecursiveCharacterTextSplitter.new!(%{
          separators: LanguageSeparators.ts(),
          keep_separator: :start,
          chunk_size: @chunk_size,
          chunk_overlap: 0
        })

      splits =
        splitter
        |> RecursiveCharacterTextSplitter.split_text(code)

      assert splits == expected_splits
    end
  end
end
