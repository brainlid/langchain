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

    test "Java splitting" do
      code = "
public class HelloWorld {
    public static void main(String[] args) {
        System.out.println(\"Hello, World!\");
    }
}
    "

      expected_splits = [
        "public class",
        "HelloWorld {",
        "public",
        "static void",
        "main(String[]",
        "args) {",
        "System.out.prin",
        "tln(\"Hello,",
        "World!\");",
        "}\n}"
      ]

      splitter =
        RecursiveCharacterTextSplitter.new!(%{
          separators: LanguageSeparators.java(),
          keep_separator: :start,
          chunk_size: @chunk_size,
          chunk_overlap: 0
        })

      splits =
        splitter
        |> RecursiveCharacterTextSplitter.split_text(code)

      assert splits == expected_splits
    end

    test "Kotlin splitting" do
      code = "
class HelloWorld {
    companion object {
        @JvmStatic
        fun main(args: Array<String>) {
            println(\"Hello, World!\")
        }
    }
}
    "

      expected_splits = [
        "class",
        "HelloWorld {",
        "companion",
        "object {",
        "@JvmStatic",
        "fun",
        "main(args:",
        "Array<String>)",
        "{",
        "println(\"Hello,",
        "World!\")",
        "}\n    }",
        "}"
      ]

      splitter =
        RecursiveCharacterTextSplitter.new!(%{
          separators: LanguageSeparators.kotlin(),
          keep_separator: :start,
          chunk_size: @chunk_size,
          chunk_overlap: 0
        })

      splits =
        splitter
        |> RecursiveCharacterTextSplitter.split_text(code)

      assert splits == expected_splits
    end

    test "Csharp splitting" do
      code = "
using System;
class Program
{
    static void Main()
    {
        int age = 30; // Change the age value as needed

        // Categorize the age without any console output
        if (age < 18)
        {
            // Age is under 18
        }
        else if (age >= 18 && age < 65)
        {
            // Age is an adult
        }
        else
        {
            // Age is a senior citizen
        }
    }
}
    "

      expected_splits = [
        "using System;",
        "class Program\n{",
        "static void",
        "Main()",
        "{",
        "int age",
        "= 30; // Change",
        "the age value",
        "as needed",
        "//",
        "Categorize the",
        "age without any",
        "console output",
        "if (age",
        "< 18)",
        "{",
        "//",
        "Age is under 18",
        "}",
        "else if",
        "(age >= 18 &&",
        "age < 65)",
        "{",
        "//",
        "Age is an adult",
        "}",
        "else",
        "{",
        "//",
        "Age is a senior",
        "citizen",
        "}\n    }",
        "}"
      ]

      splitter =
        RecursiveCharacterTextSplitter.new!(%{
          separators: LanguageSeparators.csharp(),
          keep_separator: :start,
          chunk_size: @chunk_size,
          chunk_overlap: 0
        })

      splits =
        splitter
        |> RecursiveCharacterTextSplitter.split_text(code)

      assert splits == expected_splits
    end
    
    test "C and C++ splitting" do
      code = "
#include <iostream>

int main() {
    std::cout << \"Hello, World!\" << std::endl;
    return 0;
}
    "

      expected_splits = [
       "#include",
        "<iostream>",
        "int main() {",
        "std::cout",
        "<< \"Hello,",
        "World!\" <<",
        "std::endl;",
        "return 0;\n}",
      ]

      splitter =
        RecursiveCharacterTextSplitter.new!(%{
          separators: LanguageSeparators.c(),
          keep_separator: :start,
          chunk_size: @chunk_size,
          chunk_overlap: 0
        })

      splits =
        splitter
        |> RecursiveCharacterTextSplitter.split_text(code)

      assert splits == expected_splits
    end
    
    test "Scala splitting" do
      code = "
object HelloWorld {
  def main(args: Array[String]): Unit = {
    println(\"Hello, World!\")
  }
}
    "

      expected_splits = [
       "object",
        "HelloWorld {",
        "def",
        "main(args:",
        "Array[String]):",
        "Unit = {",
        "println(\"Hello,",
        "World!\")",
        "}\n}",
      ]

      splitter =
        RecursiveCharacterTextSplitter.new!(%{
          separators: LanguageSeparators.scala(),
          keep_separator: :start,
          chunk_size: @chunk_size,
          chunk_overlap: 0
        })

      splits =
        splitter
        |> RecursiveCharacterTextSplitter.split_text(code)

      assert splits == expected_splits
    end
    
    test "Ruby splitting" do
      code = "
def hello_world
  puts \"Hello, World!\"
end

hello_world
    "

      expected_splits = [
        "def hello_world",
        "puts \"Hello,",
        "World!\"",
        "end",
        "hello_world",
      ]

      splitter =
        RecursiveCharacterTextSplitter.new!(%{
          separators: LanguageSeparators.ruby(),
          keep_separator: :start,
          chunk_size: @chunk_size,
          chunk_overlap: 0
        })

      splits =
        splitter
        |> RecursiveCharacterTextSplitter.split_text(code)

      assert splits == expected_splits
    end
    
    test "Php splitting" do
      code = "
<?php
function hello_world() {
    echo \"Hello, World!\";
}

hello_world();
?>
    "

      expected_splits = [
        "<?php",
        "function",
        "hello_world() {",
        "echo",
        "\"Hello,",
        "World!\";",
        "}",
        "hello_world();",
        "?>",
      ]

      splitter =
        RecursiveCharacterTextSplitter.new!(%{
          separators: LanguageSeparators.php(),
          keep_separator: :start,
          chunk_size: @chunk_size,
          chunk_overlap: 0
        })

      splits =
        splitter
        |> RecursiveCharacterTextSplitter.split_text(code)

      assert splits == expected_splits
    end

    test "Swift splitting" do
      code = "
func helloWorld() {
    print(\"Hello, World!\")
}

helloWorld()
    "

      expected_splits = [
        "func",
        "helloWorld() {",
        "print(\"Hello,",
        "World!\")",
        "}",
        "helloWorld()",
      ]

      splitter =
        RecursiveCharacterTextSplitter.new!(%{
          separators: LanguageSeparators.swift(),
          keep_separator: :start,
          chunk_size: @chunk_size,
          chunk_overlap: 0
        })

      splits =
        splitter
        |> RecursiveCharacterTextSplitter.split_text(code)

      assert splits == expected_splits
    end
    
    test "Rust splitting" do
      code = "
fn main() {
    println!(\"Hello, World!\");
}
    "

      expected_splits = [
        "fn main() {", "println!(\"Hello", ",", "World!\");", "}"
      ]

      splitter =
        RecursiveCharacterTextSplitter.new!(%{
          separators: LanguageSeparators.rust(),
          keep_separator: :start,
          chunk_size: @chunk_size,
          chunk_overlap: 0
        })

      splits =
        splitter
        |> RecursiveCharacterTextSplitter.split_text(code)

      assert splits == expected_splits
    end
    test "Markdown splitting" do
      code = "
# Sample Document

## Section

This is the content of the section.

## Lists

- Item 1
- Item 2
- Item 3

### Horizontal lines

***********
____________
-------------------

#### Code blocks
```
This is a code block

# sample code
a = 1
b = 2
```
    "

      expected_splits = [
     "# Sample",
        "Document",
        "## Section",
        "This is the",
        "content of the",
        "section.",
        "## Lists",
        "- Item 1",
        "- Item 2",
        "- Item 3",
        "### Horizontal",
        "lines",
        "***********",
        "____________",
        "---------------",
        "----",
        "#### Code",
        "blocks",
        "```",
        "This is a code",
        "block",
        "# sample code",
        "a = 1\nb = 2",
        "```",
      ]

      splitter =
        RecursiveCharacterTextSplitter.new!(%{
          separators: LanguageSeparators.markdown(),
          keep_separator: :start,
          chunk_size: @chunk_size,
          chunk_overlap: 0
        })

      splits =
        splitter
        |> RecursiveCharacterTextSplitter.split_text(code)

      assert splits == expected_splits
    end
    
    test "Latex splitting" do
      code = "
Hi Harrison!
\\chapter{1}
    "

      expected_splits = ["Hi Harrison!", "\\chapter{1}"]

      splitter =
        RecursiveCharacterTextSplitter.new!(%{
          separators: LanguageSeparators.latex(),
          is_separator_regex: true,
          keep_separator: :start,
          chunk_size: @chunk_size,
          chunk_overlap: 0
        })

      splits =
        splitter
        |> RecursiveCharacterTextSplitter.split_text(code)

      assert splits == expected_splits
    end
    
    test "Html splitting" do
      code = "
<h1>Sample Document</h1>
    <h2>Section</h2>
        <p id=\"1234\">Reference content.</p>

    <h2>Lists</h2>
        <ul>
            <li>Item 1</li>
            <li>Item 2</li>
            <li>Item 3</li>
        </ul>

        <h3>A block</h3>
            <div class=\"amazing\">
                <p>Some text</p>
                <p>Some more text</p>
            </div>
    "

      expected_splits = [
        "<h1>Sample Document</h1>\n    <h2>Section</h2>",
        "<p id=\"1234\">Reference content.</p>",
        "<h2>Lists</h2>\n        <ul>",
        "<li>Item 1</li>\n            <li>Item 2</li>",
        "<li>Item 3</li>\n        </ul>",
        "<h3>A block</h3>",
        "<div class=\"amazing\">",
        "<p>Some text</p>",
        "<p>Some more text</p>\n            </div>",        
      ]

      splitter =
        RecursiveCharacterTextSplitter.new!(%{
          separators: LanguageSeparators.html(),
          keep_separator: :start,
          chunk_size: 60,
          chunk_overlap: 0
        })

      splits =
        splitter
        |> RecursiveCharacterTextSplitter.split_text(code)

      assert splits == expected_splits
    end
    
    test "Solidity splitting" do
      code = "
pragma solidity ^0.8.20;
  contract HelloWorld {
    function add(uint a, uint b) pure public returns(uint) {
      return  a + b;
    }
  }
    "

      expected_splits = [
        "pragma solidity",
        "^0.8.20;",
        "contract",
        "HelloWorld {",
        "function",
        "add(uint a,",
        "uint b) pure",
        "public",
        "returns(uint) {",
        "return  a",
        "+ b;",
        "}\n  }",      
      ]

      splitter =
        RecursiveCharacterTextSplitter.new!(%{
          separators: LanguageSeparators.sol(),
          keep_separator: :start,
          chunk_size: @chunk_size,
          chunk_overlap: 0
        })

      splits =
        splitter
        |> RecursiveCharacterTextSplitter.split_text(code)

      assert splits == expected_splits
    end
    
    test "Lua splitting" do
      code = "
local variable = 10

function add(a, b)
    return a + b
end

if variable > 5 then
    for i=1, variable do
        while i < variable do
            repeat
                print(i)
                i = i + 1
            until i >= variable
        end
    end
end
    "

      expected_splits = [
        "local variable",
        "= 10",
        "function add(a,",
        "b)",
        "return a +",
        "b",
        "end",
        "if variable > 5",
        "then",
        "for i=1,",
        "variable do",
        "while i",
        "< variable do",
        "repeat",
        "print(i)",
        "i = i + 1",
        "until i >=",
        "variable",
        "end",
        "end\nend",
      ]

      splitter =
        RecursiveCharacterTextSplitter.new!(%{
          separators: LanguageSeparators.lua(),
          keep_separator: :start,
          chunk_size: @chunk_size,
          chunk_overlap: 0
        })

      splits =
        splitter
        |> RecursiveCharacterTextSplitter.split_text(code)

      assert splits == expected_splits
    end
    test "Haskell splitting" do
      code = "
        main :: IO ()
        main = do
          putStrLn \"Hello, World!\"

        -- Some sample functions
        add :: Int -> Int -> Int
        add x y = x + y      
    "

      expected_splits = [
        "main ::",
        "IO ()",
        "main = do",
        "putStrLn",
        "\"Hello, World!\"",
        "--",
        "Some sample",
        "functions",
        "add :: Int ->",
        "Int -> Int",
        "add x y = x",
        "+ y",
      ]

      splitter =
        RecursiveCharacterTextSplitter.new!(%{
          separators: LanguageSeparators.haskell(),
          is_separator_regex: true,
          keep_separator: :start,
          chunk_size: @chunk_size,
          chunk_overlap: 0
        })

      splits =
        splitter
        |> RecursiveCharacterTextSplitter.split_text(code)

      assert splits == expected_splits
    end
    
    test "Powershell short code splitting" do
      code = "
# Check if a file exists
$filePath = \"C:\\temp\\file.txt\"
if (Test-Path $filePath) {
    # File exists
} else {
    # File does not exist
}
    "

      expected_splits = [
        "# Check if a file exists\n$filePath = \"C:\\temp\\file.txt\"",
        "if (Test-Path $filePath) {\n    # File exists\n} else {",
        "# File does not exist\n}",
      ]

      splitter =
        RecursiveCharacterTextSplitter.new!(%{
          separators: LanguageSeparators.powershell(),
          is_separator_regex: true,
          keep_separator: :start,
          chunk_size: 60,
          chunk_overlap: 0
        })

      splits =
        splitter
        |> RecursiveCharacterTextSplitter.split_text(code)

      assert splits == expected_splits
    end
    
    test "Powershell long code splitting" do
      code = "
# Get a list of all processes and export to CSV
$processes = Get-Process
$processes | Export-Csv -Path \"C:\\temp\\processes.csv\" -NoTypeInformation

# Read the CSV file and display its content
$csvContent = Import-Csv -Path \"C:\\temp\\processes.csv\"
$csvContent | ForEach-Object {
    $_.ProcessName
}

# End of script
    "

      expected_splits = [
        "# Get a list of all processes and export to CSV",
        "$processes = Get-Process",
        "$processes | Export-Csv -Path \"C:\\temp\\processes.csv\"",
        "-NoTypeInformation",
        "# Read the CSV file and display its content",
        "$csvContent = Import-Csv -Path \"C:\\temp\\processes.csv\"",
        "$csvContent | ForEach-Object {\n    $_.ProcessName\n}",
        "# End of script",        
      ]

      splitter =
        RecursiveCharacterTextSplitter.new!(%{
          separators: LanguageSeparators.powershell(),
          is_separator_regex: true,
          keep_separator: :start,
          chunk_size: 60,
          chunk_overlap: 0
        })

      splits =
        splitter
        |> RecursiveCharacterTextSplitter.split_text(code)

      assert splits == expected_splits
    end
  end
end
