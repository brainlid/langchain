defmodule LangChain.Utils.GoogleSchemaTest do
  use ExUnit.Case
  doctest LangChain.Utils.GoogleSchema

  alias LangChain.Utils.GoogleSchema

  describe "sanitize/1" do
    test "removes keywords Google's Schema type has no field for" do
      schema = %{
        "type" => "object",
        "additionalProperties" => false,
        "$schema" => "http://json-schema.org/draft-07/schema#",
        "unevaluatedProperties" => false,
        "patternProperties" => %{"^x-" => %{"type" => "string"}},
        "propertyNames" => %{"pattern" => "^[a-z]+$"},
        "required" => ["answer"],
        "properties" => %{"answer" => %{"type" => "string"}}
      }

      assert %{
               "type" => "object",
               "required" => ["answer"],
               "properties" => %{"answer" => %{"type" => "string"}}
             } == GoogleSchema.sanitize(schema)
    end

    test "removes the composition keywords Google does not support" do
      schema = %{
        "type" => "object",
        "properties" => %{
          "value" => %{
            "type" => "string",
            "oneOf" => [%{"const" => "a"}, %{"const" => "b"}],
            "allOf" => [%{"minLength" => 1}],
            "not" => %{"const" => ""},
            "if" => %{"minLength" => 1},
            "then" => %{"maxLength" => 5},
            "else" => %{"maxLength" => 10}
          }
        }
      }

      assert %{"properties" => %{"value" => %{"type" => "string"}}} =
               GoogleSchema.sanitize(schema)
    end

    test "keeps the keywords Google's Schema type declares" do
      schema = %{
        "type" => "integer",
        "format" => "int32",
        "title" => "Score",
        "description" => "The score.",
        "nullable" => true,
        "minimum" => 0,
        "maximum" => 10,
        "default" => 5,
        "example" => 7,
        "pattern" => "^[0-9]+$",
        "minLength" => 1,
        "maxLength" => 2,
        "propertyOrdering" => ["a", "b"]
      }

      assert schema == GoogleSchema.sanitize(schema)
    end

    test "recurses through properties, items, and anyOf" do
      schema = %{
        "type" => "object",
        "additionalProperties" => false,
        "properties" => %{
          "people" => %{
            "type" => "array",
            "additionalProperties" => false,
            "items" => %{
              "type" => "object",
              "additionalProperties" => false,
              "properties" => %{"name" => %{"type" => "string", "const" => "x"}}
            }
          },
          "either" => %{
            "anyOf" => [
              %{"type" => "string", "$ref" => "#/$defs/thing"},
              %{"type" => "object", "additionalProperties" => false, "properties" => %{}}
            ]
          }
        }
      }

      assert %{
               "type" => "object",
               "properties" => %{
                 "people" => %{
                   "type" => "array",
                   "items" => %{
                     "type" => "object",
                     "properties" => %{"name" => %{"type" => "string"}}
                   }
                 },
                 "either" => %{
                   "anyOf" => [
                     %{"type" => "string"},
                     %{"type" => "object", "properties" => %{}}
                   ]
                 }
               }
             } == GoogleSchema.sanitize(schema)
    end

    test "keeps properties whose name matches an unsupported keyword" do
      # `properties` maps caller-chosen names to schemas. A property named
      # `const` is a property name, not a schema keyword.
      schema = %{
        "type" => "object",
        "additionalProperties" => false,
        "required" => ["const"],
        "properties" => %{
          "const" => %{"type" => "string"},
          "examples" => %{"type" => "string"},
          "not" => %{"type" => "boolean"},
          "if" => %{"type" => "string", "additionalProperties" => false}
        }
      }

      assert %{
               "type" => "object",
               "required" => ["const"],
               "properties" => %{
                 "const" => %{"type" => "string"},
                 "examples" => %{"type" => "string"},
                 "not" => %{"type" => "boolean"},
                 "if" => %{"type" => "string"}
               }
             } == GoogleSchema.sanitize(schema)
    end

    test "leaves instance data untouched" do
      # `enum` members, a `default`, and an `example` are values rather than
      # schemas. Filtering them would corrupt data that happens to use a
      # keyword as one of its own field names.
      schema = %{
        "type" => "object",
        "properties" => %{
          "settings" => %{
            "type" => "object",
            "default" => %{"additionalProperties" => true, "const" => "keep me"},
            "example" => %{"$ref" => "not-a-reference"}
          },
          "choice" => %{
            "type" => "string",
            "enum" => ["const", "examples", "$ref"]
          }
        }
      }

      assert %{
               "properties" => %{
                 "settings" => %{
                   "default" => %{"additionalProperties" => true, "const" => "keep me"},
                   "example" => %{"$ref" => "not-a-reference"}
                 },
                 "choice" => %{"enum" => ["const", "examples", "$ref"]}
               }
             } = GoogleSchema.sanitize(schema)
    end

    test "handles atom keys" do
      assert %{type: "object", properties: %{"a" => %{"type" => "string"}}} ==
               GoogleSchema.sanitize(%{
                 type: "object",
                 additionalProperties: false,
                 properties: %{"a" => %{"type" => "string"}}
               })
    end

    test "passes through non-map input unchanged" do
      assert nil == GoogleSchema.sanitize(nil)
      assert true == GoogleSchema.sanitize(true)
      assert "text" == GoogleSchema.sanitize("text")
    end
  end

  describe "empty_object?/1" do
    test "detects an object schema with no properties" do
      assert GoogleSchema.empty_object?(%{"type" => "object", "properties" => %{}})

      # extra keys do not stop Google from rejecting the empty object
      assert GoogleSchema.empty_object?(%{
               "type" => "object",
               "properties" => %{},
               "description" => "Takes nothing."
             })
    end

    test "is false for anything else" do
      refute GoogleSchema.empty_object?(%{"type" => "object", "properties" => %{"a" => %{}}})
      refute GoogleSchema.empty_object?(%{"type" => "string"})
      refute GoogleSchema.empty_object?(nil)
    end
  end
end
