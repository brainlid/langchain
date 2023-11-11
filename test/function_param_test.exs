defmodule LangChain.FunctionParamTest do
  use ExUnit.Case

  doctest LangChain.FunctionParam

  alias LangChain.FunctionParam

  describe "new!/1" do
    test "creates the function parameter" do
      %FunctionParam{} =
        param =
        FunctionParam.new!(%{
          name: "code",
          type: :string,
          description: "A unique code used to identify the object.",
          required: true
        })

      assert param.name == "code"
      assert param.type == :string
      assert param.description == "A unique code used to identify the object."
      assert param.required == true
    end

    test "description and required are optional" do
      param = FunctionParam.new!(%{name: "code", type: :string})

      assert param.name == "code"
      assert param.type == :string
      assert param.description == nil
      assert param.required == false
    end

    test "supports enum values" do
      param = FunctionParam.new!(%{name: "color", type: :string, enum: ["red", "green"]})
      assert param.name == "color"
      assert param.type == :string
      assert param.enum == ["red", "green"]
    end

    test "supports array type" do
      param = FunctionParam.new!(%{name: "colors", type: :array})
      assert param.name == "colors"
      assert param.type == :array
      assert param.item_type == nil

      param = FunctionParam.new!(%{name: "colors", type: :array, item_type: "string"})
      assert param.name == "colors"
      assert param.type == :array
      assert param.item_type == "string"
    end

    test "supports object type" do
      person_properties = [
        FunctionParam.new!(%{name: "name", type: :string, required: true}),
        FunctionParam.new!(%{name: "age", type: :integer}),
        FunctionParam.new!(%{name: "employee", type: :boolean})
      ]

      param =
        FunctionParam.new!(%{name: "person", type: :object, object_properties: person_properties})

      assert param.name == "person"
      assert param.type == :object
      assert param.object_properties == person_properties
    end

    test "supports nested objects type"

    test "does not allow field data for non-matching types" do
      {:error, changeset} =
        FunctionParam.new(%{name: "thing", type: :string, item_type: "number"})

      assert {"not allowed for type :string", _} = changeset.errors[:item_type]

      {:error, changeset} =
        FunctionParam.new(%{
          name: "thing",
          type: :string,
          object_properties: [FunctionParam.new!(%{name: "name", type: :string})]
        })

      assert {"not allowed for type :string", _} = changeset.errors[:object_properties]
    end
  end

  describe "to_json_schema/2" do
    test "basic types - integer, string, number, boolean" do
      param = FunctionParam.new!(%{name: "name", type: :string})
      expected = %{"name" => %{"type" => "string"}}
      assert expected == FunctionParam.to_json_schema(%{}, param)

      param = FunctionParam.new!(%{name: "age", type: :integer})
      expected = %{"age" => %{"type" => "integer"}}
      assert expected == FunctionParam.to_json_schema(%{}, param)

      param = FunctionParam.new!(%{name: "height", type: :number})
      expected = %{"height" => %{"type" => "number"}}
      assert expected == FunctionParam.to_json_schema(%{}, param)

      # includes description
      param = FunctionParam.new!(%{name: "name", type: :string, description: "Applicant's name"})
      expected = %{"name" => %{"type" => "string", "description" => "Applicant's name"}}
      assert expected == FunctionParam.to_json_schema(%{}, param)

      param = FunctionParam.new!(%{name: "enabled", type: :boolean})
      expected = %{"enabled" => %{"type" => "boolean"}}
      assert expected == FunctionParam.to_json_schema(%{}, param)

      param =
        FunctionParam.new!(%{name: "enabled", type: :boolean, description: "If option is enabled"})

      expected = %{"enabled" => %{"type" => "boolean", "description" => "If option is enabled"}}
      assert expected == FunctionParam.to_json_schema(%{}, param)
    end

    test "basic types support enum values" do
      param = FunctionParam.new!(%{name: "name", type: :string, enum: ["John", "Mary"]})
      expected = %{"name" => %{"type" => "string", "enum" => ["John", "Mary"]}}
      assert expected == FunctionParam.to_json_schema(%{}, param)

      param = FunctionParam.new!(%{name: "age", type: :integer, enum: [1, 2, 10]})
      expected = %{"age" => %{"type" => "integer", "enum" => [1, 2, 10]}}
      assert expected == FunctionParam.to_json_schema(%{}, param)

      param = FunctionParam.new!(%{name: "height", type: :number, enum: [5.0, 5.5, 6, 6.5]})
      expected = %{"height" => %{"type" => "number", "enum" => [5.0, 5.5, 6, 6.5]}}
      assert expected == FunctionParam.to_json_schema(%{}, param)
    end

    test "array of basic types" do
      # no defined item_type
      param =
        FunctionParam.new!(%{name: "list_data", type: :array, description: "A list of things"})

      expected = %{"list_data" => %{"type" => "array", "description" => "A list of things"}}
      assert expected == FunctionParam.to_json_schema(%{}, param)

      # with a specified item type
      param = FunctionParam.new!(%{name: "tags", type: :array, item_type: "string"})
      expected = %{"tags" => %{"type" => "array", "items" => %{"type" => "string"}}}
      assert expected == FunctionParam.to_json_schema(%{}, param)

      # includes description
      param =
        FunctionParam.new!(%{
          name: "tags",
          type: :array,
          item_type: "string",
          description: "tag values"
        })

      expected = %{
        "tags" => %{
          "type" => "array",
          "items" => %{"type" => "string"},
          "description" => "tag values"
        }
      }

      assert expected == FunctionParam.to_json_schema(%{}, param)
    end

    test "string type with enum values" do
      param = FunctionParam.new!(%{name: "color", type: :string, enum: ["red", "green"]})
      expected = %{"color" => %{"type" => "string", "enum" => ["red", "green"]}}
      assert expected == FunctionParam.to_json_schema(%{}, param)

      # includes description
      param =
        FunctionParam.new!(%{
          name: "color",
          type: :string,
          enum: ["red", "green"],
          description: "Allowed colors"
        })

      expected = %{
        "color" => %{
          "type" => "string",
          "enum" => ["red", "green"],
          "description" => "Allowed colors"
        }
      }

      assert expected == FunctionParam.to_json_schema(%{}, param)
    end

    test "object type" do
      param =
        FunctionParam.new!(%{
          name: "attributes",
          type: :object,
          description: "Set of attributes for a new thing",
          object_properties: [
            FunctionParam.new!(%{
              name: "name",
              type: :string,
              description: "The name of the thing"
            }),
            FunctionParam.new!(%{
              name: "code",
              type: :string,
              description: "Unique code",
              required: true
            })
          ]
        })

      expected =
        %{
          "attributes" => %{
            "type" => "object",
            "description" => "Set of attributes for a new thing",
            "properties" => %{
              "name" => %{
                "type" => "string",
                "description" => "The name of the thing"
              },
              "code" => %{
                "type" => "string",
                "description" => "Unique code"
              }
            },
            "required" => ["code"]
          }
        }

      assert expected == FunctionParam.to_json_schema(%{}, param)
    end

    test "array of objects"
  end

  describe "to_parameters_schema/1" do
    test "basic example" do
      expected = %{
        "type" => "object",
        "properties" => %{
          "code" => %{
            "type" => "string",
            "description" => "Unique code"
          },
          "other" => %{
            "type" => "string"
          }
        },
        "required" => ["code"]
      }

      params = [
        FunctionParam.new!(%{
          name: "code",
          type: "string",
          description: "Unique code",
          required: true
        }),
        FunctionParam.new!(%{
          name: "other",
          type: "string"
        })
      ]

      assert expected == FunctionParam.to_parameters_schema(params)
    end

    test "generates the full JSONSchema structured map for the list of parameters"
    test "supports nested objects"
    test "supports listing required parameters"
  end

  describe "required_properties/1" do
    test "return empty when nothing required" do
      params = [
        FunctionParam.new!(%{
          name: "optional_thing",
          type: "string"
        })
      ]

      assert [] == FunctionParam.required_properties(params)
    end

    test "return a list of the property names flagged as required" do
      params = [
        FunctionParam.new!(%{
          name: "code",
          type: "string",
          description: "Unique code",
          required: true
        }),
        FunctionParam.new!(%{
          name: "other",
          type: "string"
        }),
        FunctionParam.new!(%{
          name: "important",
          type: "integer",
          required: true
        })
      ]

      assert ["code", "important"] == FunctionParam.required_properties(params)
    end
  end
end
