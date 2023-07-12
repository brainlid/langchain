defmodule Langchain.Functions.FunctionParameterTest do
  use ExUnit.Case

  doctest Langchain.Functions.FunctionParameter
  alias Langchain.Functions.FunctionParameter

  describe "new/1" do
    test "works with minimal attrs" do
      assert {:ok, %FunctionParameter{} = param} =
               FunctionParameter.new(%{"name" => "thing1", "type" => "string"})

      assert param.name == "thing1"
      assert param.type == "string"
    end

    test "returns error when invalid" do
      assert {:error, changeset} = FunctionParameter.new(%{"name" => nil})
      refute changeset.valid?
      assert {"can't be blank", _} = changeset.errors[:name]
      assert {"can't be blank", _} = changeset.errors[:type]
    end
  end

  describe "for_api/1" do
    test "supports basic name and type" do
      {:ok, param} = FunctionParameter.new(%{"name" => "p1", "type" => "string"})
      result = FunctionParameter.for_api(param, %{})
      assert result == %{"p1" => %{"description" => nil, "type" => "string"}}
    end

    test "supports enums" do
      {:ok, param} = FunctionParameter.new(%{"name" => "p1", "type" => "string", "enum" => ["a", "b"]})
      result = FunctionParameter.for_api(param, %{})
      assert result == %{"p1" => %{"description" => nil, "type" => "string", "enum" => ["a", "b"]}}
    end
  end
end
