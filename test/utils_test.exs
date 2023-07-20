defmodule Langchain.UtilsTest do
  use ExUnit.Case

  doctest Langchain.Utils
  alias Langchain.Utils

  defmodule FakeSchema do
    use Ecto.Schema
    import Ecto.Changeset

    embedded_schema do
      field :name, :string
      field :age, :integer
      field :required, :string
      field :multiple, :string
    end

    def changeset(struct, attrs) do
      struct
      |> cast(attrs, [:name, :age, :required, :multiple])
      |> validate_required([:name, :required])
      |> validate_number(:age, greater_than_or_equal_to: 0)
      |> validate_inclusion(:multiple, ["north", "east", "south", "west"])
      |> validate_length(:multiple, max: 10)
    end
  end

  describe "changeset_error_to_string/1" do
    test "returns multiple errors as comma separated" do
      changeset = FakeSchema.changeset(%FakeSchema{}, %{})
      refute changeset.valid?
      result = Utils.changeset_error_to_string(changeset)
      assert result == "name: can't be blank; required: can't be blank"
    end

    test "returns multiple errors on a field" do
      changeset =
        FakeSchema.changeset(%FakeSchema{}, %{
          name: "Tom",
          required: "value",
          multiple: "too long to be valid"
        })

      refute changeset.valid?
      result = Utils.changeset_error_to_string(changeset)
      assert result == "multiple: should be at most 10 character(s), is invalid"
    end

    test "combines errors for multiple fields and multiple on single field" do
      changeset =
        FakeSchema.changeset(%FakeSchema{}, %{
          name: nil,
          required: nil,
          multiple: "too long to be valid"
        })

      refute changeset.valid?
      result = Utils.changeset_error_to_string(changeset)
      assert result == "multiple: should be at most 10 character(s), is invalid; name: can't be blank; required: can't be blank"
    end

    test "handles ecto enum type errors" do
      {:error, changeset} = Langchain.MessageDelta.new(%{role: "invalid"})
      result = Utils.changeset_error_to_string(changeset)
      assert result == "role: is invalid"
    end

    test "handles multiple errors on a field" do
      {:error, changeset} = Langchain.MessageDelta.new(%{role: "invalid"})
      changeset = Ecto.Changeset.add_error(changeset, :role, "is required")
      result = Utils.changeset_error_to_string(changeset)
      assert result == "role: is required, is invalid"
    end

    test "handles errors on multiple fields" do
      {:error, changeset} = Langchain.MessageDelta.new(%{role: "invalid", index: "abc"})
      result = Utils.changeset_error_to_string(changeset)
      assert result == "role: is invalid; index: is invalid"
    end

    test "handles multiple errors on multiple fields" do
      {:error, changeset} = Langchain.MessageDelta.new(%{role: "invalid", index: "abc"})
      changeset =
        changeset
        |> Ecto.Changeset.add_error(:index, "is numeric")
        |> Ecto.Changeset.add_error(:role, "is important")

      result = Utils.changeset_error_to_string(changeset)
      assert result == "role: is important, is invalid; index: is numeric, is invalid"
    end
  end
end
