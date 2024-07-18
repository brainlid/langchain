defmodule LangChain.UtilsTest do
  use ExUnit.Case

  doctest LangChain.Utils
  alias LangChain.ChatModels.ChatOpenAI
  alias LangChain.Utils

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

      assert result ==
               "multiple: should be at most 10 character(s), is invalid; name: can't be blank; required: can't be blank"
    end

    test "handles ecto enum type errors" do
      {:error, changeset} = LangChain.MessageDelta.new(%{role: "invalid"})
      result = Utils.changeset_error_to_string(changeset)
      assert result == "role: is invalid"
    end

    test "handles multiple errors on a field" do
      {:error, changeset} = LangChain.MessageDelta.new(%{role: "invalid"})
      changeset = Ecto.Changeset.add_error(changeset, :role, "is required")
      result = Utils.changeset_error_to_string(changeset)
      assert result == "role: is required, is invalid"
    end

    test "handles errors on multiple fields" do
      {:error, changeset} = LangChain.MessageDelta.new(%{role: "invalid", index: "abc"})
      result = Utils.changeset_error_to_string(changeset)
      assert result == "role: is invalid; index: is invalid"
    end

    test "handles multiple errors on multiple fields" do
      {:error, changeset} = LangChain.MessageDelta.new(%{role: "invalid", index: "abc"})

      changeset =
        changeset
        |> Ecto.Changeset.add_error(:index, "is numeric")
        |> Ecto.Changeset.add_error(:role, "is important")

      result = Utils.changeset_error_to_string(changeset)
      assert result == "role: is important, is invalid; index: is numeric, is invalid"
    end
  end

  describe "put_in_list/3" do
    test "adds to empty list" do
      assert [1] == Utils.put_in_list([], 0, 1)
    end

    test "adds to list with values" do
      result =
        [1, 2]
        |> Utils.put_in_list(2, "3")

      assert result == [1, 2, "3"]
    end

    test "replaces existing value at the index" do
      result =
        [1, 2, 3]
        |> Utils.put_in_list(0, "a")
        |> Utils.put_in_list(1, "b")
        |> Utils.put_in_list(2, "c")

      assert result == ["a", "b", "c"]
    end
  end

  describe "to_serializable_map/3" do
    test "converts a chat model to a string keyed map with a version included" do
      model =
        ChatOpenAI.new!(%{
          model: "gpt-4o",
          temperature: 0,
          frequency_penalty: 0.5,
          seed: 123,
          max_tokens: 1234,
          stream_options: %{include_usage: true}
        })

      result =
        Utils.to_serializable_map(model, [
          :model,
          :temperature,
          :frequency_penalty,
          :seed,
          :max_tokens,
          :stream_options
        ])

      assert result == %{
               "model" => "gpt-4o",
               "temperature" => 0.0,
               "frequency_penalty" => 0.5,
               "seed" => 123,
               "max_tokens" => 1234,
               "stream_options" => %{"include_usage" => true},
               "version" => 1,
               "module" => "Elixir.LangChain.ChatModels.ChatOpenAI"
             }
    end
  end

  describe "module_from_name/1" do
    test "returns :ok tuple with module when valid" do
      assert {:ok, DateTime} = Utils.module_from_name("Elixir.DateTime")
    end

    test "returns error when not a module" do
      assert {:error, reason} = Utils.module_from_name("not a module")
      assert reason == "Not an Elixir module: \"not a module\""
    end

    test "returns error when not an existing atom" do
      assert {:error, reason} = Utils.module_from_name("Elixir.Missing.Module")
      assert reason == "ChatModel module \"Elixir.Missing.Module\" not found"
    end
  end

  describe "handle_stream_fn/3" do
    test "skips transformed messages with :skip" do
      function =
        Utils.handle_stream_fn(%{callbacks: []}, fn _ -> {[[]], ""} end, fn _ ->
          :skip
        end)

      {:cont, {_model, %{body: body}}} = function.({:data, ""}, {1, %Req.Response{status: 200}})

      assert body == []
    end

    test "skips transformed messages with empty list" do
      empty_transform_message = []

      function =
        Utils.handle_stream_fn(%{callbacks: []}, fn _ -> {[[]], ""} end, fn _ ->
          empty_transform_message
        end)

      {:cont, {_model, %{body: body}}} = function.({:data, ""}, {1, %Req.Response{status: 200}})

      assert body == []
    end
  end
end
