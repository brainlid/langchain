defmodule LangChain.UtilsTest do
  use ExUnit.Case

  doctest LangChain.Utils
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

  def setup_expected_json(_) do
    json_1 = %{
      "choices" => [
        %{
          "delta" => %{
            "content" => nil,
            "function_call" => %{"arguments" => "", "name" => "calculator"},
            "role" => "assistant"
          },
          "finish_reason" => nil,
          "index" => 0
        }
      ],
      "created" => 1_689_801_995,
      "id" => "chatcmpl-7e8yp1xBhriNXiqqZ0xJkgNrmMuGS",
      "model" => "gpt-4-0613",
      "object" => "chat.completion.chunk"
    }

    json_2 = %{
      "choices" => [
        %{
          "delta" => %{"function_call" => %{"arguments" => "{\n"}},
          "finish_reason" => nil,
          "index" => 0
        }
      ],
      "created" => 1_689_801_995,
      "id" => "chatcmpl-7e8yp1xBhriNXiqqZ0xJkgNrmMuGS",
      "model" => "gpt-4-0613",
      "object" => "chat.completion.chunk"
    }

    %{json_1: json_1, json_2: json_2}
  end

  defp send_parsed_data(%{} = parsed_data) do
    send(self(), {:parsed_data, parsed_data})
    parsed_data
  end

  describe "decode_streamed_data/2" do
    setup :setup_expected_json

    test "correctly handles fully formed chat completion chunks", %{json_1: json_1, json_2: json_2} do
      data =
        "data: {\"id\":\"chatcmpl-7e8yp1xBhriNXiqqZ0xJkgNrmMuGS\",\"object\":\"chat.completion.chunk\",\"created\":1689801995,\"model\":\"gpt-4-0613\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":null,\"function_call\":{\"name\":\"calculator\",\"arguments\":\"\"}},\"finish_reason\":null}]}\n\n
         data: {\"id\":\"chatcmpl-7e8yp1xBhriNXiqqZ0xJkgNrmMuGS\",\"object\":\"chat.completion.chunk\",\"created\":1689801995,\"model\":\"gpt-4-0613\",\"choices\":[{\"index\":0,\"delta\":{\"function_call\":{\"arguments\":\"{\\n\"}},\"finish_reason\":null}]}\n\n"

      {parsed, incomplete} = Utils.decode_streamed_data({data, ""}, &send_parsed_data/1)

      # callback should have fired with matching parsed data
      assert_received {:parsed_data, ^json_1}
      assert_received {:parsed_data, ^json_2}

      # nothing incomplete. Parsed 2 objects.
      assert incomplete == ""
      assert parsed == [json_1, json_2]
    end

    test "correctly parses when data split over received messages", %{json_1: json_1} do
      # split the data over multiple messages
      data =
        "data: {\"id\":\"chatcmpl-7e8yp1xBhriNXiqqZ0xJkgNrmMuGS\",\"object\":\"chat.comple
         data: tion.chunk\",\"created\":1689801995,\"model\":\"gpt-4-0613\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":null,\"function_call\":{\"name\":\"calculator\",\"arguments\":\"\"}},\"finish_reason\":null}]}\n\n"

      {parsed, incomplete} = Utils.decode_streamed_data({data, ""}, &send_parsed_data/1)

      # callback should have fired with matching parsed data
      assert_received {:parsed_data, ^json_1}

      # nothing incomplete. Parsed 1 object.
      assert incomplete == ""
      assert parsed == [json_1]
    end

    test "correctly parses when data split over decode calls", %{json_1: json_1} do
      buffered = "{\"id\":\"chatcmpl-7e8yp1xBhriNXiqqZ0xJkgNrmMuGS\",\"object\":\"chat.comple"

      # incomplete message chunk processed in next call
      data = "data: tion.chunk\",\"created\":1689801995,\"model\":\"gpt-4-0613\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":null,\"function_call\":{\"name\":\"calculator\",\"arguments\":\"\"}},\"finish_reason\":null}]}\n\n"

      {parsed, incomplete} = Utils.decode_streamed_data({data, buffered}, &send_parsed_data/1)

      # callback should have fired with matching parsed data
      assert_received {:parsed_data, ^json_1}

      # nothing incomplete. Parsed 1 object.
      assert incomplete == ""
      assert parsed == [json_1]
    end

    test "correctly parses when data previously buffered and responses split and has leftovers", %{json_1: json_1, json_2: json_2} do
      buffered = "{\"id\":\"chatcmpl-7e8yp1xBhriNXiqqZ0xJkgNrmMuGS\",\"object\":\"chat.comple"

      # incomplete message chunk processed in next call
      data =
        "data: tion.chunk\",\"created\":1689801995,\"model\":\"gpt-4-0613\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":null,\"function_call\":{\"name\":\"calculator\",\"arguments\":\"\"}},\"finish_reason\":null}]}\n\n
         data: {\"id\":\"chatcmpl-7e8yp1xBhriNXiqqZ0xJkgNrmMuGS\",\"object\":\"chat.completion.chunk\",\"crea
         data: ted\":1689801995,\"model\":\"gpt-4-0613\",\"choices\":[{\"index\":0,\"delta\":{\"function_call\":{\"argu
         data: ments\":\"{\\n\"}},\"finish_reason\":null}]}\n\n
         data: {\"id\":\"chatcmpl-7e8yp1xBhriNXiqqZ0xJkgNrmMuGS\",\"object\":\"chat.comp"

      {parsed, incomplete} = Utils.decode_streamed_data({data, buffered}, &send_parsed_data/1)

      # callback should have fired with matching parsed data
      assert_received {:parsed_data, ^json_1}
      assert_received {:parsed_data, ^json_2}

      # nothing incomplete. Parsed 1 object.
      assert incomplete == "{\"id\":\"chatcmpl-7e8yp1xBhriNXiqqZ0xJkgNrmMuGS\",\"object\":\"chat.comp"
      assert parsed == [json_1, json_2]
    end
  end
end
