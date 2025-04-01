defmodule LangChain.Message.ContentPartTest do
  use ExUnit.Case
  doctest LangChain.Message.ContentPart
  alias LangChain.Message.ContentPart

  describe "new/1" do
    test "accepts valid settings" do
      {:ok, %ContentPart{} = part} = ContentPart.new(%{type: :text, content: "Hello!"})
      assert part.type == :text
      assert part.content == "Hello!"
    end

    test "returns error when invalid" do
      assert {:error, changeset} = ContentPart.new(%{"type" => nil})
      refute changeset.valid?
      assert {"can't be blank", _} = changeset.errors[:type]
    end
  end

  describe "new!/1" do
    test "accepts valid settings" do
      %ContentPart{} = part = ContentPart.new!(%{type: :text, content: "Hello!"})
      assert part.type == :text
      assert part.content == "Hello!"
    end

    test "returns error when invalid" do
      assert_raise LangChain.LangChainError, "type: can't be blank", fn ->
        ContentPart.new!(%{type: nil})
      end
    end
  end

  describe("text!/1") do
    test "returns a text configured ContentPart" do
      %ContentPart{} = part = ContentPart.text!("Hello!")
      assert part.type == :text
      assert part.content == "Hello!"
    end
  end

  describe("image!/1") do
    test "returns an image data configured ContentPart" do
      %ContentPart{} = part = ContentPart.image!(:base64.encode("fake_image_data"))
      assert part.type == :image
      assert part.content == "ZmFrZV9pbWFnZV9kYXRh"
    end

    test "supports 'detail' option" do
      %ContentPart{} = part = ContentPart.image!(Base.encode64("fake_image_data"), detail: "low")
      assert part.type == :image
      assert part.content == "ZmFrZV9pbWFnZV9kYXRh"
      assert part.options == [detail: "low"]
    end
  end

  describe("file/1") do
    test "returns a file data configured ContentPart" do
      %ContentPart{} = part = ContentPart.file!(:base64.encode("fake_file_data"))
      assert part.type == :file
      assert part.content == "ZmFrZV9maWxlX2RhdGE="
    end

    test "supports 'filename' option" do
      %ContentPart{} =
        part = ContentPart.file!(Base.encode64("fake_file_data"), filename: "my_file.pdf")

      assert part.type == :file
      assert part.content == "ZmFrZV9maWxlX2RhdGE="
      assert part.options == [filename: "my_file.pdf"]
    end
  end

  describe("image_url!/1") do
    test "returns a image_url configured ContentPart" do
      url =
        "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg"

      %ContentPart{} = part = ContentPart.image_url!(url)
      assert part.type == :image_url
      assert part.content == url
    end

    test "supports 'detail' option with ChatGPT" do
      url =
        "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg"

      %ContentPart{} = part = ContentPart.image_url!(url, detail: "low")
      assert part.type == :image_url
      assert part.content == url
      assert part.options == [detail: "low"]
    end
  end

  describe "merge_part/2" do
    test "merges two text content parts" do
      part_1 = ContentPart.text!("Hello")
      part_2 = ContentPart.text!(" world")
      merged = ContentPart.merge_part(part_1, part_2)
      assert merged.content == "Hello world"
    end

    test "merges two thinking content parts" do
      part_1 = ContentPart.new!(%{type: :thinking, content: "I'm thinking"})
      part_2 = ContentPart.new!(%{type: :thinking, content: " about how lovely"})
      merged = ContentPart.merge_part(part_1, part_2)
      assert merged.type == :thinking
      assert merged.content == "I'm thinking about how lovely"
    end

    test "merges a thinking signature" do
      part_1 = ContentPart.new!(%{type: :thinking, content: "I'm thinking about how lovely"})
      part_2 = ContentPart.new!(%{type: :thinking, options: [signature: "woofwoofwoof"]})
      merged = ContentPart.merge_part(part_1, part_2)
      assert merged.type == :thinking
      assert merged.content == "I'm thinking about how lovely"
      assert merged.options == [signature: "woofwoofwoof"]

      # assert that the signature can be added to.
      part_3 = ContentPart.new!(%{type: :thinking, options: [signature: "bowwowwow"]})
      merged = ContentPart.merge_part(merged, part_3)
      assert merged.type == :thinking
      assert merged.content == "I'm thinking about how lovely"
      assert merged.options == [signature: "woofwoofwoofbowwowwow"]
    end

    test "merges a redacted thinking content" do
      part_1 =
        ContentPart.new!(%{
          type: :unsupported,
          options: [redacted: "redactedREDACTEDredacted"]
        })

      part_2 = ContentPart.new!(%{type: :unsupported, options: [redacted: "MOREmoreMORE"]})
      merged = ContentPart.merge_part(part_1, part_2)
      assert merged.type == :unsupported
      assert merged.content == nil
      assert merged.options == [redacted: "redactedREDACTEDredactedMOREmoreMORE"]
    end
  end
end
