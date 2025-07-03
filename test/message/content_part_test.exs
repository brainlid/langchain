defmodule LangChain.Message.ContentPartTest do
  use ExUnit.Case
  doctest LangChain.Message.ContentPart, import: true
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

    test "accepts an empty string content as-is" do
      {:ok, %ContentPart{} = part} = ContentPart.new(%{type: :text, content: ""})
      assert part.type == :text
      assert part.content == ""

      {:ok, %ContentPart{} = part} = ContentPart.new(%{type: :text, content: " "})
      assert part.type == :text
      assert part.content == " "
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

  describe("file_url/1") do
    test "returns a file data configured ContentPart" do
      %ContentPart{} =
        part = ContentPart.file_url!("example.com/file.pdf", media: "application/pdf")

      assert part.type == :file_url
      assert part.content == "example.com/file.pdf"
    end

    test "supports 'media' option" do
      %ContentPart{} =
        part = ContentPart.file_url!("example.com/file.pdf", media: "application/pdf")

      assert part.type == :file_url
      assert part.content == "example.com/file.pdf"
      assert part.options == [media: "application/pdf"]
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

  describe "parts_to_string/2" do
    test "joins text content parts with double newlines" do
      parts = [
        ContentPart.text!("Hello"),
        ContentPart.text!("world"),
        ContentPart.text!("how are you")
      ]

      assert ContentPart.parts_to_string(parts) == "Hello\n\nworld\n\nhow are you"
    end

    test "ignores non-text content parts" do
      parts = [
        ContentPart.text!("Hello"),
        ContentPart.image!("base64data"),
        ContentPart.text!("world"),
        ContentPart.image_url!("https://example.com/image.jpg"),
        ContentPart.text!("how are you")
      ]

      assert ContentPart.parts_to_string(parts) == "Hello\n\nworld\n\nhow are you"
    end

    test "matches on content type and can return thinking blocks" do
      parts = [
        ContentPart.new!(%{type: :thinking, content: "Let's think about this..."}),
        ContentPart.text!("regular text"),
        ContentPart.new!(%{type: :thinking, content: "I think this is a good idea"})
      ]

      assert ContentPart.parts_to_string(parts, :thinking) ==
               "Let's think about this...\n\nI think this is a good idea"
    end

    test "returns nil for empty list" do
      assert ContentPart.parts_to_string([]) == nil
    end

    test "returns nil for list with no text parts" do
      parts = [
        ContentPart.image!("base64data"),
        ContentPart.image_url!("https://example.com/image.jpg")
      ]

      assert ContentPart.parts_to_string(parts) == nil
    end
  end

  describe "content_to_string/1" do
    test "returns nil when content is nil" do
      assert ContentPart.content_to_string(nil) == nil
    end

    test "returns string when content is binary" do
      assert ContentPart.content_to_string("Hello world") == "Hello world"
      assert ContentPart.content_to_string("") == ""
      assert ContentPart.content_to_string(" ") == " "
    end

    test "processes list of content parts" do
      parts = [
        ContentPart.text!("Hello"),
        ContentPart.text!("world")
      ]

      assert ContentPart.content_to_string(parts) == "Hello\n\nworld"
    end

    test "processes list with mixed content part types" do
      parts = [
        ContentPart.text!("Hello"),
        ContentPart.image!("base64data"),
        ContentPart.text!("world"),
        ContentPart.image_url!("https://example.com/image.jpg")
      ]

      assert ContentPart.content_to_string(parts) == "Hello\n\nworld"
    end

    test "returns nil for empty list of content parts" do
      assert ContentPart.content_to_string([]) == nil
    end

    test "returns nil for list with no text content parts" do
      parts = [
        ContentPart.image!("base64data"),
        ContentPart.image_url!("https://example.com/image.jpg")
      ]

      assert ContentPart.content_to_string(parts) == nil
    end
  end

  describe "set_option_on_last_part/3" do
    test "adds cache_control option to the last content part" do
      parts = [
        ContentPart.text!("First part"),
        ContentPart.text!("Second part"),
        ContentPart.text!("Last part")
      ]

      updated_parts = ContentPart.set_option_on_last_part(parts, :cache_control, true)

      # First two parts should be unchanged
      assert Enum.at(updated_parts, 0).options == []
      assert Enum.at(updated_parts, 1).options == []

      # Last part should have the new option
      last_part = List.last(updated_parts)
      assert last_part.options == [cache_control: true]
      assert last_part.content == "Last part"
    end

    test "adds option to single content part" do
      parts = [ContentPart.text!("Only part")]

      updated_parts = ContentPart.set_option_on_last_part(parts, :cache_control, true)

      assert length(updated_parts) == 1
      assert List.first(updated_parts).options == [cache_control: true]
    end

    test "preserves existing options and adds new one" do
      parts = [
        ContentPart.text!("First part"),
        ContentPart.text!("Last part", existing_option: "value")
      ]

      updated_parts = ContentPart.set_option_on_last_part(parts, :cache_control, true)

      last_part = List.last(updated_parts)
      assert last_part.options == [cache_control: true, existing_option: "value"]
    end

    test "works with different content part types" do
      parts = [
        ContentPart.text!("Text part"),
        ContentPart.image!("base64data")
      ]

      updated_parts = ContentPart.set_option_on_last_part(parts, :cache_control, true)

      # Text part should be unchanged
      assert Enum.at(updated_parts, 0).options == []

      # Image part should have the new option
      last_part = List.last(updated_parts)
      assert last_part.options == [cache_control: true]
      assert last_part.type == :image
    end

    test "adds different option types" do
      parts = [ContentPart.text!("Test part")]

      # Test with atom value
      updated_parts = ContentPart.set_option_on_last_part(parts, :detail, :high)
      assert List.first(updated_parts).options == [detail: :high]

      # Test with string value
      updated_parts = ContentPart.set_option_on_last_part(parts, :filename, "test.pdf")
      assert List.first(updated_parts).options == [filename: "test.pdf"]
    end
  end
end
