defmodule LangChain.Message.MessagePartTest do
  use ExUnit.Case
  doctest LangChain.Message.MessagePart
  alias LangChain.Message.MessagePart

  describe "new/1" do
    test "accepts valid settings" do
      {:ok, %MessagePart{} = part} = MessagePart.new(%{type: :text, content: "Hello!"})
      assert part.type == :text
      assert part.content == "Hello!"
    end

    test "returns error when invalid" do
      assert {:error, changeset} = MessagePart.new(%{"type" => nil, "content" => nil})
      refute changeset.valid?
      assert {"can't be blank", _} = changeset.errors[:type]
      assert {"can't be blank", _} = changeset.errors[:content]
    end
  end

  describe "new!/1" do
    test "accepts valid settings" do
      %MessagePart{} = part = MessagePart.new!(%{type: :text, content: "Hello!"})
      assert part.type == :text
      assert part.content == "Hello!"
    end

    test "returns error when invalid" do
      assert_raise LangChain.LangChainError, "content: can't be blank", fn ->
        MessagePart.text!(nil)
      end
    end
  end

  describe("text!/1") do
    test "returns a text configured MessagePart" do
      %MessagePart{} = part = MessagePart.text!("Hello!")
      assert part.type == :text
      assert part.content == "Hello!"
    end
  end

  describe("image!/1") do
    test "returns an image data configured MessagePart" do
      %MessagePart{} = part = MessagePart.image!(:base64.encode("fake_image_data"))
      assert part.type == :image
      assert part.content == "ZmFrZV9pbWFnZV9kYXRh"
    end
  end

  describe("image_url!/1") do
    test "returns a image_url configured MessagePart" do
      url =
        "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg"

      %MessagePart{} = part = MessagePart.image_url!(url)
      assert part.type == :image_url
      assert part.content == url
    end
  end
end
