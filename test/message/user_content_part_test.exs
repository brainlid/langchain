defmodule LangChain.Message.UserContentPartTest do
  use ExUnit.Case
  doctest LangChain.Message.UserContentPart
  alias LangChain.Message.UserContentPart

  describe "new/1" do
    test "accepts valid settings" do
      {:ok, %UserContentPart{} = part} = UserContentPart.new(%{type: :text, content: "Hello!"})
      assert part.type == :text
      assert part.content == "Hello!"
    end

    test "returns error when invalid" do
      assert {:error, changeset} = UserContentPart.new(%{"type" => nil, "content" => nil})
      refute changeset.valid?
      assert {"can't be blank", _} = changeset.errors[:type]
      assert {"can't be blank", _} = changeset.errors[:content]
    end
  end

  describe "new!/1" do
    test "accepts valid settings" do
      %UserContentPart{} = part = UserContentPart.new!(%{type: :text, content: "Hello!"})
      assert part.type == :text
      assert part.content == "Hello!"
    end

    test "returns error when invalid" do
      assert_raise LangChain.LangChainError, "content: can't be blank", fn ->
        UserContentPart.text!(nil)
      end
    end
  end

  describe("text!/1") do
    test "returns a text configured UserContentPart" do
      %UserContentPart{} = part = UserContentPart.text!("Hello!")
      assert part.type == :text
      assert part.content == "Hello!"
    end
  end

  describe("image!/1") do
    test "returns an image data configured UserContentPart" do
      %UserContentPart{} = part = UserContentPart.image!(:base64.encode("fake_image_data"))
      assert part.type == :image
      assert part.content == "ZmFrZV9pbWFnZV9kYXRh"
    end
  end

  describe("image_url!/1") do
    test "returns a image_url configured UserContentPart" do
      url =
        "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg"

      %UserContentPart{} = part = UserContentPart.image_url!(url)
      assert part.type == :image_url
      assert part.content == url
    end
  end
end
