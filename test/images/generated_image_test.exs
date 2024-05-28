defmodule LangChain.Images.GeneratedImageTest do
  use ExUnit.Case
  doctest LangChain.Images.GeneratedImage
  alias LangChain.Images.GeneratedImage
  alias LangChain.LangChainError

  describe "new/1" do
    test "creates valid model with minimal setup" do
      prompt = "A happy little chipmunk with full cheeks."

      {:ok, %GeneratedImage{} = img} =
        GeneratedImage.new(%{prompt: prompt, content: "https://example.com/image.png"})

      assert img.prompt == prompt
    end

    test "supports all valid settings" do
      {:ok, img} =
        GeneratedImage.new(%{
          prompt: "A well worn baseball sitting in a well worn glove",
          content: "base64_data",
          image_type: :jpg,
          type: :base64
        })

      assert img.prompt == "A well worn baseball sitting in a well worn glove"
      assert img.content == "base64_data"
      assert img.image_type == :jpg
      assert img.type == :base64
    end

    test "returns error when invalid" do
      {:error, changeset} = GeneratedImage.new(%{content: nil})
      assert {"can't be blank", _} = changeset.errors[:content]
    end
  end

  describe "new!/1" do
    test "returns struct when valid" do
      prompt = "A happy little chipmunk with full cheeks."

      %GeneratedImage{} =
        img = GeneratedImage.new!(%{prompt: prompt, content: "https://example.com/chipmunk.png"})

      assert img.prompt == prompt
    end

    test "raises LangChain.LangChainError when invalid" do
      assert_raise LangChainError, "content: can't be blank", fn ->
        GeneratedImage.new!(%{})
      end
    end
  end
end
