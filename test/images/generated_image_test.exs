defmodule LangChain.Images.GeneratedImageTest do
	use ExUnit.Case
	doctest LangChain.Images.GeneratedImage
	alias LangChain.Images.GeneratedImage
  alias LangChain.LangChainError

  describe "new/1" do
    test "creates valid model with minimal setup" do
      prompt = "A happy little chipmunk with full cheeks."
      {:ok, %GeneratedImage{} = img} = GeneratedImage.new(%{prompt: prompt})
      assert img.prompt == prompt
    end

    test "supports all valid settings" do
      {:ok, img} =
        GeneratedImage.new(%{
          endpoint: "https://example.com",
          api_key: "overridden",
          receive_timeout: 2_000,
          model: "dall-e-3",
          prompt: "A well worn baseball sitting in a well worn glove",
          quality: "hd",
          response_format: "b64_json",
          size: "1792x1024",
          style: "natural",
          user: "user-123"
        })

      assert img.endpoint == "https://example.com"
      assert img.api_key == "overridden"
      assert img.receive_timeout == 2_000
      assert img.model == "dall-e-3"
      assert img.prompt == "A well worn baseball sitting in a well worn glove"
      assert img.quality == "hd"
      assert img.response_format == "b64_json"
      assert img.size == "1792x1024"
      assert img.style == "natural"
      assert img.user == "user-123"
    end

    test "returns error when invalid" do
      {:error, changeset} = GeneratedImage.new(%{model: "dang-e-infinity"})
      assert {"can't be blank", _} = changeset.errors[:prompt]
      assert {"is invalid", _} = changeset.errors[:model]
    end
  end

  describe "new!/1" do
    test "returns struct when valid" do
      prompt = "A happy little chipmunk with full cheeks."
      %GeneratedImage{} = img = GeneratedImage.new!(%{prompt: prompt})
      assert img.prompt == prompt
    end

    test "raises LangChain.LangChainError when invalid" do
      assert_raise LangChainError, "prompt: can't be blank", fn ->
        GeneratedImage.new!(%{})
      end
    end
  end
end
