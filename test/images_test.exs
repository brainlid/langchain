defmodule LangChain.ImagesTest do
  use ExUnit.Case
  doctest LangChain.Images
  alias LangChain.Images.GeneratedImage
  alias LangChain.Images

  describe "save_to_file/2" do
    # NOTE: Use an ExUnit created temp directory for saving the file to
    @tag tmp_dir: true, live_call: true
    test "saves a URL image to a file", %{tmp_dir: tmp_dir} do
      public_image =
        GeneratedImage.new!(%{
          image_type: :png,
          type: :url,
          content: "https://pngimg.com/uploads/wikipedia/wikipedia_PNG4.png"
        })

      target = Path.join(tmp_dir, "wikipedia.png")
      assert {:ok, target} == Images.save_to_file(public_image, target)
    end

    @tag tmp_dir: true, live_call: true
    test "handles invalid URL", %{tmp_dir: tmp_dir} do
      public_image =
        GeneratedImage.new!(%{
          image_type: :png,
          type: :url,
          content: "https://pngimg.com/uploads/wikipedia/wikipedia_PNG4_missing.png"
        })

      target = Path.join(tmp_dir, "invalid_image.png")
      {:error, "Image file not found"} = Images.save_to_file(public_image, target)
    end

    @tag tmp_dir: true
    test "saves a base64 encoded image to a file", %{tmp_dir: tmp_dir} do
      public_image =
        GeneratedImage.new!(%{
          image_type: :png,
          type: :base64,
          content:
            "iVBORw0KGgoAAAANSUhEUgAAAAgAAAAIAQMAAAD+wSzIAAAABlBMVEX///+/v7+jQ3Y5AAAADklEQVQI12P4AIX8EAgALgAD/aNpbtEAAAAASUVORK5CYII="
        })

      target = Path.join(tmp_dir, "sample.png")
      assert {:ok, ^target} = Images.save_to_file(public_image, target)
    end
  end
end
