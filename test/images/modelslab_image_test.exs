defmodule LangChain.Images.ModelsLabImageTest do
  use ExUnit.Case
  alias LangChain.Images.GeneratedImage
  alias LangChain.Images.ModelsLabImage
  alias LangChain.LangChainError

  describe "new/1" do
    test "creates valid model with minimal setup" do
      prompt = "A cozy cabin in the woods at dusk."
      {:ok, %ModelsLabImage{} = img} = ModelsLabImage.new(%{prompt: prompt})
      assert img.prompt == prompt
      assert img.model == "flux"
      assert img.width == 1024
      assert img.height == 1024
    end

    test "supports all valid settings" do
      {:ok, img} =
        ModelsLabImage.new(%{
          endpoint: "https://example.com",
          api_key: "test-key",
          receive_timeout: 30_000,
          model: "sdxl",
          prompt: "A mountain landscape in oil painting style",
          negative_prompt: "blurry, low quality",
          width: 512,
          height: 512,
          samples: 2,
          num_inference_steps: 40,
          guidance_scale: 8.0,
          seed: 12345
        })

      assert img.endpoint == "https://example.com"
      assert img.api_key == "test-key"
      assert img.receive_timeout == 30_000
      assert img.model == "sdxl"
      assert img.prompt == "A mountain landscape in oil painting style"
      assert img.negative_prompt == "blurry, low quality"
      assert img.width == 512
      assert img.height == 512
      assert img.samples == 2
      assert img.num_inference_steps == 40
      assert img.guidance_scale == 8.0
      assert img.seed == 12345
    end

    test "returns error when prompt is missing" do
      {:error, changeset} = ModelsLabImage.new(%{model: "flux"})
      assert {"can't be blank", _} = changeset.errors[:prompt]
    end

    test "validates width bounds" do
      {:error, changeset} = ModelsLabImage.new(%{prompt: "test", width: 128})
      assert changeset.errors[:width] != nil
    end

    test "validates height bounds" do
      {:error, changeset} = ModelsLabImage.new(%{prompt: "test", height: 2048})
      assert changeset.errors[:height] != nil
    end

    test "validates samples range" do
      {:error, changeset} = ModelsLabImage.new(%{prompt: "test", samples: 5})
      assert changeset.errors[:samples] != nil
    end
  end

  describe "new!/1" do
    test "returns struct when valid" do
      %ModelsLabImage{} = img = ModelsLabImage.new!(%{prompt: "A cat sitting on a couch."})
      assert img.prompt == "A cat sitting on a couch."
    end

    test "raises LangChainError when invalid" do
      assert_raise LangChainError, fn ->
        ModelsLabImage.new!(%{})
      end
    end
  end

  describe "for_api/1" do
    test "includes required fields" do
      {:ok, img} = ModelsLabImage.new(%{prompt: "test prompt", api_key: "my-key"})
      params = ModelsLabImage.for_api(img)

      assert params["key"] == "my-key"
      assert params["prompt"] == "test prompt"
      assert params["model_id"] == "flux"
      assert params["safety_checker"] == "no"
    end

    test "omits nil optional fields" do
      {:ok, img} = ModelsLabImage.new(%{prompt: "test", api_key: "key"})
      params = ModelsLabImage.for_api(img)

      refute Map.has_key?(params, "negative_prompt")
      refute Map.has_key?(params, "seed")
    end

    test "includes optional fields when set" do
      {:ok, img} =
        ModelsLabImage.new(%{
          prompt: "test",
          api_key: "key",
          negative_prompt: "blurry",
          seed: 999
        })

      params = ModelsLabImage.for_api(img)
      assert params["negative_prompt"] == "blurry"
      assert params["seed"] == 999
    end
  end

  describe "do_process_response/2" do
    test "handles success response with URL output" do
      {:ok, img} = ModelsLabImage.new(%{prompt: "A test image", api_key: "key"})

      response = %{
        "status" => "success",
        "output" => ["https://cdn.modelslab.com/images/abc123.png"]
      }

      {:ok, [%GeneratedImage{} = generated]} = ModelsLabImage.do_process_response(response, img)
      assert generated.type == :url
      assert generated.content == "https://cdn.modelslab.com/images/abc123.png"
      assert generated.image_type == :png
      assert generated.prompt == "A test image"
      assert generated.metadata["provider"] == "modelslab"
      assert generated.metadata["model"] == "flux"
    end

    test "handles error response" do
      {:ok, img} = ModelsLabImage.new(%{prompt: "test", api_key: "key"})
      response = %{"status" => "error", "message" => "Invalid API key"}
      {:error, reason} = ModelsLabImage.do_process_response(response, img)
      assert reason == "Invalid API key"
    end

    test "handles error response with typo field" do
      {:ok, img} = ModelsLabImage.new(%{prompt: "test", api_key: "key"})
      response = %{"status" => "error", "messege" => "Rate limited"}
      {:error, reason} = ModelsLabImage.do_process_response(response, img)
      assert reason == "Rate limited"
    end

    test "handles processing response" do
      {:ok, img} = ModelsLabImage.new(%{prompt: "test", api_key: "key"})
      response = %{"status" => "processing", "id" => 12345}
      {:error, reason} = ModelsLabImage.do_process_response(response, img)
      assert String.contains?(reason, "processing")
    end
  end
end
