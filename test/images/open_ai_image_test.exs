defmodule LangChain.Images.OpenAIImageTest do
  use ExUnit.Case
  doctest LangChain.Images.OpenAIImage
  # alias LangChain.Images
  alias LangChain.Images.GeneratedImage
  alias LangChain.Images.OpenAIImage
  alias LangChain.LangChainError

  describe "new/1" do
    test "creates valid model with minimal setup" do
      prompt = "A happy little chipmunk with full cheeks."
      {:ok, %OpenAIImage{} = img} = OpenAIImage.new(%{prompt: prompt})
      assert img.prompt == prompt
    end

    test "supports all valid settings" do
      {:ok, img} =
        OpenAIImage.new(%{
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
      {:error, changeset} = OpenAIImage.new(%{model: "dang-e-infinity"})
      assert {"can't be blank", _} = changeset.errors[:prompt]
      assert {"is invalid", _} = changeset.errors[:model]
    end
  end

  describe "new!/1" do
    test "returns struct when valid" do
      prompt = "A happy little chipmunk with full cheeks."
      %OpenAIImage{} = img = OpenAIImage.new!(%{prompt: prompt})
      assert img.prompt == prompt
    end

    test "raises LangChain.LangChainError when invalid" do
      assert_raise LangChainError, "prompt: can't be blank", fn ->
        OpenAIImage.new!(%{})
      end
    end
  end

  describe "call/1" do
    @tag live_call: true, live_open_ai: true
    test "generates and saves an image using b64_json" do
      prompt = "A well worn baseball sitting in a well worn glove"

      result =
        %{
          model: "dall-e-2",
          prompt: prompt,
          quality: "standard",
          response_format: "url",
          size: "256x256",
          style: "natural"
        }
        |> OpenAIImage.new!()
        |> OpenAIImage.call()

      assert {:ok, [%GeneratedImage{} = image]} = result
      assert String.starts_with?(image.content, "http")
      assert image.type == :url
      assert image.image_type == :png
      assert image.prompt == prompt
      assert image.created_at != nil
      assert image.metadata != nil
    end
  end

  describe "do_process_response/2" do
    test "handles successful b64_json response" do
      request = OpenAIImage.new!(%{prompt: "pretend"})
      content = :base64.encode("pretend_content")
      # created is a Unix timestamp in UTC
      data = %{"created" => 1_715_047_358, "data" => [%{"b64_json" => content}]}
      {:ok, [%GeneratedImage{} = image]} = OpenAIImage.do_process_response(data, request)
      assert image.type == :base64
      assert image.image_type == :png
      assert image.content == content
      assert image.prompt == request.prompt
      assert image.created_at == ~U[2024-05-07 02:02:38Z]
      assert %{"model" => "dall-e-2"} = image.metadata
    end

    test "handles successful url response" do
      request = OpenAIImage.new!(%{prompt: "pretend"})
      content = "https://example.com/images/pretend_content.png"
      # created is a Unix timestamp in UTC
      data = %{"created" => 1_715_047_358, "data" => [%{"url" => content}]}
      {:ok, [%GeneratedImage{} = image]} = OpenAIImage.do_process_response(data, request)
      assert image.type == :url
      assert image.image_type == :png
      assert image.content == content
      assert image.prompt == request.prompt
      assert image.created_at == ~U[2024-05-07 02:02:38Z]
      assert %{"model" => "dall-e-2"} = image.metadata
    end

    test "handles when rejected for content violation" do
      response = %{
        "error" => %{
          "code" => "content_policy_violation",
          "message" =>
            "Your request was rejected as a result of our safety system. Your prompt may contain text that is not allowed by our safety system.",
          "param" => nil,
          "type" => "invalid_request_error"
        }
      }

      request = OpenAIImage.new!(%{prompt: "violation"})

      assert {:error, "content_policy_violation"} =
               OpenAIImage.do_process_response(response, request)
    end
  end
end
