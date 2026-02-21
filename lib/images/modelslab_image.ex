defmodule LangChain.Images.ModelsLabImage do
  @moduledoc """
  Represents the [ModelsLab Images API](https://docs.modelslab.com/image-generation/overview)
  for text-to-image generation using Flux, SDXL, Stable Diffusion, and 10,000+
  community fine-tuned models.

  ## Configuration

  Set your API key in your application config:

      config :langchain, :modelslab_key, "your-api-key"

  Or pass it directly when creating the struct:

      {:ok, ml_image} = ModelsLabImage.new(%{
        api_key: "your-api-key",
        prompt: "A cozy cabin in the woods at dusk"
      })

  ## Usage

      {:ok, ml_image} = LangChain.Images.ModelsLabImage.new(%{
        prompt: "A sunset over mountains in watercolor style",
        model: "flux",
        width: 1024,
        height: 1024
      })

      {:ok, images} = LangChain.Images.ModelsLabImage.call(ml_image)
      LangChain.Images.save_images({:ok, images}, "/tmp", "my_image_")

  ## Available models

  - `"flux"` — High-quality photorealistic Flux model (default)
  - `"flux-dev"` — Flux development variant
  - `"sdxl"` — Stable Diffusion XL
  - `"realistic-vision-v6"` — Photorealistic portraits
  - `"dreamshaper-8"` — Artistic and creative styles
  - `"anything-v5"` — Anime and illustration style
  - Any community model ID from [modelslab.com/models](https://modelslab.com/models)

  ## API docs

  - [ModelsLab text-to-image API](https://docs.modelslab.com/image-generation/overview)
  - [Get API key](https://modelslab.com/dashboard/api-keys)
  """
  use Ecto.Schema
  require Logger
  import Ecto.Changeset
  alias __MODULE__
  alias LangChain.Images.GeneratedImage
  alias LangChain.Config
  alias LangChain.LangChainError
  alias LangChain.Utils

  # allow up to 2 minutes for response
  @receive_timeout 120_000

  @primary_key false
  embedded_schema do
    field :endpoint, :string,
      default: "https://modelslab.com/api/v6/images/text2img"

    # API key for ModelsLab. If not set, resolves from config :modelslab_key.
    field :api_key, :string, redact: true

    # Duration in seconds for the response to be received.
    field :receive_timeout, :integer, default: @receive_timeout

    # Model ID. Can be "flux", "sdxl", or any community model from modelslab.com.
    field :model, :string, default: "flux"

    # Text description of the desired image.
    field :prompt, :string

    # Text describing what to exclude from the image.
    field :negative_prompt, :string

    # Width in pixels. Must be between 256 and 1024.
    field :width, :integer, default: 1024

    # Height in pixels. Must be between 256 and 1024.
    field :height, :integer, default: 1024

    # Number of images to generate. Must be between 1 and 4.
    field :samples, :integer, default: 1

    # Number of inference steps. Higher means better quality, more time.
    field :num_inference_steps, :integer, default: 30

    # Guidance scale (CFG). Higher means the image sticks more closely to the prompt.
    field :guidance_scale, :float, default: 7.5

    # Optional seed for reproducible results.
    field :seed, :integer
  end

  @type t :: %ModelsLabImage{}

  @create_fields [
    :endpoint,
    :api_key,
    :receive_timeout,
    :model,
    :prompt,
    :negative_prompt,
    :width,
    :height,
    :samples,
    :num_inference_steps,
    :guidance_scale,
    :seed
  ]
  @required_fields [:endpoint, :model, :prompt]

  @spec get_api_key(t()) :: String.t()
  defp get_api_key(%ModelsLabImage{api_key: api_key}) do
    api_key || Config.resolve(:modelslab_key, "")
  end

  @doc """
  Build a new `ModelsLabImage` configuration.
  """
  @spec new(attrs :: map()) :: {:ok, t()} | {:error, Ecto.Changeset.t()}
  def new(%{} = attrs) do
    %ModelsLabImage{}
    |> cast(attrs, @create_fields)
    |> common_validation()
    |> apply_action(:insert)
  end

  @doc """
  Build a new `ModelsLabImage` configuration and return it or raise on error.
  """
  @spec new!(attrs :: map()) :: t() | no_return()
  def new!(attrs) do
    case new(attrs) do
      {:ok, ml_image} ->
        ml_image

      {:error, changeset} ->
        raise LangChainError, changeset
    end
  end

  defp common_validation(changeset) do
    changeset
    |> validate_required(@required_fields)
    |> validate_number(:receive_timeout, greater_than_or_equal_to: 0)
    |> validate_number(:width, greater_than_or_equal_to: 256, less_than_or_equal_to: 1024)
    |> validate_number(:height, greater_than_or_equal_to: 256, less_than_or_equal_to: 1024)
    |> validate_number(:samples, greater_than_or_equal_to: 1, less_than_or_equal_to: 4)
    |> validate_number(:num_inference_steps, greater_than_or_equal_to: 1, less_than_or_equal_to: 50)
    |> validate_number(:guidance_scale, greater_than_or_equal_to: 1.0, less_than_or_equal_to: 20.0)
  end

  @doc """
  Return the params formatted for a ModelsLab API request.
  """
  @spec for_api(t()) :: map()
  def for_api(%ModelsLabImage{} = ml_image) do
    %{
      "key" => get_api_key(ml_image),
      "prompt" => ml_image.prompt,
      "model_id" => ml_image.model,
      "width" => to_string(ml_image.width),
      "height" => to_string(ml_image.height),
      "samples" => to_string(ml_image.samples),
      "num_inference_steps" => to_string(ml_image.num_inference_steps),
      "guidance_scale" => ml_image.guidance_scale,
      "safety_checker" => "no"
    }
    |> maybe_add("negative_prompt", ml_image.negative_prompt)
    |> maybe_add("seed", ml_image.seed)
  end

  defp maybe_add(map, _key, nil), do: map
  defp maybe_add(map, key, value), do: Map.put(map, key, value)

  @doc """
  Call the ModelsLab API and return generated images.

  Returns `{:ok, [GeneratedImage.t()]}` on success or `{:error, reason}` on
  failure.
  """
  @spec call(t()) :: {:ok, [GeneratedImage.t()]} | {:error, String.t()}
  def call(%ModelsLabImage{} = ml_image) do
    try do
      do_api_request(ml_image)
    rescue
      err in LangChainError ->
        {:error, err.message}
    end
  end

  @doc false
  @spec do_api_request(t(), retry_count :: integer()) ::
          {:ok, [GeneratedImage.t()]} | {:error, String.t()}
  def do_api_request(ml_image, retry_count \\ 3)

  def do_api_request(_ml_image, 0) do
    raise LangChainError, "Retries exceeded. Connection failed."
  end

  def do_api_request(%ModelsLabImage{} = ml_image, retry_count) do
    req =
      Req.new(
        url: ml_image.endpoint,
        json: for_api(ml_image),
        receive_timeout: ml_image.receive_timeout,
        retry: :transient,
        max_retries: 3,
        retry_delay: fn attempt -> 300 * attempt end
      )

    req
    |> Req.post()
    |> case do
      {:ok, %Req.Response{body: data}} ->
        do_process_response(data, ml_image)

      {:error, %Req.TransportError{reason: :timeout}} ->
        {:error, "Request timed out"}

      {:error, %Req.TransportError{reason: :closed}} ->
        Logger.debug(fn -> "Connection closed: retry count = #{inspect(retry_count)}" end)
        do_api_request(ml_image, retry_count - 1)

      other ->
        Logger.error("Unexpected ModelsLab API response: #{inspect(other)}")
        {:error, "Unexpected response from ModelsLab API"}
    end
  end

  @doc false
  @spec do_process_response(data :: map(), t()) ::
          {:ok, [GeneratedImage.t()]} | {:error, String.t()}
  def do_process_response(%{"status" => "success", "output" => urls} = _data, %ModelsLabImage{} = request)
      when is_list(urls) do
    created_at = DateTime.utc_now()

    results =
      Enum.map(urls, fn url ->
        GeneratedImage.new!(%{
          type: :url,
          image_type: :png,
          content: url,
          created_at: created_at,
          prompt: request.prompt,
          metadata: %{"model" => request.model, "provider" => "modelslab"}
        })
      end)

    {:ok, results}
  end

  def do_process_response(%{"status" => "error"} = data, _request) do
    message = data["message"] || data["messege"] || "Unknown ModelsLab error"
    Logger.warning("ModelsLab error: #{message}")
    {:error, message}
  end

  def do_process_response(%{"status" => "processing"} = data, _request) do
    id = data["id"]
    message = "Image generation is still processing (id: #{id}). Try again shortly."
    Logger.warning(message)
    {:error, message}
  end

  def do_process_response(data, _request) do
    Logger.error("Unexpected ModelsLab response: #{inspect(data)}")
    {:error, "Unexpected response format from ModelsLab"}
  end
end
