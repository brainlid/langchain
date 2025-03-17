defmodule LangChain.Images.OpenAIImage do
  @moduledoc """
  Represents the [OpenAI Images API
  endpoint](https://platform.openai.com/docs/api-reference/images) for working
  with DALL-E-2 and DALL-E-3.

  Parses and validates inputs for making a request from the OpenAI Image API.

  Converts responses into more specialized `LangChain` data structures and
  provide functions for saving generated images.

  """
  use Ecto.Schema
  require Logger
  import Ecto.Changeset
  alias __MODULE__
  alias LangChain.Images.GeneratedImage
  alias LangChain.Config
  alias LangChain.LangChainError
  alias LangChain.Utils

  # allow up to 2 minutes for response.
  @receive_timeout 1_200_000

  @primary_key false
  embedded_schema do
    field :endpoint, :string, default: "https://api.openai.com/v1/images/generations"
    # API key for OpenAI. If not set, will use global api key. Allows for usage
    # of a different API key per-call if desired. For instance, allowing a
    # customer to provide their own.
    field :api_key, :string, redact: true
    # Duration in seconds for the response to be received. When streaming a very
    # lengthy response, a longer time limit may be required. However, when it
    # goes on too long by itself, it tends to hallucinate more.
    field :receive_timeout, :integer, default: @receive_timeout

    # Defaults to `dall-e-2`. The other model option is `dall-e-3`.
    # The model to use for image generation.
    field :model, :string, default: "dall-e-2"

    # A text description of the desired image(s). The maximum length is 1000
    # characters for `dall-e-2` and 4000 characters for `dall-e-3`.
    field :prompt, :string

    # The number of images to generate. Must be between 1 and 10. For dall-e-3,
    # only n=1 is supported.
    field :n, :integer, default: 1

    #  The quality of the image that will be generated. `hd` creates images with
    #  finer details and greater consistency across the image. This param is
    #  only supported for `dall-e-3`.
    field :quality, :string, default: "standard"

    # The format in which the generated images are returned. Must be one of
    # `url` or `b64_json`. URLs are only valid for 60 minutes after the image
    # has been generated.
    field :response_format, :string, default: "url"

    # The size of the generated images. Must be one of `256x256`, `512x512`, or
    # `1024x1024` for `dall-e-2`. Must be one of `1024x1024`, `1792x1024`, or `1024x1792`
    # for `dall-e-3` models.
    field :size, :string, default: "1024x1024"

    # The style of the generated images. Must be one of `vivid` or `natural`.
    # Vivid causes the model to lean towards generating hyper-real and dramatic
    # images. Natural causes the model to produce more natural, less hyper-real
    # looking images. This param is only supported for `dall-e-3`.
    field :style, :string, default: "vivid"

    # A unique identifier representing your end-user, which can help OpenAI to
    # monitor and detect abuse
    field :user, :string
  end

  @type t :: %OpenAIImage{}

  @create_fields [
    :endpoint,
    :api_key,
    :receive_timeout,
    :model,
    :prompt,
    :n,
    :quality,
    :response_format,
    :size,
    :style,
    :user
  ]
  @required_fields [:endpoint, :model, :prompt]

  @spec get_api_key(t()) :: String.t()
  defp get_api_key(%OpenAIImage{api_key: api_key}) do
    # if no API key is set default to `""` which will raise a OpenAI API error
    api_key || Config.resolve(:openai_key, "")
  end

  @spec get_org_id() :: String.t() | nil
  defp get_org_id() do
    Config.resolve(:openai_org_id)
  end

  @spec get_proj_id() :: String.t() | nil
  defp get_proj_id() do
    Config.resolve(:openai_proj_id)
  end

  @doc """
  Setup a OpenAIImage client configuration.
  """
  @spec new(attrs :: map()) :: {:ok, t} | {:error, Ecto.Changeset.t()}
  def new(%{} = attrs) do
    %OpenAIImage{}
    |> cast(attrs, @create_fields)
    |> common_validation()
    |> conditional_validations_for_model()
    |> apply_action(:insert)
  end

  @doc """
  Setup a OpenAIImage client configuration and return it or raise an error if
  invalid.
  """
  @spec new!(attrs :: map()) :: t() | no_return()
  def new!(attrs) do
    case new(attrs) do
      {:ok, chain} ->
        chain

      {:error, changeset} ->
        raise LangChainError, changeset
    end
  end

  defp common_validation(changeset) do
    changeset
    |> validate_required(@required_fields)
    |> validate_inclusion(:model, ["dall-e-2", "dall-e-3"])
    |> validate_inclusion(:quality, ["standard", "hd"])
    |> validate_inclusion(:response_format, ["url", "b64_json"])
    |> validate_inclusion(:style, ["vivid", "natural"])
    |> validate_number(:receive_timeout, greater_than_or_equal_to: 0)
  end

  defp conditional_validations_for_model(changeset) do
    case get_field(changeset, :model) do
      "dall-e-3" ->
        changeset
        |> validate_length(:prompt, max: 4_000)
        |> validate_number(:n, equal_to: 1)
        |> validate_inclusion(:size, ["1024x1024", "1792x1024", "1024x1792"])

      "dall-e-2" ->
        changeset
        |> validate_length(:prompt, max: 1_000)
        |> validate_number(:n, greater_than_or_equal_to: 1, less_than_or_equal_to: 10)
        |> validate_inclusion(:size, ["256x256", "512x512", "1024x1024"])

      _other ->
        changeset
    end
  end

  @doc """
  Return the params formatted for an API request.
  """
  @spec for_api(t) :: %{atom() => any()}
  def for_api(%OpenAIImage{} = openai) do
    %{
      model: openai.model,
      prompt: openai.prompt,
      n: openai.n,
      quality: openai.quality,
      response_format: openai.response_format,
      size: openai.size,
      style: openai.style
    }
    |> Utils.conditionally_add_to_map(:user, openai.user)
  end

  @doc """
  Calls the OpenAI API passing the OpenAIImage struct with configuration.

  When successful, it returns `{:ok, generated_images}` where that is a list of
  `LangChain.Images.GeneratedImage` structs.
  """
  @spec call(t()) :: {:ok, [GeneratedImage.t()]} | {:error, String.t()}
  def call(openai)

  def call(%OpenAIImage{} = openai) do
    try do
      # make base api request and perform high-level success/failure checks
      case do_api_request(openai) do
        {:error, reason} ->
          {:error, reason}

        {:ok, parsed_data} ->
          {:ok, parsed_data}
      end
    rescue
      err in LangChainError ->
        {:error, err.message}
    end
  end

  # Make the API request from the OpenAI server.
  #
  # The result of the function is:
  #
  # - `{:ok, %{images: [images], prompt: "the re-written prompt"}}
  # - `{:error, reason}` - Where reason is a string explanation of what went
  #   wrong.
  #

  # Retries the request up to 3 times on transient errors with a brief delay
  @doc false
  @spec do_api_request(t(), retry_count :: integer()) :: {:ok, list()} | {:error, String.t()}
  def do_api_request(openai, retry_count \\ 3)

  def do_api_request(_openai, 0) do
    raise LangChainError, "Retries exceeded. Connection failed."
  end

  def do_api_request(%OpenAIImage{} = openai, retry_count) do
    req =
      Req.new(
        url: openai.endpoint,
        json: for_api(openai),
        # required for OpenAI API
        auth: {:bearer, get_api_key(openai)},
        # required for Azure OpenAI version
        headers: [
          {"api-key", get_api_key(openai)}
        ],
        receive_timeout: openai.receive_timeout,
        retry: :transient,
        max_retries: 3,
        retry_delay: fn attempt -> 300 * attempt end
      )

    req
    |> maybe_add_org_id_header()
    |> maybe_add_proj_id_header()
    |> Req.post()
    # parse the body and return it as parsed structs
    |> case do
      {:ok, %Req.Response{body: data}} ->
        case do_process_response(data, openai) do
          {:error, reason} ->
            {:error, reason}

          result ->
            result
        end

      {:error, %Req.TransportError{reason: :timeout}} ->
        {:error, "Request timed out"}

      {:error, %Req.TransportError{reason: :closed}} ->
        # Force a retry by making a recursive call decrementing the counter
        Logger.debug(fn -> "Mint connection closed: retry count = #{inspect(retry_count)}" end)
        do_api_request(openai, retry_count - 1)

      other ->
        Logger.error("Unexpected and unhandled API response! #{inspect(other)}")
        other
    end
  end

  @doc false
  @spec do_process_response(data :: %{String.t() => any()} | {:error, any()}, t()) ::
          {:ok, [GeneratedImage.t()]} | {:error, String.t()}
  def do_process_response(%{"data" => images} = response, %OpenAIImage{} = request)
      when is_list(images) do
    created_at = DateTime.from_unix!(response["created"])

    results =
      Enum.map(images, fn
        %{"b64_json" => base64_raw_content} = image_info ->
          GeneratedImage.new!(%{
            type: :base64,
            image_type: :png,
            content: base64_raw_content,
            created_at: created_at,
            prompt: Map.get(image_info, "revised_prompt", request.prompt),
            metadata: %{"model" => request.model, "quality" => request.quality}
          })

        %{"url" => base64_raw_content} = image_info ->
          GeneratedImage.new!(%{
            type: :url,
            image_type: :png,
            content: base64_raw_content,
            created_at: created_at,
            prompt: Map.get(image_info, "revised_prompt", request.prompt),
            metadata: %{"model" => request.model, "quality" => request.quality}
          })

        other ->
          message = "Unsupported image data response from OpenAI! #{inspect(other)}"
          Logger.error(message)
          nil
      end)

    {:ok, results |> Enum.reject(&is_nil(&1))}
  end

  def do_process_response(%{"error" => %{"code" => code} = error}, %OpenAIImage{} = _request) do
    Logger.warning("Error from OpenAI: #{error["message"]}")

    reason =
      case code do
        "content_policy_violation" = value ->
          value

        other ->
          Logger.error("Unhandled error code from API: #{inspect(other)}")
          other
      end

    {:error, reason}
  end

  defp maybe_add_org_id_header(%Req.Request{} = req) do
    org_id = get_org_id()

    if org_id do
      Req.Request.put_header(req, "OpenAI-Organization", org_id)
    else
      req
    end
  end

  defp maybe_add_proj_id_header(%Req.Request{} = req) do
    proj_id = get_proj_id()

    if proj_id do
      Req.Request.put_header(req, "OpenAI-Project", proj_id)
    else
      req
    end
  end
end
