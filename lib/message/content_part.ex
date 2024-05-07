defmodule LangChain.Message.ContentPart do
  @moduledoc """
  Models a `ContentPart`. Some LLMs support combining text, images, and possibly
  other content as part of a single user message. A `ContentPart` represents a
  block, or part, of a message's content that is all of one type.

  ## Types

  - `:text` - The message part is text.
  - `:image_url` - The message part is a URL to an image.
  - `:image` - The message part is image data that is base64 encoded text.

  ## Fields

  - `:content` - Text content.
  - `:options` - Options that may be specific to the LLM for a particular
    message type. For example, multi-modal message (ones that include image
    data) use the `:media` option to specify the mimetype information.

  ## Image mime types

  The `:media` option is used to specify the mime type of the image. Various
  LLMs handle this differently or perhaps not at all.

  Examples:

  - `media: :jpg` - turns into `"image/jpeg"` or `"image/jpg"`, depending on
    what the LLM accepts.
  - `media: :png` - turns into `"image/png"`
  - `media: "image/webp" - stays as `"image/webp"`. Any specified string value
    is passed through unchanged. This allows for future formats to be supported
    quickly.
  - When omitted, the LLM may error or some will accept it but may require the
    `base64` encoded content data to be prefixed with the mime type information.
    Basically, you must handle the content needs yourself.

  """
  use Ecto.Schema
  import Ecto.Changeset
  require Logger
  alias __MODULE__
  alias LangChain.LangChainError

  @primary_key false
  embedded_schema do
    field :type, Ecto.Enum, values: [:text, :image_url, :image], default: :text
    field :content, :string
    field :options, :any, virtual: true
  end

  @type t :: %ContentPart{}

  @update_fields [:type, :content, :options]
  @create_fields @update_fields
  @required_fields [:type, :content]

  @doc """
  Build a new message and return an `:ok`/`:error` tuple with the result.
  """
  @spec new(attrs :: map()) :: {:ok, t()} | {:error, Ecto.Changeset.t()}
  def new(attrs \\ %{}) do
    %ContentPart{}
    |> cast(attrs, @create_fields)
    |> common_validations()
    |> apply_action(:insert)
  end

  @doc """
  Build a new message and return it or raise an error if invalid.

  ## Example

      ContentPart.new!(%{type: :text, content: "Greetings!"})

      ContentPart.new!(%{type: :image_url, content: "https://example.com/images/house.jpg"})
  """
  @spec new!(attrs :: map()) :: t() | no_return()
  def new!(attrs \\ %{}) do
    case new(attrs) do
      {:ok, message} ->
        message

      {:error, changeset} ->
        raise LangChainError, changeset
    end
  end

  @doc """
  Create a new ContentPart that contains text. Raises an exception if not valid.
  """
  @spec text!(String.t()) :: t() | no_return()
  def text!(content) do
    new!(%{type: :text, content: content})
  end

  @doc """
  Create a new ContentPart that contains an image encoded as base64 data. Raises
  an exception if not valid.

  ## Options

  - `:media` - Provide the "media type" for the image. Examples: "image/jpeg",
    "image/png", etc. ChatGPT does not require this but other LLMs may.
  - `:detail` - if the LLM supports it, most images must be resized or cropped
    before given to the LLM for analysis. A detail option may specify the level
    detail of the image to present to the LLM. The higher the detail, the more
    tokens consumed. Currently only supported by OpenAI and the values of "low",
    "high", and "auto".

  ChatGPT requires media type information to prefix the base64 content. Setting
  the `media: "image/jpeg"` type will do that. Otherwise the data must be
  provided with the required prefix.

  Anthropic requires the media type information to be submitted as separate
  information with the JSON request. This media option provides an abstraction
  to normalize the behavior.
  """
  @spec image!(String.t(), Keyword.t()) :: t() | no_return()
  def image!(content, opts \\ []) do
    new!(%{type: :image, content: content, options: opts})
  end

  @doc """
  Create a new ContentPart that contains a URL to an image. Raises an exception if not valid.
  """
  @spec image_url!(String.t(), Keyword.t()) :: t() | no_return()
  def image_url!(content, opts \\ []) do
    new!(%{type: :image_url, content: content, options: opts})
  end

  @doc false
  def changeset(message, attrs) do
    message
    |> cast(attrs, @update_fields)
    |> common_validations()
  end

  defp common_validations(changeset) do
    changeset
    |> validate_required(@required_fields)
  end
end
