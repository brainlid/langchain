defmodule LangChain.Message.ContentPart do
  @moduledoc """
  Models a `ContentPart`. Some LLMs support combining text, images, and possibly
  other content as part of a single user message or other content parts like
  "thinking" may be part of an assistant's response message. A `ContentPart`
  represents a block, or part, of a message's content that is all of one type.

  ## Types

  - `:text` - The message part is text.
  - `:image_url` - The message part is a URL to an image.
  - `:image` - The message part is image data that is base64 encoded text.
  - `:file` - The message part is file data that is base64 encoded text.
  - `:thinking` - A thinking block from a reasoning model like Anthropic.
  - `:unsupported` - A part that is not supported but may need to be present.
    This includes Anthropic's `redacted_thinking` block which has no value in
    being displayed because it is encrypted, but can be provided back to the LLM
    to maintain reasoning continuity. The specific parts of the data are stored
    in `:options`.

  ## Fields

  - `:content` - Text content.
  - `:options` - Options are a keyword list of values that may be specific to
    the LLM for a particular message type. For example, multi-modal message
    (ones that include image data) use the `:media` option to specify the
    mimetype information. Options may also contain key-value settings like
    `cache_control: true` for models like Anthropic that support caching.

    When receiving content parts like with Anthropic Claude's thinking model,
    the options may contain LLM specific data that is recommended to be
    preserved like a `signature` or `redacted_thinking` data used by the LLM.

  ## Image mime types

  The `:media` option is used to specify the mime type of the image. Various
  LLMs handle this differently or perhaps not at all.

  Examples:

  - `media: :jpg` - turns into `"image/jpeg"` or `"image/jpg"`, depending on
    what the LLM accepts.
  - `media: :png` - turns into `"image/png"`
  - `media: "image/webp"` - stays as `"image/webp"`. Any specified string value
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
  alias LangChain.Utils

  @primary_key false
  embedded_schema do
    field :type, Ecto.Enum,
      values: [:text, :image_url, :image, :file, :thinking, :unsupported],
      default: :text

    field :content, :string
    field :options, :any, virtual: true, default: []
  end

  @type t :: %ContentPart{}

  @update_fields [:type, :content, :options]
  @create_fields @update_fields
  @required_fields [:type]

  @doc """
  Build a new message and return an `:ok`/`:error` tuple with the result.
  """
  @spec new(attrs :: map()) :: {:ok, t()} | {:error, Ecto.Changeset.t()}
  def new(attrs \\ %{}) do
    %ContentPart{}
    |> cast(attrs, @create_fields)
    |> Utils.assign_string_value(:content, attrs)
    |> common_validations()
    |> apply_action(:insert)
  end

  @doc """
  Build a new message and return it or raise an error if invalid.

  ## Example

      ContentPart.new!(%{type: :text, content: "Greetings!"})

      ContentPart.new!(%{type: :image_url, content: "https://example.com/images/house.jpg"})

      ContentPart.new!(%{type: :thinking, content: "I've been asked...", options: [signature: "SIGNATURE_DATA"]}

      ContentPart.new!(%{type: :unsupported, content: "redacted_data", options: [type: "redacted_thinking"]}
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
  def text!(content, opts \\ []) do
    new!(%{type: :text, content: content, options: opts})
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
  Create a new ContentPart that contains a file encoded as base64 data.
  """
  @spec file!(String.t(), Keyword.t()) :: t() | no_return()
  def file!(content, opts \\ []) do
    new!(%{type: :file, content: content, options: opts})
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

  @doc """
  Merge two `ContentPart` structs for the same index in a MessageDelta. The
  first `ContentPart` is the `primary` one that smaller deltas are merged into.
  The primary is what is being accumulated.

  A set of ContentParts can be merged like this:

      Enum.reduce(list_of_content_parts, nil, fn new_part, acc ->
        ContentPart.merge_part(acc, new_part)
      end)

  """
  @spec merge_part(nil | t(), t()) :: t()
  def merge_part(nil, %ContentPart{} = new_part), do: new_part

  def merge_part(%ContentPart{} = primary, %ContentPart{} = content_part) do
    primary
    |> append_content(content_part)
    |> update_options(content_part)
  end

  # text content being merged
  defp append_content(
         %ContentPart{type: primary_type, content: primary_content} = primary,
         %ContentPart{
           type: new_type,
           content: new_content
         }
       )
       when is_binary(primary_content) and is_binary(new_content) and primary_type == new_type do
    %ContentPart{primary | content: (primary.content || "") <> new_content}
  end

  defp append_content(%ContentPart{} = primary, %ContentPart{content: nil}), do: primary

  # Merge options from content_part into primary. When text, combine the options
  # and merge in newly encountered keys.
  defp update_options(%ContentPart{} = primary, %ContentPart{options: nil}), do: primary

  defp update_options(%ContentPart{options: nil} = primary, %ContentPart{options: new_opts}) do
    %ContentPart{primary | options: new_opts}
  end

  defp update_options(%ContentPart{options: primary_opts} = primary, %ContentPart{
         options: new_opts
       })
       when is_list(primary_opts) and is_list(new_opts) do
    # Concatenate any string values with the same key, otherwise use the new
    # value
    merged_opts = Keyword.merge(primary_opts, new_opts, fn
      _k, v1, v2 when is_binary(v1) and is_binary(v2) -> v1 <> v2
      _k, _v1, v2 -> v2
    end)
    %ContentPart{primary | options: merged_opts}
  end

  defp update_options(%ContentPart{} = primary, %ContentPart{}), do: primary

  @doc """
  Helper function for easily getting plain text from a list of ContentParts.

  This function processes a list of ContentParts and joins the text parts together
  using "\n\n" characters. Only parts where `type: :text` are used. All other parts
  are ignored.

  ## Examples

      iex> parts = [
      ...>   text!("Hello"),
      ...>   image!("base64data"),
      ...>   text!("world")
      ...> ]
      iex> parts_to_string(parts)
      "Hello\\n\\nworld"

      iex> parts_to_string([])
      nil
  """
  @spec parts_to_string([t()]) :: nil | String.t()
  def parts_to_string(parts) when is_list(parts) do
    parts
    |> Enum.filter(fn part -> part.type == :text end)
    |> Enum.map(fn part -> part.content end)
    |> Enum.join("\n\n")
    |> case do
      "" -> nil
      content -> content
    end
  end
end
