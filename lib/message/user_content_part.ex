defmodule LangChain.Message.UserContentPart do
  @moduledoc """
  Models a `UserContentPart`. Some LLMs support combining text, images, and possibly
  other content as part of a single user message. A `UserContentPart` represents a
  block, or part, of a message's content that is all of one type.

  ## Types

  - `:text` - The message part is text.
  - `:image_url` - The message part is a URL to an image.
  - `:image` - The message part is image data that is base64 encoded text.
  - `:tool_call` - An assistant can specify a tool it wants called and provide
    arguments.

  ## Fields

  - `:content` - Text content.
  - `:tool_type` - Type of tool being called. `:function` is the only one
    currently defined.
  - `:tool_name` - The name of the requested tool to call.
  - `:tool_arguments` - Arguments provided for the tool call.
  - `:options` - Options that may be specific to the LLM for a particular
    message type. For example, Anthropic requires an image's `media_type` to be
    provided by the caller. This can be provided using `media: "image/png"`.

  """
  use Ecto.Schema
  import Ecto.Changeset
  require Logger
  alias __MODULE__
  alias LangChain.LangChainError

  @primary_key false
  embedded_schema do
    field :type, Ecto.Enum, values: [:text, :image_url, :image, :tool_call], default: :text
    # tool results are sent through content as well.
    field :content, :string
    field :options, :any, virtual: true


    # tool id links the result back to a specific tool request
    field :tool_id, :string
    field :tool_type, Ecto.Enum, values: [:function], default: :function
    field :tool_name, :string
    field :tool_arguments, :any, virtual: true
  end

  # https://cookbook.openai.com/examples/how_to_call_functions_with_chat_models
  #
  # if assistant_message.tool_calls:
  #   results = execute_function_call(assistant_message)
  #   messages.append({"role": "function", "tool_call_id": assistant_message.tool_calls[0].id, "name": assistant_message.tool_calls[0].function.name, "content": results})

  @type t :: %UserContentPart{}

  @update_fields [:type, :content, :options]
  @create_fields @update_fields
  @required_fields [:type, :content]

  @tool_create_fields [:type, :tool_id, :tool_type, :tool_name, :tool_arguments]

  @doc """
  Build a new message and return an `:ok`/`:error` tuple with the result.
  """
  @spec new(attrs :: map()) :: {:ok, t()} | {:error, Ecto.Changeset.t()}
  def new(attrs \\ %{}) do
    %UserContentPart{}
    |> cast(attrs, @create_fields)
    |> common_validations()
    |> apply_action(:insert)
  end

  @doc """
  Build a new message and return it or raise an error if invalid.
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
  Create a new UserContentPart that contains text. Raises an exception if not valid.
  """
  @spec text!(String.t()) :: t() | no_return()
  def text!(content) do
    new!(%{type: :text, content: content})
  end

  @doc """
  Create a new UserContentPart that contains an image encoded as base64 data. Raises
  an exception if not valid.

  ## Options

  - `:media` - Provide the "media type" for the image. Examples: "image/jpeg",
    "image/png", etc. ChatGPT does not require this but other LLMs may.
  """
  @spec image!(String.t(), Keyword.t()) :: t() | no_return()
  def image!(content, opts \\ []) do
    new!(%{type: :image, content: content, options: opts})
  end

  @doc """
  Create a new UserContentPart that contains a URL to an image. Raises an exception if not valid.
  """
  @spec image_url!(String.t()) :: t() | no_return()
  def image_url!(content) do
    new!(%{type: :image_url, content: content})
  end

  # @doc """
  # Create a new UserContentPart that represents a tool call.
  # """
  # @spec tool_call(attrs :: map()) :: {:ok, t()} | {:error, Ecto.Changeset.t()}
  # def tool_call(attrs) do
  #   attrs
  #   |> tool_changeset()
  #   |> apply_action(:insert)
  # end

  # @doc """
  # Create a new UserContentPart that represents a tool call. Returns the UserContentPart
  # or raises an exception if invalid.
  # """
  # @spec tool_call!(attrs :: map()) :: t() | no_return()
  # def tool_call!(attrs) do
  #   case tool_call(attrs) do
  #     {:ok, part} ->
  #       part

  #     {:error, changeset} ->
  #       raise LangChainError, changeset
  #   end
  # end

  # @doc """
  # Changeset when setting up a tool_call.
  # """
  # def tool_changeset(attrs) do
  #   %UserContentPart{}
  #   |> cast(attrs, @tool_create_fields)
  #   |> put_change(:type, :tool_call)
  #   |> validate_required([:type, :tool_name])
  #   |> parse_tool_arguments()
  # end

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

  defp parse_tool_arguments(changeset) do
    case get_change(changeset, :tool_arguments) do
      nil ->
        changeset

      text when is_binary(text) ->
        # assume JSON and convert. If invalid, add an error
        case Jason.decode(text) do
          {:ok, json} ->
            put_change(changeset, :tool_arguments, json)

          {:error, reason} ->
            add_error(changeset, :tool_arguments, "invalid json")
        end

      data when is_map(data) ->
        # return unmodified
        changeset
    end
  end
end
