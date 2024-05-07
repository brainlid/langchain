defmodule LangChain.Images.GeneratedImage do
  @moduledoc """
  Represents a generated image where we have either the base64 encoded contents
  or a temporary URL to it.
  """
  use Ecto.Schema
  import Ecto.Changeset
  require Logger
  alias __MODULE__
  alias LangChain.LangChainError

  @primary_key false
  embedded_schema do
    field :image_type, Ecto.Enum, values: [:png, :jpg], default: :png
    field :type, Ecto.Enum, values: [:base64, :url], default: :url
    # When a :url, content is the URL. When base64, content is the encoded data.
    field :content, :string

    # The prompt used when generating the image. It may have been altered by the
    # LLM from the original request.
    field :prompt, :string
    field :metadata, :map
    field :created_at, :utc_datetime
  end

  @type t :: %GeneratedImage{}

  @update_fields [:image_type, :type, :content, :prompt, :metadata, :created_at]
  @create_fields @update_fields

  @doc """
  Build a new GeneratedImage and return an `:ok`/`:error` tuple with the result.
  """
  @spec new(attrs :: map()) :: {:ok, t()} | {:error, Ecto.Changeset.t()}
  def new(attrs \\ %{}) do
    %GeneratedImage{}
    |> cast(attrs, @create_fields)
    |> common_validations()
    |> apply_action(:insert)
  end

  @doc """
  Build a new GeneratedImage and return it or raise an error if invalid.
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

  defp common_validations(changeset) do
    changeset
    |> validate_required([:image_type, :type, :content])
  end

end
