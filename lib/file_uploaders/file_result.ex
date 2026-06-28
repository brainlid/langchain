defmodule LangChain.FileUploader.FileResult do
  @moduledoc """
  Represents the result of a file upload to an LLM provider.

  Different providers return different types of file references:

  - **OpenAI** and **Anthropic** return a `file_id` string
  - **Google Gemini** returns a `file_uri` URL

  At least one of `file_id` or `file_uri` will always be present.

  The `raw` field contains the full provider response for accessing
  provider-specific fields not covered by the struct.
  """
  use Ecto.Schema
  import Ecto.Changeset
  alias __MODULE__
  alias LangChain.LangChainError

  @primary_key false
  embedded_schema do
    field :file_id, :string
    field :file_uri, :string
    field :filename, :string
    field :mime_type, :string
    field :size_bytes, :integer
    field :provider, Ecto.Enum, values: [:openai, :anthropic, :google]
    field :raw, :map, virtual: true, default: %{}
  end

  @type t :: %FileResult{}

  @create_fields [:file_id, :file_uri, :filename, :mime_type, :size_bytes, :provider, :raw]
  @required_fields [:provider]

  @doc """
  Build a new FileResult and return an `:ok`/`:error` tuple with the result.
  """
  @spec new(attrs :: map()) :: {:ok, t()} | {:error, Ecto.Changeset.t()}
  def new(%{} = attrs) do
    %FileResult{}
    |> cast(attrs, @create_fields)
    |> validate_required(@required_fields)
    |> validate_reference_present()
    |> apply_action(:insert)
  end

  @doc """
  Build a new FileResult and return it or raise an error if invalid.
  """
  @spec new!(attrs :: map()) :: t() | no_return()
  def new!(attrs) do
    case new(attrs) do
      {:ok, result} -> result
      {:error, changeset} -> raise LangChainError, changeset
    end
  end

  defp validate_reference_present(changeset) do
    file_id = get_field(changeset, :file_id)
    file_uri = get_field(changeset, :file_uri)

    if is_nil(file_id) and is_nil(file_uri) do
      add_error(changeset, :file_id, "either file_id or file_uri must be present")
    else
      changeset
    end
  end
end
