defmodule LangChain.FileUploader do
  @moduledoc """
  Behaviour for uploading files to LLM providers.

  Provides a unified interface for uploading files to OpenAI, Anthropic,
  and Google Gemini. Each provider returns a `LangChain.FileUploader.FileResult`
  containing the file reference needed for subsequent API calls.

  ## Usage

      # OpenAI
      {:ok, uploader} = LangChain.FileUploader.FileOpenAI.new(%{api_key: "sk-..."})
      {:ok, result} = LangChain.FileUploader.upload(uploader, file_bytes, %{
        filename: "doc.pdf",
        mime_type: "application/pdf"
      })
      result.file_id
      #=> "file-abc123"

      # Google Gemini
      {:ok, uploader} = LangChain.FileUploader.FileGoogle.new(%{api_key: "AI..."})
      {:ok, result} = LangChain.FileUploader.upload(uploader, file_bytes, %{
        filename: "doc.pdf",
        mime_type: "application/pdf"
      })
      result.file_uri
      #=> "https://generativelanguage.googleapis.com/v1beta/files/..."

  ## File Metadata

  The `file_meta` map passed to `upload/3` accepts the following keys:

  - `:filename` (required) - The name to give the uploaded file.
  - `:mime_type` (required) - The MIME type of the file content.
  - `:purpose` - Provider-specific purpose string. Used by OpenAI
    (e.g. `"user_data"`, `"assistants"`). Ignored by other providers.
  - `:display_name` - A human-readable display name. Used by Google.
    Falls back to `:filename` if not provided.
  """

  alias LangChain.FileUploader.FileResult
  alias LangChain.LangChainError

  @type file_meta :: %{
          required(:filename) => String.t(),
          required(:mime_type) => String.t(),
          optional(:purpose) => String.t(),
          optional(:display_name) => String.t()
        }

  @type upload_result :: {:ok, FileResult.t()} | {:error, LangChainError.t()}
  @type get_result :: {:ok, FileResult.t()} | {:error, LangChainError.t()}
  @type delete_result :: :ok | {:error, LangChainError.t()}
  @type list_result :: {:ok, [FileResult.t()]} | {:error, LangChainError.t()}

  @doc """
  Upload file bytes to the provider and return a FileResult on success.
  """
  @callback upload(config :: struct(), file_bytes :: binary(), file_meta :: file_meta()) ::
              upload_result()

  @doc """
  Retrieve metadata for a previously uploaded file by its ID.
  """
  @callback get(config :: struct(), file_ref :: FileResult.t() | String.t()) :: get_result()

  @doc """
  Delete a previously uploaded file by its ID or URI.
  """
  @callback delete(config :: struct(), file_ref :: FileResult.t() | String.t()) :: delete_result()

  @doc """
  List files previously uploaded to the provider.
  """
  @callback list(config :: struct()) :: list_result()

  @optional_callbacks [get: 2, delete: 2, list: 1]

  @doc """
  Upload a file using any configured uploader implementation.

  Delegates to the provider module's `upload/3` callback.
  """
  @spec upload(struct(), binary(), file_meta()) :: upload_result()
  def upload(%mod{} = uploader, file_bytes, %{} = file_meta) do
    mod.upload(uploader, file_bytes, file_meta)
  end

  @doc """
  Retrieve file metadata using any configured uploader implementation.

  Returns an error if the provider does not implement `get/2`.
  """
  @spec get(struct(), FileResult.t() | String.t()) :: get_result()
  def get(%mod{} = uploader, file_ref) do
    if function_exported?(mod, :get, 2) do
      mod.get(uploader, file_ref)
    else
      {:error, LangChainError.exception(message: "get/2 not implemented for #{inspect(mod)}")}
    end
  end

  @doc """
  Delete a file using any configured uploader implementation.

  Returns an error if the provider does not implement `delete/2`.
  """
  @spec delete(struct(), FileResult.t() | String.t()) :: delete_result()
  def delete(%mod{} = uploader, file_ref) do
    if function_exported?(mod, :delete, 2) do
      mod.delete(uploader, file_ref)
    else
      {:error, LangChainError.exception(message: "delete/2 not implemented for #{inspect(mod)}")}
    end
  end

  @doc """
  List files using any configured uploader implementation.

  Returns an error if the provider does not implement `list/1`.
  """
  @spec list(struct()) :: list_result()
  def list(%mod{} = uploader) do
    if function_exported?(mod, :list, 1) do
      mod.list(uploader)
    else
      {:error, LangChainError.exception(message: "list/1 not implemented for #{inspect(mod)}")}
    end
  end
end
