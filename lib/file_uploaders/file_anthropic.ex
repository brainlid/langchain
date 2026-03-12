defmodule LangChain.FileUploader.FileAnthropic do
  @moduledoc """
  Uploads files to Anthropic's [Files API](https://docs.anthropic.com/en/api/files-create).

  Requires the `files-api-2025-04-14` beta header, which is automatically included.

  ## Usage

      {:ok, uploader} = LangChain.FileUploader.FileAnthropic.new(%{api_key: "sk-ant-..."})

      {:ok, result} = LangChain.FileUploader.upload(uploader, file_bytes, %{
        filename: "document.pdf",
        mime_type: "application/pdf"
      })

      result.file_id
      #=> "file-abc123"

  The returned `file_id` can be used with `ContentPart.file!/2` to reference
  the uploaded file in messages:

      ContentPart.file!(result.file_id, type: :file_id)
  """
  use Ecto.Schema
  import Ecto.Changeset
  alias __MODULE__
  alias LangChain.Config
  alias LangChain.LangChainError
  alias LangChain.FileUploader
  alias LangChain.FileUploader.FileResult

  @behaviour FileUploader

  @default_endpoint "https://api.anthropic.com/v1/files"
  @default_api_version "2023-06-01"
  @files_api_beta "files-api-2025-04-14"
  @receive_timeout 120_000

  @primary_key false
  embedded_schema do
    field :endpoint, :string, default: @default_endpoint
    field :api_key, :string, redact: true
    field :api_version, :string, default: @default_api_version
    field :receive_timeout, :integer, default: @receive_timeout
    field :req_opts, :any, virtual: true, default: []
  end

  @type t :: %FileAnthropic{}

  @create_fields [:endpoint, :api_key, :api_version, :receive_timeout, :req_opts]
  @required_fields [:endpoint, :api_version]

  @doc """
  Setup an Anthropic file uploader configuration.
  """
  @spec new(attrs :: map()) :: {:ok, t()} | {:error, Ecto.Changeset.t()}
  def new(%{} = attrs \\ %{}) do
    %FileAnthropic{}
    |> cast(attrs, @create_fields)
    |> common_validation()
    |> apply_action(:insert)
  end

  @doc """
  Setup an Anthropic file uploader configuration and return it or raise an error if invalid.
  """
  @spec new!(attrs :: map()) :: t() | no_return()
  def new!(attrs \\ %{}) do
    case new(attrs) do
      {:ok, uploader} -> uploader
      {:error, changeset} -> raise LangChainError, changeset
    end
  end

  defp common_validation(changeset) do
    changeset
    |> validate_required(@required_fields)
    |> validate_number(:receive_timeout, greater_than_or_equal_to: 0)
  end

  @impl FileUploader
  def upload(%FileAnthropic{} = uploader, file_bytes, %{} = file_meta)
      when is_binary(file_bytes) do
    filename = Map.fetch!(file_meta, :filename)
    mime_type = Map.fetch!(file_meta, :mime_type)

    Req.new(
      url: uploader.endpoint,
      headers: headers(uploader),
      receive_timeout: uploader.receive_timeout,
      retry: :transient,
      max_retries: 3,
      retry_delay: fn attempt -> 300 * attempt end
    )
    |> Req.merge(uploader.req_opts)
    |> Req.post(
      form_multipart: [
        {"file", {file_bytes, filename: filename, content_type: mime_type}}
      ]
    )
    |> handle_response()
  end

  @impl FileUploader
  def get(%FileAnthropic{} = uploader, %FileResult{file_id: file_id})
      when is_binary(file_id) do
    get(uploader, file_id)
  end

  def get(%FileAnthropic{} = uploader, file_id) when is_binary(file_id) do
    base_url = String.trim_trailing(uploader.endpoint, "/")

    Req.new(
      url: "#{base_url}/#{file_id}",
      headers: headers(uploader),
      receive_timeout: uploader.receive_timeout
    )
    |> Req.merge(uploader.req_opts)
    |> Req.get()
    |> handle_response()
  end

  @impl FileUploader
  def delete(%FileAnthropic{} = uploader, %FileResult{file_id: file_id})
      when is_binary(file_id) do
    delete(uploader, file_id)
  end

  def delete(%FileAnthropic{} = uploader, file_id) when is_binary(file_id) do
    base_url = String.trim_trailing(uploader.endpoint, "/")

    Req.new(
      url: "#{base_url}/#{file_id}",
      headers: headers(uploader),
      receive_timeout: uploader.receive_timeout
    )
    |> Req.merge(uploader.req_opts)
    |> Req.delete()
    |> case do
      {:ok, %Req.Response{status: 200}} ->
        :ok

      {:ok, %Req.Response{body: body}} ->
        {:error, LangChainError.exception(message: "Delete failed", original: body)}

      {:error, err} ->
        {:error,
         LangChainError.exception(message: "Request error: #{inspect(err)}", original: err)}
    end
  end

  @impl FileUploader
  def list(%FileAnthropic{} = uploader) do
    Req.new(
      url: uploader.endpoint,
      headers: headers(uploader),
      receive_timeout: uploader.receive_timeout
    )
    |> Req.merge(uploader.req_opts)
    |> Req.get()
    |> case do
      {:ok, %Req.Response{status: 200, body: %{"data" => files}}} ->
        {:ok, Enum.map(files, &parse_file_object/1)}

      {:ok, %Req.Response{body: body}} ->
        {:error, LangChainError.exception(message: "List failed", original: body)}

      {:error, err} ->
        {:error,
         LangChainError.exception(message: "Request error: #{inspect(err)}", original: err)}
    end
  end

  defp get_api_key(%FileAnthropic{api_key: api_key}) do
    api_key || Config.resolve(:anthropic_key, "")
  end

  defp headers(%FileAnthropic{} = uploader) do
    %{
      "x-api-key" => get_api_key(uploader),
      "anthropic-version" => uploader.api_version,
      "anthropic-beta" => @files_api_beta
    }
  end

  defp handle_response({:ok, %Req.Response{status: 200, body: body}}) do
    {:ok, parse_file_object(body)}
  end

  defp handle_response({:ok, %Req.Response{status: 401}}) do
    {:error,
     LangChainError.exception(type: "authentication_error", message: "Authentication failed")}
  end

  defp handle_response(
         {:ok, %Req.Response{body: %{"error" => %{"type" => type, "message" => msg}} = body}}
       ) do
    {:error, LangChainError.exception(type: type, message: msg, original: body)}
  end

  defp handle_response({:ok, %Req.Response{body: body}}) do
    {:error,
     LangChainError.exception(message: "Unexpected response: #{inspect(body)}", original: body)}
  end

  defp handle_response({:error, %Req.TransportError{reason: :timeout} = err}) do
    {:error,
     LangChainError.exception(type: "timeout", message: "Request timed out", original: err)}
  end

  defp handle_response({:error, err}) do
    {:error, LangChainError.exception(message: "Request error: #{inspect(err)}", original: err)}
  end

  defp parse_file_object(body) do
    FileResult.new!(%{
      file_id: body["id"],
      filename: body["filename"],
      mime_type: body["mime_type"],
      size_bytes: body["size_bytes"],
      provider: :anthropic,
      raw: body
    })
  end
end
