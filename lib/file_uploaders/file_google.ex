defmodule LangChain.FileUploader.FileGoogle do
  @moduledoc """
  Uploads files to Google Gemini's [File API](https://ai.google.dev/gemini-api/docs/files).

  Google uses a two-step resumable upload protocol internally, but this is
  abstracted away — callers simply call `upload/3`.

  ## Usage

      {:ok, uploader} = LangChain.FileUploader.FileGoogle.new(%{api_key: "AI..."})

      {:ok, result} = LangChain.FileUploader.upload(uploader, file_bytes, %{
        filename: "document.pdf",
        mime_type: "application/pdf"
      })

      result.file_id
      #=> "files/abc-123"

      result.file_uri
      #=> "https://generativelanguage.googleapis.com/v1beta/files/abc-123"

  Google identifies files by resource name (e.g. `"files/abc-123"`), which is
  stored as `file_id` in the returned `FileResult`. The `file_uri` is also
  available for use with `ContentPart.file_url!/2`.

  Note: Files uploaded to Gemini expire after 48 hours.
  """
  use Ecto.Schema
  import Ecto.Changeset
  alias __MODULE__
  alias LangChain.Config
  alias LangChain.LangChainError
  alias LangChain.FileUploader
  alias LangChain.FileUploader.FileResult

  @behaviour FileUploader

  @default_endpoint "https://generativelanguage.googleapis.com"
  @upload_path "/upload/v1beta/files"
  @files_path "/v1beta/files"
  @receive_timeout 300_000

  @primary_key false
  embedded_schema do
    field :endpoint, :string, default: @default_endpoint
    field :api_key, :string, redact: true
    field :receive_timeout, :integer, default: @receive_timeout
    field :req_opts, :any, virtual: true, default: []
  end

  @type t :: %FileGoogle{}

  @create_fields [:endpoint, :api_key, :receive_timeout, :req_opts]
  @required_fields [:endpoint]

  @doc """
  Setup a Google file uploader configuration.
  """
  @spec new(attrs :: map()) :: {:ok, t()} | {:error, Ecto.Changeset.t()}
  def new(%{} = attrs \\ %{}) do
    %FileGoogle{}
    |> cast(attrs, @create_fields)
    |> common_validation()
    |> apply_action(:insert)
  end

  @doc """
  Setup a Google file uploader configuration and return it or raise an error if invalid.
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
  def upload(%FileGoogle{} = uploader, file_bytes, %{} = file_meta) when is_binary(file_bytes) do
    filename = Map.fetch!(file_meta, :filename)
    mime_type = Map.fetch!(file_meta, :mime_type)
    display_name = Map.get(file_meta, :display_name, filename)
    byte_count = byte_size(file_bytes)

    with {:ok, upload_url} <- request_upload_url(uploader, display_name, mime_type, byte_count),
         {:ok, result} <- upload_file_bytes(uploader, upload_url, file_bytes, byte_count) do
      {:ok, result}
    end
  end

  @impl FileUploader
  def get(%FileGoogle{} = uploader, %FileResult{file_id: file_id})
      when is_binary(file_id) do
    get(uploader, file_id)
  end

  @doc """
  Retrieve file metadata by its resource name.

  `file_name` must be the Google resource name in the form `"files/{id}"`,
  e.g. `"files/abc-123"`. This is the `file_id` value stored in `FileResult`.
  """
  def get(%FileGoogle{} = uploader, file_name) when is_binary(file_name) do
    api_key = get_api_key(uploader)
    url = "#{uploader.endpoint}/v1beta/#{file_name}?key=#{api_key}"

    Req.new(url: url, receive_timeout: uploader.receive_timeout)
    |> Req.merge(uploader.req_opts)
    |> Req.get()
    |> case do
      {:ok, %Req.Response{status: 200, body: body}} ->
        {:ok, parse_file_object(body)}

      {:ok, %Req.Response{body: body}} ->
        {:error, LangChainError.exception(message: "Get failed", original: body)}

      {:error, err} ->
        {:error,
         LangChainError.exception(message: "Request error: #{inspect(err)}", original: err)}
    end
  end

  @impl FileUploader
  def delete(%FileGoogle{} = uploader, %FileResult{file_id: file_id})
      when is_binary(file_id) do
    delete(uploader, file_id)
  end

  @doc """
  Delete a file by its resource name.

  `file_name` must be the Google resource name in the form `"files/{id}"`,
  e.g. `"files/abc-123"`. This is the `file_id` value stored in `FileResult`.
  """
  def delete(%FileGoogle{} = uploader, file_name) when is_binary(file_name) do
    api_key = get_api_key(uploader)
    url = "#{uploader.endpoint}/v1beta/#{file_name}?key=#{api_key}"

    Req.new(url: url, receive_timeout: uploader.receive_timeout)
    |> Req.merge(uploader.req_opts)
    |> Req.delete()
    |> case do
      {:ok, %Req.Response{status: status}} when status in [200, 204] ->
        :ok

      {:ok, %Req.Response{body: body}} ->
        {:error, LangChainError.exception(message: "Delete failed", original: body)}

      {:error, err} ->
        {:error,
         LangChainError.exception(message: "Request error: #{inspect(err)}", original: err)}
    end
  end

  @impl FileUploader
  def list(%FileGoogle{} = uploader) do
    api_key = get_api_key(uploader)
    url = "#{uploader.endpoint}#{@files_path}?key=#{api_key}"

    Req.new(url: url, receive_timeout: uploader.receive_timeout)
    |> Req.merge(uploader.req_opts)
    |> Req.get()
    |> case do
      {:ok, %Req.Response{status: 200, body: %{"files" => files}}} when is_list(files) ->
        {:ok, Enum.map(files, &parse_file_object/1)}

      {:ok, %Req.Response{status: 200}} ->
        {:ok, []}

      {:ok, %Req.Response{body: body}} ->
        {:error, LangChainError.exception(message: "List failed", original: body)}

      {:error, err} ->
        {:error,
         LangChainError.exception(message: "Request error: #{inspect(err)}", original: err)}
    end
  end

  @doc """
  Request a presigned upload URL from Google's resumable upload endpoint (step 1 of 2).

  Returns `{:ok, upload_url}` on success, or `{:error, LangChainError.t()}` on failure.
  The returned URL can be passed to `upload_file_bytes/4` to complete the upload, or
  forwarded directly to a client (e.g. a browser or mobile app) so it can upload
  the file bytes itself without routing them through your server.
  """
  @spec request_upload_url(t(), String.t(), String.t(), non_neg_integer()) ::
          {:ok, String.t()} | {:error, LangChain.LangChainError.t()}
  def request_upload_url(%FileGoogle{} = uploader, display_name, mime_type, byte_count) do
    api_key = get_api_key(uploader)
    url = "#{uploader.endpoint}#{@upload_path}?key=#{api_key}"

    Req.new(
      url: url,
      json: %{"file" => %{"display_name" => display_name}},
      headers: %{
        "x-goog-upload-protocol" => "resumable",
        "x-goog-upload-command" => "start",
        "x-goog-upload-header-content-length" => to_string(byte_count),
        "x-goog-upload-header-content-type" => mime_type
      },
      receive_timeout: uploader.receive_timeout
    )
    |> Req.merge(uploader.req_opts)
    |> Req.post()
    |> case do
      {:ok, %Req.Response{status: 200, headers: resp_headers}} ->
        case Map.get(resp_headers, "x-goog-upload-url") do
          [upload_url | _] ->
            {:ok, upload_url}

          _ ->
            {:error,
             LangChainError.exception(
               message: "Missing x-goog-upload-url header in initiate response"
             )}
        end

      {:ok, %Req.Response{body: body}} ->
        {:error,
         LangChainError.exception(message: "Failed to request upload URL", original: body)}

      {:error, err} ->
        {:error,
         LangChainError.exception(message: "Request error: #{inspect(err)}", original: err)}
    end
  end

  @doc """
  Upload raw file bytes to a presigned upload URL (step 2 of 2).

  `upload_url` is obtained from `request_upload_url/4`. Returns `{:ok, FileResult.t()}` on
  success, or `{:error, LangChainError.t()}` on failure.
  """
  @spec upload_file_bytes(t(), String.t(), binary(), non_neg_integer()) ::
          {:ok, FileResult.t()} | {:error, LangChain.LangChainError.t()}
  def upload_file_bytes(%FileGoogle{} = uploader, upload_url, file_bytes, byte_count) do
    Req.new(
      url: upload_url,
      body: file_bytes,
      headers: %{
        "content-length" => to_string(byte_count),
        "x-goog-upload-offset" => "0",
        "x-goog-upload-command" => "upload, finalize"
      },
      receive_timeout: uploader.receive_timeout
    )
    |> Req.merge(uploader.req_opts)
    |> Req.post()
    |> case do
      {:ok, %Req.Response{status: 200, body: %{"file" => file_data}}} ->
        {:ok, parse_file_object(file_data)}

      {:ok, %Req.Response{body: body}} ->
        {:error, LangChainError.exception(message: "Failed to upload file bytes", original: body)}

      {:error, %Req.TransportError{reason: :timeout} = err} ->
        {:error,
         LangChainError.exception(type: "timeout", message: "Request timed out", original: err)}

      {:error, err} ->
        {:error,
         LangChainError.exception(message: "Request error: #{inspect(err)}", original: err)}
    end
  end

  defp get_api_key(%FileGoogle{api_key: api_key}) do
    api_key || Config.resolve(:google_ai_key, "")
  end

  defp parse_file_object(body) do
    FileResult.new!(%{
      file_id: body["name"],
      file_uri: body["uri"],
      filename: body["displayName"] || body["name"],
      mime_type: body["mimeType"],
      provider: :google,
      raw: body
    })
  end
end
