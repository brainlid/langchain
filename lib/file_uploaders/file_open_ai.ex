defmodule LangChain.FileUploader.FileOpenAI do
  @moduledoc """
  Uploads files to OpenAI's [Files API](https://platform.openai.com/docs/api-reference/files).

  ## Usage

      {:ok, uploader} = LangChain.FileUploader.FileOpenAI.new(%{api_key: "sk-..."})

      {:ok, result} = LangChain.FileUploader.upload(uploader, file_bytes, %{
        filename: "document.pdf",
        mime_type: "application/pdf"
      })

      result.file_id
      #=> "file-abc123"

  The `purpose` field defaults to `"user_data"` and can be overridden per-call
  via the `file_meta` map or globally via the `default_purpose` schema field.
  """
  use Ecto.Schema
  import Ecto.Changeset
  alias __MODULE__
  alias LangChain.Config
  alias LangChain.LangChainError
  alias LangChain.FileUploader
  alias LangChain.FileUploader.FileResult

  @behaviour FileUploader

  @default_endpoint "https://api.openai.com/v1/files"
  @default_purpose "user_data"
  @receive_timeout 120_000

  @primary_key false
  embedded_schema do
    field :endpoint, :string, default: @default_endpoint
    field :api_key, :string, redact: true
    field :receive_timeout, :integer, default: @receive_timeout
    field :default_purpose, :string, default: @default_purpose
    field :req_opts, :any, virtual: true, default: []
  end

  @type t :: %FileOpenAI{}

  @create_fields [:endpoint, :api_key, :receive_timeout, :default_purpose, :req_opts]
  @required_fields [:endpoint]

  @doc """
  Setup an OpenAI file uploader configuration.
  """
  @spec new(attrs :: map()) :: {:ok, t()} | {:error, Ecto.Changeset.t()}
  def new(%{} = attrs \\ %{}) do
    %FileOpenAI{}
    |> cast(attrs, @create_fields)
    |> common_validation()
    |> apply_action(:insert)
  end

  @doc """
  Setup an OpenAI file uploader configuration and return it or raise an error if invalid.
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
  def upload(%FileOpenAI{} = uploader, file_bytes, %{} = file_meta) when is_binary(file_bytes) do
    filename = Map.fetch!(file_meta, :filename)
    mime_type = Map.fetch!(file_meta, :mime_type)
    purpose = Map.get(file_meta, :purpose, uploader.default_purpose)

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
        {"purpose", purpose},
        {"file", {file_bytes, filename: filename, content_type: mime_type}}
      ]
    )
    |> handle_response()
  end

  @impl FileUploader
  def get(%FileOpenAI{} = uploader, %FileResult{file_id: file_id}) when is_binary(file_id) do
    get(uploader, file_id)
  end

  def get(%FileOpenAI{} = uploader, file_id) when is_binary(file_id) do
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
  def delete(%FileOpenAI{} = uploader, %FileResult{file_id: file_id}) when is_binary(file_id) do
    delete(uploader, file_id)
  end

  def delete(%FileOpenAI{} = uploader, file_id) when is_binary(file_id) do
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
  def list(%FileOpenAI{} = uploader) do
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

  defp get_api_key(%FileOpenAI{api_key: api_key}) do
    api_key || Config.resolve(:openai_key, "")
  end

  defp headers(%FileOpenAI{} = uploader) do
    %{"authorization" => "Bearer #{get_api_key(uploader)}"}
  end

  defp handle_response({:ok, %Req.Response{status: 200, body: body}}) do
    {:ok, parse_file_object(body)}
  end

  defp handle_response({:ok, %Req.Response{status: 401}}) do
    {:error,
     LangChainError.exception(type: "authentication_error", message: "Authentication failed")}
  end

  defp handle_response({:ok, %Req.Response{body: body}}) do
    message = get_in(body, ["error", "message"]) || inspect(body)
    {:error, LangChainError.exception(message: message, original: body)}
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
      size_bytes: body["bytes"],
      provider: :openai,
      raw: body
    })
  end
end
