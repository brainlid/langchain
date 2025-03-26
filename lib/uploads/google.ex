defmodule LangChain.Uploads.Google do
  @base_url "https://generativelanguage.googleapis.com"

  alias LangChain.Message.ContentPart

  def upload_file(google_config, file_path) do
    # Get file metadata
    api_key = google_config.api_key
    base_url = google_config.endpoint

    {:ok, mime_type} = get_mime_type(file_path)
    num_bytes = File.stat!(file_path).size
    display_name = "TEXT"

    # Initial resumable upload request
    {:ok, upload_url} =
      initialize_upload(
        base_url,
        api_key,
        num_bytes,
        mime_type,
        display_name
      )

    # Upload the actual file
    do_upload_file(upload_url, file_path, num_bytes)
  end

  defp get_mime_type(path) do
    result = System.cmd("file", ["-b", "--mime-type", path])

    case result do
      {mime, 0} -> {:ok, String.trim(mime)}
      _ -> {:error, "Could not determine MIME type"}
    end
  end

  defp initialize_upload(base_url, api_key, num_bytes, mime_type, display_name) do
    url = "#{base_url}/upload/v1beta/files?key=#{api_key}"

    headers = [
      {"X-Goog-Upload-Protocol", "resumable"},
      {"X-Goog-Upload-Command", "start"},
      {"X-Goog-Upload-Header-Content-Length", to_string(num_bytes)},
      {"X-Goog-Upload-Header-Content-Type", mime_type}
    ]

    response =
      Req.post!(url,
        headers: headers,
        json: %{file: %{display_name: display_name}}
      )

    upload_url =
      Enum.find_value(response.headers, fn
        {"x-goog-upload-url", [url]} -> url
        _ -> nil
      end)

    {:ok, upload_url}
  end

  defp do_upload_file(upload_url, img_path, num_bytes) do
    headers = [
      {"Content-Length", to_string(num_bytes)},
      {"X-Goog-Upload-Offset", "0"},
      {"X-Goog-Upload-Command", "upload, finalize"}
    ]

    {:ok, file_binary} = File.read(img_path)

    response =
      Req.post!(upload_url,
        headers: headers,
        body: file_binary
      )

    {:ok,
     %ContentPart{
       type: :file_url,
       content: response.body["file"]["uri"],
       options: %{media: response.body["file"]["mimeType"]}
     }}
  end
end
