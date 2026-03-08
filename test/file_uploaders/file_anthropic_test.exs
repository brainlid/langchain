defmodule LangChain.FileUploader.FileAnthropicTest do
  use ExUnit.Case
  use Mimic

  setup :verify_on_exit!

  alias LangChain.FileUploader
  alias LangChain.FileUploader.FileAnthropic
  alias LangChain.FileUploader.FileResult
  alias LangChain.LangChainError

  @file_meta %{filename: "doc.pdf", mime_type: "application/pdf"}

  describe "new/1" do
    test "creates with defaults" do
      assert {:ok, %FileAnthropic{} = uploader} = FileAnthropic.new(%{})
      assert uploader.endpoint == "https://api.anthropic.com/v1/files"
      assert uploader.api_version == "2023-06-01"
      assert uploader.receive_timeout == 120_000
    end

    test "accepts api_key override" do
      {:ok, uploader} = FileAnthropic.new(%{api_key: "sk-ant-test"})
      assert uploader.api_key == "sk-ant-test"
    end

    test "returns error when endpoint is nil" do
      {:error, changeset} = FileAnthropic.new(%{endpoint: nil})
      assert {"can't be blank", _} = changeset.errors[:endpoint]
    end
  end

  describe "new!/1" do
    test "returns struct when valid" do
      assert %FileAnthropic{} = FileAnthropic.new!(%{})
    end

    test "raises LangChainError when invalid" do
      assert_raise LangChainError, fn ->
        FileAnthropic.new!(%{endpoint: nil})
      end
    end
  end

  describe "upload/3" do
    test "returns FileResult with file_id on success" do
      expect(Req, :post, fn _req, _opts ->
        {:ok,
         %Req.Response{
           status: 200,
           body: %{
             "id" => "file-ant-abc123",
             "type" => "file",
             "filename" => "doc.pdf",
             "mime_type" => "application/pdf",
             "size_bytes" => 2048
           }
         }}
      end)

      uploader = FileAnthropic.new!(%{api_key: "sk-ant-test"})

      assert {:ok, %FileResult{} = result} =
               FileUploader.upload(uploader, "file content", @file_meta)

      assert result.file_id == "file-ant-abc123"
      assert result.filename == "doc.pdf"
      assert result.mime_type == "application/pdf"
      assert result.size_bytes == 2048
      assert result.provider == :anthropic
    end

    test "includes anthropic-beta header in request" do
      expect(Req, :post, fn req, _opts ->
        headers = req.headers
        assert Map.get(headers, "anthropic-beta") == ["files-api-2025-04-14"]

        {:ok,
         %Req.Response{
           status: 200,
           body: %{
             "id" => "file-ant-abc",
             "filename" => "doc.pdf",
             "mime_type" => "application/pdf",
             "size_bytes" => 100
           }
         }}
      end)

      uploader = FileAnthropic.new!(%{api_key: "sk-ant-test"})
      assert {:ok, _result} = FileUploader.upload(uploader, "content", @file_meta)
    end

    test "returns authentication error on 401" do
      expect(Req, :post, fn _req, _opts ->
        {:ok,
         %Req.Response{
           status: 401,
           body: %{"error" => %{"type" => "authentication_error", "message" => "invalid api key"}}
         }}
      end)

      uploader = FileAnthropic.new!(%{api_key: "bad-key"})

      assert {:error, %LangChainError{type: "authentication_error"}} =
               FileUploader.upload(uploader, "content", @file_meta)
    end

    test "returns error with API error details on failure" do
      expect(Req, :post, fn _req, _opts ->
        {:ok,
         %Req.Response{
           status: 400,
           body: %{
             "error" => %{
               "type" => "invalid_request_error",
               "message" => "File too large"
             }
           }
         }}
      end)

      uploader = FileAnthropic.new!(%{api_key: "sk-ant-test"})

      assert {:error, %LangChainError{type: "invalid_request_error", message: "File too large"}} =
               FileUploader.upload(uploader, "content", @file_meta)
    end

    test "returns timeout error on transport timeout" do
      expect(Req, :post, fn _req, _opts ->
        {:error, %Req.TransportError{reason: :timeout}}
      end)

      uploader = FileAnthropic.new!(%{api_key: "sk-ant-test"})

      assert {:error, %LangChainError{type: "timeout"}} =
               FileUploader.upload(uploader, "content", @file_meta)
    end
  end

  describe "get/2" do
    test "returns FileResult on success" do
      expect(Req, :get, fn _req ->
        {:ok,
         %Req.Response{
           status: 200,
           body: %{
             "id" => "file-ant-abc",
             "filename" => "doc.pdf",
             "mime_type" => "application/pdf",
             "size_bytes" => 2048
           }
         }}
      end)

      uploader = FileAnthropic.new!(%{api_key: "sk-ant-test"})

      assert {:ok, %FileResult{} = result} = FileUploader.get(uploader, "file-ant-abc")
      assert result.file_id == "file-ant-abc"
      assert result.filename == "doc.pdf"
      assert result.mime_type == "application/pdf"
      assert result.size_bytes == 2048
    end

    test "accepts FileResult" do
      expect(Req, :get, fn _req ->
        {:ok,
         %Req.Response{
           status: 200,
           body: %{
             "id" => "file-ant-abc",
             "filename" => "doc.pdf",
             "mime_type" => "application/pdf",
             "size_bytes" => 2048
           }
         }}
      end)

      uploader = FileAnthropic.new!(%{api_key: "sk-ant-test"})
      ref = FileResult.new!(%{file_id: "file-ant-abc", provider: :anthropic})
      assert {:ok, %FileResult{file_id: "file-ant-abc"}} = FileUploader.get(uploader, ref)
    end
  end

  describe "delete/2" do
    test "returns :ok on success" do
      expect(Req, :delete, fn _req ->
        {:ok,
         %Req.Response{status: 200, body: %{"id" => "file-ant-abc", "type" => "file_deleted"}}}
      end)

      uploader = FileAnthropic.new!(%{api_key: "sk-ant-test"})
      assert :ok = FileUploader.delete(uploader, "file-ant-abc")
    end

    test "accepts FileResult" do
      expect(Req, :delete, fn _req ->
        {:ok, %Req.Response{status: 200, body: %{}}}
      end)

      uploader = FileAnthropic.new!(%{api_key: "sk-ant-test"})
      result = FileResult.new!(%{file_id: "file-ant-abc", provider: :anthropic})
      assert :ok = FileUploader.delete(uploader, result)
    end
  end

  describe "list/1" do
    test "returns list of FileResults on success" do
      expect(Req, :get, fn _req ->
        {:ok,
         %Req.Response{
           status: 200,
           body: %{
             "data" => [
               %{
                 "id" => "file-1",
                 "filename" => "a.pdf",
                 "mime_type" => "application/pdf",
                 "size_bytes" => 100
               },
               %{
                 "id" => "file-2",
                 "filename" => "b.pdf",
                 "mime_type" => "text/plain",
                 "size_bytes" => 200
               }
             ]
           }
         }}
      end)

      uploader = FileAnthropic.new!(%{api_key: "sk-ant-test"})

      assert {:ok, [%FileResult{file_id: "file-1"}, %FileResult{file_id: "file-2"}]} =
               FileUploader.list(uploader)
    end
  end
end
