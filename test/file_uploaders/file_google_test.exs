defmodule LangChain.FileUploader.FileGoogleTest do
  use ExUnit.Case
  use Mimic

  setup :verify_on_exit!

  alias LangChain.FileUploader
  alias LangChain.FileUploader.FileGoogle
  alias LangChain.FileUploader.FileResult
  alias LangChain.LangChainError

  @file_meta %{filename: "doc.pdf", mime_type: "application/pdf"}

  describe "new/1" do
    test "creates with defaults" do
      assert {:ok, %FileGoogle{} = uploader} = FileGoogle.new(%{})
      assert uploader.endpoint == "https://generativelanguage.googleapis.com"
      assert uploader.receive_timeout == 300_000
    end

    test "accepts api_key override" do
      {:ok, uploader} = FileGoogle.new(%{api_key: "AIza-test"})
      assert uploader.api_key == "AIza-test"
    end

    test "returns error when endpoint is nil" do
      {:error, changeset} = FileGoogle.new(%{endpoint: nil})
      assert {"can't be blank", _} = changeset.errors[:endpoint]
    end
  end

  describe "new!/1" do
    test "returns struct when valid" do
      assert %FileGoogle{} = FileGoogle.new!(%{})
    end

    test "raises LangChainError when invalid" do
      assert_raise LangChainError, fn ->
        FileGoogle.new!(%{endpoint: nil})
      end
    end
  end

  describe "upload/3" do
    test "performs two-step upload and returns FileResult with file_uri" do
      # Step 1: request_upload_url
      expect(Req, :post, fn req ->
        # Verify this is the initiate request
        assert String.contains?(URI.to_string(req.url), "/upload/v1beta/files")

        {:ok,
         %Req.Response{
           status: 200,
           headers: %{"x-goog-upload-url" => ["https://upload.example.com/upload?id=123"]},
           body: %{}
         }}
      end)

      # Step 2: upload_file_bytes
      expect(Req, :post, fn req ->
        assert URI.to_string(req.url) == "https://upload.example.com/upload?id=123"

        {:ok,
         %Req.Response{
           status: 200,
           body: %{
             "file" => %{
               "name" => "files/abc123",
               "uri" => "https://generativelanguage.googleapis.com/v1beta/files/abc123",
               "displayName" => "doc.pdf",
               "mimeType" => "application/pdf"
             }
           }
         }}
      end)

      uploader = FileGoogle.new!(%{api_key: "AIza-test"})

      assert {:ok, %FileResult{} = result} =
               FileUploader.upload(uploader, "file content", @file_meta)

      assert result.file_id == "files/abc123"
      assert result.file_uri == "https://generativelanguage.googleapis.com/v1beta/files/abc123"
      assert result.filename == "doc.pdf"
      assert result.mime_type == "application/pdf"
      assert result.provider == :google
    end

    test "uses display_name from file_meta when provided" do
      expect(Req, :post, fn _req ->
        {:ok,
         %Req.Response{
           status: 200,
           headers: %{"x-goog-upload-url" => ["https://upload.example.com/upload?id=456"]},
           body: %{}
         }}
      end)

      expect(Req, :post, fn _req ->
        {:ok,
         %Req.Response{
           status: 200,
           body: %{
             "file" => %{
               "name" => "files/abc",
               "uri" => "https://example.com/files/abc",
               "displayName" => "My Custom Name",
               "mimeType" => "application/pdf"
             }
           }
         }}
      end)

      uploader = FileGoogle.new!(%{api_key: "AIza-test"})
      meta = Map.put(@file_meta, :display_name, "My Custom Name")
      assert {:ok, _result} = FileUploader.upload(uploader, "content", meta)
    end

    test "returns error when initiate request fails" do
      expect(Req, :post, fn _req ->
        {:ok,
         %Req.Response{
           status: 400,
           body: %{"error" => %{"message" => "Invalid request"}}
         }}
      end)

      uploader = FileGoogle.new!(%{api_key: "AIza-test"})

      assert {:error, %LangChainError{}} =
               FileUploader.upload(uploader, "content", @file_meta)
    end

    test "returns error when upload URL header is missing" do
      expect(Req, :post, fn _req ->
        {:ok,
         %Req.Response{
           status: 200,
           headers: %{},
           body: %{}
         }}
      end)

      uploader = FileGoogle.new!(%{api_key: "AIza-test"})

      assert {:error, %LangChainError{message: "Missing x-goog-upload-url" <> _}} =
               FileUploader.upload(uploader, "content", @file_meta)
    end

    test "returns error when finalize fails" do
      expect(Req, :post, fn _req ->
        {:ok,
         %Req.Response{
           status: 200,
           headers: %{"x-goog-upload-url" => ["https://upload.example.com/upload?id=789"]},
           body: %{}
         }}
      end)

      expect(Req, :post, fn _req ->
        {:ok, %Req.Response{status: 500, body: %{"error" => "Internal error"}}}
      end)

      uploader = FileGoogle.new!(%{api_key: "AIza-test"})

      assert {:error, %LangChainError{}} =
               FileUploader.upload(uploader, "content", @file_meta)
    end

    test "returns timeout error on transport timeout during finalize" do
      expect(Req, :post, fn _req ->
        {:ok,
         %Req.Response{
           status: 200,
           headers: %{"x-goog-upload-url" => ["https://upload.example.com/upload?id=t"]},
           body: %{}
         }}
      end)

      expect(Req, :post, fn _req ->
        {:error, %Req.TransportError{reason: :timeout}}
      end)

      uploader = FileGoogle.new!(%{api_key: "AIza-test"})

      assert {:error, %LangChainError{type: "timeout"}} =
               FileUploader.upload(uploader, "content", @file_meta)
    end
  end

  describe "get/2" do
    test "returns FileResult on success with resource name" do
      expect(Req, :get, fn req ->
        assert String.contains?(URI.to_string(req.url), "/v1beta/files/abc123")

        {:ok,
         %Req.Response{
           status: 200,
           body: %{
             "name" => "files/abc123",
             "uri" => "https://generativelanguage.googleapis.com/v1beta/files/abc123",
             "displayName" => "doc.pdf",
             "mimeType" => "application/pdf"
           }
         }}
      end)

      uploader = FileGoogle.new!(%{api_key: "AIza-test"})

      assert {:ok, %FileResult{} = result} = FileUploader.get(uploader, "files/abc123")
      assert result.file_id == "files/abc123"
      assert result.filename == "doc.pdf"
      assert result.mime_type == "application/pdf"
    end

    test "accepts FileResult and uses file_id" do
      expect(Req, :get, fn _req ->
        {:ok,
         %Req.Response{
           status: 200,
           body: %{
             "name" => "files/abc123",
             "uri" => "https://generativelanguage.googleapis.com/v1beta/files/abc123",
             "displayName" => "doc.pdf",
             "mimeType" => "application/pdf"
           }
         }}
      end)

      uploader = FileGoogle.new!(%{api_key: "AIza-test"})

      ref =
        FileResult.new!(%{
          file_id: "files/abc123",
          provider: :google
        })

      assert {:ok, %FileResult{file_id: "files/abc123"}} = FileUploader.get(uploader, ref)
    end
  end

  describe "delete/2" do
    test "returns :ok on success with resource name" do
      expect(Req, :delete, fn req ->
        assert String.contains?(URI.to_string(req.url), "/v1beta/files/abc123")
        {:ok, %Req.Response{status: 200, body: %{}}}
      end)

      uploader = FileGoogle.new!(%{api_key: "AIza-test"})
      assert :ok = FileUploader.delete(uploader, "files/abc123")
    end

    test "accepts FileResult and uses file_id" do
      expect(Req, :delete, fn req ->
        assert String.contains?(URI.to_string(req.url), "/v1beta/files/abc123")
        {:ok, %Req.Response{status: 204, body: ""}}
      end)

      uploader = FileGoogle.new!(%{api_key: "AIza-test"})

      result =
        FileResult.new!(%{
          file_id: "files/abc123",
          file_uri: "https://generativelanguage.googleapis.com/v1beta/files/abc123",
          provider: :google,
          raw: %{"name" => "files/abc123"}
        })

      assert :ok = FileUploader.delete(uploader, result)
    end
  end

  describe "list/1" do
    test "returns list of FileResults with file_uri" do
      expect(Req, :get, fn _req ->
        {:ok,
         %Req.Response{
           status: 200,
           body: %{
             "files" => [
               %{
                 "name" => "files/abc",
                 "uri" => "https://example.com/files/abc",
                 "displayName" => "a.pdf",
                 "mimeType" => "application/pdf"
               },
               %{
                 "name" => "files/def",
                 "uri" => "https://example.com/files/def",
                 "displayName" => "b.pdf",
                 "mimeType" => "text/plain"
               }
             ]
           }
         }}
      end)

      uploader = FileGoogle.new!(%{api_key: "AIza-test"})

      assert {:ok,
              [
                %FileResult{file_id: "files/abc", file_uri: "https://example.com/files/abc"},
                %FileResult{file_id: "files/def", file_uri: "https://example.com/files/def"}
              ]} =
               FileUploader.list(uploader)
    end

    test "returns empty list when no files" do
      expect(Req, :get, fn _req ->
        {:ok, %Req.Response{status: 200, body: %{}}}
      end)

      uploader = FileGoogle.new!(%{api_key: "AIza-test"})
      assert {:ok, []} = FileUploader.list(uploader)
    end
  end
end
