defmodule LangChain.FileUploader.FileOpenAITest do
  use ExUnit.Case
  use Mimic

  setup :verify_on_exit!

  alias LangChain.FileUploader
  alias LangChain.FileUploader.FileOpenAI
  alias LangChain.FileUploader.FileResult
  alias LangChain.LangChainError

  @file_meta %{filename: "doc.pdf", mime_type: "application/pdf"}

  describe "new/1" do
    test "creates with defaults" do
      assert {:ok, %FileOpenAI{} = uploader} = FileOpenAI.new(%{})
      assert uploader.endpoint == "https://api.openai.com/v1/files"
      assert uploader.default_purpose == "user_data"
      assert uploader.receive_timeout == 120_000
    end

    test "accepts api_key override" do
      {:ok, uploader} = FileOpenAI.new(%{api_key: "sk-test"})
      assert uploader.api_key == "sk-test"
    end

    test "returns error when endpoint is nil" do
      {:error, changeset} = FileOpenAI.new(%{endpoint: nil})
      assert {"can't be blank", _} = changeset.errors[:endpoint]
    end
  end

  describe "new!/1" do
    test "returns struct when valid" do
      assert %FileOpenAI{} = FileOpenAI.new!(%{})
    end

    test "raises LangChainError when invalid" do
      assert_raise LangChainError, fn ->
        FileOpenAI.new!(%{endpoint: nil})
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
             "id" => "file-abc123",
             "object" => "file",
             "bytes" => 1024,
             "filename" => "doc.pdf",
             "purpose" => "user_data"
           }
         }}
      end)

      uploader = FileOpenAI.new!(%{api_key: "sk-test"})

      assert {:ok, %FileResult{} = result} =
               FileUploader.upload(uploader, "file content", @file_meta)

      assert result.file_id == "file-abc123"
      assert result.filename == "doc.pdf"
      assert result.size_bytes == 1024
      assert result.provider == :openai
    end

    test "passes custom purpose from file_meta" do
      expect(Req, :post, fn _req, opts ->
        multipart = Keyword.get(opts, :form_multipart)
        assert {"purpose", "assistants"} in multipart

        {:ok,
         %Req.Response{
           status: 200,
           body: %{"id" => "file-abc", "filename" => "doc.pdf", "bytes" => 100}
         }}
      end)

      uploader = FileOpenAI.new!(%{api_key: "sk-test"})
      meta = Map.put(@file_meta, :purpose, "assistants")
      assert {:ok, _result} = FileUploader.upload(uploader, "content", meta)
    end

    test "returns authentication error on 401" do
      expect(Req, :post, fn _req, _opts ->
        {:ok, %Req.Response{status: 401, body: %{"error" => %{"message" => "Unauthorized"}}}}
      end)

      uploader = FileOpenAI.new!(%{api_key: "bad-key"})

      assert {:error, %LangChainError{type: "authentication_error"}} =
               FileUploader.upload(uploader, "content", @file_meta)
    end

    test "returns error with API message on failure" do
      expect(Req, :post, fn _req, _opts ->
        {:ok,
         %Req.Response{
           status: 400,
           body: %{"error" => %{"message" => "Invalid file format"}}
         }}
      end)

      uploader = FileOpenAI.new!(%{api_key: "sk-test"})

      assert {:error, %LangChainError{message: "Invalid file format"}} =
               FileUploader.upload(uploader, "content", @file_meta)
    end

    test "returns timeout error on transport timeout" do
      expect(Req, :post, fn _req, _opts ->
        {:error, %Req.TransportError{reason: :timeout}}
      end)

      uploader = FileOpenAI.new!(%{api_key: "sk-test"})

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
           body: %{"id" => "file-abc123", "filename" => "doc.pdf", "bytes" => 1024}
         }}
      end)

      uploader = FileOpenAI.new!(%{api_key: "sk-test"})

      assert {:ok, %FileResult{} = result} = FileUploader.get(uploader, "file-abc123")
      assert result.file_id == "file-abc123"
      assert result.filename == "doc.pdf"
      assert result.size_bytes == 1024
    end

    test "accepts FileResult" do
      expect(Req, :get, fn _req ->
        {:ok,
         %Req.Response{
           status: 200,
           body: %{"id" => "file-abc123", "filename" => "doc.pdf", "bytes" => 1024}
         }}
      end)

      uploader = FileOpenAI.new!(%{api_key: "sk-test"})
      ref = FileResult.new!(%{file_id: "file-abc123", provider: :openai})
      assert {:ok, %FileResult{file_id: "file-abc123"}} = FileUploader.get(uploader, ref)
    end
  end

  describe "delete/2" do
    test "returns :ok on success" do
      expect(Req, :delete, fn _req ->
        {:ok, %Req.Response{status: 200, body: %{"deleted" => true}}}
      end)

      uploader = FileOpenAI.new!(%{api_key: "sk-test"})
      assert :ok = FileUploader.delete(uploader, "file-abc123")
    end

    test "accepts FileResult" do
      expect(Req, :delete, fn _req ->
        {:ok, %Req.Response{status: 200, body: %{"deleted" => true}}}
      end)

      uploader = FileOpenAI.new!(%{api_key: "sk-test"})
      result = FileResult.new!(%{file_id: "file-abc123", provider: :openai})
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
               %{"id" => "file-1", "filename" => "a.pdf", "bytes" => 100},
               %{"id" => "file-2", "filename" => "b.pdf", "bytes" => 200}
             ]
           }
         }}
      end)

      uploader = FileOpenAI.new!(%{api_key: "sk-test"})

      assert {:ok, [%FileResult{file_id: "file-1"}, %FileResult{file_id: "file-2"}]} =
               FileUploader.list(uploader)
    end
  end
end
