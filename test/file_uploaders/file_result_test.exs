defmodule LangChain.FileUploader.FileResultTest do
  use ExUnit.Case
  alias LangChain.FileUploader.FileResult
  alias LangChain.LangChainError

  describe "new/1" do
    test "creates result with file_id" do
      assert {:ok, %FileResult{} = result} =
               FileResult.new(%{
                 file_id: "file-abc123",
                 filename: "doc.pdf",
                 mime_type: "application/pdf",
                 size_bytes: 1024,
                 provider: :openai
               })

      assert result.file_id == "file-abc123"
      assert result.filename == "doc.pdf"
      assert result.provider == :openai
    end

    test "creates result with file_uri" do
      assert {:ok, %FileResult{} = result} =
               FileResult.new(%{
                 file_uri: "https://example.com/files/abc",
                 filename: "doc.pdf",
                 provider: :google
               })

      assert result.file_uri == "https://example.com/files/abc"
      assert result.provider == :google
      assert result.file_id == nil
    end

    test "creates result with both file_id and file_uri" do
      assert {:ok, %FileResult{}} =
               FileResult.new(%{
                 file_id: "file-abc",
                 file_uri: "https://example.com/files/abc",
                 provider: :openai
               })
    end

    test "returns error when neither file_id nor file_uri is present" do
      assert {:error, changeset} =
               FileResult.new(%{
                 filename: "doc.pdf",
                 provider: :openai
               })

      assert {"either file_id or file_uri must be present", _} = changeset.errors[:file_id]
    end

    test "returns error when provider is missing" do
      assert {:error, changeset} =
               FileResult.new(%{
                 file_id: "file-abc"
               })

      assert {"can't be blank", _} = changeset.errors[:provider]
    end

    test "returns error for invalid provider" do
      assert {:error, changeset} =
               FileResult.new(%{
                 file_id: "file-abc",
                 provider: :unsupported
               })

      assert {"is invalid", _} = changeset.errors[:provider]
    end

    test "raw defaults to empty map" do
      {:ok, result} = FileResult.new(%{file_id: "file-abc", provider: :openai})
      assert result.raw == %{}
    end
  end

  describe "new!/1" do
    test "returns struct when valid" do
      %FileResult{} = result = FileResult.new!(%{file_id: "file-abc", provider: :anthropic})
      assert result.file_id == "file-abc"
    end

    test "raises LangChainError when invalid" do
      assert_raise LangChainError, fn ->
        FileResult.new!(%{filename: "doc.pdf"})
      end
    end
  end
end
