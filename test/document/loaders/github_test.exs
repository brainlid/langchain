defmodule LangChain.Document.Loaders.GithubTest do
  use ExUnit.Case, async: true

  alias LangChain.Document.Loaders.Github
  alias LangChain.Document
  alias LangChain.LangChainError

  describe "new/1" do
    test "creates a new Github struct with default values" do
      assert {:ok, %Github{endpoint: "https://api.github.com", receive_timeout: 60_000}} =
               Github.new(%{})
    end

    test "creates a new Github struct with custom values" do
      attrs = %{
        endpoint: "https://custom.github.com",
        api_key: "test_key",
        receive_timeout: 30_000
      }

      assert {:ok,
              %Github{
                endpoint: "https://custom.github.com",
                api_key: "test_key",
                receive_timeout: 30_000
              }} =
               Github.new(attrs)
    end
  end

  describe "load/1 with unsupported type" do
    test "raises LangChainError when type is unsupported" do
      options = %{type: :unsupported_type}

      assert_raise LangChainError, "Unsupported type: :unsupported_type", fn ->
        Github.load(options)
      end
    end
  end

  describe "new!/1" do
    test "creates a new Github struct with default values" do
      assert %Github{endpoint: "https://api.github.com", receive_timeout: 60_000} =
               Github.new!(%{})
    end
  end

  describe "to_documents/1" do
    test "converts a list of issues into Document structs" do
      issues = [
        %{id: 1, title: "Issue 1", body: "Body 1"},
        %{id: 2, title: "Issue 2", body: "Body 2"}
      ]

      documents = Github.to_documents(issues)

      assert [
               %Document{
                 content: "Body 1",
                 metadata: %{id: 1, title: "Issue 1"},
                 type: "github_issue"
               },
               %Document{
                 content: "Body 2",
                 metadata: %{id: 2, title: "Issue 2"},
                 type: "github_issue"
               }
             ] = documents
    end
  end

  describe "process_response/1" do
    test "processes a list of GitHub issues from the API response" do
      response = [
        %{"id" => 1, "title" => "Issue 1", "body" => "Body 1"},
        %{"id" => 2, "title" => "Issue 2", "body" => "Body 2"}
      ]

      processed_issues = Github.process_response(response)

      assert [
               %{id: 1, title: "Issue 1", body: "Body 1"},
               %{id: 2, title: "Issue 2", body: "Body 2"}
             ] = processed_issues
    end
  end
end
