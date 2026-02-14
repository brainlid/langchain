defmodule LangChain.Tools.DeepResearch.ResearchResultTest do
  use LangChain.BaseCase

  alias LangChain.Tools.DeepResearch.ResearchResult
  alias LangChain.Message.Citation
  alias LangChain.Message.CitationSource

  describe "changeset/2" do
    test "creates valid changeset with required fields" do
      attrs = %{
        id: "req_12345",
        output_text: "Research findings"
      }

      changeset = ResearchResult.changeset(%ResearchResult{}, attrs)
      assert changeset.valid?
    end

    test "requires id and output_text" do
      changeset = ResearchResult.changeset(%ResearchResult{}, %{})
      refute changeset.valid?

      # Check that the required fields are in the errors
      errors = changeset.errors
      assert Keyword.has_key?(errors, :id)
      assert Keyword.has_key?(errors, :output_text)
    end

    test "casts embedded sources as Citation structs" do
      attrs = %{
        id: "req_12345",
        output_text: "Research findings",
        sources: [
          %{
            cited_text: "Solar power is growing",
            start_index: 0,
            end_index: 20,
            source: %{type: :web, title: "Source 1", url: "https://example.com/1"}
          },
          %{
            cited_text: "Wind energy is efficient",
            start_index: 25,
            end_index: 50,
            source: %{type: :web, title: "Source 2", url: "https://example.com/2"}
          }
        ]
      }

      changeset = ResearchResult.changeset(%ResearchResult{}, attrs)
      assert changeset.valid?

      result = Ecto.Changeset.apply_changes(changeset)
      assert length(result.sources) == 2

      first = hd(result.sources)
      assert %Citation{} = first
      assert first.cited_text == "Solar power is growing"
      assert first.start_index == 0
      assert first.end_index == 20
      assert %CitationSource{} = first.source
      assert first.source.type == :web
      assert first.source.title == "Source 1"
      assert first.source.url == "https://example.com/1"
    end

    test "casts embedded usage statistics" do
      attrs = %{
        id: "req_12345",
        output_text: "Research findings",
        usage: %{
          input_tokens: 100,
          output_tokens: 500,
          total_tokens: 600,
          reasoning_tokens: 50
        }
      }

      changeset = ResearchResult.changeset(%ResearchResult{}, attrs)
      assert changeset.valid?

      result = Ecto.Changeset.apply_changes(changeset)
      assert result.usage.input_tokens == 100
      assert result.usage.total_tokens == 600
    end

    test "validates usage token counts are non-negative" do
      attrs = %{
        id: "req_12345",
        output_text: "Research findings",
        usage: %{
          input_tokens: -10,
          output_tokens: -5
        }
      }

      changeset = ResearchResult.changeset(%ResearchResult{}, attrs)
      refute changeset.valid?

      # Check that validation fails for negative numbers
      assert changeset.changes.usage.errors != []
    end
  end

  describe "from_api_response/1" do
    test "creates result from valid API response" do
      api_response = %{
        "id" => "req_12345",
        "model" => "o3-deep-research-2025-06-26",
        "created_at" => 1_234_567_890,
        "output" => [
          %{
            "type" => "message",
            "content" => [
              %{
                "type" => "output_text",
                "text" => "Comprehensive research findings on renewable energy.",
                "annotations" => [
                  %{
                    "title" => "Solar Power Study",
                    "url" => "https://example.com/solar",
                    "start_index" => 0,
                    "end_index" => 20,
                    "snippet" => "Solar power is growing rapidly"
                  }
                ]
              }
            ]
          },
          %{
            "type" => "web_search_call",
            "status" => "completed",
            "action" => %{"type" => "search", "query" => "renewable energy 2024"}
          }
        ],
        "usage" => %{
          "input_tokens" => 150,
          "output_tokens" => 800,
          "total_tokens" => 950
        }
      }

      assert {:ok, result} = ResearchResult.from_api_response(api_response)

      assert result.id == "req_12345"
      assert result.model == "o3-deep-research-2025-06-26"
      assert result.output_text == "Comprehensive research findings on renewable energy."
      assert result.created_at == 1_234_567_890

      assert length(result.sources) == 1
      citation = hd(result.sources)
      assert %Citation{} = citation
      assert citation.cited_text == "Solar power is growing rapidly"
      assert citation.start_index == 0
      assert citation.end_index == 20
      assert citation.source.type == :web
      assert citation.source.title == "Solar Power Study"
      assert citation.source.url == "https://example.com/solar"
      assert citation.metadata == %{"provider" => "openai_deep_research"}

      assert result.usage.input_tokens == 150
      assert result.usage.total_tokens == 950

      assert length(result.tool_calls) == 1
      tool_call = hd(result.tool_calls)
      assert tool_call.type == "web_search_call"
      assert tool_call.status == "completed"
    end

    test "handles response with minimal content" do
      api_response = %{
        "id" => "req_minimal",
        "output" => [
          %{
            "type" => "message",
            "content" => [
              %{"type" => "output_text", "text" => "Basic findings"}
            ]
          }
        ]
      }

      assert {:ok, result} = ResearchResult.from_api_response(api_response)

      assert result.id == "req_minimal"
      assert result.output_text == "Basic findings"
      assert result.sources == []
      assert result.tool_calls == []
    end

    test "handles malformed API response" do
      api_response = %{"invalid" => "response"}

      assert {:error, changeset} = ResearchResult.from_api_response(api_response)
      refute changeset.valid?

      # Check that the required fields are missing
      errors = changeset.errors
      assert Keyword.has_key?(errors, :id)
    end

    test "handles multiple annotations creating multiple Citation structs" do
      api_response = %{
        "id" => "req_multi",
        "output" => [
          %{
            "type" => "message",
            "content" => [
              %{
                "type" => "output_text",
                "text" => "Research on multiple topics.",
                "annotations" => [
                  %{
                    "title" => "First Source",
                    "url" => "https://example.com/1",
                    "start_index" => 0,
                    "end_index" => 10,
                    "snippet" => "First finding"
                  },
                  %{
                    "title" => "Second Source",
                    "url" => "https://example.com/2",
                    "start_index" => 15,
                    "end_index" => 25,
                    "snippet" => "Second finding"
                  }
                ]
              }
            ]
          }
        ]
      }

      assert {:ok, result} = ResearchResult.from_api_response(api_response)
      assert length(result.sources) == 2

      [first, second] = result.sources
      assert first.source.url == "https://example.com/1"
      assert first.cited_text == "First finding"
      assert second.source.url == "https://example.com/2"
      assert second.cited_text == "Second finding"
    end
  end

  describe "source_count/1" do
    test "returns correct source count" do
      result = %ResearchResult{
        sources: [
          Citation.new!(%{source: %{type: :web, url: "https://example.com/1"}}),
          Citation.new!(%{source: %{type: :web, url: "https://example.com/2"}})
        ]
      }

      assert ResearchResult.source_count(result) == 2
    end

    test "returns zero for no sources" do
      result = %ResearchResult{sources: []}
      assert ResearchResult.source_count(result) == 0
    end
  end

  describe "tool_call_count/1" do
    test "returns correct tool call count" do
      result = %ResearchResult{
        tool_calls: [
          %ResearchResult.ToolCall{type: "web_search_call"},
          %ResearchResult.ToolCall{type: "code_interpreter_call"}
        ]
      }

      assert ResearchResult.tool_call_count(result) == 2
    end
  end

  describe "format_for_display/1" do
    test "formats result with sources for display" do
      result = %ResearchResult{
        output_text: "This is the research content.",
        sources: [
          Citation.new!(%{
            source: %{type: :web, title: "Source 1", url: "https://example.com/1"}
          }),
          Citation.new!(%{
            source: %{type: :web, title: "Source 2", url: "https://example.com/2"}
          })
        ]
      }

      formatted = ResearchResult.format_for_display(result)

      assert formatted =~ "## Research Findings"
      assert formatted =~ "This is the research content."
      assert formatted =~ "## Sources"
      assert formatted =~ "1. Source 1 - https://example.com/1"
      assert formatted =~ "2. Source 2 - https://example.com/2"
    end

    test "formats result without sources" do
      result = %ResearchResult{
        output_text: "Research without sources.",
        sources: []
      }

      formatted = ResearchResult.format_for_display(result)

      assert formatted =~ "## Research Findings"
      assert formatted =~ "Research without sources."
      refute formatted =~ "## Sources"
    end
  end

  describe "source_urls/1" do
    test "extracts URLs from citation sources" do
      result = %ResearchResult{
        sources: [
          Citation.new!(%{source: %{type: :web, url: "https://example.com/1"}}),
          Citation.new!(%{source: %{type: :web, url: "https://example.com/2"}})
        ]
      }

      urls = ResearchResult.source_urls(result)
      assert urls == ["https://example.com/1", "https://example.com/2"]
    end

    test "returns empty list for no sources" do
      result = %ResearchResult{sources: []}
      assert ResearchResult.source_urls(result) == []
    end

    test "filters out nil URLs" do
      result = %ResearchResult{
        sources: [
          Citation.new!(%{source: %{type: :web, url: "https://example.com/1"}}),
          Citation.new!(%{source: %{type: :document, title: "Local Doc"}})
        ]
      }

      urls = ResearchResult.source_urls(result)
      assert urls == ["https://example.com/1"]
    end
  end
end
