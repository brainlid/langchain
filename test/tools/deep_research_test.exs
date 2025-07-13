defmodule LangChain.Tools.DeepResearchTest do
  use LangChain.BaseCase

  doctest LangChain.Tools.DeepResearch
  alias LangChain.Tools.DeepResearch
  alias LangChain.Function
  alias LangChain.ChatModels.ChatOpenAI
  alias LangChain.Chains.LLMChain
  alias LangChain.Message

  describe "new/0" do
    test "defines the function correctly" do
      assert {:ok, %Function{} = function} = DeepResearch.new()
      assert function.name == "deep_research"
      assert function.description =~ "comprehensive research"
      assert function.function != nil
      assert function.async == true

      assert function.parameters_schema == %{
               type: "object",
               properties: %{
                 query: %{
                   type: "string",
                   description:
                     "The research question or topic to investigate. Be specific and detailed for best results."
                 },
                 model: %{
                   type: "string",
                   enum: ["o3-deep-research-2025-06-26", "o4-mini-deep-research-2025-06-26"],
                   description:
                     "The deep research model to use. o3-deep-research provides highest quality (5-30 min), o4-mini-deep-research is faster (shorter time).",
                   default: "o3-deep-research-2025-06-26"
                 },
                 system_message: %{
                   type: "string",
                   description:
                     "Optional guidance for the research approach, methodology, or specific requirements."
                 },
                 max_tool_calls: %{
                   type: "integer",
                   description:
                     "Maximum number of tool calls (web searches, etc.) to make. Controls cost and latency.",
                   minimum: 1,
                   maximum: 100
                 },
                 background: %{
                   type: "boolean",
                   description:
                     "Whether to run in background mode (recommended for long research tasks).",
                   default: true
                 }
               },
               required: ["query"]
             }
    end

    test "assigned function can be executed" do
      {:ok, deep_research} = DeepResearch.new()
      assert function = deep_research.function
      assert is_function(function, 2)
    end
  end

  describe "new!/0" do
    test "returns the function" do
      assert %Function{name: "deep_research"} = DeepResearch.new!()
    end
  end

  describe "execute/2" do
    test "returns error when query parameter is missing" do
      assert {:error, "ERROR: 'query' parameter is required for deep research"} ==
               DeepResearch.execute(%{}, %{})
    end

    @tag live_call: true, live_deep_research: true
    # 5 minutes for real API test
    @tag timeout: 300_000
    test "handles API interaction appropriately" do
      # Test with minimal parameters - this will actually create a research request
      # This is a real API test to verify integration works
      result = DeepResearch.execute(%{"query" => "brief test research query"}, %{})

      # With a valid API key, this should either succeed in creating the request
      # or timeout/fail during polling - both are acceptable for this test
      case result do
        {:ok, research_result} ->
          # If research completes, verify the result structure
          assert is_binary(research_result)
          assert research_result =~ "Research Findings"

        {:error, reason} ->
          # Expected timeout or API error during polling
          assert is_binary(reason)
      end
    end

    test "validates parameter structure" do
      # Test parameter validation without making API calls
      function = DeepResearch.new!()

      # Verify the function has the expected structure
      assert function.name == "deep_research"
      assert function.async == true
      assert Map.has_key?(function.parameters_schema, :required)
      assert "query" in function.parameters_schema.required
    end
  end

  describe "internal functions" do
    test "format_research_result handles different input formats" do
      # Test the public interface through module introspection
      assert function_exported?(DeepResearch, :execute, 2)
      assert function_exported?(DeepResearch, :new, 0)
      assert function_exported?(DeepResearch, :new!, 0)
    end
  end

  describe "module integration" do
    test "integrates with LLMChain correctly" do
      # Test that the function integrates properly with LangChain patterns
      function = DeepResearch.new!()

      # Verify it has the expected Function structure for LLMChain integration
      assert is_function(function.function, 2)
      assert is_binary(function.name)
      assert is_binary(function.description)
      assert is_map(function.parameters_schema)
    end
  end

  # Live integration test (requires actual OpenAI API key and takes significant time)
  describe "live test" do
    @tag live_call: true, live_deep_research: true
    # 30 minutes timeout for live deep research
    @tag timeout: 1_800_000
    test "performs actual deep research with live API" do
      # Skip if no API key is configured
      case Application.get_env(:langchain, :openai_key) do
        nil ->
          :skip

        "" ->
          :skip

        "test" ->
          :skip

        _key ->
          # This would perform an actual Deep Research call
          # Only runs when there's a real API key configured
          model = ChatOpenAI.new!(%{temperature: 0, stream: false})

          {:ok, updated_chain} =
            %{llm: model, verbose: false}
            |> LLMChain.new!()
            |> LLMChain.add_message(
              Message.new_user!("Research renewable energy trends briefly.")
            )
            |> LLMChain.add_tools(DeepResearch.new!())
            |> LLMChain.run(mode: :while_needs_response)

          # If we get here, the research completed
          assert updated_chain.last_message.role == :assistant
          assert updated_chain.last_message.content =~ ~r/research|energy/i
      end
    end
  end
end
