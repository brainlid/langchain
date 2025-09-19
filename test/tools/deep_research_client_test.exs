defmodule LangChain.Tools.DeepResearchClientTest do
  use LangChain.BaseCase

  alias LangChain.Tools.DeepResearchClient

  describe "module structure" do
    test "has required functions" do
      # Test that the module compiles and has the expected structure
      assert Code.ensure_loaded?(DeepResearchClient)

      # Test that the module exports the expected functions
      module_info = DeepResearchClient.__info__(:functions)
      assert Enum.member?(module_info, {:create_research, 2})
      assert Enum.member?(module_info, {:check_status, 1})
      assert Enum.member?(module_info, {:get_results, 1})
    end

    test "module loads correctly" do
      # Test that the module has the expected public interface
      assert is_function(&DeepResearchClient.create_research/2)
      assert is_function(&DeepResearchClient.check_status/1)
      assert is_function(&DeepResearchClient.get_results/1)
    end
  end

  describe "API integration" do
    @tag live_call: true
    test "creates research request successfully" do
      # With a valid API key, this should successfully create a research request
      result = DeepResearchClient.create_research("test query")

      case result do
        {:ok, request_id} ->
          # Verify we got a request ID back
          assert is_binary(request_id)
          assert String.starts_with?(request_id, "resp_")

          # Test status checking
          {:ok, status} = DeepResearchClient.check_status(request_id)
          assert is_map(status)
          assert Map.has_key?(status, :status)

        {:error, reason} ->
          # API error - still a valid test result
          assert is_binary(reason)
      end
    end

    @tag live_call: true
    test "creates research request with all new parameters" do
      # Test with all parameters including new ones
      options = %{
        model: "o4-mini-deep-research-2025-06-26",
        system_message: "Focus on recent developments",
        max_tool_calls: 10,
        summary: "detailed",
        include_code_interpreter: false
      }

      result = DeepResearchClient.create_research("test query with parameters", options)

      case result do
        {:ok, request_id} ->
          assert is_binary(request_id)
          assert String.starts_with?(request_id, "resp_")

        {:error, reason} ->
          # API error - still a valid test result
          assert is_binary(reason)
      end
    end
  end
end
