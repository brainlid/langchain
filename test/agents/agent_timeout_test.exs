defmodule LangChain.Agents.AgentTimeoutTest do
  use LangChain.BaseCase, async: true

  alias LangChain.Agents.Agent

  describe "async_tool_timeout field" do
    test "accepts integer timeout" do
      {:ok, agent} =
        Agent.new(
          %{
            model: mock_model(),
            async_tool_timeout: 10 * 60 * 1000
          },
          replace_default_middleware: true
        )

      assert agent.async_tool_timeout == 600_000
    end

    test "accepts :infinity" do
      {:ok, agent} =
        Agent.new(
          %{
            model: mock_model(),
            async_tool_timeout: :infinity
          },
          replace_default_middleware: true
        )

      assert agent.async_tool_timeout == :infinity
    end

    test "defaults to nil (uses chain/app default)" do
      {:ok, agent} =
        Agent.new(
          %{model: mock_model()},
          replace_default_middleware: true
        )

      assert agent.async_tool_timeout == nil
    end

    test "new! also accepts timeout values" do
      agent =
        Agent.new!(
          %{
            model: mock_model(),
            async_tool_timeout: 5 * 60 * 1000
          },
          replace_default_middleware: true
        )

      assert agent.async_tool_timeout == 300_000
    end
  end
end
