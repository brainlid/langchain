defmodule LangChain.Chains.LLMChainTimeoutTest do
  # Must be async: false because we modify global application config
  use LangChain.BaseCase, async: false

  alias LangChain.Chains.LLMChain
  alias LangChain.ChatModels.ChatOpenAI

  # Store and restore application config for all tests in this module
  setup do
    original = Application.get_env(:langchain, :async_tool_timeout)

    on_exit(fn ->
      if original do
        Application.put_env(:langchain, :async_tool_timeout, original)
      else
        Application.delete_env(:langchain, :async_tool_timeout)
      end
    end)

    # Clear any existing config to start fresh for each test
    Application.delete_env(:langchain, :async_tool_timeout)

    {:ok, chat} = ChatOpenAI.new(%{temperature: 0})
    %{chat: chat, original_config: original}
  end

  describe "async_tool_timeout configuration" do
    test "defaults to nil when not configured", %{chat: chat} do
      {:ok, chain} = LLMChain.new(%{llm: chat})
      assert chain.async_tool_timeout == nil
      # The actual default is applied at execution time via default_async_tool_timeout()
    end

    test "accepts integer timeout", %{chat: chat} do
      {:ok, chain} =
        LLMChain.new(%{
          llm: chat,
          async_tool_timeout: 5 * 60 * 1000
        })

      assert chain.async_tool_timeout == 300_000
    end

    test "accepts :infinity", %{chat: chat} do
      {:ok, chain} =
        LLMChain.new(%{
          llm: chat,
          async_tool_timeout: :infinity
        })

      assert chain.async_tool_timeout == :infinity
    end

    test "new! also accepts timeout values", %{chat: chat} do
      chain =
        LLMChain.new!(%{
          llm: chat,
          async_tool_timeout: 10 * 60 * 1000
        })

      assert chain.async_tool_timeout == 600_000
    end
  end

  describe "application config" do
    test "application config can be set to integer" do
      Application.put_env(:langchain, :async_tool_timeout, 600_000)
      assert Application.get_env(:langchain, :async_tool_timeout) == 600_000
    end

    test "application config can be set to :infinity" do
      Application.put_env(:langchain, :async_tool_timeout, :infinity)
      assert Application.get_env(:langchain, :async_tool_timeout) == :infinity
    end

    test "chain-level timeout takes precedence over application config", %{chat: chat} do
      Application.put_env(:langchain, :async_tool_timeout, 600_000)

      {:ok, chain} =
        LLMChain.new(%{
          llm: chat,
          async_tool_timeout: 300_000
        })

      # Chain-level setting should be stored on the struct
      assert chain.async_tool_timeout == 300_000
      # Application config is still what was set
      assert Application.get_env(:langchain, :async_tool_timeout) == 600_000
    end
  end
end
