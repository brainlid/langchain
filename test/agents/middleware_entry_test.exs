defmodule LangChain.Agents.MiddlewareEntryTest do
  use LangChain.BaseCase, async: true

  alias LangChain.Agents.MiddlewareEntry

  describe "to_raw_spec/1" do
    test "converts MiddlewareEntry with no config to module atom" do
      entry = %MiddlewareEntry{
        id: MyMiddleware,
        module: MyMiddleware,
        config: %{id: MyMiddleware}
      }

      assert MiddlewareEntry.to_raw_spec(entry) == MyMiddleware
    end

    test "converts MiddlewareEntry with internal-only keys to module atom" do
      entry = %MiddlewareEntry{
        id: MyMiddleware,
        module: MyMiddleware,
        config: %{id: MyMiddleware, middleware_id: MyMiddleware}
      }

      assert MiddlewareEntry.to_raw_spec(entry) == MyMiddleware
    end

    test "converts MiddlewareEntry with config options to {module, opts} tuple" do
      entry = %MiddlewareEntry{
        id: MyMiddleware,
        module: MyMiddleware,
        config: %{
          id: MyMiddleware,
          max_items: 100,
          enabled: true
        }
      }

      result = MiddlewareEntry.to_raw_spec(entry)

      assert {MyMiddleware, opts} = result
      assert Keyword.get(opts, :max_items) == 100
      assert Keyword.get(opts, :enabled) == true
      # Internal keys should be removed
      refute Keyword.has_key?(opts, :id)
      refute Keyword.has_key?(opts, :middleware_id)
    end

    test "removes internal keys but preserves all user options" do
      entry = %MiddlewareEntry{
        id: "custom_id",
        module: MyMiddleware,
        config: %{
          id: "custom_id",
          middleware_id: "custom_id",
          agent_id: "parent",
          model: :some_model,
          custom_option: "value",
          another_option: 42
        }
      }

      result = MiddlewareEntry.to_raw_spec(entry)

      assert {MyMiddleware, opts} = result
      # User options should be preserved
      assert Keyword.get(opts, :agent_id) == "parent"
      assert Keyword.get(opts, :model) == :some_model
      assert Keyword.get(opts, :custom_option) == "value"
      assert Keyword.get(opts, :another_option) == 42
      # Internal keys should be removed
      refute Keyword.has_key?(opts, :id)
      refute Keyword.has_key?(opts, :middleware_id)
    end

    test "passes through raw module atom unchanged" do
      assert MiddlewareEntry.to_raw_spec(MyMiddleware) == MyMiddleware
    end

    test "passes through raw {module, opts} tuple unchanged" do
      spec = {MyMiddleware, [opt: "value", another: 123]}
      assert MiddlewareEntry.to_raw_spec(spec) == spec
    end

    test "handles empty options keyword list in raw tuple" do
      spec = {MyMiddleware, []}
      assert MiddlewareEntry.to_raw_spec(spec) == spec
    end

    test "preserves order of options in config" do
      # Note: Maps don't guarantee order, but we can verify all keys are present
      entry = %MiddlewareEntry{
        id: MyMiddleware,
        module: MyMiddleware,
        config: %{
          id: MyMiddleware,
          z_option: "z",
          a_option: "a",
          m_option: "m"
        }
      }

      {MyMiddleware, opts} = MiddlewareEntry.to_raw_spec(entry)

      # Verify all options are present (order may vary)
      assert length(opts) == 3
      assert Keyword.get(opts, :z_option) == "z"
      assert Keyword.get(opts, :a_option) == "a"
      assert Keyword.get(opts, :m_option) == "m"
    end
  end

  describe "to_raw_specs/1" do
    test "converts list of MiddlewareEntry structs to raw specs" do
      entries = [
        %MiddlewareEntry{
          id: Middleware1,
          module: Middleware1,
          config: %{id: Middleware1}
        },
        %MiddlewareEntry{
          id: Middleware2,
          module: Middleware2,
          config: %{id: Middleware2, opt: "value"}
        },
        %MiddlewareEntry{
          id: Middleware3,
          module: Middleware3,
          config: %{id: Middleware3, max_items: 10, enabled: false}
        }
      ]

      result = MiddlewareEntry.to_raw_specs(entries)

      assert length(result) == 3
      assert Enum.at(result, 0) == Middleware1

      assert {Middleware2, opts2} = Enum.at(result, 1)
      assert Keyword.get(opts2, :opt) == "value"

      assert {Middleware3, opts3} = Enum.at(result, 2)
      assert Keyword.get(opts3, :max_items) == 10
      assert Keyword.get(opts3, :enabled) == false
    end

    test "handles mixed list of MiddlewareEntry structs and raw specs" do
      entries = [
        %MiddlewareEntry{
          id: Middleware1,
          module: Middleware1,
          config: %{id: Middleware1, opt: "value"}
        },
        Middleware2,
        {Middleware3, [existing: "option"]}
      ]

      result = MiddlewareEntry.to_raw_specs(entries)

      assert length(result) == 3

      assert {Middleware1, opts1} = Enum.at(result, 0)
      assert Keyword.get(opts1, :opt) == "value"

      assert Enum.at(result, 1) == Middleware2
      assert Enum.at(result, 2) == {Middleware3, [existing: "option"]}
    end

    test "handles empty list" do
      assert MiddlewareEntry.to_raw_specs([]) == []
    end

    test "handles list with only raw specs" do
      specs = [
        Middleware1,
        {Middleware2, [opt: "value"]},
        Middleware3
      ]

      result = MiddlewareEntry.to_raw_specs(specs)

      assert result == specs
    end
  end

  describe "integration with actual middleware configs" do
    test "converts real Summarization middleware entry correctly" do
      # Simulate what happens when Summarization middleware is initialized
      entry = %MiddlewareEntry{
        id: LangChain.Agents.Middleware.Summarization,
        module: LangChain.Agents.Middleware.Summarization,
        config: %{
          id: LangChain.Agents.Middleware.Summarization,
          model: :some_chat_model,
          max_tokens_before_summary: 170_000,
          messages_to_keep: 6,
          summary_prompt: "Custom prompt",
          token_counter: &Function.identity/1
        }
      }

      {module, opts} = MiddlewareEntry.to_raw_spec(entry)

      assert module == LangChain.Agents.Middleware.Summarization
      assert Keyword.get(opts, :model) == :some_chat_model
      assert Keyword.get(opts, :max_tokens_before_summary) == 170_000
      assert Keyword.get(opts, :messages_to_keep) == 6
      assert Keyword.get(opts, :summary_prompt) == "Custom prompt"
      assert is_function(Keyword.get(opts, :token_counter), 1)
      refute Keyword.has_key?(opts, :id)
    end

    test "converts real FileSystem middleware entry correctly" do
      entry = %MiddlewareEntry{
        id: LangChain.Agents.Middleware.FileSystem,
        module: LangChain.Agents.Middleware.FileSystem,
        config: %{
          id: LangChain.Agents.Middleware.FileSystem,
          middleware_id: LangChain.Agents.Middleware.FileSystem,
          agent_id: "test-agent-123",
          enabled_tools: ["read_file", "write_file"],
          custom_tool_descriptions: %{"read_file" => "Read a file"}
        }
      }

      {module, opts} = MiddlewareEntry.to_raw_spec(entry)

      assert module == LangChain.Agents.Middleware.FileSystem
      assert Keyword.get(opts, :agent_id) == "test-agent-123"
      assert Keyword.get(opts, :enabled_tools) == ["read_file", "write_file"]
      assert Keyword.get(opts, :custom_tool_descriptions) == %{"read_file" => "Read a file"}
      refute Keyword.has_key?(opts, :id)
      refute Keyword.has_key?(opts, :middleware_id)
    end

    test "converts middleware entry with model struct containing thinking config" do
      # This is the specific case that triggered the atom limit bug
      mock_model = %{
        __struct__: LangChain.ChatModels.ChatAnthropic,
        model: "claude-sonnet-4-5-20250929",
        thinking: %{type: "enabled", budget_tokens: 2000},
        temperature: 1,
        stream: true
      }

      entry = %MiddlewareEntry{
        id: LangChain.Agents.Middleware.Summarization,
        module: LangChain.Agents.Middleware.Summarization,
        config: %{
          id: LangChain.Agents.Middleware.Summarization,
          model: mock_model,
          max_tokens_before_summary: 170_000
        }
      }

      # This should not blow up trying to create atoms
      {module, opts} = MiddlewareEntry.to_raw_spec(entry)

      assert module == LangChain.Agents.Middleware.Summarization
      assert Keyword.get(opts, :model) == mock_model
      assert Keyword.get(opts, :max_tokens_before_summary) == 170_000
      refute Keyword.has_key?(opts, :id)
    end
  end
end
