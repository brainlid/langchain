defmodule LangChain.OpenTelemetry.ConfigTest do
  use ExUnit.Case, async: true

  alias LangChain.OpenTelemetry.Config

  describe "new/1" do
    test "returns defaults when no options given" do
      config = Config.new()

      assert %Config{
               capture_input_messages: false,
               capture_output_messages: false,
               capture_tool_arguments: false,
               capture_tool_results: false,
               enable_metrics: true
             } = config
    end

    test "accepts keyword options" do
      config = Config.new(capture_input_messages: true, enable_metrics: false)

      assert config.capture_input_messages == true
      assert config.enable_metrics == false
      assert config.capture_output_messages == false
    end

    test "raises on unknown keys" do
      assert_raise KeyError, fn ->
        Config.new(bogus_option: true)
      end
    end
  end
end
