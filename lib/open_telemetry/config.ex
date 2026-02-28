defmodule LangChain.OpenTelemetry.Config do
  @moduledoc """
  Configuration for the LangChain OpenTelemetry integration.

  ## Options

    * `:capture_input_messages` - When `true`, records `gen_ai.input.messages` as a
      span attribute containing the serialized input messages. Defaults to `false`.

    * `:capture_output_messages` - When `true`, records `gen_ai.output.messages` as a
      span attribute containing the serialized output messages. Defaults to `false`.

    * `:capture_tool_arguments` - When `true`, records `gen_ai.tool.call.arguments` as a
      span attribute. Defaults to `false`.

    * `:capture_tool_results` - When `true`, records `gen_ai.tool.call.result` as a
      span attribute. Defaults to `false`.

    * `:enable_metrics` - When `true`, records OTel histogram metrics for operation
      duration and token usage. Defaults to `true`.
  """

  @type t :: %__MODULE__{
          capture_input_messages: boolean(),
          capture_output_messages: boolean(),
          capture_tool_arguments: boolean(),
          capture_tool_results: boolean(),
          enable_metrics: boolean()
        }

  defstruct capture_input_messages: false,
            capture_output_messages: false,
            capture_tool_arguments: false,
            capture_tool_results: false,
            enable_metrics: true

  @doc """
  Creates a new config from the given options.

  ## Examples

      iex> LangChain.OpenTelemetry.Config.new(capture_input_messages: true)
      %LangChain.OpenTelemetry.Config{capture_input_messages: true}
  """
  @spec new(keyword()) :: t()
  def new(opts \\ []) do
    struct!(__MODULE__, opts)
  end
end
