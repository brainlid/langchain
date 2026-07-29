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

    * `:enable_metrics` - When `true`, attaches the metrics handler, which re-emits
      LangChain telemetry as intermediary `[:langchain, :otel, …]` metric events for a
      consumer (`Telemetry.Metrics`, PromEx, …) to record. It does **not** record OTel
      histograms directly. Defaults to `true`. See `LangChain.OpenTelemetry.MetricsHandler`.

    * `:inherit_attributes` - When `true`, attributes supplied via
      `custom_context[:otel_attributes]` on a chain are inherited by the `chat` and
      `execute_tool` spans nested under it, instead of landing only on the chain span.
      Defaults to `true`.

      This is what gets application context (user, tenant, feature) onto the LLM span,
      which carries no `custom_context` of its own. Inherited values always lose to
      attributes the span derives itself, so an inherited `gen_ai.request.model` can
      never mask the real per-call model.

      Inheritance is carried on the OpenTelemetry context, so it follows the trace
      across process boundaries — including `async: true` tools running in their own
      `Task` — but is never serialized onto outbound requests the way baggage is. Turn
      it off if a very large attribute map makes the per-span copy undesirable.
  """

  @type t :: %__MODULE__{
          capture_input_messages: boolean(),
          capture_output_messages: boolean(),
          capture_tool_arguments: boolean(),
          capture_tool_results: boolean(),
          enable_metrics: boolean(),
          inherit_attributes: boolean()
        }

  defstruct capture_input_messages: false,
            capture_output_messages: false,
            capture_tool_arguments: false,
            capture_tool_results: false,
            enable_metrics: true,
            inherit_attributes: true

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
