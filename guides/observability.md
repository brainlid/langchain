# Observability with Telemetry and OpenTelemetry

When you run an LLM chain in production, you want to answer questions like: How
long did that request take? How many tokens did it burn? Which tool call failed?
Where is the latency — the model, or my own tool code? LangChain answers these
through two layers you can adopt independently:

1. **`LangChain.Telemetry`** — vendor-neutral [`:telemetry`](https://hexdocs.pm/telemetry)
   events emitted for every LLM call, chain execution, and tool call. Attach your
   own handlers, feed them into `Telemetry.Metrics`, PromEx, `Logger`, or anything
   else. No extra dependencies.
2. **`LangChain.OpenTelemetry`** — an optional, opt-in integration that translates
   those events into [OpenTelemetry](https://opentelemetry.io/) spans and metrics
   following a subset of the [GenAI Semantic Conventions](https://opentelemetry.io/docs/specs/semconv/gen-ai/)
   (v1.40+). This is what you want for distributed tracing backends like
   [Langfuse](https://langfuse.com/), Honeycomb, Grafana Tempo, Jaeger, or any
   OTLP collector.

Layer 2 is built on layer 1, so everything below applies whether you consume the
raw events yourself or let the OpenTelemetry handler do it for you.

## The telemetry layer

`LangChain.Telemetry` emits standard `:telemetry` events using the naming
convention `[:langchain, component, operation, stage]`. The core lifecycle events
are:

| Event | Emitted when |
|-------|--------------|
| `[:langchain, :llm, :call, :start \| :stop \| :exception]` | An LLM call begins / completes / raises |
| `[:langchain, :chain, :execute, :start \| :stop \| :exception]` | A chain run begins / completes / raises |
| `[:langchain, :tool, :call, :start \| :stop \| :exception]` | A tool (function) call begins / completes / raises |
| `[:langchain, :llm, :prompt]` | The prompt is sent (carries message content) |
| `[:langchain, :llm, :response]` | A response is received (carries content) |

Each `:start`/`:stop`/`:exception` triple shares a `:call_id` (a UUID) so you can
correlate them. `:stop` events carry a `duration` measurement (in native time
units); `:exception` events carry one too, so failed operations remain visible to
latency metrics.

### Key metadata fields

- `:provider` — the LLM provider (`"openai"`, `"anthropic"`, `"google"`, …),
  available on LLM call events.
- `:token_usage` — a `%LangChain.TokenUsage{}` struct on LLM call `:stop` events
  (when the model reports usage) and on chain `:stop` events (aggregated across
  **all** assistant messages in the run, so multi-turn/tool-calling chains don't
  lose earlier turns' tokens).
- `:custom_context` — your `LLMChain.custom_context`, surfaced on chain and tool
  events.
- `:last_message` — the final assembled `%Message{}` on chain `:stop` events.

### Attaching your own handler

```elixir
:telemetry.attach(
  "my-llm-logger",
  [:langchain, :llm, :call, :stop],
  fn _event, measurements, metadata, _config ->
    ms = System.convert_time_unit(measurements.duration, :native, :millisecond)

    require Logger
    Logger.info(
      "LLM #{metadata.provider}/#{metadata.model} took #{ms}ms, " <>
        "tokens: #{inspect(metadata[:token_usage])}"
    )
  end,
  nil
)
```

### A privacy note

Message content is intentionally **excluded** from the lifecycle events
(`:start` / `:stop` / `:exception`) to avoid unconditionally exposing user or PII
data to every handler. Content is only available through the purpose-specific
`[:langchain, :llm, :prompt]` and `[:langchain, :llm, :response]` events —
subscribing to those is an explicit opt-in.

See `LangChain.Telemetry` for the full list of events and the exact metadata
shape of each.

## The OpenTelemetry layer

### Installation

The OpenTelemetry integration ships in LangChain but its dependencies are
**optional** — nothing is pulled in and no code is compiled unless you add them.
Add these to your application's `mix.exs`:

```elixir
def deps do
  [
    {:langchain, "~> 0.8"},
    # OpenTelemetry — required only if you want the OTel integration
    {:opentelemetry_api, "~> 1.4"},
    {:opentelemetry, "~> 1.5"},
    {:opentelemetry_exporter, "~> 1.8"}
  ]
end
```

### Setup

Call `setup/1` once during application startup — for example in your
`Application.start/2`:

```elixir
defmodule MyApp.Application do
  use Application

  def start(_type, _args) do
    LangChain.OpenTelemetry.setup()

    children = [
      # ...
    ]

    Supervisor.start_link(children, strategy: :one_for_one, name: MyApp.Supervisor)
  end
end
```

That attaches the span and metrics handlers to LangChain's telemetry events. From
then on, every chain run, LLM call, and tool call produces OTel spans
automatically. To detach later, call `LangChain.OpenTelemetry.teardown/0`.

### What spans you get

Spans follow a subset of the GenAI Semantic Conventions (see
`LangChain.OpenTelemetry.Attributes` for the attributes emitted) and nest
automatically for synchronous work (they share the process dictionary):

```
invoke_agent llm_chain              (:internal)   gen_ai.operation.name = invoke_agent
  └─ chat gpt-4o                     (:client)     gen_ai.operation.name = chat
       └─ execute_tool get_weather   (:internal)   gen_ai.operation.name = execute_tool
```

| Telemetry event | Span name | Kind |
|-----------------|-----------|------|
| `[:langchain, :chain, :execute, …]` | `invoke_agent {chain_type}` | `:internal` |
| `[:langchain, :llm, :call, …]` | `chat {model}` | `:client` |
| `[:langchain, :tool, :call, …]` | `execute_tool {tool_name}` | `:internal` |

Attributes recorded on the spans include (a subset, always present when
available):

- `gen_ai.operation.name`, `gen_ai.provider.name`
- `gen_ai.output.type` — `"json"` when the model requests structured output
  (`json_response` / `json_schema` / a JSON `response_format`), otherwise `"text"`
- `gen_ai.request.model`, `gen_ai.response.model`
- `server.address`, `server.port` — derived from the model's request endpoint
- Request parameters, when the model sets them: `gen_ai.request.temperature`,
  `gen_ai.request.max_tokens`, `gen_ai.request.top_p`, `gen_ai.request.top_k`,
  `gen_ai.request.frequency_penalty`, `gen_ai.request.presence_penalty`,
  `gen_ai.request.seed`, `gen_ai.request.choice.count`, `gen_ai.request.stream`,
  `gen_ai.request.stop_sequences`, `gen_ai.request.reasoning.level`
- `gen_ai.usage.input_tokens`, `gen_ai.usage.output_tokens`, and — best-effort,
  for providers that report them — `gen_ai.usage.cache_read.input_tokens`,
  `gen_ai.usage.cache_creation.input_tokens`, `gen_ai.usage.reasoning.output_tokens`
- `gen_ai.response.finish_reasons` — best-effort (`"stop"` / `"length"` /
  `"tool_calls"`), reconstructed from the normalized message status
- `gen_ai.response.time_to_first_token` — for streaming calls, seconds from the
  request start to the first streamed chunk. A `gen_ai.first_token` span event is
  also recorded at that moment so backends can place it on the trace timeline.
- `gen_ai.tool.name`, `gen_ai.tool.call.id`, `gen_ai.tool.type`, `gen_ai.tool.description`
- `gen_ai.agent.name` (the chain type), `gen_ai.conversation.id` (see below)

> #### How request parameters are captured {: .info}
>
> Each chat model's standard parameters (temperature, max tokens, top-p, seed, …)
> are read from the model struct by conventional field name and carried on the
> telemetry event as `:request_options`, then mapped to `gen_ai.request.*`. A
> parameter the model doesn't set — or exposes under a non-standard field name —
> is simply omitted, so treat the set as a best-effort subset.

> #### `gen_ai.response.model` reflects the requested model {: .info}
>
> LangChain's telemetry carries the model you requested, not the exact model id
> the provider returns. `gen_ai.response.model` is therefore set to the request
> model — when a provider echoes a more specific id (e.g. `gpt-4o` →
> `gpt-4o-2024-08-06`), that specific id is not currently captured. Use
> `gen_ai.request.model` as the source of truth for the model.

Provider names are normalized to the OTel registry — e.g. `"google"` becomes
`gcp.gemini`, `"vertex_ai"` becomes `gcp.vertex_ai`, `"xai"` becomes `x_ai`, and
`"mistralai"` becomes `mistral_ai`. Unknown providers pass through unchanged. See
`LangChain.OpenTelemetry.ProviderMapping`.

### Grouping traces into conversations

LangChain has no first-class conversation id, so `gen_ai.conversation.id` (which
backends use to group a multi-turn session's traces) is read from your chain's
`custom_context`. Set a `:conversation_id` key — or reuse the
`:langfuse_session_id` you may already set — and it lands on the root chain span:

```elixir
chain
|> Map.put(:custom_context, %{conversation_id: session_id})
```

### Errors

When an operation raises, its span is closed with an error status, the exception
is recorded on the span, and an `error.type` attribute (the exception's module
name) is set so backends can group and filter by error kind.

### Configuration options

All options are passed to `setup/1` and default to the safest choice. See
`LangChain.OpenTelemetry.Config`.

| Option | Default | Effect |
|--------|---------|--------|
| `:capture_input_messages` | `false` | Serialize input messages into `gen_ai.input.messages`. **May contain PII.** |
| `:capture_output_messages` | `false` | Serialize the response into `gen_ai.output.messages`. **May contain PII.** |
| `:capture_tool_arguments` | `false` | Record tool call arguments into `gen_ai.tool.call.arguments`. |
| `:capture_tool_results` | `false` | Record tool return values into `gen_ai.tool.call.result`. |
| `:enable_metrics` | `true` | Re-emit duration and token-usage metric events (see below). |

```elixir
# Trace latency and token usage, but never record message content (default):
LangChain.OpenTelemetry.setup()

# Full-fidelity tracing, including message content — only where PII is acceptable:
LangChain.OpenTelemetry.setup(
  capture_input_messages: true,
  capture_output_messages: true,
  capture_tool_arguments: true,
  capture_tool_results: true
)
```

Message and tool-content capture is **off by default** precisely because those
attributes can carry sensitive data. Turn them on deliberately, and only where
your tracing backend and data-retention policy allow it.

### Metrics require a consumer

> #### Important {: .warning}
>
> `enable_metrics: true` does **not** record OpenTelemetry histograms on its own.
> It re-emits LangChain telemetry as two intermediary `:telemetry` events:
>
> - `[:langchain, :otel, :operation, :duration]` — `%{duration_s: float()}`
> - `[:langchain, :otel, :token, :usage]` — `%{tokens: integer()}`, tagged with
>   `gen_ai.token.type` of `"input"` or `"output"`
> - `[:langchain, :otel, :operation, :time_to_first_token]` — `%{duration_s: float()}`,
>   emitted once per streaming LLM call (aligns with the semantic-convention
>   `gen_ai.server.time_to_first_token` metric)
>
> To turn these into real metrics, attach a consumer such as `Telemetry.Metrics`
> (with an OpenTelemetry reporter), `PromEx`, or equivalent. Without a consumer,
> `enable_metrics: true` has no observable effect.

Example wiring with `Telemetry.Metrics`. Name the metrics with the canonical
GenAI semantic-convention names (`gen_ai.client.operation.duration` and
`gen_ai.client.token.usage`) so they line up with other OTel GenAI
instrumentation, and configure the spec's recommended histogram buckets:

```elixir
defmodule MyApp.Telemetry do
  import Telemetry.Metrics

  # Recommended explicit bucket boundaries from the GenAI semantic conventions.
  @duration_buckets [0.01, 0.02, 0.04, 0.08, 0.16, 0.32, 0.64, 1.28, 2.56, 5.12,
                     10.24, 20.48, 40.96, 81.92]
  @token_buckets [1, 4, 16, 64, 256, 1024, 4096, 16_384, 65_536, 262_144,
                  1_048_576, 4_194_304, 16_777_216, 67_108_864]

  def metrics do
    [
      distribution("gen_ai.client.operation.duration",
        event_name: [:langchain, :otel, :operation, :duration],
        measurement: :duration_s,
        unit: :second,
        reporter_options: [buckets: @duration_buckets],
        tags: [:"gen_ai.operation.name", :"gen_ai.provider.name", :"gen_ai.request.model"]
      ),
      distribution("gen_ai.client.token.usage",
        event_name: [:langchain, :otel, :token, :usage],
        measurement: :tokens,
        reporter_options: [buckets: @token_buckets],
        tags: [:"gen_ai.operation.name", :"gen_ai.provider.name", :"gen_ai.token.type"]
      )
    ]
  end
end
```

> #### Bucket configuration is reporter-specific {: .info}
>
> `:reporter_options` is honored by some reporters directly; for the
> OpenTelemetry reporter, explicit histogram buckets are typically set through an
> OTel **View** on the instrument name. Either way, use the boundaries above so
> LangChain's histograms match the semantic-convention recommendation.

See `LangChain.OpenTelemetry.MetricsHandler` for the exact event shapes.

### Suppressing traces for utility chains

Some chains — a translation pass, a title generator, a routing decision — are
implementation details you may not want cluttering your traces. Wrap them in
`without_tracing/1` and no spans are exported — and, when metrics are enabled, no
duration/token metric events are emitted — for anything inside:

```elixir
LangChain.OpenTelemetry.without_tracing(fn ->
  {:ok, updated_chain} = LLMChain.run(title_chain)
  updated_chain
end)
```

Suppression is scoped to the calling process. Work that runs in a separate
process spawned inside the block (e.g. an `async: true` tool) does not inherit
it, the same way OTel span context does not cross a process boundary.

### Async tools

Tools declared with `async: true` execute in a separate `Task` process. The
OpenTelemetry context lives in the parent process's dictionary and is **not**
inherited by a spawned process, so an async tool's span would otherwise become
its own root span instead of a child of the chain span.

For tools run by `LLMChain`'s built-in tool executor this is now handled
**automatically**: the chain captures the current OpenTelemetry context before
spawning each async tool `Task` and re-attaches it inside, so async tool spans
nest under the chain span with no extra work. (The propagation is a no-op unless
you have the OpenTelemetry dependencies installed.)

If you spawn your own processes that run LangChain operations, the same rule
applies to you — the context doesn't cross the process boundary by itself.
Capture `OpenTelemetry.Ctx.get_current/0` before spawning and
`OpenTelemetry.Ctx.attach/1` inside the spawned process. The
`:on_tool_pre_execution` callback (which fires inside the spawned process) is one
convenient place to do this.

## Langfuse integration

[Langfuse](https://langfuse.com/) ingests OpenTelemetry traces via its
OTLP-compatible endpoint. Point the OTel exporter at your Langfuse instance:

```elixir
# config/runtime.exs
config :opentelemetry_exporter,
  otlp_protocol: :http_protobuf,
  otlp_endpoint: "https://your-langfuse-host/api/public/otel",
  otlp_headers: [
    {"Authorization", "Basic " <> Base.encode64("pk-lf-...:sk-lf-...")}
  ]
```

### Trace-level attributes via `custom_context`

Set well-known keys on your chain's `custom_context` to propagate Langfuse trace
attributes onto the root chain span:

```elixir
chain =
  %{llm: llm, messages: messages}
  |> LLMChain.new!()
  |> Map.put(:custom_context, %{
    langfuse_user_id: current_user.id,
    langfuse_session_id: session_id,
    langfuse_tags: ["production", "v2"],
    langfuse_metadata: %{env: "prod", feature: "chat"}
  })
```

| `custom_context` key | Span attribute |
|----------------------|----------------|
| `:langfuse_trace_name` | `langfuse.trace.name` |
| `:langfuse_user_id` | `langfuse.user.id` |
| `:langfuse_session_id` | `langfuse.session.id` |
| `:langfuse_tags` | `langfuse.trace.tags` (comma-joined) |
| `:langfuse_metadata` | `langfuse.trace.metadata.*` (flattened) |

> #### `custom_context` is shared with tools {: .info}
>
> `custom_context` is your own data map — it's also passed to your tool functions
> and surfaced on tool/chain telemetry events. The `langfuse_*` keys shown here
> are simply reserved names the tracing layer recognizes; you can mix them with
> your application's own context freely.

## Robustness

The span handler traps and logs any exception it raises rather than letting it
propagate. This matters because `:telemetry` permanently detaches a handler that
raises — a single bad payload would otherwise silently disable tracing VM-wide for
the rest of the run. Instead, that one event is skipped and tracing stays
attached.

## See also

- `LangChain.Telemetry` — the full event catalog and metadata shapes
- `LangChain.OpenTelemetry` — setup, teardown, and `without_tracing/1`
- `LangChain.OpenTelemetry.Config` — every configuration option
- `LangChain.OpenTelemetry.MetricsHandler` — emitted metric event shapes
- `LangChain.OpenTelemetry.ProviderMapping` — provider name normalization
