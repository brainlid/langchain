# Changelog

## v0.8.6

### Fixed

- **`DataExtractionChain` accepts a single object in the extraction tool `info` argument**: The tool schema describes `info` as a JSON array of rows, but some models return one object when a single entity is extracted, which previously surfaced as `{:error, "Unexpected response…"}`. `run/4` now normalises a lone map to a one-element list so callers consistently receive `{:ok, [row | rows]}`. Adds public `normalize_extraction_info/1` for coercion (and unit tests) without mocking the LLM.

## v0.8.5

A small reliability and observability release. Streaming runs now recover from delta-conversion errors instead of failing the chain outright, and a new tool callback fires inside the per-tool process so per-process context (tenancy, OTel, Sentry) can be re-applied across the async Task boundary.

### Added

- **`:on_tool_pre_execution` chain callback**: New callback that fires inside the process that will actually run the tool, immediately before the tool function is invoked. For `async: true` tools it fires inside the spawned `Task.async/1`; for `async: false` tools and tools run through `execute_tool_calls_with_decisions/3` it fires in the chain's own process. Use this hook (instead of `:on_tool_execution_started`, which always fires in the parent chain process) when you need to re-apply per-process state — tenant context, OpenTelemetry spans, Sentry scope, etc. — that does not propagate across the async boundary on its own. https://github.com/brainlid/langchain/pull/530

### Fixed

- **Streaming delta-conversion errors are now retried like transport errors**: When `delta_to_message_when_complete/1` returned `{:error, chain, reason}` (for example, `"delta_conversion_failed"` from invalid streamed JSON), `LLMChain` previously surfaced the error immediately and abandoned the run. The chain now routes that failure through the same path as other LLM errors: it fires `:on_llm_error`, increments `current_failure_count`, and recurses into `do_run/1` if retries remain. On final exhaustion the original reason is preserved (rather than being rewritten to `"exceeded_failure_count"`), so downstream apps that pattern-match on the reason to render user-facing copy continue to work. https://github.com/brainlid/langchain/pull/528

## v0.8.4

A security-focused release. Fixes an API-key leak in `ChatGrok` verbose logging, hardens an atom-table exposure in `ChatReqLLM`, documents `PromptTemplate`'s EEx trust boundary, and wires `mix sobelow` into `mix precommit`.

### Added

- **`PromptTemplate` EEx security documentation**: Added a security admonition to `PromptTemplate` and the raw-text constructors warning that `text` is evaluated as EEx. Developer-controlled templates are safe; user- or LLM-controlled strings are not. https://github.com/brainlid/langchain/pull/526
- **`mix sobelow` in `mix precommit`**: Sobelow runs between `format` and `test` with a committed `.sobelow-conf`. Reviewed call sites carry inline `# sobelow_skip` annotations. https://github.com/brainlid/langchain/pull/526

### Changed

- **`ChatReqLLM.merge_provider_opts/2` uses `String.to_existing_atom/1`**: Unknown provider-opt keys now raise `ArgumentError` instead of growing the atom table. https://github.com/brainlid/langchain/pull/526

### Fixed

- **`ChatGrok` verbose logging leaked the API key**: An unconditional `IO.inspect/1` ran before the redaction branch, so `verbose_api: true` printed the raw `Bearer` token to stdout. Rotate any Grok keys that may have been captured in logs. https://github.com/brainlid/langchain/pull/526

## v0.8.3

### Added

- **`ChatOllamaAI`: native `:format` (structured outputs) and image support**: Two gaps in the Ollama chat model are now closed.
  - `:format` accepts either `"json"` (plain JSON mode) or a JSON Schema map, and is forwarded as the request's top-level `format` field. Schema-enforced generation is handled server-side by Ollama, mirroring what `/api/generate` has always supported.
  - User messages containing `:image` `ContentPart`s now have their base64 payloads split out of the message content and re-attached as Ollama's native top-level `images` array on the message. Multiple image parts are preserved in order. `:image_url` parts raise a clear error because Ollama has no server-side URL fetcher — callers must fetch bytes themselves and pass them as `:image` parts.
  - Prior behavior dropped image parts silently via `ContentPart.parts_to_string/1` and had no way to request structured output, forcing users to route through `ChatOpenAI` against Ollama's `/v1/chat/completions` endpoint — which doesn't reliably enforce schemas. https://github.com/brainlid/langchain/pull/520

### Fixed

- **Cross-model message serialization for `:thinking` and `:unsupported` content parts**: When conversation history crosses model providers (e.g. a Claude response with extended thinking is replayed to an OpenAI-compatible endpoint, or vice versa), the receiving model's serializer could encounter `ContentPart` types it doesn't recognize.
  - `ChatAnthropic.content_parts_for_api/1` now filters out `nil` entries after mapping, preventing Anthropic from rejecting requests with unsigned or unrecognised `:thinking` parts.
  - `ChatAwsMantle.strip_thinking_parts/1` now also strips `:unsupported` parts (in addition to `:thinking`), so Anthropic `redacted_thinking` blocks don't crash the Mantle serializer when forwarded as conversation history. https://github.com/brainlid/langchain/pull/523

## v0.8.2

This release introduces **experimental** support for AWS Bedrock's Mantle endpoint via a new `ChatAwsMantle` chat model. Mantle is AWS's OpenAI-compatible gateway for third-party Bedrock models, so one new module unlocks a whole family of providers at once: Moonshot's Kimi K2 line, OpenAI's gpt-oss series, and any models AWS adds in the future.

### Added

- **`ChatAwsMantle` chat model (experimental)**: New module for AWS Bedrock's Mantle endpoint. Supports both Bearer auth (Bedrock API key) and AWS SigV4 (IAM credentials via a zero-arity `:credentials` function). Region-aware URL building derives `https://bedrock-mantle.{region}.api.aws/v1/chat/completions` from `:region`. Reasoning extraction surfaces Mantle's `message.reasoning` / `delta.reasoning` field as `:thinking` `ContentPart`s on the assistant message (including streaming), so reasoning-model output renders the same way as Anthropic extended thinking. Streaming (with separate reasoning and content deltas), tool calling, K2.5 multimodal input, and the OpenAI-standard `:reasoning_effort` field all work. Defaults to `receive_timeout: 120_000` and `max_tokens: 4096` to bound Mantle's intermittent slow-starts and Kimi's occasional token-repetition loops. Verified against `moonshotai.kimi-k2-thinking`, `moonshotai.kimi-k2.5`, and `openai.gpt-oss-120b` https://github.com/brainlid/langchain/pull/521

### Changed

- **`MessageDelta` merges batched tool-call fragments**: The tool-call merge path now folds over every fragment in a chunk's `tool_calls` array rather than assuming one fragment per chunk. Most providers emit one fragment per chunk (no behavior change); gpt-oss-120b on AWS Mantle batches multiple argument fragments per SSE event, and this generalization handles either batching style correctly https://github.com/brainlid/langchain/pull/521
- **`LLMChain` `max_runs` error includes the configured value**: When the chain halts because the `max_runs` ceiling was reached, the returned error now reports the configured limit so callers can log or surface the exact threshold that was hit. Additional test coverage was also added around this mode's termination paths https://github.com/brainlid/langchain/pull/519

### Fixed

- **`ContentPart.parts_to_string/2` tolerates `nil` entries**: When `MessageDelta.merge_content_part_at_index/3` pads `merged_content` with `nil` (common mid-stream when reasoning lands at index 0 and visible content at index 1), the string-serializer now filters those nils instead of raising `BadMapError` on the first mid-stream read https://github.com/brainlid/langchain/pull/521

## v0.8.1

A small follow-up to v0.8.0. The headline change is improved sub-agent cancellation: tools that spawn sub-agents (or any other long-running side-effect) now receive the originating `tool_call_id` in their execution context, making it possible to correlate the spawned work back to the tool call and update its status when the user cancels.

### Added

- **`tool_call_id` available in tool execution context**: When `LLMChain` executes a tool, the call's `tool_call_id` is now injected into the tool's `context` map under the `:tool_call_id` key. Tools that spawn sub-agents or other asynchronous side-effects can use this to correlate the spawned work back to the originating tool call, enabling a cleaner cancellation UX where a cancelled sub-agent can mark its tool call as cancelled instead of leaving it dangling https://github.com/brainlid/langchain/pull/514

### Changed

- **Relaxed `req_llm` dependency constraint**: The optional `req_llm` dependency now uses `>= 1.6.0` instead of `~> 1.6`, allowing host applications to pull in newer major versions of `req_llm` (such as 2.x) without waiting on a LangChain release. Because `req_llm` is `optional: true`, applications that use `ChatReqLLM` should pin the version they want in their own `mix.exs` https://github.com/brainlid/langchain/pull/513
- **Hardened GitHub Actions CI configuration**: Applied [zizmor](https://github.com/woodruffw/zizmor) security recommendations to the Elixir workflow and added a `dependabot.yml` for automated workflow updates. No effect on library users; relevant for contributors running CI https://github.com/brainlid/langchain/pull/515

## v0.8.0

**Breaking changes** in this release. Delta-related functions in `LLMChain` now return `{:ok, chain}` / `{:error, chain, reason}` tuples instead of bare chain structs. See upgrade guide below.

This release greatly improves the stability of the library. There were long-standing issues where errors were difficult to detect or handle, and this release is focused on improving error detection, insight, and the general stability of longer running multi-turn agents.

In particular, a validation was introduced some time back when trying to deal with older Mistral models that was causing problems for modern LLMs. It is valid for an LLM to return an empty `MessageDelta` (no content and no tool calls) when closing out an exchange where it has nothing else to say. This commonly happened when the final action was a tool call and result with nothing left to add. The overly strict validation was rejecting these as errors, causing difficult to diagnose, intermittent failures.

That long-standing issue has finally been resolved! If you've had stability issues with the library in the past when building agents or working with multi-turn LLM conversations, this release is worth evaluating. With the empty delta issue resolved, improved error detection and handling within the streaming pipeline, and the new error-focused callbacks (`on_error` and `on_llm_error`), it's a great time to try it out.

### Upgrading from v0.7.0 - v0.8.0

#### Breaking: Delta functions return `:ok`/`:error` tuples

The following `LLMChain` functions now return `{:ok, chain}` or `{:error, chain, reason}` instead of a bare `%LLMChain{}` struct:

- `apply_deltas/2`
- `merge_deltas/2` (returns `%LLMChain{}` on success, `{:error, chain, reason}` on failure)
- `delta_to_message_when_complete/1`

If you call these functions directly, update your pattern matches:

```elixir
# Before (v0.7.0)
chain = LLMChain.apply_deltas(chain, deltas)
chain = LLMChain.delta_to_message_when_complete(chain)

# After (v0.8.0)
{:ok, chain} = LLMChain.apply_deltas(chain, deltas)
{:ok, chain} = LLMChain.delta_to_message_when_complete(chain)
```

These changes allow errors during streaming delta processing (e.g., failed delta-to-message conversion) to propagate as structured errors rather than being silently swallowed.

### Added

- **`on_error` and `on_llm_error` callbacks**: Two new chain callbacks for error observability. `on_llm_error` fires on every individual LLM call failure (including transient ones that may be retried or recovered via fallbacks). `on_error` fires exactly once when the chain has exhausted all recovery options and is returning a terminal error to the caller https://github.com/brainlid/langchain/pull/511
- **Top-level `cache_control` for `ChatAnthropic`**: New `cache_control` field on the `ChatAnthropic` struct enables Anthropic's automatic caching feature, which applies cache breakpoints to the last cacheable block in each request. This is the simplest way to enable prompt caching for multi-turn conversations without manual breakpoint management. Existing `cache_messages` option is clarified as legacy client-side behavior https://github.com/brainlid/langchain/pull/509

### Changed

- **Delta functions return `:ok`/`:error` tuples** (breaking): `apply_deltas/2`, `delta_to_message_when_complete/1`, and `merge_deltas/2` now return structured result tuples, enabling proper error propagation through the streaming pipeline. Failed delta-to-message conversions now return `{:error, chain, %LangChainError{type: "delta_conversion_failed"}}` instead of silently resetting streaming state https://github.com/brainlid/langchain/pull/511
- **Removed `validate_not_empty` from `MessageDelta.to_message/1`**: Empty assistant deltas (no content, no tool calls) are no longer rejected during delta-to-message conversion. This validation was overly strict and could mask legitimate responses https://github.com/brainlid/langchain/pull/511

### Fixed

- **Compiler warnings with optional `mint_web_socket` dependency**: Fixed warnings in `ChatOpenAIResponses` and `WebSocket` modules when the optional `mint_web_socket` library is not included as a dependency https://github.com/brainlid/langchain/pull/510

## v0.7.0

### Changed

- **Req-level HTTP retry disabled by default**: All chat model and image generator modules now set `retry: false` on Req HTTP requests. Previously, Req's `retry: :transient` (up to 3 retries) compounded with LangChain's own connection retry, potentially causing up to 12 HTTP requests per call. Server errors (503/429) now bubble up immediately to the caller, which is the correct behavior for applications using job queues like Oban with proper backoff strategies. The `req_config` option can still be used to re-enable Req retry if needed https://github.com/brainlid/langchain/pull/504

### Added

- **Configurable `retry_count` on chat models**: All chat model and image generator structs now expose a `retry_count` field (default: `2`) controlling how many times LangChain retries on connection errors. Set `retry_count: 0` to disable retries entirely and let your application's job queue handle retry strategy. Default behavior is unchanged (1 initial attempt + 2 retries = 3 total requests) https://github.com/brainlid/langchain/pull/505
- **WebSocket transport for `ChatOpenAIResponses`**: New optional WebSocket transport allows requests over a persistent connection instead of HTTP. Includes a generic `LangChain.WebSocket` client GenServer built on `mint_web_socket`. Use `connect_websocket!/1` and `disconnect_websocket!/1` to manage the lifecycle. Best suited for short-lived, synchronous sessions -- not for long-lived agents or HITL workflows https://github.com/brainlid/langchain/pull/497
- **Streaming tool calls for `ChatOllamaAI`**: Tool calls in streaming responses are now correctly handled instead of being silently dropped. Supports both complete and incomplete tool call responses https://github.com/brainlid/langchain/pull/498
- **Regression tests for DeepSeek streaming token usage**: Added test coverage for the DeepSeek streaming format where usage data arrives bundled in the final chunk alongside non-empty choices, verifying existing code handles both DeepSeek and OpenAI formats correctly https://github.com/brainlid/langchain/pull/499

### Fixed

- **`{:interrupt, ...}` and `{:pause, ...}` with fallback LLMs**: These valid execution results from HITL middleware now pass through `try_chain_with_llm/4` correctly instead of causing a `CaseClauseError` or being incorrectly retried on fallback models. Fallback models and HITL interrupts are no longer mutually exclusive https://github.com/brainlid/langchain/pull/502
- **Streaming `overloaded_error` from Anthropic API**: When Anthropic sends an `overloaded_error` as an SSE event during streaming on a 200 connection, it is now extracted and returned as a proper `{:error, %LangChainError{}}` instead of causing a crash. The `"overloaded_error"` type is also now recognized by `retry_on_fallback?/1` for fallback model retry https://github.com/brainlid/langchain/pull/500

## v0.6.3

### Added

- **ChatOpenAI Logprobs**: Added `logprobs` and `top_logprobs` parameters to `ChatOpenAI`, allowing users to request log probability information from the OpenAI API. When present, logprobs data is surfaced in message/delta metadata under the `"logprobs"` key. Supports both streaming and non-streaming modes https://github.com/brainlid/langchain/pull/494
- **LangChain.Trajectory**: New `LangChain.Trajectory` module for easier evaluation of agents, providing a structured way to assess agent execution paths https://github.com/brainlid/langchain/pull/481
- **ChatVertexAI Multimodal Tool Results**: Tool results in `ChatVertexAI` now support multimodal content including images, files (base64-encoded PDFs, etc.), JSON responses, and display names https://github.com/brainlid/langchain/pull/491

### Changed

- **Reduced Logger.error usage**: Replaced excessive `Logger.error` calls across the library with a more nuanced approach — errors already captured in return values no longer log redundantly, and catch-all/rescue clauses now use `Logger.warning` with lazy evaluation. This gives library consumers control over their logging levels https://github.com/brainlid/langchain/pull/492

### Fixed

- **Mistral responses without tool_calls**: Fixed crash when Mistral (or Azure-hosted Mistral) returns a complete message without a `"tool_calls"` key. The response previously fell through to the catch-all error handler https://github.com/brainlid/langchain/pull/495
- **Tool schema compatibility**: Added `additionalProperties: false` to tool parameter schemas for Anthropic API compatibility, which strictly requires this field on all object-type schemas https://github.com/brainlid/langchain/pull/490
- **Azure streaming keepalive**: Fixed crash caused by Azure OpenAI keepalive SSE events during long-running streaming responses, which were not matched by any `do_process_response/2` clause https://github.com/brainlid/langchain/pull/485
- **Empty/unexpected LLM streaming responses**: Added catch-all clauses in `do_run/1` for `{:ok, []}` and other unrecognized response formats (e.g., from thinking/reasoning models like Qwen3 and DeepSeek-R1 that emit `<think>` tokens), returning typed `LangChainError` instead of crashing with `CaseClauseError` https://github.com/brainlid/langchain/pull/484
- **Malformed tool call JSON loop**: Added regression tests verifying that truncated/malformed JSON in `tool_call.arguments` during streaming is properly cleared, preventing infinite loops https://github.com/brainlid/langchain/pull/493

## v0.6.2

### Upgrading from v0.6.1 - v0.6.2

#### Developer Environment: `.envrc` → `.env`

The library now uses [dotenvy](https://hex.pm/packages/dotenvy) to automatically load API keys from a `.env` file when running `mix test` or starting `iex -S mix`. This replaces the previous `direnv`/`.envrc` workflow.

**Who is affected**: Developers running live tests locally.

**How to migrate**:

```bash
# From the my_langchain/ directory:
cp .envrc .env
```

That's it. The `.env` file is gitignored and will be loaded automatically — no shell tool or `source .envrc` step required. AI coding assistants running `mix test` will also pick up the keys without any additional configuration.

The untracked `.envrc` file remains for anyone who prefers to continue using `direnv`. The new `.env.example` is the canonical starting point:

```bash
cp .env.example .env
# Populate .env with your API keys
```

### Added

- **ChatReqLLM**: New experimental `LangChain.ChatModels.ChatReqLLM` adapter that delegates HTTP, authentication, and provider encoding to the [`req_llm`](https://hex.pm/packages/req_llm) library. A single `"provider:model_id"` string (e.g. `"anthropic:claude-haiku-4-5"`, `"openai:gpt-4o"`, `"ollama:llama3"`) unlocks 20+ providers without requiring per-provider adapter code. Supports streaming, tool use, multi-modal content, token usage, `serialize_config/1`, and `restore_from_map/1` https://github.com/brainlid/langchain/pull/486
- **dotenvy dependency**: Added `dotenvy ~> 1.1` for automatic `.env` file loading in dev/test. The test suite now calls `Dotenvy.source!/1` before reading API keys, so no shell setup is needed to run live tests
- **`.env.example`**: New template file (dotenv format, no `export` prefix) for setting up local API keys

## v0.6.1

### Added

- **ModelsLabImage Provider**: New `LangChain.Images.ModelsLabImage` module for text-to-image generation via the ModelsLab REST API, supporting Flux, SDXL, and community models https://github.com/brainlid/langchain/pull/468
- **ChatAnthropic Structured Output**: Native structured output support via `json_response` and `json_schema` fields, using Anthropic's `output_config.format` API https://github.com/brainlid/langchain/pull/474
- **ChatAnthropic file_id Support**: `ContentPart` `:file` and `:image` types can now reference files uploaded via Anthropic's Files API using `type: :file_id` option https://github.com/brainlid/langchain/pull/475
- **Gemini Inline Data for PDF/CSV**: Added `inline_data` support for PDF and CSV content parts in `ChatGoogleAI` and `ChatVertexAI` https://github.com/brainlid/langchain/pull/478
- **ChatOpenAIResponses Verbosity**: Added `verbosity` parameter to control response length via the OpenAI Responses API `text` configuration https://github.com/brainlid/langchain/pull/470
- **Tool Result Interrupt/Resume**: `ToolResult` gains `is_interrupt` and `interrupt_data` fields, allowing tools to return `{:interrupt, message, data}` to pause execution for external input (e.g., Human-in-the-Loop approval). Includes `Message.replace_tool_result/3` and `LLMChain.replace_tool_result/3` for resuming with completed results, and a new `on_tool_interrupted` callback https://github.com/brainlid/langchain/pull/479

### Changed

- **Streaming Error Handling**: `LLMChain.merge_delta/2` now gracefully handles any `LangChainError` received during streaming (content moderation, etc.) instead of only `"overloaded"` errors. Cancelled messages store the error in `metadata[:streaming_error]` for higher layers to detect. `MessageDelta.merge_delta/2` absorbs `{:error, _}` tuples mid-stream without crashing https://github.com/brainlid/langchain/pull/480

### Fixed

- **ChatOpenAIResponses**: Fixed verbosity parameter to be passed inside the `text` parameter as required by the API, and improved error handling for malformed API responses https://github.com/brainlid/langchain/pull/476
- **Message Processors**: Fixed `FunctionClauseError` in `run_message_processors` when assistant messages have nil content (e.g., Groq tool-call-only responses) by using `content_to_string` which handles nil, string, and list content types https://github.com/brainlid/langchain/pull/473

## v0.6.0

### Breaking Changes

- **DeepResearch `ResearchResult.Source` removed**: The inline `Source` embedded schema in `ResearchResult` has been replaced with the unified `Citation`/`CitationSource` structs. See migration steps below.
- **Perplexity response format**: Response content is now returned as a list of `ContentPart` structs (consistent with other providers) rather than a plain string.
- **Google AI `groundingMetadata` restructured**: Raw grounding metadata has moved from top-level keys in `message.metadata` to `message.metadata["grounding_metadata"]`.

### Upgrading from v0.5.2 - v0.6.0

**ContentPart `citations` field (all providers)**: `ContentPart` structs returned from any provider may now have a non-empty `citations` field (list of `Citation` structs). Code that pattern-matches on the exact `ContentPart` struct shape should be unaffected since `citations` defaults to `[]`.

**Anthropic round-trip**: Code that sends assistant messages back to the Anthropic API will automatically include citations in the round-trip. No action required.

**Google AI `groundingMetadata`**: Code reading `message.metadata["groundingChunks"]` should update to `message.metadata["grounding_metadata"]["groundingChunks"]`. The structured citations on `ContentPart` are now the canonical representation.

**OpenAI Responses API**: `ContentPart` structs may now include citation data from annotations that were previously discarded. This is purely additive.

**Perplexity**: Code that expects `message.content` to be a string should update to handle `[%ContentPart{}]`.

**DeepResearch `ResearchResult.Source`**: Code referencing `ResearchResult.Source` must update to use `Citation` and `CitationSource`:
  - `%ResearchResult.Source{title: t, url: u}` → `%Citation{source: %CitationSource{title: t, url: u}}`
  - `source.title` → `citation.source.title`
  - `source.url` → `citation.source.url`
  - `source.snippet` → `citation.cited_text`
  - `source.start_index` → `citation.start_index`
  - `ResearchResult.source_urls/1` now filters out `nil` URLs (previously returned all, including nil)

### Added

- **Citation Support**: Unified citation abstraction shared across all providers https://github.com/brainlid/langchain/pull/461
  - New `LangChain.Message.Citation` struct with `cited_text`, `source`, `start_index`, `end_index`, `confidence`, and `metadata` fields
  - New `LangChain.Message.CitationSource` struct with `type` (`:web`, `:document`, `:place`), `title`, `url`, `document_id`, and `metadata` fields
  - `ContentPart` gains a `citations` field (virtual, defaults to `[]`) with merging support during streaming delta accumulation
  - Helper functions: `ContentPart.citations/1`, `ContentPart.has_citations?/1`, `Message.all_citations/1`
  - **Anthropic**: Non-streaming and streaming citation parsing, API round-trip serialization, all three citation types (`char_location`, `page_location`, `content_block_location`)
  - **OpenAI Responses API**: Annotation parsing for `url_citation` and `file_citation` types (not yet verified against live API)
  - **Google AI (Gemini)**: Grounding metadata decomposed into `Citation` structs by `partIndex` with confidence scores
  - **Perplexity**: `citations` URLs and `search_results` parsed into `Citation` structs with `[N]` inline reference matching
  - **DeepResearch**: Migrated `ResearchResult` to use `Citation`/`CitationSource`; added `Citation.changeset/2` for Ecto `cast_embed/3`
- **Customizable Run Modes**: Extracted LLMChain execution modes into discrete, composable modules with a `Mode` behaviour https://github.com/brainlid/langchain/pull/469
  - Built-in modes: `WhileNeedsResponse`, `UntilSuccess`, `Step`, `UntilToolUsed`
  - Pipe-friendly API for composing custom modes
  - Backward compatible with existing `LLMChain.run/2` usage
- **Streaming Reasoning Callbacks**: Added `on_llm_reasoning_delta` callback for reasoning summary streaming from Azure OpenAI Responses API https://github.com/brainlid/langchain/pull/456
- **ChatVertexAI**: Added `req_config` option for custom Req configuration https://github.com/brainlid/langchain/pull/457

### Changed

- **Error Handling**: Consistently include `original` error info in "Unexpected response" errors across all providers https://github.com/brainlid/langchain/pull/392

### Fixed

- **ChatAnthropic**: Auth error (HTTP 401) for invalid API keys is now handled gracefully https://github.com/brainlid/langchain/pull/464
- **ChatAnthropic**: Fixed error handling regression introduced by PR #392, added additional catch for `{:error, thing}` shape and tracking of `original` error info on `LangChainError` https://github.com/brainlid/langchain/pull/462
- **LLMChain**: Fixed exception handling in `run_message_processors` — exceptions now correctly return 3-tuple expected by `process_message` https://github.com/brainlid/langchain/pull/460

## v0.5.2

### Changed

- **Tool Detection Improvements**: Simplified tool execution flow and enhanced UI feedback https://github.com/brainlid/langchain/pull/458
  - Removed special detection of malformed tool calls (partial rollback of https://github.com/brainlid/langchain/pull/449). The previous approach created a more complicated multi-path flow and could add internally defined user messages that appeared unexpectedly to users
  - Added `display_text` as a first-class attribute on `ToolCall` for better UI feedback and display control
  - Fixed early tool use detection and notification timing

### Fixed

- **ChatVertexAI**: Fixed tool calls for ChatVertexAI and added support for Gemini 3 models https://github.com/brainlid/langchain/pull/452
  - Added `thought_signature` support for Gemini 3 function calls
  - Removed unsupported "strict" field from function declarations for Vertex AI compatibility
  - Added Jason encoder derivation for `ContentPart` to ensure proper JSON serialization
  - Implemented `thoughtSignature` handling for Vertex AI tool calls

---

## v0.5.1

### Enhancements

**Enhanced Tool Execution Callbacks**: Added four new callbacks that provide granular visibility into the complete tool execution lifecycle:

1. **`:on_tool_call_identified`** - Fires as soon as a tool name is detected during streaming (before arguments are fully received). May not have `call_id` yet. Enables early UI feedback like "Searching web..." while the LLM is still streaming.

2. **`:on_tool_execution_started`** - Fires immediately before tool execution begins. Always has `call_id`. Allows tracking when actual execution starts.

3. **`:on_tool_execution_completed`** - Fires after successful tool execution with the result. Useful for logging, metrics, and updating UI state.

4. **`:on_tool_execution_failed`** - Fires when a tool call fails, is invalid, or is rejected during human-in-the-loop approval. Includes error details.

**Key distinction**: The early `:on_tool_call_identified` callback fires during streaming as soon as the tool name appears, while `:on_tool_execution_started` fires later when execution actually begins. This two-phase notification enables responsive UIs that can show immediate feedback.

**Example usage**:

```elixir
callbacks = %{
  on_tool_call_identified: fn _chain, tool_call, func ->
    # Show early UI feedback during streaming (tool args may be incomplete)
    IO.puts("Tool identified: #{func.display_text || tool_call.name}")
  end,
  on_tool_execution_started: fn _chain, tool_call, func ->
    # Update UI when execution actually begins (tool args complete)
    IO.puts("Executing: #{func.display_text || tool_call.name}")
  end,
  on_tool_execution_completed: fn _chain, tool_call, func, result ->
    # Handle successful execution
    IO.puts("Completed: #{tool_call.name}")
  end,
  on_tool_execution_failed: fn _chain, tool_call, func, error ->
    # Handle failures
    IO.puts("Failed: #{tool_call.name} - #{error}")
  end
}

chain = LLMChain.new!(%{
  llm: model,
  tools: [my_tool],
  callbacks: [callbacks]
})
```

**Additional improvements**:
- `MessageDelta` now tracks tool display information in metadata for better UI rendering
- Callbacks fire correctly across all execution paths (normal, async, HITL workflows)
- Internal tracking prevents duplicate identification notifications for the same tool call

**Non-breaking change**: Existing applications continue to work without modifications. All new callbacks are optional.

---

## v0.5.0

### Breaking Changes

**Elixir 1.17+ Required**: This release requires Elixir 1.17 or higher. The library uses the `get_in` macro which is only available in Elixir 1.17 onwards. Using Elixir 1.16 will throw a compile error. https://github.com/brainlid/langchain/pull/427

**Async Tool Timeout Default Changed**: The default timeout for async tools (tools with `async: true`) has changed from 2 minutes (`120_000` ms) to `:infinity` (no timeout).

### Upgrading from v0.4.1 - v0.5.0

#### Elixir Version Requirement

**What changed**: Minimum Elixir version is now 1.17.

**Who is affected**: Users running Elixir 1.16 or earlier.

**How to migrate**: Upgrade to Elixir 1.17 or later.

#### Async Tool Timeout Default Change

**What changed**: The default timeout for async tools (tools with `async: true`) has changed from 2 minutes (`120_000` ms) to `:infinity` (no timeout).

**Why**: The previous 2-minute default was problematic for human-interactive agents because:
- Web research tools often take 3-5+ minutes
- Sub-agent workflows can run indefinitely
- Deep analysis tools may need extended processing time
- Users observing the agent can manually stop it if needed

**Who is affected**: If your application relied on the implicit 2-minute timeout to catch runaway tools, you will need to explicitly set a timeout.

**How to migrate**:

1. **If you were happy with the 2-minute timeout**, add this to your `config/runtime.exs`:

       config :langchain, async_tool_timeout: 2 * 60 * 1000  # 2 minutes (previous default)

2. **If you set an explicit `async_tool_timeout`**, no changes needed - your explicit value is still respected.

3. **If the new `:infinity` default works for you** (human-interactive agents), no changes needed.

**Configuration precedence** (highest to lowest):
1. `LLMChain.async_tool_timeout` - explicit value on chain
2. `Agent.async_tool_timeout` - passed through to chain when building
3. `Application.get_env(:langchain, :async_tool_timeout)` - runtime config
4. Library default - `:infinity`

**Example configurations**:

    # Application-level (config/runtime.exs)
    config :langchain, async_tool_timeout: 5 * 60 * 1000  # 5 minutes

    # Agent-level
    {:ok, agent} = Agent.new(%{
      model: model,
      async_tool_timeout: 10 * 60 * 1000  # 10 minutes
    })

    # Chain-level
    {:ok, chain} = LLMChain.new(%{
      llm: model,
      async_tool_timeout: 35 * 60 * 1000  # 35 minutes for Deep Research
    })

### Added
- **Agent Framework Foundation**: Base work for new agent library with middleware-based architecture, including agent orchestration, state management, virtual filesystem, human-in-the-loop (HITL) workflows, sub-agents, summarization, and presence tracking https://github.com/brainlid/langchain/pull/442
- **ChatOpenAIResponses**: Added `req_config` option for custom Req configuration https://github.com/brainlid/langchain/pull/415
- **ChatOpenAIResponses**: Added reasoning/thinking events support https://github.com/brainlid/langchain/pull/421
- **ChatOpenAIResponses**: Added new reasoning effort values https://github.com/brainlid/langchain/pull/419
- **ChatOpenAIResponses**: Added stateful context support for Response API https://github.com/brainlid/langchain/pull/425
- **ChatVertexAI**: Added JSON schema support https://github.com/brainlid/langchain/pull/424
- **ChatVertexAI**: Added thinking configuration support https://github.com/brainlid/langchain/pull/423
- **ChatGoogleAI**: Added `thought_signature` support for Gemini 3 function calls https://github.com/brainlid/langchain/pull/431
- **ChatMistralAI**: Added support for parallel tool calls https://github.com/brainlid/langchain/pull/433
- **ChatMistralAI**: Added thinking content parts support https://github.com/brainlid/langchain/pull/418
- **ChatPerplexity and ChatMistralAI**: Added `verbose_api` field https://github.com/brainlid/langchain/pull/416
- **LLMChain**: Changed default `async_tool_timeout` from 2 minutes to `:infinity` https://github.com/brainlid/langchain/pull/442

### Changed
- **Dependencies**: Updated Elixir requirement to `~> 1.17` https://github.com/brainlid/langchain/pull/427
- **ChatOpenAIResponses**: Don't include `top_p` parameter for gpt-5.2+ models https://github.com/brainlid/langchain/pull/428

### Fixed
- **ChatDeepSeek**: Fixed UI bug in deepseek-chat model introduced by reasoning_content support https://github.com/brainlid/langchain/pull/429
- **Core**: Fixed missing error handling and fallback mechanism on server outages https://github.com/brainlid/langchain/pull/435
- **ChatOpenAIResponses**: Fixed image `file_id` content type handling https://github.com/brainlid/langchain/pull/438

---

## v0.4.1

### Added
- **ChatDeepSeek**: Added DeepSeek chat model integration with reasoning_content support (#394, #407)
- **ChatOpenAI**: Added strict tool use support (#301)
- **ChatOpenAI**: Added support for file_url with link to file (#395)
- **ChatAnthropic**: Added strict tool use support (#409)
- **ChatAnthropic**: Added support for file_url (#404)
- **ChatAnthropic**: Added PDF reading support via Anthropic API (#403)
- **ChatAnthropic**: Added `cache_messages` option to improve cache utilization (#398)
- **ChatAnthropic**: Added `req_opts` option for custom Req configuration (#408)
- **MessageDelta**: Added `MessageDelta.merge_deltas/2` function for merging multiple deltas (#401)
- **ChatAnthropic**: Added `disable_parallel_tool_use` tool_choice pass-through (#390)
- **Core**: Added multi-part tool responses support (#410)

### Changed
- **ChatOpenAI**: Enhanced OpenAI responses API (#391)
- **Dependencies**: Updated gettext requirement to `~> 1.0` (#393, #399)
- **Documentation**: Updated README install instructions

### Fixed
- **Core**: Fixed compiler typing warnings

---

## v0.4.0

### Added
- **ChatOpenAI**: Added support for json-schema in OpenAI responses API (#387)
- **Documentation**: Added AGENTS.md and CLAUDE.md file support (#385)
- **CI**: Added support for OTP 28 (#382)

### Changed
- **ChatOpenAI**: Enhanced OpenAI responses handling (#381)
- **Documentation**: Use moduledoc instead of doc for LLMChain documentation (#384)
- **Utils.ChainResult**: Added clarity to message stopped for length handling

### Fixed
- **ChatBumblebee**: Suppressed compiler warning messages when used as a dependency (#386)
- **Core**: Fixed Ecto field formatting

---

## v0.4.0-rc.3

### Added
- **ChatOrqAI**: Added Orq AI chat model support (#377)
- **ChatOpenAI**: Added OpenAI Deep Research integration (#336)
- **ChatOpenAI**: Added `parallel_tool_calls` option (#371)
- **ChatOpenAI**: Added `req_config` option for custom Req configuration (#376)
- **ChatOpenAI**: Added verbosity parameter support (#379)
- **ChatVertexAI**: Added support for native tool calls (#359)
- **ChatGoogleAI**: Added full thinking configuration support (#375)
- **Bedrock**: Added optional AWS session token handling in BedrockHelpers (#372)
- **LLMChain**: Added `should_continue?` function for automatic looping on mode `:step` (#361)
- **Core**: Added `retry_on_fallback?` to chat model definition and all models (#350)

### Fixed
- **Images**: Fixed handling of LiteLLM responses with null `b64_json` in OpenAI image generation (#368)
- **Core**: Fixed handling of missing `finish_reason` in streaming responses for LiteLLM compatibility (#367)
- **ChatGoogleAI**: Fixed error prevention from thinking content parts (#374)
- **ChatGoogleAI**: Fixed handling of Gemini's cumulative token usage (#373)

---

## v0.4.0-rc.2

### Added
- **ChatGrok**: Added xAI Grok chat model support (#338)
- **ChatGoogleAI**: Added thinking support (#354)
- **ChatGoogleAI**: Added `req_config` option for custom Req configuration (#357)
- **ChatOllamaAI**: Added missing `verbose_api` field for streaming compatibility (#341)
- **ChatVertexAI**: Added usage data to Message response metadata (#335)
- **Images**: Added support for `gpt-image-1` model in OpenAI image generation (#360)
- **LLMChain**: Added new run mode `:step` for step-by-step execution (#343)
- **LLMChain**: Added support for multiple tools in `run_until_tool_used` (#345)
- **OpenAI**: Added organization ID as a parameter for API requests (#337)
- New callback `on_llm_response_headers` supports receiving the full Req HTTP response headers for a request (#358)

### Changed
- **Bedrock**: Added OpenAI-compatible API compatibility (#356)
- **ChatAnthropic**: Expanded logging for API errors (#349)
- **ChatAnthropic**: Added transient Req retry support in stream mode (#329)
- **ChatGoogleAI**: Cleaned up MessageDelta handling (#353)
- **ChatOpenAI**: Only include "user" field with requests when a value is provided (#364)
- **Dependencies**: Updated gettext requirement to `~> 0.26` (#332)

### Fixed
- **ChatGoogleAI**: Handle responses with no content parts (#365)
- **ChatGoogleAI**: Prevent crash when ToolResult contains string content (#352)
- **Core**: Fixed issue with poorly matching list in case statements (#334)
- **Core**: Filter out empty lists in message responses (#333)

### Breaking Changes
- **ChatOllamaAI**: Fixed `stop` field type from `:string` to `{:array, :string}` to match Ollama API requirements. Previously, stop sequences were non-functional due to API type mismatch. Now accepts arrays like `["\\n", "Human:", "<|eot_id|>"]`. Empty arrays are excluded from API requests to preserve modelfile defaults (#342)

---

## v0.4.0-rc.1

---

### Breaking Changes
- ToolResult `content` now supports a list of ContentParts, not just strings. Functions can return a ToolResult directly for advanced control (e.g., cache control, processed_content).
- Expanded multi-modal support: messages and tool results can now include text, images, files, and thinking blocks as ContentParts.
- LLMChain: Added `async_tool_timeout` config; improved fallback and error handling.
- `LangChain.Function` changed the default for `async` to `false`. If you want async execution, set `async: true` explicitly when defining your function.
- The `on_llm_new_delta` callback now receives a list of `MessageDelta` structs instead of a single one. To merge the received deltas into your chain for display, use:

```elixir
updated_chain = LLMChain.merge_deltas(current_llm_chain, deltas)
```

### Upgrading from v0.4.0-rc.0 - v0.4.0-rc.1
- If you return a ToolResult from a function, you can now use ContentParts for richer responses. See module docs for details.
- If you use custom chunking logic, see the new tokenizer support in TextSplitter.
- If you are displaying streamed MessageDelta results using the `on_llm_new_delta` callback, you will need to update your callback function to expect a list of MessageDeltas and you can use the new `LLMChain.merge_deltas` function for merging them into your chain. The resulting merged delta can be used for display.

#### Model Compatibility
- The following models have been verified with this version:
  - ChatOpenAI
  - ChatAnthropic
  - ChatGoogleAI
- There are known broken live tests with Perplexity and likely others. Not all models are currently verified or supported in this release.

**Assistance is requested** for verifying/updating other models and their tests.

### Added
- Telemetry to `LLMChain.run_until_tool_used` for better observability.
- Google Gemini 2.0+ supports native Google Search as a tool.
- MistralAI: Structured output support.
- ChatGoogleAI: `verbose_api` option; updated default model to `gemini-2.5-pro`.
- TextSplitter: Added configurable tokenizer support for chunking by tokens, not just characters.

### Changed
- ChatOpenAI: Improved handling of ContentParts in ToolResults; better support for reasoning models and robust API options.
- ChatGoogleAI: Improved ToolResult handling for ContentParts; better error and token usage reporting.
- ChatAnthropic: Expanded prompt caching support and documentation; improved error and token usage handling.
- LLMChain: Improved fallback and error handling; added async tool timeout config.
- TextSplitter: Now supports custom tokenizers for chunking.

### Fixed
- ToolCalls: Fixed issues with nil tool_calls and tool call processing.
- Token Usage: Fixed token usage reporting for GoogleAI.
- Bedrock Stream Decoder: Fixed chunk order issue.

## v0.4.0-rc.0

This includes several breaking changes:

- Not all chat models are supported and updated yet. Currently only **OpenAI** and **Claude**
- Assistant messages are all assumed to be a list of `ContentPart` structs, supporting text, thinking, and more in the future like images
- A Message includes the TokenUsage in `Message.metadata.usage` after received.
- To display a MessageDelta as it is being streamed back, use `MessageDelta.merged_content`.

Use the v0.3.x releases for models that are not yet supported.

| Model | v0.3.x | v0.4.x |
|-------|---------|---------|
| OpenAI ChatGPT | ✓ | ✓ |
| OpenAI DALL-e 2 (image generation) | ✓ | ? |
| Anthropic Claude | ✓ | ✓ |
| Anthropic Claude (thinking) | X | ✓ |
| Google Gemini | ✓ | ✓ |
| Google Vertex AI | ✓ | ✓ |
| Ollama | ✓ | ? |
| Mistral | ✓ | X |
| Bumblebee self-hosted models | ✓ | ? |
| LMStudio | ✓ | ? |
| Perplexity | ✓ | ? |

### Upgrade from v0.3.3 to v0.4.x

As LLM services get more advanced, they have begun returning multi-modal responses. For some time, they have been accepting multi-modal requests, meaning an image and text could be submitted at the same time.

Now, LLMs have changed to return multi-modal responses. This means they may return text along with an image. This is currently most common with receiving a "thinking" response separate from their text response.

In an effort to provide a consistent interface to many different LLMs, now **all** message responses with content (text, image, thinking, etc.) will be represented as a list of `ContentPart` structs.

This is a breaking change and may require application updates to adapt.

### Message Changes

Where this was received before:

```elixir
%Message{content: "this is a string"}
```

This is received now:

```elixir
%Message{content: [%ContentPart{type: :text, content: "this is a string"}]}
```

This can be quickly turned back into plain text using `LangChain.Message.ContentPart.parts_to_string/1`.

It looks like this:
```elixir
message = %Message{content: [%ContentPart{type: :text, content: "this is a string"}]}
ContentPart.parts_to_string(message.content)
#=> "this is a string"
```

This also handles if multiple text content parts are received:
```elixir
message = %Message{content: [
  %ContentPart{type: :text, content: "this is a string"},
  %ContentPart{type: :text, content: "this is another string"},
]}
ContentPart.parts_to_string(message.content)
#=> "this is a string\n\nthisis another string"
```

For constructing your own messages, this is auto-converted for you:

```elixir
Message.new_user!("Howdy!")
#=> %Message{role: :user, content: [%ContentPart{type: :text, content: "Howdy!"}]}
```

This can also be constructed like this:

```elixir
Message.new_user!([ContentPart.text!("Howdy!")])
#=> %Message{role: :user, content: [%ContentPart{type: :text, content: "Howdy!"}]}
```

The change is more significant when handling an assistant response message.

### MessageDelta Changes

When streaming a response and getting back `MessageDelta`s, these now have a `merged_content` field that combines the different streamed back content types into their complete pieces. These pieces can represent different indexes in the list of received ContentParts.

See the MessageDelta module docs for more information on `merged_content`.

This is important because when needing to display the deltas as they are being received, it is now the `merged_content` field that should be used.

### TokenUsage

Another significant change is the moving of TokenUsage from a separated callback to being directly attached to a Message's `metadata`. Token usage is accumulated, as it is split out typically on the first and last delta's received.

After an LLMChain.run, the `updated_chain.last_message.metadata.usage` will contain the %TokenUsage{} information.

A related change was to move the TokenUsage callback from the OpenAI and Anthropic chat models to the LLMChain. This means the same event will fire, but it will fire when it's fully received and assembled.



## v0.3.3 (2025-03-17)

This is a milestone release before staring v0.4.0 which introduces breaking changes, but importantly adds support for "thinking" models.

### Added
- Added telemetry support https://github.com/brainlid/langchain/pull/284
- Added `LLMChain.run_until_tool_used/3` function https://github.com/brainlid/langchain/pull/292
- Support for file uploads with file_id in ChatOpenAI https://github.com/brainlid/langchain/pull/283
- Support for json_response in ChatGoogleAI https://github.com/brainlid/langchain/pull/277
- Support for streaming responses from Mistral https://github.com/brainlid/langchain/pull/287
- Support for file URLs in Google AI https://github.com/brainlid/langchain/pull/286
- Support for PDF content with OpenAI model https://github.com/brainlid/langchain/pull/275
- Support for caching tool results in Anthropic calls https://github.com/brainlid/langchain/pull/269
- Support for choosing Anthropic beta headers https://github.com/brainlid/langchain/pull/273

### Changed
- Fixed options being passed to the Ollama chat API https://github.com/brainlid/langchain/pull/179
- Fixed media URIs for Google Vertex https://github.com/brainlid/langchain/pull/242
- Fixed OpenAI verbose_api https://github.com/brainlid/langchain/pull/274
- Improved documentation for callbacks and content parts
- Upgraded gettext and migrated https://github.com/brainlid/langchain/pull/271

### Fixed
- Added validation to check if requested tool_name exists in chain
- Fixed various documentation issues and typos
- Fixed callback links in documentation

## v0.3.2 (2025-03-17)

### Added
- Support for Perplexity AI https://github.com/brainlid/langchain/pull/261
- Enable tool support for ollama (if the model supports it and only when not streaming) https://github.com/brainlid/langchain/pull/164
- Added `on_message_processed` callback when tool response is created: When a Tool response message is created, it already fired an on_tool_response_created, but it now also fires the more general on_message_processed, because a tool result can certainly be considered being processed. https://github.com/brainlid/langchain/pull/248
- Added Tool Calls and TokenUsage for Mistral.ai https://github.com/brainlid/langchain/pull/253
- Added `LangChain.TextSplitter` with character and recursive character splitting support https://github.com/brainlid/langchain/pull/256
- Add native tool functionality (e.g. `google_search` for Gemini) https://github.com/brainlid/langchain/pull/250

### Changes
- Improved System instruction support for Vertex AI https://github.com/brainlid/langchain/pull/260
- Redact api-key from models when logged https://github.com/brainlid/langchain/pull/266

## v0.3.1 (2025-02-05)

### Added
- Include stacktrace context in messages for caught exceptions from LLM functions & function callbacks. (#241)

### Changes
- Support LMStudio when using ChatOpenAI (#243)
- Fix issue with OpenAI converting an assistant message to JSON when the AI is talking while making tool calls in the same message (#245)

## v0.3.0 (2025-01-22)

No more breaking changes.

### Upgrading from v0.2.0 - v0.3.0

There were several breaking changes made in the different Release Candidates. All changes were kept. Refer to the CHANGELOG documentation for rc.0, rc.1 and rc.2 for specific examples and coverage on needed code updates.

### Added
- LLAMA 3.1 JSON tool call support with Bumblebee (#198)
- Raw field to TokenUsage (#236) - this returns the raw LLM token usage information, giving access to LLM-specific data.
- Prompt caching support for Claude (#226)
- Support for Ollama keep_alive API parameter (#237)
- Support for o1 OpenAI model (#234)
- Bumblebee Phi-4 support (#233)

### Changed
- Apply chat template from callback (#231)

## v0.3.0-rc.2 (2025-01-08)

### Breaking Changes

How LLM callbacks are registered has changed. The callback function's arguments have also changed.

Specifically, this refers to the callbacks:

- `on_llm_new_delta`
- `on_llm_new_message`
- `on_llm_ratelimit_info`
- `on_llm_token_usage`

The callbacks are still supported, but _how_ they are registered and the arguments passed to the linked functions has changed.

Previously, an LLM callback's first argument was the chat model, it is now the LLMChain that is running it.

A ChatModel still has the `callbacks` struct attribute, but it should be considered private.

#### Why the change
Having some callback functions registered on the chat model and some registered on the chain was confusing. What goes where? Why the difference?

This change moves them all to the same place, removing a source of confusion.

The primary reason for the change is that important information about the **context** of the callback event was not available to the callback function. Information stored in the chain's `custom_context` can be valuable and important, like a user's account ID, but it was not easily accessible in a callback like `on_llm_token_usage` where we might want to record the user's token usage linked to their account.

This important change passes the entire `LLMChain` through to the callback function, giving the function access to the `custom_context`. This makes the LLM (aka chat model) callback functions expect the same arguments as the other chain focused callback functions.

This both unifies how the callbacks operate and what data they have available, and it groups them all together.

#### Adapting to the change
A before example:

```elixir
llm_events = %{
  # 1st argument was the chat model
  on_llm_new_delta: fn _chat_model, %MessageDelta{} = delta ->
    # ...
  end,
  on_llm_token_usage: fn _chat_model, usage_data ->
    # ...
  end
}

chain_events = %{
  on_message_processed: fn _chain, tool_msg ->
    # ...
  end
}

# LLM callback events were registered on the chat model
chat_model = ChatOpenAI.new!(%{stream: true, callbacks: [llm_events]})

{:ok, updated_chain} =
  %{
    llm: chat_model,
    custom_context: %{user_id: 123}
  }
  |> LLMChain.new!()
  |> LLMChain.add_message(Message.new_system!())
  |> LLMChain.add_message(Message.new_user!("Say hello!"))
  # Chain callback events were registered on the chain
  |> LLMChain.add_callback(chain_events)
  |> LLMChain.run()
```

This is updated to: (comments highlight changes)

```elixir
# Events are all combined together
events = %{
  # 1st argument is now the LLMChain
  on_llm_new_delta: fn _chain, %MessageDelta{} = delta ->
    # ...
  end,
  on_llm_token_usage: fn %LLMChain{} = chain, usage_data ->
    # ... `chain.custom_context` is available
  end,
  on_message_processed: fn _chain, tool_msg ->
    # ...
  end
}

# callbacks removed from Chat Model setup
chat_model = ChatOpenAI.new!(%{stream: true})

{:ok, updated_chain} =
  %{
    llm: chat_model,
    custom_context: %{user_id: 123}
  }
  |> LLMChain.new!()
  |> LLMChain.add_message(Message.new_system!())
  |> LLMChain.add_message(Message.new_user!("Say hello!"))
  # All events are registered through `add_callback`
  |> LLMChain.add_callback(events)
  |> LLMChain.run()
```

If you still need access to the LLM in the callback functions, it's available in `chain.llm`.

The change is a breaking change, but should be fairly easy to update.

This consolidates how callback events work and them more powerful by exposing important information to the callback functions.

If you were using the `LLMChain.add_llm_callback/2`, the change is even easier:

From:
```elixir
  %{
    llm: chat_model,
    custom_context: %{user_id: 123}
  }
  |> LLMChain.new!()
  # ...
  # LLM callback events could be added later this way
  |> LLMChain.add_llm_callback(llm_events)
  |> LLMChain.run()
```

To:
```elixir
  %{
    llm: chat_model,
    custom_context: %{user_id: 123}
  }
  |> LLMChain.new!()
  # ...
  # Use the `add_callback` function instead
  |> LLMChain.add_callback(llm_events)
  |> LLMChain.run()
```

#### Details of the change
- Removal of the `LangChain.ChatModels.LLMCallbacks` module.
- The LLM-specific callbacks were migrated to `LangChain.Chains.ChainCallbacks`.
- Removal of `LangChain.Chains.LLMChain.add_llm_callback/2`
- `LangChain.ChatModels.ChatOpenAI.new/1` and `LangChain.ChatModels.ChatOpenAI.new!/1` no longer accept `:callbacks` on the chat model.
- Removal of `LangChain.ChatModels.ChatModel.add_callback/2`

### What else Changed
* add explicit message support in summarizer by @brainlid in https://github.com/brainlid/langchain/pull/220
* Change abacus to optional dep by @nallwhy in https://github.com/brainlid/langchain/pull/223
* Remove constraint of alternating user, assistant by @GenericJam in https://github.com/brainlid/langchain/pull/222
* Breaking change: consolidate LLM callback functions by @brainlid in https://github.com/brainlid/langchain/pull/228
* feat: Enable :inet6 for Req.new for Ollama by @mpope9 in https://github.com/brainlid/langchain/pull/227
* fix: enable verbose_deltas by @cristineguadelupe in https://github.com/brainlid/langchain/pull/197

### New Contributors
* @nallwhy made their first contribution in https://github.com/brainlid/langchain/pull/223
* @GenericJam made their first contribution in https://github.com/brainlid/langchain/pull/222
* @mpope9 made their first contribution in https://github.com/brainlid/langchain/pull/227

## v0.3.0-rc.1 (2024-12-15)

### Breaking Changes
- Change return of LLMChain.run/2 ([#170](https://github.com/brainlid/langchain/pull/170))
- Revamped error handling and handles Anthropic's "overload_error" - ([#194](https://github.com/brainlid/langchain/pull/194))

#### Change return of LLMChain.run/2 ([#170](https://github.com/brainlid/langchain/pull/170))

##### Why the change

Before this change, an `LLMChain`'s `run` function returned `{:ok, updated_chain, last_message}`.

When an assistant (ie LLM) issues a ToolCall and when `run` is in the mode `:until_success` or `:while_need_response`, the `LLMChain` will automatically execute the function and return the result as a new Message back to the LLM. This works great!

The problem comes when an application needs to keep track of all the messages being exchanged during a run operation. That can be done by using callbacks and sending and receiving messages, but that's far from ideal. It makes more sense to have access to that information directly after the `run` operation completes.

##### What this change does

This PR changes the returned type to `{:ok, updated_chain}`.

The `last_message` is available in `updated_chain.last_message`. This cleans up the return API.

This change also adds `%LLMChain{exchanged_messages: exchanged_messages}`,or `updated_chain.exchanged_messages` which is a list of all the messages exchanged between the application and the LLM during the execution of the `run` function.

This breaks the return contract for the `run` function.

##### How to adapt to this change

To adapt to this, if the application isn't using the `last_message` in `{:ok, updated_chain, _last_message}`, then delete the third position in the tuple. Ex: `{:ok, updated_chain}`.

Access to the `last_message` is available on the `updated_chain`.

```elixir
{:ok, updated_chain} =
  %{llm: model}
  |> LLMChain.new!()
  |> LLMChain.run()

last_message = updated_chain.last_message
```

NOTE: that the `updated_chain` now includes `updated_chain.exchanged_messages` which can also be used.

#### Revamped error handling and handles Anthropic's "overload_error" - ([#194](https://github.com/brainlid/langchain/pull/194))

**What you need to do:**
Check your application code for how it is responding to and handling error responses.

If you want to keep the same previous behavior, the following code change will do that:

```elixir
case LLMChain.run(chain) do
  {:ok, _updated_chain} ->
    :ok

  # return the error for display
  {:error, _updated_chain, %LangChainError{message: reason}} ->
    {:error, reason}
end
```

The change from:

```
{:error, _updated_chain, reason}
```

To:

```
{:error, _updated_chain, %LangChainError{message: reason}}
```

When possible, a `type` value may be set on the `LangChainError`, making it easier to handle some error types programmatically.

### Features
- Added ability to summarize LLM conversations (#216)
- Implemented initial support for fallbacks (#207)
- Added AWS Bedrock support for ChatAnthropic (#154)
- Added OpenAI's new structured output API (#180)
- Added support for examples to title chain (#191)
- Added tool_choice support for OpenAI and Anthropic (#142)
- Added support for passing safety settings to Google AI (#186)
- Added OpenAI project authentication (#166)

### Fixes
- Fixed specs and examples (#211)
- Fixed content-part encoding and decoding for Google API (#212)
- Fixed ChatOllamaAI streaming response (#162)
- Fixed streaming issue with Azure OpenAI Service (#158, #161)
- Fixed OpenAI stream decode issue (#156)
- Fixed typespec error on Message.new_user/1 (#151)
- Fixed duplicate tool call parameters (#174)

### Improvements
- Added error type support for Azure token rate limit exceeded
- Improved error handling (#194)
- Enhanced function execution failure response
- Added "processed_content" to ToolResult struct (#192)
- Implemented support for strict mode for tools (#173)
- Updated documentation for ChatOpenAI use on Azure
- Updated config documentation for API keys
- Updated README examples

### Azure & Google AI Updates
- Added Azure test for ChatOpenAI usage
- Added support for system instructions for Google AI (#182)
- Handle functions with no parameters for Google AI (#183)
- Handle missing token usage fields for Google AI (#184)
- Handle empty text parts from GoogleAI responses (#181)
- Handle all possible finishReasons for ChatGoogleAI (#188)

### Documentation
- Added LLM Model documentation for tool_choice
- Updated documentation using new functions
- Added custom functions notebook
- Improved documentation formatting (#145)
- Added links to models in the config section
- Updated getting started doc for callbacks

## v0.3.0-rc.0 (2024-06-05)

**Added:**

* `LangChain.ChatModels.ChatGoogleAI` which differed too significantly from `LangChain.ChatModels.ChatGoogleAI`.  What's up with that? I'm looking at you Google! 👀
  * Thanks for the [contribution](https://github.com/brainlid/langchain/pull/124) Raul Chedrese!
* New callback mechanism was introduced to ChatModels and LLMChain. It was inspired by the approach used in the TS/JS LangChain library.
* Ability to provide plug-like middleware functions for pre-processing an assistant response message. Most helpful when coupled with a new run mode called `:until_success`. The first built-in one is `LangChain.MessageProcessors.JsonProcessor`.
* LLMChain has an internally managed `current_failure_count` and a publicly managed `max_retry_count`.
* New run mode `:until_success` uses failure and retry counts to repeatedly run the chain when the LLMs responses fail a MessageProcessor.
* `LangChain.MessageProcessors.JsonProcessor` is capable of extracting JSON contents and converting it to an Elixir map using `Jason`. Parsing errors are returned to the LLM for it to try again.
* The attribute `processed_content` was added to a `LangChain.Message`. When a MessageProcessor is run on a received assistant message, the results of the processing are accumulated there. The original `content` remains unchanged for when it is sent back to the LLM and used when fixing or correcting it's generated content.
* Callback support for LLM ratelimit information returned in API response headers. These are currently implemented for Anthropic and OpenAI.
* Callback support for LLM token usage information returned when available.
* `LangChain.ChatModels.ChatModel` additions
  * Added `add_callback/2` makes it easier to add a callback to an chat model.
  * Added `serialize_config/1` to serialize an LLM chat model configuration to a map that can be restored later.
  * Added `restore_from_map/1` to restore a configured LLM chat model from a database (for example).
* `LangChain.Chain.LLMChain` additions
  * New function `add_callback/2` makes it easier to add a callback to an existing `LLMChain`.
  * New function `add_llm_callback/2` makes it easier to add a callback to a chain's LLM. This is particularly useful when an LLM model is restored from a database when loading a past conversation and wanting to preserve the original configuration.


**Changed:**

* `LLMChain.run/2` error result now includes the failed chain up to the point of failure. This is helpful for debugging.
* `ChatOpenAI` and `ChatAnthropic` both support the new callbacks.
* Many smaller changes and contributions were made. This includes updates to the README for clarity,
* `LangChain.Utils.fire_callback/3` was refactored into `LangChain.Utils.fire_streamed_callback/2` where it is only used for processing deltas and uses the new callback mechanism.
* Notebooks were moved to the separate demo project
* `LangChain.ChatModels.ChatGoogleAI`'s key `:version` was changed to `:api_version` to be more consistent with other models and allow for model serializers to use the `:version` key.

### Migrations Steps

The `LLMChain.run/2` function changed. Migrating should be easy.

**From:**

```elixir
chain
|> LLMChain.run(while_needs_response: true)
```

**Is changed to:**

```elixir
chain
|> LLMChain.run(mode: :while_needs_response)
```

This change enabled adding the new mode `:until_success`, which is mutually exclusive with `:while_needs_response`.

Additionally, the error return value was changed to include the chain itself.

**From:**

```elixir
{:error, reason} = LLMChain.run(chain)
```

**Is changed to:**

```elixir
{:error, _updated_chain, reason} = LLMChain.run(chain)
```

You can disregard the updated chain if you don't need it.

Callback events work differently now. Previously, a single `callback_fn` was executed and the developer needed to pattern match on a `%Message{}` or `%MessageDelta{}`. Callbacks work differently now.

When creating an LLM chat model, we can optionally pass in a map of callbacks where the event name is linked to the function to execute.

**From:**

```elixir
live_view_pid = self()

callback_fn = fn
  %MessageDelta{} = delta ->
    send(live_view_pid, {:received_delta, delta})

  %Message{} = message ->
    send(live_view_pid, {:received_message, message})
end

{:ok, _result_chain, last_message} =
  LLMChain.new!(%{llm: %ChatAnthropic{stream: false}})
  |> LLMChain.add_message(Message.new_user!("Say, 'Hi!'!"))
  |> LLMChain.run(callback_fn: callback_fn)
```

The equivalent code would look like this:

**Is changed to:**

```elixir
live_view_pid = self()

handler = %{
  on_llm_new_delta: fn _model, delta ->
    send(live_view_pid, {:received_delta, delta})
  end,
  on_llm_new_message: fn _model, message ->
    send(live_view_pid, {:received_message, message})
  end
}

{:ok, _result_chain, last_message} =
  LLMChain.new!(%{llm: %ChatAnthropic{stream: false, callbacks: [handler]}})
  |> LLMChain.add_message(Message.new_user!("Say, 'Hi!'!"))
  |> LLMChain.run()
```

The `Message` and `MessageDelta` callbacks are now set on the model. The callbacks are more granular and new callbacks are supported on the `LLMChain` as well. This more flexible configuration allows for more callbacks to be added as we move forward.

Also of note, is that the callbacks are set as a list of handler maps. This means we can assign multiple sets of callbacks for different purposes and they all get executed.

## v0.2.0 (2024-04-30)

For LLMs that support it (verified with ChatGPT and Anthropic), a user message can now contain multiple `ContentPart`s, making it "multi-modal". This means images and text can be combined into a single message allowing for interactions about the images to now be possible.

**Added:**

* `LangChain.Message.ContentPart` - used for User messages and multi-modal support. Google's AI assistant can return multiple parts as well.
* `LangChain.Message.ToolCall` - an assistant can request multiple tool calls in the same message.
* `LangChain.Message.ToolResult` - the system's answer to a `ToolCall`. It adds an is_error boolean flag. This an be helpful in the UI, but Anthropic specifically wants it.
* Add llama-3 chat template by @bowyern in https://github.com/brainlid/langchain/pull/102

**Changed:**

* The roles of `:function` and `:function_call` are removed. The equivalent of a `function_call` is expressed by an `:assistant` role making one or more `ToolCall` requests. The `:function` was the system's answer to a function call. This is now in the `:tool` role.
* Role `:tool` was added. A tool message contains one or more `ToolResult` messages.

## v0.1.10 (2024-03-07)

**Changes**

- Fix invalid default url for google ai by @pkrawat1 in https://github.com/brainlid/langchain/pull/82

## v0.1.9 (2024-02-29) - The Leap Release!

This adds support for Bumblebee as a Chat model, making it easy to have conversations with Llama 2, Mistral, and Zephyr LLMs.

See the documentation in `LangChain.ChatModels.ChatBumblebee` for getting started.

NOTE: That at this time, none of the models support the `function` ability, so that is not supported yet.

This release includes an experimental change for better support of streamed responses that are broken up over multiple messages from services like ChatGPT and others.

Other library dependencies requirements were relaxed, making it easier to support different versions of libraries like `req` and `nx`.

* Add mistral chat by @michalwarda in https://github.com/brainlid/langchain/pull/76
* handle receiving JSON data broken up over multiple messages by @brainlid in https://github.com/brainlid/langchain/pull/80
* Add initial support for Zephyr 7b Beta by @brainlid in https://github.com/brainlid/langchain/pull/41

## v0.1.8 (2024-02-16)

**Breaking change**: `RoutingChain`'s required values changed. Previously, `default_chain` was assigned an `%LLMChain{}` to return when no more specific routes matched.

This was changed to be `default_route`. It now expects a `%PromptRoute{}` to be provided.

Here's how to make the change:

      selected_route =
        RoutingChain.new(%{
          llm: ChatOpenAI.new(%{model: "gpt-3.5-turbo", stream: false}),
          input_text: user_input_text,
          routes: routes,
          default_route: PromptRoute.new!(%{name: "DEFAULT", chain: fallback_chain})
        })
        |> RoutingChain.evaluate()

The `default_chain` was changed to `default_route` and now expects a `PromptRoute` to be provided. The above example includes a sample default route that includes an optional `fallback_chain`.

Previously, the returned value from `RoutingChain.evaluate/1` was a `selected_chain`; it now returns the `selected_route`.

**Why was this changed?**

This was changed to make it easier to use a `PromptChain` when there isn't an associated `%LLMChain{}` for it. The application must just need the answer of which route was selected.

This includes the change of not requiring a `%PromptChain{}`'s `description` or `chain` field.

**Other Changes**
* Add support for Ollama open source models by @medoror in https://github.com/brainlid/langchain/pull/70
* Add clause to match call_response spec by @medoror in https://github.com/brainlid/langchain/pull/72
* Add max_tokens option for OpenAI calls by @matthusby in https://github.com/brainlid/langchain/pull/73

## v0.1.7 (2024-01-18)

- Improvements for more intelligent agents - https://github.com/brainlid/langchain/pull/61
  - adds `LangChain.Chains.RoutingChain` - first-pass LLM chain to select the best route to take given the user's initial prompt
  - adds `LangChain.Chains.TextToTitleChain` - turn the user's prompt into a title for the conversation
- Removed the need for a function to send a message to the process for how to display the function being executed
- Updated dependencies
- Add support for Google AI / Gemini Pro model by @jadengis in https://github.com/brainlid/langchain/pull/59
- Built-in automatic retries when underlying Mint connection is closed in https://github.com/brainlid/langchain/pull/68

## v0.1.6 (2023-12-12)

- Fix for correct usage of new Req retry setting. PR #57

## v0.1.5 (2023-12-11)

- Upgraded Req to v0.4.8. It contains a needed retry fix for certain situations.
- Fix OpenAI returns "Unrecognized request argument supplied: api_key" PR #54

## v0.1.4 (2023-12-11)

- Merged PR #45 - https://github.com/brainlid/langchain/pull/45
  - Added `LangChain.Utils.ChainResult` for helper functions when working with LLMChain result values.
- Merged PR #46 - https://github.com/brainlid/langchain/pull/46
  - Add possibility to use api_key per chat invocation.
- Merged PR #51 - https://github.com/brainlid/langchain/pull/51
  - Update req 0.4.7
  - Hopefully resolves issue where Finch connections would be closed and a now does a built-in retry.
- Merged PR #52 - https://github.com/brainlid/langchain/pull/52
  - Allow overriding OpenAI compatible API endpoint. Caller can pass an alternate `endpoint`.

## v0.1.3 (2023-12-01)

- Merged PR #43 - https://github.com/brainlid/langchain/pull/43
  - Add Finch retry strategy to OpenAI Chat API requests
- Merged PR #39 - https://github.com/brainlid/langchain/pull/39
  - Changed ENV key from `OPENAI_KEY` to `OPENAI_API_KEY` to be consistent with the OpenAI docs.
- Merged PR #36 - https://github.com/brainlid/langchain/pull/36
  - Support specifying the `seed` with OpenAI calls. Used in testing for more deterministic behavior.
- Merged PR #34 - https://github.com/brainlid/langchain/pull/34
  - Enable sending the `json_response` flag with OpenAI model requests.
- Created `LangChain.FunctionParam` to express JSONSchema-friendly data structures. Supports basic types, arrays, enums, objects, arrays of objects and nested objects.
  - Still allows for full control over JSONSchema by providing an override `parameters_schema` object to full self-describe it.

## v0.1.2 (2023-10-26)

- refactor(chat_open_ai): Harden `do_process_response` by @Cardosaum in https://github.com/brainlid/langchain/pull/21
  - Improve JSON error handling result from ChatGPT
- Update req to 0.4.4 by @medoror in https://github.com/brainlid/langchain/pull/25
  - Updated to Req 0.4.4


## v0.1.1 (2023-10-10)

Minor update release.

- added "update_custom_context" to LLMChain
- added support for setting the OpenAI-Organization header in requests
- fixed data extraction chain and improved the prompt
- make chatgpt response tests more robust


## v0.1.0 (2023-09-18)

Initial release when published to [hex.pm](https://hex.pm).
