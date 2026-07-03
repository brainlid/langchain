defmodule LangChain.ChatModels.ChatModel do
  alias LangChain.Message
  alias LangChain.MessageDelta
  alias LangChain.Function
  alias LangChain.LangChainError
  alias LangChain.TokenUsage
  alias LangChain.Utils

  @type call_response ::
          {:ok, Message.t() | [Message.t()] | [MessageDelta.t()]} | {:error, LangChainError.t()}

  @type tool :: Function.t()
  @type tools :: [tool()]

  @type t :: Ecto.Schema.t()

  @callback call(
              t(),
              String.t() | [Message.t()],
              [LangChain.Function.t()]
            ) :: call_response()

  @doc """
  Returns the provider name for this chat model (e.g. "openai", "anthropic").

  Used in telemetry metadata to identify the LLM provider without inspecting
  the module name. This is an optional callback — if not implemented, the
  provider can be derived from the module name via `provider/1`.
  """
  @callback provider() :: String.t()

  @callback retry_on_fallback?(LangChainError.t()) :: boolean()

  @callback serialize_config(t()) :: %{String.t() => any()}

  @callback restore_from_map(%{String.t() => any()}) :: {:ok, struct()} | {:error, String.t()}

  @optional_callbacks [provider: 0]

  @doc """
  Returns the provider name for a given chat model struct.

  Dispatches to the model module's `provider/0` callback when implemented.
  Otherwise it falls back to a **best-effort** guess derived from the module
  name (drop a leading `Chat`, then `Macro.underscore/1`), e.g.
  `LangChain.ChatModels.ChatAnthropic` -> `"anthropic"`.

  The fallback is only a heuristic and cannot recover canonical names for
  multi-word or acronym module names — `ChatOpenAIResponses` derives to
  `"open_ai_responses"`, not `"openai_responses"`, so it won't round-trip through
  `LangChain.OpenTelemetry.ProviderMapping`. **Every model should implement
  `provider/0`** to get a stable, canonical provider string; all in-tree models
  do. (A multi-provider adapter like `ChatReqLLM`, whose provider varies per
  call, instead sets `:provider` directly in its telemetry metadata rather than
  via `provider/0`.)
  """
  @spec provider(t()) :: String.t()
  def provider(%module{}) do
    if function_exported?(module, :provider, 0) do
      module.provider()
    else
      module
      |> Module.split()
      |> List.last()
      |> String.replace_leading("Chat", "")
      |> Macro.underscore()
    end
  end

  @doc """
  Extracts token usage from an LLM call result for use as a `span/4` `:enrich_stop` callback.

  Returns a map with `:token_usage` set to the `%TokenUsage{}` struct when
  available, or `nil` otherwise.
  """
  @spec token_usage_from_result(call_response()) :: %{token_usage: TokenUsage.t() | nil}
  def token_usage_from_result({:ok, %Message{} = msg}) do
    %{token_usage: get_in(msg.metadata, [:usage])}
  end

  def token_usage_from_result({:ok, %MessageDelta{} = delta}) do
    %{token_usage: get_in(delta.metadata, [:usage])}
  end

  def token_usage_from_result({:ok, [_ | _] = items}) do
    # Streaming calls return a list of `%MessageDelta{}` structs; the accumulated
    # `%TokenUsage{}` rides on the final delta's metadata. Non-streaming calls
    # that return a list of `%Message{}` structs carry it the same way. Scan the
    # list for the first item whose metadata holds a `%TokenUsage{}`.
    usage =
      Enum.find_value(items, fn
        %Message{metadata: %{usage: %TokenUsage{} = usage}} -> usage
        %MessageDelta{metadata: %{usage: %TokenUsage{} = usage}} -> usage
        _ -> nil
      end)

    %{token_usage: usage}
  end

  def token_usage_from_result(_result), do: %{token_usage: nil}

  @doc """
  Wraps an LLM `call/3` body in the standard `[:langchain, :llm, :call]`
  telemetry span with token-usage enrichment and request-parameter capture
  already wired.

  Every chat model must open its LLM-call span through this helper. Building the
  `LangChain.Telemetry.span/4` call by hand in each model made it easy to forget
  the `:enrich_stop` callback — and a model that forgets it silently drops token
  usage from the `[:langchain, :llm, :call, :stop]` event (and the OTEL span) for
  that provider, with nothing failing to signal the mistake. Centralizing the
  event name and `:enrich_stop` here removes that footgun: a new provider only
  has to build its metadata and call this function.

  `model` is the chat model struct (or `nil`). It is read for three pieces of
  telemetry metadata (each via `put_new`, so a provider that builds its own richer
  value keeps precedence):

    * `:request_options` — standard `gen_ai.request.*` parameters, via `request_options/1`.
    * `:output_type` — `"text"` or `"json"`, via `output_type/1`.
    * `:endpoint` — the request URL (when the model exposes `:endpoint`), which the
      OTEL layer turns into `server.address` / `server.port`.

  This lets the OTEL layer emit those attributes without every provider re-plumbing
  them by hand. `metadata` is the LLM-call metadata map (`:model`, `:provider`,
  `:message_count`, `:tools_count`, ...). `fun` is the zero-arity function that
  performs the request and returns the `call_response()`.
  """
  @spec llm_telemetry_span(struct() | nil, map(), (-> result)) :: result when result: any()
  def llm_telemetry_span(model, metadata, fun)
      when is_map(metadata) and is_function(fun, 0) do
    metadata =
      metadata
      |> Map.put_new(:request_options, request_options(model))
      |> Map.put_new(:output_type, output_type(model))

    # Only carry the endpoint when the model actually exposes one, to avoid a
    # `nil` key on every LLM telemetry event for models with no default endpoint.
    metadata =
      case endpoint(model) do
        nil -> metadata
        url -> Map.put_new(metadata, :endpoint, url)
      end

    # Prime time-to-first-token tracking. Streaming decode runs in THIS process
    # (Req's `into` collector is synchronous), so the shared
    # `LangChain.Utils.fire_streamed_callback/2` can read this start time from the
    # process dictionary when the first delta lands and emit the
    # `[:langchain, :llm, :stream, :first_token]` event. Reset the "seen" flag so a
    # reused process measures each call afresh; a non-streaming call simply never
    # fires the streamed callback, so nothing is emitted.
    Process.put(:langchain_llm_call_start, System.monotonic_time())
    Process.delete(:langchain_stream_first_token_seen)

    LangChain.Telemetry.span(
      [:langchain, :llm, :call],
      metadata,
      fun,
      enrich_stop: &token_usage_from_result/1
    )
  end

  @doc """
  Best-effort `gen_ai.output.type` for a chat model.

  Returns `"json"` when the model is configured to request structured/JSON output
  (`:json_response` set to `true`, a non-nil `:json_schema`, or a JSON-typed
  `:response_format`), otherwise `"text"`. All LangChain chat models are
  chat-completion style, so only `"text"` and `"json"` are distinguished. Mirrors
  the conventional-field heuristic used by `request_options/1`.
  """
  @spec output_type(struct() | nil) :: String.t()
  def output_type(nil), do: "text"

  def output_type(%_module{} = model) do
    cond do
      Map.get(model, :json_response) == true -> "json"
      not is_nil(Map.get(model, :json_schema)) -> "json"
      json_response_format?(Map.get(model, :response_format)) -> "json"
      true -> "text"
    end
  end

  def output_type(_), do: "text"

  # `:response_format` (Grok/Perplexity) is a raw map; treat it as JSON output only
  # when its `type` explicitly names json (e.g. "json_object", "json_schema").
  # Anything else — including `%{"type" => "text"}` or an indeterminate map — stays
  # "text" so we never mislabel a plain-text response.
  defp json_response_format?(%{"type" => type}) when is_binary(type),
    do: String.contains?(type, "json")

  defp json_response_format?(%{type: type}) when is_binary(type),
    do: String.contains?(type, "json")

  defp json_response_format?(_), do: false

  # Request endpoint URL for `server.address`/`server.port`, when the model exposes
  # one under the conventional `:endpoint` field.
  @spec endpoint(struct() | nil) :: String.t() | nil
  defp endpoint(nil), do: nil
  defp endpoint(%_module{} = model), do: Map.get(model, :endpoint)

  @doc """
  Best-effort extraction of standard request parameters from a chat model struct
  for telemetry, using the conventional public schema field names.

  Reads the well-known fields (`:temperature`, `:max_tokens`, `:top_p`,
  `:top_k`, `:frequency_penalty`, `:presence_penalty`, `:seed`, `:n`, `:stream`,
  `:stop`/`:stop_sequences`, `:reasoning_effort`) when the struct defines them,
  dropping any that are absent or `nil`. This mirrors the heuristic used by
  `provider/1`: models expose these parameters as public schema fields under
  conventional names, so one generic reader avoids per-provider duplication. A
  model that names a parameter differently simply won't have it captured — the
  result is a useful subset, not a guarantee.

  The returned map uses provider-neutral keys; `LangChain.OpenTelemetry.Attributes`
  maps them to their `gen_ai.request.*` semantic-convention attribute names.
  """
  @spec request_options(struct() | nil) :: map()
  def request_options(nil), do: %{}

  def request_options(%_module{} = model) do
    %{
      temperature: Map.get(model, :temperature),
      max_tokens: Map.get(model, :max_tokens),
      top_p: Map.get(model, :top_p),
      top_k: Map.get(model, :top_k),
      frequency_penalty: Map.get(model, :frequency_penalty),
      presence_penalty: Map.get(model, :presence_penalty),
      seed: Map.get(model, :seed),
      choice_count: Map.get(model, :n),
      stream: Map.get(model, :stream),
      stop_sequences: Map.get(model, :stop) || Map.get(model, :stop_sequences),
      reasoning_level: Map.get(model, :reasoning_effort)
    }
    |> Enum.reject(fn {_key, value} -> is_nil(value) end)
    |> Map.new()
  end

  def request_options(_), do: %{}

  @doc """
  Create a serializable map from a ChatModel's current configuration that can
  later be restored.
  """
  def serialize_config(%chat_module{} = model) do
    # plucks the module from the struct and, because of the behaviour, assumes
    # the module defines a `serialize_config/1` function that is executed.
    chat_module.serialize_config(model)
  end

  @doc """
  Restore a ChatModel from a serialized config map.
  """
  @spec restore_from_map(nil | %{String.t() => any()}) :: {:ok, struct()} | {:error, String.t()}
  def restore_from_map(nil), do: {:error, "No data to restore"}

  def restore_from_map(%{"module" => module_name} = data) do
    case Utils.module_from_name(module_name) do
      {:ok, module} ->
        module.restore_from_map(data)

      {:error, _reason} = error ->
        error
    end
  end
end
