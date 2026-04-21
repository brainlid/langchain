defmodule LangChain.ChatModels.ChatAwsMantle do
  @moduledoc """
  Represents a chat model hosted by AWS Bedrock's **Mantle** endpoint — the
  OpenAI-compatible gateway AWS introduced for third-party models such as
  Moonshot AI's Kimi K2 family and OpenAI's gpt-oss series.

  Mantle accepts standard OpenAI Chat Completions requests, so much of the
  wire format mirrors `LangChain.ChatModels.ChatOpenAI`. This module exists as
  a separate chat model because Mantle has several differences that warrant
  dedicated handling:

  - **Region-aware URL building** — `https://bedrock-mantle.{region}.api.aws/v1/chat/completions`
  - **Two auth modes** — Bedrock API key (Bearer) **or** AWS IAM (SigV4)
  - **Reasoning extraction** — Mantle returns model reasoning at `message.reasoning`
    (or `delta.reasoning` when streaming), which `ChatOpenAI` silently drops
  - **Higher default `receive_timeout`** — Mantle exhibits intermittent slow
    starts of 60s+, so the default is 120s here vs OpenAI's 60s
  - **Bounded default `max_tokens: 4096`** — Kimi occasionally falls into
    token-repetition loops; streaming keeps the HTTP layer alive as chunks
    arrive, so an uncapped request can run indefinitely. Override as needed
    when reasoning budgets require more
  - **Per-model quirks** — Kimi prepends a leading space to text content; uses
    `functions.NAME:N` for `call_id` shape; narrates before tool calls

  ## Available Models (as of writing)

  | Model ID                         | Vendor    | Notes                                  |
  | -------------------------------- | --------- | -------------------------------------- |
  | `moonshotai.kimi-k2-thinking`    | Moonshot  | Reasoning by default, 128K ctx         |
  | `moonshotai.kimi-k2.5`           | Moonshot  | Multimodal, hybrid thinking via `:reasoning_effort` |
  | `openai.gpt-oss-120b`            | OpenAI    | Open-source GPT, hosted by AWS         |

  More models may be available — call `GET /v1/models` against the Mantle
  endpoint to discover what's currently published.

  ## Authentication

  Two mutually-exclusive auth modes:

  ### Bearer (Bedrock API key) — simplest

  Generate a long-term Bedrock API key in the AWS console
  ([Bedrock API keys](https://console.aws.amazon.com/bedrock/home#/api-keys/long-term/create))
  and set it as the `:api_key`:

      ChatAwsMantle.new!(%{
        model: "moonshotai.kimi-k2.5",
        region: "us-east-1",
        api_key: System.fetch_env!("AWS_BEARER_TOKEN_BEDROCK")
      })

  ### AWS SigV4 (IAM credentials) — production-friendly

  Pass a zero-arity function returning IAM credentials. Useful when the host
  already has IAM-based credentials available (e.g. ExAws):

      ChatAwsMantle.new!(%{
        model: "moonshotai.kimi-k2.5",
        region: "us-east-1",
        credentials: fn ->
          ExAws.Config.new(:s3)
          |> Map.take([:access_key_id, :secret_access_key])
          |> Map.to_list()
        end
      })

  ## Reasoning / Thinking

  K2.5 is a hybrid thinking model — pass OpenAI's standard `:reasoning_effort`
  to enable structured reasoning:

      ChatAwsMantle.new!(%{
        model: "moonshotai.kimi-k2.5",
        region: "us-east-1",
        api_key: System.fetch_env!("AWS_BEARER_TOKEN_BEDROCK"),
        reasoning_effort: "high"
      })

  When reasoning is active, the response message will include a `ContentPart`
  of `type: :thinking` containing the model's chain of thought, alongside the
  normal `:text` content parts.

  K2 Thinking always reasons (it's the model's default mode); the field is
  populated regardless of `:reasoning_effort`.

  ## Sampling controls

  Standard OpenAI sampling parameters are supported and passed through to
  Mantle unchanged:

  - `:temperature` — 0.0 to 2.0 (default `1.0`)
  - `:top_p` — 0.0 to 1.0 nucleus sampling cutoff. OpenAI recommends tuning
    this *or* temperature, not both
  - `:frequency_penalty` — -2.0 to 2.0. Positive values discourage reuse of
    tokens proportional to how often they've already appeared. **Kimi K2.5
    on Mantle has been observed to occasionally lock into single-token
    repetition loops (e.g. streams of "!"); `frequency_penalty: 0.5` is a
    reasonable starting defense.**
  - `:presence_penalty` — -2.0 to 2.0. Binary variant of frequency_penalty
    (penalizes any token that has appeared at all)

  ## Streaming

  Set `stream: true` to receive incremental `MessageDelta` updates via the
  `on_llm_new_delta` callback. Mantle emits standard OpenAI SSE chunks for
  content and tool calls, and adds a sibling `delta.reasoning` field when
  reasoning is active. `ChatAwsMantle` extracts those into `:thinking`
  ContentParts so the merged final message carries `[thinking_part, text_part]`
  in order.

  ## Multimodal (K2.5 vision)

  Kimi K2.5 is natively multimodal. Send images via standard LangChain
  `ContentPart` structs — `ChatAwsMantle` delegates serialization to
  `ChatOpenAI.content_part_for_api/2`, which emits Mantle's expected
  `{"type": "image_url", "image_url": {"url": "data:<media>;base64,..."}}`
  shape:

      {:ok, bytes} = File.read("photo.jpg")

      Message.new_user!([
        ContentPart.text!("What's in this image?"),
        ContentPart.image!(Base.encode64(bytes), media: :jpeg)
      ])
      |> then(&ChatAwsMantle.call(model, [&1]))

  Mantle runs images through an upstream sanitizer that rejects degenerate
  inputs (tiny or unusual images may return a 400 with
  `"Failed to sanitize image"`). Use real photographs or reasonably-sized
  source images. Vision tokens add meaningfully to `prompt_tokens` — a
  1200×675 JPG consumes roughly 1100 prompt tokens.

  ## Open Notes

  Streaming and tool-calling support follow the same wire format as
  `ChatOpenAI` — see the smoke tests for verified behavior.
  """
  use Ecto.Schema
  require Logger
  import Ecto.Changeset
  alias __MODULE__
  alias LangChain.ChatModels.ChatModel
  alias LangChain.ChatModels.ChatOpenAI
  alias LangChain.Config
  alias LangChain.LangChainError
  alias LangChain.Message
  alias LangChain.Message.ContentPart
  alias LangChain.MessageDelta
  alias LangChain.Callbacks
  alias LangChain.Utils

  @behaviour ChatModel

  @default_receive_timeout 120_000
  # Kimi K2.5 has been observed to fall into degenerate token-repetition loops
  # ("!!!!!!" runs) on a small fraction of requests. Streaming keeps the HTTP
  # layer's receive_timeout alive because chunks keep arriving, so without a
  # max_tokens cap a degenerate run can stream indefinitely. 4096 is enough
  # for normal agent turns (including `reasoning_effort: "medium"`) and caps
  # runaway generation at a small, bounded cost. Override when reasoning
  # needs more headroom.
  @default_max_tokens 4096

  @primary_key false
  embedded_schema do
    field :model, :string
    field :region, :string

    # Optional explicit endpoint override; default is derived from :region.
    field :endpoint, :string

    # Auth: exactly one of :api_key or :credentials must be set.
    field :api_key, :string, redact: true
    # Zero-arity fn returning a keyword list with :access_key_id, :secret_access_key,
    # and optionally :token. Used to build SigV4 signing options for Req.
    field :credentials, :any, virtual: true

    # Standard OpenAI-shaped knobs
    field :temperature, :float, default: 1.0
    field :max_tokens, :integer, default: @default_max_tokens
    field :stream, :boolean, default: false

    # Nucleus sampling (0.0–1.0). OpenAI docs recommend altering *either* this
    # or :temperature, not both. Default nil → Mantle uses its own default.
    field :top_p, :float

    # Discourage reuse of tokens already seen in the generation. Positive
    # values (0.1–2.0) reduce repetition; useful for Kimi K2.5 which has
    # exhibited occasional token-repetition loops ("!!!!!" runs).
    field :frequency_penalty, :float

    # Discourage reuse of tokens that have appeared at all (binary rather
    # than frequency-weighted). Positive values encourage topic diversity.
    field :presence_penalty, :float

    # OpenAI-standard reasoning control. Passed through to Mantle, which
    # translates into the upstream model's thinking mode (verified working
    # for Kimi K2.5).
    field :reasoning_effort, :string

    # Tool choice option, mirrors ChatOpenAI's shape
    field :tool_choice, :map

    # Structured response format
    field :json_response, :boolean, default: false
    field :json_schema, :map

    # Mantle-specific: longer default than ChatOpenAI. Mantle has intermittent
    # 60s+ slow starts; 120s gives those a chance to resolve.
    field :receive_timeout, :integer, default: @default_receive_timeout

    # Stream options (e.g. include_usage)
    field :stream_options, :map, default: nil

    # Callback handlers (treated as internal — not part of API request)
    field :callbacks, {:array, :map}, default: []

    # Debug helper — prints raw request/response when true
    field :verbose_api, :boolean, default: false

    # Req options to merge into the request
    field :req_config, :map, default: %{}
  end

  @type t :: %ChatAwsMantle{}

  @create_fields [
    :model,
    :region,
    :endpoint,
    :api_key,
    :credentials,
    :temperature,
    :max_tokens,
    :stream,
    :top_p,
    :frequency_penalty,
    :presence_penalty,
    :reasoning_effort,
    :tool_choice,
    :json_response,
    :json_schema,
    :receive_timeout,
    :stream_options,
    :verbose_api,
    :req_config
  ]
  @required_fields [:model]

  @valid_reasoning_efforts ~w(low medium high)

  @doc """
  Build a new `ChatAwsMantle` instance from attributes.
  """
  @spec new(attrs :: map()) :: {:ok, t()} | {:error, Ecto.Changeset.t()}
  def new(attrs \\ %{}) do
    %ChatAwsMantle{}
    |> cast(attrs, @create_fields)
    |> common_validation()
    |> apply_action(:insert)
  end

  @doc """
  Build a new `ChatAwsMantle` instance, raising on validation failure.
  """
  @spec new!(attrs :: map()) :: t() | no_return()
  def new!(attrs \\ %{}) do
    case new(attrs) do
      {:ok, model} ->
        model

      {:error, changeset} ->
        raise LangChainError, changeset
    end
  end

  defp common_validation(changeset) do
    changeset
    |> validate_required(@required_fields)
    |> validate_inclusion(:reasoning_effort, @valid_reasoning_efforts,
      message: "must be one of: #{Enum.join(@valid_reasoning_efforts, ", ")}"
    )
    |> validate_number(:temperature, greater_than_or_equal_to: 0, less_than_or_equal_to: 2)
    |> validate_number(:top_p, greater_than_or_equal_to: 0, less_than_or_equal_to: 1)
    |> validate_number(:frequency_penalty, greater_than_or_equal_to: -2, less_than_or_equal_to: 2)
    |> validate_number(:presence_penalty, greater_than_or_equal_to: -2, less_than_or_equal_to: 2)
    |> validate_number(:receive_timeout, greater_than_or_equal_to: 0)
    |> validate_auth()
    |> validate_endpoint_resolvable()
  end

  # Exactly one of :api_key or :credentials must be set.
  defp validate_auth(changeset) do
    api_key = get_field(changeset, :api_key)
    credentials = get_field(changeset, :credentials)

    cond do
      is_binary(api_key) and is_function(credentials, 0) ->
        add_error(
          changeset,
          :api_key,
          "cannot set both :api_key and :credentials — pick one auth mode"
        )

      is_nil(api_key) and is_nil(credentials) ->
        add_error(
          changeset,
          :api_key,
          "must set either :api_key (Bearer) or :credentials (SigV4)"
        )

      not is_nil(credentials) and not is_function(credentials, 0) ->
        add_error(
          changeset,
          :credentials,
          "must be a zero-arity function returning IAM credentials"
        )

      true ->
        changeset
    end
  end

  # Either :endpoint is set explicitly, or :region is set (so we can build one).
  defp validate_endpoint_resolvable(changeset) do
    endpoint = get_field(changeset, :endpoint)
    region = get_field(changeset, :region)

    if is_binary(endpoint) or is_binary(region) do
      changeset
    else
      add_error(
        changeset,
        :region,
        "must set :region (or override :endpoint) so the Mantle URL can be built"
      )
    end
  end

  # Build the request URL for this model. Prefer :endpoint override; otherwise
  # build from :region.
  @doc false
  def url(%ChatAwsMantle{endpoint: endpoint}) when is_binary(endpoint), do: endpoint

  def url(%ChatAwsMantle{region: region}) when is_binary(region) do
    "https://bedrock-mantle.#{region}.api.aws/v1/chat/completions"
  end

  # Resolve the auth header / signing options for a Req call. Returns a keyword
  # list of options to merge into Req.new/1.
  @doc false
  def auth_opts(%ChatAwsMantle{api_key: api_key}) when is_binary(api_key) do
    [auth: {:bearer, api_key}]
  end

  def auth_opts(%ChatAwsMantle{credentials: credentials, region: region})
      when is_function(credentials, 0) and is_binary(region) do
    sigv4 =
      credentials.()
      |> Keyword.merge(region: region, service: :bedrock)

    [aws_sigv4: sigv4]
  end

  @doc """
  Format the request body for the Mantle API. Reuses `ChatOpenAI`'s per-message
  formatting (since the wire format is OpenAI-shaped), but assembles the
  top-level body with Mantle-relevant fields only.
  """
  @spec for_api(t(), [Message.t()], [LangChain.Function.t()]) :: %{atom() => any()}
  def for_api(%ChatAwsMantle{} = model, messages, tools) do
    %{
      model: model.model,
      stream: model.stream,
      messages:
        messages
        |> Enum.map(&strip_thinking_parts/1)
        |> Enum.reduce([], fn m, acc ->
          case ChatOpenAI.for_api(model, m) do
            %{} = data -> [data | acc]
            data when is_list(data) -> Enum.reverse(data) ++ acc
          end
        end)
        |> Enum.reverse()
    }
    |> Utils.conditionally_add_to_map(:temperature, model.temperature)
    |> Utils.conditionally_add_to_map(:max_tokens, model.max_tokens)
    |> Utils.conditionally_add_to_map(:top_p, model.top_p)
    |> Utils.conditionally_add_to_map(:frequency_penalty, model.frequency_penalty)
    |> Utils.conditionally_add_to_map(:presence_penalty, model.presence_penalty)
    |> Utils.conditionally_add_to_map(:reasoning_effort, model.reasoning_effort)
    |> Utils.conditionally_add_to_map(:response_format, response_format(model))
    |> Utils.conditionally_add_to_map(:tools, tools_for_api(model, tools))
    |> Utils.conditionally_add_to_map(:tool_choice, tool_choice_for_api(model))
    |> Utils.conditionally_add_to_map(
      :stream_options,
      stream_options_for_api(model.stream_options)
    )
  end

  # Strip :thinking ContentParts before sending a message back to Mantle.
  # Mantle's wire format (OpenAI Chat Completions) has no representation for
  # reasoning blocks — they're a response-side artifact we surface for UI
  # display. `ChatOpenAI.content_part_for_api/2` has no clause for :thinking
  # and will crash if one round-trips, so we filter them here.
  @spec strip_thinking_parts(Message.t()) :: Message.t()
  defp strip_thinking_parts(%Message{content: content} = msg) when is_list(content) do
    cleaned = Enum.reject(content, &match?(%ContentPart{type: :thinking}, &1))
    %{msg | content: cleaned}
  end

  defp strip_thinking_parts(msg), do: msg

  defp response_format(%ChatAwsMantle{json_response: true, json_schema: schema})
       when not is_nil(schema) do
    %{"type" => "json_schema", "json_schema" => schema}
  end

  defp response_format(%ChatAwsMantle{json_response: true}), do: %{"type" => "json_object"}
  defp response_format(%ChatAwsMantle{json_response: false}), do: nil

  defp tools_for_api(_model, nil), do: []
  defp tools_for_api(_model, []), do: []

  defp tools_for_api(%ChatAwsMantle{} = model, tools) do
    Enum.map(tools, fn %LangChain.Function{} = function ->
      %{"type" => "function", "function" => ChatOpenAI.for_api(model, function)}
    end)
  end

  defp tool_choice_for_api(%ChatAwsMantle{tool_choice: nil}), do: nil
  defp tool_choice_for_api(%ChatAwsMantle{tool_choice: choice}), do: choice

  defp stream_options_for_api(nil), do: nil

  defp stream_options_for_api(%{} = data) do
    %{"include_usage" => Map.get(data, :include_usage, Map.get(data, "include_usage"))}
  end

  # ---------------------------------------------------------------------------
  # ChatModel behavior
  # ---------------------------------------------------------------------------

  @doc """
  Make a call to the Mantle API. Returns `{:ok, [%Message{}]}` on success or
  `{:error, %LangChainError{}}` on failure.
  """
  @impl ChatModel
  def call(model, prompt, tools \\ [])

  def call(%ChatAwsMantle{} = model, prompt, tools) when is_binary(prompt) do
    call(model, [Message.new_user!(prompt)], tools)
  end

  def call(%ChatAwsMantle{} = model, messages, tools) when is_list(messages) do
    metadata = %{
      model: model.model,
      message_count: length(messages),
      tools_count: length(tools)
    }

    LangChain.Telemetry.span([:langchain, :llm, :call], metadata, fn ->
      try do
        LangChain.Telemetry.llm_prompt(
          %{system_time: System.system_time()},
          %{model: model.model, messages: messages}
        )

        case do_api_request(model, messages, tools) do
          {:error, %LangChainError{} = err} ->
            {:error, err}

          parsed ->
            LangChain.Telemetry.llm_response(
              %{system_time: System.system_time()},
              %{model: model.model, response: parsed}
            )

            {:ok, parsed}
        end
      rescue
        err in LangChainError ->
          {:error, err}
      end
    end)
  end

  @impl ChatModel
  def retry_on_fallback?(%LangChainError{type: type}) when type in ["timeout", "connection"],
    do: true

  def retry_on_fallback?(_), do: false

  @impl ChatModel
  def serialize_config(%ChatAwsMantle{} = model) do
    Map.from_struct(model)
    |> Map.drop([:credentials, :callbacks, :api_key])
    |> Map.put(:module, Atom.to_string(__MODULE__))
    |> Map.put(:version, 1)
    |> stringify_keys()
  end

  @impl ChatModel
  def restore_from_map(%{"version" => 1} = data) do
    attrs = data |> Map.delete("module") |> Map.delete("version") |> atomize_keys()
    new(attrs)
  end

  def restore_from_map(_other), do: {:error, "Unsupported ChatAwsMantle config version"}

  defp stringify_keys(map) when is_map(map) do
    Map.new(map, fn {k, v} -> {to_string(k), v} end)
  end

  defp atomize_keys(map) when is_map(map) do
    Map.new(map, fn {k, v} -> {String.to_existing_atom(to_string(k)), v} end)
  end

  # ---------------------------------------------------------------------------
  # HTTP request — non-streaming for now. Streaming added in a follow-up.
  # ---------------------------------------------------------------------------

  @doc false
  def do_api_request(%ChatAwsMantle{stream: false} = model, messages, tools) do
    body = for_api(model, messages, tools)

    if model.verbose_api do
      IO.inspect(body, label: "RAW DATA BEING SUBMITTED (Mantle)")
    end

    req_opts =
      [
        url: url(model),
        json: body,
        receive_timeout: model.receive_timeout,
        retry: :transient,
        max_retries: 2,
        retry_delay: fn attempt -> 500 * attempt end
      ] ++ auth_opts(model)

    req = Req.new(req_opts)

    req
    |> Req.merge(model.req_config |> Keyword.new())
    |> Req.post()
    |> case do
      {:ok, %Req.Response{status: status, body: body} = response} when status in 200..299 ->
        if model.verbose_api do
          IO.inspect(response, label: "RAW REQ RESPONSE (Mantle)")
        end

        Callbacks.fire(model.callbacks, :on_llm_response_headers, [response.headers])
        do_process_response(model, body)

      {:ok, %Req.Response{status: status, body: body}} ->
        {:error,
         LangChainError.exception(
           type: "api_error",
           message: "Mantle returned HTTP #{status}: #{inspect(body)}"
         )}

      {:error, %Req.TransportError{reason: reason}} ->
        {:error,
         LangChainError.exception(
           type: "connection",
           message: "Mantle connection error: #{inspect(reason)}"
         )}

      {:error, error} ->
        {:error, LangChainError.exception(type: "unknown", message: inspect(error))}
    end
  end

  def do_api_request(%ChatAwsMantle{stream: true} = model, messages, tools) do
    body = for_api(model, messages, tools)

    if model.verbose_api do
      IO.inspect(body, label: "RAW DATA BEING SUBMITTED (Mantle stream)")
    end

    req_opts =
      [
        url: url(model),
        json: body,
        receive_timeout: model.receive_timeout
      ] ++ auth_opts(model)

    Req.new(req_opts)
    |> Req.merge(model.req_config |> Keyword.new())
    |> Req.post(
      into:
        Utils.handle_stream_fn(
          model,
          &ChatOpenAI.decode_stream/1,
          &do_process_response(model, &1)
        )
    )
    |> case do
      {:ok, %Req.Response{status: status, body: data} = response} when status in 200..299 ->
        Callbacks.fire(model.callbacks, :on_llm_response_headers, [response.headers])
        data

      {:ok, %Req.Response{status: status, body: body}} ->
        {:error,
         LangChainError.exception(
           type: "api_error",
           message: "Mantle returned HTTP #{status}: #{inspect(body)}"
         )}

      {:error, %LangChainError{} = error} ->
        {:error, error}

      {:error, %Req.TransportError{reason: :timeout} = err} ->
        {:error,
         LangChainError.exception(type: "timeout", message: "Request timed out", original: err)}

      {:error, %Req.TransportError{reason: reason}} ->
        {:error,
         LangChainError.exception(
           type: "connection",
           message: "Mantle stream connection error: #{inspect(reason)}"
         )}

      other ->
        {:error,
         LangChainError.exception(
           type: "unexpected_response",
           message: "Unexpected streamed response: #{inspect(other)}"
         )}
    end
  end

  # ---------------------------------------------------------------------------
  # Response parsing — delegates the OpenAI-shaped bits to ChatOpenAI then
  # extracts the Mantle-specific `message.reasoning` field into a thinking
  # ContentPart.
  # ---------------------------------------------------------------------------

  @doc false
  # Streaming delta chunk — matched first because each choice has a "delta" key.
  def do_process_response(
        %ChatAwsMantle{} = model,
        %{"choices" => [%{"delta" => _} | _] = choices} = msg
      ) do
    choices
    |> Enum.flat_map(&process_stream_choice(model, &1, msg))
    |> case do
      [] -> :skip
      [single] -> single
      many -> many
    end
  end

  # Non-streaming complete response — every choice carries a fully-formed "message".
  def do_process_response(%ChatAwsMantle{} = model, %{"choices" => choices} = body)
      when is_list(choices) and choices != [] do
    choices
    |> Enum.map(&process_choice(model, &1, body))
    |> Enum.reject(&is_nil/1)
  end

  # Usage-only terminal event (when stream_options.include_usage is set, Mantle
  # sends a final chunk with empty `choices` and a `usage` map).
  def do_process_response(_model, %{"choices" => [], "usage" => _} = _msg) do
    :skip
  end

  def do_process_response(_model, %{"error" => %{"message" => message}} = body) do
    {:error,
     LangChainError.exception(
       type: Map.get(body, "code") || "api_error",
       message: message
     )}
  end

  def do_process_response(_model, other) do
    {:error,
     LangChainError.exception(
       type: "unexpected_response",
       message: "Unexpected response shape: #{inspect(other)}"
     )}
  end

  defp process_choice(model, %{"message" => message_data} = choice, body) do
    # Hand off to ChatOpenAI's parser for the standard OpenAI-shaped fields
    # (content, tool_calls, role, etc.), then layer reasoning extraction on top.
    case ChatOpenAI.do_process_response(model, choice) do
      %Message{} = msg ->
        msg
        |> maybe_add_reasoning(message_data)
        |> attach_usage(body)

      other ->
        other
    end
  end

  defp process_choice(_model, _choice, _body), do: nil

  # Per-choice streaming delta processing. Returns a list of MessageDelta
  # structs (zero, one, or two). The list form lets one SSE event emit both
  # a thinking delta and a content/tool_calls delta when they co-occur.
  #
  # Positioning policy: thinking fragments land at MessageDelta.index 0;
  # text fragments land at index 1. That maps directly to merged_content
  # positions during merge so the final message reads [thinking, text].
  defp process_stream_choice(model, %{"delta" => delta_body} = choice, _msg) do
    role = role_for_delta(delta_body)
    finish_reason = Map.get(choice, "finish_reason")
    status = finish_reason_to_status(finish_reason)
    reasoning = Map.get(delta_body, "reasoning")
    content = Map.get(delta_body, "content")
    tool_calls_raw = Map.get(delta_body, "tool_calls")

    deltas =
      []
      |> maybe_reasoning_delta(reasoning, role, status)
      |> maybe_content_delta(content, role, status)
      |> maybe_tool_calls_delta(model, tool_calls_raw, role, status)

    case deltas do
      [] ->
        # Role-only / keep-alive / terminal delta with no payload.
        if role == :assistant or status in [:complete, :length] do
          [build_delta(%{role: role, status: status, index: 0})]
        else
          []
        end

      _ ->
        deltas
    end
  end

  defp process_stream_choice(_model, _choice, _msg), do: []

  defp maybe_reasoning_delta(acc, nil, _role, _status), do: acc
  defp maybe_reasoning_delta(acc, "", _role, _status), do: acc

  defp maybe_reasoning_delta(acc, reasoning, role, status) when is_binary(reasoning) do
    acc ++
      [
        build_delta(%{
          content: ContentPart.thinking!(reasoning),
          role: role,
          status: status,
          index: 0
        })
      ]
  end

  defp maybe_content_delta(acc, nil, _role, _status), do: acc
  defp maybe_content_delta(acc, "", _role, _status), do: acc

  defp maybe_content_delta(acc, content, role, status) when is_binary(content) do
    acc ++
      [
        build_delta(%{
          content: content,
          role: role,
          status: status,
          index: 1
        })
      ]
  end

  defp maybe_tool_calls_delta(acc, _model, nil, _role, _status), do: acc
  defp maybe_tool_calls_delta(acc, _model, [], _role, _status), do: acc

  defp maybe_tool_calls_delta(acc, model, tool_calls_raw, role, status)
       when is_list(tool_calls_raw) do
    tool_calls = Enum.map(tool_calls_raw, &ChatOpenAI.do_process_response(model, &1))

    acc ++
      [
        build_delta(%{
          role: role,
          status: status,
          index: 0,
          tool_calls: tool_calls
        })
      ]
  end

  defp build_delta(attrs) do
    attrs =
      attrs
      |> Map.put_new(:role, :unknown)
      |> Map.put_new(:status, :incomplete)

    struct!(MessageDelta, attrs)
  end

  defp role_for_delta(%{"role" => "assistant"}), do: :assistant
  defp role_for_delta(_), do: :unknown

  defp finish_reason_to_status(nil), do: :incomplete
  defp finish_reason_to_status("stop"), do: :complete
  defp finish_reason_to_status("tool_calls"), do: :complete
  defp finish_reason_to_status("content_filter"), do: :complete
  defp finish_reason_to_status("length"), do: :length
  defp finish_reason_to_status("max_tokens"), do: :length
  defp finish_reason_to_status(_other), do: nil

  # If Mantle returned `message.reasoning`, add it as a leading thinking
  # ContentPart on the message.
  defp maybe_add_reasoning(%Message{} = msg, %{"reasoning" => reasoning})
       when is_binary(reasoning) and byte_size(reasoning) > 0 do
    thinking_part = ContentPart.thinking!(reasoning)
    %{msg | content: [thinking_part | msg.content || []]}
  end

  defp maybe_add_reasoning(%Message{} = msg, _), do: msg

  defp attach_usage(%Message{} = msg, %{"usage" => usage}) when is_map(usage) do
    token_usage = %LangChain.TokenUsage{
      input: Map.get(usage, "prompt_tokens"),
      output: Map.get(usage, "completion_tokens"),
      raw: usage
    }

    metadata = Map.merge(msg.metadata || %{}, %{usage: token_usage})
    %{msg | metadata: metadata}
  end

  defp attach_usage(%Message{} = msg, _), do: msg

  # Resolve API key from struct or app config. Currently unused (auth_opts/1
  # uses the struct field directly), kept for future config-based auth.
  @doc false
  @spec get_api_key(t()) :: String.t() | nil
  def get_api_key(%ChatAwsMantle{api_key: api_key}) when is_binary(api_key), do: api_key
  def get_api_key(%ChatAwsMantle{}), do: Config.resolve(:aws_bearer_token_bedrock, nil)
end
