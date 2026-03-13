if Code.ensure_loaded?(ReqLLM) do
  defmodule LangChain.ChatModels.ChatReqLLM do
    @moduledoc """
    ChatModel adapter using the `req_llm` library as the HTTP/LLM backend.

    Provides access to any provider supported by req_llm (Anthropic, OpenAI, Google
    Gemini, Groq, Ollama, AWS Bedrock, etc.) through the unified LangChain framework.

    ## Model Specification

    The `model` field takes a req_llm-format specifier string: `"provider:model_id"`.

    ## Usage

        alias LangChain.ChatModels.ChatReqLLM
        alias LangChain.Chains.LLMChain
        alias LangChain.Message

        # Anthropic via req_llm
        llm = ChatReqLLM.new!(%{model: "anthropic:claude-haiku-4-5"})

        # OpenAI
        llm = ChatReqLLM.new!(%{model: "openai:gpt-4o"})

        # Ollama local model
        llm = ChatReqLLM.new!(%{model: "ollama:llama3", base_url: "http://localhost:11434"})

        # Groq with streaming
        llm = ChatReqLLM.new!(%{model: "groq:llama-3.3-70b-versatile", stream: true})

        {:ok, chain} =
          %{llm: llm}
          |> LLMChain.new!()
          |> LLMChain.add_message(Message.new_user!("Hello!"))
          |> LLMChain.run()

    ## Tool Use

    Tools are translated to req_llm format automatically. The `callback` field in the
    req_llm Tool struct is set to a stub — tool execution remains the LLMChain's
    responsibility, as with all other ChatModel adapters.

    ## Provider Options

    Provider-specific options (e.g. `thinking`, `tool_choice`, `seed`) can be passed
    via `provider_opts`:

        ChatReqLLM.new!(%{
          model: "anthropic:claude-haiku-4-5",
          provider_opts: %{"thinking" => %{"type" => "enabled", "budget_tokens" => 2000}}
        })
    """

    use Ecto.Schema
    require Logger
    import Ecto.Changeset
    alias __MODULE__
    alias LangChain.ChatModels.ChatModel
    alias LangChain.LangChainError
    alias LangChain.Message
    alias LangChain.Message.ContentPart
    alias LangChain.Message.ToolCall
    alias LangChain.Message.ToolResult
    alias LangChain.MessageDelta
    alias LangChain.TokenUsage
    alias LangChain.Function
    alias LangChain.Callbacks
    alias LangChain.Utils

    @behaviour ChatModel

    @current_config_version 1

    @primary_key false
    embedded_schema do
      # Required: req_llm model specifier, e.g. "anthropic:claude-haiku-4-5"
      field :model, :string

      # Optional: override API key (if nil, req_llm uses its layered key resolution)
      field :api_key, :string, redact: true

      # Optional: override base URL (useful for Ollama, Azure, VLLM, etc.)
      field :base_url, :string

      # Stream the response? (Phase 2: streaming support)
      field :stream, :boolean, default: false

      # Max tokens for the response
      field :max_tokens, :integer

      # Temperature (0.0–2.0 depending on provider)
      field :temperature, :float

      # Receive timeout in ms for non-streaming requests
      field :receive_timeout, :integer, default: 60_000

      # Pass-through opts forwarded verbatim to req_llm calls.
      # Allows provider-specific options: thinking, tool_choice, seed, top_p, etc.
      field :provider_opts, :map, default: %{}

      # Callbacks for LLM events (internal, treated as private)
      field :callbacks, {:array, :map}, default: []

      # Log raw req_llm requests/responses for debugging
      field :verbose_api, :boolean, default: false

      # Req options merged into the underlying Req.Request (advanced use)
      field :req_opts, :any, virtual: true, default: []
    end

    @type t :: %ChatReqLLM{}

    @create_fields [
      :model,
      :api_key,
      :base_url,
      :stream,
      :max_tokens,
      :temperature,
      :receive_timeout,
      :provider_opts,
      :verbose_api,
      :req_opts
    ]

    @required_fields [:model]

    @doc """
    Create a ChatReqLLM configuration.
    """
    @spec new(attrs :: map()) :: {:ok, t()} | {:error, Ecto.Changeset.t()}
    def new(attrs \\ %{}) do
      %ChatReqLLM{}
      |> cast(attrs, @create_fields)
      |> common_validation()
      |> apply_action(:insert)
    end

    @doc """
    Create a ChatReqLLM configuration, raising on error if invalid.
    """
    @spec new!(attrs :: map()) :: t() | no_return()
    def new!(attrs \\ %{}) do
      case new(attrs) do
        {:ok, model} -> model
        {:error, changeset} -> raise LangChainError, changeset
      end
    end

    defp common_validation(changeset) do
      changeset
      |> validate_required(@required_fields)
      |> validate_length(:model, min: 1)
      |> validate_number(:temperature, greater_than_or_equal_to: 0, less_than_or_equal_to: 2)
      |> validate_number(:max_tokens, greater_than: 0)
      |> validate_number(:receive_timeout, greater_than_or_equal_to: 0)
    end

    # ============================================================
    # ChatModel behaviour implementation
    # ============================================================

    @doc """
    Call the LLM via req_llm with a prompt or list of messages.
    """
    @impl ChatModel
    def call(model, prompt, functions \\ [])

    def call(%ChatReqLLM{} = model, prompt, functions) when is_binary(prompt) do
      messages = [
        Message.new_system!(),
        Message.new_user!(prompt)
      ]

      call(model, messages, functions)
    end

    def call(%ChatReqLLM{} = model, messages, functions) when is_list(messages) do
      metadata = %{
        model: model.model,
        message_count: length(messages),
        tools_count: length(functions)
      }

      LangChain.Telemetry.span([:langchain, :llm, :call], metadata, fn ->
        try do
          LangChain.Telemetry.llm_prompt(
            %{system_time: System.system_time()},
            %{model: model.model, messages: messages}
          )

          case do_api_request(model, messages, functions) do
            {:error, %LangChainError{} = error} ->
              {:error, error}

            parsed_data ->
              LangChain.Telemetry.llm_response(
                %{system_time: System.system_time()},
                %{model: model.model, response: parsed_data}
              )

              {:ok, parsed_data}
          end
        rescue
          err in LangChainError ->
            {:error, err}
        end
      end)
    end

    @doc """
    Determine if an error should be retried via a fallback LLM.
    """
    @impl ChatModel
    @spec retry_on_fallback?(LangChainError.t()) :: boolean()
    def retry_on_fallback?(%LangChainError{type: "rate_limit_exceeded"}), do: true
    def retry_on_fallback?(%LangChainError{type: "rate_limited"}), do: true
    def retry_on_fallback?(%LangChainError{type: "overloaded"}), do: true
    def retry_on_fallback?(%LangChainError{type: "timeout"}), do: true
    def retry_on_fallback?(_), do: false

    @impl ChatModel
    def serialize_config(%ChatReqLLM{} = model) do
      Utils.to_serializable_map(
        model,
        [:model, :stream, :max_tokens, :temperature, :receive_timeout, :base_url, :provider_opts],
        @current_config_version
      )
    end

    @impl ChatModel
    def restore_from_map(%{"version" => 1} = data) do
      ChatReqLLM.new(data)
    end

    # ============================================================
    # API Request
    # ============================================================

    @doc false
    @spec do_api_request(t(), [Message.t()], ChatModel.tools(), non_neg_integer()) ::
            Message.t() | {:error, LangChainError.t()} | no_return()
    def do_api_request(model, messages, tools, retry_count \\ 3)

    def do_api_request(_model, _messages, _tools, 0) do
      raise LangChainError,
        type: "retries_exceeded",
        message: "Retries exceeded. Connection failed."
    end

    def do_api_request(%ChatReqLLM{stream: false} = model, messages, tools, retry_count) do
      context = messages_to_req_llm_context(messages)
      req_llm_tools = functions_to_req_llm_tools(tools)
      opts = build_req_llm_opts(model, req_llm_tools)

      if model.verbose_api do
        IO.inspect(context, label: "CHAT_REQ_LLM CONTEXT")
        IO.inspect(opts, label: "CHAT_REQ_LLM OPTS")
      end

      case ReqLLM.generate_text(model.model, context, opts) do
        {:ok, %ReqLLM.Response{} = response} ->
          if model.verbose_api do
            IO.inspect(response, label: "CHAT_REQ_LLM RAW RESPONSE")
          end

          case do_process_response(model, response) do
            {:error, _reason} = error ->
              error

            result ->
              Callbacks.fire(model.callbacks, :on_llm_new_message, [result])
              result
          end

        {:error, %Req.TransportError{reason: :timeout} = err} ->
          {:error,
           LangChainError.exception(type: "timeout", message: "Request timed out", original: err)}

        {:error, %Req.TransportError{reason: :closed}} ->
          Logger.debug(fn ->
            "Mint connection closed: retry count = #{inspect(retry_count)}"
          end)

          do_api_request(model, messages, tools, retry_count - 1)

        {:error, %LangChainError{}} = error ->
          error

        {:error, error} ->
          translate_req_llm_error(error)

        other ->
          Logger.error("Unexpected response from ReqLLM: #{inspect(other)}")

          {:error,
           LangChainError.exception(
             type: "unexpected_response",
             message: "Unexpected response",
             original: other
           )}
      end
    end

    def do_api_request(%ChatReqLLM{stream: true} = model, messages, tools, retry_count) do
      context = messages_to_req_llm_context(messages)
      req_llm_tools = functions_to_req_llm_tools(tools)
      opts = build_req_llm_opts(model, req_llm_tools)

      if model.verbose_api do
        IO.inspect(context, label: "CHAT_REQ_LLM STREAM CONTEXT")
        IO.inspect(opts, label: "CHAT_REQ_LLM STREAM OPTS")
      end

      LangChain.Telemetry.llm_prompt(
        %{system_time: System.system_time(), streaming: true},
        %{model: model.model, messages: messages}
      )

      case ReqLLM.stream_text(model.model, context, opts) do
        {:ok, stream_response} ->
          # Stateful reduce: assigns a monotonic content index to each new content
          # block type (thinking, text, etc.) so that MessageDelta merging places
          # each type in the correct merged_content slot. Tool call argument
          # fragments are emitted incrementally (mirroring ChatAnthropic's approach).
          initial_state = %{next_content_index: 0, type_index_map: %{}}

          {all_deltas, _final_state} =
            stream_response.stream
            |> Enum.reduce({[], initial_state}, fn chunk, {acc_deltas, state} ->
              {new_deltas, new_state} = process_stream_chunk(chunk, state)
              if new_deltas != [], do: Utils.fire_streamed_callback(model, new_deltas)
              {acc_deltas ++ new_deltas, new_state}
            end)

          LangChain.Telemetry.emit_event(
            [:langchain, :llm, :response],
            %{system_time: System.system_time()},
            %{model: model.model, streaming: true}
          )

          all_deltas

        {:error, %Req.TransportError{reason: :timeout} = err} ->
          {:error,
           LangChainError.exception(type: "timeout", message: "Request timed out", original: err)}

        {:error, %Req.TransportError{reason: :closed}} ->
          Logger.debug(fn ->
            "Mint connection closed: retry count = #{inspect(retry_count)}"
          end)

          do_api_request(model, messages, tools, retry_count - 1)

        {:error, %LangChainError{}} = error ->
          error

        {:error, error} ->
          translate_req_llm_error(error)

        other ->
          Logger.error("Unexpected response from ReqLLM stream: #{inspect(other)}")

          {:error,
           LangChainError.exception(
             type: "unexpected_response",
             message: "Unexpected response from stream",
             original: other
           )}
      end
    end

    # Process a stream chunk with state tracking.
    # Assigns a monotonic content index per chunk type so each content block type
    # (thinking, text, image, etc.) gets a stable merged_content slot.
    # Emits tool call start and argument fragment deltas incrementally (mirroring ChatAnthropic).

    # Thinking chunk: get or assign a stable index for :thinking blocks
    defp process_stream_chunk(
           %ReqLLM.StreamChunk{type: :thinking, text: text},
           state
         )
         when is_binary(text) do
      {index, new_state} = get_or_assign_content_index(state, :thinking)

      delta =
        MessageDelta.new!(%{
          role: :assistant,
          content: ContentPart.new!(%{type: :thinking, content: text}),
          status: :incomplete,
          index: index
        })

      {[delta], new_state}
    end

    # Text content: get or assign a stable index for :content blocks
    defp process_stream_chunk(
           %ReqLLM.StreamChunk{type: :content, text: text},
           state
         )
         when is_binary(text) and text != "" do
      {index, new_state} = get_or_assign_content_index(state, :content)

      delta =
        MessageDelta.new!(%{
          role: :assistant,
          content: ContentPart.text!(text),
          status: :incomplete,
          index: index
        })

      {[delta], new_state}
    end

    # Tool call start (Anthropic streaming: metadata has start: true):
    # emit initial incomplete ToolCall delta with name/id so UI can show tool in progress
    defp process_stream_chunk(
           %ReqLLM.StreamChunk{type: :tool_call, name: name, metadata: %{start: true} = meta},
           state
         )
         when is_binary(name) do
      id = meta[:id] || "tool_#{:erlang.unique_integer([:positive])}"
      block_index = meta[:index] || 0

      tool_call =
        ToolCall.new!(%{
          type: :function,
          status: :incomplete,
          call_id: id,
          name: name,
          index: block_index
        })

      delta =
        MessageDelta.new!(%{
          role: :assistant,
          tool_calls: [tool_call],
          status: :incomplete,
          index: 0
        })

      {[delta], state}
    end

    # Tool call arg fragment: emit incomplete ToolCall delta with the partial JSON string.
    # ToolCall.merge/2 will concatenate binary arguments strings across deltas.
    defp process_stream_chunk(
           %ReqLLM.StreamChunk{
             type: :meta,
             metadata: %{tool_call_args: %{index: block_index, fragment: fragment}}
           },
           state
         )
         when is_binary(fragment) and fragment != "" do
      tool_call =
        ToolCall.new!(%{
          type: :function,
          status: :incomplete,
          arguments: fragment,
          index: block_index
        })

      delta =
        MessageDelta.new!(%{
          role: :assistant,
          tool_calls: [tool_call],
          status: :incomplete,
          index: 0
        })

      {[delta], state}
    end

    # All other chunks: delegate to stateless translation
    defp process_stream_chunk(chunk, state) do
      {translate_stream_chunk(chunk), state}
    end

    # Assigns a monotonic content index per chunk type. The first time a chunk type
    # is seen it gets the next available index; subsequent chunks of the same type
    # reuse that index so MessageDelta merging accumulates into the correct slot.
    defp get_or_assign_content_index(state, chunk_type) do
      case Map.get(state.type_index_map, chunk_type) do
        nil ->
          index = state.next_content_index

          new_state = %{
            state
            | next_content_index: index + 1,
              type_index_map: Map.put(state.type_index_map, chunk_type, index)
          }

          {index, new_state}

        existing_index ->
          {existing_index, state}
      end
    end

    @doc """
    Translate a single `ReqLLM.StreamChunk` to a list of `LangChain.MessageDelta` structs.

    Returns an empty list for chunks that produce no LangChain deltas (e.g. empty content,
    non-terminal metadata).
    """
    @spec translate_stream_chunk(ReqLLM.StreamChunk.t()) :: [MessageDelta.t()]
    def translate_stream_chunk(%ReqLLM.StreamChunk{type: :content, text: text})
        when is_binary(text) and text != "" do
      delta =
        MessageDelta.new!(%{
          role: :assistant,
          content: ContentPart.text!(text),
          status: :incomplete,
          index: 0
        })

      [delta]
    end

    def translate_stream_chunk(%ReqLLM.StreamChunk{type: :thinking, text: text})
        when is_binary(text) do
      delta =
        MessageDelta.new!(%{
          role: :assistant,
          content: ContentPart.new!(%{type: :thinking, content: text}),
          status: :incomplete,
          index: 0
        })

      [delta]
    end

    def translate_stream_chunk(%ReqLLM.StreamChunk{
          type: :tool_call,
          name: name,
          arguments: args,
          metadata: meta
        })
        when is_binary(name) do
      id = (meta || %{})[:id] || "tool_#{:erlang.unique_integer([:positive])}"

      args_map = if is_map(args), do: args, else: %{}

      tool_call =
        ToolCall.new!(%{
          type: :function,
          status: :complete,
          call_id: id,
          name: name,
          arguments: args_map
        })

      delta =
        MessageDelta.new!(%{
          role: :assistant,
          tool_calls: [tool_call],
          status: :incomplete,
          index: 0
        })

      [delta]
    end

    def translate_stream_chunk(%ReqLLM.StreamChunk{type: :meta, metadata: meta})
        when is_map(meta) do
      usage_deltas =
        case meta[:usage] do
          nil ->
            []

          usage_map ->
            case translate_usage(usage_map) do
              nil ->
                []

              token_usage ->
                [MessageDelta.new!(%{role: :assistant, metadata: %{usage: token_usage}})]
            end
        end

      finish_deltas =
        if meta[:terminal?] do
          status = translate_finish_reason(meta[:finish_reason])
          [MessageDelta.new!(%{role: :assistant, status: status, index: 0})]
        else
          []
        end

      usage_deltas ++ finish_deltas
    end

    def translate_stream_chunk(_), do: []

    defp build_req_llm_opts(%ChatReqLLM{} = model, tools) do
      []
      |> then(fn opts ->
        if tools != [], do: Keyword.put(opts, :tools, tools), else: opts
      end)
      |> maybe_put(:max_tokens, model.max_tokens)
      |> maybe_put(:temperature, model.temperature)
      |> maybe_put(:api_key, model.api_key)
      |> maybe_put(:base_url, model.base_url)
      |> merge_provider_opts(model.provider_opts)
    end

    defp maybe_put(opts, _key, nil), do: opts
    defp maybe_put(opts, key, value), do: Keyword.put(opts, key, value)

    defp merge_provider_opts(opts, nil), do: opts
    defp merge_provider_opts(opts, provider_opts) when provider_opts == %{}, do: opts

    defp merge_provider_opts(opts, provider_opts) when is_map(provider_opts) do
      extra =
        Enum.map(provider_opts, fn {k, v} ->
          key = if is_binary(k), do: String.to_atom(k), else: k
          {key, v}
        end)

      Keyword.merge(opts, extra)
    end

    defp translate_req_llm_error(error) do
      case error do
        %{status: 401} ->
          {:error,
           LangChainError.exception(
             type: "authentication_error",
             message: "Authentication failed",
             original: error
           )}

        %{status: 429} ->
          {:error,
           LangChainError.exception(
             type: "rate_limit_exceeded",
             message: "Rate limit exceeded",
             original: error
           )}

        %{status: 529} ->
          {:error,
           LangChainError.exception(
             type: "overloaded",
             message: "Service overloaded",
             original: error
           )}

        %{status: status} when is_integer(status) and status >= 500 ->
          {:error,
           LangChainError.exception(
             type: "server_error",
             message: "Server error",
             original: error
           )}

        _ ->
          Logger.error("Unhandled error from ReqLLM: #{inspect(error)}")

          {:error,
           LangChainError.exception(
             type: "unhandled_error",
             message: "Unhandled error from ReqLLM",
             original: error
           )}
      end
    end

    # ============================================================
    # Outbound Translation: LangChain → ReqLLM
    # ============================================================

    @doc """
    Convert a list of LangChain messages to a `ReqLLM.Context`.

    Tool messages are expanded: a single LangChain `:tool` message (which may carry
    multiple `ToolResult` structs) becomes one `ReqLLM.Message` per result, matching
    the one-result-per-message convention expected by OpenAI-compatible providers.
    """
    @spec messages_to_req_llm_context([Message.t()]) :: ReqLLM.Context.t()
    def messages_to_req_llm_context(messages) do
      req_llm_messages =
        messages
        |> Enum.flat_map(&message_to_req_llm_messages/1)

      ReqLLM.Context.new(req_llm_messages)
    end

    @doc """
    Convert a single LangChain `Message` to a list of `ReqLLM.Message` structs.

    Most roles map 1-to-1. The `:tool` role expands to one message per `ToolResult`.
    """
    @spec message_to_req_llm_messages(Message.t()) :: [ReqLLM.Message.t()]
    def message_to_req_llm_messages(%Message{role: :tool, tool_results: results})
        when is_list(results) do
      Enum.map(results, fn %ToolResult{} = result ->
        content = tool_result_content_to_req_llm(result.content)

        %ReqLLM.Message{
          role: :tool,
          tool_call_id: result.tool_call_id,
          content: content
        }
      end)
    end

    def message_to_req_llm_messages(%Message{role: :assistant, tool_calls: calls} = msg)
        when is_list(calls) and calls != [] do
      content = lc_content_to_req_llm(msg.content)
      req_tool_calls = Enum.map(calls, &lc_tool_call_to_req_llm/1)

      [%ReqLLM.Message{role: :assistant, content: content, tool_calls: req_tool_calls}]
    end

    def message_to_req_llm_messages(%Message{} = msg) do
      content = lc_content_to_req_llm(msg.content)
      [%ReqLLM.Message{role: msg.role, content: content}]
    end

    defp lc_content_to_req_llm(nil), do: []

    defp lc_content_to_req_llm(content) when is_binary(content) do
      [ReqLLM.Message.ContentPart.text(content)]
    end

    defp lc_content_to_req_llm(parts) when is_list(parts) do
      parts
      |> Enum.map(&content_part_to_req_llm/1)
      |> Enum.reject(&is_nil/1)
    end

    @doc """
    Convert a LangChain `ContentPart` to a `ReqLLM.Message.ContentPart`.

    Returns `nil` for unsupported types (they are filtered out of the content list).
    """
    @spec content_part_to_req_llm(ContentPart.t()) :: ReqLLM.Message.ContentPart.t() | nil
    def content_part_to_req_llm(%ContentPart{type: :text, content: text}) do
      ReqLLM.Message.ContentPart.text(text || "")
    end

    def content_part_to_req_llm(%ContentPart{type: :thinking, content: text}) do
      ReqLLM.Message.ContentPart.thinking(text || "")
    end

    def content_part_to_req_llm(%ContentPart{type: :image_url, content: url}) do
      ReqLLM.Message.ContentPart.image_url(url)
    end

    def content_part_to_req_llm(%ContentPart{type: :image, content: b64, options: opts}) do
      media_type = opts |> Keyword.get(:media, :png) |> media_to_mime()
      decoded = Base.decode64!(b64)
      ReqLLM.Message.ContentPart.image(decoded, media_type)
    end

    def content_part_to_req_llm(%ContentPart{type: :file, content: b64, options: opts}) do
      media_type =
        (opts || []) |> Keyword.get(:media, "application/octet-stream") |> media_to_mime()

      filename = (opts || []) |> Keyword.get(:filename, "file")
      decoded = Base.decode64!(b64)
      ReqLLM.Message.ContentPart.file(decoded, filename, media_type)
    end

    def content_part_to_req_llm(%ContentPart{type: :file_url, content: url}) do
      Logger.warning(
        "ContentPart type :file_url is not directly supported by ReqLLM; converting to text URL reference"
      )

      ReqLLM.Message.ContentPart.text("URL: #{url}")
    end

    def content_part_to_req_llm(%ContentPart{type: :unsupported}) do
      Logger.warning("Unsupported ContentPart type skipped during ChatReqLLM translation")
      nil
    end

    defp tool_result_content_to_req_llm(nil) do
      [ReqLLM.Message.ContentPart.text("")]
    end

    defp tool_result_content_to_req_llm(content) when is_binary(content) do
      [ReqLLM.Message.ContentPart.text(content)]
    end

    defp tool_result_content_to_req_llm(parts) when is_list(parts) do
      parts
      |> Enum.map(&content_part_to_req_llm/1)
      |> Enum.reject(&is_nil/1)
    end

    defp lc_tool_call_to_req_llm(%ToolCall{call_id: id, name: name, arguments: args}) do
      args_json = Jason.encode!(args || %{})
      ReqLLM.ToolCall.new(id, name, args_json)
    end

    @doc """
    Convert a list of LangChain `Function` structs to `ReqLLM.Tool` structs.

    Each tool gets a stub callback — tool execution remains the LLMChain's responsibility.
    """
    @spec functions_to_req_llm_tools([Function.t()] | nil) :: [ReqLLM.Tool.t()]
    def functions_to_req_llm_tools(nil), do: []
    def functions_to_req_llm_tools([]), do: []

    def functions_to_req_llm_tools(functions) when is_list(functions) do
      Enum.map(functions, &function_to_req_llm_tool/1)
    end

    @doc """
    Convert a single `LangChain.Function` to a `ReqLLM.Tool` with a stub callback.

    The stub callback is never invoked in normal LangChain operation — the tool
    definition is only used for schema generation (telling the LLM what tools exist).
    """
    @spec function_to_req_llm_tool(Function.t()) :: ReqLLM.Tool.t()
    def function_to_req_llm_tool(%Function{} = fun) do
      schema = fun.parameters_schema || %{}

      ReqLLM.Tool.new!(
        name: fun.name,
        description: fun.description || "",
        parameter_schema: schema,
        callback: fn _ -> {:ok, "stub"} end
      )
    end

    defp media_to_mime(:png), do: "image/png"
    defp media_to_mime(:jpg), do: "image/jpeg"
    defp media_to_mime(:jpeg), do: "image/jpeg"
    defp media_to_mime(:gif), do: "image/gif"
    defp media_to_mime(:webp), do: "image/webp"
    defp media_to_mime(:pdf), do: "application/pdf"
    defp media_to_mime(:text), do: "text/plain"
    defp media_to_mime(s) when is_binary(s), do: s
    defp media_to_mime(_), do: "application/octet-stream"

    # ============================================================
    # Inbound Translation: ReqLLM → LangChain
    # ============================================================

    @doc """
    Convert a `ReqLLM.Response` to a `LangChain.Message`.
    """
    @spec do_process_response(t(), ReqLLM.Response.t()) ::
            Message.t() | {:error, LangChainError.t()}
    def do_process_response(%ChatReqLLM{} = _model, %ReqLLM.Response{error: error})
        when not is_nil(error) do
      Logger.error("ReqLLM returned error in response body: #{inspect(error)}")

      {:error,
       LangChainError.exception(
         type: "api_error",
         message: "API returned an error",
         original: error
       )}
    end

    def do_process_response(%ChatReqLLM{} = _model, %ReqLLM.Response{message: nil}) do
      {:error,
       LangChainError.exception(
         type: "unexpected_response",
         message: "Response contained no message"
       )}
    end

    def do_process_response(%ChatReqLLM{} = _model, %ReqLLM.Response{} = response) do
      content_parts = translate_response_content(response.message.content)
      tool_calls = translate_response_tool_calls(response.message.tool_calls)
      status = translate_finish_reason(response.finish_reason)
      usage = translate_usage(response.usage)

      %{
        role: :assistant,
        content: content_parts,
        tool_calls: tool_calls,
        status: status
      }
      |> Message.new()
      |> TokenUsage.set_wrapped(usage)
      |> unwrap_message()
    end

    defp unwrap_message({:ok, message}), do: message

    defp unwrap_message({:error, %Ecto.Changeset{} = changeset}) do
      {:error, LangChainError.exception(changeset)}
    end

    defp translate_response_content(nil), do: []
    defp translate_response_content([]), do: []

    defp translate_response_content(parts) when is_list(parts) do
      parts
      |> Enum.map(&req_llm_content_part_to_lc/1)
      |> Enum.reject(&is_nil/1)
    end

    defp req_llm_content_part_to_lc(%ReqLLM.Message.ContentPart{type: :text, text: text}) do
      ContentPart.text!(text || "")
    end

    defp req_llm_content_part_to_lc(%ReqLLM.Message.ContentPart{
           type: :thinking,
           text: text,
           metadata: meta
         }) do
      signature = (meta || %{})[:signature]
      opts = if signature, do: [signature: signature], else: []
      ContentPart.new!(%{type: :thinking, content: text, options: opts})
    end

    defp req_llm_content_part_to_lc(%ReqLLM.Message.ContentPart{type: :image_url, url: url}) do
      ContentPart.new!(%{type: :image_url, content: url})
    end

    defp req_llm_content_part_to_lc(%ReqLLM.Message.ContentPart{
           type: :image,
           data: data,
           media_type: media_type
         }) do
      ContentPart.new!(%{
        type: :image,
        content: Base.encode64(data),
        options: [media: media_type]
      })
    end

    defp req_llm_content_part_to_lc(%ReqLLM.Message.ContentPart{
           type: :file,
           data: data,
           media_type: media_type,
           filename: filename
         }) do
      ContentPart.new!(%{
        type: :file,
        content: Base.encode64(data),
        options: [media: media_type, filename: filename]
      })
    end

    defp req_llm_content_part_to_lc(other) do
      Logger.warning("Unknown ReqLLM ContentPart type skipped: #{inspect(other)}")
      nil
    end

    defp translate_response_tool_calls(nil), do: nil
    defp translate_response_tool_calls([]), do: nil

    defp translate_response_tool_calls(tool_calls) when is_list(tool_calls) do
      Enum.map(tool_calls, fn %ReqLLM.ToolCall{
                                id: id,
                                function: %{name: name, arguments: args_json}
                              } ->
        arguments =
          case Jason.decode(args_json || "{}") do
            {:ok, map} when is_map(map) -> map
            _ -> %{}
          end

        ToolCall.new!(%{
          type: :function,
          status: :complete,
          call_id: id,
          name: name,
          arguments: arguments
        })
      end)
    end

    @doc """
    Translate a `req_llm` `finish_reason` atom to a `LangChain.Message` status atom.
    """
    @spec translate_finish_reason(atom() | nil) :: atom()
    def translate_finish_reason(:stop), do: :complete
    def translate_finish_reason(:tool_calls), do: :complete
    def translate_finish_reason(:length), do: :length
    def translate_finish_reason(:content_filter), do: :complete
    def translate_finish_reason(:cancelled), do: :complete
    def translate_finish_reason(:incomplete), do: :length
    def translate_finish_reason(:error), do: :complete
    def translate_finish_reason(:unknown), do: :complete
    def translate_finish_reason(nil), do: :complete

    def translate_finish_reason(other) do
      Logger.warning("Unknown finish_reason from ReqLLM: #{inspect(other)}")
      :complete
    end

    @doc """
    Translate a `req_llm` usage map to a `LangChain.TokenUsage` struct.
    """
    @spec translate_usage(map() | nil) :: TokenUsage.t() | nil
    def translate_usage(nil), do: nil

    def translate_usage(usage) when is_map(usage) do
      input = usage[:input_tokens] || 0
      output = usage[:output_tokens] || 0

      case TokenUsage.new(%{input: input, output: output, raw: usage}) do
        {:ok, token_usage} -> token_usage
        _ -> nil
      end
    end
  end
end
