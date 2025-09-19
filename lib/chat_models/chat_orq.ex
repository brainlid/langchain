defmodule LangChain.ChatModels.ChatOrq do
  @moduledoc """
  Chat adapter for orq.ai Deployments API.

  Non-streaming:
  - POST https://api.orq.ai/v2/deployments/invoke

  Streaming (SSE, sentinel "[DONE]"):
  - POST https://api.orq.ai/v2/deployments/stream

  Security:
  - HTTP Bearer token (Authorization: Bearer ...). Configure via application env :langchain, :orq_key

  Required body field:
  - key: Deployment key to invoke.

  Messages:
  - Accepts roles: developer | system | user | assistant | tool
  - Content supports text, image_url, file, input_audio
  - Assistant may include tool_calls; Tool results are role: tool with tool_call_id

  Notes:
  - Azure is not supported.
  """
  use Ecto.Schema
  require Logger
  import Ecto.Changeset

  alias __MODULE__
  alias LangChain.Config
  alias LangChain.ChatModels.ChatModel
  alias LangChain.Message
  alias LangChain.Message.ContentPart
  alias LangChain.Message.ToolCall
  alias LangChain.Message.ToolResult
  alias LangChain.MessageDelta
  alias LangChain.TokenUsage
  alias LangChain.Function
  alias LangChain.FunctionParam
  alias LangChain.LangChainError
  alias LangChain.Utils
  alias LangChain.Callbacks

  @behaviour ChatModel

  @current_config_version 1

  # allow up to 1 minute for response (overrideable)
  @receive_timeout 60_000

  @primary_key false
  embedded_schema do
    # Full endpoints; stream uses dedicated endpoint
    field :endpoint, :string, default: "https://api.orq.ai/v2/deployments/invoke"
    field :stream_endpoint, :string, default: "https://api.orq.ai/v2/deployments/stream"

    # Deployment key (required in request body)
    field :key, :string

    # Optional: model label for telemetry/readability (not sent to API)
    field :model, :string, default: "orq"

    # API key for orq.ai. If not set, will use global orq_key
    field :api_key, :string, redact: true

    # Streaming flag
    field :stream, :boolean, default: false

    # Duration in milliseconds for the response to be received.
    field :receive_timeout, :integer, default: @receive_timeout

    # Request body optional fields (supported by both invoke and stream endpoints)
    field :inputs, :map, default: nil
    field :context, :map, default: nil
    field :prefix_messages, {:array, :map}, default: nil
    # optional passthrough when messages already mapped
    field :messages_passthrough, {:array, :map}, default: nil
    field :file_ids, {:array, :string}, default: nil
    field :metadata, :map, default: nil
    field :extra_params, :map, default: nil
    field :documents, {:array, :map}, default: nil
    field :invoke_options, :map, default: nil
    field :thread, :map, default: nil
    field :knowledge_filter, :map, default: nil

    # A list of maps for callback handlers (treated as internal)
    field :callbacks, {:array, :map}, default: []

    # For help with debugging. It outputs the RAW Req response received and the
    # RAW Elixir map being submitted to the API.
    field :verbose_api, :boolean, default: false
  end

  @type t :: %ChatOrq{}

  @create_fields [
    :endpoint,
    :stream_endpoint,
    :key,
    :model,
    :api_key,
    :stream,
    :receive_timeout,
    :inputs,
    :context,
    :prefix_messages,
    :messages_passthrough,
    :file_ids,
    :metadata,
    :extra_params,
    :documents,
    :invoke_options,
    :thread,
    :knowledge_filter,
    :callbacks,
    :verbose_api
  ]
  @required_fields [:endpoint, :stream_endpoint, :key]

  @spec get_api_key(t()) :: String.t()
  defp get_api_key(%ChatOrq{api_key: api_key}) do
    # if no API key is set default to `""` which will raise an API error
    api_key || Config.resolve(:orq_key, "")
  end

  @doc """
  Setup a ChatOrq client configuration.
  """
  @spec new(attrs :: map()) :: {:ok, t} | {:error, Ecto.Changeset.t()}
  def new(%{} = attrs \\ %{}) do
    %ChatOrq{}
    |> cast(attrs, @create_fields)
    |> common_validation()
    |> apply_action(:insert)
  end

  @doc """
  Setup a ChatOrq client configuration and return it or raise an error if invalid.
  """
  @spec new!(attrs :: map()) :: t() | no_return()
  def new!(attrs \\ %{}) do
    case new(attrs) do
      {:ok, chain} ->
        chain

      {:error, changeset} ->
        raise LangChainError, changeset
    end
  end

  defp common_validation(changeset) do
    changeset
    |> validate_required(@required_fields)
    |> validate_number(:receive_timeout, greater_than_or_equal_to: 0)
  end

  @doc """
  Return the params formatted for an API request.
  """
  @spec for_api(
          t | Message.t() | ToolCall.t() | ToolResult.t() | ContentPart.t(),
          message :: [map()],
          ChatModel.tools()
        ) :: %{
          atom() => any()
        }
  def for_api(%ChatOrq{} = orq, messages, tools) do
    base =
      %{
        key: orq.key,
        stream: orq.stream,
        # a single ToolResult can expand into multiple tool messages
        messages:
          messages
          |> Enum.reduce([], fn m, acc ->
            case for_api(orq, m) do
              %{} = data ->
                [data | acc]

              data when is_list(data) ->
                Enum.reverse(data) ++ acc
            end
          end)
          |> Enum.reverse()
      }
      |> Utils.conditionally_add_to_map(:inputs, orq.inputs)
      |> Utils.conditionally_add_to_map(:context, orq.context)
      |> Utils.conditionally_add_to_map(
        :prefix_messages,
        if(orq.prefix_messages, do: prefix_messages_for_api(orq, orq.prefix_messages), else: nil)
      )
      |> Utils.conditionally_add_to_map(
        :messages,
        if(orq.messages_passthrough, do: orq.messages_passthrough, else: nil)
      )
      |> Utils.conditionally_add_to_map(:file_ids, orq.file_ids)
      |> Utils.conditionally_add_to_map(:metadata, orq.metadata)
      |> Utils.conditionally_add_to_map(:extra_params, get_extra_params_with_tools(orq, tools))
      |> Utils.conditionally_add_to_map(:documents, orq.documents)
      |> Utils.conditionally_add_to_map(:invoke_options, orq.invoke_options)
      |> Utils.conditionally_add_to_map(:thread, orq.thread)
      |> Utils.conditionally_add_to_map(:knowledge_filter, orq.knowledge_filter)

    base
  end

  # Helper function to merge tools into extra_params
  defp get_extra_params_with_tools(%ChatOrq{} = orq, tools) do
    tools_for_api = get_tools_for_api(orq, tools)

    base_extra_params = orq.extra_params || %{}

    case tools_for_api do
      tools when is_list(tools) and length(tools) > 0 ->
        Map.put(base_extra_params, "tools", tools_for_api)

      _ ->
        if map_size(base_extra_params) > 0, do: base_extra_params, else: nil
    end
  end

  # Convert tools to the format expected by ORQ API (same as OpenAI format)
  defp get_tools_for_api(%_{} = _model, nil), do: []

  defp get_tools_for_api(%_{} = model, tools) do
    Enum.map(tools, fn
      %Function{} = function ->
        %{"type" => "function", "function" => for_api(model, function)}
    end)
  end

  # Message conversions (compatible with orq schema)
  @spec for_api(
          struct(),
          Message.t()
          | ToolCall.t()
          | ToolResult.t()
          | ContentPart.t()
          | Function.t()
        ) ::
          %{String.t() => any()} | [%{String.t() => any()}]
  def for_api(%_{} = model, %Message{content: content} = msg) when is_list(content) do
    %{
      "role" => msg.role,
      "content" => content_parts_to_string(content)
    }
    |> Utils.conditionally_add_to_map("name", msg.name)
    |> Utils.conditionally_add_to_map(
      "tool_calls",
      Enum.map(msg.tool_calls || [], &for_api(model, &1))
    )
  end

  def for_api(%_{} = model, %Message{role: :assistant, tool_calls: tool_calls} = msg)
      when is_list(tool_calls) do
    content =
      case msg.content do
        content when is_list(content) -> content_parts_to_string(content)
        content -> content
      end

    %{
      "role" => :assistant,
      "content" => content
    }
    |> Utils.conditionally_add_to_map("tool_calls", Enum.map(tool_calls, &for_api(model, &1)))
  end

  def for_api(%_{} = _model, %Message{role: :user, content: content} = msg)
      when is_list(content) do
    %{
      "role" => msg.role,
      "content" => content_parts_to_string(content)
    }
    |> Utils.conditionally_add_to_map("name", msg.name)
  end

  def for_api(%_{} = _model, %ToolResult{type: :function} = result) do
    # a ToolResult becomes a stand-alone %Message{role: :tool} response.
    %{
      "role" => :tool,
      "tool_call_id" => result.tool_call_id,
      "content" => content_parts_to_string(result.content)
    }
  end

  def for_api(%_{} = _model, %Message{role: :tool, tool_results: tool_results} = _msg)
      when is_list(tool_results) do
    # ToolResults turn into a list of tool messages
    Enum.map(tool_results, fn result ->
      %{
        "role" => :tool,
        "tool_call_id" => result.tool_call_id,
        "content" => content_parts_to_string(result.content)
      }
    end)
  end

  # ToolCall support
  def for_api(%_{} = _model, %ToolCall{type: :function} = fun) do
    %{
      "id" => fun.call_id,
      "type" => "function",
      "function" => %{
        "name" => fun.name,
        "arguments" => Jason.encode!(fun.arguments)
      }
    }
  end

  # Function support (for tools)
  def for_api(%_{} = _model, %Function{} = fun) do
    %{
      "name" => fun.name,
      "parameters" => get_parameters(fun)
    }
    |> Utils.conditionally_add_to_map("description", fun.description)
  end

  # Handle ContentPart structures directly
  def for_api(%_{} = _model, %ContentPart{} = part) do
    content_part_for_api(part)
  end

  @doc """
  Convert a list of ContentParts to the expected map of data for the API.
  """
  def content_parts_for_api(content_parts) when is_list(content_parts) do
    Enum.map(content_parts, &content_part_for_api(&1))
  end

  @doc """
  Convert a list of ContentParts to a string for tool results.
  ORQ API expects tool result content to be a string, not an array.
  """
  def content_parts_to_string(content_parts) when is_list(content_parts) do
    content_parts
    |> Enum.map(fn
      %ContentPart{type: :text, content: text} -> text
      %ContentPart{type: _other, content: content} -> content
    end)
    |> Enum.join("")
  end

  def content_parts_to_string(content) when is_binary(content), do: content

  @doc """
  Convert content to a list of ContentParts. Content may be a string or already a list of ContentParts.
  """
  def content_to_parts(content) when is_binary(content) do
    [ContentPart.text!(content)]
  end

  def content_to_parts(content) when is_list(content) do
    content
  end

  def content_to_parts(nil), do: []

  @doc """
  Convert content to a single ContentPart for MessageDelta. Content may be a string or already a ContentPart.
  """
  def content_to_single_part(content) when is_binary(content) and content != "" do
    ContentPart.text!(content)
  end

  def content_to_single_part(""), do: nil

  def content_to_single_part(content) when is_list(content) and length(content) == 0 do
    nil
  end

  def content_to_single_part(content) when is_list(content) and length(content) == 1 do
    List.first(content)
  end

  def content_to_single_part(content) when is_list(content) do
    # Join multiple parts into a single text part for MessageDelta compatibility
    text_content =
      content
      |> Enum.map(fn
        %ContentPart{type: :text, content: text} -> text
        %ContentPart{content: other} -> other
      end)
      |> Enum.join("")

    ContentPart.text!(text_content)
  end

  def content_to_single_part(%ContentPart{} = part), do: part
  def content_to_single_part(nil), do: nil

  @doc """
  Convert a ContentPart to the expected map of data for the API.
  """
  def content_part_for_api(%ContentPart{type: :text} = part) do
    %{"type" => "text", "text" => part.content}
  end

  def content_part_for_api(%ContentPart{type: :file, options: opts} = part) do
    # orq requires file_data + filename (no file_id in schema)
    file_params =
      case Keyword.get(opts, :type, :base64) do
        :base64 ->
          media = Keyword.get(opts, :media, :pdf)

          prefix =
            case media do
              type when is_binary(type) ->
                "data:#{type};base64,"

              :pdf ->
                "data:application/pdf;base64,"

              :png ->
                "data:image/png;base64,"

              :jpeg ->
                "data:image/jpg;base64,"

              :jpg ->
                "data:image/jpg;base64,"

              :gif ->
                "data:image/gif;base64,"

              :webp ->
                "data:image/webp;base64,"

              _ ->
                "data:application/octet-stream;base64,"
            end

          %{
            "filename" => Keyword.get(opts, :filename, "file.bin"),
            "file_data" => prefix <> part.content
          }

        :file_id ->
          # Not part of orq schema; best-effort include as string content (may be rejected server-side)
          %{
            "filename" => Keyword.get(opts, :filename, "file.bin"),
            "file_data" => part.content
          }
      end

    %{
      "type" => "file",
      "file" => file_params
    }
  end

  def content_part_for_api(%ContentPart{type: image} = part)
      when image in [:image, :image_url] do
    media_prefix =
      case Keyword.get(part.options || [], :media, nil) do
        nil ->
          ""

        type when is_binary(type) ->
          "data:#{type};base64,"

        type when type in [:jpeg, :jpg] ->
          "data:image/jpg;base64,"

        :png ->
          "data:image/png;base64,"

        :gif ->
          "data:image/gif;base64,"

        :webp ->
          "data:image/webp;base64,"

        _other ->
          ""
      end

    detail_option = Keyword.get(part.options || [], :detail, nil)

    %{
      "type" => "image_url",
      "image_url" =>
        %{"url" => media_prefix <> part.content}
        |> Utils.conditionally_add_to_map("detail", detail_option)
    }
  end

  @doc false
  def get_parameters(%Function{parameters: [], parameters_schema: nil} = _fun) do
    %{
      "type" => "object",
      "properties" => %{}
    }
  end

  def get_parameters(%Function{parameters: [], parameters_schema: schema} = _fun)
      when is_map(schema) do
    schema
  end

  def get_parameters(%Function{parameters: params} = _fun) do
    FunctionParam.to_parameters_schema(params)
  end

  @doc false
  def prefix_messages_for_api(%_{} = orq, list) when is_list(list) do
    list
    |> Enum.reduce([], fn item, acc ->
      cond do
        match?(%Message{}, item) ->
          case for_api(orq, item) do
            %{} = data -> [data | acc]
            data when is_list(data) -> Enum.reverse(data) ++ acc
          end

        is_map(item) ->
          # Assume already in API shape
          [item | acc]

        true ->
          acc
      end
    end)
    |> Enum.reverse()
  end

  @doc """
  Calls the orq.ai API passing the ChatOrq struct with configuration, plus
  either a simple message or the list of messages to act as the prompt.

  Optionally pass in a list of tools available to the LLM for requesting
  execution in response (tools schema is not sent to orq; tool messages are included in messages).
  """
  @impl ChatModel
  def call(orq, prompt, tools \\ [])

  def call(%ChatOrq{} = orq, prompt, tools) when is_binary(prompt) do
    messages = [
      Message.new_system!(),
      Message.new_user!(prompt)
    ]

    call(orq, messages, tools)
  end

  def call(%ChatOrq{} = orq, messages, tools) when is_list(messages) do
    metadata = %{
      model: orq.model,
      message_count: length(messages),
      tools_count: length(tools)
    }

    LangChain.Telemetry.span([:langchain, :llm, :call], metadata, fn ->
      try do
        # Track the prompt being sent
        LangChain.Telemetry.llm_prompt(
          %{system_time: System.system_time()},
          %{model: orq.model, messages: messages}
        )

        case do_api_request(orq, messages, tools) do
          {:error, reason} ->
            {:error, reason}

          parsed_data ->
            # Track the response being received
            LangChain.Telemetry.llm_response(
              %{system_time: System.system_time()},
              %{model: orq.model, response: parsed_data}
            )

            {:ok, parsed_data}
        end
      rescue
        err in LangChainError ->
          {:error, err}
      end
    end)
  end

  # Make the API request to orq.ai
  @doc false
  @spec do_api_request(t(), [Message.t()], ChatModel.tools(), integer()) ::
          list() | struct() | {:error, LangChainError.t()}
  def do_api_request(orq, messages, tools, retry_count \\ 3)

  def do_api_request(_orq, _messages, _tools, 0) do
    raise LangChainError, "Retries exceeded. Connection failed."
  end

  def do_api_request(
        %ChatOrq{stream: false} = orq,
        messages,
        tools,
        retry_count
      ) do
    raw_data = for_api(orq, messages, tools)

    if orq.verbose_api do
      IO.inspect(raw_data, label: "RAW DATA BEING SUBMITTED")
    end

    req =
      Req.new(
        url: orq.endpoint,
        json: raw_data,
        auth: {:bearer, get_api_key(orq)},
        receive_timeout: orq.receive_timeout,
        retry: :transient,
        max_retries: 3,
        retry_delay: fn attempt -> 300 * attempt end
      )

    req
    |> Req.post()
    |> case do
      {:ok, %Req.Response{body: data} = response} ->
        if orq.verbose_api do
          IO.inspect(response, label: "RAW REQ RESPONSE")
        end

        Callbacks.fire(orq.callbacks, :on_llm_response_headers, [response.headers])

        case do_process_response(orq, data) do
          {:error, %LangChainError{} = reason} ->
            {:error, reason}

          result ->
            Callbacks.fire(orq.callbacks, :on_llm_new_message, [result])

            # Fire on_message_processed for assistant messages
            case result do
              %Message{role: :assistant} = message ->
                Callbacks.fire(orq.callbacks, :on_message_processed, [message])

              [%Message{role: :assistant} = message | _] ->
                Callbacks.fire(orq.callbacks, :on_message_processed, [message])

              messages when is_list(messages) ->
                Enum.each(messages, fn
                  %Message{role: :assistant} = message ->
                    Callbacks.fire(orq.callbacks, :on_message_processed, [message])

                  _ ->
                    :ok
                end)

              _ ->
                :ok
            end

            # Track non-streaming response completion
            LangChain.Telemetry.emit_event(
              [:langchain, :llm, :response, :non_streaming],
              %{system_time: System.system_time()},
              %{
                model: orq.model,
                response_size: byte_size(inspect(result))
              }
            )

            result
        end

      {:error, %Req.TransportError{reason: :timeout} = err} ->
        {:error,
         LangChainError.exception(type: "timeout", message: "Request timed out", original: err)}

      {:error, %Req.TransportError{reason: :closed}} ->
        # Force a retry by making a recursive call decrementing the counter
        Logger.debug(fn -> "Mint connection closed: retry count = #{inspect(retry_count)}" end)
        do_api_request(orq, messages, tools, retry_count - 1)

      other ->
        Logger.error("Unexpected and unhandled API response! #{inspect(other)}")
        other
    end
  end

  def do_api_request(
        %ChatOrq{stream: true} = orq,
        messages,
        tools,
        retry_count
      ) do
    raw_data = for_api(orq, messages, tools)

    if orq.verbose_api do
      IO.inspect(raw_data, label: "RAW DATA BEING SUBMITTED")
    end

    Req.new(
      url: orq.stream_endpoint,
      json: raw_data,
      auth: {:bearer, get_api_key(orq)},
      headers: [{"accept", "text/event-stream"}],
      receive_timeout: orq.receive_timeout
    )
    |> Req.post(
      into:
        Utils.handle_stream_fn(
          orq,
          &decode_stream/1,
          &do_process_response(orq, &1)
        )
    )
    |> case do
      {:ok, %Req.Response{body: data} = response} ->
        Callbacks.fire(orq.callbacks, :on_llm_response_headers, [response.headers])
        data

      {:error, %LangChainError{} = error} ->
        {:error, error}

      {:error, %Req.TransportError{reason: :timeout} = err} ->
        {:error,
         LangChainError.exception(type: "timeout", message: "Request timed out", original: err)}

      {:error, %Req.TransportError{reason: :closed}} ->
        # Force a retry by making a recursive call decrementing the counter
        Logger.debug(fn -> "Connection closed: retry count = #{inspect(retry_count)}" end)
        do_api_request(orq, messages, tools, retry_count - 1)

      other ->
        Logger.error(
          "Unhandled and unexpected response from streamed post call. #{inspect(other)}"
        )

        {:error,
         LangChainError.exception(type: "unexpected_response", message: "Unexpected response")}
    end
  end

  @doc """
  Decode a streamed response (SSE). Delegates to ChatOpenAI-compatible decoder.
  """
  @spec decode_stream({String.t(), String.t()}, list()) ::
          {%{String.t() => any()}}
  defdelegate decode_stream(data, done \\ []),
    to: LangChain.ChatModels.ChatOpenAI,
    as: :decode_stream

  # Parse responses (compatible with OpenAI-like shapes used by orq)
  @doc false
  @spec do_process_response(
          %{:callbacks => [map()]},
          data :: %{String.t() => any()} | {:error, any()}
        ) ::
          :skip
          | Message.t()
          | [Message.t()]
          | MessageDelta.t()
          | [MessageDelta.t()]
          | {:error, String.t()}
  def do_process_response(model, %{"choices" => _choices} = data) do
    token_usage = get_token_usage(data)

    case data do
      # no choices data but got token usage.
      %{"choices" => [], "usage" => _usage} ->
        token_usage

      # no data and no token usage. Skip.
      %{"choices" => []} ->
        :skip

      %{"choices" => choices} ->
        # process each response individually. Return a list of all processed choices.
        choices
        |> Enum.map(&do_process_response(model, &1))
        |> Enum.map(&TokenUsage.set(&1, token_usage))
    end
  end

  # Full message with tool call (non-streaming)
  def do_process_response(
        %{stream: false} = model,
        %{"finish_reason" => finish_reason, "message" => %{"tool_calls" => calls} = message} =
          data
      )
      when finish_reason in ["tool_calls", "stop"] do
    case Message.new(%{
           "role" => "assistant",
           "content" => content_to_parts(message["content"]),
           "complete" => true,
           "index" => data["index"],
           "tool_calls" => Enum.map(calls || [], &do_process_response(model, &1))
         }) do
      {:ok, message} ->
        message

      {:error, %Ecto.Changeset{} = changeset} ->
        {:error, LangChainError.exception(changeset)}
    end
  end

  # Full message with tool call (streaming) - return MessageDelta
  def do_process_response(
        %{stream: true} = model,
        %{"finish_reason" => finish_reason, "message" => %{"tool_calls" => calls} = message} =
          data
      )
      when finish_reason in ["tool_calls", "stop"] do
    status =
      case finish_reason do
        "stop" ->
          :complete

        "tool_calls" ->
          :complete

        other ->
          Logger.warning("Unsupported finish_reason in message. Reason: #{inspect(other)}")
          :complete
      end

    data_map =
      message
      |> Map.update("content", nil, &content_to_single_part/1)
      |> Map.put("index", data["index"])
      |> Map.put("status", status)
      |> Map.put("tool_calls", Enum.map(calls || [], &do_process_response(model, &1)))

    case LangChain.MessageDelta.new(data_map) do
      {:ok, message} ->
        message

      {:error, %Ecto.Changeset{} = changeset} ->
        {:error, LangChainError.exception(changeset)}
    end
  end

  # Full message without tool calls (non-streaming)
  def do_process_response(
        %{stream: false} = _model,
        %{"finish_reason" => finish_reason, "message" => message, "index" => index} = _data
      )
      when finish_reason in ["stop", "length", "max_tokens"] and
             not is_map_key(message, "tool_calls") do
    case Message.new(%{
           "role" => message["role"],
           "content" => content_to_parts(message["content"]),
           "complete" => true,
           "index" => index
         }) do
      {:ok, message} ->
        message

      {:error, %Ecto.Changeset{} = changeset} ->
        {:error, LangChainError.exception(changeset)}
    end
  end

  # Full message without tool calls (streaming) - return MessageDelta
  def do_process_response(
        %{stream: true} = _model,
        %{"finish_reason" => finish_reason, "message" => message, "index" => index} = _data
      )
      when finish_reason in ["stop", "length", "max_tokens"] and
             not is_map_key(message, "tool_calls") do
    status =
      case finish_reason do
        "stop" ->
          :complete

        "length" ->
          :length

        "max_tokens" ->
          :length

        other ->
          Logger.warning("Unsupported finish_reason in message. Reason: #{inspect(other)}")
          :complete
      end

    data =
      message
      |> Map.update("content", nil, &content_to_single_part/1)
      |> Map.put("index", index)
      |> Map.put("status", status)

    case LangChain.MessageDelta.new(data) do
      {:ok, message} ->
        message

      {:error, %Ecto.Changeset{} = changeset} ->
        {:error, LangChainError.exception(changeset)}
    end
  end

  # Orq.ai streaming message (has "message" instead of "delta") - only for incomplete streaming chunks
  def do_process_response(
        model,
        %{"finish_reason" => nil, "message" => message_body, "index" => index} = _msg
      ) do
    # For streaming chunks with finish_reason: nil, status is always incomplete
    status = :incomplete

    # Process tool calls if present
    tool_calls =
      case message_body do
        %{"tool_calls" => tools_data} when is_list(tools_data) ->
          Enum.map(tools_data, &do_process_response(model, &1))

        _other ->
          nil
      end

    data =
      message_body
      |> Map.update("content", nil, &content_to_single_part/1)
      |> Map.put("index", index)
      |> Map.put("status", status)
      |> Map.put("tool_calls", tool_calls)

    case LangChain.MessageDelta.new(data) do
      {:ok, message} ->
        message

      {:error, %Ecto.Changeset{} = changeset} ->
        {:error, LangChainError.exception(changeset)}
    end
  end

  # Orq.ai final streaming message (has "message" instead of "delta") - for complete streaming chunks
  def do_process_response(
        model,
        %{"finish_reason" => finish, "message" => message_body, "index" => index} = _msg
      )
      when finish in ["stop", "tool_calls", "length", "max_tokens"] do
    status =
      case finish do
        "stop" ->
          :complete

        "tool_calls" ->
          :complete

        "length" ->
          :length

        "max_tokens" ->
          :length

        other ->
          Logger.warning(
            "Unsupported finish_reason in streaming message. Reason: #{inspect(other)}"
          )

          :complete
      end

    # Process tool calls if present
    tool_calls =
      case message_body do
        %{"tool_calls" => tools_data} when is_list(tools_data) ->
          Enum.map(tools_data, &do_process_response(model, &1))

        _other ->
          nil
      end

    data =
      message_body
      |> Map.update("content", nil, &content_to_single_part/1)
      |> Map.put("index", index)
      |> Map.put("status", status)
      |> Map.put("tool_calls", tool_calls)

    case LangChain.MessageDelta.new(data) do
      {:ok, message} ->
        message

      {:error, %Ecto.Changeset{} = changeset} ->
        {:error, LangChainError.exception(changeset)}
    end
  end

  # OpenAI-style delta message (streaming)
  def do_process_response(
        model,
        %{"delta" => delta_body, "finish_reason" => finish, "index" => index} = _msg
      ) do
    status =
      case finish do
        nil ->
          :incomplete

        "stop" ->
          :complete

        "tool_calls" ->
          :complete

        "content_filter" ->
          :complete

        "length" ->
          :length

        "max_tokens" ->
          :length

        other ->
          Logger.warning("Unsupported finish_reason in message. Reason: #{inspect(other)}")
          nil
      end

    tool_calls =
      case delta_body do
        %{"tool_calls" => tools_data} when is_list(tools_data) ->
          Enum.map(tools_data, &do_process_response(model, &1))

        _other ->
          nil
      end

    role =
      case delta_body do
        %{"role" => role} -> role
        _other -> "unknown"
      end

    data =
      delta_body
      |> Map.update("content", nil, &content_to_single_part/1)
      |> Map.put("role", role)
      |> Map.put("index", index)
      |> Map.put("status", status)
      |> Map.put("tool_calls", tool_calls)

    case LangChain.MessageDelta.new(data) do
      {:ok, message} ->
        message

      {:error, %Ecto.Changeset{} = changeset} ->
        {:error, LangChainError.exception(changeset)}
    end
  end

  # Tool call as part of a delta message - handles both OpenAI and ORQ formats
  def do_process_response(_model, %{"function" => func_body, "index" => index} = tool_call) do
    case ToolCall.new(%{
           status: :incomplete,
           type: :function,
           call_id: tool_call["id"],
           name: Map.get(func_body, "name", nil),
           arguments: Map.get(func_body, "arguments", nil),
           index: index
         }) do
      {:ok, %ToolCall{} = call} ->
        call

      {:error, %Ecto.Changeset{} = changeset} ->
        reason = Utils.changeset_error_to_string(changeset)
        Logger.error("Failed to process ToolCall for a function. Reason: #{reason}")
        {:error, LangChainError.exception(changeset)}
    end
  end

  # Tool call with direct function data (ORQ streaming format)
  def do_process_response(_model, %{
        "function" => %{"name" => name, "arguments" => args},
        "id" => call_id,
        "index" => index,
        "type" => "function"
      }) do
    case ToolCall.new(%{
           status: :incomplete,
           type: :function,
           call_id: call_id,
           name: name,
           arguments: args,
           index: index
         }) do
      {:ok, %ToolCall{} = call} ->
        call

      {:error, %Ecto.Changeset{} = changeset} ->
        reason = Utils.changeset_error_to_string(changeset)
        Logger.error("Failed to process ToolCall for a function. Reason: #{reason}")
        {:error, LangChainError.exception(changeset)}
    end
  end

  # Tool call with partial data (streaming chunks)
  def do_process_response(
        _model,
        %{"id" => call_id, "index" => index, "type" => "function"} = tool_call
      ) do
    func_data = Map.get(tool_call, "function", %{})

    case ToolCall.new(%{
           status: :incomplete,
           type: :function,
           call_id: call_id,
           name: Map.get(func_data, "name", nil),
           arguments: Map.get(func_data, "arguments", nil),
           index: index
         }) do
      {:ok, %ToolCall{} = call} ->
        call

      {:error, %Ecto.Changeset{} = changeset} ->
        reason = Utils.changeset_error_to_string(changeset)
        Logger.error("Failed to process ToolCall for a function. Reason: #{reason}")
        {:error, LangChainError.exception(changeset)}
    end
  end

  # Tool call from a complete message
  def do_process_response(_model, %{
        "function" => %{
          "arguments" => args,
          "name" => name
        },
        "id" => call_id,
        "type" => "function"
      }) do
    case ToolCall.new(%{
           type: :function,
           status: :complete,
           name: name,
           arguments: args,
           call_id: call_id
         }) do
      {:ok, %ToolCall{} = call} ->
        call

      {:error, %Ecto.Changeset{} = changeset} ->
        reason = Utils.changeset_error_to_string(changeset)
        Logger.error("Failed to process ToolCall for a function. Reason: #{reason}")
        {:error, LangChainError.exception(changeset)}
    end
  end

  def do_process_response(_model, %{
        "finish_reason" => finish_reason,
        "message" => message,
        "index" => index
      }) do
    status =
      case finish_reason do
        nil ->
          :incomplete

        "stop" ->
          :complete

        "tool_calls" ->
          :complete

        "content_filter" ->
          :complete

        "length" ->
          :length

        "max_tokens" ->
          :length

        other ->
          Logger.warning("Unsupported finish_reason in message. Reason: #{inspect(other)}")
          nil
      end

    # Normalize content to ContentPart arrays for consistency
    normalized_message = Map.update(message, "content", [], &content_to_parts/1)

    case Message.new(Map.merge(normalized_message, %{"status" => status, "index" => index})) do
      {:ok, message} ->
        message

      {:error, %Ecto.Changeset{} = changeset} ->
        {:error, LangChainError.exception(changeset)}
    end
  end

  def do_process_response(_model, %{"error" => %{"message" => reason}} = response) do
    Logger.error("Received error from API: #{inspect(reason)}")
    {:error, LangChainError.exception(message: reason, original: response)}
  end

  def do_process_response(_model, {:error, %Jason.DecodeError{} = response}) do
    error_message = "Received invalid JSON: #{inspect(response)}"
    Logger.error(error_message)

    {:error,
     LangChainError.exception(type: "invalid_json", message: error_message, original: response)}
  end

  def do_process_response(_model, other) do
    Logger.error("Trying to process an unexpected response. #{inspect(other)}")

    {:error,
     LangChainError.exception(
       type: "unexpected_response",
       message: "Unexpected response",
       original: other
     )}
  end

  defp get_token_usage(%{"usage" => usage}) when is_map(usage) do
    TokenUsage.new!(%{
      input: Map.get(usage, "prompt_tokens"),
      output: Map.get(usage, "completion_tokens"),
      raw: usage
    })
  end

  defp get_token_usage(_response_body), do: nil

  @doc """
  Determine if an error should be retried. If `true`, a fallback LLM may be
  used. If `false`, the error is understood to be more fundamental with the
  request rather than a service issue and it should not be retried or fallback
  to another service.
  """
  @impl ChatModel
  @spec retry_on_fallback?(LangChainError.t()) :: boolean()
  def retry_on_fallback?(%LangChainError{type: "rate_limited"}), do: true
  def retry_on_fallback?(%LangChainError{type: "rate_limit_exceeded"}), do: true
  def retry_on_fallback?(%LangChainError{type: "timeout"}), do: true
  def retry_on_fallback?(%LangChainError{type: "too_many_requests"}), do: true
  def retry_on_fallback?(_), do: false

  @doc """
  Generate a config map that can later restore the model's configuration.
  """
  @impl ChatModel
  @spec serialize_config(t()) :: %{String.t() => any()}
  def serialize_config(%ChatOrq{} = model) do
    Utils.to_serializable_map(
      model,
      [
        :endpoint,
        :stream_endpoint,
        :key,
        :model,
        :receive_timeout,
        :stream,
        :inputs,
        :context,
        :prefix_messages,
        :messages_passthrough,
        :file_ids,
        :metadata,
        :extra_params,
        :documents,
        :invoke_options,
        :thread,
        :knowledge_filter
      ],
      @current_config_version
    )
  end

  @doc """
  Restores the model from the config.
  """
  @impl ChatModel
  def restore_from_map(%{"version" => 1} = data) do
    ChatOrq.new(data)
  end
end
