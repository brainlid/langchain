defmodule LangChain.ChatModels.ChatCompletionsFormat do
  @moduledoc false
  # The Chat Completions wire format: encoding LangChain messages, tools and
  # content parts into request data, and decoding responses and streamed
  # chunks back into LangChain structs.
  #
  # Shared by `ChatOpenAI`, `ChatOpenAICompatible` and the other chat models
  # whose services speak this format. It holds no defaults and no provider
  # policy. Any choice that differs between services is passed in by the
  # caller as an option.

  require Logger
  alias LangChain.PromptTemplate
  alias LangChain.Message
  alias LangChain.Message.ContentPart
  alias LangChain.Message.ToolCall
  alias LangChain.Message.ToolResult
  alias LangChain.TokenUsage
  alias LangChain.Function
  alias LangChain.FunctionParam
  alias LangChain.LangChainError
  alias LangChain.Utils
  alias LangChain.MessageDelta

  @typedoc """
  Encoding options.

  - `:system_role` - the role a `:system` message is sent as. Defaults to
    `:system`. OpenAI's reasoning models take `:developer`.
  """
  @type encode_opts :: [system_role: :system | :developer]

  @type encodable ::
          Message.t()
          | PromptTemplate.t()
          | ToolCall.t()
          | ToolResult.t()
          | ContentPart.t()
          | Function.t()

  @doc """
  Encode a list of messages for the request's `messages` field.

  A single `:tool` message holding several `ToolResult`s expands into one tool
  message per result, so the returned list can be longer than the input.
  """
  @spec messages_for_api([Message.t()], encode_opts()) :: [%{String.t() => any()}]
  def messages_for_api(messages, opts \\ []) do
    messages
    |> Enum.reduce([], fn m, acc ->
      case item_for_api(m, opts) do
        %{} = data ->
          [data | acc]

        data when is_list(data) ->
          Enum.reverse(data) ++ acc
      end
    end)
    |> Enum.reverse()
  end

  @doc """
  Encode a single LangChain structure.
  """
  @spec item_for_api(encodable(), encode_opts()) ::
          %{String.t() => any()} | [%{String.t() => any()}]
  def item_for_api(item, opts \\ [])

  def item_for_api(%Message{content: content} = msg, opts) when is_list(content) do
    %{
      "role" => message_role(msg.role, opts),
      "content" => content_parts_for_api(content)
    }
    |> Utils.conditionally_add_to_map("name", msg.name)
    |> Utils.conditionally_add_to_map(
      "tool_calls",
      Enum.map(msg.tool_calls || [], &item_for_api(&1, opts))
    )
  end

  def item_for_api(%Message{role: :assistant, tool_calls: tool_calls} = msg, opts)
      when is_list(tool_calls) do
    %{
      "role" => :assistant,
      "content" => msg.content
    }
    |> Utils.conditionally_add_to_map(
      "tool_calls",
      Enum.map(tool_calls, &item_for_api(&1, opts))
    )
  end

  def item_for_api(%ToolResult{type: :function} = result, _opts) do
    # a ToolResult becomes a stand-alone %Message{role: :tool} response.
    tool_result_for_api(result)
  end

  def item_for_api(%Message{role: :tool, tool_results: tool_results} = _msg, _opts)
      when is_list(tool_results) do
    # Each ToolResult becomes its own tool message.
    Enum.map(tool_results, &tool_result_for_api/1)
  end

  def item_for_api(%ToolCall{type: :function} = fun, _opts) do
    %{
      "id" => fun.call_id,
      "type" => "function",
      "function" => %{
        "name" => fun.name,
        "arguments" => Jason.encode!(fun.arguments)
      }
    }
  end

  def item_for_api(%Function{} = fun, _opts) do
    %{
      "name" => fun.name,
      "parameters" => get_parameters(fun),
      "strict" => fun.strict
    }
    |> Utils.conditionally_add_to_map("description", fun.description)
  end

  def item_for_api(%PromptTemplate{} = _template, _opts) do
    raise LangChainError, "PromptTemplates must be converted to messages."
  end

  def item_for_api(%ContentPart{} = part, _opts) do
    content_part_for_api(part)
  end

  defp tool_result_for_api(%ToolResult{} = result) do
    %{
      "role" => :tool,
      "tool_call_id" => result.tool_call_id,
      "content" => content_parts_for_api(result.content)
    }
  end

  defp message_role(:system, opts), do: Keyword.get(opts, :system_role, :system)
  defp message_role(role, _opts), do: role

  @doc """
  Encode functions for the request's `tools` field.
  """
  @spec tools_for_api(nil | [Function.t()]) :: [%{String.t() => any()}]
  def tools_for_api(nil), do: []

  def tools_for_api(tools) when is_list(tools) do
    Enum.map(tools, fn %Function{} = function ->
      %{"type" => "function", "function" => item_for_api(function)}
    end)
  end

  @doc """
  Encode a list of content parts.

  Thinking and unsupported parts are omitted. Both are response-side artifacts
  that this API surface has no request representation for, and both reach it by
  round-tripping a message the provider itself produced.

  There is no agreed request representation for thinking across the services
  that speak this API. A provider returning it in `reasoning_content` may
  accept that field back, ignore it, or accept it only in a particular mode,
  and the field is absent from the OpenAI request schema the rest of these
  services are modeled on. Unsupported parts, such as the `redacted_thinking`
  block Anthropic returns, hold opaque provider data with no meaning here at
  all.

  Nothing depends on returning either. Reasoning on this surface carries no
  signature to validate and no continuity requirement, so a conversation sends
  the answer text and any tool calls, and the model reasons afresh on the next
  turn.

  The omission is unconditional, which is what keeps prompt caching working.
  A prefix cache is built from what the client sends rather than from what the
  model produced, so a conversation that always omits thinking presents a
  prefix that matches itself turn after turn. Omitting it on some turns and
  including it on others would break the prefix at the first message that
  differs and cost a cache miss for everything after it.
  """
  @spec content_parts_for_api([ContentPart.t()]) :: [%{String.t() => any()}]
  def content_parts_for_api(content_parts) when is_list(content_parts) do
    content_parts
    |> Enum.reject(&(&1.type in [:thinking, :unsupported]))
    |> Enum.map(&content_part_for_api/1)
  end

  @doc """
  Encode a single content part.
  """
  @spec content_part_for_api(ContentPart.t()) :: %{String.t() => any()}
  def content_part_for_api(%ContentPart{type: :text} = part) do
    %{"type" => "text", "text" => part.content}
  end

  def content_part_for_api(%ContentPart{type: :file, options: opts} = part) do
    file_params =
      case Keyword.get(opts, :type, :base64) do
        :file_id ->
          %{
            "file_id" => part.content
          }

        :base64 ->
          %{
            "filename" => Keyword.get(opts, :filename, "file.pdf"),
            "file_data" => "data:application/pdf;base64," <> part.content
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

        other ->
          message = "Received unsupported media type for ContentPart: #{inspect(other)}"
          raise LangChainError, message
      end

    detail_option = Keyword.get(part.options, :detail, nil)

    %{
      "type" => "image_url",
      "image_url" =>
        %{"url" => media_prefix <> part.content}
        |> Utils.conditionally_add_to_map("detail", detail_option)
    }
  end

  @doc """
  Return the JSON Schema for a function's parameters.
  """
  @spec get_parameters(Function.t()) :: %{String.t() => any()}
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

  @doc """
  Decode a streamed response. Parses a string of received content into an
  Elixir map data structure using string keys.

  If a partial response was received, meaning the JSON text is split across
  multiple data frames, then the incomplete portion is returned as-is in the
  buffer. The function will be successively called, receiving the incomplete
  buffer data from a previous call, and assembling it to parse.
  """
  @spec decode_stream({String.t(), String.t()}, list()) ::
          {[%{String.t() => any()}], String.t()}
  def decode_stream({raw_data, buffer}, done \\ []) do
    # Data comes back like this:
    #
    # "data: {\"id\":\"chatcmpl-7e8yp1xBhriNXiqqZ0xJkgNrmMuGS\",\"object\":\"chat.completion.chunk\",\"created\":1689801995,\"model\":\"gpt-4-0613\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":null,\"function_call\":{\"name\":\"calculator\",\"arguments\":\"\"}},\"finish_reason\":null}]}\n\n
    #  data: {\"id\":\"chatcmpl-7e8yp1xBhriNXiqqZ0xJkgNrmMuGS\",\"object\":\"chat.completion.chunk\",\"created\":1689801995,\"model\":\"gpt-4-0613\",\"choices\":[{\"index\":0,\"delta\":{\"function_call\":{\"arguments\":\"{\\n\"}},\"finish_reason\":null}]}\n\n"
    #
    # In that form, the data is not ready to be interpreted as JSON. Let's clean
    # it up first.

    # as we start, the initial accumulator is an empty set of parsed results and
    # any left-over buffer from a previous processing.
    raw_data
    |> String.split("data: ")
    |> Enum.reduce({done, buffer}, fn str, {done, incomplete} = acc ->
      # auto filter out "" and "[DONE]" by not including the accumulator
      str
      |> String.trim()
      |> case do
        ":" <> _sse_comment ->
          # A line starting with a colon is an SSE comment and can be ignored per
          # https://html.spec.whatwg.org/multipage/server-sent-events.html#event-stream-interpretation
          # OpenRouter sends ": OPENROUTER PROCESSING" keep-alive comments which
          # otherwise poison the incomplete-JSON buffer and break the whole stream.
          acc

        "" ->
          acc

        "[DONE]" ->
          acc

        json ->
          parse_combined_data(incomplete, json, done)
      end
    end)
  end

  defp parse_combined_data("", json, done) do
    json
    |> Jason.decode()
    |> case do
      {:ok, parsed} ->
        {done ++ [parsed], ""}

      {:error, _reason} ->
        {done, json}
    end
  end

  defp parse_combined_data(incomplete, json, done) do
    # combine with any previous incomplete data
    starting_json = incomplete <> json

    # recursively call decode_stream so that the combined message data is split on "data: " again.
    # the combined data may need re-splitting if the last message ended in the middle of the "data: " key.
    # i.e. incomplete ends with "dat" and the new message starts with "a: {".
    decode_stream({starting_json, ""}, done)
  end

  @doc """
  Convert a decoded response body, a choice, a delta or a tool call into
  LangChain structs.
  """
  @spec process_response(data :: any()) ::
          :skip
          | TokenUsage.t()
          | Message.t()
          | [Message.t() | MessageDelta.t() | TokenUsage.t() | {:error, LangChainError.t()}]
          | MessageDelta.t()
          | [MessageDelta.t()]
          | ToolCall.t()
          | {:error, LangChainError.t()}
  def process_response(%{"choices" => _choices} = data) do
    token_usage = get_token_usage(data)

    case data do
      # no choices data but got token usage.
      %{"choices" => [], "usage" => _usage} ->
        token_usage

      # no data and no token usage. Skip.
      %{"choices" => []} ->
        :skip

      %{"choices" => choices} ->
        # process each response individually. Return a list of all processed
        # choices. If we received TokenUsage, attach it to each returned item.
        # Merging will work out later.
        choices
        |> Enum.map(&process_response/1)
        |> Enum.map(&TokenUsage.set(&1, token_usage))
    end
  end

  # Complete message carrying a reasoning model's thinking.
  #
  # OpenAI-compatible providers that expose reasoning models return the
  # thinking in a `reasoning_content` field beside `content`. The two become
  # separate content parts, thinking first, matching the order the model
  # produced them.
  #
  # Handled ahead of the tool call and plain message clauses so a reasoning
  # model that also calls a tool keeps both. Once the thinking is folded into
  # `content`, the message is dispatched again for the clause that matches its
  # shape.
  def process_response(%{"message" => %{"reasoning_content" => reasoning} = message} = data)
      when is_binary(reasoning) and reasoning != "" do
    content = reasoning_content_parts(reasoning, message["content"])

    data
    |> Map.put(
      "message",
      message |> Map.delete("reasoning_content") |> Map.put("content", content)
    )
    |> process_response()
  end

  # Full message with tool call
  def process_response(
        %{"finish_reason" => finish_reason, "message" => %{"tool_calls" => calls} = message} =
          data
      )
      when finish_reason in ["tool_calls", "stop"] do
    %{
      "role" => "assistant",
      "content" => message["content"],
      "complete" => true,
      "index" => data["index"],
      "tool_calls" => Enum.map(calls || [], &process_response/1)
    }
    |> Map.merge(logprobs_metadata(data))
    |> Message.new()
    |> case do
      {:ok, message} ->
        message

      {:error, %Ecto.Changeset{} = changeset} ->
        {:error, LangChainError.exception(changeset)}
    end
  end

  # Delta carrying a reasoning model's thinking.
  #
  # Providers streaming a reasoning model send `reasoning_content` alongside
  # `content` on every chunk, carrying the thinking while it is being produced
  # and `nil` once the answer begins. Thinking accumulates as a content part at
  # position 0 and the answer text at position 1, keeping them distinct in the
  # assembled message.
  #
  # `index` selects the position to merge into, so this repurposes the choice
  # index. A request asking for multiple choices from a reasoning model would
  # collapse them together.
  def process_response(%{"delta" => %{"reasoning_content" => _} = delta_body} = msg) do
    {index, content} =
      case delta_body["reasoning_content"] do
        reasoning when is_binary(reasoning) and reasoning != "" ->
          {0, ContentPart.thinking!(reasoning)}

        _no_reasoning ->
          {1, delta_body["content"]}
      end

    delta_body =
      delta_body
      |> Map.delete("reasoning_content")
      |> Map.put("content", content)

    msg
    |> Map.put("delta", delta_body)
    |> Map.put("index", index)
    |> process_response()
  end

  # Delta message tool call
  def process_response(%{"delta" => delta_body, "index" => index} = msg) do
    # finish_reason might not be present in all streaming responses (e.g., LiteLLM proxy)
    finish = Map.get(msg, "finish_reason", nil)
    status = finish_reason_to_status(finish)

    tool_calls =
      case delta_body do
        %{"tool_calls" => tools_data} when is_list(tools_data) ->
          Enum.map(tools_data, &process_response/1)

        _other ->
          nil
      end

    # more explicitly interpret the role. We treat a "function_call" as a a role
    # while OpenAI addresses it as an "assistant". Technically, they are correct
    # that the assistant is issuing the function_call.
    role =
      case delta_body do
        %{"role" => role} -> role
        _other -> "unknown"
      end

    data =
      delta_body
      |> Map.put("role", role)
      |> Map.put("index", index)
      |> Map.put("status", status)
      |> Map.put("tool_calls", tool_calls)
      |> Map.merge(logprobs_metadata(msg))

    case MessageDelta.new(data) do
      {:ok, message} ->
        message

      {:error, %Ecto.Changeset{} = changeset} ->
        {:error, LangChainError.exception(changeset)}
    end
  end

  # Tool call as part of a delta message
  def process_response(%{"function" => func_body, "index" => index} = tool_call) do
    # function parts may or may not be present on any given delta chunk
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
        {:error, LangChainError.exception(changeset)}
    end
  end

  # Tool call from a complete message
  def process_response(%{
        "function" => %{
          "arguments" => args,
          "name" => name
        },
        "id" => call_id,
        "type" => "function"
      }) do
    # No "index". It is a complete message.
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
        {:error, LangChainError.exception(changeset)}
    end
  end

  def process_response(
        %{
          "finish_reason" => finish_reason,
          "message" => message,
          "index" => index
        } = data
      ) do
    status = finish_reason_to_status(finish_reason)

    merged =
      message
      |> Map.merge(%{"status" => status, "index" => index})
      |> Map.merge(logprobs_metadata(data))

    case Message.new(merged) do
      {:ok, message} ->
        message

      {:error, %Ecto.Changeset{} = changeset} ->
        {:error, LangChainError.exception(changeset)}
    end
  end

  # MS Azure returns numeric error codes. Interpret them when possible to give a computer-friendly reason
  #
  # https://learn.microsoft.com/en-us/troubleshoot/azure/azure-kubernetes/create-upgrade-delete/429-too-many-requests-errors
  def process_response(
        %{"error" => %{"code" => code, "message" => reason} = error_data} = response
      ) do
    type =
      case code do
        "429" ->
          "rate_limit_exceeded"

        "unsupported_value" ->
          if String.contains?(reason, "does not support 'system' with this model") do
            # return the API error type as the exception type information
            error_data["type"]
          end

        _other ->
          nil
      end

    {:error, LangChainError.exception(type: type, message: reason, original: response)}
  end

  def process_response(%{"error" => %{"message" => reason}} = response) do
    {:error, LangChainError.exception(message: reason, original: response)}
  end

  def process_response({:error, %Jason.DecodeError{} = response}) do
    error_message = "Received invalid JSON: #{inspect(response)}"

    {:error,
     LangChainError.exception(type: "invalid_json", message: error_message, original: response)}
  end

  def process_response(other) do
    {:error,
     LangChainError.exception(
       type: "unexpected_response",
       message: "Unexpected response",
       original: other
     )}
  end

  # Build the content parts for a message that carried thinking, keeping the
  # thinking ahead of the answer text. A response can be thinking-only, such as
  # when the model stops on a token limit before answering.
  defp reasoning_content_parts(reasoning, content) when is_binary(content) and content != "" do
    [ContentPart.thinking!(reasoning), ContentPart.text!(content)]
  end

  defp reasoning_content_parts(reasoning, _content) do
    [ContentPart.thinking!(reasoning)]
  end

  # Extract logprobs from a choice-level response map and return a metadata map.
  # Returns an empty map when logprobs is nil or absent, so it can be safely merged.
  defp logprobs_metadata(%{"logprobs" => logprobs}) when not is_nil(logprobs),
    do: %{"metadata" => %{"logprobs" => logprobs}}

  defp logprobs_metadata(_data), do: %{}

  defp finish_reason_to_status(nil), do: :incomplete
  defp finish_reason_to_status("stop"), do: :complete
  defp finish_reason_to_status("tool_calls"), do: :complete
  defp finish_reason_to_status("content_filter"), do: :content_filtered
  defp finish_reason_to_status("length"), do: :length
  defp finish_reason_to_status("max_tokens"), do: :length

  defp finish_reason_to_status(other) do
    Logger.warning("Unsupported finish_reason in message. Reason: #{inspect(other)}")
    nil
  end

  defp get_token_usage(%{"usage" => usage} = response_body) when is_map(usage) do
    # extract out the reported response token usage
    #
    #  https://platform.openai.com/docs/api-reference/chat/object#chat/object-usage
    #
    # The tier that served the request is reported beside `usage`, not inside
    # it. Keeping it in `raw` carries it to the final message along with the
    # usage, which is also where ChatAnthropic reports its tier.
    TokenUsage.new!(%{
      input: Map.get(usage, "prompt_tokens"),
      output: Map.get(usage, "completion_tokens"),
      raw: Utils.conditionally_add_to_map(usage, "service_tier", response_body["service_tier"])
    })
  end

  defp get_token_usage(_response_body), do: nil

  @doc """
  Extract the rate-limit headers a Chat Completions service reports.

  https://platform.openai.com/docs/guides/rate-limits/rate-limits-in-headers
  """
  @spec get_ratelimit_info(%{String.t() => any()}) :: %{String.t() => any()}
  def get_ratelimit_info(response_headers) do
    {return, _} =
      Map.split(response_headers, [
        "x-ratelimit-limit-requests",
        "x-ratelimit-limit-tokens",
        "x-ratelimit-remaining-requests",
        "x-ratelimit-remaining-tokens",
        "x-ratelimit-reset-requests",
        "x-ratelimit-reset-tokens",
        "x-request-id"
      ])

    return
  end
end
