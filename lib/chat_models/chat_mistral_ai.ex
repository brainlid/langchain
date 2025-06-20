defmodule LangChain.ChatModels.ChatMistralAI do
  use Ecto.Schema
  require Logger
  import Ecto.Changeset
  alias __MODULE__
  alias LangChain.Config
  alias LangChain.ChatModels.ChatModel
  alias LangChain.ChatModels.ChatOpenAI
  alias LangChain.Function
  alias LangChain.Message
  alias LangChain.Message.ContentPart
  alias LangChain.Message.ToolCall
  alias LangChain.Message.ToolResult
  alias LangChain.TokenUsage
  alias LangChain.MessageDelta
  alias LangChain.LangChainError
  alias LangChain.Utils
  alias LangChain.Callbacks

  @behaviour ChatModel

  @current_config_version 1
  @receive_timeout 60_000

  @default_endpoint "https://api.mistral.ai/v1/chat/completions"

  @primary_key false
  embedded_schema do
    field :endpoint, :string, default: @default_endpoint

    # The version/model of the Mistral API to use.
    field :model, :string
    field :api_key, :string, redact: true

    # Sampling temperature, 0..1 for Mistral
    field :temperature, :float, default: 0.9

    # top_p param to shape token selection
    field :top_p, :float, default: 1.0

    # Duration in milliseconds for the response to be received.
    field :receive_timeout, :integer, default: @receive_timeout

    # Maximum tokens to generate in the response
    field :max_tokens, :integer

    # Some Mistral deployments allow a "safe_prompt" option
    field :safe_prompt, :boolean, default: false

    # Optional random seed for reproducible outputs.
    field :random_seed, :integer

    # Whether to stream partial/delta responses
    field :stream, :boolean, default: false

    # For choosing a specific tool call (like forcing a function execution).
    field :tool_choice, :map

    # JSON Schema to validate the output format (for structured JSON output)
    field :json_schema, :map

    # Whether to force a JSON response format
    field :json_response, :boolean, default: false

    # A list of callback handlers
    field :callbacks, {:array, :map}, default: []
  end

  @type t :: %ChatMistralAI{}

  @create_fields [
    :endpoint,
    :model,
    :api_key,
    :temperature,
    :top_p,
    :receive_timeout,
    :max_tokens,
    :safe_prompt,
    :random_seed,
    :stream,
    :tool_choice,
    :json_schema,
    :json_response
  ]
  @required_fields [
    :model
  ]

  @spec get_api_key(t) :: String.t()
  defp get_api_key(%ChatMistralAI{api_key: api_key}) do
    # If no API key is set, fall back to the globally configured Mistral API key
    api_key || Config.resolve(:mistral_api_key)
  end

  @spec new(attrs :: map()) :: {:ok, t} | {:error, Ecto.Changeset.t()}
  def new(%{} = attrs \\ %{}) do
    %ChatMistralAI{}
    |> cast(attrs, @create_fields)
    |> common_validation()
    |> apply_action(:insert)
  end

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
  end

  @doc """
  Formats this struct plus the given messages and tools as a request payload.
  """
  @spec for_api(t(), [Message.t()], ChatModel.tools()) :: %{atom() => any()}
  def for_api(%ChatMistralAI{} = mistral, messages, tools) do
    %{
      model: mistral.model,
      temperature: mistral.temperature,
      top_p: mistral.top_p,
      safe_prompt: mistral.safe_prompt,
      stream: mistral.stream,
      messages: Enum.map(messages, &for_api(mistral, &1))
    }
    |> Utils.conditionally_add_to_map(:random_seed, mistral.random_seed)
    |> Utils.conditionally_add_to_map(:max_tokens, mistral.max_tokens)
    |> Utils.conditionally_add_to_map(:tools, get_tools_for_api(mistral, tools))
    |> Utils.conditionally_add_to_map(:tool_choice, get_tool_choice(mistral))
    |> Utils.conditionally_add_to_map(:response_format, set_response_format(mistral))
  end

  # Creates the response_format field for JSON output when json_response is true.
  # If json_schema is provided, it will be included in the response format.
  #
  # For Mistral, the format is as follows:
  # https://docs.mistral.ai/capabilities/structured-output/custom_structured_output/
  # {
  #   "type": "json_schema",
  #   "json_schema": {
  #     "schema": { ... },
  #     "name": "output",
  #     "strict": true
  #   }
  # }
  @spec set_response_format(t()) :: map() | nil
  defp set_response_format(%ChatMistralAI{json_response: true, json_schema: schema})
       when is_map(schema) and map_size(schema) > 0 do
    # The schema should already be in the correct format
    schema
  end

  defp set_response_format(%ChatMistralAI{json_response: true}) do
    # For Mistral, when no schema is provided, we use json_object type
    %{
      "type" => "json_object"
    }
  end

  defp set_response_format(%ChatMistralAI{}) do
    nil
  end

  # Add a more complete function to map tools. This mirrors ChatOpenAI approach.
  defp get_tools_for_api(%__MODULE__{} = _model, nil), do: []

  defp get_tools_for_api(%__MODULE__{} = model, tools) when is_list(tools) do
    Enum.map(tools, fn
      %Function{} = function ->
        %{"type" => "function", "function" => for_api(model, function)}
    end)
  end

  defp get_tool_choice(%ChatMistralAI{
         tool_choice: %{"type" => "function", "function" => %{"name" => name}} = _tool_choice
       })
       when is_binary(name) and byte_size(name) > 0,
       do: %{"type" => "function", "function" => %{"name" => name}}

  defp get_tool_choice(%ChatMistralAI{tool_choice: %{"type" => type} = _tool_choice})
       when is_binary(type) and byte_size(type) > 0,
       do: type

  defp get_tool_choice(%ChatMistralAI{}), do: nil

  @doc """
  Converts a LangChain Message-based structure into the expected map of data for
  Mistral. We also include any `tool_calls` stored on the message.
  """
  @spec for_api(
          struct(),
          Message.t()
          | ContentPart.t()
          | ToolCall.t()
          | ToolResult.t()
          | Function.t()
        ) ::
          %{String.t() => any()} | [%{String.t() => any()}]
  def for_api(%_{} = model, %Message{content: content} = msg) when is_binary(content) do
    role = get_message_role(model, msg.role)

    %{
      "role" => role,
      "content" => msg.content
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
        content when is_binary(content) -> content
        content when is_list(content) -> ContentPart.parts_to_string(content)
        nil -> nil
      end

    %{
      "role" => :assistant,
      "content" => content
    }
    |> Utils.conditionally_add_to_map("tool_calls", Enum.map(tool_calls, &for_api(model, &1)))
  end

  def for_api(%_{} = _model, %Message{role: :user, content: content} = msg)
      when is_list(content) do
    # A user message can hold an array of ContentParts
    %{
      "role" => msg.role,
      "content" => ContentPart.parts_to_string(content)
    }
    |> Utils.conditionally_add_to_map("name", msg.name)
  end

  # Handle messages with ContentPart content for non-user roles
  def for_api(%_{} = model, %Message{content: content} = msg) when is_list(content) do
    role = get_message_role(model, msg.role)

    %{
      "role" => role,
      "content" => ContentPart.parts_to_string(content)
    }
    |> Utils.conditionally_add_to_map("name", msg.name)
    |> Utils.conditionally_add_to_map(
      "tool_calls",
      Enum.map(msg.tool_calls || [], &for_api(model, &1))
    )
  end

  # Handle ContentPart structures
  def for_api(%_{} = _model, %ContentPart{type: :text, content: content}) do
    content
  end

  # ToolResult => stand-alone message with "role: :tool"
  def for_api(%_{} = _model, %ToolResult{type: :function} = result) do
    %{
      "role" => :tool,
      "tool_call_id" => result.tool_call_id,
      "content" => result.content
    }
  end

  # When an assistant message has go-betweens for tool results, for example
  def for_api(%_{} = _model, %Message{role: :tool, tool_results: [result | _]} = _msg) do
    %{
      "role" => "tool",
      "content" => result.content,
      "tool_call_id" => result.tool_call_id
    }
  end

  # Handle empty tool_results
  def for_api(%_{} = _model, %Message{role: :tool, tool_results: []} = _msg) do
    %{
      "role" => "tool",
      "content" => ""
    }
  end

  # ToolCall => "function" style request
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

  def for_api(_model, %Function{} = fun) do
    %{
      "name" => fun.name,
      "description" => fun.description,
      "parameters" => fun.parameters_schema || %{}
    }
  end

  # Implementation only: more straightforward approach for Mistral
  defp get_message_role(%ChatMistralAI{}, role), do: role

  @doc """
  Calls the Mistral API passing the ChatMistralAI struct plus either a simple string
  prompt or a list of messages as the prompt. Optionally pass in a list of tools.
  """
  @impl ChatModel
  def call(%__MODULE__{} = mistralai, prompt, tools) when is_binary(prompt) and is_list(tools) do
    messages = [
      Message.new_system!(),
      Message.new_user!(prompt)
    ]

    call(mistralai, messages, tools)
  end

  def call(%__MODULE__{} = mistralai, messages, tools)
      when is_list(messages) and is_list(tools) do
    metadata = %{
      model: mistralai.model,
      message_count: length(messages),
      tools_count: length(tools)
    }

    LangChain.Telemetry.span([:langchain, :llm, :call], metadata, fn ->
      try do
        # Track the prompt being sent
        LangChain.Telemetry.llm_prompt(
          %{system_time: System.system_time()},
          %{model: mistralai.model, messages: messages}
        )

        case do_api_request(mistralai, messages, tools) do
          {:error, reason} ->
            {:error, reason}

          parsed_data ->
            # Track the response being received
            LangChain.Telemetry.llm_response(
              %{system_time: System.system_time()},
              %{model: mistralai.model, response: parsed_data}
            )

            {:ok, parsed_data}
        end
      rescue
        err in LangChainError ->
          {:error, err}
      end
    end)
  end

  # Make the API request. If `stream: true`, we handle partial chunk deltas;
  # otherwise, we parse a single complete body.
  @doc false
  @spec do_api_request(t(), [Message.t()], ChatModel.tools(), integer()) ::
          list() | struct() | {:error, LangChainError.t()}
  def do_api_request(openai, messages, tools, retry_count \\ 3)

  def do_api_request(_mistralai, _messages, _tools, 0) do
    raise LangChainError, "Retries exceeded. Connection failed."
  end

  def do_api_request(
        %__MODULE__{stream: false} = mistralai,
        messages,
        tools,
        retry_count
      ) do
    raw_data = for_api(mistralai, messages, tools)

    req =
      Req.new(
        url: mistralai.endpoint,
        json: raw_data,
        auth: {:bearer, get_api_key(mistralai)},
        headers: [
          {"api-key", get_api_key(mistralai)}
        ],
        receive_timeout: mistralai.receive_timeout,
        retry: :transient,
        max_retries: 3,
        retry_delay: fn attempt -> 300 * attempt end
      )

    req
    |> Req.post()
    |> case do
      {:ok, %Req.Response{body: data} = _response} ->
        Callbacks.fire(mistralai.callbacks, :on_llm_token_usage, [
          get_token_usage(data)
        ])

        case do_process_response(mistralai, data) do
          {:error, %LangChainError{} = reason} ->
            {:error, reason}

          result ->
            # Track non-streaming response completion
            LangChain.Telemetry.emit_event(
              [:langchain, :llm, :response, :non_streaming],
              %{system_time: System.system_time()},
              %{
                model: mistralai.model,
                response_size: byte_size(inspect(result))
              }
            )

            Callbacks.fire(mistralai.callbacks, :on_llm_new_message, [result])
            result
        end

      {:error, %Req.TransportError{reason: :timeout} = err} ->
        {:error,
         LangChainError.exception(type: "timeout", message: "Request timed out", original: err)}

      {:error, %Req.TransportError{reason: :closed}} ->
        Logger.debug(fn -> "Connection closed: retry count = #{inspect(retry_count)}" end)
        do_api_request(mistralai, messages, tools, retry_count - 1)

      other ->
        Logger.error("Unexpected and unhandled API response! #{inspect(other)}")
        other
    end
  end

  def do_api_request(
        %__MODULE__{stream: true} = mistralai,
        messages,
        tools,
        retry_count
      ) do
    raw_data = for_api(mistralai, messages, tools)

    req =
      Req.new(
        url: mistralai.endpoint,
        json: raw_data,
        auth: {:bearer, get_api_key(mistralai)},
        headers: [
          {"api-key", get_api_key(mistralai)}
        ],
        receive_timeout: mistralai.receive_timeout
      )

    req
    |> Req.post(
      into:
        Utils.handle_stream_fn(
          mistralai,
          # Mistral's streaming API is mostly compatible with OpenAI's,
          # so we can reuse the same decoder
          &ChatOpenAI.decode_stream/1,
          &do_process_response(mistralai, &1)
        )
    )
    |> case do
      {:ok, %Req.Response{body: data} = _response} ->
        data

      {:error, %Req.TransportError{reason: :timeout} = err} ->
        {:error,
         LangChainError.exception(type: "timeout", message: "Request timed out", original: err)}

      {:error, %Req.TransportError{reason: :closed}} ->
        Logger.debug(fn -> "Connection closed: retry count = #{inspect(retry_count)}" end)
        do_api_request(mistralai, messages, tools, retry_count - 1)

      other ->
        Logger.error("Unhandled and unexpected response from streamed call. #{inspect(other)}")
        {:error, LangChainError.exception(type: "unexpected_response", message: "Unexpected")}
    end
  end

  # Parse final or partial responses to produce the appropriate LangChain structure.
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
  # The last chunk of the response contains both the final delta in the "choices" key,
  # and the token usage in the "usage" key
  def do_process_response(model, %{"choices" => choices, "usage" => %{} = _usage} = data) do
    case get_token_usage(data) do
      %TokenUsage{} = token_usage ->
        Callbacks.fire(model.callbacks, :on_llm_token_usage, [token_usage])
        :ok

      nil ->
        :ok
    end

    Enum.map(choices, &do_process_response(model, &1))
  end

  def do_process_response(_model, %{"choices" => []}), do: :skip

  def do_process_response(model, %{"choices" => choices}) when is_list(choices) do
    Enum.map(choices, &do_process_response(model, &1))
  end

  def do_process_response(_model, %{"choices" => _not_a_list} = data) do
    Logger.warning("""
    Mistral returned a response with a "choices" key that is not a list.
    data: #{inspect(data)}
    """)

    :skip
  end

  # Partial 'delta' format: look for any embedded "tool_calls"
  def do_process_response(
        model,
        %{
          "delta" => delta_body,
          "finish_reason" => finish,
          "index" => index
        }
      ) do
    status =
      case finish do
        nil ->
          :incomplete

        "stop" ->
          :complete

        "length" ->
          :length

        "model_length" ->
          :length

        "tool_calls" ->
          :complete

        other ->
          Logger.warning("Unsupported finish_reason in delta message. Reason: #{inspect(other)}")
          nil
      end

    # If partial chunk references some tool calls
    tool_calls =
      case delta_body do
        %{"tool_calls" => calls} when is_list(calls) ->
          Enum.map(calls, &do_process_response(model, &1))

        _ ->
          nil
      end

    role =
      case delta_body do
        %{"role" => role} -> role
        # Mistral doesn't include a `role` key in the delta.
        # Defaulting to `:assistant`. seems like it makes sense.
        _ -> "assistant"
      end

    data =
      delta_body
      |> Map.put("role", role)
      |> Map.put("index", index)
      |> Map.put("status", status)
      |> Map.put("tool_calls", tool_calls)

    case MessageDelta.new(data) do
      {:ok, message} ->
        message

      {:error, %Ecto.Changeset{} = changeset} ->
        {:error, LangChainError.exception(changeset)}
    end
  end

  # Complete message with tool calls:
  def do_process_response(
        model,
        %{"finish_reason" => finish_reason, "message" => %{"tool_calls" => calls} = message} =
          data
      )
      when finish_reason in ["tool_calls", "stop"] do
    tool_calls =
      if is_list(calls),
        do: Enum.map(calls, &do_process_response(model, &1)),
        else: []

    case Message.new(%{
           "role" => "assistant",
           "content" => message["content"],
           "complete" => true,
           "index" => data["index"],
           "tool_calls" => tool_calls
         }) do
      {:ok, msg} ->
        msg

      {:error, %Ecto.Changeset{} = changeset} ->
        {:error, LangChainError.exception(changeset)}
    end
  end

  # Tool call from a complete message
  def do_process_response(
        _model,
        %{
          "function" => %{"arguments" => args, "name" => name},
          "id" => call_id,
          "index" => _maybe_index
        } = _data
      ) do
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
        Logger.error("Failed to process ToolCall. Reason: #{reason}")
        {:error, LangChainError.exception(changeset)}
    end
  end

  def do_process_response(_model, %{
        "function" => %{"arguments" => args, "name" => name},
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
        Logger.error("Failed to process ToolCall. Reason: #{reason}")
        {:error, LangChainError.exception(changeset)}
    end
  end

  def do_process_response(_model, %{"error" => %{"message" => reason}} = response) do
    Logger.error("Received error from Mistral API: #{inspect(reason)}")
    {:error, LangChainError.exception(message: reason, original: response)}
  end

  def do_process_response(_model, {:error, %Jason.DecodeError{} = response}) do
    error_message = "Received invalid JSON: #{inspect(response)}"
    Logger.error(error_message)

    {:error,
     LangChainError.exception(type: "invalid_json", message: error_message, original: response)}
  end

  def do_process_response(_model, other) do
    Logger.error("Trying to process an unexpected response from Mistral: #{inspect(other)}")

    {:error,
     LangChainError.exception(
       type: "unexpected_response",
       message: "Unexpected response",
       original: other
     )}
  end

  defp get_token_usage(%{"usage" => usage} = _response_body) do
    # extract out the reported response token usage
    #
    #  https://platform.mistralai.com/docs/api-reference/chat/object#chat/object-usage
    TokenUsage.new!(%{
      input: Map.get(usage, "prompt_tokens"),
      output: Map.get(usage, "completion_tokens"),
      raw: usage
    })
  end

  defp get_token_usage(_response_body), do: nil

  @doc """
  Generate a config map that can later restore the model's configuration.
  """
  @impl ChatModel
  @spec serialize_config(t()) :: %{String.t() => any()}
  def serialize_config(%ChatMistralAI{} = model) do
    Utils.to_serializable_map(
      model,
      [
        :endpoint,
        :model,
        :temperature,
        :top_p,
        :receive_timeout,
        :max_tokens,
        :safe_prompt,
        :random_seed,
        :stream,
        :json_schema,
        :json_response
      ],
      @current_config_version
    )
  end

  @doc """
  Restores the model from the config map.
  """
  @impl ChatModel
  def restore_from_map(%{"version" => 1} = data) do
    ChatMistralAI.new(data)
  end
end
