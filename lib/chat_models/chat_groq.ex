defmodule LangChain.ChatModels.ChatGroq do
  @moduledoc """
  Represents the [Groq Chat API](https://console.groq.com/docs/openai).
  
  Parses and validates inputs for making requests to the Groq Chat API.
  Converts responses into more specialized `LangChain` data structures.
  
  ## Features
  
  - Native API support for Groq's LLM capabilities
  - Support for streaming responses
  - Tool/function calling capabilities
  - Proper error handling for Groq-specific responses
  
  ## Configuration
  
  The Groq API key can be configured through the environment:
  
  ```elixir
  # In config/config.exs
  config :langchain, :groq,
    api_key: System.get_env("GROQ_API_KEY")
  ```
  
  For testing, you can set the API key in your environment:
  
  ```bash
  GROQ_API_KEY=your-api-key mix test --only groq
  ```
  
  ## Supported Models
  
  ### Production Models
  
  - `gemma2-9b-it` (Google)
  - `llama-3.3-70b-versatile` (Meta)
  - `llama-3.1-8b-instant` (Meta)
  - `llama-guard-3-8b` (Meta)
  - `llama3-70b-8192` (Meta)
  - `llama3-8b-8192` (Meta)
  
  ### Preview Models (evaluation only)
  
  - `meta-llama/llama-4-maverick-17b-128e-instruct` (Meta)
  - `meta-llama/llama-4-scout-17b-16e-instruct` (Meta)
  - `mistral-saba-24b` (Mistral)
  - `deepseek-r1-distill-llama-70b` (DeepSeek)
  - `qwen-qwq-32b` (Alibaba Cloud)
  
  ## Example Usage
  
  ```elixir
  # Basic usage
  {:ok, chat} = ChatGroq.new(%{
    model: "llama3-70b-8192",
    temperature: 0.7,
    max_tokens: 1000
  })
  
  # With streaming enabled
  {:ok, chat} = ChatGroq.new(%{
    model: "llama3-70b-8192",
    temperature: 0.7,
    stream: true
  })
  ```
  """
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

  @default_endpoint "https://api.groq.com/openai/v1/chat/completions"

  @primary_key false
  embedded_schema do
    field :endpoint, :string, default: @default_endpoint

    # The model to use for the Groq API
    field :model, :string
    field :api_key, :string, redact: true

    # Sampling temperature, 0..2 (same as OpenAI)
    field :temperature, :float, default: 0.7

    # Duration in milliseconds for the response to be received
    field :receive_timeout, :integer, default: @receive_timeout

    # Maximum tokens to generate in the response
    field :max_tokens, :integer

    # Optional random seed for reproducible outputs
    field :seed, :integer

    # top_p param to shape token selection
    field :top_p, :float, default: 1.0

    # Whether to stream partial/delta responses
    field :stream, :boolean, default: false

    # For choosing a specific tool call (like forcing a function execution)
    field :tool_choice, :map

    # A list of callback handlers
    field :callbacks, {:array, :map}, default: []

    # For JSON response
    field :json_response, :boolean, default: false
    field :json_schema, :map, default: nil

    # Can send a string user_id to help services detect abuse
    field :user, :string

    # For help with debugging. It outputs the RAW Req response received and the
    # RAW Elixir map being submitted to the API.
    field :verbose_api, :boolean, default: false
  end

  @type t :: %ChatGroq{}

  @create_fields [
    :endpoint,
    :model,
    :api_key,
    :temperature,
    :top_p,
    :receive_timeout,
    :max_tokens,
    :seed,
    :stream,
    :tool_choice,
    :json_response,
    :json_schema,
    :user,
    :verbose_api
  ]
  @required_fields [
    :model
  ]

  @spec get_api_key(t()) :: String.t()
  defp get_api_key(%ChatGroq{api_key: api_key}) do
    # If no API key is set, fall back to the globally configured Groq API key
    api_key || Config.resolve(:groq_api_key, "")
  end

  @doc """
  Setup a ChatGroq client configuration.
  """
  @spec new(attrs :: map()) :: {:ok, t} | {:error, Ecto.Changeset.t()}
  def new(%{} = attrs \\ %{}) do
    %ChatGroq{}
    |> cast(attrs, @create_fields)
    |> common_validation()
    |> apply_action(:insert)
  end

  @doc """
  Setup a ChatGroq client configuration and return it or raise an error if invalid.
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
    |> validate_number(:temperature, greater_than_or_equal_to: 0, less_than_or_equal_to: 2)
    |> validate_number(:top_p, greater_than_or_equal_to: 0, less_than_or_equal_to: 1)
    |> validate_number(:receive_timeout, greater_than_or_equal_to: 0)
  end

  @doc """
  Formats this struct plus the given messages and tools as a request payload.
  """
  @spec for_api(t(), [Message.t()], ChatModel.tools()) :: %{atom() => any()}
  def for_api(%ChatGroq{} = groq, messages, tools) do
    %{
      model: groq.model,
      temperature: groq.temperature,
      top_p: groq.top_p,
      stream: groq.stream,
      messages: Enum.map(messages, &for_api(groq, &1)),
      user: groq.user
    }
    |> Utils.conditionally_add_to_map(:response_format, set_response_format(groq))
    |> Utils.conditionally_add_to_map(:max_tokens, groq.max_tokens)
    |> Utils.conditionally_add_to_map(:seed, groq.seed)
    |> Utils.conditionally_add_to_map(:tools, get_tools_for_api(groq, tools))
    |> Utils.conditionally_add_to_map(:tool_choice, get_tool_choice(groq))
  end

  defp get_tools_for_api(%_{} = _model, nil), do: []

  defp get_tools_for_api(%_{} = model, tools) when is_list(tools) do
    Enum.map(tools, fn
      %Function{} = function ->
        %{"type" => "function", "function" => for_api(model, function)}
    end)
  end

  defp get_tool_choice(%ChatGroq{
         tool_choice: %{"type" => "function", "function" => %{"name" => name}} = _tool_choice
       })
       when is_binary(name) and byte_size(name) > 0,
       do: %{"type" => "function", "function" => %{"name" => name}}

  defp get_tool_choice(%ChatGroq{tool_choice: %{"type" => type} = _tool_choice})
       when is_binary(type) and byte_size(type) > 0,
       do: type

  defp get_tool_choice(%ChatGroq{}), do: nil

  defp set_response_format(%ChatGroq{json_response: true, json_schema: json_schema})
       when not is_nil(json_schema) do
    %{
      "type" => "json_schema",
      "json_schema" => json_schema
    }
  end

  defp set_response_format(%ChatGroq{json_response: true}) do
    %{"type" => "json_object"}
  end

  defp set_response_format(%ChatGroq{json_response: false}) do
    nil
  end

  @doc """
  Converts a LangChain Message-based structure into the expected map of data for
  Groq. We also include any `tool_calls` stored on the message.
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
    %{
      "role" => msg.role,
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
    %{
      "role" => :assistant,
      "content" => msg.content
    }
    |> Utils.conditionally_add_to_map("tool_calls", Enum.map(tool_calls, &for_api(model, &1)))
  end

  def for_api(%_{} = model, %Message{role: :user, content: content} = msg)
      when is_list(content) do
    # A user message can hold an array of ContentParts
    %{
      "role" => msg.role,
      "content" => Enum.map(content, &for_api(model, &1))
    }
    |> Utils.conditionally_add_to_map("name", msg.name)
  end

  # ToolResult => stand-alone message with "role: :tool"
  def for_api(%_{} = _model, %ToolResult{type: :function} = result) do
    %{
      "role" => :tool,
      "tool_call_id" => result.tool_call_id,
      "content" => result.content
    }
  end

  # Tool message handling
  def for_api(%_{} = _model, %Message{role: :tool, tool_results: tool_results} = _msg)
      when is_list(tool_results) do
    # ToolResults turn into a list of tool messages for Groq
    Enum.map(tool_results, fn result ->
      %{
        "role" => :tool,
        "tool_call_id" => result.tool_call_id,
        "content" => result.content
      }
    end)
  end

  # ContentPart handling
  def for_api(%_{} = _model, %ContentPart{type: :text} = part) do
    %{"type" => "text", "text" => part.content}
  end

  def for_api(%_{} = _model, %ContentPart{type: :image, options: opts} = part) do
    detail_option = Keyword.get(opts || [], :detail, nil)

    %{
      "type" => "image_url",
      "image_url" =>
        %{"url" => part.content}
        |> Utils.conditionally_add_to_map("detail", detail_option)
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

  @doc """
  Calls the Groq API passing the ChatGroq struct plus either a simple string
  prompt or a list of messages as the prompt. Optionally pass in a list of tools.
  """
  @impl ChatModel
  def call(%__MODULE__{} = groq, prompt, tools) when is_binary(prompt) and is_list(tools) do
    messages = [
      Message.new_system!(),
      Message.new_user!(prompt)
    ]

    call(groq, messages, tools)
  end

  def call(%__MODULE__{} = groq, messages, tools)
      when is_list(messages) and is_list(tools) do
    metadata = %{
      model: groq.model,
      message_count: length(messages),
      tools_count: length(tools)
    }

    LangChain.Telemetry.span([:langchain, :llm, :call], metadata, fn ->
      try do
        # Track the prompt being sent
        LangChain.Telemetry.llm_prompt(
          %{system_time: System.system_time()},
          %{model: groq.model, messages: messages}
        )

        case do_api_request(groq, messages, tools) do
          {:error, reason} ->
            {:error, reason}

          parsed_data ->
            # Track the response being received
            LangChain.Telemetry.llm_response(
              %{system_time: System.system_time()},
              %{model: groq.model, response: parsed_data}
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
  def do_api_request(groq, messages, tools, retry_count \\ 3)

  def do_api_request(_groq, _messages, _tools, 0) do
    raise LangChainError, "Retries exceeded. Connection failed."
  end

  def do_api_request(
        %__MODULE__{stream: false} = groq,
        messages,
        tools,
        retry_count
      ) do
    raw_data = for_api(groq, messages, tools)

    if groq.verbose_api do
      IO.inspect(raw_data, label: "RAW DATA BEING SUBMITTED")
    end

    req =
      Req.new(
        url: groq.endpoint,
        json: raw_data,
        auth: {:bearer, get_api_key(groq)},
        headers: [
          {"api-key", get_api_key(groq)}
        ],
        receive_timeout: groq.receive_timeout,
        retry: :transient,
        max_retries: 3,
        retry_delay: fn attempt -> 300 * attempt end
      )

    req
    |> Req.post()
    |> case do
      {:ok, %Req.Response{body: data} = response} ->
        if groq.verbose_api do
          IO.inspect(response, label: "RAW REQ RESPONSE")
        end

        Callbacks.fire(groq.callbacks, :on_llm_token_usage, [
          get_token_usage(data)
        ])

        case do_process_response(groq, data) do
          {:error, %LangChainError{} = reason} ->
            {:error, reason}

          result ->
            # Track non-streaming response completion
            LangChain.Telemetry.emit_event(
              [:langchain, :llm, :response, :non_streaming],
              %{system_time: System.system_time()},
              %{
                model: groq.model,
                response_size: byte_size(inspect(result))
              }
            )

            Callbacks.fire(groq.callbacks, :on_llm_new_message, [result])
            result
        end

      {:error, %Req.TransportError{reason: :timeout} = err} ->
        {:error,
         LangChainError.exception(type: "timeout", message: "Request timed out", original: err)}

      {:error, %Req.TransportError{reason: :closed}} ->
        Logger.debug(fn -> "Connection closed: retry count = #{inspect(retry_count)}" end)
        do_api_request(groq, messages, tools, retry_count - 1)

      other ->
        Logger.error("Unexpected and unhandled API response! #{inspect(other)}")
        other
    end
  end

  def do_api_request(
        %__MODULE__{stream: true} = groq,
        messages,
        tools,
        retry_count
      ) do
    req =
      Req.new(
        url: groq.endpoint,
        json: for_api(groq, messages, tools),
        auth: {:bearer, get_api_key(groq)},
        headers: [
          {"api-key", get_api_key(groq)}
        ],
        receive_timeout: groq.receive_timeout
      )

    req
    |> Req.post(
      into:
        Utils.handle_stream_fn(
          groq,
          # Groq's streaming API is compatible with OpenAI's format
          &ChatOpenAI.decode_stream/1,
          &do_process_response(groq, &1)
        )
    )
    |> case do
      {:ok, %Req.Response{body: data, headers: headers} = _response} ->
        # Handle token usage from x-groq-* headers if present
        case get_token_usage_from_headers(headers) do
          %TokenUsage{} = usage ->
            Callbacks.fire(groq.callbacks, :on_llm_token_usage, [usage])
          _ -> :ok
        end
        
        data

      {:error, %Req.TransportError{reason: :timeout} = err} ->
        {:error,
         LangChainError.exception(type: "timeout", message: "Request timed out", original: err)}

      {:error, %Req.TransportError{reason: :closed}} ->
        Logger.debug(fn -> "Connection closed: retry count = #{inspect(retry_count)}" end)
        do_api_request(groq, messages, tools, retry_count - 1)

      other ->
        Logger.error("Unhandled and unexpected response from streamed call. #{inspect(other)}")
        {:error, LangChainError.exception(type: "unexpected_response", message: "Unexpected")}
    end
  end

  # Parse response to produce the appropriate LangChain structure
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
          
  # Handle token usage from Groq's x_groq field in streaming responses
  def do_process_response(model, %{"x_groq" => %{"usage" => usage}} = data) do
    token_usage = TokenUsage.new!(%{
      input: Map.get(usage, "prompt_tokens"),
      output: Map.get(usage, "completion_tokens"),
      raw: usage
    })
    
    Callbacks.fire(model.callbacks, :on_llm_token_usage, [token_usage])
    
    # Continue processing the main response data
    data = Map.delete(data, "x_groq")
    do_process_response(model, data)
  end
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

  # Partial 'delta' format handling
  def do_process_response(
        model,
        %{"delta" => delta_body, "finish_reason" => finish, "index" => index} = _msg
      ) do
    status = finish_reason_to_status(finish)

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
        # Groq includes role on the first chunk, but we default to assistant for consistency
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

  # Complete message with tool calls
  def do_process_response(
        model,
        %{"finish_reason" => finish_reason, "message" => %{"tool_calls" => calls} = message} = data
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

  # Tool call from a delta or complete message
  def do_process_response(
        _model,
        %{
          "function" => %{"arguments" => args, "name" => name},
          "id" => call_id
        } = data
      ) do
    # Note: index might be present or not, depending on context
    status = if Map.has_key?(data, "index"), do: :incomplete, else: :complete

    case ToolCall.new(%{
           type: :function,
           status: status,
           name: name,
           arguments: args,
           call_id: call_id,
           index: Map.get(data, "index")
         }) do
      {:ok, %ToolCall{} = call} ->
        call

      {:error, %Ecto.Changeset{} = changeset} ->
        reason = Utils.changeset_error_to_string(changeset)
        Logger.error("Failed to process ToolCall. Reason: #{reason}")
        {:error, LangChainError.exception(changeset)}
    end
  end

  # Standard API error from Groq
  def do_process_response(_model, %{"error" => %{"message" => reason}}) do
    Logger.error("Received error from Groq API: #{inspect(reason)}")
    {:error, LangChainError.exception(message: reason)}
  end

  # JSON decode errors
  def do_process_response(_model, {:error, %Jason.DecodeError{} = response}) do
    error_message = "Received invalid JSON: #{inspect(response)}"
    Logger.error(error_message)

    {:error,
     LangChainError.exception(type: "invalid_json", message: error_message, original: response)}
  end

  # Fallback for unexpected responses
  # Handle the direct message response format from Groq
  def do_process_response(_model, %{"message" => message, "index" => index, "finish_reason" => finish_reason}) do
    status = finish_reason_to_status(finish_reason)
    
    case Message.new(%{
           "status" => status,
           "role" => message["role"],
           "content" => message["content"],
           "complete" => true,
           "index" => index,
           "tool_calls" => message["tool_calls"] || []
         }) do
      {:ok, msg} ->
        msg

      {:error, %Ecto.Changeset{} = changeset} ->
        {:error, LangChainError.exception(changeset)}
    end
  end

  def do_process_response(_model, other) do
    Logger.error("Trying to process an unexpected response from Groq: #{inspect(other)}")

    {:error,
     LangChainError.exception(type: "unexpected_response", message: "Unexpected response")}
  end

  # Maps finish reasons to LangChain completion statuses
  defp finish_reason_to_status(nil), do: :incomplete
  defp finish_reason_to_status("stop"), do: :complete
  defp finish_reason_to_status("tool_calls"), do: :complete
  defp finish_reason_to_status("content_filter"), do: :complete
  defp finish_reason_to_status("length"), do: :length

  defp finish_reason_to_status(other) do
    Logger.warning("Unsupported finish_reason in message. Reason: #{inspect(other)}")
    nil
  end

  # Extract token usage information from response
  defp get_token_usage(%{"usage" => usage} = _response_body) do
    TokenUsage.new!(%{
      input: Map.get(usage, "prompt_tokens"),
      output: Map.get(usage, "completion_tokens"),
      raw: usage
    })
  end
  
  defp get_token_usage(_response_body), do: nil
  
  # Check for token usage in x-groq-* headers
  defp get_token_usage_from_headers(headers) do
    with prompt_tokens when not is_nil(prompt_tokens) <- get_header_value(headers, "x-groq-prompt-tokens"),
         completion_tokens when not is_nil(completion_tokens) <- get_header_value(headers, "x-groq-completion-tokens") do
      TokenUsage.new!(%{
        input: String.to_integer(prompt_tokens),
        output: String.to_integer(completion_tokens),
        raw: %{
          "prompt_tokens" => String.to_integer(prompt_tokens),
          "completion_tokens" => String.to_integer(completion_tokens),
          "total_tokens" => String.to_integer(prompt_tokens) + String.to_integer(completion_tokens)
        }
      })
    else
      _ -> nil
    end
  end
  
  defp get_header_value(headers, key) do
    # Try multiple case formats since header names can vary
    Enum.find_value(headers, fn {header_key, value} ->
      if String.downcase(header_key) == String.downcase(key), do: value, else: nil
    end)
  end

  @doc """
  Generate a config map that can later restore the model's configuration.
  """
  @impl ChatModel
  @spec serialize_config(t()) :: %{String.t() => any()}
  def serialize_config(%ChatGroq{} = model) do
    Utils.to_serializable_map(
      model,
      [
        :endpoint,
        :model,
        :temperature,
        :top_p,
        :receive_timeout,
        :max_tokens,
        :seed,
        :stream,
        :json_response,
        :json_schema
      ],
      @current_config_version
    )
  end

  @doc """
  Restores the model from a serialized config map.
  """
  @impl ChatModel
  def restore_from_map(%{"version" => 1} = data) do
    ChatGroq.new(data)
  end
end