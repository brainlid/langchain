defmodule LangChain.ChatModels.ChatDeepSeek do
  @moduledoc """
  Module for interacting with [DeepSeek models](https://www.deepseek.com/).

  DeepSeek provides an API that is compatible with OpenAI's API format, making it
  easy to integrate with existing OpenAI-based code.

  ## Model Options

  DeepSeek supports the following models:
  - `deepseek-chat` - Non-thinking mode of DeepSeek-V3.2-Exp
  - `deepseek-reasoner` - Thinking mode of DeepSeek-V3.2-Exp

  ## API Configuration

  The DeepSeek API uses the following configuration:
  - Base URL: `https://api.deepseek.com` (or `https://api.deepseek.com/v1` for OpenAI compatibility)
  - Authentication: Bearer token (API key)

  ## Example Usage

      # Basic usage
      model = ChatDeepSeek.new!(%{
        model: "deepseek-chat",
        api_key: "your-api-key-here"
      })

      # Using with LLMChain
      {:ok, chain} =
        LLMChain.new!(%{llm: model})
        |> LLMChain.add_message(Message.new_user!("Hello!"))

      {:ok, response} = LLMChain.run(chain)

  ## Tool Support

  DeepSeek supports function calling through the OpenAI-compatible API format.
  You can use tools in the same way as with OpenAI:

      model = ChatDeepSeek.new!(%{
        model: "deepseek-chat",
        api_key: "your-api-key-here"
      })

      function = Function.new!(%{
        name: "get_weather",
        description: "Get current weather for a location",
        parameters_schema: %{
          "type" => "object",
          "properties" => %{
            "location" => %{
              "type" => "string",
              "description" => "The city and state, e.g. San Francisco, CA"
            }
          },
          "required" => ["location"]
        }
      })

  ## Callbacks

  See the set of available callbacks: `LangChain.Chains.ChainCallbacks`

  ### Token Usage

  DeepSeek returns token usage information as part of the response body. The
  `LangChain.TokenUsage` is added to the `metadata` of the `LangChain.Message`
  and `LangChain.MessageDelta` structs that are processed under the `:usage`
  key.

  The `TokenUsage` data is accumulated for `MessageDelta` structs and the final usage information will be on the `LangChain.Message`.
  """
  use Ecto.Schema
  require Logger
  import Ecto.Changeset
  alias __MODULE__
  alias LangChain.Config
  alias LangChain.ChatModels.ChatModel
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
  alias LangChain.Callbacks

  @behaviour ChatModel

  @current_config_version 1

  # allow up to 1 minute for response.
  @receive_timeout 60_000

  @primary_key false
  embedded_schema do
    field :endpoint, :string, default: "https://api.deepseek.com/chat/completions"
    field :model, :string, default: "deepseek-chat"

    # API key for DeepSeek. If not set, will use global api key. Allows for usage
    # of a different API key per-call if desired. For instance, allowing a
    # customer to provide their own.
    field :api_key, :string, redact: true

    # What sampling temperature to use, between 0 and 2. Higher values like 0.8
    # will make the output more random, while lower values like 0.2 will make it
    # more focused and deterministic.
    field :temperature, :float, default: 1.0

    # Number between -2.0 and 2.0. Positive values penalize new tokens based on
    # their existing frequency in the text so far, decreasing the model's
    # likelihood to repeat the same line verbatim.
    field :frequency_penalty, :float, default: nil

    # Duration in seconds for the response to be received. When streaming a very
    # lengthy response, a longer time limit may be required. However, when it
    # goes on too long by itself, it tends to hallucinate more.
    field :receive_timeout, :integer, default: @receive_timeout

    # Seed for more deterministic output. Helpful for testing.
    field :seed, :integer

    # How many chat completion choices to generate for each input message.
    field :n, :integer, default: 1

    field :json_response, :boolean, default: false
    field :json_schema, :map, default: nil
    field :stream, :boolean, default: false
    field :max_tokens, :integer, default: nil

    # Options for streaming response. Only set this when you set `stream: true`
    field :stream_options, :map, default: nil

    # Tool choice option
    field :tool_choice, :map

    field :parallel_tool_calls, :boolean

    # Whether to return log probabilities of the output tokens or not
    field :logprobs, :boolean, default: false

    # Include the log probabilities on most likely tokens, as well the chosen tokens
    field :top_logprobs, :integer, default: nil

    # A list of maps for callback handlers (treated as internal)
    field :callbacks, {:array, :map}, default: []

    # Can send a string user_id to help detect abuse by users of the
    # application.
    field :user, :string

    # For help with debugging. It outputs the RAW Req response received and the
    # RAW Elixir map being submitted to the API.
    field :verbose_api, :boolean, default: false

    # Req options to merge into the request.
    # Refer to `https://hexdocs.pm/req/Req.html#new/1-options` for
    # `Req.new` supported set of options.
    field :req_config, :map, default: %{}
  end

  @type t :: %ChatDeepSeek{}

  @create_fields [
    :endpoint,
    :model,
    :temperature,
    :frequency_penalty,
    :api_key,
    :seed,
    :n,
    :stream,
    :receive_timeout,
    :json_response,
    :json_schema,
    :max_tokens,
    :stream_options,
    :user,
    :tool_choice,
    :parallel_tool_calls,
    :logprobs,
    :top_logprobs,
    :verbose_api,
    :req_config
  ]
  @required_fields [:endpoint, :model]

  @spec get_api_key(t()) :: String.t()
  defp get_api_key(%ChatDeepSeek{api_key: api_key}) do
    # if no API key is set default to `""` which will raise a DeepSeek API error
    api_key || Config.resolve(:deepseek_key, "")
  end

  @doc """
  Setup a ChatDeepSeek client configuration.
  """
  @spec new(attrs :: map()) :: {:ok, t} | {:error, Ecto.Changeset.t()}
  def new(%{} = attrs \\ %{}) do
    %ChatDeepSeek{}
    |> cast(attrs, @create_fields)
    |> common_validation()
    |> apply_action(:insert)
  end

  @doc """
  Setup a ChatDeepSeek client configuration and return it or raise an error if invalid.
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
    |> validate_number(:frequency_penalty, greater_than_or_equal_to: -2, less_than_or_equal_to: 2)
    |> validate_number(:n, greater_than_or_equal_to: 1, less_than_or_equal_to: 10)
    |> validate_number(:receive_timeout, greater_than_or_equal_to: 1000)
    |> validate_number(:max_tokens, greater_than_or_equal_to: 1)
    |> validate_number(:top_logprobs, greater_than_or_equal_to: 0, less_than_or_equal_to: 20)
    |> validate_top_logprobs_requires_logprobs()
    |> validate_endpoint_url()
  end

  defp validate_endpoint_url(changeset) do
    endpoint = get_field(changeset, :endpoint)

    if endpoint && !String.starts_with?(endpoint, "https://") &&
         !String.starts_with?(endpoint, "http://") do
      add_error(changeset, :endpoint, "must be a valid URL starting with http:// or https://")
    else
      changeset
    end
  end

  defp validate_top_logprobs_requires_logprobs(changeset) do
    logprobs = get_field(changeset, :logprobs)
    top_logprobs = get_field(changeset, :top_logprobs)

    if top_logprobs && !logprobs do
      add_error(changeset, :top_logprobs, "requires logprobs to be enabled")
    else
      changeset
    end
  end

  @doc """
  Return the params formatted for an API request.
  """
  @spec for_api(t | Message.t() | Function.t(), message :: [map()], ChatModel.tools()) :: %{
          atom() => any()
        }
  def for_api(%ChatDeepSeek{} = deepseek, messages, tools) do
    %{
      model: deepseek.model,
      temperature: deepseek.temperature,
      n: deepseek.n,
      stream: deepseek.stream,
      # a single ToolResult can expand into multiple tool messages for DeepSeek
      messages:
        messages
        |> Enum.flat_map(fn m ->
          case for_api(deepseek, m) do
            %{} = data ->
              [data]

            data when is_list(data) ->
              data
          end
        end)
    }
    |> Utils.conditionally_add_to_map(:user, deepseek.user)
    |> Utils.conditionally_add_to_map(:frequency_penalty, deepseek.frequency_penalty)
    |> Utils.conditionally_add_to_map(:response_format, set_response_format(deepseek))
    |> Utils.conditionally_add_to_map(:max_tokens, deepseek.max_tokens)
    |> Utils.conditionally_add_to_map(:seed, deepseek.seed)
    |> Utils.conditionally_add_to_map(
      :stream_options,
      get_stream_options_for_api(deepseek.stream_options)
    )
    |> Utils.conditionally_add_to_map(:tools, get_tools_for_api(deepseek, tools))
    |> Utils.conditionally_add_to_map(:tool_choice, get_tool_choice(deepseek))
    |> Utils.conditionally_add_to_map(:parallel_tool_calls, deepseek.parallel_tool_calls)
    |> Utils.conditionally_add_to_map(:logprobs, deepseek.logprobs)
    |> Utils.conditionally_add_to_map(:top_logprobs, deepseek.top_logprobs)
  end

  defp get_tools_for_api(%_{} = _model, nil), do: []

  defp get_tools_for_api(%_{} = model, tools) do
    Enum.map(tools, fn
      %Function{} = function ->
        %{"type" => "function", "function" => for_api(model, function)}
    end)
  end

  defp get_stream_options_for_api(nil), do: nil

  defp get_stream_options_for_api(%{} = data) do
    %{"include_usage" => Map.get(data, :include_usage, Map.get(data, "include_usage"))}
  end

  defp set_response_format(%ChatDeepSeek{json_response: true, json_schema: json_schema})
       when not is_nil(json_schema) do
    %{
      "type" => "json_schema",
      "json_schema" => json_schema
    }
  end

  defp set_response_format(%ChatDeepSeek{json_response: true}) do
    %{"type" => "json_object"}
  end

  defp set_response_format(%ChatDeepSeek{json_response: false}) do
    nil
  end

  defp get_tool_choice(%ChatDeepSeek{
         tool_choice: %{"type" => "function", "function" => %{"name" => name}} = _tool_choice
       })
       when is_binary(name) and byte_size(name) > 0,
       do: %{"type" => "function", "function" => %{"name" => name}}

  defp get_tool_choice(%ChatDeepSeek{tool_choice: %{"type" => type} = _tool_choice})
       when is_binary(type) and byte_size(type) > 0,
       do: type

  defp get_tool_choice(%ChatDeepSeek{}), do: nil

  @doc """
  Convert a LangChain Message-based structure to the expected map of data for
  the DeepSeek API.
  """
  @spec for_api(
          struct(),
          Message.t()
          | PromptTemplate.t()
          | ToolCall.t()
          | ToolResult.t()
          | ContentPart.t()
          | Function.t()
        ) ::
          %{String.t() => any()} | [%{String.t() => any()}]
  def for_api(%_{} = model, %Message{content: content} = msg) when is_list(content) do
    %{
      "role" => msg.role,
      "content" => ContentPart.content_to_string(content)
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
      "content" => ContentPart.content_to_string(msg.content)
    }
    |> Utils.conditionally_add_to_map("tool_calls", Enum.map(tool_calls, &for_api(model, &1)))
  end

  def for_api(%_{} = model, %Message{role: :user, content: content} = msg)
      when is_list(content) do
    %{
      "role" => msg.role,
      "content" => Enum.map(content, &for_api(model, &1))
    }
    |> Utils.conditionally_add_to_map("name", msg.name)
  end

  def for_api(%_{} = _model, %ToolResult{type: :function} = result) do
    # a ToolResult becomes a stand-alone %Message{role: :tool} response.
    %{
      "role" => :tool,
      "tool_call_id" => result.tool_call_id,
      "content" => ContentPart.content_to_string(result.content)
    }
  end

  def for_api(%_{} = _model, %Message{role: :tool, tool_results: tool_results} = _msg)
      when is_list(tool_results) do
    # ToolResults turn into a list of tool messages for DeepSeek
    Enum.map(tool_results, fn result ->
      %{
        "role" => :tool,
        "tool_call_id" => result.tool_call_id,
        "content" => ContentPart.content_to_string(result.content)
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

  # Function support
  def for_api(%_{} = _model, %Function{} = fun) do
    %{
      "name" => fun.name,
      "parameters" => get_parameters(fun)
    }
    |> Utils.conditionally_add_to_map("description", fun.description)
  end

  def for_api(%_{} = _model, %PromptTemplate{} = _template) do
    raise LangChainError, "PromptTemplates must be converted to messages."
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

  @doc """
  Calls the DeepSeek API passing the ChatDeepSeek struct with configuration, plus
  either a simple message or the list of messages to act as the prompt.

  Optionally pass in a list of tools available to the LLM for requesting
  execution in response.

  Optionally pass in a callback function that can be executed as data is
  received from the API.

  **NOTE:** This function *can* be used directly, but the primary interface
  should be through `LangChain.Chains.LLMChain`. The `ChatDeepSeek` module is more
  focused on translating the `LangChain` data structures to and from the DeepSeek
  API.

  Another benefit of using `LangChain.Chains.LLMChain` is that it combines the
  storage of messages, adding tools, adding custom context that should be
  passed to tools, and automatically applying `LangChain.MessageDelta`
  structs as they are are received, then converting those to the full
  `LangChain.Message` once fully complete.
  """
  @impl ChatModel
  def call(deepseek, prompt, tools \\ [])

  def call(%ChatDeepSeek{} = deepseek, prompt, tools) when is_binary(prompt) do
    messages = [
      Message.new_system!(),
      Message.new_user!(prompt)
    ]

    call(deepseek, messages, tools)
  end

  def call(%ChatDeepSeek{} = deepseek, messages, tools) when is_list(messages) do
    metadata = %{
      model: deepseek.model,
      message_count: length(messages),
      tools_count: length(tools)
    }

    LangChain.Telemetry.span([:langchain, :llm, :call], metadata, fn ->
      try do
        # Track the prompt being sent
        LangChain.Telemetry.llm_prompt(
          %{system_time: System.system_time()},
          %{model: deepseek.model, messages: messages}
        )

        # make base api request and perform high-level success/failure checks
        case do_api_request(deepseek, messages, tools) do
          {:error, %LangChainError{} = reason} ->
            {:error, reason}

          parsed_data ->
            # Track the response being received
            LangChain.Telemetry.llm_response(
              %{system_time: System.system_time()},
              %{model: deepseek.model, response: parsed_data}
            )

            {:ok, parsed_data}
        end
      rescue
        err in LangChainError ->
          {:error, err}
      end
    end)
  end

  # Make the API request from the DeepSeek server.
  @doc false
  @spec do_api_request(t(), [Message.t()], ChatModel.tools(), integer()) ::
          list() | struct() | {:error, LangChainError.t()}
  def do_api_request(deepseek, messages, tools, retry_count \\ 3)

  def do_api_request(_deepseek, _messages, _tools, 0) do
    raise LangChainError, "Retries exceeded. Connection failed."
  end

  def do_api_request(
        %ChatDeepSeek{stream: false} = deepseek,
        messages,
        tools,
        retry_count
      ) do
    raw_data = for_api(deepseek, messages, tools)

    if deepseek.verbose_api do
      IO.inspect(raw_data, label: "RAW DATA BEING SUBMITTED")
    end

    req =
      Req.new(
        url: deepseek.endpoint,
        json: raw_data,
        # required for DeepSeek API
        auth: {:bearer, get_api_key(deepseek)},
        headers: [
          {"Content-Type", "application/json"}
        ],
        receive_timeout: deepseek.receive_timeout,
        retry: :transient,
        max_retries: 3,
        retry_delay: fn attempt -> 300 * attempt end
      )

    req
    |> Req.merge(deepseek.req_config |> Keyword.new())
    |> Req.post()
    # parse the body and return it as parsed structs
    |> case do
      {:ok, %Req.Response{status: status, body: data} = response} when status in 200..299 ->
        if deepseek.verbose_api do
          IO.inspect(response, label: "RAW REQ RESPONSE")
        end

        Callbacks.fire(deepseek.callbacks, :on_llm_response_headers, [response.headers])

        case do_process_response(deepseek, data) do
          {:error, %LangChainError{} = reason} ->
            {:error, reason}

          result ->
            Callbacks.fire(deepseek.callbacks, :on_llm_new_message, [result])

            # Track non-streaming response completion
            LangChain.Telemetry.emit_event(
              [:langchain, :llm, :response, :non_streaming],
              %{system_time: System.system_time()},
              %{
                model: deepseek.model,
                response_size: byte_size(inspect(result))
              }
            )

            result
        end

      {:ok, %Req.Response{status: 400, body: body}} ->
        {:error,
         LangChainError.exception(type: "bad_request", message: "Bad request: #{inspect(body)}")}

      {:ok, %Req.Response{status: 401}} ->
        {:error,
         LangChainError.exception(type: "authentication_error", message: "Authentication failed")}

      {:ok, %Req.Response{status: 403}} ->
        {:error,
         LangChainError.exception(type: "permission_denied", message: "Permission denied")}

      {:ok, %Req.Response{status: 404}} ->
        {:error, LangChainError.exception(type: "not_found", message: "Endpoint not found")}

      {:ok, %Req.Response{status: 429}} ->
        {:error,
         LangChainError.exception(type: "rate_limit_exceeded", message: "Rate limit exceeded")}

      {:ok, %Req.Response{status: status}} when status in 500..599 ->
        {:error,
         LangChainError.exception(type: "server_error", message: "Server error: #{status}")}

      {:ok, %Req.Response{status: 529}} ->
        {:error, LangChainError.exception(type: "overloaded", message: "Overloaded")}

      {:error, %Req.TransportError{reason: reason} = err} ->
        case reason do
          :timeout ->
            {:error,
             LangChainError.exception(
               type: "timeout",
               message: "Request timed out",
               original: err
             )}

          :closed ->
            # Force a retry by making a recursive call decrementing the counter
            Logger.debug(fn ->
              "Mint connection closed: retry count = #{inspect(retry_count)}"
            end)

            do_api_request(deepseek, messages, tools, retry_count - 1)

          :nxdomain ->
            {:error,
             LangChainError.exception(
               type: "dns_error",
               message: "DNS resolution failed",
               original: err
             )}

          :econnrefused ->
            {:error,
             LangChainError.exception(
               type: "connection_refused",
               message: "Connection refused",
               original: err
             )}

          :connect_timeout ->
            {:error,
             LangChainError.exception(
               type: "connection_timeout",
               message: "Connection timeout",
               original: err
             )}

          other ->
            {:error,
             LangChainError.exception(
               type: "transport_error",
               message: "Transport error: #{inspect(other)}",
               original: err
             )}
        end

      other ->
        Logger.error("Unexpected and unhandled API response! #{inspect(other)}")
        other
    end
  end

  def do_api_request(
        %ChatDeepSeek{stream: true} = deepseek,
        messages,
        tools,
        retry_count
      ) do
    raw_data = for_api(deepseek, messages, tools)

    if deepseek.verbose_api do
      IO.inspect(raw_data, label: "RAW DATA BEING SUBMITTED")
    end

    Logger.debug("Raw Data #{inspect(raw_data)}")

    Req.new(
      url: deepseek.endpoint,
      json: raw_data,
      # required for DeepSeek API
      auth: {:bearer, get_api_key(deepseek)},
      headers: [
        {"Content-Type", "application/json"}
      ],
      receive_timeout: deepseek.receive_timeout,
      retry: :transient,
      max_retries: 3,
      retry_delay: fn attempt -> 300 * attempt end
    )
    |> Req.merge(deepseek.req_config |> Keyword.new())
    |> Req.post(
      into:
        Utils.handle_stream_fn(
          Map.take(deepseek, [:stream]),
          &decode_stream/1,
          &do_process_response(deepseek, &1)
        )
    )
    |> case do
      {:ok, %Req.Response{body: data} = response} ->
        Callbacks.fire(deepseek.callbacks, :on_llm_response_headers, [response.headers])

        data

      {:error, %LangChainError{} = error} ->
        {:error, error}

      {:error, %Req.TransportError{reason: reason} = err} ->
        case reason do
          :timeout ->
            {:error,
             LangChainError.exception(
               type: "timeout",
               message: "Request timed out",
               original: err
             )}

          :closed ->
            # Force a retry by making a recursive call decrementing the counter
            Logger.debug(fn -> "Connection closed: retry count = #{inspect(retry_count)}" end)
            do_api_request(deepseek, messages, tools, retry_count - 1)

          :nxdomain ->
            {:error,
             LangChainError.exception(
               type: "dns_error",
               message: "DNS resolution failed",
               original: err
             )}

          :econnrefused ->
            {:error,
             LangChainError.exception(
               type: "connection_refused",
               message: "Connection refused",
               original: err
             )}

          :connect_timeout ->
            {:error,
             LangChainError.exception(
               type: "connection_timeout",
               message: "Connection timeout",
               original: err
             )}

          other ->
            {:error,
             LangChainError.exception(
               type: "transport_error",
               message: "Transport error: #{inspect(other)}",
               original: err
             )}
        end

      other ->
        Logger.error(
          "Unhandled and unexpected response from streamed post call. #{inspect(other)}"
        )

        {:error,
         LangChainError.exception(type: "unexpected_response", message: "Unexpected response")}
    end
  end

  @doc """
  Decode a streamed response from a DeepSeek server. This is the same as the OpenAI
  implementation since DeepSeek uses an OpenAI-compatible API.
  """
  @spec decode_stream({String.t(), String.t()}, list(), non_neg_integer()) ::
          {%{String.t() => any()}} | {:error, LangChainError.t()}
  def decode_stream({raw_data, buffer}, done \\ [], depth \\ 0) do
    # Data comes back in the same format as OpenAI
    raw_data
    |> String.split("data: ")
    |> Enum.reduce({done, buffer}, fn str, {done, incomplete} = acc ->
      str
      |> String.trim()
      |> case do
        "" ->
          acc

        "[DONE]" ->
          acc

        json ->
          parse_combined_data(incomplete, json, done, depth)
      end
    end)
  end

  defp parse_combined_data(incomplete, json, done, depth)

  defp parse_combined_data("", json, done, _depth) do
    json
    |> Jason.decode()
    |> case do
      {:ok, parsed} ->
        {done ++ [parsed], ""}

      {:error, _reason} ->
        {done, json}
    end
  end

  defp parse_combined_data(incomplete, json, done, depth) when depth < 10 do
    # combine with any previous incomplete data
    starting_json = incomplete <> json

    # recursively call decode_stream so that the combined message data is split on "data: " again.
    decode_stream({starting_json, ""}, done, depth + 1)
  end

  defp parse_combined_data(_incomplete, _json, done, depth) when depth >= 10 do
    Logger.error("Stream parsing recursion limit exceeded: depth = #{depth}")
    {done, ""}
  end

  # Parse a new message response
  @doc false
  @spec do_process_response(
          ChatDeepSeek.t(),
          data :: %{String.t() => any()} | {:error, any()}
        ) ::
          :skip
          | Message.t()
          | [Message.t()]
          | MessageDelta.t()
          | [MessageDelta.t()]
          | {:error, LangChainError.t()}
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
        # process each response individually. Return a list of all processed
        # choices. If we received TokenUsage, attach it to each returned item.
        # Merging will work out later.
        choices
        |> Enum.map(&do_process_response(model, &1))
        |> Enum.map(fn result ->
          result
          |> TokenUsage.set(token_usage)
          |> merge_response_metadata(data)
        end)
    end
  end

  # Full message with tool call
  def do_process_response(
        model,
        %{"finish_reason" => finish_reason, "message" => %{"tool_calls" => calls} = message} =
          data
      )
      when finish_reason in ["tool_calls", "stop"] do
    # Extract DeepSeek-specific metadata
    metadata = extract_response_metadata(data)

    case Message.new(%{
           "role" => "assistant",
           "content" => message["content"],
           "complete" => true,
           "index" => data["index"],
           "tool_calls" => Enum.map(calls || [], &do_process_response(model, &1)),
           "metadata" => metadata
         }) do
      {:ok, message} ->
        message

      {:error, %Ecto.Changeset{} = changeset} ->
        {:error, LangChainError.exception(changeset)}
    end
  end

  # Delta message tool call
  def do_process_response(
        model,
        %{"delta" => delta_body, "index" => index} = msg
      ) do
    # finish_reason might not be present in all streaming responses
    finish = Map.get(msg, "finish_reason", nil)
    status = finish_reason_to_status(finish)

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

    # Extract DeepSeek-specific metadata for streaming responses
    metadata = extract_response_metadata(msg)

    data =
      delta_body
      |> Map.put("role", role)
      |> Map.put("index", index)
      |> Map.put("status", status)
      |> Map.put("tool_calls", tool_calls)
      |> Map.put("metadata", metadata)

    case MessageDelta.new(data) do
      {:ok, message} ->
        message

      {:error, %Ecto.Changeset{} = changeset} ->
        {:error, LangChainError.exception(changeset)}
    end
  end

  # Tool call as part of a delta message
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

  # Tool call from a complete message
  def do_process_response(_model, %{
        "function" => %{
          "arguments" => args,
          "name" => name
        },
        "id" => call_id,
        "type" => "function"
      }) do
    # Validate required fields
    cond do
      is_nil(call_id) or call_id == "" ->
        {:error,
         LangChainError.exception(
           type: "invalid_tool_call",
           message: "Tool call missing required field: id"
         )}

      is_nil(name) or name == "" ->
        {:error,
         LangChainError.exception(
           type: "invalid_tool_call",
           message: "Tool call missing required field: name"
         )}

      true ->
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
            reason = Utils.changeset_error_to_string(changeset)
            Logger.error("Failed to process ToolCall for a function. Reason: #{reason}")
            {:error, LangChainError.exception(changeset)}
        end
    end
  end

  def do_process_response(
        _model,
        %{
          "finish_reason" => finish_reason,
          "message" => message,
          "index" => index
        } = data
      ) do
    status = finish_reason_to_status(finish_reason)
    # Extract DeepSeek-specific metadata
    metadata = extract_response_metadata(data)

    case Message.new(
           Map.merge(message, %{"status" => status, "index" => index, "metadata" => metadata})
         ) do
      {:ok, message} ->
        message

      {:error, %Ecto.Changeset{} = changeset} ->
        {:error, LangChainError.exception(changeset)}
    end
  end

  def do_process_response(
        _model,
        %{
          "error" => %{"code" => code, "message" => reason} = _error_data
        } = response
      ) do
    type =
      case code do
        "429" -> "rate_limit_exceeded"
        _other -> nil
      end

    Logger.error("Received error from API: #{inspect(reason)}")
    {:error, LangChainError.exception(type: type, message: reason, original: response)}
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
    {:error, LangChainError.exception(message: "Unexpected response", original: other)}
  end

  # Merge response metadata with existing message metadata
  defp merge_response_metadata(message, response_data) do
    response_metadata = extract_response_metadata(response_data)

    current_metadata = message.metadata || %{}
    %{message | metadata: Map.merge(current_metadata, response_metadata)}
  end

  defp finish_reason_to_status(nil), do: :incomplete
  defp finish_reason_to_status("stop"), do: :complete
  defp finish_reason_to_status("tool_calls"), do: :complete
  defp finish_reason_to_status("content_filter"), do: :complete
  defp finish_reason_to_status("length"), do: :length
  defp finish_reason_to_status("max_tokens"), do: :length

  defp finish_reason_to_status(other) do
    Logger.warning("Unsupported finish_reason in message. Reason: #{inspect(other)}")
    nil
  end

  # Extract DeepSeek-specific metadata from API response.
  # This function extracts fields like `system_fingerprint`, `logprobs`, and other
  # DeepSeek-specific information from the response and stores it in the message metadata.
  defp extract_response_metadata(response_data) do
    metadata = %{}

    # Add system fingerprint if present
    metadata =
      case Map.get(response_data, "system_fingerprint") do
        nil -> metadata
        fingerprint -> Map.put(metadata, :system_fingerprint, fingerprint)
      end

    # Add object type if present
    metadata =
      case Map.get(response_data, "object") do
        nil -> metadata
        object -> Map.put(metadata, :object, object)
      end

    # Add logprobs information from choices (include even if nil)
    logprobs =
      case Map.get(response_data, "choices") do
        choices when is_list(choices) ->
          case choices do
            [single_choice] -> Map.get(single_choice, "logprobs")
            multiple_choices -> Enum.map(multiple_choices, &Map.get(&1, "logprobs"))
          end

        _ ->
          nil
      end

    Map.put(metadata, :logprobs, logprobs)
  end

  defp get_token_usage(%{"usage" => usage} = _response_body) when is_map(usage) do
    # extract out the reported response token usage
    # DeepSeek provides detailed token usage information
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
  def retry_on_fallback?(%LangChainError{type: "server_error"}), do: true
  def retry_on_fallback?(%LangChainError{type: "overloaded"}), do: true
  def retry_on_fallback?(_), do: false

  @doc """
  Generate a config map that can later restore the model's configuration.
  """
  @impl ChatModel
  @spec serialize_config(t()) :: %{String.t() => any()}
  def serialize_config(%ChatDeepSeek{} = model) do
    Utils.to_serializable_map(
      model,
      [
        :endpoint,
        :model,
        :temperature,
        :frequency_penalty,
        :receive_timeout,
        :seed,
        :n,
        :json_response,
        :json_schema,
        :stream,
        :max_tokens,
        :stream_options
      ],
      @current_config_version
    )
  end

  @doc """
  Restores the model from the config.
  """
  @impl ChatModel
  def restore_from_map(%{"version" => 1} = data) do
    ChatDeepSeek.new(data)
  end
end
