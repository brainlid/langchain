defmodule LangChain.ChatModels.ChatOpenAI do
  @moduledoc """
  Represents the [OpenAI
  ChatModel](https://platform.openai.com/docs/api-reference/chat/create).

  Parses and validates inputs for making a requests from the OpenAI Chat API.

  Converts responses into more specialized `LangChain` data structures.

  - https://github.com/openai/openai-cookbook/blob/main/examples/How_to_call_functions_with_chat_models.ipynb


  ## Callbacks

  See the set of available callback: `LangChain.ChatModels.LLMCallbacks`

  ### Rate Limit API Response Headers

  OpenAI returns rate limit information in the response headers. Those can be
  accessed using the LLM callback `on_llm_ratelimit_info` like this:

      handlers = %{
        on_llm_ratelimit_info: fn _model, headers ->
          IO.inspect(headers)
        end
      }

      {:ok, chat} = ChatOpenAI.new(%{callbacks: [handlers]})

  When a request is received, something similar to the following will be output
  to the console.

      %{
        "x-ratelimit-limit-requests" => ["5000"],
        "x-ratelimit-limit-tokens" => ["160000"],
        "x-ratelimit-remaining-requests" => ["4999"],
        "x-ratelimit-remaining-tokens" => ["159973"],
        "x-ratelimit-reset-requests" => ["12ms"],
        "x-ratelimit-reset-tokens" => ["10ms"],
        "x-request-id" => ["req_1234"]
      }

  ### Token Usage

  OpenAI returns token usage information as part of the response body. That data
  can be accessed using the LLM callback `on_llm_token_usage` like this:

      handlers = %{
        on_llm_token_usage: fn _model, usage ->
          IO.inspect(usage)
        end
      }

      {:ok, chat} = ChatOpenAI.new(%{
        callbacks: [handlers],
        stream: true,
        stream_options: %{include_usage: true}
      })

  When a request is received, something similar to the following will be output
  to the console.

      %LangChain.TokenUsage{input: 15, output: 3}

  The OpenAI documentation instructs to provide the `stream_options` with the
  `include_usage: true` for the information to be provided.

  ## Tool Choice

  OpenAI's ChatGPT API supports forcing a tool to be used.
  - https://platform.openai.com/docs/api-reference/chat/create#chat-create-tool_choice

  This is supported through the `tool_choice` options. It takes a plain Elixir map to provide the configuration.

  By default, the LLM will choose a tool call if a tool is available and it determines it is needed. That's the "auto" mode.

  ### Example
  For the LLM's response to make a tool call of the "get_weather" function.

      ChatOpenAI.new(%{
        model: "...",
        tool_choice: %{"type" => "function", "function" => %{"name" => "get_weather"}}
      })

  ## Azure OpenAI Support

  To use `ChatOpenAI` with Microsoft's Azure hosted OpenAI models, the `endpoint` must be overridden and the API key needs to be provided in some way. The [MS Quickstart guide for REST access](https://learn.microsoft.com/en-us/azure/ai-services/openai/chatgpt-quickstart?tabs=command-line%2Cjavascript-keyless%2Ctypescript-keyless%2Cpython-new&pivots=rest-api) may be helpful.

  In order to use it, you must have an Azure account and from the console, a model must be deployed for your account. Use the Azure AI Foundry and Azure OpenAI Service to deploy the model you want to use. The entire URL is used as the `endpoint` and the provided `key` is used as the `api_key`.

  The following is an example of setting up `ChatOpenAI` for use with an Azure hosted model.

      endpoint = System.fetch_env!("AZURE_OPENAI_ENDPOINT")
      api_key = System.fetch_env!("AZURE_OPENAI_KEY")

      llm =
        ChatOpenAI.new!(%{
          endpoint: endpoint,
          api_key: api_key,
          seed: 0,
          temperature: 1,
          stream: false
        })

  The URL itself specifies the model to use and the `model` attribute is disregarded.

  A fake example URL for the endpoint value:

  `https://some-subdomain.cognitiveservices.azure.com/openai/deployments/gpt-4o-mini/chat/completions?api-version=2024-08-01-preview"`

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
  alias LangChain.TokenUsage
  alias LangChain.Function
  alias LangChain.FunctionParam
  alias LangChain.LangChainError
  alias LangChain.Utils
  alias LangChain.MessageDelta
  alias LangChain.Callbacks

  @behaviour ChatModel

  @current_config_version 1

  # NOTE: As of gpt-4 and gpt-3.5, only one function_call is issued at a time
  # even when multiple requests could be issued based on the prompt.

  # allow up to 1 minute for response.
  @receive_timeout 60_000

  @primary_key false
  embedded_schema do
    field :endpoint, :string, default: "https://api.openai.com/v1/chat/completions"
    # field :model, :string, default: "gpt-4"
    field :model, :string, default: "gpt-3.5-turbo"
    # API key for OpenAI. If not set, will use global api key. Allows for usage
    # of a different API key per-call if desired. For instance, allowing a
    # customer to provide their own.
    field :api_key, :string

    # What sampling temperature to use, between 0 and 2. Higher values like 0.8
    # will make the output more random, while lower values like 0.2 will make it
    # more focused and deterministic.
    field :temperature, :float, default: 1.0
    # Number between -2.0 and 2.0. Positive values penalize new tokens based on
    # their existing frequency in the text so far, decreasing the model's
    # likelihood to repeat the same line verbatim.
    field :frequency_penalty, :float, default: 0.0
    # Duration in seconds for the response to be received. When streaming a very
    # lengthy response, a longer time limit may be required. However, when it
    # goes on too long by itself, it tends to hallucinate more.
    field :receive_timeout, :integer, default: @receive_timeout
    # Seed for more deterministic output. Helpful for testing.
    # https://platform.openai.com/docs/guides/text-generation/reproducible-outputs
    field :seed, :integer
    # How many chat completion choices to generate for each input message.
    field :n, :integer, default: 1
    field :json_response, :boolean, default: false
    field :json_schema, :map, default: nil
    field :stream, :boolean, default: false
    field :max_tokens, :integer, default: nil
    # Options for streaming response. Only set this when you set `stream: true`
    # https://platform.openai.com/docs/api-reference/chat/create#chat-create-stream_options
    #
    # Set to `%{include_usage: true}` to have token usage returned when
    # streaming.
    field :stream_options, :map, default: nil

    # Tool choice option
    field :tool_choice, :map

    # A list of maps for callback handlers
    field :callbacks, {:array, :map}, default: []

    # Can send a string user_id to help ChatGPT detect abuse by users of the
    # application.
    # https://platform.openai.com/docs/guides/safety-best-practices/end-user-ids
    field :user, :string
  end

  @type t :: %ChatOpenAI{}

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
    :callbacks,
    :tool_choice
  ]
  @required_fields [:endpoint, :model]

  @spec get_api_key(t()) :: String.t()
  defp get_api_key(%ChatOpenAI{api_key: api_key}) do
    # if no API key is set default to `""` which will raise a OpenAI API error
    api_key || Config.resolve(:openai_key, "")
  end

  @spec get_org_id() :: String.t() | nil
  defp get_org_id() do
    Config.resolve(:openai_org_id)
  end

  @spec get_proj_id() :: String.t() | nil
  defp get_proj_id() do
    Config.resolve(:openai_proj_id)
  end

  @doc """
  Setup a ChatOpenAI client configuration.
  """
  @spec new(attrs :: map()) :: {:ok, t} | {:error, Ecto.Changeset.t()}
  def new(%{} = attrs \\ %{}) do
    %ChatOpenAI{}
    |> cast(attrs, @create_fields)
    |> common_validation()
    |> apply_action(:insert)
  end

  @doc """
  Setup a ChatOpenAI client configuration and return it or raise an error if invalid.
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
    |> validate_number(:n, greater_than_or_equal_to: 1)
    |> validate_number(:receive_timeout, greater_than_or_equal_to: 0)
  end

  @doc """
  Return the params formatted for an API request.
  """
  @spec for_api(t | Message.t() | Function.t(), message :: [map()], ChatModel.tools()) :: %{
          atom() => any()
        }
  def for_api(%ChatOpenAI{} = openai, messages, tools) do
    %{
      model: openai.model,
      temperature: openai.temperature,
      frequency_penalty: openai.frequency_penalty,
      n: openai.n,
      stream: openai.stream,
      # a single ToolResult can expand into multiple tool messages for OpenAI
      messages:
        messages
        |> Enum.reduce([], fn m, acc ->
          case for_api(m) do
            %{} = data ->
              [data | acc]

            data when is_list(data) ->
              Enum.reverse(data) ++ acc
          end
        end)
        |> Enum.reverse(),
      response_format: set_response_format(openai),
      user: openai.user
    }
    |> Utils.conditionally_add_to_map(:max_tokens, openai.max_tokens)
    |> Utils.conditionally_add_to_map(:seed, openai.seed)
    |> Utils.conditionally_add_to_map(
      :stream_options,
      get_stream_options_for_api(openai.stream_options)
    )
    |> Utils.conditionally_add_to_map(:tools, get_tools_for_api(tools))
    |> Utils.conditionally_add_to_map(:tool_choice, get_tool_choice(openai))
  end

  defp get_tools_for_api(nil), do: []

  defp get_tools_for_api(tools) do
    Enum.map(tools, fn
      %Function{} = function ->
        %{"type" => "function", "function" => for_api(function)}
    end)
  end

  defp get_stream_options_for_api(nil), do: nil

  defp get_stream_options_for_api(%{} = data) do
    %{"include_usage" => Map.get(data, :include_usage, Map.get(data, "include_usage"))}
  end

  defp set_response_format(%ChatOpenAI{json_response: true, json_schema: json_schema})
       when not is_nil(json_schema) do
    %{
      "type" => "json_schema",
      "json_schema" => json_schema
    }
  end

  defp set_response_format(%ChatOpenAI{json_response: true}) do
    %{"type" => "json_object"}
  end

  defp set_response_format(%ChatOpenAI{json_response: false}) do
    %{"type" => "text"}
  end

  defp get_tool_choice(%ChatOpenAI{
         tool_choice: %{"type" => "function", "function" => %{"name" => name}} = _tool_choice
       })
       when is_binary(name) and byte_size(name) > 0,
       do: %{"type" => "function", "function" => %{"name" => name}}

  defp get_tool_choice(%ChatOpenAI{tool_choice: %{"type" => type} = _tool_choice})
       when is_binary(type) and byte_size(type) > 0,
       do: type

  defp get_tool_choice(%ChatOpenAI{}), do: nil

  @doc """
  Convert a LangChain structure to the expected map of data for the OpenAI API.
  """
  @spec for_api(Message.t() | ContentPart.t() | Function.t()) ::
          %{String.t() => any()} | [%{String.t() => any()}]
  def for_api(%Message{role: :assistant, tool_calls: tool_calls} = msg)
      when is_list(tool_calls) do
    %{
      "role" => :assistant,
      "content" => msg.content
    }
    |> Utils.conditionally_add_to_map("tool_calls", Enum.map(tool_calls, &for_api(&1)))
  end

  def for_api(%Message{role: :tool, tool_results: tool_results} = _msg)
      when is_list(tool_results) do
    # ToolResults turn into a list of tool messages for OpenAI
    Enum.map(tool_results, fn result ->
      %{
        "role" => :tool,
        "tool_call_id" => result.tool_call_id,
        "content" => result.content
      }
    end)
  end

  def for_api(%Message{content: content} = msg) when is_binary(content) do
    %{
      "role" => msg.role,
      "content" => msg.content
    }
    |> Utils.conditionally_add_to_map("name", msg.name)
  end

  def for_api(%Message{role: :user, content: content} = msg) when is_list(content) do
    %{
      "role" => msg.role,
      "content" => Enum.map(content, &for_api(&1))
    }
    |> Utils.conditionally_add_to_map("name", msg.name)
  end

  def for_api(%ToolResult{type: :function} = result) do
    # a ToolResult becomes a stand-alone %Message{role: :tool} response.
    %{
      "role" => :tool,
      "tool_call_id" => result.tool_call_id,
      "content" => result.content
    }
  end

  def for_api(%LangChain.PromptTemplate{} = _template) do
    raise LangChain.LangChainError, "PromptTemplates must be converted to messages."
  end

  def for_api(%ContentPart{type: :text} = part) do
    %{"type" => "text", "text" => part.content}
  end

  def for_api(%ContentPart{type: image} = part) when image in [:image, :image_url] do
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
          Logger.error(message)
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

  # ToolCall support
  def for_api(%ToolCall{type: :function} = fun) do
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
  def for_api(%Function{} = fun) do
    %{
      "name" => fun.name,
      "parameters" => get_parameters(fun)
    }
    |> Utils.conditionally_add_to_map("description", fun.description)
  end

  defp get_parameters(%Function{parameters: [], parameters_schema: nil} = _fun) do
    %{
      "type" => "object",
      "properties" => %{}
    }
  end

  defp get_parameters(%Function{parameters: [], parameters_schema: schema} = _fun)
       when is_map(schema) do
    schema
  end

  defp get_parameters(%Function{parameters: params} = _fun) do
    FunctionParam.to_parameters_schema(params)
  end

  @doc """
  Calls the OpenAI API passing the ChatOpenAI struct with configuration, plus
  either a simple message or the list of messages to act as the prompt.

  Optionally pass in a list of tools available to the LLM for requesting
  execution in response.

  Optionally pass in a callback function that can be executed as data is
  received from the API.

  **NOTE:** This function *can* be used directly, but the primary interface
  should be through `LangChain.Chains.LLMChain`. The `ChatOpenAI` module is more
  focused on translating the `LangChain` data structures to and from the OpenAI
  API.

  Another benefit of using `LangChain.Chains.LLMChain` is that it combines the
  storage of messages, adding tools, adding custom context that should be
  passed to tools, and automatically applying `LangChain.MessageDelta`
  structs as they are are received, then converting those to the full
  `LangChain.Message` once fully complete.
  """
  @impl ChatModel
  def call(openai, prompt, tools \\ [])

  def call(%ChatOpenAI{} = openai, prompt, tools) when is_binary(prompt) do
    messages = [
      Message.new_system!(),
      Message.new_user!(prompt)
    ]

    call(openai, messages, tools)
  end

  def call(%ChatOpenAI{} = openai, messages, tools) when is_list(messages) do
    try do
      # make base api request and perform high-level success/failure checks
      case do_api_request(openai, messages, tools) do
        {:error, reason} ->
          {:error, reason}

        parsed_data ->
          {:ok, parsed_data}
      end
    rescue
      err in LangChainError ->
        {:error, err}
    end
  end

  # Make the API request from the OpenAI server.
  #
  # The result of the function is:
  #
  # - `result` - where `result` is a data-structure like a list or map.
  # - `{:error, reason}` - Where reason is a string explanation of what went wrong.
  #
  # If a callback_fn is provided, it will fire with each

  # When `stream: true` is
  # If `stream: false`, the completed message is returned.
  #
  # If `stream: true`, the `callback_fn` is executed for the returned MessageDelta
  # responses.
  #
  # Executes the callback function passing the response only parsed to the data
  # structures.
  # Retries the request up to 3 times on transient errors with a 1 second delay
  @doc false
  @spec do_api_request(t(), [Message.t()], ChatModel.tools(), integer()) ::
          list() | struct() | {:error, LangChainError.t()}
  def do_api_request(openai, messages, tools, retry_count \\ 3)

  def do_api_request(_openai, _messages, _tools, 0) do
    raise LangChainError, "Retries exceeded. Connection failed."
  end

  def do_api_request(
        %ChatOpenAI{stream: false} = openai,
        messages,
        tools,
        retry_count
      ) do
    req =
      Req.new(
        url: openai.endpoint,
        json: for_api(openai, messages, tools),
        # required for OpenAI API
        auth: {:bearer, get_api_key(openai)},
        # required for Azure OpenAI version
        headers: [
          {"api-key", get_api_key(openai)}
        ],
        receive_timeout: openai.receive_timeout,
        retry: :transient,
        max_retries: 3,
        retry_delay: fn attempt -> 300 * attempt end
      )

    req
    |> maybe_add_org_id_header()
    |> maybe_add_proj_id_header()
    |> Req.post()
    # parse the body and return it as parsed structs
    |> case do
      {:ok, %Req.Response{body: data} = response} ->
        Callbacks.fire(openai.callbacks, :on_llm_ratelimit_info, [
          openai,
          get_ratelimit_info(response.headers)
        ])

        Callbacks.fire(openai.callbacks, :on_llm_token_usage, [
          openai,
          get_token_usage(data)
        ])

        case do_process_response(openai, data) do
          {:error, %LangChainError{} = reason} ->
            {:error, reason}

          result ->
            Callbacks.fire(openai.callbacks, :on_llm_new_message, [openai, result])
            result
        end

      {:error, %Req.TransportError{reason: :timeout} = err} ->
        {:error,
         LangChainError.exception(type: "timeout", message: "Request timed out", original: err)}

      {:error, %Req.TransportError{reason: :closed}} ->
        # Force a retry by making a recursive call decrementing the counter
        Logger.debug(fn -> "Mint connection closed: retry count = #{inspect(retry_count)}" end)
        do_api_request(openai, messages, tools, retry_count - 1)

      other ->
        Logger.error("Unexpected and unhandled API response! #{inspect(other)}")
        other
    end
  end

  def do_api_request(
        %ChatOpenAI{stream: true} = openai,
        messages,
        tools,
        retry_count
      ) do
    Req.new(
      url: openai.endpoint,
      json: for_api(openai, messages, tools),
      # required for OpenAI API
      auth: {:bearer, get_api_key(openai)},
      # required for Azure OpenAI version
      headers: [
        {"api-key", get_api_key(openai)}
      ],
      receive_timeout: openai.receive_timeout
    )
    |> maybe_add_org_id_header()
    |> maybe_add_proj_id_header()
    |> Req.post(
      into: Utils.handle_stream_fn(openai, &decode_stream/1, &do_process_response(openai, &1))
    )
    |> case do
      {:ok, %Req.Response{body: data} = response} ->
        Callbacks.fire(openai.callbacks, :on_llm_ratelimit_info, [
          openai,
          get_ratelimit_info(response.headers)
        ])

        data

      {:error, %LangChainError{} = error} ->
        {:error, error}

      {:error, %Req.TransportError{reason: :timeout} = err} ->
        {:error,
         LangChainError.exception(type: "timeout", message: "Request timed out", original: err)}

      {:error, %Req.TransportError{reason: :closed}} ->
        # Force a retry by making a recursive call decrementing the counter
        Logger.debug(fn -> "Connection closed: retry count = #{inspect(retry_count)}" end)
        do_api_request(openai, messages, tools, retry_count - 1)

      other ->
        Logger.error(
          "Unhandled and unexpected response from streamed post call. #{inspect(other)}"
        )

        {:error,
         LangChainError.exception(type: "unexpected_response", message: "Unexpected response")}
    end
  end

  @doc """
  Decode a streamed response from an OpenAI-compatible server. Parses a string
  of received content into an Elixir map data structure using string keys.

  If a partial response was received, meaning the JSON text is split across
  multiple data frames, then the incomplete portion is returned as-is in the
  buffer. The function will be successively called, receiving the incomplete
  buffer data from a previous call, and assembling it to parse.
  """
  @spec decode_stream({String.t(), String.t()}) :: {%{String.t() => any()}}
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

  # Parse a new message response
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
  def do_process_response(model, %{"choices" => [], "usage" => %{} = _usage} = data) do
    case get_token_usage(data) do
      %TokenUsage{} = token_usage ->
        Callbacks.fire(model.callbacks, :on_llm_token_usage, [model, token_usage])
        :ok

      nil ->
        :ok
    end

    # this stand-alone TokenUsage message is skipped and not returned
    :skip
  end

  def do_process_response(_model, %{"choices" => []}), do: :skip

  def do_process_response(model, %{"choices" => choices} = _data) when is_list(choices) do
    # process each response individually. Return a list of all processed choices
    for choice <- choices do
      do_process_response(model, choice)
    end
  end

  # Full message with tool call
  def do_process_response(
        model,
        %{"finish_reason" => finish_reason, "message" => %{"tool_calls" => calls} = message} =
          data
      )
      when finish_reason in ["tool_calls", "stop"] do
    case Message.new(%{
           "role" => "assistant",
           "content" => message["content"],
           "complete" => true,
           "index" => data["index"],
           "tool_calls" => Enum.map(calls, &do_process_response(model, &1))
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
        %{"delta" => delta_body, "finish_reason" => finish, "index" => index} = _msg
      ) do
    status = finish_reason_to_status(finish)

    tool_calls =
      case delta_body do
        %{"tool_calls" => tools_data} when is_list(tools_data) ->
          Enum.map(tools_data, &do_process_response(model, &1))

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

    case MessageDelta.new(data) do
      {:ok, message} ->
        message

      {:error, %Ecto.Changeset{} = changeset} ->
        {:error, LangChainError.exception(changeset)}
    end
  end

  # Tool call as part of a delta message
  def do_process_response(_model, %{"function" => func_body, "index" => index} = tool_call) do
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

  def do_process_response(_model, %{
        "finish_reason" => finish_reason,
        "message" => message,
        "index" => index
      }) do
    status = finish_reason_to_status(finish_reason)

    case Message.new(Map.merge(message, %{"status" => status, "index" => index})) do
      {:ok, message} ->
        message

      {:error, %Ecto.Changeset{} = changeset} ->
        {:error, LangChainError.exception(changeset)}
    end
  end

  # MS Azure returns numeric error codes. Interpret them when possible to give a computer-friendly reason
  #
  # https://learn.microsoft.com/en-us/troubleshoot/azure/azure-kubernetes/create-upgrade-delete/429-too-many-requests-errors
  def do_process_response(_model, %{"error" => %{"code" => code, "message" => reason}}) do
    type =
      case code do
        "429" ->
          "rate_limit_exceeded"

        _other ->
          nil
      end

    Logger.error("Received error from API: #{inspect(reason)}")
    {:error, LangChainError.exception(type: type, message: reason)}
  end

  def do_process_response(_model, %{"error" => %{"message" => reason}}) do
    Logger.error("Received error from API: #{inspect(reason)}")
    {:error, LangChainError.exception(message: reason)}
  end

  def do_process_response(_model, {:error, %Jason.DecodeError{} = response}) do
    error_message = "Received invalid JSON: #{inspect(response)}"
    Logger.error(error_message)

    {:error,
     LangChainError.exception(type: "invalid_json", message: error_message, original: response)}
  end

  def do_process_response(_model, other) do
    Logger.error("Trying to process an unexpected response. #{inspect(other)}")
    {:error, LangChainError.exception(message: "Unexpected response")}
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

  defp maybe_add_org_id_header(%Req.Request{} = req) do
    org_id = get_org_id()

    if org_id do
      Req.Request.put_header(req, "OpenAI-Organization", org_id)
    else
      req
    end
  end

  defp maybe_add_proj_id_header(%Req.Request{} = req) do
    proj_id = get_proj_id()

    if proj_id do
      Req.Request.put_header(req, "OpenAI-Project", proj_id)
    else
      req
    end
  end

  defp get_ratelimit_info(response_headers) do
    # extract out all the ratelimit response headers
    #
    #  https://platform.openai.com/docs/guides/rate-limits/rate-limits-in-headers
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

  defp get_token_usage(%{"usage" => usage} = _response_body) do
    # extract out the reported response token usage
    #
    #  https://platform.openai.com/docs/api-reference/chat/object#chat/object-usage
    TokenUsage.new!(%{
      input: Map.get(usage, "prompt_tokens"),
      output: Map.get(usage, "completion_tokens")
    })
  end

  defp get_token_usage(_response_body), do: nil

  @doc """
  Generate a config map that can later restore the model's configuration.
  """
  @impl ChatModel
  @spec serialize_config(t()) :: %{String.t() => any()}
  def serialize_config(%ChatOpenAI{} = model) do
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
    ChatOpenAI.new(data)
  end
end
