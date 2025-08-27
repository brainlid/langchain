defmodule LangChain.ChatModels.ChatOpenAIResponses do
  @moduledoc """
  Represents the OpenAI Responses API

  Parses and validates inputs for making requests to the OpenAI Responses API.

  Converts responses into more specialized `LangChain` data structures.

  ## ContentPart Types

  OpenAI's Responses API supports several types of content parts that can be combined in a single message:

  ### Text Content
  Basic text content is the default and most common type:

      Message.new_user!("Hello, how are you?")

  ### Image Content
  OpenAI supports both base64-encoded images and image URLs:

      # Using a base64 encoded image
      Message.new_user!([
        ContentPart.text!("What's in this image?"),
        ContentPart.image!("base64_encoded_image_data", media: :jpg)
      ])

      # Using an image URL
      Message.new_user!([
        ContentPart.text!("Describe this image:"),
        ContentPart.image_url!("https://example.com/image.jpg")
      ])

  For images, you can specify the detail level which affects token usage:
  - `detail: "low"` - Lower resolution, fewer tokens
  - `detail: "high"` - Higher resolution, more tokens
  - `detail: "auto"` - Let the model decide

  ### File Content
  OpenAI supports both base64-encoded files and file IDs:

      # Using a base64 encoded file
      Message.new_user!([
        ContentPart.text!("Process this file:"),
        ContentPart.file!("base64_encoded_file_data",
          type: :base64,
          filename: "document.pdf"
        )
      ])

      # Using a file ID (after uploading to OpenAI)
      Message.new_user!([
        ContentPart.text!("Process this file:"),
        ContentPart.file!("file-1234", type: :file_id)
      ])

  ## Callbacks

  See the set of available callbacks: `LangChain.Chains.ChainCallbacks`

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

  OpenAI returns token usage information as part of the response body. The
  `LangChain.TokenUsage` is added to the `metadata` of the `LangChain.Message`
  and `LangChain.MessageDelta` structs that are processed under the `:usage`
  key.

  The OpenAI documentation instructs to provide the `stream_options` with the
  `include_usage: true` for the information to be provided.

  The `TokenUsage` data is accumulated for `MessageDelta` structs and the final usage information will be on the `LangChain.Message`.

  NOTE: Of special note is that the `TokenUsage` information is returned once
  for all "choices" in the response. The `LangChain.TokenUsage` data is added to
  each message, but if your usage requests multiple choices, you will see the
  same usage information for each choice but it is duplicated and only one
  response is meaningful.

  ## Native Tools (Web Search)

  Open AI's Responses API also supports built-in tools. Among those, we support Web Search currently.

  ### Example
  To optionally permit the model to use web search:

      native_web_tool = NativeTool.new!(%{name: "web_search_preview", configuration: %{}})

      %{llm: ChatOpenAIResponses.new!(%{model: "gpt-4o"})}
      |> LLMChain.new!()
      |> LLMChain.add_message(Message.new_user!("Can you tell me something that happened today in Texas?"))
      |> LLMChain.add_tools(web_tool)
      |> LLMChain.run()

  You may provide additional configuration per the OpenAI documentation:

      web_config = %{
        search_context_size: "medium",
        user_location: %{
          type: "approximate",
          city: "Humble",
          country: "US",
          region: "Texas",
          timezone: "America/Chicago"
        }
      }
      native_web_tool = NativeTool.new!(%{name: "web_search_preview", configuration: web_config)

  You may reference a prior web_search_call in subsequent runs as:

      Message.new_assistant!([
        ContentPart.new!(%{
          type: :unsupported,
            options: %{
              id: "ws_123456789", # ID as provided from Open AI
              status: "completed",
              type: "web_search_call"
            }
          }
        ),
        ContentPart.text!("The Astros won today 5-4...")
      ])

  Note: Not all Open AI models support `web_search_preview`. OpenAI will return an error if you request web_search_preview for when using a model that doesn't support it.

  ## Tool Choice

  OpenAI's ChatGPT API supports forcing a tool to be used.
  - https://platform.openai.com/docs/api-reference/chat/create#chat-create-tool_choice

  This is supported through the `tool_choice` options. It takes a plain Elixir
  map to provide the configuration.

  By default, the LLM will choose a tool call if a tool is available and it
  determines it is needed. That's the "auto" mode.

  ### Example
  For the LLM's response to make a tool call of the "get_weather" function.

      ChatOpenAI.new(%{
        model: "...",
        tool_choice: %{"type" => "function", "function" => %{"name" => "get_weather"}}
      })

  ...or to force a native tool (such as web search):

      ChatOpenAI.new(%{
        model: "...",
        tool_choice: "web_search_preview"
      })

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
  alias LangChain.NativeTool
  alias LangChain.FunctionParam
  alias LangChain.LangChainError
  alias LangChain.Utils
  alias LangChain.MessageDelta
  alias LangChain.Callbacks

  @behaviour ChatModel

  @current_config_version 1

  @receive_timeout 60_000

  @primary_key false

  # https://platform.openai.com/docs/api-reference/responses/create
  embedded_schema do
    field :receive_timeout, :integer, default: @receive_timeout
    field :api_key, :string, redact: true
    field :endpoint, :string, default: "https://api.openai.com/v1/responses"

    field :model, :string, default: "gpt-3.5-turbo"

    field :include, {:array, :string}, default: []
    # omit instructions becasue langchain assumes statelessness
    field :max_output_tokens, :integer, default: nil
    # omit metadata because chat_open_ai also omits it
    # omit parallel_tool_calls because chat_open_ai also omits it
    # omit previous_response_id becasue langchain assumes statelessness
    field :reasoning, :map, default: nil
    # omit service_tier because chat_open_ai also omits it
    # omit store, but set it explicitly to false later to keep statelessness. the API will default true unless we set it
    field :stream, :boolean, default: false
    field :temperature, :float, default: 1.0
    field :json_response, :boolean, default: false
    field :json_schema, :map, default: nil
    field :json_schema_name, :string, default: nil

    # This can be a string or object. We will need to allow ["none", "auto", "required", "file_search", "web_search_preview", and "computer_use_preview"] and take any other string and turn it to %{name: value, type: "function"}
    field :tool_choice, :any, default: nil, virtual: true
    field :top_p, :float, default: 1.0
    field :truncation, :string
    field :user, :string

    field :callbacks, {:array, :map}, default: []
    field :verbose_api, :boolean, default: false
  end

  @type t :: %ChatOpenAIResponses{}

  # Omits callbacks. Otherwise identical to above.
  @create_fields [
    :receive_timeout,
    :api_key,
    :endpoint,
    :model,
    :include,
    :max_output_tokens,
    :reasoning,
    :stream,
    :temperature,
    :json_response,
    :json_schema,
    :json_schema_name,
    :tool_choice,
    :top_p,
    :truncation,
    :user,
    :verbose_api
  ]
  @required_fields [:endpoint, :model]

  @spec get_api_key(t()) :: String.t()
  defp get_api_key(%ChatOpenAIResponses{api_key: api_key}) do
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
    %ChatOpenAIResponses{}
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
    |> validate_number(:top_p, greater_than_or_equal_to: 0, less_than_or_equal_to: 1)
    |> validate_number(:receive_timeout, greater_than_or_equal_to: 0)
  end

  @doc """
  Return the params formatted for an API request.
  """
  @spec for_api(t | Message.t() | Function.t(), message :: [map()], ChatModel.tools()) :: %{
          atom() => any()
        }
  def for_api(%ChatOpenAIResponses{} = openai, messages, tools) do
    %{
      model: openai.model,
      temperature: openai.temperature,
      top_p: openai.top_p,
      stream: openai.stream,
      input:
        messages
        |> Enum.reduce([], fn m, acc ->
          case for_api(openai, m) do
            %{} = data ->
              [data | acc]

            data when is_list(data) ->
              Enum.reverse(data) ++ acc
          end
        end)
        |> Enum.reverse(),
      user: openai.user
    }
    |> Utils.conditionally_add_to_map(:include, openai.include)
    |> Utils.conditionally_add_to_map(:max_output_tokens, openai.max_output_tokens)
    |> Utils.conditionally_add_to_map(:reasoning, openai.reasoning)
    |> Utils.conditionally_add_to_map(:text, set_text_format(openai))
    |> Utils.conditionally_add_to_map(:tool_choice, get_tool_choice(openai))
    |> Utils.conditionally_add_to_map(:truncation, openai.truncation)
    |> Utils.conditionally_add_to_map(:tools, get_tools_for_api(openai, tools))
  end

  defp get_tools_for_api(%ChatOpenAIResponses{} = _model, nil), do: []

  defp get_tools_for_api(%ChatOpenAIResponses{} = model, tools) do
    Enum.map(tools, fn
      %Function{} = function ->
        for_api(model, function)

      %NativeTool{} = tool ->
        for_api(model, tool)
    end)
  end

  defp set_text_format(%ChatOpenAIResponses{
         json_response: true,
         json_schema: json_schema,
         json_schema_name: json_schema_name
       })
       when not is_nil(json_schema) and not is_nil(json_schema_name) do
    %{
      "name" => json_schema_name,
      "schema" => json_schema,
      "type" => "json_schema"
    }
  end

  defp set_text_format(%ChatOpenAIResponses{json_response: true}) do
    %{"type" => "json_object"}
  end

  defp set_text_format(%ChatOpenAIResponses{json_response: false}) do
    # NOTE: The default handling when unspecified is `%{"type" => "text"}`
    # This returns a `nil` which has the same effect.
    nil
  end

  defp get_tool_choice(%ChatOpenAIResponses{tool_choice: choice})
       when choice in ["none", "auto", "required"],
       do: choice

  defp get_tool_choice(%ChatOpenAIResponses{tool_choice: choice})
       when choice in ["file_search", "web_search_preview", "computer_use_preview"],
       do: %{"type" => choice}

  defp get_tool_choice(%ChatOpenAIResponses{tool_choice: choice})
       when is_binary(choice) and byte_size(choice) > 0,
       do: %{"type" => "function", "name" => choice}

  defp get_tool_choice(%ChatOpenAIResponses{}), do: nil

  @spec for_api(
          struct(),
          Message.t()
          | PromptTemplate.t()
          | ToolCall.t()
          | ToolResult.t()
          | ContentPart.t()
          | Function.t()
          | NativeTool.t()
        ) ::
          %{String.t() => any()} | [%{String.t() => any()}]

  # Function support
  def for_api(%ChatOpenAIResponses{} = _model, %Function{} = fun) do
    %{
      "name" => fun.name,
      "parameters" => get_parameters(fun),
      "type" => "function"
    }
    |> Utils.conditionally_add_to_map("description", fun.description)
    |> Utils.conditionally_add_to_map("strict", fun.strict)
  end

  def for_api(
        %ChatOpenAIResponses{} = _model,
        %NativeTool{name: name, configuration: config}
      ) do
    Map.put_new(config, :type, name)
  end

  def for_api(%ChatOpenAIResponses{} = model, %Message{role: :system, content: content})
      when is_list(content) do
    %{
      "role" => "system",
      "type" => "message",
      "content" => content_parts_for_api(model, content)
    }
  end

  def for_api(%ChatOpenAIResponses{} = model, %Message{role: :user, content: content})
      when is_list(content) do
    %{
      "role" => "user",
      "type" => "message",
      "content" => content_parts_for_api(model, content)
    }
  end

  def for_api(
        %ChatOpenAIResponses{} = model,
        %Message{role: :tool, tool_results: tool_results}
      )
      when is_list(tool_results) do
    Enum.map(tool_results, &for_api(model, &1))
  end

  # Native tool calls (such as web_search_call) need to get plucked
  # out of the content parts and become their own input items.
  def for_api(
        %ChatOpenAIResponses{} = model,
        %Message{role: :assistant, content: content} = msg
      )
      when is_list(content) do
    native_tool_calls_for_api(model, content) ++
      [
        %{
          "role" => "user",
          "type" => "message",
          "content" => content_parts_for_api(model, content)
        }
      ] ++
      Enum.map(msg.tool_calls || [], &for_api(model, &1))
  end

  def for_api(
        %ChatOpenAIResponses{} = model,
        %Message{role: :assistant, tool_calls: tool_calls}
      )
      when is_list(tool_calls) do
    Enum.map(tool_calls, &for_api(model, &1))
  end

  def for_api(%ChatOpenAIResponses{} = _model, %ToolResult{type: :function} = result) do
    # a ToolResult becomes a stand-alone %Message{role: :tool} response.
    %{
      "call_id" => result.tool_call_id,
      "output" => result.content,
      "type" => "function_call_output"
    }
  end

  # ToolCall support
  def for_api(%ChatOpenAIResponses{} = _model, %ToolCall{type: :function} = fun) do
    %{
      "arguments" => Jason.encode!(fun.arguments),
      "call_id" => fun.call_id,
      "name" => fun.name,
      "type" => "function_call"
    }
    |> Utils.conditionally_add_to_map("status", fun.status)
  end

  def for_api(%ChatOpenAIResponses{} = _model, %PromptTemplate{} = _template) do
    raise LangChainError, "PromptTemplates must be converted to messages."
  end

  def native_tool_calls_for_api(%ChatOpenAIResponses{} = model, content_parts)
      when is_list(content_parts) do
    Enum.map(content_parts, &native_tool_call_for_api(model, &1))
    |> Enum.reject(&is_nil/1)
  end

  @spec native_tool_call_for_api(any(), any()) ::
          nil | %{id: any(), status: any(), type: <<_::120>>}
  def native_tool_call_for_api(%ChatOpenAIResponses{} = _model, %ContentPart{
        type: :unsupported,
        options: %{type: "web_search_call"} = opts
      }) do
    %{id: opts.id, type: "web_search_call", status: opts.status}
  end

  def native_tool_call_for_api(_, _), do: nil

  @doc """
  Convert a list of ContentParts to the expected map of data for the OpenAI API.
  """
  def content_parts_for_api(%ChatOpenAIResponses{} = model, content_parts)
      when is_list(content_parts) do
    Enum.map(content_parts, &content_part_for_api(model, &1))
    |> Enum.reject(&is_nil/1)
  end

  @doc """
  Convert a ContentPart to the expected map of data for the OpenAI API.
  """
  def content_part_for_api(%ChatOpenAIResponses{} = _model, %ContentPart{type: :text} = part) do
    %{"type" => "input_text", "text" => part.content}
  end

  def content_part_for_api(
        %ChatOpenAIResponses{} = _model,
        %ContentPart{type: :file, options: opts} = part
      ) do
    case Keyword.get(opts, :type, :base64) do
      :file_id ->
        %{
          "type" => "input_file",
          "file_id" => part.content
        }

      :base64 ->
        %{
          "type" => "input_file",
          "filename" => Keyword.get(opts, :filename, "file.pdf"),
          "file_data" => "data:application/pdf;base64," <> part.content
        }
    end
  end

  def content_part_for_api(%ChatOpenAIResponses{} = _model, %ContentPart{type: image} = part)
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
          Logger.error(message)
          raise LangChainError, message
      end

    detail_option = Keyword.get(part.options, :detail, nil)
    file_id = Keyword.get(part.options, :file_id, nil)

    %{
      "type" => "input_image",
      "image_url" => media_prefix <> part.content
    }
    |> Utils.conditionally_add_to_map("detail", detail_option)
    |> Utils.conditionally_add_to_map("file_id", file_id)
  end

  # Ignore unknown, unsupported content parts
  def content_part_for_api(%ChatOpenAIResponses{} = _model, %ContentPart{type: :unsupported}),
    do: nil

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

  @impl ChatModel
  def call(openai, prompt, tools \\ [])

  def call(%ChatOpenAIResponses{} = openai, prompt, tools) when is_binary(prompt) do
    messages = [
      Message.new_system!(),
      Message.new_user!(prompt)
    ]

    call(openai, messages, tools)
  end

  def call(%ChatOpenAIResponses{} = openai, messages, tools) when is_list(messages) do
    metadata = %{
      model: openai.model,
      message_count: length(messages),
      tools_count: length(tools)
    }

    LangChain.Telemetry.span([:langchain, :llm, :call], metadata, fn ->
      try do
        # Track the prompt being sent
        LangChain.Telemetry.llm_prompt(
          %{system_time: System.system_time()},
          %{model: openai.model, messages: messages}
        )

        # make base api request and perform high-level success/failure checks
        case do_api_request(openai, messages, tools) do
          {:error, reason} ->
            {:error, reason}

          parsed_data ->
            # Track the response being received
            LangChain.Telemetry.llm_response(
              %{system_time: System.system_time()},
              %{model: openai.model, response: parsed_data}
            )

            {:ok, parsed_data}
        end
      rescue
        err in LangChainError ->
          {:error, err}
      end
    end)
  end

  @spec do_api_request(t(), [Message.t()], ChatModel.tools(), integer()) ::
          list() | struct() | {:error, LangChainError.t()}
  def do_api_request(openai, messages, tools, retry_count \\ 3)

  def do_api_request(_openai, _messages, _tools, 0) do
    raise LangChainError, "Retries exceeded. Connection failed."
  end

  def do_api_request(
        %ChatOpenAIResponses{stream: false} = openai,
        messages,
        tools,
        retry_count
      ) do
    raw_data = for_api(openai, messages, tools)

    if openai.verbose_api do
      IO.inspect(raw_data, label: "RAW DATA BEING SUBMITTED")
    end

    req =
      Req.new(
        url: openai.endpoint,
        json: raw_data,
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
        if openai.verbose_api do
          IO.inspect(response, label: "RAW REQ RESPONSE")
        end

        Callbacks.fire(openai.callbacks, :on_llm_ratelimit_info, [
          get_ratelimit_info(response.headers)
        ])

        case do_process_response(openai, data) do
          {:error, %LangChainError{} = reason} ->
            {:error, reason}

          result ->
            Callbacks.fire(openai.callbacks, :on_llm_new_message, [result])

            # Track non-streaming response completion
            LangChain.Telemetry.emit_event(
              [:langchain, :llm, :response, :non_streaming],
              %{system_time: System.system_time()},
              %{
                model: openai.model,
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
        do_api_request(openai, messages, tools, retry_count - 1)

      other ->
        Logger.error("Unexpected and unhandled API response! #{inspect(other)}")
        other
    end
  end

  def do_api_request(
        %ChatOpenAIResponses{stream: true} = openai,
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

  # Complete Response with output lists
  def do_process_response(_model, %{"status" => "completed", "output" => content_items})
      when is_list(content_items) do
    {content_parts, tool_calls} = content_items_to_content_parts_and_tool_calls(content_items)

    Message.new!(%{
      content: content_parts,
      status: :complete,
      role: :assistant,
      tool_calls: tool_calls
    })
  end

  defp content_items_to_content_parts_and_tool_calls(content_items) do
    Enum.reduce(content_items, {[], []}, fn content_item, {content_parts, tool_calls} ->
      case content_item_to_content_part_or_tool_call(content_item) do
        %ContentPart{} = cp ->
          {content_parts ++ [cp], tool_calls}

        %ToolCall{} = tc ->
          {content_parts, tool_calls ++ [tc]}
      end
    end)
  end

  defp content_item_to_content_part_or_tool_call(%{
         "type" => "message",
         "content" => message_contents
       }) do
    text =
      message_contents
      |> Enum.map(fn
        %{"type" => "output_text", "text" => text} -> text
        %{"type" => "refusal", "refusal" => refusal} -> refusal
      end)
      |> Enum.join(" ")

    ContentPart.text!(text)
  end

  defp content_item_to_content_part_or_tool_call(%{
         "type" => "function_call",
         "call_id" => call_id,
         "name" => name,
         "arguments" => args
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

  # The Responses API returns web_search_call as a sibling of assistant messages, as
  # in:
  # %{
  #   ...,
  #   "output" => [
  #     %{"type" => "web_search_call", ...},
  #     %{"type" => "message", "content" => [...content_parts...]}
  #   ]
  # }
  # however we embed it within the message as an unsupported content part to maintain the
  # idiom of returning a single %Message{} per API call.
  defp content_item_to_content_part_or_tool_call(%{
         "type" => "web_search_call",
         "id" => web_search_call_id,
         "status" => "completed"
       }) do
    case ContentPart.new(%{
           type: :unsupported,
           options: %{
             id: web_search_call_id,
             status: "completed",
             type: "web_search_call"
           },
           call_id: web_search_call_id
         }) do
      {:ok, %ContentPart{} = call} ->
        call

      {:error, %Ecto.Changeset{} = changeset} ->
        reason = Utils.changeset_error_to_string(changeset)
        Logger.error("Failed to process web_search_call. Reason: #{reason}")
        {:error, LangChainError.exception(changeset)}
    end
  end

  # # Full message with tool call
  # def do_process_response(
  #       model,
  #       %{"finish_reason" => finish_reason, "message" => %{"tool_calls" => calls} = message} =
  #         data
  #     )
  #     when finish_reason in ["tool_calls", "stop"] do
  #   case Message.new(%{
  #          "role" => "assistant",
  #          "content" => message["content"],
  #          "complete" => true,
  #          "index" => data["index"],
  #          "tool_calls" => Enum.map(calls, &do_process_response(model, &1))
  #        }) do
  #     {:ok, message} ->
  #       message

  #     {:error, %Ecto.Changeset{} = changeset} ->
  #       {:error, LangChainError.exception(changeset)}
  #   end
  # end

  # # Delta message tool call
  # def do_process_response(
  #       model,
  #       %{"delta" => delta_body, "finish_reason" => finish, "index" => index} = _msg
  #     ) do
  #   status = finish_reason_to_status(finish)

  #   tool_calls =
  #     case delta_body do
  #       %{"tool_calls" => tools_data} when is_list(tools_data) ->
  #         Enum.map(tools_data, &do_process_response(model, &1))

  #       _other ->
  #         nil
  #     end

  #   # more explicitly interpret the role. We treat a "function_call" as a a role
  #   # while OpenAI addresses it as an "assistant". Technically, they are correct
  #   # that the assistant is issuing the function_call.
  #   role =
  #     case delta_body do
  #       %{"role" => role} -> role
  #       _other -> "unknown"
  #     end

  #   data =
  #     delta_body
  #     |> Map.put("role", role)
  #     |> Map.put("index", index)
  #     |> Map.put("status", status)
  #     |> Map.put("tool_calls", tool_calls)

  #   case MessageDelta.new(data) do
  #     {:ok, message} ->
  #       message

  #     {:error, %Ecto.Changeset{} = changeset} ->
  #       {:error, LangChainError.exception(changeset)}
  #   end
  # end

  # # Tool call as part of a delta message
  # def do_process_response(_model, %{"function" => func_body, "index" => index} = tool_call) do
  #   # function parts may or may not be present on any given delta chunk
  #   case ToolCall.new(%{
  #          status: :incomplete,
  #          type: :function,
  #          call_id: tool_call["id"],
  #          name: Map.get(func_body, "name", nil),
  #          arguments: Map.get(func_body, "arguments", nil),
  #          index: index
  #        }) do
  #     {:ok, %ToolCall{} = call} ->
  #       call

  #     {:error, %Ecto.Changeset{} = changeset} ->
  #       reason = Utils.changeset_error_to_string(changeset)
  #       Logger.error("Failed to process ToolCall for a function. Reason: #{reason}")
  #       {:error, LangChainError.exception(changeset)}
  #   end
  # end

  # # Tool call from a complete message
  # def do_process_response(_model, %{
  #       "function" => %{
  #         "arguments" => args,
  #         "name" => name
  #       },
  #       "id" => call_id,
  #       "type" => "function"
  #     }) do
  #   # No "index". It is a complete message.
  #   case ToolCall.new(%{
  #          type: :function,
  #          status: :complete,
  #          name: name,
  #          arguments: args,
  #          call_id: call_id
  #        }) do
  #     {:ok, %ToolCall{} = call} ->
  #       call

  #     {:error, %Ecto.Changeset{} = changeset} ->
  #       reason = Utils.changeset_error_to_string(changeset)
  #       Logger.error("Failed to process ToolCall for a function. Reason: #{reason}")
  #       {:error, LangChainError.exception(changeset)}
  #   end
  # end

  # def do_process_response(_model, %{
  #       "finish_reason" => finish_reason,
  #       "message" => message,
  #       "index" => index
  #     }) do
  #   status = finish_reason_to_status(finish_reason)

  #   case Message.new(Map.merge(message, %{"status" => status, "index" => index})) do
  #     {:ok, message} ->
  #       message

  #     {:error, %Ecto.Changeset{} = changeset} ->
  #       {:error, LangChainError.exception(changeset)}
  #   end
  # end

  # def do_process_response(_model, %{"error" => %{"message" => reason}}) do
  #   Logger.error("Received error from API: #{inspect(reason)}")
  #   {:error, LangChainError.exception(message: reason)}
  # end

  # def do_process_response(_model, {:error, %Jason.DecodeError{} = response}) do
  #   error_message = "Received invalid JSON: #{inspect(response)}"
  #   Logger.error(error_message)

  #   {:error,
  #    LangChainError.exception(type: "invalid_json", message: error_message, original: response)}
  # end

  # def do_process_response(_model, other) do
  #   Logger.error("Trying to process an unexpected response. #{inspect(other)}")
  #   {:error, LangChainError.exception(message: "Unexpected response")}
  # end

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
  def serialize_config(%ChatOpenAIResponses{} = model) do
    Utils.to_serializable_map(
      model,
      [
        :endpoint,
        :model,
        :temperature,
        :frequency_penalty,
        :reasoning_mode,
        :reasoning_effort,
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
    ChatOpenAIResponses.new(data)
  end
end
