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

      # Using a file ID (after uploading to OpenAI)
      Message.new_user!([
        ContentPart.text!("Describe this image:"),
        ContentPart.image!("file-1234", type: :file_id)
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
  alias LangChain.ChatModels.ReasoningOptions

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
    field :previous_response_id, :string, default: nil
    # Reasoning options for gpt-5 and o-series models
    embeds_one(:reasoning, ReasoningOptions)
    # omit service_tier because chat_open_ai also omits it
    field :store, :boolean, default: false
    field :stream, :boolean, default: false
    field :temperature, :float, default: nil
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

    # Req options to merge into the request.
    # Refer to `https://hexdocs.pm/req/Req.html#new/1-options` for
    # `Req.new` supported set of options.
    field :req_config, :map, default: %{}
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
    :previous_response_id,
    :store,
    :stream,
    :temperature,
    :json_response,
    :json_schema,
    :json_schema_name,
    :tool_choice,
    :top_p,
    :truncation,
    :user,
    :verbose_api,
    :req_config
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
    |> cast_embed(:reasoning)
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
      stream: openai.stream,
      store: if(openai.previous_response_id, do: true, else: openai.store),
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
        |> Enum.reverse()
    }
    |> Utils.conditionally_add_to_map(:include, openai.include)
    |> Utils.conditionally_add_to_map(:max_output_tokens, openai.max_output_tokens)
    |> Utils.conditionally_add_to_map(:previous_response_id, openai.previous_response_id)
    |> Utils.conditionally_add_to_map(:reasoning, ReasoningOptions.to_api_map(openai.reasoning))
    |> Utils.conditionally_add_to_map(:text, set_text_format(openai))
    |> Utils.conditionally_add_to_map(:tool_choice, get_tool_choice(openai))
    |> Utils.conditionally_add_to_map(:truncation, openai.truncation)
    |> Utils.conditionally_add_to_map(:tools, get_tools_for_api(openai, tools))
    |> Utils.conditionally_add_to_map(:user, openai.user)
    |> Utils.conditionally_add_to_map(:temperature, openai.temperature)
    |> maybe_add_top_p(openai)
  end

  # gpt-5.2 and newer do not support the top_p parameter.
  # Earlier models (gpt-4.x, gpt-5.0, gpt-5.1) accept top_p.
  defp maybe_add_top_p(map, %ChatOpenAIResponses{model: model, top_p: top_p}) do
    if supports_top_p?(model) do
      Utils.conditionally_add_to_map(map, :top_p, top_p)
    else
      map
    end
  end

  @doc false
  @spec supports_top_p?(String.t()) :: boolean()
  def supports_top_p?(model) when is_binary(model) do
    # Match models known to support top_p. This set is fixed and won't grow.
    cond do
      String.starts_with?(model, "gpt-4") -> true
      String.starts_with?(model, "gpt-5.0") -> true
      String.starts_with?(model, "gpt-5.1") -> true
      true -> false
    end
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
      "format" => %{
        "type" => "json_schema",
        "name" => json_schema_name,
        "schema" => json_schema,
        "strict" => true
      }
    }
  end

  defp set_text_format(%ChatOpenAIResponses{json_response: true}) do
    %{"format" => %{"type" => "json_object"}}
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
    [%ContentPart{type: :text, content: output, options: []}] = result.content

    %{
      "call_id" => result.tool_call_id,
      "output" => output,
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
    |> Utils.conditionally_add_to_map("status", "completed")
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
        %ContentPart{type: :file_url} = part
      ) do
    %{
      "type" => "input_file",
      "file_url" => part.content
    }
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
    output =
      if Keyword.get(part.options, :type) == :file_id do
        %{"type" => "input_image", "file_id" => part.content}
      else
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

        %{
          "type" => "input_image",
          "image_url" => media_prefix <> part.content
        }
      end

    detail_option = Keyword.get(part.options, :detail, nil)

    Utils.conditionally_add_to_map(output, "detail", detail_option)
  end

  # Thinking content parts are output-only and should be omitted when sending to the API
  def content_part_for_api(%ChatOpenAIResponses{} = _model, %ContentPart{type: :thinking}),
    do: nil

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
    |> Req.merge(openai.req_config |> Keyword.new())
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
    |> Req.merge(openai.req_config |> Keyword.new())
    |> Req.post(
      into: Utils.handle_stream_fn(openai, &decode_stream/1, &do_process_response(openai, &1))
    )
    |> case do
      {:ok, %Req.Response{body: data} = response} ->
        Callbacks.fire(openai.callbacks, :on_llm_ratelimit_info, [
          get_ratelimit_info(response.headers)
        ])

        List.flatten(data)

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

  # The Responses API streams events in the form:
  # event: <event_type>\ndata: { ...json... }
  # We want to extract each pair and parse the JSON from the `data:` line.

  # A list of all events can be found here: https://platform.openai.com/docs/api-reference/responses-streaming

  # Unlike the Chat Completions API, we do not get a [DONE] token at the end of the stream.

  @spec decode_stream({String.t(), String.t()}) :: {%{String.t() => any()}}
  def decode_stream({raw_data, buffer}, done \\ []) do
    raw_data
    |> String.split(~r/event: /)
    |> Enum.map(fn
      <<"event: ", rest::binary>> -> rest
      other -> other
    end)
    |> Enum.flat_map(fn chunk ->
      case String.split(chunk, ~r/\ndata: /, parts: 2) do
        [_event, json] -> [json]
        [json_only] -> [json_only]
        _ -> []
      end
    end)
    |> Enum.reduce({done, buffer}, fn str, {done, incomplete} = acc ->
      str
      |> String.trim()
      |> case do
        "" ->
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
  def do_process_response(
        _model,
        %{"status" => "completed", "output" => content_items} = response
      )
      when is_list(content_items) do
    {content_parts, tool_calls} = content_items_to_content_parts_and_tool_calls(content_items)

    metadata =
      case get_token_usage(response) do
        nil -> %{}
        %TokenUsage{} = usage -> %{usage: usage}
      end
      |> maybe_add_response_id(response)

    Message.new!(%{
      content: content_parts,
      status: :complete,
      role: :assistant,
      tool_calls: tool_calls,
      metadata: metadata
    })
  end

  # Handle streaming events

  # Streamed events are returned as a raw list of events
  # Even if there is only one event, it is returned within a list.
  # Open to feedback this should get moved up and down the pattern-matching
  # priority here.

  def do_process_response(model, list) when is_list(list) do
    Enum.map(list, &do_process_response(model, &1))
  end

  # Deltas arrive in the following shape:
  # %{
  #   "content_index" => 0,
  #   "delta" => "Hello",
  #   "item_id" => "msg_1234567890",
  #   "output_index" => 0,
  #   "sequence_number" => 4,
  #   "type" => "response.output_text.delta"
  # }
  def do_process_response(_model, %{
        "type" => "response.reasoning.delta",
        "output_index" => output_index,
        "delta" => delta_text
      }) do
    data = %{
      content: ContentPart.new!(%{type: :thinking, content: delta_text}),
      status: :incomplete,
      role: :assistant,
      index: output_index
    }

    case MessageDelta.new(data) do
      {:ok, message} ->
        message

      {:error, %Ecto.Changeset{} = changeset} ->
        {:error, LangChainError.exception(changeset)}
    end
  end

  def do_process_response(
        _model,
        %{
          "type" => "response.output_text.delta",
          "output_index" => output_index,
          "delta" => delta_text
        }
      ) do
    data = %{
      content: delta_text,
      # Will need to be updated to :complete when the response is complete
      status: :incomplete,
      role: :assistant,
      index: output_index
    }

    case MessageDelta.new(data) do
      {:ok, message} ->
        message

      {:error, %Ecto.Changeset{} = changeset} ->
        {:error, LangChainError.exception(changeset)}
    end
  end

  def do_process_response(_model, %{"type" => "response.output_text.delta", "delta" => delta_text}) do
    data = %{
      content: delta_text,
      # Will need to be updated to :complete when the response is complete
      status: :incomplete,
      role: :assistant
    }

    case MessageDelta.new(data) do
      {:ok, message} ->
        message

      {:error, %Ecto.Changeset{} = changeset} ->
        {:error, LangChainError.exception(changeset)}
    end
  end

  # Open question: is it possible we get multiples of `response.output_text.done`?
  # It precedes `response.content_part.done` and `response.output_item.done`
  # and theoretically we could get multiple text content_parts and output_items.

  # I believe, semantically, these deltas are "outside" the output item and content part
  # and can be treated as a "global" stream of deltas -- meaning "done" is truly
  # "done" -- but that remains unconfirmed.
  def do_process_response(_model, %{"type" => "response.output_text.done"}) do
    data = %{
      content: "",
      status: :complete,
      role: :assistant
    }

    case MessageDelta.new(data) do
      {:ok, message} ->
        message

      {:error, %Ecto.Changeset{} = changeset} ->
        {:error, LangChainError.exception(changeset)}
    end
  end

  # This is the first event we get for a reasoning/thinking block.
  # It is followed by a series of `response.reasoning.delta` events.
  # Finally, it is followed by a `response.output_item.done` event.
  def do_process_response(_model, %{
        "type" => "response.output_item.added",
        "output_index" => output_index,
        "item" => %{
          "type" => "reasoning",
          "id" => _reasoning_id
        }
      }) do
    data = %{
      content: ContentPart.new!(%{type: :thinking, content: ""}),
      status: :incomplete,
      role: :assistant,
      index: output_index
    }

    case MessageDelta.new(data) do
      {:ok, delta} ->
        delta

      {:error, %Ecto.Changeset{} = changeset} ->
        {:error, LangChainError.exception(changeset)}
    end
  end

  # This is the first event we get for a function call.
  # It is followed by a series of `response.function_call_arguments.delta` events.
  # It is followed by a `response.function_call_arguments.done` event. (which we skip)
  # Finally, it is followed by a `response.output_item.done` event.
  def do_process_response(_model, %{
        "type" => "response.output_item.added",
        "output_index" => output_index,
        "item" => %{
          "type" => "function_call",
          "call_id" => call_id,
          "name" => name,
          "arguments" => args
        }
      }) do
    data = %{
      status: :incomplete,
      type: :function,
      call_id: call_id,
      name: name,
      arguments: args,
      index: output_index
    }

    with {:ok, %ToolCall{} = call} <- ToolCall.new(data),
         {:ok, delta} <-
           MessageDelta.new(%{
             content: "",
             status: :incomplete,
             role: :assistant,
             tool_calls: [call]
           }) do
      delta
    else
      {:error, %Ecto.Changeset{} = changeset} ->
        {:error, LangChainError.exception(changeset)}
    end
  end

  def do_process_response(_model, %{
        "type" => "response.function_call_arguments.delta",
        "output_index" => output_index,
        "delta" => delta_text
      }) do
    data = %{
      arguments: delta_text,
      index: output_index
    }

    with {:ok, call} <- ToolCall.new(data),
         {:ok, message} <-
           MessageDelta.new(%{
             content: "",
             status: :incomplete,
             role: :assistant,
             tool_calls: [call]
           }) do
      message
    else
      {:error, %Ecto.Changeset{} = changeset} ->
        {:error, LangChainError.exception(changeset)}
    end
  end

  def do_process_response(_model, %{
        "type" => "response.output_item.done",
        "output_index" => output_index,
        "item" => %{"type" => "reasoning"}
      }) do
    data = %{
      content: ContentPart.new!(%{type: :thinking, content: ""}),
      status: :complete,
      role: :assistant,
      index: output_index
    }

    case MessageDelta.new(data) do
      {:ok, delta} ->
        delta

      {:error, %Ecto.Changeset{} = changeset} ->
        {:error, LangChainError.exception(changeset)}
    end
  end

  def do_process_response(_model, %{
        "type" => "response.output_item.done",
        "output_index" => output_index,
        "item" => %{"type" => "function_call"} = item
      }) do
    data = %{
      status: :complete,
      index: output_index,
      call_id: item["call_id"],
      arguments: item["arguments"],
      name: item["name"]
    }

    with {:ok, call} <- ToolCall.new(data),
         {:ok, message} <-
           MessageDelta.new(%{
             status: :complete,
             role: :assistant,
             tool_calls: [call]
           }) do
      message
    else
      {:error, %Ecto.Changeset{} = changeset} ->
        {:error, LangChainError.exception(changeset)}
    end
  end

  def do_process_response(_model, %{
        "type" => "response.completed",
        "response" => response
      }) do
    usage = get_token_usage(response)
    metadata = %{usage: usage} |> maybe_add_response_id(response)

    data = %{
      content: "",
      status: :complete,
      role: :assistant,
      metadata: metadata
    }

    case MessageDelta.new(data) do
      {:ok, message} ->
        message

      {:error, %Ecto.Changeset{} = changeset} ->
        {:error, LangChainError.exception(changeset)}
    end
  end

  # Streaming events explicitly skipped

  # Items we should come back and implement:
  # - refusals
  # - function_calls
  # - error

  @reasoning_summary_events [
    "response.reasoning_summary_text.done",
    "response.reasoning_summary_part.added",
    "response.reasoning_summary_part.done",
    "response.reasoning_summary.done"
  ]

  @skippable_streaming_events [
    "response.created",
    "response.in_progress",
    "response.incomplete",
    "response.output_item.added",
    "response.output_item.done",
    "response.content_part.added",
    "response.content_part.done",
    "response.refusal.delta",
    "response.refusal.done",
    "response.function_call_arguments.done",
    "response.file_search_call.in_progress",
    "response.file_search_call.searching",
    "response.file_search_call.completed",
    "response.web_search_call.in_progress",
    "response.web_search_call.searching",
    "response.web_search_call.completed",
    "response.image_generation_call.completed",
    "response.image_generation_call.generating",
    "response.image_generation_call.in_progress",
    "response.image_generation_call.partial_image",
    "response.mcp_call.arguments.delta",
    "response.mcp_call.arguments.done",
    "response.mcp_call.completed",
    "response.mcp_call.failed",
    "response.mcp_call.in_progress",
    "response.output_text.annotation.added",
    "response.queued",
    "response.reasoning.delta",
    "error"
  ]

  # Handle reasoning summary delta events - fire callback and return :skip
  def do_process_response(model, %{
        "type" => "response.reasoning_summary_text.delta",
        "delta" => delta
      }) do
    Logger.debug("[LANGCHAIN] Reasoning text delta received")
    Callbacks.fire(model.callbacks, :on_llm_reasoning_delta, [delta])
    :skip
  end

  def do_process_response(model, %{
        "type" => "response.reasoning_summary.delta",
        "delta" => delta
      }) do
    Logger.debug("[LANGCHAIN] Reasoning summary delta received")
    Callbacks.fire(model.callbacks, :on_llm_reasoning_delta, [delta])
    :skip
  end

  def do_process_response(_model, %{"type" => event} = _data)
      when event in @reasoning_summary_events do
    Logger.debug("[LANGCHAIN] Reasoning event: #{event}")
    :skip
  end

  def do_process_response(_model, %{"type" => event})
      when event in @skippable_streaming_events do
    Logger.debug("[LANGCHAIN] Skipping streaming event: #{event}")
    :skip
  end

  def do_process_response(_model, %{"error" => %{"message" => reason}}) do
    Logger.error("Received error from API: #{inspect(reason)}")
    {:error, LangChainError.exception(message: reason)}
  end

  # Handle failed response status (streaming event with type "response.failed")
  def do_process_response(_model, %{
        "type" => "response.failed",
        "response" => %{"status" => "failed"} = response
      }) do
    build_failed_response_error(response)
  end

  # Handle failed response status (non-streaming / full response object)
  def do_process_response(_model, %{"response" => %{"status" => "failed"} = response}) do
    build_failed_response_error(response)
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

  # Extracts error details from a failed OpenAI response and builds an error tuple.
  # Handles various error formats defensively:
  # - %{"error" => %{"code" => "...", "message" => "..."}}
  # - %{"error" => "string message"}
  # - %{} (no error details)
  @spec build_failed_response_error(map()) :: {:error, LangChainError.t()}
  defp build_failed_response_error(response) do
    {error_type, error_message} = extract_error_details(response)

    Logger.error("OpenAI Responses API request failed: #{error_message}")

    {:error,
     LangChainError.exception(
       type: error_type,
       message: "OpenAI request failed: #{error_message}",
       original: response
     )}
  end

  @spec extract_error_details(map()) :: {String.t(), String.t()}
  defp extract_error_details(response) do
    case Map.get(response, "error") do
      %{} = error_info ->
        message = Map.get(error_info, "message", "Request failed")
        code = Map.get(error_info, "code", "api_error")
        {code, message}

      message when is_binary(message) ->
        {"api_error", message}

      _ ->
        {"api_error", "Request failed"}
    end
  end

  defp get_token_usage(%{"usage" => usage} = _response_body) when is_map(usage) do
    # extract out the reported response token usage
    #
    # https://platform.openai.com/docs/api-reference/responses_streaming/response/completed#responses_streaming/response/completed-response-usage
    TokenUsage.new!(%{
      input: Map.get(usage, "input_tokens"),
      output: Map.get(usage, "output_tokens"),
      raw: usage
    })
  end

  defp get_token_usage(_response_body), do: nil

  defp maybe_add_response_id(metadata, %{"id" => id}) when is_binary(id) do
    Map.put(metadata, :response_id, id)
  end

  defp maybe_add_response_id(metadata, _response), do: metadata

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

  # Handle reasoning output from gpt-5 and o-series models
  # We can either ignore it or store it as metadata
  defp content_item_to_content_part_or_tool_call(%{
         "type" => "reasoning",
         "id" => reasoning_id,
         "summary" => summary
       }) do
    # Store reasoning as an unsupported content part for now
    # This preserves the information without breaking the flow
    case ContentPart.new(%{
           type: :unsupported,
           options: %{
             id: reasoning_id,
             summary: summary,
             type: "reasoning"
           }
         }) do
      {:ok, %ContentPart{} = part} ->
        part

      {:error, %Ecto.Changeset{} = changeset} ->
        reason = Utils.changeset_error_to_string(changeset)
        Logger.warning("Failed to process reasoning output. Reason: #{reason}")
        # Return a minimal content part to avoid breaking the flow
        ContentPart.text!("")
    end
  end

  defp content_item_to_content_part_or_tool_call(
         %{
           "type" => "file_search_call"
         } = part
       ) do
    # Store reasoning as an unsupported content part for now
    # This preserves the information without breaking the flow
    case ContentPart.new(%{
           type: :unsupported,
           options: %{
             id: part["id"],
             type: "file_search_call",
             queries: part["queries"],
             results: part["results"]
           }
         }) do
      {:ok, %ContentPart{} = part} ->
        part

      {:error, %Ecto.Changeset{} = changeset} ->
        reason = Utils.changeset_error_to_string(changeset)
        Logger.warning("Failed to process file_search output. Reason: #{reason}")
        # Return a minimal content part to avoid breaking the flow
        ContentPart.text!("")
    end
  end

  # Catch-all for unknown content item types
  defp content_item_to_content_part_or_tool_call(%{"type" => type} = item) do
    Logger.warning("Unknown content item type: #{type}. Item: #{inspect(item)}")
    # Return empty text content to avoid breaking the flow
    ContentPart.text!("")
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
        :reasoning,
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

  @doc """
  Determine if an error should be retried with a fallback model.
  Aligns with other providers.
  """
  @impl ChatModel
  @spec retry_on_fallback?(LangChainError.t()) :: boolean()
  def retry_on_fallback?(%LangChainError{type: "rate_limited"}), do: true
  def retry_on_fallback?(%LangChainError{type: "rate_limit_exceeded"}), do: true
  def retry_on_fallback?(%LangChainError{type: "timeout"}), do: true
  def retry_on_fallback?(%LangChainError{type: "too_many_requests"}), do: true
  def retry_on_fallback?(_), do: false
end
