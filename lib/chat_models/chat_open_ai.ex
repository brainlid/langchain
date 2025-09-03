defmodule LangChain.ChatModels.ChatOpenAI do
  @moduledoc """
  Represents the [OpenAI
  ChatModel](https://platform.openai.com/docs/api-reference/chat/create).

  Parses and validates inputs for making a requests from the OpenAI Chat API.

  Converts responses into more specialized `LangChain` data structures.

  - https://github.com/openai/openai-cookbook/blob/main/examples/How_to_call_functions_with_chat_models.ipynb

  ## ContentPart Types

  OpenAI supports several types of content parts that can be combined in a single message:

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

  ```elixir
  chat = ChatOpenAI.new!(%{stream: true, stream_options: %{include_usage: true}})
  ```

  The `TokenUsage` data is accumulated for `MessageDelta` structs and the final usage information will be on the `LangChain.Message`.

  NOTE: Of special note is that the `TokenUsage` information is returned once
  for all "choices" in the response. The `LangChain.TokenUsage` data is added to
  each message, but if your usage requests multiple choices, you will see the
  same usage information for each choice but it is duplicated and only one
  response is meaningful.

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

  ## Azure OpenAI Support

  To use `ChatOpenAI` with Microsoft's Azure hosted OpenAI models, the
  `endpoint` must be overridden and the API key needs to be provided in some
  way. The [MS Quickstart guide for REST
  access](https://learn.microsoft.com/en-us/azure/ai-services/openai/chatgpt-quickstart?tabs=command-line%2Cjavascript-keyless%2Ctypescript-keyless%2Cpython-new&pivots=rest-api)
  may be helpful.

  In order to use it, you must have an Azure account and from the console, a
  model must be deployed for your account. Use the Azure AI Foundry and Azure
  OpenAI Service to deploy the model you want to use. The entire URL is used as
  the `endpoint` and the provided `key` is used as the `api_key`.

  The following is an example of setting up `ChatOpenAI` for use with an Azure
  hosted model.

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

  The URL itself specifies the model to use and the `model` attribute is
  disregarded.

  A fake example URL for the endpoint value:

  `https://some-subdomain.cognitiveservices.azure.com/openai/deployments/gpt-4o-mini/chat/completions?api-version=2024-08-01-preview"`

  ## Reasoning Model Support

  OpenAI made some significant API changes with the introduction of their
  "reasoning" models. This includes the `o1` and `o1-mini` models.

  To enable this mode, set `:reasoning_mode` to `true`:

      model = ChatOpenAI.new!(%{reasoning_mode: true})

  Setting `reasoning_mode` to `true` does at least the two following things:

  - Set `:developer` as the `role` for system messages. The OpenAI documentation
    says API calls to `o1` and newer models must use the `role: :developer`
    instead of `role: :system` and errors if not set correctly.
  - The `:reasoning_effort` option included in LLM requests. This setting is
    only permitted on a reasoning model. The `:reasoning_effort` values support
    the "low", "medium" (default), and "high" options specified in the OpenAI
    documentation. This instructs the LLM on how much time, and tokens, should
    be spent on thinking through and reasoning about the request and the
    response.
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
    field :api_key, :string, redact: true

    # Organization ID for OpenAI. If not set, will use global org_id. Allows for usage
    # of a different organization ID per-call if desired.
    field :org_id, :string, redact: true

    # What sampling temperature to use, between 0 and 2. Higher values like 0.8
    # will make the output more random, while lower values like 0.2 will make it
    # more focused and deterministic.
    field :temperature, :float, default: 1.0
    # Number between -2.0 and 2.0. Positive values penalize new tokens based on
    # their existing frequency in the text so far, decreasing the model's
    # likelihood to repeat the same line verbatim.
    field :frequency_penalty, :float, default: nil

    # Used when working with a reasoning model like `o1` and newer. This setting
    # is required when working with those models as the API behavior needs to
    # change.
    field :reasoning_mode, :boolean, default: false

    # o1 models only
    #
    # Constrains effort on reasoning for reasoning models. Currently supported
    # values are `low`, `medium`, and `high`. Reducing reasoning effort can result in
    # faster responses and fewer tokens used on reasoning in a response.
    field :reasoning_effort, :string, default: "medium"

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

    # A list of maps for callback handlers (treated as internal)
    field :callbacks, {:array, :map}, default: []

    # Can send a string user_id to help ChatGPT detect abuse by users of the
    # application.
    # https://platform.openai.com/docs/guides/safety-best-practices/end-user-ids
    field :user, :string

    # For help with debugging. It outputs the RAW Req response received and the
    # RAW Elixir map being submitted to the API.
    field :verbose_api, :boolean, default: false
  end

  @type t :: %ChatOpenAI{}

  @create_fields [
    :endpoint,
    :model,
    :temperature,
    :frequency_penalty,
    :api_key,
    :org_id,
    :seed,
    :n,
    :stream,
    :reasoning_mode,
    :reasoning_effort,
    :receive_timeout,
    :json_response,
    :json_schema,
    :max_tokens,
    :stream_options,
    :user,
    :tool_choice,
    :verbose_api
  ]
  @required_fields [:endpoint, :model]

  @spec get_api_key(t()) :: String.t()
  defp get_api_key(%ChatOpenAI{api_key: api_key}) do
    # if no API key is set default to `""` which will raise a OpenAI API error
    api_key || Config.resolve(:openai_key, "")
  end

  @spec get_org_id(t()) :: String.t() | nil
  defp get_org_id(%ChatOpenAI{org_id: org_id}) when is_binary(org_id), do: org_id
  defp get_org_id(%ChatOpenAI{}), do: Config.resolve(:openai_org_id)

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
    {input_key, input_value} =
      if is_gpt5_model?(openai) do
        {
          :input,
          messages
          |> Enum.reduce([], fn m, acc ->
            case for_api(openai, m) do
              %{} = data -> [data | acc]
              data when is_list(data) -> Enum.reverse(data) ++ acc
            end
          end)
          |> Enum.reverse()
        }
      else
        {
          :messages,
          messages
          |> Enum.reduce([], fn m, acc ->
            case for_api(openai, m) do
              %{} = data -> [data | acc]
              data when is_list(data) -> Enum.reverse(data) ++ acc
            end
          end)
          |> Enum.reverse()
        }
      end

    base =
      %{
        model: openai.model,
        stream: openai.stream
      }
      |> Utils.conditionally_add_to_map(
        :temperature,
        if(is_gpt5_model?(openai), do: nil, else: openai.temperature)
      )
      |> Map.put(input_key, input_value)
      |> Utils.conditionally_add_to_map(:user, openai.user)
      |> Utils.conditionally_add_to_map(
        :frequency_penalty,
        if(is_gpt5_model?(openai), do: nil, else: openai.frequency_penalty)
      )
      |> Utils.conditionally_add_to_map(:response_format, set_response_format(openai))
      |> Utils.conditionally_add_to_map(
        :reasoning_effort,
        if(openai.reasoning_mode, do: openai.reasoning_effort, else: nil)
      )
      |> Utils.conditionally_add_to_map(:max_tokens, openai.max_tokens)
      # GPT-5 (Responses API) does not support 'seed' param
      |> Utils.conditionally_add_to_map(
        :seed,
        if(is_gpt5_model?(openai), do: nil, else: openai.seed)
      )
      |> Utils.conditionally_add_to_map(
        :stream_options,
        if(is_gpt5_model?(openai),
          do: nil,
          else: get_stream_options_for_api(openai.stream_options)
        )
      )
      |> Utils.conditionally_add_to_map(:tools, get_tools_for_api(openai, tools))
      |> Utils.conditionally_add_to_map(:tool_choice, get_tool_choice(openai))

    if is_gpt5_model?(openai) do
      base
    else
      Utils.conditionally_add_to_map(base, :n, openai.n)
    end
  end

  defp get_tools_for_api(%_{} = _model, nil), do: []

  defp get_tools_for_api(%ChatOpenAI{} = model, tools) do
    Enum.map(tools, fn
      %Function{} = function ->
        if is_gpt5_model?(model) do
          # GPT-5 (Responses API) expects flattened function declarations
          %{
            "type" => "function",
            "name" => function.name,
            "parameters" => get_parameters(function)
          }
          |> Utils.conditionally_add_to_map("description", function.description)
        else
          %{"type" => "function", "function" => for_api(model, function)}
        end
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
    # NOTE: The default handling when unspecified is `%{"type" => "text"}`
    #
    # For improved compatibility with other APIs like LMStudio, this returns a
    # `nil` which has the same effect.
    nil
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
  Convert a LangChain Message-based structure to the expected map of data for
  the OpenAI API. This happens within the context of the model configuration as
  well. The additional context is needed to correctly convert a role to either
  `:system` or `:developer`.

  NOTE: The `ChatOpenAI` model's functions are reused in other modules. For this
  reason, model is more generally defined as a struct.
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
    role = get_message_role(model, msg.role)

    %{
      "role" => role,
      "content" => content_parts_for_api(model, content, role)
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
        list when is_list(list) -> content_parts_for_api(model, list, :assistant)
        other -> other
      end

    %{
      "role" => :assistant,
      "content" => content
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

  def for_api(%_{} = model, %ToolResult{type: :function} = result) do
    # a ToolResult becomes a stand-alone %Message{role: :tool} response.
    %{
      "role" => :tool,
      "tool_call_id" => result.tool_call_id,
      "content" => content_parts_for_api(model, result.content)
    }
  end

  def for_api(%_{} = model, %Message{role: :tool, tool_results: tool_results} = _msg)
      when is_list(tool_results) do
    # ToolResults turn into a list of tool messages for OpenAI
    Enum.map(tool_results, fn result ->
      %{
        "role" => :tool,
        "tool_call_id" => result.tool_call_id,
        "content" => content_parts_for_api(model, result.content)
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

  # Handle ContentPart structures directly
  def for_api(%_{} = model, %ContentPart{} = part) do
    content_part_for_api(model, part)
  end

  @doc """
  Convert a list of ContentParts to the expected map of data for the OpenAI API.
  """
  def content_parts_for_api(%_{} = model, content_parts) when is_list(content_parts) do
    Enum.map(content_parts, &content_part_for_api(model, &1))
  end

  # With role context (GPT-5 needs assistant vs user mapping)
  def content_parts_for_api(%_{} = model, content_parts, role) when is_list(content_parts) do
    Enum.map(content_parts, fn
      %ContentPart{} = part -> content_part_for_api_with_role(model, part, role)
    end)
  end

  @doc """
  Convert a ContentPart to the expected map of data for the OpenAI API.
  """
  def content_part_for_api(%_{} = model, %ContentPart{type: :text} = part) do
    # For GPT-5 use "input_text"; for older models use "text"
    type = if is_gpt5_model?(model), do: "input_text", else: "text"
    %{"type" => type, "text" => part.content}
  end

  # (3-arity text variant is defined earlier at lines ~623)

  def content_part_for_api(%_{} = _model, %ContentPart{type: :file, options: opts} = part) do
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

  def content_part_for_api(%_{} = model, %ContentPart{type: image} = part)
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

    if is_gpt5_model?(model) do
      %{
        "type" => "input_image",
        # Responses API expects a string URL for image_url
        "image_url" => media_prefix <> part.content
      }
    else
      %{
        "type" => "image_url",
        "image_url" =>
          %{"url" => media_prefix <> part.content}
          |> Utils.conditionally_add_to_map("detail", detail_option)
      }
    end
  end

  # Role-aware mapping helper for GPT-5
  defp content_part_for_api_with_role(%_{} = model, %ContentPart{type: :text} = part, role) do
    if is_gpt5_model?(model) do
      type = if role == :assistant, do: "output_text", else: "input_text"
      %{"type" => type, "text" => part.content}
    else
      content_part_for_api(model, part)
    end
  end

  defp content_part_for_api_with_role(%_{} = model, %ContentPart{} = part, _role) do
    content_part_for_api(model, part)
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

  # Convert a message role into either `:system` or :developer` based on the
  # message role and the system config.
  defp get_message_role(%ChatOpenAI{reasoning_mode: true}, :system), do: :developer
  defp get_message_role(%ChatOpenAI{}, role), do: role
  defp get_message_role(_model, role), do: role

  # Detect if endpoint query has api-version set to preview or latest
  defp is_preview_latest?(endpoint) when is_binary(endpoint) do
    case URI.parse(endpoint).query do
      nil -> false
      query ->
        case URI.decode_query(query)["api-version"] do
          nil -> false
          version when is_binary(version) -> version == "latest" or version == "preview"
          _ -> false
        end
    end
  end

  # Normalize base endpoint for Azure next-gen (preview/latest): ensure /openai/v1/
  defp azure_normalize_base_endpoint(%ChatOpenAI{endpoint: endpoint} = _openai) do
    if is_preview_latest?(endpoint) do
      case URI.parse(endpoint) do
        %URI{path: path} = uri when is_binary(path) ->
          if String.contains?(path, "/openai/v1/") do
            endpoint
          else
            adjusted = String.replace(path, "/openai/", "/openai/v1/")
            URI.to_string(%URI{uri | path: adjusted})
          end

        _ -> endpoint
      end
    else
      endpoint
    end
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
    # GPT-5 (Responses API) with tools requires a non-streaming create + continue loop
    # regardless of the external stream flag. Handle that here.
    if is_gpt5_model?(openai) and is_list(tools) and length(tools) > 0 do
      return = do_responses_continue_loop(openai, messages, tools, retry_count)

      case return do
        {:ok, [%Message{} | _] = messages} ->
          Callbacks.fire(openai.callbacks, :on_llm_new_message, [messages])
          messages

        other ->
          other
      end
    else
      raw_data = for_api(openai, messages, tools)

      if openai.verbose_api do
        IO.inspect(raw_data, label: "RAW DATA BEING SUBMITTED")
      end

      req =
        Req.new(
          url: azure_normalize_base_endpoint(openai),
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
      |> maybe_add_org_id_header(openai)
      |> maybe_add_proj_id_header()
      |> Req.post()
      # parse the body and return it as parsed structs
      |> case do
        {:ok, %Req.Response{body: data} = response} ->
          if openai.verbose_api do
            IO.inspect(response, label: "RAW REQ RESPONSE")
          end

          Callbacks.fire(openai.callbacks, :on_llm_response_headers, [response.headers])

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
  end

  def do_api_request(
        %ChatOpenAI{stream: true} = openai,
        messages,
        tools,
        retry_count
      ) do
    # GPT-5 (Responses API) with tools requires a non-streaming create + continue loop
    # SSE does not reach a terminal event until the continue call occurs.
    if is_gpt5_model?(openai) and is_list(tools) and length(tools) > 0 do
      # Force the non-streaming continue loop even if streaming is requested
      return = do_responses_continue_loop(%ChatOpenAI{openai | stream: false}, messages, tools, retry_count)

      case return do
        {:ok, [%Message{} | _] = messages} ->
          Callbacks.fire(openai.callbacks, :on_llm_new_message, [messages])
          messages

        other ->
          other
      end
    else
      raw_data = for_api(openai, messages, tools)

      if openai.verbose_api do
        IO.inspect(raw_data, label: "RAW DATA BEING SUBMITTED")
      end

      Req.new(
        url: azure_normalize_base_endpoint(openai),
        json: raw_data,
        # required for OpenAI API
        auth: {:bearer, get_api_key(openai)},
        # required for Azure OpenAI version
        headers: [
          {"api-key", get_api_key(openai)}
        ],
        receive_timeout: openai.receive_timeout
      )
      |> maybe_add_org_id_header(openai)
      |> maybe_add_proj_id_header()
      |> Req.post(
        into:
          Utils.handle_stream_fn(
            openai,
            &decode_stream/1,
            &do_process_response(openai, &1)
          )
      )
      |> case do
        {:ok, %Req.Response{body: data} = response} ->
          Callbacks.fire(openai.callbacks, :on_llm_response_headers, [response.headers])

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
  end

  # --- GPT-5 Responses API: create + continue tool-calling loop (non-streaming driver) ---

  # Drive the Responses API until completion, executing LangChain tools locally
  # and feeding their outputs back via the continue request using prompt_state.
  defp do_responses_continue_loop(%ChatOpenAI{} = openai, messages, tools, retry_count) do
    # Always call the base request with stream=false for the driver loop
    base_model = %ChatOpenAI{openai | stream: false}
    raw_data = for_api(base_model, messages, tools)

    with {:ok, first_resp, headers} <- post_openai_json(openai, raw_data),
         :ok <- fire_headers_callbacks(openai, headers) do
      Logger.debug(fn -> "Responses initial body: #{inspect(first_resp)}" end)
      continue_until_complete(openai, tools, first_resp, 0)
    else
      {:error, %Req.TransportError{reason: :closed}} when retry_count > 0 ->
        Logger.debug(fn -> "Connection closed: retry count = #{inspect(retry_count)}" end)
        do_responses_continue_loop(openai, messages, tools, retry_count - 1)

      {:error, %Req.TransportError{reason: :timeout} = err} ->
        {:error,
         LangChainError.exception(type: "timeout", message: "Request timed out", original: err)}

      {:error, %LangChainError{} = err} ->
        {:error, err}

      other ->
        Logger.error("Unexpected error starting Responses loop: #{inspect(other)}")
        {:error,
         LangChainError.exception(type: "unexpected_response", message: "Unexpected response")}
    end
  end

  # Continue loop: parse response for tool calls; if present, execute and continue; else finalize
  defp continue_until_complete(%ChatOpenAI{} = openai, tools, %{} = resp, depth)
       when depth < 25 do
    case extract_required_tool_calls(resp) do
      {:continue, response_id, prompt_state, tool_calls} ->
        metadata = extract_response_metadata(resp)
        {tool_outputs, _had_errors} = execute_responses_tool_calls(tools, tool_calls)

        continue_payload =
          %{
            tool_outputs:
              Enum.map(tool_outputs, fn
                %{"tool_call_id" => alt_id, "call_id" => call_id, "output_text" => text} ->
                  %{"call_id" => (call_id || alt_id), "output" => text}

                %{"tool_call_id" => alt_id, "output_text" => text} ->
                  %{"call_id" => alt_id, "output" => text}
              end),
            metadata: metadata || %{}
          }
          |> Utils.conditionally_add_to_map(:prompt_state, prompt_state)

        case submit_tool_outputs_with_fallbacks(openai, response_id, continue_payload) do
          {:ok, next_resp, _headers} ->
            continue_until_complete(openai, tools, next_resp, depth + 1)

          {:error, %LangChainError{} = err} ->
            {:error, err}

          {:error, %Req.TransportError{reason: :closed}} ->
            # single retry path for closed connections during continue
            case submit_tool_outputs_with_fallbacks(openai, response_id, continue_payload) do
              {:ok, next_resp, _headers} -> continue_until_complete(openai, tools, next_resp, depth + 1)
              other -> other
            end

          other ->
            Logger.error("Unexpected continue response: #{inspect(other)}")
            {:error,
             LangChainError.exception(type: "unexpected_response", message: "Unexpected response")}
        end

      :final ->
        # Convert final Responses object to a Message list using existing parser
        case do_process_response(openai, resp) do
          {:error, %LangChainError{} = reason} -> {:error, reason}
          %Message{} = message -> {:ok, [message]}
          [%Message{} | _] = messages -> {:ok, messages}
          other ->
            Logger.error("Unexpected final parse in Responses loop: #{inspect(other)}")
            {:error,
             LangChainError.exception(type: "unexpected_response", message: "Unexpected final parse")}
        end
    end
  end

  defp continue_until_complete(_openai, _tools, resp, _depth) do
    Logger.error("Exceeded maximum continue iterations. Resp: #{inspect(resp)}")
    {:error,
     LangChainError.exception(type: "exceeded_max_runs", message: "Exceeded maximum continues")}
  end

  # If response indicates tools are required, returns {:continue, response_id, prompt_state, tool_calls}
  # Otherwise returns :final
  defp extract_required_tool_calls(%{"required_action" => required_action} = resp)
       when is_map(required_action) do
    # Two shapes are seen: directly under required_action["tool_calls"] or nested under submit_tool_outputs
    tool_calls =
      Map.get(required_action, "tool_calls") ||
        get_in(required_action, ["submit_tool_outputs", "tool_calls"]) || []

    if is_list(tool_calls) and tool_calls != [] do
      response_id = resp["id"] || get_in(resp, ["response", "id"]) || get_in(resp, ["id"])
      prompt_state = Map.get(resp, "prompt_state") || get_in(resp, ["response", "prompt_state"])
      {:continue, response_id, prompt_state, tool_calls}
    else
      :final
    end
  end

  # Fallback shape: non-streaming may expose function calls as output items
  defp extract_required_tool_calls(%{"output" => outputs} = resp) when is_list(outputs) do
    tool_calls =
      outputs
      |> Enum.filter(&match?(%{"type" => "function_call"}, &1))
      |> Enum.map(fn %{"id" => id, "name" => name} = item ->
        %{
          "id" => id,
          "type" => "function",
          "name" => name,
          "arguments" => Map.get(item, "arguments"),
          "call_id" => Map.get(item, "call_id")
        }
      end)

    if tool_calls != [] do
      response_id = resp["id"] || get_in(resp, ["response", "id"]) || get_in(resp, ["id"])
      prompt_state = Map.get(resp, "prompt_state") || get_in(resp, ["response", "prompt_state"])
      {:continue, response_id, prompt_state, tool_calls}
    else
      :final
    end
  end

  defp extract_required_tool_calls(%{"status" => status}) when status in ["completed", "finished"] do
    :final
  end

  defp extract_required_tool_calls(_resp), do: :final

  defp extract_response_metadata(%{"metadata" => meta}) when is_map(meta), do: meta
  defp extract_response_metadata(%{"response" => %{"metadata" => meta}}) when is_map(meta),
    do: meta
  defp extract_response_metadata(_), do: nil

  # Execute LangChain tools and return the Responses API's expected tool_outputs list
  defp execute_responses_tool_calls(tools, tool_calls) when is_list(tools) do
    tool_map =
      tools
      |> Enum.reduce(%{}, fn
        %Function{name: name} = f, acc -> Map.put(acc, name, f)
        _other, acc -> acc
      end)

    results =
      Enum.map(tool_calls, fn call ->
        call_id = call["id"] || call["call_id"] || call["tool_call_id"]
        name = call["name"]
        raw_args = Map.get(call, "arguments")
        args =
          cond do
            is_map(raw_args) -> raw_args
            is_binary(raw_args) ->
              case Jason.decode(raw_args) do
                {:ok, %{} = m} -> m
                _ -> %{}
              end
            true -> %{}
          end

        output_text =
          case Map.get(tool_map, name) do
            %Function{} = function ->
              case Function.execute(function, args, nil) do
                {:ok, %ToolResult{} = tr} -> tool_result_to_string(tr)
                {:ok, llm_result, _processed} -> to_output_string(llm_result)
                {:ok, llm_result} -> to_output_string(llm_result)
                {:error, reason} when is_binary(reason) -> reason
                {:error, other} -> inspect(other)
              end

            nil ->
              "ERROR: Tool '#{name}' not found"
          end

        %{"tool_call_id" => call_id, "output_text" => output_text}
      end)

    {results, Enum.any?(results, fn r -> String.starts_with?(r["output_text"], "ERROR:") end)}
  end

  defp tool_result_to_string(%ToolResult{content: content}) when is_binary(content), do: content

  defp tool_result_to_string(%ToolResult{content: content}) when is_list(content) do
    # ContentParts list -> stringify text
    LangChain.Message.ContentPart.content_to_string(content, :text) || Jason.encode!(content)
  end

  defp tool_result_to_string(%ToolResult{content: content}) when is_map(content),
    do: Jason.encode!(content)

  defp tool_result_to_string(%ToolResult{content: content}), do: to_output_string(content)

  defp to_output_string(content) when is_binary(content), do: content
  defp to_output_string(content) when is_map(content), do: Jason.encode!(content)
  defp to_output_string(content) when is_list(content), do: Jason.encode!(content)
  defp to_output_string(content), do: inspect(content)

  # Low-level POST wrapper returning {:ok, body, headers} | {:error, reason}
  defp post_openai_json(%ChatOpenAI{} = openai, %{} = json) do
    req =
      Req.new(
        url: azure_normalize_base_endpoint(openai),
        json: json,
        auth: {:bearer, get_api_key(openai)},
        headers: [
          {"api-key", get_api_key(openai)}
        ],
        receive_timeout: openai.receive_timeout,
        retry: :transient,
        max_retries: 3,
        retry_delay: fn attempt -> 300 * attempt end
      )
      |> maybe_add_org_id_header(openai)
      |> maybe_add_proj_id_header()

    case Req.post(req) do
      {:ok, %Req.Response{body: data, headers: headers}} ->
        {:ok, data, headers}

      {:error, %Req.TransportError{} = err} ->
        {:error, err}

      {:ok, %Req.Response{body: data}} ->
        {:ok, data, %{}}

      other ->
        Logger.error("Unexpected POST response: #{inspect(other)}")
        {:error,
         LangChainError.exception(type: "unexpected_response", message: "Unexpected response")}
    end
  end

  # Azure/Responses tool output submission endpoint helper
  defp submit_tool_outputs_request(%ChatOpenAI{} = openai, response_id, payload) do
    # For Azure Responses API, the continue/tool outputs endpoint is
    # POST {base_path}/{response_id}?api-version=...
    # The configured endpoint includes the query; inject the id and segment into the path.
    uri = URI.parse(openai.endpoint)
    base_path = uri.path || ""
    new_path =
      base_path
      |> String.trim_trailing("/")
      |> Kernel.<>("/" <> to_string(response_id) <> "/tool_outputs")

    continue_url = URI.to_string(%URI{uri | path: new_path})

    req =
      Req.new(
        url: continue_url,
        json: payload,
        auth: {:bearer, get_api_key(openai)},
        headers: [
          {"api-key", get_api_key(openai)}
        ],
        receive_timeout: openai.receive_timeout,
        retry: :transient,
        max_retries: 3,
        retry_delay: fn attempt -> 300 * attempt end
      )
      |> maybe_add_org_id_header(openai)
      |> maybe_add_proj_id_header()

    Logger.debug(fn -> "Submitting tool outputs to #{continue_url} payload=#{inspect(payload)}" end)
    case Req.post(req) do
      {:ok, %Req.Response{body: data, headers: headers}} -> {:ok, data, headers}
      {:error, %Req.TransportError{} = err} -> {:error, err}
      {:ok, %Req.Response{body: data}} -> {:ok, data, %{}}
      other ->
        Logger.error("Unexpected tool_outputs POST response: #{inspect(other)}")
        {:error,
         LangChainError.exception(type: "unexpected_response", message: "Unexpected response")}
    end
  end

  # Try multiple Azure Responses API shapes for submitting tool outputs
  defp submit_tool_outputs_with_fallbacks(%ChatOpenAI{} = openai, response_id, payload) do
    # Build two payload variants differing by id key
    outputs = Map.get(payload, :tool_outputs) || Map.get(payload, "tool_outputs") || []
    with_call_id = Map.put(payload, :tool_outputs, Enum.map(outputs, fn o -> %{ "call_id" => o["call_id"] || o[:call_id] || o["tool_call_id"], "output" => o["output"] || o[:output] } end))
    with_tool_call_id = Map.put(payload, :tool_outputs, Enum.map(outputs, fn o -> %{ "tool_call_id" => o["tool_call_id"] || o[:tool_call_id] || o["call_id"], "output" => o["output"] || o[:output] } end))

    attempts = [
      {:post_path, :tool_outputs, with_call_id},
      {:post_path, :tool_outputs, with_tool_call_id},
      {:post_path, :continue, with_call_id},
      {:post_path, :continue, with_tool_call_id}
    ]

    Enum.reduce_while(attempts, {:error, LangChainError.exception(message: "all attempts failed")}, fn {mode, segment, pl}, _acc ->
      result =
        case mode do
          :post_path ->
            segment_path =
              case segment do
                :tool_outputs -> "/tool_outputs"
                :continue -> "/continue"
              end

            submit_tool_outputs_request_to_path(openai, response_id, pl, segment_path)
        end

      case result do
        {:ok, %{"error" => _} = data, _headers} ->
          # treat as failure and try next
          {:cont, {:error, LangChainError.exception(original: data, message: get_in(data, ["error", "message"]))}}

        {:ok, %{} = data, headers} ->
          {:halt, {:ok, data, headers}}

        {:error, _} = err ->
          {:cont, err}
      end
    end)
  end

  defp submit_tool_outputs_request_to_path(%ChatOpenAI{} = openai, response_id, payload, segment_path) do
    uri = URI.parse(openai.endpoint)
    base_path = uri.path || ""

    # Azure Responses endpoints may omit the deployment segment. If missing,
    # insert "/deployments/<model>" using the configured model name.
    adjusted_base_path =
      cond do
        String.contains?(base_path, "/openai/deployments/") ->
          base_path

        is_preview_latest?(openai.endpoint) ->
          # For preview/latest, ensure v1 responses path and do not inject deployments
          cond do
            String.contains?(base_path, "/openai/v1/responses") -> base_path
            String.contains?(base_path, "/openai/responses") ->
              String.replace(base_path, "/openai/responses", "/openai/v1/responses")
            true -> base_path
          end

        true ->
          # Older dated api-versions without deployments: keep base path unchanged
          base_path
      end

    new_path =
      adjusted_base_path
      |> String.trim_trailing("/")
      |> Kernel.<>("/" <> to_string(response_id) <> segment_path)

    url = URI.to_string(%URI{uri | path: new_path})

    req =
      Req.new(
        url: url,
        json: payload,
        auth: {:bearer, get_api_key(openai)},
        headers: [
          {"api-key", get_api_key(openai)}
        ],
        receive_timeout: openai.receive_timeout,
        retry: :transient,
        max_retries: 3,
        retry_delay: fn attempt -> 300 * attempt end
      )
      |> maybe_add_org_id_header(openai)
      |> maybe_add_proj_id_header()

    Logger.debug(fn -> "Submitting tool outputs to #{url} payload=#{inspect(payload)}" end)

    case Req.post(req) do
      {:ok, %Req.Response{body: data, headers: headers}} -> {:ok, data, headers}
      {:error, %Req.TransportError{} = err} -> {:error, err}
      {:ok, %Req.Response{body: data}} -> {:ok, data, %{}}
      other ->
        Logger.error("Unexpected tool_outputs POST response: #{inspect(other)}")
        {:error,
         LangChainError.exception(type: "unexpected_response", message: "Unexpected response")}
    end
  end

  defp fire_headers_callbacks(%{callbacks: callbacks} = _openai, headers) do
    # Best-effort header callback compatibility
    try do
      Callbacks.fire(callbacks, :on_llm_response_headers, [headers])
      Callbacks.fire(callbacks, :on_llm_ratelimit_info, [get_ratelimit_info(headers)])
      :ok
    rescue
      _ -> :ok
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
  @spec decode_stream({String.t(), String.t()}, list()) ::
          {%{String.t() => any()}}
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
          # Remove any SSE event lines like "event: response.output_text.delta"
          cleaned =
            json
            |> String.split("\n")
            |> Enum.reject(&String.starts_with?(&1, "event:"))
            |> Enum.join("\n")

          parse_combined_data(incomplete, cleaned, done)
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
        |> Enum.map(&TokenUsage.set(&1, token_usage))
    end
  end

  # Responses API (GPT-5) - non-streaming final response
  def do_process_response(
        _model,
        %{"object" => "response", "content" => parts} = data
      )
      when is_list(parts) do
    text_parts =
      parts
      |> Enum.filter(&match?(%{"type" => "output_text"}, &1))
      |> Enum.map(fn %{"text" => text} -> ContentPart.text!(text) end)

    case Message.new(%{
           "role" => "assistant",
           "content" => text_parts,
           "complete" => true,
           "index" => 0
         }) do
      {:ok, message} ->
        message = TokenUsage.set(message, get_token_usage(data))
        [message]

      {:error, %Ecto.Changeset{} = changeset} ->
        {:error, LangChainError.exception(changeset)}
    end
  end

  # Responses API (GPT-5) - non-streaming final response with "output" array
  def do_process_response(
        _model,
        %{"object" => "response", "output" => outputs} = data
      )
      when is_list(outputs) do
    text_parts =
      outputs
      |> Enum.filter(&match?(%{"type" => "message"}, &1))
      |> Enum.flat_map(fn %{"content" => parts} -> parts end)
      |> Enum.filter(&match?(%{"type" => "output_text"}, &1))
      |> Enum.map(fn %{"text" => text} -> ContentPart.text!(text) end)

    case Message.new(%{
           "role" => "assistant",
           "content" => text_parts,
           "complete" => true,
           "index" => 0
         }) do
      {:ok, message} ->
        message = TokenUsage.set(message, get_token_usage(data))
        [message]

      {:error, %Ecto.Changeset{} = changeset} ->
        {:error, LangChainError.exception(changeset)}
    end
  end

  # Responses API (GPT-5) - streaming events
  # Text delta
  def do_process_response(
        _model,
        %{"type" => "response.output_text.delta", "delta" => text} = data
      ) do
    index = Map.get(data, "output_index", 0)

    MessageDelta.new!(%{
      role: :unknown,
      content: ContentPart.text!(text),
      status: :incomplete,
      index: index
    })
  end

  # Responses API - non-content lifecycle events (skip)
  def do_process_response(_model, %{"type" => type})
      when type in [
             "response.created",
             "response.in_progress"
           ] do
    :skip
  end

  # Responses API - output item lifecycle (reasoning/messages scaffolding and tool calls)
  # Handle function_call item creation; ignore other scaffolding
  def do_process_response(
        _model,
        %{"type" => "response.output_item.added", "item" => item} = data
      ) do
    case item do
      %{"type" => "function_call"} ->
        index = Map.get(data, "output_index", 0)
        call_id = item["id"] || Map.get(data, "item_id")
        name = item["name"]

        tool_call =
          ToolCall.new!(%{
            status: :incomplete,
            type: :function,
            call_id: call_id,
            name: name,
            index: index
          })

        MessageDelta.new!(%{
          role: :assistant,
          content: nil,
          status: :incomplete,
          index: index,
          tool_calls: [tool_call]
        })

      _ ->
        :skip
    end
  end

  def do_process_response(_model, %{"type" => "response.output_item.done"}), do: :skip

  # Responses API - content part lifecycle and deltas
  def do_process_response(_model, %{"type" => "response.content_part.added"} = data) do
    index = Map.get(data, "output_index", 0)

    content =
      case get_in(data, ["part", "type"]) do
        "output_text" -> ContentPart.text!(get_in(data, ["part", "text"]) || "")
        _ -> nil
      end

    MessageDelta.new!(%{role: :unknown, content: content, status: :incomplete, index: index})
  end

  def do_process_response(
        _model,
        %{"type" => "response.content_part.delta", "delta" => text} = data
      ) do
    index = Map.get(data, "output_index", 0)

    MessageDelta.new!(%{
      role: :unknown,
      content: ContentPart.text!(text),
      status: :incomplete,
      index: index
    })
  end

  def do_process_response(_model, %{"type" => "response.content_part.done"} = data) do
    index = Map.get(data, "output_index", 0)

    MessageDelta.new!(%{role: :unknown, content: nil, status: :complete, index: index})
  end

  # Responses API - tool call arguments streaming
  def do_process_response(
        _model,
        %{"type" => "response.function_call_arguments.delta", "delta" => text} = data
      ) do
    index = Map.get(data, "output_index", 0)
    call_id = Map.get(data, "item_id")

    tool_call =
      ToolCall.new!(%{
        status: :incomplete,
        type: :function,
        call_id: call_id,
        arguments: text,
        index: index
      })

    MessageDelta.new!(%{
      role: :assistant,
      content: nil,
      status: :incomplete,
      index: index,
      tool_calls: [tool_call]
    })
  end

  def do_process_response(
        _model,
        %{"type" => "response.function_call_arguments.done", "arguments" => args} = data
      ) do
    index = Map.get(data, "output_index", 0)
    call_id = Map.get(data, "item_id")
    name = Map.get(data, "name")

    # Mark the tool call as complete with the final arguments
    tool_call =
      ToolCall.new!(%{
        status: :complete,
        type: :function,
        call_id: call_id,
        arguments: args,
        name: name,
        index: index
      })

    MessageDelta.new!(%{
      role: :assistant,
      content: nil,
      status: :incomplete,
      index: index,
      tool_calls: [tool_call]
    })
  end

  # Text done/completed marker
  def do_process_response(
        _model,
        %{"type" => "response.output_text.done"} = data
      ) do
    index = Map.get(data, "output_index", 0)

    MessageDelta.new!(%{
      role: :unknown,
      content: nil,
      status: :complete,
      index: index
    })
  end

  # Responses API - completed event carries usage
  def do_process_response(_model, %{"type" => "response.completed", "response" => resp}) do
    get_token_usage(%{"usage" => Map.get(resp, "usage", %{})})
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
           "tool_calls" => Enum.map(calls || [], &do_process_response(model, &1))
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
  def do_process_response(
        _model,
        %{
          "error" => %{"code" => code, "message" => reason} = error_data
        } = response
      ) do
    type =
      case code do
        "429" ->
          "rate_limit_exceeded"

        "unsupported_value" ->
          if String.contains?(reason, "does not support 'system' with this model") do
            Logger.error(
              "This model requires 'reasoning_mode' to be enabled. Reason: #{inspect(reason)}"
            )

            # return the API error type as the exception type information
            error_data["type"]
          end

        _other ->
          nil
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

  # Detect GPT-5 family models to switch to Responses API shape
  defp is_gpt5_model?(%ChatOpenAI{model: model}) when is_binary(model) do
    String.starts_with?(model, "gpt-5")
  end

  defp is_gpt5_model?(_), do: false

  defp maybe_add_org_id_header(%Req.Request{} = req, %ChatOpenAI{} = openai) do
    org_id = get_org_id(openai)

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

  defp get_token_usage(%{"usage" => usage} = _response_body) when is_map(usage) do
    # extract out the reported response token usage
    #
    #  https://platform.openai.com/docs/api-reference/chat/object#chat/object-usage
    TokenUsage.new!(%{
      input: Map.get(usage, "prompt_tokens"),
      output: Map.get(usage, "completion_tokens"),
      raw: usage
    })
  end

  # GPT-5 Responses API usage shape
  defp get_token_usage(%{"usage" => usage} = _response_body) when is_map(usage) do
    TokenUsage.new!(%{
      input: Map.get(usage, "input_tokens") || Map.get(usage, "prompt_tokens"),
      output: Map.get(usage, "output_tokens") || Map.get(usage, "completion_tokens"),
      raw: usage
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
    ChatOpenAI.new(data)
  end
end
