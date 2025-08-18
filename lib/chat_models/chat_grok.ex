defmodule LangChain.ChatModels.ChatGrok do
  @moduledoc """
  Module for interacting with [xAI's Grok models](https://docs.x.ai/docs/models).

  Parses and validates inputs for making requests to [xAI's chat completions API](https://docs.x.ai/docs/api-reference).

  Converts responses into more specialized `LangChain` data structures.

  ## Tested with models

  - `grok-4` - The latest and most advanced reasoning model with 130K+ context window
  - `grok-3-mini` - Faster, lightweight model optimized for speed and efficiency
  and other, please look in the tests `--include live_grok` for more.

  ## OpenAI API Compatibility

  Grok's API is fully compatible with OpenAI's format, making integration straightforward.
  The main differences are:
  - Base URL: `https://api.x.ai/v1/chat/completions`
  - Model names: `grok-4`, `grok-3-mini`, etc.
  - Enhanced context window and reasoning capabilities

  ## Usage Example

      # Basic usage with Grok-4
      {:ok, chat} = ChatGrok.new(%{
        model: "grok-4",
        temperature: 0.7,
        max_tokens: 1000
      })

      # Fast and efficient with Grok-3-mini
      {:ok, grok_mini} = ChatGrok.new(%{
        model: "grok-3-mini",
        temperature: 0.8,
        max_tokens: 5000,
        api_key: System.get_env("XAI_API_KEY"),
        callbacks: [handlers]
      })

  ## Callbacks

  See the set of available callbacks: `LangChain.Chains.ChainCallbacks`

  ### Rate Limit API Response Headers

  xAI returns rate limit information in the response headers. Those can be
  accessed using the LLM callback `on_llm_ratelimit_info` like this:

      handlers = %{
        on_llm_ratelimit_info: fn _model, headers ->
          IO.inspect(headers, label: )
        end
      }

      {:ok, grok_mini} = ChatGrok.new(%{callbacks: [handlers]})

  ### Token Usage

  xAI returns token usage information as part of the response body. The
  `LangChain.TokenUsage` is added to the `metadata` of the `LangChain.Message`
  and `LangChain.MessageDelta` structs that are processed under the `:usage`
  key.

  ## Tool Choice

  Grok supports forcing a tool to be used, following OpenAI's format:

      ChatGrok.new(%{
        model: "grok-4",
        tool_choice: %{"type" => "function", "function" => %{"name" => "get_weather"}}
      })

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
  alias LangChain.Function
  alias LangChain.MessageDelta
  alias LangChain.Utils
  alias LangChain.Callbacks
  alias LangChain.LangChainError

  @behaviour ChatModel

  @current_config_version 1

  # Allow up to 2 minutes for response due to large context and reasoning time
  @receive_timeout 120_000

  @primary_key false
  embedded_schema do
    # API endpoint to use. Defaults to xAI's API
    field :endpoint, :string, default: "https://api.x.ai/v1/chat/completions"

    # API key for xAI. If not set, will use global api key.
    field :api_key, :string, redact: true

    # Duration in seconds for the response to be received. When streaming a very
    # lengthy response, a longer time limit may be required.
    field :receive_timeout, :integer, default: @receive_timeout

    # Model to use. Defaults to grok-4
    field :model, :string, default: "grok-4"

    # The maximum tokens allowed for generating a response.
    field :max_tokens, :integer, default: 4096

    # Amount of randomness injected into the response. Ranges from 0.0 to 1.0.
    field :temperature, :float, default: 0.7

    # Use nucleus sampling. Controls diversity via nucleus sampling.
    field :top_p, :float

    # Frequency penalty. Penalizes repeated tokens.
    field :frequency_penalty, :float

    # Presence penalty. Penalizes new tokens based on whether they appear in the text so far.
    field :presence_penalty, :float

    # Random seed for deterministic outputs
    field :seed, :integer

    # Number of chat completion choices to generate for each input message
    field :n, :integer, default: 1

    # Whether to stream the response
    field :stream, :boolean, default: false

    # Stream options for usage tracking
    field :stream_options, :map

    # Tool choice option for forcing specific function calls
    field :tool_choice, :map

    # Response format (for JSON responses)
    field :response_format, :map

    # A list of maps for callback handlers
    field :callbacks, {:array, :map}, default: []

    # Additional level of raw api request and response data
    field :verbose_api, :boolean, default: false

    # Grok-specific: Enhanced reasoning mode for complex problems
    field :reasoning_mode, :boolean, default: false

    # Grok-specific: Multi-agent coordination for Grok-4 Heavy
    field :multi_agent, :boolean, default: false

    # Grok-specific: Context window optimization for large contexts
    field :large_context, :boolean, default: false
  end

  @type t :: %ChatGrok{}

  @create_fields [
    :endpoint,
    :api_key,
    :receive_timeout,
    :model,
    :max_tokens,
    :temperature,
    :top_p,
    :frequency_penalty,
    :presence_penalty,
    :seed,
    :n,
    :stream,
    :stream_options,
    :tool_choice,
    :response_format,
    :callbacks,
    :verbose_api,
    :reasoning_mode,
    :multi_agent,
    :large_context
  ]

  @required_fields [:endpoint, :model]

  @doc """
  Setup a ChatGrok client configuration.
  """
  @spec new(attrs :: map()) :: {:ok, t} | {:error, Ecto.Changeset.t()}
  def new(%{} = attrs \\ %{}) do
    %ChatGrok{}
    |> cast(attrs, @create_fields)
    |> common_validation()
    |> apply_action(:insert)
  end

  @doc """
  Setup a ChatGrok client configuration and return it or raise an error if invalid.
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
    |> validate_number(:frequency_penalty, greater_than_or_equal_to: -2, less_than_or_equal_to: 2)
    |> validate_number(:presence_penalty, greater_than_or_equal_to: -2, less_than_or_equal_to: 2)
    |> validate_number(:max_tokens, greater_than: 0)
    |> validate_number(:n, greater_than: 0)
    |> validate_number(:receive_timeout, greater_than_or_equal_to: 0)
    |> validate_grok_specific_features()
  end

  defp validate_grok_specific_features(changeset) do
    model = get_field(changeset, :model)
    multi_agent = get_field(changeset, :multi_agent)

    cond do
      multi_agent && model != "grok-4-heavy" ->
        add_error(changeset, :multi_agent, "can only be enabled with grok-4-heavy model")

      true ->
        changeset
    end
  end

  @doc """
  Return the params formatted for an API request.
  """
  def for_api(%ChatGrok{} = grok, messages, tools \\ []) do
    base_params =
      %{
        model: grok.model,
        messages: messages |> Enum.map(&for_api_message/1),
        stream: grok.stream
      }
      |> Utils.conditionally_add_to_map(:max_tokens, grok.max_tokens)
      |> Utils.conditionally_add_to_map(:temperature, grok.temperature)
      |> Utils.conditionally_add_to_map(:top_p, grok.top_p)
      |> Utils.conditionally_add_to_map(:frequency_penalty, grok.frequency_penalty)
      |> Utils.conditionally_add_to_map(:presence_penalty, grok.presence_penalty)
      |> Utils.conditionally_add_to_map(:seed, grok.seed)
      |> Utils.conditionally_add_to_map(:n, grok.n)
      |> Utils.conditionally_add_to_map(:stream_options, grok.stream_options)
      |> Utils.conditionally_add_to_map(:response_format, grok.response_format)
      |> Utils.conditionally_add_to_map(:tools, format_tools_for_api(tools))
      |> Utils.conditionally_add_to_map(:tool_choice, grok.tool_choice)

    # Add Grok-specific enhancements
    base_params
    |> add_grok_specific_params(grok)
  end

  defp add_grok_specific_params(params, grok) do
    params
    |> maybe_add_reasoning_enhancement(grok)
    |> maybe_add_multi_agent_coordination(grok)
    |> maybe_add_large_context_optimization(grok)
  end

  defp maybe_add_reasoning_enhancement(params, %{reasoning_mode: true}) do
    # Enhanced reasoning instructions for complex problems
    Map.update(params, :messages, [], fn messages ->
      case messages do
        [%{role: "system"} = system | rest] ->
          enhanced_content =
            system.content <>
              "\n\nUse step-by-step reasoning and first principles thinking for complex problems."

          [%{system | content: enhanced_content} | rest]

        messages ->
          system_msg = %{
            role: "system",
            content:
              "Use step-by-step reasoning and first principles thinking for complex problems."
          }

          [system_msg | messages]
      end
    end)
  end

  defp maybe_add_reasoning_enhancement(params, _), do: params

  defp maybe_add_multi_agent_coordination(params, %{multi_agent: true, model: "grok-4-heavy"}) do
    # Add instructions for multi-agent collaboration
    Map.update(params, :messages, [], fn messages ->
      case messages do
        [%{role: "system"} = system | rest] ->
          coordination_instruction =
            system.content <>
              "\n\nCoordinate multiple reasoning perspectives and synthesize the best approach like a collaborative study group."

          [%{system | content: coordination_instruction} | rest]

        messages ->
          system_msg = %{
            role: "system",
            content:
              "Coordinate multiple reasoning perspectives and synthesize the best approach like a collaborative study group."
          }

          [system_msg | messages]
      end
    end)
  end

  defp maybe_add_multi_agent_coordination(params, _), do: params

  defp maybe_add_large_context_optimization(params, %{large_context: true}) do
    # Optimize for large context window usage (130K tokens)
    params
    |> Map.put(:max_tokens, min(params[:max_tokens] || 4096, 4096))
    |> maybe_add_context_instruction()
  end

  defp maybe_add_large_context_optimization(params, _), do: params

  defp maybe_add_context_instruction(params) do
    Map.update(params, :messages, [], fn messages ->
      case messages do
        [%{role: "system"} = system | rest] ->
          context_instruction =
            system.content <>
              "\n\nUtilize the full context window efficiently for comprehensive analysis."

          [%{system | content: context_instruction} | rest]

        messages ->
          system_msg = %{
            role: "system",
            content: "Utilize the full context window efficiently for comprehensive analysis."
          }

          [system_msg | messages]
      end
    end)
  end

  @doc """
  Convert a LangChain structure to the expected xAI API format.
  """
  def for_api_message(%Message{role: :system} = message) do
    %{
      role: "system",
      content: get_content_string(message)
    }
  end

  def for_api_message(%Message{role: :user} = message) do
    content =
      case message.content do
        text when is_binary(text) ->
          text

        content_parts when is_list(content_parts) ->
          # If it's just a single text part, extract the text
          case content_parts do
            [%ContentPart{type: :text, content: text}] -> text
            _ -> Enum.map(content_parts, &format_content_part_for_api/1)
          end
      end

    %{
      role: "user",
      content: content
    }
  end

  def for_api_message(%Message{role: :assistant} = message) do
    base = %{
      role: "assistant"
    }

    base
    |> maybe_add_content(message)
    |> maybe_add_tool_calls(message)
  end

  def for_api_message(%Message{role: :tool, tool_results: tool_results} = _message)
      when is_list(tool_results) do
    # ToolResults turn into a list of tool messages for Grok (following OpenAI format)
    Enum.map(tool_results, fn result ->
      content =
        case result.content do
          text when is_binary(text) ->
            text

          content_parts when is_list(content_parts) ->
            content_parts
            |> Enum.filter(&(&1.type == :text))
            |> Enum.map(& &1.content)
            |> Enum.join(" ")
        end

      %{
        role: "tool",
        tool_call_id: result.tool_call_id,
        content: content
      }
    end)
  end

  defp maybe_add_content(base, %Message{content: content})
       when is_binary(content) and content != "" do
    Map.put(base, :content, content)
  end

  defp maybe_add_content(base, %Message{content: content}) when is_list(content) do
    # Handle ContentParts - if it's just a single text part, extract the text
    case content do
      [%ContentPart{type: :text, content: text}] when text != "" ->
        Map.put(base, :content, text)

      [] ->
        base

      _ ->
        # Multiple content parts - need more complex handling
        text_content =
          content
          |> Enum.filter(&(&1.type == :text))
          |> Enum.map(& &1.content)
          |> Enum.join(" ")

        if text_content != "" do
          Map.put(base, :content, text_content)
        else
          base
        end
    end
  end

  defp maybe_add_content(base, _message), do: base

  defp maybe_add_tool_calls(base, %Message{tool_calls: tool_calls})
       when is_list(tool_calls) and tool_calls != [] do
    formatted_calls = Enum.map(tool_calls, &format_tool_call_for_api/1)
    Map.put(base, :tool_calls, formatted_calls)
  end

  defp maybe_add_tool_calls(base, _message), do: base

  defp format_tool_call_for_api(%ToolCall{} = tool_call) do
    %{
      id: tool_call.call_id,
      type: "function",
      function: %{
        name: tool_call.name,
        arguments: tool_call.arguments
      }
    }
  end

  defp format_content_part_for_api(%ContentPart{type: :text, content: text}) do
    %{type: "text", text: text}
  end

  defp format_content_part_for_api(%ContentPart{type: :image_url, content: url}) do
    %{type: "image_url", image_url: %{url: url}}
  end

  defp format_content_part_for_api(%ContentPart{type: :image, content: data, options: options}) do
    %{
      type: "image_url",
      image_url: %{
        url: "data:#{options[:media_type] || "image/jpeg"};base64,#{data}"
      }
    }
  end

  defp get_content_string(%Message{content: content}) when is_binary(content), do: content

  defp get_content_string(%Message{content: content}) when is_list(content) do
    content
    |> Enum.filter(&(&1.type == :text))
    |> Enum.map(& &1.content)
    |> Enum.join(" ")
  end

  defp format_tools_for_api([]), do: nil

  defp format_tools_for_api(tools) when is_list(tools) do
    Enum.map(tools, &format_tool_for_api/1)
  end

  defp format_tool_for_api(%Function{} = function) do
    %{
      type: "function",
      function: %{
        name: function.name,
        description: function.description,
        parameters: function.parameters_schema || %{}
      }
    }
  end

  @doc """
  Calls the xAI API with the given messages and tools.
  """
  def call(grok, prompt, tools \\ [])

  def call(%ChatGrok{} = grok, prompt, tools) when is_binary(prompt) do
    messages = [
      Message.new_system!(),
      Message.new_user!(prompt)
    ]

    call(grok, messages, tools)
  end

  def call(%ChatGrok{} = grok, messages, tools) when is_list(messages) do
    metadata = %{
      model: grok.model,
      message_count: length(messages),
      tools_count: length(tools)
    }

    try do
      case do_api_request(grok, messages, tools, metadata) do
        {:ok, data} ->
          {:ok, data}

        {:error, reason} ->
          {:error, reason}
      end
    rescue
      err in LangChainError ->
        {:error, err}
    end
  end

  defp do_api_request(grok, messages, tools, metadata) do
    api_payload = for_api(grok, messages, tools)
    headers = req_headers(grok)

    # Log request details if verbose_api is enabled
    if grok.verbose_api do
      IO.puts("🚀 Grok API Request:")
      IO.puts("  URL: #{grok.endpoint}")
      IO.puts("  Headers:")

      Enum.each(headers, fn {k, v} ->
        IO.inspect({k, v})

        if k == "authorization" do
          IO.puts("    #{k}: Bearer ***#{String.slice(v, -10, 10)}")
        else
          IO.puts("    #{k}: #{v}")
        end
      end)

      IO.puts("  Payload:")
      IO.inspect(api_payload, pretty: true, limit: :infinity)
    end

    Req.post(
      url: grok.endpoint,
      json: api_payload,
      headers: headers,
      receive_timeout: grok.receive_timeout,
      retry: false,
      max_retries: 0
    )
    |> case do
      {:ok, %Req.Response{status: 200} = response} ->
        Callbacks.fire(grok.callbacks, :on_llm_response_headers, [response.headers])

        case grok.stream do
          true -> handle_stream_response(response, grok, metadata)
          false -> handle_response(response, grok, metadata)
        end

      {:ok, %Req.Response{} = response} ->
        handle_error_response(response)

      {:error, %Mint.TransportError{reason: :timeout}} ->
        {:error, LangChainError.exception(type: :timeout, message: "Request timed out")}

      {:error, %Mint.TransportError{reason: reason}} ->
        detailed_msg = "Transport error: #{inspect(reason)}"

        if Application.get_env(:langchain, :debug_api_errors, true) do
          IO.puts("🚨 Grok Transport Error: #{detailed_msg}")
        end

        {:error, LangChainError.exception(type: :transport_error, message: detailed_msg)}

      {:error, %Req.TransportError{reason: reason}} ->
        detailed_msg = "Req transport error: #{inspect(reason)}"

        if Application.get_env(:langchain, :debug_api_errors, true) do
          IO.puts("🚨 Grok Req Error: #{detailed_msg}")
        end

        {:error, LangChainError.exception(type: :transport_error, message: detailed_msg)}

      {:error, reason} ->
        detailed_msg = "HTTP request failed: #{inspect(reason)}"

        if Application.get_env(:langchain, :debug_api_errors, true) do
          IO.puts("🚨 Grok Request Error: #{detailed_msg}")
        end

        {:error, LangChainError.exception(type: :http_error, message: detailed_msg)}
    end
  end

  defp req_headers(grok) do
    api_key = grok.api_key || Config.resolve(:xai_api_key, "")

    [
      {"authorization", "Bearer #{api_key}"},
      {"content-type", "application/json"}
    ]
  end

  defp handle_response(%Req.Response{body: data}, grok, metadata) do
    case data do
      %{"choices" => choices} = response_data ->
        grok
        |> maybe_execute_callback(:on_llm_token_usage, [response_data])

        # Extract usage information from response and add to metadata
        updated_metadata = Map.put(metadata, :usage, response_data["usage"])
        messages = Enum.map(choices, &(&1 |> choice_to_message(updated_metadata)))
        {:ok, messages}

      %{"error" => error} ->
        {:error, LangChainError.exception(type: :api_error, message: error["message"])}

      other ->
        {:error,
         LangChainError.exception(
           type: :unexpected_response,
           message: "Unexpected response: #{inspect(other)}"
         )}
    end
  end

  defp handle_stream_response(%Req.Response{body: body}, _grok, metadata) when is_binary(body) do
    body
    |> String.split("\n")
    |> Enum.filter(&String.starts_with?(&1, "data: "))
    |> Enum.map(&String.slice(&1, 6..-1//1))
    |> Enum.filter(&(&1 != "[DONE]"))
    |> Enum.map(&Jason.decode!/1)
    |> Enum.map(&choice_delta_to_message(&1, metadata))
    |> then(&{:ok, &1})
  rescue
    error ->
      {:error, LangChainError.exception(type: :stream_parse_error, message: inspect(error))}
  end

  defp handle_error_response(%Req.Response{status: status, body: body, headers: headers}) do
    # Log detailed error information for debugging
    if Application.get_env(:langchain, :debug_api_errors, true) do
      IO.puts("🚨 Grok API Error Details:")
      IO.puts("  Status: #{status}")
      IO.puts("  Headers:")
      Enum.each(headers, fn {k, v} -> IO.puts("    #{k}: #{v}") end)
      IO.puts("  Body:")
      IO.inspect(body, pretty: true, limit: :infinity)
    end

    error_message =
      case body do
        %{"error" => %{"message" => message, "type" => type}} ->
          "#{type}: #{message}"

        %{"error" => %{"message" => message}} ->
          message

        %{"error" => message} when is_binary(message) ->
          message

        %{"detail" => detail} when is_binary(detail) ->
          detail

        text when is_binary(text) and text != "" ->
          text

        _ ->
          "HTTP #{status} error"
      end

    detailed_message = "#{error_message} (HTTP #{status})"

    {:error,
     LangChainError.exception(
       type: :api_error,
       message: detailed_message,
       original: %{status: status, body: body, headers: Map.new(headers)}
     )}
  end

  defp choice_to_message(%{"message" => message_data}, metadata) do
    usage = metadata[:usage]

    %Message{
      role: :assistant,
      content: message_data["content"],
      tool_calls: parse_tool_calls(message_data["tool_calls"]),
      metadata: %{usage: usage}
    }
  end

  defp choice_delta_to_message(%{"choices" => [choice | _]}, metadata) do
    delta = choice["delta"]

    %MessageDelta{
      role: :assistant,
      content: delta["content"] || "",
      tool_calls: parse_tool_calls(delta["tool_calls"]),
      metadata: metadata
    }
  end

  defp choice_delta_to_message(%{"choices" => []}, metadata) do
    %MessageDelta{
      role: :assistant,
      content: "",
      tool_calls: [],
      metadata: metadata
    }
  end

  defp choice_delta_to_message(data, metadata) do
    # Fallback for unexpected streaming data format
    %MessageDelta{
      role: :assistant,
      content: "",
      tool_calls: [],
      metadata: Map.put(metadata, :raw_data, data)
    }
  end

  defp parse_tool_calls(nil), do: []

  defp parse_tool_calls(tool_calls) when is_list(tool_calls) do
    Enum.map(tool_calls, fn tool_call ->
      %ToolCall{
        call_id: tool_call["id"],
        name: tool_call["function"]["name"],
        arguments: tool_call["function"]["arguments"]
      }
    end)
  end

  defp maybe_execute_callback(grok, callback_name, args) do
    Callbacks.fire(grok.callbacks, callback_name, [grok | args])
  end

  @doc """
  Serialize the configuration of a ChatGrok struct to a map for saving.
  """
  def serialize_config(%ChatGrok{} = grok) do
    grok
    |> Map.from_struct()
    |> Map.put(:module, ChatGrok)
    |> Map.put(:version, @current_config_version)
  end

  @doc """
  Restore a ChatGrok struct from a serialized configuration map.
  """
  def restore_from_map(%{"module" => "Elixir.LangChain.ChatModels.ChatGrok"} = data) do
    new(data)
  end

  def restore_from_map(%{module: ChatGrok} = data) do
    new(data)
  end

  def restore_from_map(_), do: {:error, "Invalid ChatGrok configuration"}
end
