defmodule LangChain.ChatModels.ChatPerplexity do
  @moduledoc """
  Represents the [Perplexity Chat model](https://docs.perplexity.ai/api-reference/chat-completions).

  This module implements a client for the Perplexity Chat API, providing functions to validate input parameters,
  format API requests, and parse API responses into LangChain's structured data types.

  Perplexity does not natively support tool calling in the same manner as some other chat models.
  To overcome this limitation, this module employs a workaround using structured outputs via a JSON schema.
  When tools are provided, the API request is augmented with a JSON schema that defines the expected format
  for tool calls. The response processing logic then detects and decodes these tool call details, converting them
  into corresponding ToolCall structs. This approach allows LangChain to seamlessly emulate tool calling functionality
  and integrate it with its standard workflow, similar to how ChatOpenAI handles function calls.

  In addition, this module supports various configuration options such as temperature, top_p, top_k,
  and streaming, as well as callbacks for token usage and new message events.

  Overall, this implementation provides a unified interface for interacting with the Perplexity Chat API
  while working around its limitations regarding tool calling.
  """
  use Ecto.Schema
  require Logger
  import Ecto.Changeset
  alias __MODULE__
  alias LangChain.Config
  alias LangChain.ChatModels.ChatModel
  alias LangChain.Message
  alias LangChain.MessageDelta
  alias LangChain.Message.ToolCall
  alias LangChain.TokenUsage
  alias LangChain.LangChainError
  alias LangChain.Utils
  alias LangChain.Callbacks

  @behaviour ChatModel

  @current_config_version 1

  # Default endpoint for Perplexity API
  @default_endpoint "https://api.perplexity.ai/chat/completions"

  # Default timeout of 1 minute
  @receive_timeout 60_000

  @primary_key false
  embedded_schema do
    field :endpoint, :string, default: @default_endpoint
    field :model, :string, default: "sonar-reasoning-pro"
    field :api_key, :string

    # What sampling temperature to use, between 0 and 2.
    # Higher values make output more random, lower values more deterministic.
    field :temperature, :float, default: 0.2

    # The nucleus sampling threshold, between 0 and 1.
    field :top_p, :float, default: 0.9

    # The number of tokens for highest top-k filtering (0-2048).
    field :top_k, :integer, default: 0

    # Maximum number of tokens to generate
    field :max_tokens, :integer

    # Whether to stream the response
    field :stream, :boolean, default: false

    # Presence penalty between -2.0 and 2.0
    field :presence_penalty, :float, default: 0.0

    # Frequency penalty greater than 0
    field :frequency_penalty, :float, default: 1.0

    # Search domain filter for limiting citations
    field :search_domain_filter, {:array, :string}

    # Whether to return images in response
    field :return_images, :boolean, default: false

    # Whether to return related questions
    field :return_related_questions, :boolean, default: false

    # Time interval for search recency
    field :search_recency_filter, :string

    # Response format for structured outputs
    field :response_format, :map

    # Duration in seconds for response timeout
    field :receive_timeout, :integer, default: @receive_timeout

    # A list of maps for callback handlers
    field :callbacks, {:array, :map}, default: []
  end

  @type t :: %ChatPerplexity{}

  @create_fields [
    :endpoint,
    :model,
    :api_key,
    :temperature,
    :top_p,
    :top_k,
    :max_tokens,
    :stream,
    :presence_penalty,
    :frequency_penalty,
    :search_domain_filter,
    :return_images,
    :return_related_questions,
    :search_recency_filter,
    :response_format,
    :receive_timeout
  ]

  @required_fields [:model]

  @spec get_api_key(t()) :: String.t()
  defp get_api_key(%ChatPerplexity{api_key: api_key}) do
    api_key || Config.resolve(:perplexity_key, "")
  end

  @doc """
  Setup a ChatPerplexity client configuration.
  """
  @spec new(attrs :: map()) :: {:ok, t} | {:error, Ecto.Changeset.t()}
  def new(%{} = attrs \\ %{}) do
    %ChatPerplexity{}
    |> cast(attrs, @create_fields)
    |> common_validation()
    |> apply_action(:insert)
  end

  @doc """
  Setup a ChatPerplexity client configuration and return it or raise an error if invalid.
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
    |> validate_number(:temperature, greater_than_or_equal_to: 0, less_than: 2)
    |> validate_number(:top_p, greater_than: 0, less_than_or_equal_to: 1)
    |> validate_number(:top_k, greater_than_or_equal_to: 0, less_than_or_equal_to: 2048)
    |> validate_number(:presence_penalty, greater_than_or_equal_to: -2, less_than_or_equal_to: 2)
    |> validate_number(:frequency_penalty, greater_than: 0)
    |> validate_number(:receive_timeout, greater_than_or_equal_to: 0)
  end

  @doc """
  Return the params formatted for an API request.
  """
  @spec for_api(t(), [Message.t()], ChatModel.tools()) :: %{atom() => any()}
  def for_api(%ChatPerplexity{} = perplexity, messages, tools) do
    # If tools are provided, we'll create a JSON schema to emulate tool calls
    response_format =
      if length(tools) > 0 do
        %{
          "type" => "json_object_with_schema",
          "schema" => %{
            "type" => "object",
            "required" => ["tool_calls"],
            "properties" => %{
              "tool_calls" => %{
                "type" => "array",
                "items" => %{
                  "type" => "object",
                  "required" => ["name", "arguments"],
                  "properties" => %{
                    "name" => %{"type" => "string", "enum" => Enum.map(tools, & &1.name)},
                    "arguments" => %{"type" => "object"}
                  }
                }
              }
            }
          }
        }
      else
        perplexity.response_format
      end

    %{
      model: perplexity.model,
      messages: Enum.map(messages, &for_api(perplexity, &1)),
      temperature: perplexity.temperature,
      top_p: perplexity.top_p,
      top_k: perplexity.top_k,
      stream: perplexity.stream
    }
    |> Utils.conditionally_add_to_map(:max_tokens, perplexity.max_tokens)
    |> Utils.conditionally_add_to_map(:presence_penalty, perplexity.presence_penalty)
    |> Utils.conditionally_add_to_map(:frequency_penalty, perplexity.frequency_penalty)
    |> Utils.conditionally_add_to_map(:search_domain_filter, perplexity.search_domain_filter)
    |> Utils.conditionally_add_to_map(:return_images, perplexity.return_images)
    |> Utils.conditionally_add_to_map(
      :return_related_questions,
      perplexity.return_related_questions
    )
    |> Utils.conditionally_add_to_map(:search_recency_filter, perplexity.search_recency_filter)
    |> Utils.conditionally_add_to_map(:response_format, response_format)
  end

  @doc """
  Convert a LangChain Message-based structure to the expected map of data for
  the Perplexity API.
  """
  @spec for_api(t(), Message.t()) :: %{String.t() => any()}
  def for_api(%ChatPerplexity{}, %Message{} = msg) do
    %{
      "role" => msg.role,
      "content" => msg.content
    }
  end

  @impl ChatModel
  def call(perplexity, prompt, tools \\ [])

  def call(%ChatPerplexity{} = perplexity, prompt, tools) when is_binary(prompt) do
    messages = [
      Message.new_system!(),
      Message.new_user!(prompt)
    ]

    call(perplexity, messages, tools)
  end

  def call(%ChatPerplexity{} = perplexity, messages, tools) when is_list(messages) do
    try do
      case do_api_request(perplexity, messages, tools) do
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

  @doc false
  @spec do_api_request(t(), [Message.t()], ChatModel.tools(), integer()) ::
          list() | struct() | {:error, LangChainError.t()}
  def do_api_request(perplexity, messages, tools, retry_count \\ 3)

  def do_api_request(_perplexity, _messages, _tools, 0) do
    raise LangChainError, "Retries exceeded. Connection failed."
  end

  def do_api_request(
        %ChatPerplexity{stream: false} = perplexity,
        messages,
        tools,
        retry_count
      ) do
    req =
      Req.new(
        url: perplexity.endpoint,
        json: for_api(perplexity, messages, tools),
        auth: {:bearer, get_api_key(perplexity)},
        receive_timeout: perplexity.receive_timeout
      )

    req
    |> Req.post()
    |> case do
      {:ok, %Req.Response{body: data} = _response} ->
        Callbacks.fire(perplexity.callbacks, :on_llm_token_usage, [
          get_token_usage(data)
        ])

        case do_process_response(perplexity, data) do
          {:error, %LangChainError{} = reason} ->
            {:error, reason}

          result ->
            Callbacks.fire(perplexity.callbacks, :on_llm_new_message, [result])
            result
        end

      {:error, %Req.TransportError{reason: :timeout} = err} ->
        {:error,
         LangChainError.exception(type: "timeout", message: "Request timed out", original: err)}

      {:error, %Req.TransportError{reason: :closed}} ->
        Logger.debug(fn -> "Connection closed: retry count = #{inspect(retry_count)}" end)
        do_api_request(perplexity, messages, tools, retry_count - 1)

      other ->
        Logger.error("Unexpected and unhandled API response! #{inspect(other)}")
        other
    end
  end

  def do_api_request(
        %ChatPerplexity{stream: true} = perplexity,
        messages,
        tools,
        retry_count
      ) do
    Req.new(
      url: perplexity.endpoint,
      json: for_api(perplexity, messages, tools),
      auth: {:bearer, get_api_key(perplexity)},
      receive_timeout: perplexity.receive_timeout
    )
    |> Req.post(
      into:
        Utils.handle_stream_fn(perplexity, &decode_stream/1, &do_process_response(perplexity, &1))
    )
    |> case do
      {:ok, %Req.Response{body: data}} ->
        data

      {:error, %LangChainError{} = error} ->
        {:error, error}

      {:error, %Req.TransportError{reason: :timeout} = err} ->
        {:error,
         LangChainError.exception(type: "timeout", message: "Request timed out", original: err)}

      {:error, %Req.TransportError{reason: :closed}} ->
        Logger.debug(fn -> "Connection closed: retry count = #{inspect(retry_count)}" end)
        do_api_request(perplexity, messages, tools, retry_count - 1)

      other ->
        Logger.error(
          "Unhandled and unexpected response from streamed post call. #{inspect(other)}"
        )

        {:error,
         LangChainError.exception(type: "unexpected_response", message: "Unexpected response")}
    end
  end

  @doc """
  Decode a streamed response from the Perplexity API.
  """
  @spec decode_stream({String.t(), String.t()}) :: {%{String.t() => any()}}
  def decode_stream({raw_data, buffer}, done \\ []) do
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
    starting_json = incomplete <> json
    decode_stream({starting_json, ""}, done)
  end

  @doc false
  def do_process_response(_model, %{"error" => %{"message" => reason, "type" => type}}) do
    Logger.error("Received error from API: #{inspect(reason)}")
    {:error, LangChainError.exception(type: type, message: reason)}
  end

  def do_process_response(_model, %{"error" => %{"message" => reason}}) do
    Logger.error("Received error from API: #{inspect(reason)}")
    {:error, LangChainError.exception(message: reason)}
  end

  def do_process_response(model, %{"choices" => [], "usage" => %{} = usage} = _data) do
    case get_token_usage(%{"usage" => usage}) do
      %TokenUsage{} = token_usage ->
        Callbacks.fire(model.callbacks, :on_llm_token_usage, [token_usage])
        :skip

      nil ->
        :skip
    end
  end

  def do_process_response(_model, %{"choices" => []}), do: :skip

  def do_process_response(model, %{"choices" => choices} = data) when is_list(choices) do
    # Fire token usage callback if present
    if usage = Map.get(data, "usage") do
      case get_token_usage(%{"usage" => usage}) do
        %TokenUsage{} = token_usage ->
          Callbacks.fire(model.callbacks, :on_llm_token_usage, [token_usage])

        nil ->
          :ok
      end
    end

    # Process each response individually
    for choice <- choices do
      do_process_response(model, choice)
    end
  end

  def do_process_response(
        _model,
        %{"finish_reason" => finish_reason, "message" => %{"content" => content}} = data
      ) do
    status = finish_reason_to_status(finish_reason)

    # Try to parse content as JSON for potential tool calls
    case Jason.decode(content) do
      {:ok, %{"tool_calls" => tool_calls}} when is_list(tool_calls) ->
        # Convert JSON tool calls to Message struct with tool calls
        case Message.new(%{
               "role" => :assistant,
               "content" => nil,
               "status" => status,
               "index" => data["index"],
               "tool_calls" =>
                 Enum.map(tool_calls, fn call ->
                   tool_call =
                     ToolCall.new!(%{
                       type: :function,
                       status: :complete,
                       name: call["name"],
                       arguments: Jason.encode!(call["arguments"]),
                       call_id: Ecto.UUID.generate()
                     })

                   # Force the arguments field to be a JSON string even if the ToolCall schema casts it
                   %{tool_call | arguments: Jason.encode!(call["arguments"])}
                 end)
             }) do
          {:ok, message} -> message
          {:error, changeset} -> {:error, LangChainError.exception(changeset)}
        end

      _ ->
        # Regular message processing
        case Message.new(%{
               "role" => :assistant,
               "content" => content,
               "status" => status,
               "index" => data["index"]
             }) do
          {:ok, message} -> message
          {:error, changeset} -> {:error, LangChainError.exception(changeset)}
        end
    end
  end

  def do_process_response(
        _model,
        %{"delta" => delta_body, "finish_reason" => finish, "index" => index} = _msg
      ) do
    status = finish_reason_to_status(finish)

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

    case MessageDelta.new(data) do
      {:ok, message} ->
        message

      {:error, %Ecto.Changeset{} = changeset} ->
        {:error, LangChainError.exception(changeset)}
    end
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
  defp finish_reason_to_status("length"), do: :length

  defp finish_reason_to_status(other) do
    Logger.warning("Unsupported finish_reason in message. Reason: #{inspect(other)}")
    nil
  end

  defp get_token_usage(%{"usage" => usage} = _response_body) do
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
  def serialize_config(%ChatPerplexity{} = model) do
    model
    |> Utils.to_serializable_map(
      [
        :endpoint,
        :model,
        :temperature,
        :top_p,
        :top_k,
        :max_tokens,
        :stream,
        :presence_penalty,
        :frequency_penalty,
        :search_domain_filter,
        :return_images,
        :return_related_questions,
        :search_recency_filter,
        :response_format,
        :receive_timeout
      ],
      @current_config_version
    )
    |> Map.delete("module")
  end

  @doc """
  Restores the model from the config.
  """
  @impl ChatModel
  def restore_from_map(%{"version" => 1} = data) do
    ChatPerplexity.new(data)
  end
end
