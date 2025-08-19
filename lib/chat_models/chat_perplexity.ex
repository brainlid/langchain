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

  ## Full Response Data

  The full Perplexity API response, including citations and search results, is captured in the
  `processed_content` field of the returned Message. This includes:

  - `id`: Unique identifier for the chat completion
  - `model`: The model that generated the response
  - `created`: Unix timestamp of when the completion was created
  - `usage`: Token usage information
  - `citations`: Array of citation sources for the response
  - `search_results`: Array of search results related to the response

  You can access this metadata like:

      {:ok, [message]} = ChatPerplexity.call(perplexity, "Tell me about climate change")
      citations = message.processed_content.citations
      search_results = message.processed_content.search_results

      # Example of what citations might look like:
      # ["https://climate.nasa.gov/", "https://www.ipcc.ch/"]

      # Example of what search_results might look like:
      # [
      #   %{
      #     "title" => "Climate Change and Global Warming",
      #     "url" => "https://climate.nasa.gov/",
      #     "date" => "2023-12-25"
      #   }
      # ]

  ## Tool Calls

  In order to use tool calls, you need specifically prompt Perplexity as outlined in their
  [Prompt Guide](https://docs.perplexity.ai/guides/prompt-guide) as well as the
  [Structured Outputs Guide](https://docs.perplexity.ai/guides/structured-outputs).

  Provide it additional prompting like:

  ```
  Rules:
  1. Provide only the final answer. It is important that you do not include any explanation on the steps below.
  2. Do not show the intermediate steps information.

  Output a JSON object with the following fields:
  - title: The article title
  - keywords: An array of SEO keywords
  - meta_description: The SEO meta description
  ```
  """
  use Ecto.Schema
  require Logger
  import Ecto.Changeset
  alias __MODULE__
  alias LangChain.Config
  alias LangChain.ChatModels.ChatModel
  alias LangChain.Message
  alias LangChain.MessageDelta
  alias LangChain.Message.ContentPart
  alias LangChain.Message.ToolCall
  alias LangChain.TokenUsage
  alias LangChain.LangChainError
  alias LangChain.Utils
  alias LangChain.Callbacks
  alias LangChain.Telemetry

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
    response_format =
      if tools && !Enum.empty?(tools) do
        %{
          "type" => "json_schema",
          "json_schema" => %{
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
                      "name" => %{
                        "type" => "string",
                        "enum" => Enum.map(tools, & &1.name)
                      },
                      "arguments" => build_arguments_schema(tools)
                    }
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

  defp build_arguments_schema([tool | _]) do
    properties =
      tool.parameters
      |> Enum.map(fn param ->
        {param.name, param_to_json_schema(param)}
      end)
      |> Map.new()

    required =
      tool.parameters
      |> Enum.filter(& &1.required)
      |> Enum.map(& &1.name)

    %{
      "type" => "object",
      "required" => required,
      "properties" => properties
    }
  end

  defp param_to_json_schema(param) do
    base = %{"type" => atom_to_json_type(param.type)}

    base
    |> add_enum(param)
    |> add_items(param)
  end

  defp atom_to_json_type(:string), do: "string"
  defp atom_to_json_type(:number), do: "number"
  defp atom_to_json_type(:integer), do: "integer"
  defp atom_to_json_type(:boolean), do: "boolean"
  defp atom_to_json_type(:array), do: "array"
  defp atom_to_json_type(:object), do: "object"

  defp add_enum(schema, %{enum: enum}) when enum in [nil, []], do: schema
  defp add_enum(schema, %{enum: enum}), do: Map.put(schema, "enum", enum)

  defp add_items(schema, %{type: :array, item_type: item_type}) do
    Map.put(schema, "items", %{"type" => atom_to_json_type(String.to_existing_atom(item_type))})
  end

  defp add_items(schema, _), do: schema

  @doc """
  Convert a LangChain Message-based structure to the expected map of data for
  the Perplexity API.
  """
  @spec for_api(t(), Message.t()) :: %{String.t() => any()}
  def for_api(%ChatPerplexity{}, %Message{} = msg) do
    content =
      case msg.content do
        content when is_binary(content) -> content
        content when is_list(content) -> ContentPart.parts_to_string(content)
        nil -> nil
      end

    %{
      "role" => msg.role,
      "content" => content
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
    metadata = %{
      model: perplexity.model,
      message_count: length(messages),
      tools_count: length(tools)
    }

    Telemetry.span([:langchain, :llm, :call], metadata, fn ->
      try do
        # Track the prompt being sent
        Telemetry.llm_prompt(
          %{system_time: System.system_time()},
          %{model: perplexity.model, messages: messages}
        )

        case do_api_request(perplexity, messages, tools) do
          {:error, reason} ->
            {:error, reason}

          parsed_data ->
            # Track the response being received
            Telemetry.llm_response(
              %{system_time: System.system_time()},
              %{model: perplexity.model, response: parsed_data}
            )

            {:ok, [parsed_data]}
        end
      rescue
        err in LangChainError ->
          {:error, err}
      end
    end)
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
      {:ok, %Req.Response{body: data} = response} ->
        Callbacks.fire(perplexity.callbacks, :on_llm_response_headers, [response.headers])

        Callbacks.fire(perplexity.callbacks, :on_llm_token_usage, [
          get_token_usage(data)
        ])

        case do_process_response(perplexity, data, tools) do
          {:error, %LangChainError{} = reason} ->
            Logger.error("Error processing response: #{inspect(reason)}")
            {:error, reason}

          result ->
            Callbacks.fire(perplexity.callbacks, :on_llm_new_message, [result])

            # Track non-streaming response completion
            Telemetry.emit_event(
              [:langchain, :llm, :response, :non_streaming],
              %{system_time: System.system_time()},
              %{
                model: perplexity.model,
                response_size: byte_size(inspect(result))
              }
            )

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
        Utils.handle_stream_fn(
          perplexity,
          &decode_stream/1,
          &process_stream_chunk(perplexity, &1)
        )
    )
    |> case do
      {:ok, response} ->
        Callbacks.fire(perplexity.callbacks, :on_llm_response_headers, [response.headers])

        {:ok, response}

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
  @spec do_process_response(t(), map(), ChatModel.tools()) ::
          Message.t() | {:error, LangChainError.t()}
  def do_process_response(model, %{"choices" => [choice | _]} = data, tools) do
    # Fire token usage callback if present
    if usage = Map.get(data, "usage") do
      case get_token_usage(%{"usage" => usage}) do
        %TokenUsage{} = token_usage ->
          Callbacks.fire(model.callbacks, :on_llm_token_usage, [token_usage])

        nil ->
          :ok
      end
    end

    # Process the first choice and add full response metadata
    case do_process_response(model, choice, tools) do
      {:error, _} = error ->
        error

      message ->
        # Add the full API response data to processed_content
        full_response_data = %{
          id: Map.get(data, "id"),
          model: Map.get(data, "model"),
          created: Map.get(data, "created"),
          usage: Map.get(data, "usage"),
          citations: Map.get(data, "citations"),
          search_results: Map.get(data, "search_results")
        }

        %{message | processed_content: full_response_data}
    end
  end

  def do_process_response(
        _model,
        %{"finish_reason" => finish_reason, "message" => %{"content" => content, "role" => role}} =
          data,
        tools
      )
      when tools != [] do
    status = finish_reason_to_status(finish_reason)

    # Try to parse content as JSON since we expect structured output
    case Jason.decode(content) do
      {:ok, _parsed} ->
        # Create a tool call from the parsed content
        tool_call =
          ToolCall.new!(%{
            type: :function,
            status: :complete,
            name: List.first(tools).name,
            # Keep the original JSON string
            arguments: content,
            call_id: Ecto.UUID.generate()
          })

        case Message.new(%{
               "role" => role,
               "content" => nil,
               "status" => status,
               "index" => data["index"],
               "tool_calls" => [tool_call]
             }) do
          {:ok, message} -> message
          {:error, changeset} -> {:error, LangChainError.exception(changeset)}
        end

      {:error, _} ->
        {:error,
         LangChainError.exception(
           type: "bad_request",
           message: "response_format: expected JSON structured output but got: #{content}"
         )}
    end
  end

  def do_process_response(
        _model,
        %{"finish_reason" => finish_reason, "message" => %{"content" => content, "role" => role}} =
          data,
        tools
      )
      when tools == [] do
    status = finish_reason_to_status(finish_reason)

    case Message.new(%{
           "role" => role,
           "content" => content,
           "status" => status,
           "index" => data["index"]
         }) do
      {:ok, message} -> message
      {:error, changeset} -> {:error, LangChainError.exception(changeset)}
    end
  end

  def do_process_response(_model, %{"error" => %{"message" => reason, "type" => type}}, _tools) do
    Logger.error("Received error from API: #{inspect(reason)}")
    {:error, LangChainError.exception(type: type, message: reason)}
  end

  def do_process_response(_model, %{"error" => %{"message" => reason}}, _tools) do
    Logger.error("Received error from API: #{inspect(reason)}")
    {:error, LangChainError.exception(message: reason)}
  end

  def do_process_response(model, %{"choices" => %{} = usage} = _data, _tools) do
    case get_token_usage(%{"usage" => usage}) do
      %TokenUsage{} = token_usage ->
        Callbacks.fire(model.callbacks, :on_llm_token_usage, [token_usage])
        :skip

      nil ->
        :skip
    end
  end

  def do_process_response(_model, %{"choices" => []}, _tools), do: :skip

  def do_process_response(model, %{"choices" => choices} = data, tools) when is_list(choices) do
    # Fire token usage callback if present
    if usage = Map.get(data, "usage") do
      case get_token_usage(%{"usage" => usage}) do
        %TokenUsage{} = token_usage ->
          Callbacks.fire(model.callbacks, :on_llm_token_usage, [token_usage])

        nil ->
          :ok
      end
    end

    # Process each response individually and add full response metadata
    full_response_data = %{
      id: Map.get(data, "id"),
      model: Map.get(data, "model"),
      created: Map.get(data, "created"),
      usage: Map.get(data, "usage"),
      citations: Map.get(data, "citations"),
      search_results: Map.get(data, "search_results")
    }

    for choice <- choices do
      case do_process_response(model, choice, tools) do
        {:error, _} = error ->
          error

        message ->
          %{message | processed_content: full_response_data}
      end
    end
  end

  def do_process_response(
        _model,
        %{"finish_reason" => finish_reason, "message" => %{"content" => content}} = data,
        _tools
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
        %{
          "choices" => [
            %{
              "delta" => %{"role" => role, "content" => content},
              "finish_reason" => finish,
              "index" => index
            } = _choice
          ]
        },
        _tools
      ) do
    status = finish_reason_to_status(finish)

    data =
      %{}
      |> Map.put("role", role)
      |> Map.put("content", content)
      |> Map.put("index", index)
      |> Map.put("status", status)

    case MessageDelta.new(data) do
      {:ok, message} ->
        send(self(), {:message_delta, message})
        message

      {:error, %Ecto.Changeset{} = changeset} ->
        {:error, LangChainError.exception(changeset)}
    end
  end

  def do_process_response(
        _model,
        %{
          "choices" => [
            %{
              "delta" => %{"content" => content},
              "finish_reason" => finish,
              "index" => index
            }
            | _
          ]
        },
        _tools
      ) do
    status = finish_reason_to_status(finish)

    data =
      %{}
      |> Map.put("role", "assistant")
      |> Map.put("content", content)
      |> Map.put("index", index)
      |> Map.put("status", status)

    case MessageDelta.new(data) do
      {:ok, message} ->
        send(self(), {:message_delta, message})
        message

      {:error, %Ecto.Changeset{} = changeset} ->
        {:error, LangChainError.exception(changeset)}
    end
  end

  def do_process_response(_model, %{"choices" => []} = _msg, _tools), do: :skip

  def do_process_response(_model, {:error, %Jason.DecodeError{} = response}, _tools) do
    error_message = "Received invalid JSON: #{inspect(response)}"
    Logger.error(error_message)

    {:error,
     LangChainError.exception(type: "invalid_json", message: error_message, original: response)}
  end

  def do_process_response(_model, other, _tools) do
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

  defp process_stream_chunk(_perplexity, %{
         "choices" => [
           %{
             "delta" => %{"content" => content},
             "finish_reason" => finish_reason
           }
           | _
         ]
       })
       when finish_reason in [nil, "stop"] do
    delta = %MessageDelta{content: content, role: :assistant, status: :complete}
    send(self(), {:message_delta, delta})
    {:cont, delta}
  end

  defp process_stream_chunk(_perplexity, %{
         "choices" => [
           %{"delta" => %{"content" => content}} | _
         ]
       }) do
    delta = %MessageDelta{content: content, role: :assistant}
    send(self(), {:message_delta, delta})
    {:cont, delta}
  end

  defp process_stream_chunk(_perplexity, %{
         "choices" => [
           %{"finish_reason" => finish_reason} | _
         ]
       })
       when finish_reason in [nil, "stop"] do
    delta = %MessageDelta{status: :complete}
    send(self(), {:message_delta, delta})
    {:cont, delta}
  end

  defp process_stream_chunk(_perplexity, %{
         "choices" => [
           %{"message" => message} | _
         ]
       }) do
    send(self(), {:message, message})
    {:cont, message}
  end

  defp process_stream_chunk(_perplexity, _chunk) do
    {:cont, %MessageDelta{}}
  end
end
