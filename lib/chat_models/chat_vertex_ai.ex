defmodule LangChain.ChatModels.ChatVertexAI do
  @moduledoc """
  Parses and validates inputs for making a request for the Google AI  Chat API.

  Converts response into more specialized `LangChain` data structures.

  Example Usage:

  ```elixir
  alias LangChain.Chains.LLMChain
  alias LangChain.Message
  alias LangChain.Message.ContentPart
  alias LangChain.ChatModels.ChatVertexAI


  config = %{
        model: "gemini-2.0-flash",
        api_key: ..., # vertex requires gcloud auth token https://cloud.google.com/vertex-ai/generative-ai/docs/start/quickstarts/quickstart-multimodal#rest
        temperature: 1.0,
        top_p: 0.8,
        receive_timeout: ...
      }
   model = ChatVertexAI.new!(config)

      %{llm: model, verbose: false, stream: false}
      |> LLMChain.new!()
      |> LLMChain.add_message(
        Message.new_user!([
          ContentPart.new!(%{type: :text, content: "Analyse the provided file and share a summary"}),
          ContentPart.new!(%{
            type: :file_url,
            content: ...,
            options: [media: ...]
          })
        ])
      )
      |> LLMChain.run()
  The above call will return summary of the media content.
  ```
  """
  use Ecto.Schema
  require Logger
  import Ecto.Changeset
  alias __MODULE__
  alias LangChain.Config
  alias LangChain.ChatModels.ChatModel
  alias LangChain.ChatModels.ChatOpenAI
  alias LangChain.Message
  alias LangChain.MessageDelta
  alias LangChain.Message.ContentPart
  alias LangChain.Message.ToolCall
  alias LangChain.Message.ToolResult
  alias LangChain.LangChainError
  alias LangChain.Utils
  alias LangChain.Callbacks

  @behaviour ChatModel

  @current_config_version 1

  # allow up to 2 minutes for response.
  @receive_timeout 60_000

  @primary_key false
  embedded_schema do
    field :endpoint, :string

    field :model, :string, default: "gemini-pro"
    field :api_key, :string, redact: true

    # What sampling temperature to use, between 0 and 2. Higher values like 0.8
    # will make the output more random, while lower values like 0.2 will make it
    # more focused and deterministic.
    field :temperature, :float, default: 0.9

    # The topP parameter changes how the model selects tokens for output. Tokens
    # are selected from the most to least probable until the sum of their
    # probabilities equals the topP value. For example, if tokens A, B, and C have
    # a probability of 0.3, 0.2, and 0.1 and the topP value is 0.5, then the model
    # will select either A or B as the next token by using the temperature and exclude
    # C as a candidate. The default topP value is 0.95.
    field :top_p, :float, default: 1.0

    # The topK parameter changes how the model selects tokens for output. A topK of
    # 1 means the selected token is the most probable among all the tokens in the
    # model's vocabulary (also called greedy decoding), while a topK of 3 means that
    # the next token is selected from among the 3 most probable using the temperature.
    # For each token selection step, the topK tokens with the highest probabilities
    # are sampled. Tokens are then further filtered based on topP with the final token
    # selected using temperature sampling.
    field :top_k, :float, default: 1.0

    # Duration in seconds for the response to be received. When streaming a very
    # lengthy response, a longer time limit may be required. However, when it
    # goes on too long by itself, it tends to hallucinate more.
    field :receive_timeout, :integer, default: @receive_timeout

    field :stream, :boolean, default: false
    field :json_response, :boolean, default: false

    # A list of maps for callback handlers (treated as internal)
    field :callbacks, {:array, :map}, default: []

    # Additional level of raw api request and response data
    field :verbose_api, :boolean, default: false
  end

  @type t :: %ChatVertexAI{}

  @create_fields [
    :endpoint,
    :model,
    :api_key,
    :temperature,
    :top_p,
    :top_k,
    :receive_timeout,
    :stream,
    :json_response
  ]
  @required_fields [
    :endpoint,
    :model
  ]

  @spec get_api_key(t) :: String.t()
  defp get_api_key(%ChatVertexAI{api_key: api_key}) do
    # if no API key is set default to `""` which will raise an API error
    api_key || Config.resolve(:vertex_ai_key, "")
  end

  @doc """
  Setup a ChatVertexAI client configuration.
  """
  @spec new(attrs :: map()) :: {:ok, t} | {:error, Ecto.Changeset.t()}
  def new(%{} = attrs \\ %{}) do
    %ChatVertexAI{}
    |> cast(attrs, @create_fields)
    |> common_validation()
    |> apply_action(:insert)
  end

  @doc """
  Setup a ChatVertexAI client configuration and return it or raise an error if invalid.
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
  end

  def for_api(%ChatVertexAI{} = vertex_ai, messages, functions) do
    {sys_instructions, other_messages} = Utils.split_system_message(messages)

    messages_for_api =
      other_messages
      |> Enum.map(&for_api/1)
      |> List.flatten()
      |> List.wrap()

    req =
      %{
        "contents" => messages_for_api,
        "generationConfig" => %{
          "temperature" => vertex_ai.temperature,
          "topP" => vertex_ai.top_p,
          "topK" => vertex_ai.top_k
        }
      }
      |> Utils.conditionally_add_to_map("system_instruction", for_api(sys_instructions))

    req =
      if vertex_ai.json_response do
        req
        |> put_in(["generationConfig", "response_mime_type"], "application/json")
      else
        req
      end

    if functions && not Enum.empty?(functions) do
      req
      |> Map.put("tools", [
        %{
          # Google AI functions use an OpenAI compatible format.
          # See: https://ai.google.dev/docs/function_calling#how_it_works
          "functionDeclarations" => Enum.map(functions, &ChatOpenAI.for_api(vertex_ai, &1))
        }
      ])
    else
      req
    end
  end

  defp for_api(%Message{role: :assistant} = message) do
    content_parts = get_message_contents(message) || []
    tool_calls = Enum.map(message.tool_calls || [], &for_api/1)

    %{
      "role" => map_role(:assistant),
      "parts" => content_parts ++ tool_calls
    }
  end

  defp for_api(%Message{role: :tool} = message) do
    %{
      "role" => map_role(:tool),
      "parts" => Enum.map(message.tool_results, &for_api/1)
    }
  end

  defp for_api(%Message{role: :system} = message) do
    # System messages should return a single text part, not a list
    case get_message_contents(message) do
      [%{"text" => text}] -> %{"parts" => %{"text" => text}}
      _ -> %{"parts" => %{"text" => message.content}}
    end
  end

  defp for_api(%Message{role: :user, content: content}) when is_list(content) do
    %{
      "role" => map_role(:user),
      "parts" => Enum.map(content, &for_api(&1))
    }
  end

  defp for_api(%Message{} = message) do
    content_parts = get_message_contents(message) || []

    %{
      "role" => map_role(message.role),
      "parts" => content_parts
    }
  end

  defp for_api(%ContentPart{type: :text} = part) do
    %{"text" => part.content}
  end

  defp for_api(%ContentPart{type: :image} = part) do
    %{
      "inlineData" => %{
        "mimeType" => Keyword.fetch!(part.options, :media),
        "data" => part.content
      }
    }
  end

  defp for_api(%ContentPart{type: :image_url} = part) do
    %{
      "fileData" => %{
        "mimeType" => Keyword.fetch!(part.options, :media),
        "fileUri" => part.content
      }
    }
  end

  defp for_api(%ContentPart{type: :file_url} = part) do
    %{
      "fileData" => %{
        "mimeType" => Keyword.fetch!(part.options, :media),
        "fileUri" => part.content
      }
    }
  end

  defp for_api(%ToolCall{} = call) do
    %{
      "functionCall" => %{
        "args" => call.arguments,
        "name" => call.name
      }
    }
  end

  defp for_api(%ToolResult{} = result) do
    content =
      result.content
      |> ContentPart.parts_to_string()
      |> Jason.decode()
      |> case do
        {:ok, data} ->
          # content was converted through JSON
          data

        {:error, %Jason.DecodeError{}} ->
          # assume the result is intended to be a string and return it as-is
          %{"result" => result.content}
      end

    %{
      "functionResponse" => %{
        "name" => result.name,
        "response" => content
      }
    }
  end

  defp for_api(nil), do: nil

  @doc """
  Calls the Google AI API passing the ChatVertexAI struct with configuration,
  plus either a simple message or the list of messages to act as the prompt.

  Optionally pass in a list of tools available to the LLM for requesting
  execution in response.

  **NOTE:** This function *can* be used directly, but the primary interface
  should be through `LangChain.Chains.LLMChain`. The `ChatVertexAI` module is
  more focused on translating the `LangChain` data structures to and from the
  OpenAI API.

  Another benefit of using `LangChain.Chains.LLMChain` is that it combines the
  storage of messages, adding tools, adding custom context that should be passed
  to tools, and automatically applying `LangChain.MessageDelta` structs as they
  are are received, then converting those to the full `LangChain.Message` once
  fully complete.
  """
  @impl ChatModel
  def call(openai, prompt, tools \\ [])

  def call(%ChatVertexAI{} = vertex_ai, prompt, tools) when is_binary(prompt) do
    messages = [
      Message.new_system!(),
      Message.new_user!(prompt)
    ]

    call(vertex_ai, messages, tools)
  end

  def call(%ChatVertexAI{} = vertex_ai, messages, tools)
      when is_list(messages) do
    metadata = %{
      model: vertex_ai.model,
      message_count: length(messages),
      tools_count: length(tools)
    }

    LangChain.Telemetry.span([:langchain, :llm, :call], metadata, fn ->
      try do
        # Track the prompt being sent
        LangChain.Telemetry.llm_prompt(
          %{system_time: System.system_time()},
          %{model: vertex_ai.model, messages: messages}
        )

        case do_api_request(vertex_ai, messages, tools) do
          {:error, reason} ->
            {:error, reason}

          parsed_data ->
            # Track the response being received
            LangChain.Telemetry.llm_response(
              %{system_time: System.system_time()},
              %{model: vertex_ai.model, response: parsed_data}
            )

            {:ok, parsed_data}
        end
      rescue
        err in LangChainError ->
          {:error, err}
      end
    end)
  end

  @doc false
  @spec do_api_request(t(), [Message.t()], [Function.t()]) ::
          list() | struct() | {:error, LangChainError.t()}
  def do_api_request(%ChatVertexAI{stream: false} = vertex_ai, messages, tools) do
    req =
      Req.new(
        url: build_url(vertex_ai),
        json: for_api(vertex_ai, messages, tools),
        receive_timeout: vertex_ai.receive_timeout,
        retry: :transient,
        max_retries: 3,
        auth: {:bearer, get_api_key(vertex_ai)},
        retry_delay: fn attempt -> 300 * attempt end
      )

    req
    |> Req.post()
    |> case do
      {:ok, %Req.Response{body: data}} ->
        case do_process_response(data) do
          {:error, reason} ->
            {:error, reason}

          result ->
            Callbacks.fire(vertex_ai.callbacks, :on_llm_new_message, [result])

            # Track non-streaming response completion
            LangChain.Telemetry.emit_event(
              [:langchain, :llm, :response, :non_streaming],
              %{system_time: System.system_time()},
              %{
                model: vertex_ai.model,
                response_size: byte_size(inspect(result))
              }
            )

            result
        end

      {:error, %Req.TransportError{reason: :timeout} = err} ->
        {:error,
         LangChainError.exception(type: "timeout", message: "Request timed out", original: err)}

      other ->
        Logger.error("Unexpected and unhandled API response! #{inspect(other)}")
        other
    end
  end

  def do_api_request(%ChatVertexAI{stream: true} = vertex_ai, messages, tools) do
    Req.new(
      url: build_url(vertex_ai),
      json: for_api(vertex_ai, messages, tools),
      auth: {:bearer, get_api_key(vertex_ai)},
      receive_timeout: vertex_ai.receive_timeout
    )
    |> Req.Request.put_header("accept-encoding", "utf-8")
    |> Req.post(
      into:
        Utils.handle_stream_fn(
          vertex_ai,
          &ChatOpenAI.decode_stream/1,
          &do_process_response(&1, MessageDelta)
        )
    )
    |> case do
      {:ok, %Req.Response{body: data}} ->
        # Google AI uses `finishReason: "STOP` for all messages in the stream.
        # This field can't be used to terminate the list of deltas, so simulate
        # this behavior by forcing the final delta to have `status: :complete`.
        complete_final_delta(data)

      {:error, %LangChainError{} = error} ->
        {:error, error}

      {:error, %Req.TransportError{reason: :timeout} = err} ->
        {:error,
         LangChainError.exception(type: "timeout", message: "Request timed out", original: err)}

      other ->
        Logger.error(
          "Unhandled and unexpected response from streamed post call. #{inspect(other)}"
        )

        {:error,
         LangChainError.exception(type: "unexpected_response", message: "Unexpected response")}
    end
  end

  @spec build_url(t()) :: String.t()
  defp build_url(%ChatVertexAI{endpoint: endpoint, model: model} = vertex_ai) do
    "#{endpoint}/models/#{model}:#{get_action(vertex_ai)}?key=#{get_api_key(vertex_ai)}"
    |> use_sse(vertex_ai)
  end

  @spec use_sse(String.t(), t()) :: String.t()
  defp use_sse(url, %ChatVertexAI{stream: true}), do: url <> "&alt=sse"
  defp use_sse(url, _model), do: url

  @spec get_action(t()) :: String.t()
  defp get_action(%ChatVertexAI{stream: false}), do: "generateContent"
  defp get_action(%ChatVertexAI{stream: true}), do: "streamGenerateContent"

  def complete_final_delta(data) when is_list(data) do
    update_in(data, [Access.at(-1), Access.at(-1)], &%{&1 | status: :complete})
  end

  def do_process_response(response, message_type \\ Message)

  def do_process_response(%{"candidates" => candidates}, message_type) when is_list(candidates) do
    candidates
    |> Enum.map(&do_process_response(&1, message_type))
  end

  def do_process_response(%{"content" => %{"parts" => parts} = content_data} = data, Message) do
    text_part =
      parts
      |> filter_parts_for_types(["text"])
      |> Enum.map(fn part ->
        ContentPart.new!(%{type: :text, content: part["text"]})
      end)

    tool_calls_from_parts =
      parts
      |> filter_parts_for_types(["functionCall"])
      |> Enum.map(fn part ->
        do_process_response(part, nil)
      end)

    tool_result_from_parts =
      parts
      |> filter_parts_for_types(["functionResponse"])
      |> Enum.map(fn part ->
        do_process_response(part, nil)
      end)

    %{
      role: unmap_role(content_data["role"]),
      content: text_part,
      complete: false,
      index: data["index"]
    }
    |> Utils.conditionally_add_to_map(:tool_calls, tool_calls_from_parts)
    |> Utils.conditionally_add_to_map(:tool_results, tool_result_from_parts)
    |> Message.new()
    |> case do
      {:ok, message} ->
        message

      {:error, %Ecto.Changeset{} = changeset} ->
        {:error, LangChainError.exception(changeset)}
    end
  end

  def do_process_response(%{"content" => %{"parts" => parts} = content_data} = data, MessageDelta) do
    text_content =
      case parts do
        [%{"text" => text}] ->
          text

        _other ->
          nil
      end

    parts
    |> filter_parts_for_types(["text"])
    |> Enum.map(fn part ->
      ContentPart.new!(%{type: :text, content: part["text"]})
    end)

    tool_calls_from_parts =
      parts
      |> filter_parts_for_types(["functionCall"])
      |> Enum.map(fn part ->
        do_process_response(part, nil)
      end)

    %{
      role: unmap_role(content_data["role"]),
      content: text_content,
      complete: true,
      index: data["index"]
    }
    |> Utils.conditionally_add_to_map(:tool_calls, tool_calls_from_parts)
    |> MessageDelta.new()
    |> case do
      {:ok, message} ->
        message

      {:error, %Ecto.Changeset{} = changeset} ->
        {:error, LangChainError.exception(changeset)}
    end
  end

  def do_process_response(%{"functionCall" => %{"args" => raw_args, "name" => name}} = data, _) do
    %{
      call_id: "call-#{name}",
      name: name,
      arguments: raw_args,
      complete: true,
      index: data["index"]
    }
    |> ToolCall.new()
    |> case do
      {:ok, message} ->
        message

      {:error, %Ecto.Changeset{} = changeset} ->
        {:error, LangChainError.exception(changeset)}
    end
  end

  def do_process_response(
        %{
          "finishReason" => finish,
          "content" => %{"parts" => parts, "role" => role},
          "index" => index
        },
        message_type
      )
      when is_list(parts) do
    status =
      case message_type do
        MessageDelta ->
          :incomplete

        Message ->
          case finish do
            "STOP" ->
              :complete

            "SAFETY" ->
              :complete

            other ->
              Logger.warning("Unsupported finishReason in response. Reason: #{inspect(other)}")
              nil
          end
      end

    content = Enum.map_join(parts, & &1["text"])

    case message_type.new(%{
           "content" => content,
           "role" => unmap_role(role),
           "status" => status,
           "index" => index
         }) do
      {:ok, message} ->
        message

      {:error, %Ecto.Changeset{} = changeset} ->
        {:error, LangChainError.exception(changeset)}
    end
  end

  def do_process_response(%{"error" => %{"message" => reason}} = response, _) do
    Logger.error("Received error from API: #{inspect(reason)}")
    {:error, LangChainError.exception(message: reason, original: response)}
  end

  def do_process_response({:error, %Jason.DecodeError{} = response}, _) do
    error_message = "Received invalid JSON: #{inspect(response)}"
    Logger.error(error_message)

    {:error,
     LangChainError.exception(type: "invalid_json", message: error_message, original: response)}
  end

  def do_process_response(other, _) do
    Logger.error("Trying to process an unexpected response. #{inspect(other)}")

    {:error,
     LangChainError.exception(
       type: "unexpected_response",
       message: "Unexpected response",
       original: other
     )}
  end

  @doc false
  def filter_parts_for_types(parts, types) when is_list(parts) and is_list(types) do
    Enum.filter(parts, fn p ->
      Enum.any?(types, &Map.has_key?(p, &1))
    end)
  end

  @doc """
  Return the content parts for the message.
  """
  @spec get_message_contents(MessageDelta.t() | Message.t()) :: [%{String.t() => any()}]
  def get_message_contents(%{content: content} = _message) when is_binary(content) do
    [%{"text" => content}]
  end

  def get_message_contents(%{content: contents} = _message) when is_list(contents) do
    Enum.map(contents, &for_api/1)
  end

  def get_message_contents(%{content: nil} = _message) do
    nil
  end

  defp map_role(role) do
    case role do
      :assistant -> :model
      :tool -> :function
      # System prompts are not supported yet. Google recommends using user prompt.
      :system -> :user
      role -> role
    end
  end

  defp unmap_role("model"), do: "assistant"
  defp unmap_role("function"), do: "tool"
  defp unmap_role(role), do: role

  @doc """
  Generate a config map that can later restore the model's configuration.
  """
  @impl ChatModel
  @spec serialize_config(t()) :: %{String.t() => any()}
  def serialize_config(%ChatVertexAI{} = model) do
    Utils.to_serializable_map(
      model,
      [
        :endpoint,
        :model,
        :temperature,
        :top_p,
        :top_k,
        :receive_timeout,
        :json_response,
        :stream
      ],
      @current_config_version
    )
  end

  @doc """
  Restores the model from the config.
  """
  @impl ChatModel
  def restore_from_map(%{"version" => 1} = data) do
    ChatVertexAI.new(data)
  end
end
