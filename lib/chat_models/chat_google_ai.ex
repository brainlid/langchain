defmodule LangChain.ChatModels.ChatGoogleAI do
  @moduledoc """
  Parses and validates inputs for making a request for the Google AI  Chat API.

  Converts response into more specialized `LangChain` data structures.

  **NOTE:** The GoogleAI service is unique in how it reports TokenUsage
  information. So far, it's the only API that returns TokenUsage for each
  returned delta, where the generated token count is incremented with one. Other
  services return the total TokenUsage data at the end. This Chat model fires
  the callback each time it is received.

  **Google Search Integration**

  Starting with Gemini 2.0, this module supports Google Search as a native tool,
  allowing the model to automatically search the web for recent information to ground
  its responses and improve factuality. Check out the [Google AI Documentation](https://ai.google.dev/gemini-api/docs/grounding?lang=rest)
  for more information.

  Example Usage:

  ```elixir
  alias LangChain.Chains.LLMChain
  alias LangChain.Message
  alias LangChain.NativeTool

  model = ChatGoogleAI.new!(%{temperature: 0, stream: false, model: "gemini-2.0-flash"})

  {:ok, updated_chain} =
     %{llm: model, verbose: false, stream: false}
     |> LLMChain.new!()
     |> LLMChain.add_message(
       Message.new_user!("What is the current Google stock price?")
     )
     |> LLMChain.add_tools(NativeTool.new!(%{name: "google_search", configuration: %{}}))
     |> LLMChain.run()
  ```

  The above call will return the current Google stock price.

  When `google_search` is used, the model will also return grounding information in the metadata attribute of the assistant message.
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
  alias LangChain.Function
  alias LangChain.TokenUsage
  alias LangChain.LangChainError
  alias LangChain.Utils
  alias LangChain.Callbacks
  alias LangChain.NativeTool

  @behaviour ChatModel

  @current_config_version 1

  @default_base_url "https://generativelanguage.googleapis.com"
  @default_api_version "v1beta"
  @default_endpoint @default_base_url

  # allow up to 2 minutes for response.
  @receive_timeout 60_000

  @primary_key false
  embedded_schema do
    field :endpoint, :string, default: @default_endpoint

    # The version of the API to use.
    field :api_version, :string, default: @default_api_version
    field :model, :string, default: "gemini-2.5-pro"
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
    field :json_response, :boolean, default: false
    field :json_schema, :map, default: nil
    field :stream, :boolean, default: false

    # The safety settings for the model, specified as a list of maps. Each map
    # should contain a `category` and a `threshold` for that category.
    # e.g. [%{"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_ONLY_HIGH"}]
    # see https://ai.google.dev/api/generate-content#v1beta.SafetySetting
    # for the list of categories and thresholds
    field :safety_settings, {:array, :map}, default: []

    # A list of maps for callback handlers (treat as private)
    field :callbacks, {:array, :map}, default: []

    # Additional level of raw api request and response data
    field :verbose_api, :boolean, default: false
  end

  @type t :: %ChatGoogleAI{}

  @create_fields [
    :endpoint,
    :api_version,
    :model,
    :api_key,
    :temperature,
    :top_p,
    :top_k,
    :receive_timeout,
    :json_response,
    :json_schema,
    :stream,
    :safety_settings
  ]
  @required_fields [
    :endpoint,
    :api_version,
    :model
  ]

  @spec get_api_key(t) :: String.t()
  defp get_api_key(%ChatGoogleAI{api_key: api_key}) do
    # if no API key is set default to `""` which will raise an API error
    api_key || Config.resolve(:google_ai_key, "")
  end

  @doc """
  Setup a ChatGoogleAI client configuration.
  """
  @spec new(attrs :: map()) :: {:ok, t} | {:error, Ecto.Changeset.t()}
  def new(%{} = attrs \\ %{}) do
    %ChatGoogleAI{}
    |> cast(attrs, @create_fields)
    |> common_validation()
    |> apply_action(:insert)
  end

  @doc """
  Setup a ChatGoogleAI client configuration and return it or raise an error if invalid.
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

  def for_api(%ChatGoogleAI{} = google_ai, messages, functions) do
    {system, messages} =
      Utils.split_system_message(messages, "Google AI only supports a single System message")

    system_instruction =
      case system do
        nil ->
          nil

        %Message{role: :system, content: content} when is_binary(content) ->
          %{"parts" => [%{"text" => content}]}

        %Message{role: :system, content: content} when is_list(content) ->
          # Extract text from ContentPart structures
          text_content =
            content
            |> Enum.filter(&match?(%ContentPart{type: :text}, &1))
            |> Enum.map(& &1.content)
            |> Enum.join(" ")

          %{"parts" => [%{"text" => text_content}]}
      end

    messages_for_api =
      messages
      |> Enum.map(&for_api/1)
      |> List.flatten()
      |> List.wrap()

    {response_mime_type, response_schema} =
      case google_ai.json_response do
        true ->
          {"application/json", google_ai.json_schema}

        false ->
          {nil, nil}
      end

    generation_config_params =
      %{
        "temperature" => google_ai.temperature,
        "topP" => google_ai.top_p,
        "topK" => google_ai.top_k
      }
      |> Utils.conditionally_add_to_map("response_mime_type", response_mime_type)
      |> Utils.conditionally_add_to_map("response_schema", response_schema)

    req =
      %{
        "contents" => messages_for_api,
        "generationConfig" => generation_config_params
      }
      |> Utils.conditionally_add_to_map("system_instruction", system_instruction)
      |> Utils.conditionally_add_to_map("safetySettings", google_ai.safety_settings)

    if functions && not Enum.empty?(functions) do
      native_tools = Enum.filter(functions, &match?(%NativeTool{}, &1))
      function_tools = Enum.filter(functions, &match?(%Function{}, &1))

      tools_array = []

      tools_array =
        if function_tools != [] do
          tools_array ++ [%{"functionDeclarations" => Enum.map(function_tools, &for_api/1)}]
        else
          tools_array
        end

      tools_array =
        if native_tools != [] do
          tools_array ++ Enum.map(native_tools, &for_api/1)
        else
          tools_array
        end

      Map.put(req, "tools", tools_array)
    else
      req
    end
  end

  @doc false
  def for_api(%Message{role: :assistant} = message) do
    content_parts = get_message_contents(message) || []
    tool_calls = Enum.map(message.tool_calls || [], &for_api/1)

    %{
      "role" => map_role(:assistant),
      "parts" => content_parts ++ tool_calls
    }
  end

  def for_api(%Message{role: :tool} = message) do
    %{
      "role" => map_role(:tool),
      "parts" => Enum.map(message.tool_results, &for_api/1)
    }
  end

  def for_api(%Message{content: content} = message) when is_binary(content) do
    %{
      "role" => map_role(message.role),
      "parts" => [%{"text" => message.content}]
    }
  end

  def for_api(%Message{content: content} = message) when is_list(content) do
    %{
      "role" => message.role,
      "parts" => Enum.map(content, &for_api/1)
    }
  end

  def for_api(%Message{content: content} = message) when is_list(content) do
    %{
      "role" => message.role,
      "parts" => Enum.map(content, &for_api/1)
    }
  end

  def for_api(%ContentPart{type: :text} = part) do
    %{"text" => part.content}
  end

  def for_api(%ContentPart{type: :file_url} = part) do
    %{
      "file_data" => %{
        "mime_type" => part.options[:media],
        "file_uri" => part.content
      }
    }
  end

  # Supported image types: png, jpeg, webp, heic, heif: https://ai.google.dev/gemini-api/docs/vision?lang=rest#technical-details-image
  def for_api(%ContentPart{type: :image} = part) do
    mime_type =
      case Keyword.get(part.options || [], :media, nil) do
        :png ->
          "image/png"

        type when type in [:jpeg, :jpg] ->
          "image/jpeg"

        :webp ->
          "image/webp"

        :heic ->
          "image/heic"

        :heif ->
          "image/heif"

        type when is_binary(type) ->
          "image/type"

        other ->
          message = "Received unsupported media type for ContentPart: #{inspect(other)}"
          Logger.error(message)
          raise LangChainError, message
      end

    %{
      "inline_data" => %{
        "mime_type" => mime_type,
        "data" => part.content
      }
    }
  end

  def for_api(%ToolCall{} = call) do
    %{
      "functionCall" => %{
        "args" => call.arguments,
        "name" => call.name
      }
    }
  end

  def for_api(%ToolResult{} = result) do
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

    # There is no explanation for why they want it nested like this. Odd.
    #
    # https://ai.google.dev/gemini-api/docs/function-calling#expandable-7
    %{
      "functionResponse" => %{
        "name" => result.name,
        "response" => %{
          "name" => result.name,
          "content" => content
        }
      }
    }
  end

  def for_api(%Function{} = function) do
    encoded =
      %{
        "name" => function.name,
        "parameters" => ChatOpenAI.get_parameters(function)
      }
      |> Utils.conditionally_add_to_map("description", function.description)

    # For functions with no parameters, Google AI needs the parameters field removing, otherwise it will error
    # with "* GenerateContentRequest.tools[0].function_declarations[0].parameters.properties: should be non-empty for OBJECT type\n"
    if encoded["parameters"] == %{"properties" => %{}, "type" => "object"} do
      Map.delete(encoded, "parameters")
    else
      encoded
    end
  end

  def for_api(%NativeTool{name: name, configuration: %{} = config}) do
    %{name => config}
  end

  def for_api(%NativeTool{name: name, configuration: nil}) do
    name
  end

  @doc """
  Calls the Google AI API passing the ChatGoogleAI struct with configuration, plus
  either a simple message or the list of messages to act as the prompt.

  Optionally pass in a list of tools available to the LLM for requesting
  execution in response.

  Optionally pass in a callback function that can be executed as data is
  received from the API.

  **NOTE:** This function *can* be used directly, but the primary interface
  should be through `LangChain.Chains.LLMChain`. The `ChatGoogleAI` module is more focused on
  translating the `LangChain` data structures to and from the Google AI API.

  Another benefit of using `LangChain.Chains.LLMChain` is that it combines the
  storage of messages, adding tools, adding custom context that should be
  passed to tools, and automatically applying `LangChain.MessageDelta`
  structs as they are are received, then converting those to the full
  `LangChain.Message` once fully complete.
  """
  @impl ChatModel
  def call(google_ai, prompt, tools \\ [])

  def call(%ChatGoogleAI{} = google_ai, prompt, tools) when is_binary(prompt) do
    messages = [
      Message.new_system!(),
      Message.new_user!(prompt)
    ]

    call(google_ai, messages, tools)
  end

  def call(%ChatGoogleAI{} = google_ai, messages, tools)
      when is_list(messages) do
    metadata = %{
      model: google_ai.model,
      message_count: length(messages),
      tools_count: length(tools)
    }

    LangChain.Telemetry.span([:langchain, :llm, :call], metadata, fn ->
      try do
        # Track the prompt being sent
        LangChain.Telemetry.llm_prompt(
          %{system_time: System.system_time()},
          %{model: google_ai.model, messages: messages}
        )

        case do_api_request(google_ai, messages, tools) do
          {:error, reason} ->
            {:error, reason}

          parsed_data ->
            # Track the response being received
            LangChain.Telemetry.llm_response(
              %{system_time: System.system_time()},
              %{model: google_ai.model, response: parsed_data}
            )

            {:ok, parsed_data}
        end
      rescue
        err in LangChainError ->
          {:error, err.message}
      end
    end)
  end

  @doc false
  @spec do_api_request(t(), [Message.t()], [Function.t()]) ::
          list() | struct() | {:error, LangChainError.t()}
  def do_api_request(%ChatGoogleAI{stream: false} = google_ai, messages, tools) do
    req =
      Req.new(
        url: build_url(google_ai),
        json: for_api(google_ai, messages, tools),
        receive_timeout: google_ai.receive_timeout,
        retry: :transient,
        max_retries: 3,
        retry_delay: fn attempt -> 300 * attempt end
      )

    req
    |> Req.post()
    |> case do
      {:ok, %Req.Response{status: 200, body: data}} ->
        case do_process_response(google_ai, data) do
          {:error, reason} ->
            {:error, reason}

          result ->
            # Track non-streaming response completion
            LangChain.Telemetry.emit_event(
              [:langchain, :llm, :response, streaming: false],
              %{system_time: System.system_time()},
              %{
                model: google_ai.model,
                response_size: byte_size(inspect(result))
              }
            )

            Callbacks.fire(google_ai.callbacks, :on_llm_new_message, [result])
            result
        end

      {:ok, %Req.Response{status: status} = err} ->
        {:error,
         LangChainError.exception(
           message: "Failed with status: #{inspect(status)}",
           original: err
         )}

      {:error, %Req.TransportError{reason: :timeout} = err} ->
        {:error,
         LangChainError.exception(type: "timeout", message: "Request timed out", original: err)}

      other ->
        Logger.error("Unexpected and unhandled API response! #{inspect(other)}")
        other
    end
  end

  def do_api_request(%ChatGoogleAI{stream: true} = google_ai, messages, tools) do
    Req.new(
      url: build_url(google_ai),
      json: for_api(google_ai, messages, tools),
      receive_timeout: google_ai.receive_timeout
    )
    |> Req.Request.put_header("accept-encoding", "utf-8")
    |> Req.post(
      into:
        Utils.handle_stream_fn(
          google_ai,
          &ChatOpenAI.decode_stream/1,
          &do_process_response(google_ai, &1, MessageDelta)
        )
    )
    |> case do
      {:ok, %Req.Response{status: 200, body: data}} ->
        # Google AI uses `finishReason: "STOP` for all messages in the stream.
        # This field can't be used to terminate the list of deltas, so simulate
        # this behavior by forcing the final delta to have `status: :complete`.
        complete_final_delta(data)

      {:ok, %Req.Response{status: status} = err} ->
        {:error,
         LangChainError.exception(
           message: "Failed with status: #{inspect(status)}",
           original: err
         )}

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

  @doc false
  @spec build_url(t()) :: String.t()
  def build_url(
        %ChatGoogleAI{endpoint: endpoint, api_version: api_version, model: model} = google_ai
      ) do
    "#{endpoint}/#{api_version}/models/#{model}:#{get_action(google_ai)}?key=#{get_api_key(google_ai)}"
    |> use_sse(google_ai)
  end

  @spec use_sse(String.t(), t()) :: String.t()
  defp use_sse(url, %ChatGoogleAI{stream: true}), do: url <> "&alt=sse"
  defp use_sse(url, _model), do: url

  @spec get_action(t()) :: String.t()
  defp get_action(%ChatGoogleAI{stream: false}), do: "generateContent"
  defp get_action(%ChatGoogleAI{stream: true}), do: "streamGenerateContent"

  def complete_final_delta(data) when is_list(data) do
    update_in(data, [Access.at(-1), Access.at(-1)], &%{&1 | status: :complete})
  end

  def do_process_response(model, response, message_type \\ Message)

  def do_process_response(model, %{"candidates" => candidates} = data, message_type)
      when is_list(candidates) do
    # Google is odd in that it returns token usage for each MessageDelta as it
    # goes, incrementing the number of generated tokens. I haven't seen anyone
    # else do this. For now, we fire each and every TokenUsage we receive.
    token_usage = get_token_usage(data)

    case token_usage do
      %TokenUsage{} = usage ->
        Callbacks.fire(model.callbacks, :on_llm_token_usage, [usage])
        :ok

      nil ->
        :ok
    end

    candidates
    |> Enum.map(&do_process_response(model, &1, message_type))
    |> Enum.map(&TokenUsage.set(&1, token_usage))
  end

  # Function Call in a Message
  def do_process_response(
        model,
        %{"content" => %{"parts" => parts} = content_data} = data,
        Message
      ) do
    text_part =
      parts
      |> filter_parts_for_types(["text"])
      |> filter_text_parts()
      |> Enum.map(fn part ->
        ContentPart.new!(%{type: :text, content: part["text"]})
      end)

    tool_calls_from_parts =
      parts
      |> filter_parts_for_types(["functionCall"])
      |> Enum.map(fn part ->
        do_process_response(model, part, nil)
      end)

    tool_result_from_parts =
      parts
      |> filter_parts_for_types(["functionResponse"])
      |> Enum.map(fn part ->
        do_process_response(model, part, nil)
      end)

    %{
      role: unmap_role(content_data["role"]),
      content: text_part,
      complete: true,
      index: data["index"],
      metadata: if(data["groundingMetadata"], do: data["groundingMetadata"], else: nil)
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

  # Function Call in a MessageDelta
  def do_process_response(
        model,
        %{"content" => %{"parts" => parts} = content_data} = data,
        MessageDelta
      ) do
    text_content =
      case parts do
        [%{"text" => text}] ->
          text

        _other ->
          nil
      end

    tool_calls_from_parts =
      parts
      |> filter_parts_for_types(["functionCall"])
      |> Enum.map(fn part ->
        do_process_response(model, part, nil)
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

  def do_process_response(
        _model,
        %{"functionCall" => %{"args" => raw_args, "name" => name}} = data,
        _
      ) do
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
        _model,
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
          finish_reason_to_status(finish)
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

  def do_process_response(_model, %{"error" => %{"message" => reason}} = response, _) do
    Logger.error("Received error from API: #{inspect(reason)}")
    {:error, LangChainError.exception(message: reason, original: response)}
  end

  def do_process_response(_model, {:error, %Jason.DecodeError{} = response}, _) do
    error_message = "Received invalid JSON: #{inspect(response)}"
    Logger.error(error_message)

    {:error,
     LangChainError.exception(type: "invalid_json", message: error_message, original: response)}
  end

  def do_process_response(_model, other, _) do
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

  @doc false
  def filter_text_parts(parts) when is_list(parts) do
    Enum.filter(parts, fn p ->
      case p do
        %{"text" => text} -> text && text != ""
        _ -> false
      end
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
  def serialize_config(%ChatGoogleAI{} = model) do
    Utils.to_serializable_map(
      model,
      [
        :endpoint,
        :model,
        :api_version,
        :temperature,
        :top_p,
        :top_k,
        :receive_timeout,
        :json_response,
        :json_schema,
        :stream,
        :safety_settings
      ],
      @current_config_version
    )
  end

  @doc """
  Restores the model from the config.
  """
  @impl ChatModel
  def restore_from_map(%{"version" => 1} = data) do
    ChatGoogleAI.new(data)
  end

  defp get_token_usage(%{"usageMetadata" => usage} = _response_body) do
    # extract out the reported response token usage
    TokenUsage.new!(%{
      input: Map.get(usage, "promptTokenCount", 0),
      output: Map.get(usage, "candidatesTokenCount", 0),
      raw: usage
    })
  end

  defp get_token_usage(_response_body), do: nil

  # A full list of finish reasons and their meanings can be found here:
  # https://ai.google.dev/api/generate-content#FinishReason
  defp finish_reason_to_status("STOP"), do: :complete
  defp finish_reason_to_status("SAFETY"), do: :complete
  defp finish_reason_to_status("MAX_TOKENS"), do: :length
  defp finish_reason_to_status("RECITATION"), do: :complete
  defp finish_reason_to_status("LANGUAGE"), do: :complete
  defp finish_reason_to_status("OTHER"), do: :complete
  defp finish_reason_to_status("BLOCKLIST"), do: :complete
  defp finish_reason_to_status("PROHIBITED_CONTENT"), do: :complete
  defp finish_reason_to_status("SPII"), do: :complete
  defp finish_reason_to_status("MALFORMED_FUNCTION_CALL"), do: :complete

  defp finish_reason_to_status(other) do
    Logger.warning("Unsupported finishReason in response. Reason: #{inspect(other)}")
    nil
  end
end
