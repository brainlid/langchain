defmodule LangChain.ChatModels.ChatOllamaAI do
  @moduledoc """
  Represents the [Ollama AI Chat model](https://github.com/jmorganca/ollama/blob/main/docs/api.md#generate-a-chat-completion)

  Parses and validates inputs for making a requests from the Ollama Chat API.

  Converts responses into more specialized `LangChain` data structures.

  The module's functionalities include:

  - Initializing a new `ChatOllamaAI` struct with defaults or specific attributes.
  - Validating and casting input data to fit the expected schema.
  - Preparing and sending requests to the Ollama AI service API.
  - Managing both streaming and non-streaming API responses.
  - Processing API responses to convert them into suitable message formats.

  The `ChatOllamaAI` struct has fields to configure the AI, including but not limited to:

  - `endpoint`: URL of the Ollama AI service.
  - `model`: The AI model used, e.g., "llama2:latest".
  - `receive_timeout`: Max wait time for AI service responses.
  - `temperature`: Influences the AI's response creativity.

  For detailed info on on all other parameters see documentation here:
  https://github.com/jmorganca/ollama/blob/main/docs/modelfile.md#valid-parameters-and-values

  This module is for use within LangChain and follows the `ChatModel` behavior,
  outlining callbacks AI chat models must implement.

  Usage examples and more details are in the LangChain documentation or the
  module's function docs.

  ## Tool Support

  Currently, `ChatOllamaAI` supports tool calls when not streaming the responses.
  Streaming tool calls is not yet supported.
  """
  use Ecto.Schema
  require Logger
  import Ecto.Changeset
  alias __MODULE__
  alias LangChain.ChatModels.ChatModel
  alias LangChain.ChatModels.ChatOpenAI
  alias LangChain.Message
  alias LangChain.Message.ContentPart
  alias LangChain.Message.ToolCall
  alias LangChain.Message.ToolResult
  alias LangChain.MessageDelta
  alias LangChain.Function
  alias LangChain.FunctionParam
  alias LangChain.LangChainError
  alias LangChain.Utils
  alias LangChain.Callbacks

  @behaviour ChatModel

  @current_config_version 1

  @type t :: %ChatOllamaAI{}

  @create_fields [
    :endpoint,
    :keep_alive,
    :mirostat,
    :mirostat_eta,
    :mirostat_tau,
    :model,
    :num_ctx,
    :num_gqa,
    :num_gpu,
    :num_predict,
    :num_thread,
    :receive_timeout,
    :repeat_last_n,
    :repeat_penalty,
    :seed,
    :stop,
    :stream,
    :temperature,
    :tfs_z,
    :top_k,
    :top_p,
    :verbose_api
  ]

  @required_fields [:endpoint, :model]

  @receive_timeout 60_000 * 5

  @primary_key false
  embedded_schema do
    field :endpoint, :string, default: "http://localhost:11434/api/chat"

    # Change Keep Alive setting for unloading the model from memory.
    # (Default: "5m", set to a negative interval to disable)
    field :keep_alive, :string, default: "5m"

    # Enable Mirostat sampling for controlling perplexity.
    # (default: 0, 0 = disabled, 1 = Mirostat, 2 = Mirostat 2.0)
    field :mirostat, :integer, default: 0

    # Influences how quickly the algorithm responds to feedback from the generated text. A lower learning rate
    # will result in slower adjustments, while a higher learning rate will make the algorithm more responsive.
    # (Default: 0.1)
    field :mirostat_eta, :float, default: 0.1

    # Controls the balance between coherence and diversity of the output. A lower value will result in more focused
    # and coherent text. (Default: 5.0)
    field :mirostat_tau, :float, default: 5.0

    field :model, :string, default: "llama2:latest"

    # Sets the size of the context window used to generate the next token. (Default: 2048)
    field :num_ctx, :integer, default: 2048

    # The number of GQA groups in the transformer layer. Required for some models, for example it is 8 for llama2:70b
    field :num_gqa, :integer

    # The number of layers to send to the GPU(s). On macOS it defaults to 1 to enable metal support, 0 to disable.
    field :num_gpu, :integer

    # Maximum number of tokens to predict when generating text. (Default: 128, -1 = infinite generation, -2 = fill context)
    field :num_predict, :integer, default: 128

    # Sets the number of threads to use during computation. By default, Ollama will detect this for optimal
    # performance. It is recommended to set this value to the number of physical CPU cores your system has (as
    # opposed to the logical number of cores).
    field :num_thread, :integer

    # Duration in seconds for the response to be received. When streaming a very
    # lengthy response, a longer time limit may be required. However, when it
    # goes on too long by itself, it tends to hallucinate more.
    # Seems like the default for ollama is 5 minutes? https://github.com/jmorganca/ollama/pull/1257
    field :receive_timeout, :integer, default: @receive_timeout

    # Sets how far back for the model to look back to prevent repetition. (Default: 64, 0 = disabled, -1 = num_ctx)
    field :repeat_last_n, :integer, default: 64

    # Sets how strongly to penalize repetitions. A higher value (e.g., 1.5) will penalize repetitions more strongly,
    # while a lower value (e.g., 0.9) will be more lenient. (Default: 1.1)
    field :repeat_penalty, :float, default: 1.1

    # Sets the random number seed to use for generation. Setting this to a specific number will make the
    # model generate the same text for the same prompt. (Default: 0)
    field :seed, :integer, default: 0

    # Sets the stop sequences to use. When this pattern is encountered the LLM will stop generating text and return.
    # Multiple stop patterns may be set by specifying multiple separate stop parameters in a modelfile.
    # Empty arrays [] are excluded from API requests via Utils.conditionally_add_to_map to preserve modelfile defaults.
    # This prevents overriding the model's built-in stop tokens (e.g., <|eot_id|>, Human:, Assistant:).
    field :stop, {:array, :string}, default: []

    field :stream, :boolean, default: false

    # The temperature of the model. Increasing the temperature will make the model
    # answer more creatively. (Default: 0.8)
    field :temperature, :float, default: 0.8

    # Tail free sampling is used to reduce the impact of less probable tokens from the output. A higher value (e.g., 2.0)
    # will reduce the impact more, while a value of 1.0 disables this setting. (default: 1)
    field :tfs_z, :float, default: 1.0

    # Reduces the probability of generating nonsense. A higher value (e.g. 100) will give more diverse answers,
    # while a lower value (e.g. 10) will be more conservative. (Default: 40)
    field :top_k, :integer, default: 40

    # Works together with top-k. A higher value (e.g., 0.95) will lead to more diverse text,
    # while a lower value (e.g., 0.5) will generate more focused and conservative text. (Default: 0.9)
    field :top_p, :float, default: 0.9

    # A list of maps for callback handlers (treat as private)
    field :callbacks, {:array, :map}, default: []

    # For help with debugging. It outputs the RAW Req response received and the
    # RAW Elixir map being submitted to the API.
    field :verbose_api, :boolean, default: false
  end

  @doc """
  Creates a new `ChatOllamaAI` struct with the given attributes.
  """
  @spec new(attrs :: map()) :: {:ok, t} | {:error, Ecto.Changeset.t()}
  def new(%{} = attrs \\ %{}) do
    %ChatOllamaAI{}
    |> cast(attrs, @create_fields, empty_values: [""])
    |> common_validation()
    |> apply_action(:insert)
  end

  @doc """
  Creates a new `ChatOllamaAI` struct with the given attributes. Will raise an error if the changeset is invalid.
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
    |> validate_number(:temperature, greater_than_or_equal_to: 0.0, less_than_or_equal_to: 1.0)
    |> validate_number(:mirostat_eta, greater_than_or_equal_to: 0.0, less_than_or_equal_to: 1.0)
  end

  defp messages_for_api(messages) do
    Enum.reduce(messages, [], fn m, acc ->
      case for_api(m) do
        data when is_map(data) -> [data | acc]
        data when is_list(data) -> Enum.reverse(data) ++ acc
      end
    end)
    |> Enum.reverse()
  end

  @doc """
  Return the params formatted for an API request.
  """
  def for_api(%ChatOllamaAI{} = model, messages, tools) do
    %{
      model: model.model,
      messages: messages_for_api(messages),
      stream: model.stream,
      options:
        %{
          temperature: model.temperature,
          seed: model.seed,
          num_ctx: model.num_ctx,
          num_predict: model.num_predict,
          repeat_last_n: model.repeat_last_n,
          repeat_penalty: model.repeat_penalty,
          mirostat: model.mirostat,
          mirostat_eta: model.mirostat_eta,
          mirostat_tau: model.mirostat_tau,
          num_gqa: model.num_gqa,
          num_gpu: model.num_gpu,
          num_thread: model.num_thread,
          tfs_z: model.tfs_z,
          top_k: model.top_k,
          top_p: model.top_p
        }
        # Conditionally add stop sequences: excludes empty arrays [] and nil to preserve modelfile defaults
        |> Utils.conditionally_add_to_map(:stop, model.stop),
      receive_timeout: model.receive_timeout
    }
    |> Utils.conditionally_add_to_map(:tools, get_tools_for_api(tools))
  end

  def for_api(%Message{role: :assistant, tool_calls: tool_calls} = msg)
      when is_list(tool_calls) do
    content =
      case msg.content do
        content when is_binary(content) -> content
        content when is_list(content) -> ContentPart.parts_to_string(content)
        nil -> nil
      end

    %{
      "role" => :assistant,
      "content" => content
    }
    |> Utils.conditionally_add_to_map("tool_calls", Enum.map(tool_calls, &for_api(&1)))
  end

  # ToolCall support
  def for_api(%ToolCall{type: :function} = fun) do
    %{
      "id" => fun.call_id,
      "type" => "function",
      "function" => %{
        "name" => fun.name,
        "arguments" => fun.arguments
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

  def for_api(%Message{role: :tool, tool_results: tool_results}) when is_list(tool_results) do
    Enum.map(tool_results, &for_api/1)
  end

  def for_api(%ToolResult{content: content}) do
    %{
      "role" => :tool,
      "content" => ContentPart.parts_to_string(content)
    }
  end

  def for_api(%Message{content: content} = msg) when is_binary(content) do
    %{
      "role" => msg.role,
      "content" => ContentPart.content_to_string(msg.content)
    }
    |> Utils.conditionally_add_to_map("name", msg.name)
  end

  def for_api(%Message{role: :user, content: content} = msg) when is_list(content) do
    %{
      "role" => msg.role,
      "content" => ContentPart.content_to_string(content)
    }
    |> Utils.conditionally_add_to_map("name", msg.name)
  end

  # Handle messages with ContentPart content for non-user roles
  def for_api(%Message{content: content} = msg) when is_list(content) do
    %{
      "role" => msg.role,
      "content" => ContentPart.parts_to_string(content)
    }
    |> Utils.conditionally_add_to_map("name", msg.name)
  end

  # Handle ContentPart structures
  def for_api(%ContentPart{type: :text, content: content}) do
    content
  end

  defp get_tools_for_api(nil), do: []

  defp get_tools_for_api(tools) do
    Enum.map(tools, fn %Function{} = function ->
      %{"type" => "function", "function" => for_api(function)}
    end)
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
  Calls the Ollama Chat Completion API struct with configuration, plus
  either a simple message or the list of messages to act as the prompt.

  **NOTE:** This function *can* be used directly, but the primary interface
  should be through `LangChain.Chains.LLMChain`. The `ChatOllamaAI` module is more focused on
  translating the `LangChain` data structures to and from the Ollama API.

  Another benefit of using `LangChain.Chains.LLMChain` is that it combines the
  storage of messages, adding functions, adding custom context that should be
  passed to functions, and automatically applying `LangChain.MessageDelta`
  structs as they are are received, then converting those to the full
  `LangChain.Message` once fully complete.
  """

  @impl ChatModel
  def call(ollama_ai, prompt, tools \\ [])

  def call(%ChatOllamaAI{} = ollama_ai, prompt, tools) when is_binary(prompt) do
    messages = [
      Message.new_system!(),
      Message.new_user!(prompt)
    ]

    call(ollama_ai, messages, tools)
  end

  def call(%ChatOllamaAI{} = ollama_ai, messages, tools) when is_list(messages) do
    metadata = %{
      model: ollama_ai.model,
      message_count: length(messages),
      tools_count: length(tools)
    }

    LangChain.Telemetry.span([:langchain, :llm, :call], metadata, fn ->
      try do
        # Track the prompt being sent
        LangChain.Telemetry.llm_prompt(
          %{system_time: System.system_time()},
          %{model: ollama_ai.model, messages: messages}
        )

        case __MODULE__.do_api_request(ollama_ai, messages, tools) do
          {:error, reason} ->
            {:error, reason}

          parsed_data ->
            # Track the response being received
            LangChain.Telemetry.llm_response(
              %{system_time: System.system_time()},
              %{model: ollama_ai.model, response: parsed_data}
            )

            {:ok, parsed_data}
        end
      rescue
        err in LangChainError ->
          {:error, err.message}
      end
    end)
  end

  # Make the API request from the Ollama server.
  #
  # The result of the function is:
  #
  # - `result` - where `result` is a data-structure like a list or map.
  # - `{:error, reason}` - Where reason is a `LangChain.LangChainError`
  #   explanation of what went wrong.
  #
  # **NOTE:** callback function are IGNORED for ollama ai When `stream: true` is
  # If `stream: false`, the completed message is returned.
  #
  # If `stream: true`, the completed message is returned after MessageDelta's.
  #
  # Retries the request up to 3 times on transient errors with a 1 second delay
  @doc false
  @spec do_api_request(t(), [Message.t()], ChatModel.tools(), integer()) ::
          list() | struct() | {:error, String.t()}
  def do_api_request(ollama_ai, messages, tools, retry_count \\ 3)

  def do_api_request(_ollama_ai, _messages, _tools, 0) do
    raise LangChainError, "Retries exceeded. Connection failed."
  end

  def do_api_request(
        %ChatOllamaAI{stream: false} = ollama_ai,
        messages,
        tools,
        retry_count
      ) do
    raw_data = for_api(ollama_ai, messages, tools)

    if ollama_ai.verbose_api do
      IO.inspect(raw_data, label: "RAW DATA BEING SUBMITTED")
    end

    req =
      Req.new(
        url: ollama_ai.endpoint,
        json: raw_data,
        receive_timeout: ollama_ai.receive_timeout,
        retry: :transient,
        max_retries: 3,
        inet6: true,
        retry_delay: fn attempt -> 300 * attempt end
      )

    req
    |> Req.post()
    |> case do
      {:ok, %Req.Response{body: data} = response} ->
        if ollama_ai.verbose_api do
          IO.inspect(response, label: "RAW REQ RESPONSE")
        end

        Callbacks.fire(ollama_ai.callbacks, :on_llm_response_headers, [response.headers])

        case do_process_response(ollama_ai, data) do
          {:error, reason} ->
            {:error, reason}

          result ->
            # Track non-streaming response completion
            LangChain.Telemetry.emit_event(
              [:langchain, :llm, :response, :non_streaming],
              %{system_time: System.system_time()},
              %{
                model: ollama_ai.model,
                response_size: byte_size(inspect(result))
              }
            )

            result
        end

      {:error, %Req.TransportError{reason: :timeout}} ->
        {:error, "Request timed out"}

      {:error, %Req.TransportError{reason: :closed}} ->
        # Force a retry by making a recursive call decrementing the counter
        Logger.debug(fn -> "Mint connection closed: retry count = #{inspect(retry_count)}" end)
        do_api_request(ollama_ai, messages, tools, retry_count - 1)

      other ->
        Logger.error("Unexpected and unhandled API response! #{inspect(other)}")
        other
    end
  end

  def do_api_request(
        %ChatOllamaAI{stream: true} = ollama_ai,
        messages,
        tools,
        retry_count
      ) do
    raw_data = for_api(ollama_ai, messages, tools)

    if ollama_ai.verbose_api do
      IO.inspect(raw_data, label: "RAW DATA BEING SUBMITTED")
    end

    Req.new(
      url: ollama_ai.endpoint,
      json: raw_data,
      inet6: true,
      receive_timeout: ollama_ai.receive_timeout
    )
    |> Req.post(
      into:
        Utils.handle_stream_fn(
          ollama_ai,
          &ChatOpenAI.decode_stream/1,
          &do_process_response(ollama_ai, &1)
        )
    )
    |> case do
      {:ok, %Req.Response{body: data} = response} ->
        Callbacks.fire(ollama_ai.callbacks, :on_llm_response_headers, [response.headers])

        data

      {:error, %LangChainError{} = error} ->
        {:error, error}

      {:error, %Req.TransportError{reason: :timeout} = err} ->
        {:error,
         LangChainError.exception(type: "timeout", message: "Request timed out", original: err)}

      {:error, %Req.TransportError{reason: :closed}} ->
        # Force a retry by making a recursive call decrementing the counter
        Logger.debug(fn -> "Mint connection closed: retry count = #{inspect(retry_count)}" end)
        do_api_request(ollama_ai, messages, tools, retry_count - 1)

      other ->
        Logger.error(
          "Unhandled and unexpected response from streamed post call. #{inspect(other)}"
        )

        {:error,
         LangChainError.exception(type: "unexpected_response", message: "Unexpected response")}
    end
  end

  def do_process_response(%{stream: true} = _model, %{"message" => message, "done" => true}) do
    create_message(message, :complete, MessageDelta)
  end

  def do_process_response(model, %{
        "message" => %{"tool_calls" => calls} = message,
        "done" => true
      })
      when calls != [] do
    message
    |> Map.merge(%{
      "tool_calls" => Enum.map(calls, &do_process_response(model, &1))
    })
    |> create_message(:complete, Message)
  end

  def do_process_response(_model, %{"message" => message, "done" => true}) do
    create_message(message, :complete, Message)
  end

  def do_process_response(_model, %{"message" => message, "done" => _other}) do
    create_message(message, :incomplete, MessageDelta)
  end

  def do_process_response(_model, %{"error" => reason} = response) do
    Logger.error("Received error from API: #{inspect(reason)}")
    {:error, LangChainError.exception(message: reason, original: response)}
  end

  def do_process_response(_model, %{
        "function" => %{
          "arguments" => args,
          "name" => name
        }
      }) do
    case ToolCall.new(%{
           call_id: Ecto.UUID.generate(),
           type: :function,
           name: name,
           arguments: args
         }) do
      {:ok, %ToolCall{} = call} ->
        call

      {:error, changeset} ->
        reason = Utils.changeset_error_to_string(changeset)
        Logger.error("Failed to process ToolCall for a function. Reason: #{reason}")
        {:error, reason}
    end
  end

  defp create_message(message, status, message_type) do
    case message_type.new(Map.merge(message, %{"status" => status})) do
      {:ok, new_message} ->
        new_message

      {:error, %Ecto.Changeset{} = changeset} ->
        {:error, LangChainError.exception(changeset)}
    end
  end

  @doc """
  Generate a config map that can later restore the model's configuration.
  """
  @impl ChatModel
  @spec serialize_config(t()) :: %{String.t() => any()}
  def serialize_config(%ChatOllamaAI{} = model) do
    Utils.to_serializable_map(
      model,
      [
        :endpoint,
        :keep_alive,
        :model,
        :mirostat,
        :mirostat_eta,
        :mirostat_tau,
        :num_ctx,
        :num_gqa,
        :num_gpu,
        :num_predict,
        :num_thread,
        :receive_timeout,
        :repeat_last_n,
        :repeat_penalty,
        :seed,
        :stop,
        :stream,
        :temperature,
        :tfs_z,
        :top_k,
        :top_p,
        :verbose_api
      ],
      @current_config_version
    )
  end

  @doc """
  Restores the model from the config.
  """
  @impl ChatModel
  def restore_from_map(%{"version" => 1} = data) do
    ChatOllamaAI.new(data)
  end
end
