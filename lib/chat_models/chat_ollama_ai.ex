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
  """
  use Ecto.Schema
  require Logger
  import Ecto.Changeset
  alias __MODULE__
  alias LangChain.ChatModels.ChatModel
  alias LangChain.ChatModels.ChatOpenAI
  alias LangChain.Message
  alias LangChain.MessageDelta
  alias LangChain.LangChainError
  alias LangChain.Utils

  @behaviour ChatModel

  @type t :: %ChatOllamaAI{}

  @create_fields [
    :endpoint,
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
    :top_p
  ]

  @required_fields [:endpoint, :model]

  @receive_timeout 60_000 * 5

  @primary_key false
  embedded_schema do
    field :endpoint, :string, default: "http://localhost:11434/api/chat"

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
    field :stop, :string

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

  @doc """
  Return the params formatted for an API request.
  """
  def for_api(%ChatOllamaAI{} = model, messages, _functions) do
    %{
      model: model.model,
      temperature: model.temperature,
      messages: messages |> Enum.map(&ChatOpenAI.for_api/1),
      stream: model.stream,
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
      receive_timeout: model.receive_timeout,
      stop: model.stop,
      tfs_z: model.tfs_z,
      top_k: model.top_k,
      top_p: model.top_p
    }
  end

  @doc """
  Calls the Ollama Chat Completion API struct with configuration, plus
  either a simple message or the list of messages to act as the prompt.

  **NOTE:** This API as of right now does not support functions. More
  information here: https://github.com/jmorganca/ollama/issues/1729

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
  def call(ollama_ai, prompt, functions \\ [], callback_fn \\ nil)

  def call(%ChatOllamaAI{} = ollama_ai, prompt, functions, callback_fn) when is_binary(prompt) do
    messages = [
      Message.new_system!(),
      Message.new_user!(prompt)
    ]

    call(ollama_ai, messages, functions, callback_fn)
  end

  def call(%ChatOllamaAI{} = ollama_ai, messages, functions, callback_fn)
      when is_list(messages) do
    try do
      case do_api_request(ollama_ai, messages, functions, callback_fn) do
        {:error, reason} ->
          {:error, reason}

        parsed_data ->
          {:ok, parsed_data}
      end
    rescue
      err in LangChainError ->
        {:error, err.message}
    end
  end

  # Make the API request from the Ollama server.
  #
  # The result of the function is:
  #
  # - `result` - where `result` is a data-structure like a list or map.
  # - `{:error, reason}` - Where reason is a string explanation of what went wrong.
  #
  # **NOTE:** callback function are IGNORED for ollama ai
  # When `stream: true` is
  # If `stream: false`, the completed message is returned.
  #
  # If `stream: true`, the completed message is returned after MessageDelta's.
  #
  # Retries the request up to 3 times on transient errors with a 1 second delay
  @doc false
  @spec do_api_request(t(), [Message.t()], [Function.t()], (any() -> any())) ::
          list() | struct() | {:error, String.t()}
  def do_api_request(ollama_ai, messages, functions, callback_fn, retry_count \\ 3)

  def do_api_request(_ollama_ai, _messages, _functions, _callback_fn, 0) do
    raise LangChainError, "Retries exceeded. Connection failed."
  end

  def do_api_request(
        %ChatOllamaAI{stream: false} = ollama_ai,
        messages,
        functions,
        callback_fn,
        retry_count
      ) do
    req =
      Req.new(
        url: ollama_ai.endpoint,
        json: for_api(ollama_ai, messages, functions),
        receive_timeout: ollama_ai.receive_timeout,
        retry: :transient,
        max_retries: 3,
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
            result
        end

      {:error, %Req.TransportError{reason: :timeout}} ->
        {:error, "Request timed out"}

      {:error, %Req.TransportError{reason: :closed}} ->
        # Force a retry by making a recursive call decrementing the counter
        Logger.debug(fn -> "Mint connection closed: retry count = #{inspect(retry_count)}" end)
        do_api_request(ollama_ai, messages, functions, callback_fn, retry_count - 1)

      other ->
        Logger.error("Unexpected and unhandled API response! #{inspect(other)}")
        other
    end
  end

  def do_api_request(
        %ChatOllamaAI{stream: true} = ollama_ai,
        messages,
        functions,
        callback_fn,
        retry_count
      ) do
    Req.new(
      url: ollama_ai.endpoint,
      json: for_api(ollama_ai, messages, functions),
      receive_timeout: ollama_ai.receive_timeout
    )
    |> Req.post(
      into:
        Utils.handle_stream_fn(
          ollama_ai,
          &ChatOpenAI.decode_stream/1,
          &do_process_response/1,
          callback_fn
        )
    )
    |> case do
      {:ok, %Req.Response{body: data}} ->
        data

      {:error, %LangChainError{message: reason}} ->
        {:error, reason}

      {:error, %Req.TransportError{reason: :timeout}} ->
        {:error, "Request timed out"}

      {:error, %Req.TransportError{reason: :closed}} ->
        # Force a retry by making a recursive call decrementing the counter
        Logger.debug(fn -> "Mint connection closed: retry count = #{inspect(retry_count)}" end)
        do_api_request(ollama_ai, messages, functions, callback_fn, retry_count - 1)

      other ->
        Logger.error(
          "Unhandled and unexpected response from streamed post call. #{inspect(other)}"
        )

        {:error, "Unexpected response"}
    end
  end

  def do_process_response(%{"message" => message, "done" => true}) do
    create_message(message, :complete, Message)
  end

  def do_process_response(%{"message" => message, "done" => _other}) do
    create_message(message, :incomplete, MessageDelta)
  end

  def do_process_response(%{"error" => reason}) do
    Logger.error("Received error from API: #{inspect(reason)}")
    {:error, reason}
  end

  defp create_message(message, status, message_type) do
    case message_type.new(Map.merge(message, %{"status" => status})) do
      {:ok, new_message} ->
        new_message

      {:error, changeset} ->
        {:error, Utils.changeset_error_to_string(changeset)}
    end
  end
end
