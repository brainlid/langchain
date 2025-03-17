defmodule LangChain.ChatModels.ChatMistralAI do
  use Ecto.Schema
  require Logger
  import Ecto.Changeset
  alias __MODULE__
  alias LangChain.Config
  alias LangChain.ChatModels.ChatOpenAI
  alias LangChain.ChatModels.ChatModel
  alias LangChain.Function
  alias LangChain.Message
  alias LangChain.MessageDelta
  alias LangChain.LangChainError
  alias LangChain.Utils
  alias LangChain.Callbacks

  @behaviour ChatModel

  @current_config_version 1
  @receive_timeout 60_000

  @default_endpoint "https://api.mistral.ai/v1/chat/completions"

  @primary_key false
  embedded_schema do
    field :endpoint, :string, default: @default_endpoint

    # The version of the API to use.
    field :model, :string
    field :api_key, :string, redact: true

    # What sampling temperature to use, between 0 and 1. Higher values like 0.8
    # will make the output more random, while lower values like 0.2 will make it
    # more focused and deterministic.
    field :temperature, :float, default: 0.9

    # The topP parameter changes how the model selects tokens for output. Tokens
    # are selected from the most to least probable until the sum 3of their
    # probabilities equals the topP value. For example, if tokens A, B, and C have
    # a probability of 0.3, 0.2, and 0.1 and the topP value is 0.5, then the model
    # will select either A or B as the next token by using the temperature and exclude
    # C as a candidate. The default topP value is 0.95.
    field :top_p, :float, default: 1.0

    # Duration in seconds for the response to be received. When streaming a very
    # lengthy response, a longer time limit may be required. However, when it
    # goes on too long by itself, it tends to hallucinate more.
    field :receive_timeout, :integer, default: @receive_timeout

    field :max_tokens, :integer

    field :safe_prompt, :boolean, default: false

    field :random_seed, :integer

    field :stream, :boolean, default: false

    # A list of maps for callback handlers
    field :callbacks, {:array, :map}, default: []
  end

  @type t :: %ChatMistralAI{}

  @create_fields [
    :endpoint,
    :model,
    :api_key,
    :temperature,
    :top_p,
    :receive_timeout,
    :max_tokens,
    :safe_prompt,
    :random_seed,
    :stream,
    :callbacks
  ]
  @required_fields [
    :model
  ]

  @spec get_api_key(t) :: String.t()
  defp get_api_key(%ChatMistralAI{api_key: api_key}) do
    # if no API key is set default to `""` which will raise an API error
    api_key || Config.resolve(:mistral_api_key)
  end

  @spec get_headers(t) :: [tuple()]
  defp get_headers(%ChatMistralAI{} = chat) do
    api_key = get_api_key(chat)

    [
      Authorization: "Bearer #{api_key}",
      "Content-Type": "application/json",
      Accept: "application/json"
    ]
  end

  @spec new(attrs :: map()) :: {:ok, t} | {:error, Ecto.Changeset.t()}
  def new(%{} = attrs \\ %{}) do
    %ChatMistralAI{}
    |> cast(attrs, @create_fields)
    |> common_validation()
    |> apply_action(:insert)
  end

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

  @spec for_api(t, message :: [map()], functions :: [map()]) :: %{atom() => any()}
  def for_api(%ChatMistralAI{} = mistral, messages, _functions) do
    %{
      model: mistral.model,
      temperature: mistral.temperature,
      top_p: mistral.top_p,
      safe_prompt: mistral.safe_prompt,
      stream: mistral.stream,
      messages: Enum.map(messages, &ChatOpenAI.for_api/1)
    }
    |> Utils.conditionally_add_to_map(:random_seed, mistral.random_seed)
    |> Utils.conditionally_add_to_map(:max_tokens, mistral.max_tokens)
  end

  @impl ChatModel
  def call(mistral, prompt, functions \\ [])

  def call(%ChatMistralAI{} = mistral, prompt, functions) when is_binary(prompt) do
    messages = [
      Message.new_system!(),
      Message.new_user!(prompt)
    ]

    call(mistral, messages, functions)
  end

  def call(%ChatMistralAI{} = mistral, messages, functions) when is_list(messages) do
    try do
      # make base api request and perform high-level success/failure checks
      case do_api_request(mistral, messages, functions) do
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

  @spec do_api_request(t(), [Message.t()], [Function.t()], integer()) ::
          list() | struct() | {:error, String.t()}
  def do_api_request(mistral, messages, functions, retry_count \\ 3)

  def do_api_request(_mistral, _messages, _functions, 0) do
    raise LangChainError.exception(
            type: "retries_exceeded",
            message: "Retries exceeded. Connection failed."
          )
  end

  def do_api_request(
        %ChatMistralAI{stream: false} = mistral,
        messages,
        functions,
        retry_count
      ) do
    req =
      Req.new(
        url: mistral.endpoint,
        json: for_api(mistral, messages, functions),
        headers: get_headers(mistral),
        receive_timeout: mistral.receive_timeout,
        retry: :transient,
        max_retries: 3,
        retry_delay: fn attempt -> 300 * attempt end
      )

    req
    |> Req.post()
    # parse the body and return it as parsed structs
    |> case do
      {:ok, %Req.Response{body: data}} ->
        case do_process_response(mistral, data) do
          {:error, reason} ->
            {:error, reason}

          result ->
            Callbacks.fire(mistral.callbacks, :on_llm_new_message, [mistral, result])
            result
        end

      {:error, %Req.TransportError{reason: :timeout} = err} ->
        {:error,
         LangChainError.exception(type: "timeout", message: "Request timed out", original: err)}

      {:error, %Req.TransportError{reason: :closed}} ->
        # Force a retry by making a recursive call decrementing the counter
        Logger.debug(fn -> "Mint connection closed: retry count = #{inspect(retry_count)}" end)
        do_api_request(mistral, messages, functions, retry_count - 1)

      other ->
        Logger.error("Unexpected and unhandled API response! #{inspect(other)}")
        other
    end
  end

  def do_api_request(
        %ChatMistralAI{stream: true} = mistral,
        messages,
        functions,
        retry_count
      ) do
    Req.new(
      url: mistral.endpoint,
      json: for_api(mistral, messages, functions),
      headers: get_headers(mistral),
      receive_timeout: mistral.receive_timeout
    )
    |> Req.post(
      into:
        Utils.handle_stream_fn(
          mistral,
          &ChatOpenAI.decode_stream/1,
          &do_process_response(mistral, &1)
        )
    )
    |> case do
      {:ok, %Req.Response{body: data}} ->
        data

      {:error, %LangChainError{} = err} ->
        {:error, err}

      {:error, %Req.TransportError{reason: :timeout} = err} ->
        {:error,
         LangChainError.exception(type: "timeout", message: "Request timed out", original: err)}

      {:error, %Req.TransportError{reason: :closed}} ->
        # Force a retry by making a recursive call decrementing the counter
        Logger.debug(fn -> "Mint connection closed: retry count = #{inspect(retry_count)}" end)
        do_api_request(mistral, messages, functions, retry_count - 1)

      other ->
        Logger.error(
          "Unhandled and unexpected response from streamed post call. #{inspect(other)}"
        )

        {:error,
         LangChainError.exception(type: "unexpected_response", message: "Unexpected response")}
    end
  end

  # Parse a new message response
  @doc false
  @spec do_process_response(t(), data :: %{String.t() => any()} | {:error, any()}) ::
          Message.t()
          | [Message.t()]
          | MessageDelta.t()
          | [MessageDelta.t()]
          | {:error, LangChainError.t()}
  def do_process_response(model, %{"choices" => choices}) when is_list(choices) do
    # process each response individually. Return a list of all processed choices
    for choice <- choices do
      do_process_response(model, choice)
    end
  end

  def do_process_response(
        _model,
        %{"delta" => delta_body, "finish_reason" => finish, "index" => index} = _msg
      ) do
    status =
      case finish do
        nil ->
          :incomplete

        "stop" ->
          :complete

        "length" ->
          :length

        "model_length" ->
          :length

        other ->
          Logger.warning("Unsupported finish_reason in delta message. Reason: #{inspect(other)}")
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

    case MessageDelta.new(data) do
      {:ok, message} ->
        message

      {:error, changeset} ->
        {:error, Utils.changeset_error_to_string(changeset)}
    end
  end

  def do_process_response(_model, %{
        "finish_reason" => finish_reason,
        "message" => message,
        "index" => index
      }) do
    status =
      case finish_reason do
        "stop" ->
          :complete

        "length" ->
          :length

        "model_length" ->
          :length

        other ->
          Logger.warning("Unsupported finish_reason in message. Reason: #{inspect(other)}")
          nil
      end

    case Message.new(Map.merge(message, %{"status" => status, "index" => index})) do
      {:ok, message} ->
        message

      {:error, %Ecto.Changeset{} = changeset} ->
        {:error, LangChainError.exception(changeset)}
    end
  end

  def do_process_response(_model, %{"error" => %{"message" => reason}}) do
    Logger.error("Received error from API: #{inspect(reason)}")
    {:error, LangChainError.exception(message: reason)}
  end

  def do_process_response(_model, {:error, %Jason.DecodeError{} = response}) do
    error_message = "Received invalid JSON: #{inspect(response)}"
    Logger.error(error_message)

    {:error,
     LangChainError.exception(type: "invalid_json", message: error_message, original: response)}
  end

  def do_process_response(_model, other) do
    Logger.error("Trying to process an unexpected response. #{inspect(other)}")

    {:error,
     LangChainError.exception(type: "unexpected_response", message: "Unexpected response")}
  end

  @doc """
  Generate a config map that can later restore the model's configuration.
  """
  @impl ChatModel
  @spec serialize_config(t()) :: %{String.t() => any()}
  def serialize_config(%ChatMistralAI{} = model) do
    Utils.to_serializable_map(
      model,
      [
        :endpoint,
        :model,
        :temperature,
        :top_p,
        :receive_timeout,
        :max_tokens,
        :safe_prompt,
        :random_seed,
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
    ChatMistralAI.new(data)
  end
end
