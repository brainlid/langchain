defmodule LangChain.ChatModels.ChatGoogleAI do
  @moduledoc """
  Parses and validates inputs for making a request for the Google AI  Chat API.

  Converts response into more specialized `LangChain` data structures.
  """
  use Ecto.Schema
  require Logger
  import Ecto.Changeset
  alias __MODULE__
  alias LangChain.Config
  alias LangChain.ChatModels.ChatModel
  alias LangChain.Message
  alias LangChain.MessageDelta
  alias LangChain.LangChainError
  alias LangChain.ForOpenAIApi
  alias LangChain.Utils

  @behaviour ChatModel

  @default_base_url "https://generativelanguage.googleapis.com"
  @default_api_version "v1beta"
  @default_endpoint "#{@default_base_url}#{@default_api_version}"

  # allow up to 2 minutes for response.
  @receive_timeout 60_000

  @primary_key false
  embedded_schema do
    field :endpoint, :string, default: @default_endpoint

    # The version of the API to use.
    field :version, :string, default: @default_api_version
    field :model, :string, default: "gemini-pro"
    field :api_key, :string

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
  end

  @type t :: %ChatGoogleAI{}

  @create_fields [
    :endpoint,
    :model,
    :api_key,
    :temperature,
    :top_p,
    :top_k,
    :receive_timeout,
    :stream
  ]
  @required_fields [
    :endpoint,
    :version,
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
    req = %{
      "contents" =>
        Stream.map(messages, &for_api/1)
        |> Enum.flat_map(fn
          list when is_list(list) -> list
          not_list -> [not_list]
        end),
      "generationConfig" => %{
        "temperature" => google_ai.temperature,
        "topP" => google_ai.top_p,
        "topK" => google_ai.top_k
      }
    }

    if functions && not Enum.empty?(functions) do
      req
      |> Map.put("tools", [
        %{
          # Google AI functions use an OpenAI compatible format.
          # See: https://ai.google.dev/docs/function_calling#how_it_works
          "functionDeclarations" => Enum.map(functions, &ForOpenAIApi.for_api/1)
        }
      ])
    end
  end

  defp for_api(%Message{role: :assistant, function_name: fun_name} = fun)
       when is_binary(fun_name) do
    %{
      "role" => map_role(:assistant),
      "parts" => [
        %{
          "functionCall" => %{
            "name" => fun_name,
            "args" => fun.arguments
          }
        }
      ]
    }
  end

  defp for_api(%Message{role: :function} = message) do
    %{
      "role" => map_role(:function),
      "parts" => [
        %{
          "functionResponse" => %{
            "name" => message.function_name,
            "response" => message.function_response
          }
        }
      ]
    }
  end

  defp for_api(%Message{role: :system} = message) do
    # No system messages support means we need to fake a prompt and response
    # to pretend like it worked.
    [
      %{
        "role" => :user,
        "parts" => [%{"text" => message.content}]
      },
      %{
        "role" => :model,
        "parts" => [%{"text" => ""}]
      }
    ]
  end

  defp for_api(%Message{} = message) do
    %{
      "role" => map_role(message.role),
      "parts" => [%{"text" => message.content}]
    }
  end

  defp map_role(role) do
    case role do
      :assistant -> :model
      # System prompts are not supported yet. Google recommends using user prompt.
      :system -> :user
      role -> role
    end
  end

  @doc """
  Calls the Google AI API passing the ChatGoogleAI struct with configuration, plus
  either a simple message or the list of messages to act as the prompt.

  Optionally pass in a list of functions available to the LLM for requesting
  execution in response.

  Optionally pass in a callback function that can be executed as data is
  received from the API.

  **NOTE:** This function *can* be used directly, but the primary interface
  should be through `LangChain.Chains.LLMChain`. The `ChatGoogleAI` module is more focused on
  translating the `LangChain` data structures to and from the OpenAI API.

  Another benefit of using `LangChain.Chains.LLMChain` is that it combines the
  storage of messages, adding functions, adding custom context that should be
  passed to functions, and automatically applying `LangChain.MessageDelta`
  structs as they are are received, then converting those to the full
  `LangChain.Message` once fully complete.
  """
  @impl ChatModel
  def call(openai, prompt, functions \\ [], callback_fn \\ nil)

  def call(%ChatGoogleAI{} = google_ai, prompt, functions, callback_fn) when is_binary(prompt) do
    messages = [
      Message.new_system!(),
      Message.new_user!(prompt)
    ]

    call(google_ai, messages, functions, callback_fn)
  end

  def call(%ChatGoogleAI{} = google_ai, messages, functions, callback_fn)
      when is_list(messages) do
    try do
      case do_api_request(google_ai, messages, functions, callback_fn) do
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

  @doc false
  @spec do_api_request(t(), [Message.t()], [Function.t()], (any() -> any())) ::
          list() | struct() | {:error, String.t()}
  def do_api_request(%ChatGoogleAI{stream: false} = google_ai, messages, functions, callback_fn) do
    req =
      Req.new(
        url: build_url(google_ai),
        json: for_api(google_ai, messages, functions),
        receive_timeout: google_ai.receive_timeout,
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
            Utils.fire_callback(google_ai, result, callback_fn)
            result
        end

      {:error, %Mint.TransportError{reason: :timeout}} ->
        {:error, "Request timed out"}

      other ->
        Logger.error("Unexpected and unhandled API response! #{inspect(other)}")
        other
    end
  end

  def do_api_request(%ChatGoogleAI{stream: true} = google_ai, messages, functions, callback_fn) do
    Req.new(
      url: build_url(google_ai),
      json: for_api(google_ai, messages, functions),
      receive_timeout: google_ai.receive_timeout,
      finch_request:
        Utils.finch_stream_fn(google_ai, &do_process_response(&1, MessageDelta), callback_fn)
    )
    |> Req.Request.put_header("accept-encoding", "utf-8")
    |> Req.post()
    |> case do
      {:ok, %Req.Response{body: data}} ->
        # Google AI uses `finishReason: "STOP` for all messages in the stream.
        # This field can't be used to terminate the list of deltas, so simulate
        # this behavior by forcing the final delta to have `status: :complete`.
        complete_final_delta(data)

      {:error, %LangChainError{message: reason}} ->
        {:error, reason}

      other ->
        Logger.error(
          "Unhandled and unexpected response from streamed post call. #{inspect(other)}"
        )

        {:error, "Unexpected response"}
    end
  end

  @spec build_url(t()) :: String.t()
  defp build_url(%ChatGoogleAI{endpoint: endpoint, version: version, model: model} = google_ai) do
    "#{endpoint}/#{version}/models/#{model}:#{get_action(google_ai)}?key=#{get_api_key(google_ai)}"
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

  def do_process_response(response, message_type \\ Message)

  def do_process_response(%{"candidates" => candidates}, message_type) when is_list(candidates) do
    candidates
    |> Enum.map(&do_process_response(&1, message_type))
  end

  def do_process_response(
        %{
          "content" => %{"parts" => [%{"functionCall" => %{"args" => raw_args, "name" => name}}]}
        } = data,
        message_type
      ) do
    case message_type.new(%{
           "role" => "assistant",
           "function_name" => name,
           "arguments" => raw_args,
           "complete" => true,
           "index" => data["index"]
         }) do
      {:ok, message} ->
        message

      {:error, changeset} ->
        {:error, Utils.changeset_error_to_string(changeset)}
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
      case finish do
        "STOP" ->
          :complete

        "LENGTH" ->
          :length

        other ->
          Logger.warning("Unsupported finishReason in response. Reason: #{inspect(other)}")
          nil
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

      {:error, changeset} ->
        {:error, Utils.changeset_error_to_string(changeset)}
    end
  end

  def do_process_response(%{"error" => %{"message" => reason}}, _) do
    Logger.error("Received error from API: #{inspect(reason)}")
    {:error, reason}
  end

  def do_process_response({:error, %Jason.DecodeError{} = response}, _) do
    error_message = "Received invalid JSON: #{inspect(response)}"
    Logger.error(error_message)
    {:error, error_message}
  end

  def do_process_response(other, _) do
    Logger.error("Trying to process an unexpected response. #{inspect(other)}")
    {:error, "Unexpected response"}
  end

  defp unmap_role("model"), do: "assistant"
  defp unmap_role(role), do: role
end
