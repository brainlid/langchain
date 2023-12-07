defmodule LangChain.ChatModels.ChatZephyr do
  @moduledoc """
  Represents the Zephyr model when hosted by Nx and Bumblebee directly from
  Elixir.

  Parses and validates inputs for making a requests from the self-hosted Zephyr
  LLM.

  Converts responses into more specialized `LangChain` data structures.
  """
  use Ecto.Schema
  require Logger
  import Ecto.Changeset
  import LangChain.Utils.ApiOverride
  alias __MODULE__
  alias LangChain.Message
  alias LangChain.LangChainError
  alias LangChain.Utils
  alias LangChain.MessageDelta

  # NOTE: As of gpt-4 and gpt-3.5, only one function_call is issued at a time
  # even when multiple requests could be issued based on the prompt.

  # allow up to 2 minutes for response.
  # @receive_timeout 60_000

  @primary_key false
  embedded_schema do
    # Name of the Nx.Serving to use for working with the LLM.
    field :serving, :any, virtual: true

    # # What sampling temperature to use, between 0 and 2. Higher values like 0.8
    # # will make the output more random, while lower values like 0.2 will make it
    # # more focused and deterministic.
    # field :temperature, :float, default: 1.0

    # # Seed for randomizing behavior or giving more deterministic output. Helpful for testing.
    # field :seed, :integer

    # The bumblebee model may compile differently based on the stream true/false
    # option on the serving. Therefore, streaming should be enabled on the
    # serving and a stream option here can change the way data is received in
    # code. - https://github.com/elixir-nx/bumblebee/issues/295

    field :stream, :boolean, default: true
  end

  @type t :: %ChatZephyr{}

  @type call_response :: {:ok, Message.t() | [Message.t()]} | {:error, String.t()}
  @type callback_data ::
          {:ok, Message.t() | MessageDelta.t() | [Message.t() | MessageDelta.t()]}
          | {:error, String.t()}
  @type callback_fn :: (Message.t() | MessageDelta.t() -> any())

  @create_fields [
    :serving,
    # :temperature,
    # :seed,
    :stream
  ]
  @required_fields [:serving]

  # Tags used for formatting the messages. Don't allow the user to include these
  # themselves.
  #
  # https://huggingface.co/docs/transformers/main/chat_templating#how-do-i-use-chat-templates
  @system_tag "<|system|>"
  @user_tag "<|user|>"
  @assistant_tag "<|assistant|>"
  @text_end_tag "</s>"

  @doc """
  Setup a ChatZephyr client configuration.
  """
  @spec new(attrs :: map()) :: {:ok, t} | {:error, Ecto.Changeset.t()}
  def new(%{} = attrs \\ %{}) do
    %ChatZephyr{}
    |> cast(attrs, @create_fields)
    |> common_validation()
    |> apply_action(:insert)
  end

  @doc """
  Setup a ChatZephyr client configuration and return it or raise an error if invalid.
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

  # @doc """
  # Return the params formatted for an API request.
  # """

  # @spec for_api(t, message :: [map()], functions :: [map()]) :: %{atom() => any()}
  # def for_api(%ChatZephyr{} = zephyr, messages, functions) do
  #   %{
  #     model: zephyr.model,
  #     temperature: zephyr.temperature,
  #     frequency_penalty: zephyr.frequency_penalty,
  #     n: zephyr.n,
  #     stream: zephyr.stream,
  #     messages: Enum.map(messages, &ForZephyrApi.for_api/1)
  #   }
  #   |> Utils.conditionally_add_to_map(:seed, zephyr.seed)
  #   |> Utils.conditionally_add_to_map(:functions, get_functions_for_api(functions))
  # end

  # defp get_functions_for_api(nil), do: []

  # defp get_functions_for_api(functions) do
  #   Enum.map(functions, &ForZephyrApi.for_api/1)
  # end

  @doc """
  Calls the zephyr API passing the ChatZephyr struct with configuration, plus
  either a simple message or the list of messages to act as the prompt.

  Optionally pass in a list of functions available to the LLM for requesting
  execution in response.

  Optionally pass in a callback function that can be executed as data is
  received from the API.

  **NOTE:** This function *can* be used directly, but the primary interface
  should be through `LangChain.Chains.LLMChain`. The `ChatZephyr` module is more focused on
  translating the `LangChain` data structures to and from the zephyr API.

  Another benefit of using `LangChain.Chains.LLMChain` is that it combines the
  storage of messages, adding functions, adding custom context that should be
  passed to functions, and automatically applying `LangChain.MessageDelta`
  structs as they are are received, then converting those to the full
  `LangChain.Message` once fully complete.
  """
  @spec call(
          t(),
          String.t() | [Message.t()],
          [LangChain.Function.t()],
          nil | callback_fn()
        ) :: call_response()
  def call(zephyr, prompt, functions \\ [], callback_fn \\ nil)

  def call(%ChatZephyr{} = zephyr, prompt, functions, callback_fn) when is_binary(prompt) do
    messages = [
      Message.new_system!(),
      Message.new_user!(prompt)
    ]

    call(zephyr, messages, functions, callback_fn)
  end

  def call(%ChatZephyr{} = zephyr, messages, functions, callback_fn) when is_list(messages) do
    if override_api_return?() do
      Logger.warning("Found override API response. Will not make live API call.")

      case get_api_override() do
        {:ok, {:ok, data} = response} ->
          # fire callback for fake responses too
          fire_callback(zephyr, data, callback_fn)
          response

        _other ->
          raise LangChainError,
                "An unexpected fake API response was set. Should be an `{:ok, value}`"
      end
    else
      try do
        # make base api request and perform high-level success/failure checks
        case do_serving_request(zephyr, messages, functions, callback_fn) do
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
  end

  # Make the request from the bumblebee zephyr serving.
  #
  # The result of the function is:
  #
  # - `result` - where `result` is a data-structure like a list or map.
  # - `{:error, reason}` - Where reason is a string explanation of what went wrong.
  #
  # If a callback_fn is provided, it will fire with each

  # When `stream: true` is If `stream: false`, the completed message is
  # returned.
  #
  # If `stream: true`, the `callback_fn` is executed for the returned
  # MessageDelta responses. The deltas are accumulated and merged together into
  # the complete Message.
  #
  # Executes the callback function passing the response only parsed to the data
  # structures.
  @doc false
  @spec do_serving_request(t(), [Message.t()], [Function.t()], callback_fn()) ::
          list() | struct() | {:error, String.t()}
  def do_serving_request(%ChatZephyr{stream: false} = zephyr, messages, _functions, callback_fn) do
    prompt = messages_to_prompt(messages)

    raw_response =
      case Nx.Serving.batched_run(zephyr.serving, prompt) do
        # model serving set to stream: false. Received map response. Extract data.
        %{results: [%{text: content}]} ->
          content

        stream ->
          # Model serving setup for streaming. Requested to not stream response.
          # Consume the full stream and return as the content.
          Enum.reduce(stream, "", fn data, acc -> acc <> data end)
      end

    case Message.new(%{role: :assistant, status: :complete, content: raw_response}) do
      {:ok, message} ->
        # execute the callback with the final message
        fire_callback(zephyr, message, callback_fn)
        # return a list of the complete message. As a list for compatibility.
        [message]

      {:error, changeset} ->
        reason = Utils.changeset_error_to_string(changeset)
        Logger.error("Failed to create non-streamed full message: #{inspect(reason)}")
        {:error, reason}
    end
  end

  def do_serving_request(%ChatZephyr{stream: true} = zephyr, messages, _functions, callback_fn) do
    # Create the content from the messages.
    # prompt = messages_to_prompt(messages)
    prompt = LangChain.Utils.ChatTemplates.apply_chat_template!(messages, :zephyr)

    case Nx.Serving.batched_run(zephyr.serving, prompt) do
      # requested a stream but received a non-streaming result.
      %{results: _results} ->
        raise LangChainError, "Served model did not return a stream"

      stream ->
        # process the stream in to MessageDeltas. Fire off the callbacks as they
        # are received. It accumulates the deltas into a final combined
        # MessageDelta.
        final_delta = stream_to_deltas!(zephyr, stream, callback_fn)

        # fire the callback of the completed message
        # Assuming it's complete at this point.
        # Want a `:done` message to know it's officially complete.
        # https://github.com/elixir-nx/bumblebee/issues/287
        case MessageDelta.to_message(%MessageDelta{final_delta | status: :complete}) do
          {:ok, message} ->
            # TODO: NEED TO TEST FOR RECEIVING A FUNCTION EXECUTION

            # execute the callback with the final message
            fire_callback(zephyr, message, callback_fn)
            # return a list of the complete message. For compatibility.
            [message]

          {:error, reason} ->
            Logger.error("Failed to convert deltas to full message: #{inspect(reason)}")
            {:error, reason}
        end
    end
  end

  def message_to_text(%Message{role: :system} = message, text) do
    # to try and prevent prompt manipulation, even though the system message
    # should be the first, include whatever else came before
    text <> "#{@system_tag}\n#{message.content}#{@text_end_tag}\n"
  end

  def message_to_text(%Message{role: :user} = message, text) do
    # filter out special meaning tags from what the user can submit
    excluded_tags = [@system_tag, @user_tag, @assistant_tag, @text_end_tag]

    user_content =
      Enum.reduce(excluded_tags, message.content, fn exclude, final ->
        String.replace(final, exclude, "", global: true)
      end)

    text <> "#{@user_tag}\n#{user_content}#{@text_end_tag}\n#{@assistant_tag}\n"
  end

  def message_to_text(%Message{role: :assistant} = message, text) do
    text <> "#{message.content}\n"
  end

  @doc """
  Convert the list of messages into the expected single text format for a chat
  prompt.
  """
  @spec messages_to_prompt([Message.t()]) :: String.t()
  def messages_to_prompt(messages) do
    Enum.reduce(messages, "", fn msg, acc ->
      message_to_text(msg, acc)
    end)
  end

  @spec stream_to_deltas!(t(), Stream.t(), callback_fn()) :: MessageDelta.t() | no_return()
  def stream_to_deltas!(zephyr, stream, callback_fn) do
    Enum.reduce(stream, nil, fn data, acc ->
      new_delta =
        case MessageDelta.new(%{role: :assistant, content: data}) do
          {:ok, delta} ->
            delta

          {:error, changeset} ->
            reason = Utils.changeset_error_to_string(changeset)

            Logger.error(
              "Failed to process received Zephyr MessageDelta data: #{inspect(reason)}"
            )

            raise LangChainError, reason
        end

      # processed the delta, fire the callback
      fire_callback(zephyr, new_delta, callback_fn)

      # merge the delta to accumulate the full message
      case acc do
        # first time through. Set delta as the initial message chunk
        nil ->
          new_delta

        %MessageDelta{} = _previous ->
          MessageDelta.merge_delta(acc, new_delta)
      end
    end)
  end

  # fire the callback if present.
  @spec fire_callback(
          t(),
          data :: callback_data(),
          nil | callback_fn()
        ) :: :ok
  defp fire_callback(%ChatZephyr{stream: true}, _data, nil) do
    Logger.warning("Streaming call requested but no callback function was given.")
    :ok
  end

  defp fire_callback(%ChatZephyr{stream: false}, _data, nil), do: :ok

  defp fire_callback(%ChatZephyr{}, data, callback_fn) when is_function(callback_fn) do
    # OPTIONAL: Execute callback function
    callback_fn.(data)
    :ok
  end
end
