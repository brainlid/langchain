defmodule LangChain.ChatModels.ChatBumblebee do
  @moduledoc """
  Represents a chat model hosted and accessed through Bumblebee.

  Many types of models can be hosted through Bumblebee, so this attempts to
  represent the most common features and provide a single implementation when
  possible.
  """
  use Ecto.Schema
  require Logger
  import Ecto.Changeset
  import LangChain.Utils.ApiOverride
  alias __MODULE__
  alias LangChain.ChatModels.ChatModel
  alias LangChain.Message
  alias LangChain.LangChainError
  alias LangChain.Utils
  alias LangChain.MessageDelta
  alias LangChain.Utils.ChatTemplates

  @behaviour ChatModel

  @primary_key false
  embedded_schema do
    # Name of the Nx.Serving to use when working with the LLM.
    field :serving, :any, virtual: true

    # # What sampling temperature to use, between 0 and 2. Higher values like 0.8
    # # will make the output more random, while lower values like 0.2 will make it
    # # more focused and deterministic.
    # field :temperature, :float, default: 1.0

    field :template_format, Ecto.Enum, values: [:inst, :im_start, :zephyr, :llama_2]

    # The bumblebee model may compile differently based on the stream true/false
    # option on the serving. Therefore, streaming should be enabled on the
    # serving and a stream option here can change the way data is received in
    # code. - https://github.com/elixir-nx/bumblebee/issues/295

    field :stream, :boolean, default: true

    # Seed for randomizing behavior or giving more deterministic output. Helpful
    # for testing.
    field :seed, :integer
  end

  @type t :: %ChatBumblebee{}

  # @type call_response :: {:ok, Message.t() | [Message.t()]} | {:error, String.t()}
  # @type callback_data ::
  #         {:ok, Message.t() | MessageDelta.t() | [Message.t() | MessageDelta.t()]}
  #         | {:error, String.t()}
  @type callback_fn :: (Message.t() | MessageDelta.t() -> any())

  @create_fields [
    :serving,
    # :temperature,
    :seed,
    :template_format,
    :stream
  ]
  @required_fields [:serving]

  @doc """
  Setup a ChatBumblebee client configuration.
  """
  @spec new(attrs :: map()) :: {:ok, t} | {:error, Ecto.Changeset.t()}
  def new(%{} = attrs \\ %{}) do
    %ChatBumblebee{}
    |> cast(attrs, @create_fields)
    |> common_validation()
    |> apply_action(:insert)
  end

  @doc """
  Setup a ChatBumblebee client configuration and return it or raise an error if invalid.
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

  # @spec call(
  #         t(),
  #         String.t() | [Message.t()],
  #         [LangChain.Function.t()],
  #         nil | callback_fn()
  #       ) :: call_response()
  @impl ChatModel
  def call(model, prompt, functions \\ [], callback_fn \\ nil)

  def call(%ChatBumblebee{} = model, prompt, functions, callback_fn) when is_binary(prompt) do
    messages = [
      Message.new_system!(),
      Message.new_user!(prompt)
    ]

    call(model, messages, functions, callback_fn)
  end

  def call(%ChatBumblebee{} = model, messages, functions, callback_fn)
      when is_list(messages) do
    if override_api_return?() do
      Logger.warning("Found override API response. Will not make live API call.")

      case get_api_override() do
        {:ok, {:ok, data} = response} ->
          # fire callback for fake responses too
          Utils.fire_callback(model, data, callback_fn)
          response

        _other ->
          raise LangChainError,
                "An unexpected fake API response was set. Should be an `{:ok, value}`"
      end
    else
      try do
        # make base api request and perform high-level success/failure checks
        case do_serving_request(model, messages, functions, callback_fn) do
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

  @doc false
  @spec do_serving_request(t(), [Message.t()], [Function.t()], callback_fn()) ::
          list() | struct() | {:error, String.t()}
  def do_serving_request(
        %ChatBumblebee{stream: false} = model,
        messages,
        _functions,
        callback_fn
      ) do
    prompt = ChatTemplates.apply_chat_template!(messages, model.template_format)

    raw_response =
      case Nx.Serving.batched_run(model.serving, prompt) do
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
        Utils.fire_callback(model, message, callback_fn)
        # return a list of the complete message. As a list for compatibility.
        [message]

      {:error, changeset} ->
        reason = Utils.changeset_error_to_string(changeset)
        Logger.error("Failed to create non-streamed full message: #{inspect(reason)}")
        {:error, reason}
    end
  end

  def do_serving_request(
        %ChatBumblebee{stream: true} = model,
        messages,
        _functions,
        callback_fn
      ) do
    # Create the content from the messages.
    # prompt = messages_to_prompt(messages)
    prompt = ChatTemplates.apply_chat_template!(messages, model.template_format)

    case Nx.Serving.batched_run(model.serving, prompt) do
      # requested a stream but received a non-streaming result.
      %{results: _results} ->
        raise LangChainError, "Served model did not return a stream"

      stream ->
        # process the stream in to MessageDeltas. Fire off the callbacks as they
        # are received. It accumulates the deltas into a final combined
        # MessageDelta.
        final_delta = stream_to_deltas!(model, stream, callback_fn)

        # fire the callback of the completed message
        # Assuming it's complete at this point.
        # Want a `:done` message to know it's officially complete.
        # https://github.com/elixir-nx/bumblebee/issues/287
        case MessageDelta.to_message(%MessageDelta{final_delta | status: :complete}) do
          {:ok, message} ->
            # execute the callback with the final message
            Utils.fire_callback(model, message, callback_fn)
            # return a list of the complete message. For compatibility.
            [message]

          {:error, reason} ->
            Logger.error("Failed to convert deltas to full message: #{inspect(reason)}")
            {:error, reason}
        end
    end
  end

  @spec stream_to_deltas!(t(), Stream.t(), callback_fn()) :: nil | MessageDelta.t() | no_return()
  def stream_to_deltas!(model, stream, callback_fn) do
    Enum.reduce(stream, nil, fn data, acc ->
      new_delta =
        case MessageDelta.new(%{role: :assistant, content: data}) do
          {:ok, delta} ->
            delta

          {:error, changeset} ->
            reason = Utils.changeset_error_to_string(changeset)

            Logger.error(
              "Failed to process received model's MessageDelta data: #{inspect(reason)}"
            )

            raise LangChainError, reason
        end

      # processed the delta, fire the callback
      Utils.fire_callback(model, new_delta, callback_fn)

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
end
