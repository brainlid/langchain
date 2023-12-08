defmodule LangChain.ChatModels.ChatBumbleModel do
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
  alias LangChain.Message
  alias LangChain.LangChainError
  alias LangChain.Utils
  alias LangChain.MessageDelta

  # allow up to 2 minutes for response.
  # @receive_timeout 60_000

  @primary_key false
  embedded_schema do
    # Name of the Nx.Serving to use when working with the LLM.
    field :serving, :any, virtual: true

    # # What sampling temperature to use, between 0 and 2. Higher values like 0.8
    # # will make the output more random, while lower values like 0.2 will make it
    # # more focused and deterministic.
    # field :temperature, :float, default: 1.0

    # # Seed for randomizing behavior or giving more deterministic output. Helpful for testing.
    # field :seed, :integer

    field :template_format, Ecto.Enum, values: [:inst, :im_start, :zephyr, :llama_2]

    # Track if the model supports functions.
    field :supports_functions, :boolean, default: false

    # The bumblebee model may compile differently based on the stream true/false
    # option on the serving. Therefore, streaming should be enabled on the
    # serving and a stream option here can change the way data is received in
    # code. - https://github.com/elixir-nx/bumblebee/issues/295

    field :stream, :boolean, default: true
  end

  @type t :: %ChatBumbleModel{}

  @type call_response :: {:ok, Message.t() | [Message.t()]} | {:error, String.t()}
  @type callback_data ::
          {:ok, Message.t() | MessageDelta.t() | [Message.t() | MessageDelta.t()]}
          | {:error, String.t()}
  @type callback_fn :: (Message.t() | MessageDelta.t() -> any())

  @create_fields [
    :serving,
    # :temperature,
    # :seed,
    :template_format,
    :supports_functions,
    :stream
  ]
  @required_fields [:serving]

  # Tags used for formatting the messages. Don't allow the user to include these
  # themselves.
  @system_tag "<|system|>"
  @user_tag "<|user|>"
  @assistant_tag "<|assistant|>"
  @text_end_tag "</s>"

  @doc """
  Setup a ChatBumbleModel client configuration.
  """
  @spec new(attrs :: map()) :: {:ok, t} | {:error, Ecto.Changeset.t()}
  def new(%{} = attrs \\ %{}) do
    %ChatBumbleModel{}
    |> cast(attrs, @create_fields)
    |> common_validation()
    |> apply_action(:insert)
  end

  @doc """
  Setup a ChatBumbleModel client configuration and return it or raise an error if invalid.
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

  def new_mistral() do
    # https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.1
  end
end
