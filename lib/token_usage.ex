defmodule LangChain.TokenUsage do
  @moduledoc """
  Contains token usage information returned from an LLM.

  ## Example

      %TokenUsage{
        input: 30,
        output: 15,
        raw: %{
          "total_tokens" => 29
        }
      }

  Input is the tokens from the prompt. Output is the completion or generated
  tokens returned.

  """
  use Ecto.Schema
  import Ecto.Changeset
  require Logger
  alias __MODULE__
  alias LangChain.LangChainError

  @primary_key false
  embedded_schema do
    field :input, :integer
    field :output, :integer
    field :raw, :map, default: %{}
  end

  @type t :: %TokenUsage{}

  @create_fields [:input, :output, :raw]
  # Anthropic returns only the output token count when streaming deltas
  @required_fields [:output]

  @doc """
  Build a new TokenUsage and return an `:ok`/`:error` tuple with the result.
  """
  @spec new(attrs :: map()) :: {:ok, t()} | {:error, Ecto.Changeset.t()}
  def new(attrs \\ %{}) do
    %TokenUsage{}
    |> cast(attrs, @create_fields)
    |> common_validations()
    |> apply_action(:insert)
  end

  @doc """
  Build a new TokenUsage and return it or raise an error if invalid.
  """
  @spec new!(attrs :: map()) :: t() | no_return()
  def new!(attrs \\ %{}) do
    case new(attrs) do
      {:ok, usage} ->
        usage

      {:error, changeset} ->
        raise LangChainError, changeset
    end
  end

  defp common_validations(changeset) do
    changeset
    |> validate_required(@required_fields)
    |> validate_number(:input, greater_than_or_equal_to: 0)
    |> validate_number(:output, greater_than_or_equal_to: 0)
  end

  @doc """
  Return the total token usage amount. The total is the sum of input and output.
  """
  @spec total(t()) :: integer()
  def total(%TokenUsage{} = usage) do
    usage.input + usage.output
  end
end
