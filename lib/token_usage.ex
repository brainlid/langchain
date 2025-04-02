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

  Refer to the `raw` token usage information for access to LLM-specific information that may be available.

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
  @required_fields []

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

  @doc """
  Combines two TokenUsage structs by adding their respective input and output values.
  The raw maps are merged, with values being added if they are numeric.

  If both arguments are nil, returns nil.
  If one argument is nil, returns the non-nil argument.

  ## Example

      iex> usage1 = LangChain.TokenUsage.new!(%{input: 10, output: 20, raw: %{"total_tokens" => 30}})
      iex> usage2 = LangChain.TokenUsage.new!(%{input: 5, output: 15, raw: %{"total_tokens" => 20}})
      iex> combined = LangChain.TokenUsage.add(usage1, usage2)
      iex> combined.input
      15
      iex> combined.output
      35
      iex> combined.raw["total_tokens"]
      50

  """
  @spec add(t() | nil, t() | nil) :: t() | nil
  def add(nil, nil), do: nil
  def add(nil, usage), do: usage
  def add(usage, nil), do: usage
  def add(%TokenUsage{} = usage1, %TokenUsage{} = usage2) do
    new!(%{
      input: (usage1.input || 0) + (usage2.input || 0),
      output: (usage1.output || 0) + (usage2.output || 0),
      raw: merge_raw_values(usage1.raw, usage2.raw)
    })
  end

  defp merge_raw_values(raw1, raw2) do
    Map.merge(raw1, raw2, fn _k, v1, v2 ->
      if is_number(v1) and is_number(v2), do: v1 + v2, else: v2
    end)
  end

  @doc """
  Extracts token usage information from a `LangChain.Message` or
  `LangChain.MessageDelta` struct's metadata. Returns nil if no token usage
  information is found.

  ## Example

      iex> message = %LangChain.Message{metadata: %{usage: %LangChain.TokenUsage{input: 10, output: 20}}}
      iex> LangChain.TokenUsage.get(message)
      %LangChain.TokenUsage{input: 10, output: 20}

      iex> message = %LangChain.Message{metadata: %{}}
      iex> LangChain.TokenUsage.get(message)
      nil

  """
  @spec get(%{metadata: %{usage: t()}} | %{metadata: map() | nil}) :: t() | nil
  def get(%{metadata: %{usage: %TokenUsage{} = usage}}), do: usage
  def get(_), do: nil
end
