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

  ## Combining usage

  Two different questions call for two different combinations, and the caller
  knows which one it is asking:

    * `add/2` combines several readings of the **same** message, as a streamed
      response produces. Each reading is a snapshot of the message so far, so
      the combination keeps the largest count per field.

    * `add_total/2` combines the final usage of **different** messages into a
      running total for a conversation. Each operand is already settled for its
      own message, so the combination sums them.

  """
  use Ecto.Schema
  import Ecto.Changeset
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

  A count a provider never reported is `nil` and contributes nothing to the
  total. Anthropic's closing `message_delta` event, for one, carries only
  `output_tokens`, so a usage built from it alone has no input count.
  """
  @spec total(t()) :: integer()
  def total(%TokenUsage{} = usage) do
    (usage.input || 0) + (usage.output || 0)
  end

  @doc """
  Combines two readings of the same message into one.

  A streamed response reports usage more than once, and each reading is a
  snapshot of the message so far rather than a description of one delta's share.
  Anthropic opens with the input classes on `message_start` and closes with a
  total on `message_delta`; Gemini repeats the running totals on every chunk;
  OpenAI-shaped providers answer with a single snapshot on a usage-only terminal
  chunk. Combining snapshots keeps the larger of the two readings per field.

  Token counts are counters, so a count never decreases while a message is being
  generated. Providers differ in how completely each reading repeats the
  picture: some report every class every time, some report only the classes that
  changed, and some normalize an unreported class to zero. Keeping the larger
  reading is correct under all three -- a fuller reading advances the total, and
  a partial one cannot erase a class it never meant to describe.

  The `raw` maps are merged the same way, at every depth, so map-valued usage
  details such as `prompt_tokens_details` and `completion_tokens_details` advance
  per-key rather than the later map replacing the earlier one. A detail reported
  as a *list* of per-modality objects, the shape Gemini uses for
  `promptTokensDetails`, offers no key to advance against and is taken from the
  later usage whole.

  Keeping the larger count is sound for a primitive counter. A `raw` key holding
  a value *derived* from other counts within a single reading, such as a
  `total_tokens` an adapter computes as input + output, is only meaningful
  alongside the reading it came from and does not survive the merge intact. Read
  the total from `total/1` rather than from `raw`.

  To total usage across different messages, use `add_total/2`.

  If both arguments are nil, returns nil.
  If one argument is nil, returns the non-nil argument.

  ## Example

      iex> alias LangChain.TokenUsage
      iex> opening = TokenUsage.new!(%{input: 25, output: 1, raw: %{"input_tokens" => 25}})
      iex> closing = TokenUsage.new!(%{output: 15, raw: %{"output_tokens" => 15}})
      iex> combined = TokenUsage.add(opening, closing)
      iex> {combined.input, combined.output}
      {25, 15}
      iex> combined.raw
      %{"input_tokens" => 25, "output_tokens" => 15}

  """
  @spec add(t() | nil, t() | nil) :: t() | nil
  def add(nil, nil), do: nil
  def add(nil, usage), do: usage
  def add(usage, nil), do: usage

  def add(%TokenUsage{} = usage1, %TokenUsage{} = usage2) do
    %TokenUsage{
      usage2
      | input: running_max(usage1.input, usage2.input),
        output: running_max(usage1.output, usage2.output),
        raw: merge_snapshot_raw(usage1.raw || %{}, usage2.raw || %{})
    }
  end

  @doc """
  Adds one completed message's usage to a running total across messages.

  Use this, rather than `add/2`, whenever the two operands are totals for
  *different* messages. Each assembled message's usage is already final for that
  message, so the values are summed, at every depth of the `raw` map.

  If both arguments are nil, returns nil.
  If one argument is nil, returns the non-nil argument.

  ## Example

      iex> alias LangChain.TokenUsage
      iex> first = TokenUsage.new!(%{input: 100, output: 20})
      iex> second = TokenUsage.new!(%{input: 150, output: 35})
      iex> total = TokenUsage.add_total(first, second)
      iex> {total.input, total.output}
      {250, 55}

  """
  @spec add_total(t() | nil, t() | nil) :: t() | nil
  def add_total(nil, nil), do: nil
  def add_total(nil, usage), do: usage
  def add_total(usage, nil), do: usage

  def add_total(%TokenUsage{} = usage1, %TokenUsage{} = usage2) do
    new!(%{
      input: (usage1.input || 0) + (usage2.input || 0),
      output: (usage1.output || 0) + (usage2.output || 0),
      raw: sum_raw_values(usage1.raw || %{}, usage2.raw || %{})
    })
  end

  # Raw usage maps nest: OpenAI-shaped providers report cached and reasoning
  # tokens under `prompt_tokens_details` / `input_tokens_details` /
  # `completion_tokens_details`. Recursing means those inner counts are summed
  # per-key at every depth instead of the earlier map being dropped for the
  # later one. Only maps are traversed. Gemini reports the same class of detail
  # as a list of per-modality objects, which has no key to accumulate against,
  # so the later value is taken whole.
  defp sum_raw_values(raw1, raw2) do
    Map.merge(raw1, raw2, fn
      _k, v1, v2 when is_number(v1) and is_number(v2) ->
        v1 + v2

      _k, v1, v2 ->
        if plain_map?(v1) and plain_map?(v2), do: sum_raw_values(v1, v2), else: v2
    end)
  end

  # Same traversal as `sum_raw_values/2`, but keeping the larger of two counts
  # rather than their sum, because both sides describe the same message.
  defp merge_snapshot_raw(raw1, raw2) do
    Map.merge(raw1, raw2, fn
      _k, v1, v2 when is_number(v1) and is_number(v2) ->
        max(v1, v2)

      _k, v1, v2 ->
        if plain_map?(v1) and plain_map?(v2), do: merge_snapshot_raw(v1, v2), else: v2
    end)
  end

  # `max/2` compares across types, and every atom sorts above every number, so a
  # bare `max(nil, 5)` returns `nil`. An unreported count is not a larger count.
  defp running_max(nil, value), do: value
  defp running_max(value, nil), do: value
  defp running_max(v1, v2), do: max(v1, v2)

  # Structs are maps, but merging them key-by-key would produce a malformed
  # struct rather than a combination, so only bare maps recurse.
  defp plain_map?(value), do: is_map(value) and not is_struct(value)

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
  @spec get(any()) :: t() | nil
  def get(%{metadata: %{usage: %TokenUsage{} = usage}}), do: usage
  def get(_), do: nil

  @doc """
  Sets the token usage information on a `LangChain.Message` or
  `LangChain.MessageDelta` struct in the `metadata` under the `:usage` key.

  ## Example

      iex> message = %LangChain.Message{metadata: %{}}
      iex> token_usage = %LangChain.TokenUsage{input: 10, output: 20}
      iex> LangChain.TokenUsage.set(message, token_usage)
      %LangChain.Message{metadata: %{usage: %LangChain.TokenUsage{input: 10, output: 20}}}
  """
  # The first argument is typed as `any()` because `set/2` is intentionally
  # total: the catch-all clause below returns non-message values unchanged, so
  # callers can safely pipe any `do_process_response/2` shape (e.g. `:skip`,
  # `{:error, _}`, a message, or a list) through it.
  @spec set(any(), nil | t()) :: any()
  def set(%{metadata: metadata} = message, %TokenUsage{} = usage) do
    new_metadata =
      if metadata == nil do
        %{usage: usage}
      else
        Map.put(metadata, :usage, usage)
      end

    %{message | metadata: new_metadata}
  end

  def set(message, _), do: message

  @doc """
  Sets the token usage information on a `LangChain.Message` or
  `LangChain.MessageDelta` struct when wrapped in an :ok,:error tuple in the `metadata` under the `:usage` key.
  """
  @spec set_wrapped({:ok, %{metadata: nil | map()}} | {:error, any()} | any(), nil | t()) ::
          {:ok, %{metadata: %{usage: t()}}} | {:error, any()} | any()
  def set_wrapped({:ok, message}, usage) do
    {:ok, set(message, usage)}
  end

  def set_wrapped(message, _), do: message
end
