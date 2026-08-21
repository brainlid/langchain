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
  alias __MODULE__
  alias LangChain.LangChainError

  @primary_key false
  embedded_schema do
    field :input, :integer
    field :output, :integer
    field :raw, :map, default: %{}
    # For token usage attached to a MessageDelta, whether the token usage is cumulative
    # (all tokens for the message so far) or just the token usage for this delta.
    # Varies by model.
    field :cumulative, :boolean, default: false
  end

  @type t :: %TokenUsage{}

  @create_fields [:input, :output, :raw, :cumulative]
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
  Combines two TokenUsage structs by adding their respective input and output
  values. The raw maps are merged, with numeric values added at every depth, so
  map-valued usage details such as `prompt_tokens_details` and
  `completion_tokens_details` accumulate per-key rather than the later map
  replacing the earlier one. A detail reported as a *list* of per-modality
  objects, the shape Gemini uses for `promptTokensDetails`, offers no key to
  accumulate against and is taken from the later usage whole.

  When the second argument is marked `cumulative: true` it is a running total for
  the message rather than an increment, so it supersedes the accumulator instead
  of being added to it. Because a running total never decreases, the larger of
  the two readings is kept per field: a reading that omits a class, or reports it
  as zero, cannot erase what an earlier reading already established.

  This combines two readings of the *same* message. To total usage across
  different messages, use `add_total/2`.

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

  # A `cumulative: true` usage is a running total for the message so far, not an
  # increment, so it supersedes the accumulator instead of being added to it.
  #
  # Token counts are counters, so a running total never decreases while a message
  # is being generated. Providers differ in how completely each reading repeats
  # the picture: some report every class every time, some report only the classes
  # that changed, and some normalize an unreported class to zero. Keeping the
  # larger of the two readings per field is correct under all three -- a fuller
  # reading advances the total, and a partial one cannot erase a class it never
  # meant to describe.
  def add(%TokenUsage{} = usage1, %TokenUsage{cumulative: true} = usage2) do
    %TokenUsage{
      usage2
      | input: running_max(usage1.input, usage2.input),
        output: running_max(usage1.output, usage2.output),
        raw: merge_cumulative_raw(usage1.raw || %{}, usage2.raw || %{})
    }
  end

  def add(%TokenUsage{} = usage1, %TokenUsage{} = usage2) do
    new!(%{
      input: (usage1.input || 0) + (usage2.input || 0),
      output: (usage1.output || 0) + (usage2.output || 0),
      raw: merge_raw_values(usage1.raw || %{}, usage2.raw || %{})
    })
  end

  @doc """
  Returns the usage with the `:cumulative` flag cleared.

  `:cumulative` describes the relationship between a streaming delta and the
  *other deltas of the same message*. Once deltas are merged into a message the
  flag has served its purpose, and carrying it into a total across messages makes
  `add/2` supersede the accumulator and report only the last message.

  Returns non-`TokenUsage` values (including `nil`) unchanged.
  """
  # Two specs, because the catch-all clause makes this intentionally total: a
  # caller can pipe a `get/1` result straight through without a nil check, and
  # anything that is not a `%TokenUsage{}` comes back as the type it went in as.
  @spec clear_cumulative(t()) :: t()
  @spec clear_cumulative(other) :: other when other: any()
  def clear_cumulative(%TokenUsage{} = usage), do: %TokenUsage{usage | cumulative: false}
  def clear_cumulative(other), do: other

  @doc """
  Adds one completed message's usage to a running total across messages.

  Use this, rather than `add/2`, whenever the two operands are totals for
  *different* messages. Each assembled message's usage is already final for that
  message, so the values are summed, and `:cumulative` is cleared first because
  it only describes delta-to-delta relationships within a single message.

  ## Example

      iex> alias LangChain.TokenUsage
      iex> first = TokenUsage.new!(%{input: 100, output: 20, cumulative: true})
      iex> second = TokenUsage.new!(%{input: 150, output: 35, cumulative: true})
      iex> total = TokenUsage.add_total(first, second)
      iex> {total.input, total.output, total.cumulative}
      {250, 55, false}

  """
  @spec add_total(t() | nil, t() | nil) :: t() | nil
  def add_total(accumulator, usage) do
    add(clear_cumulative(accumulator), clear_cumulative(usage))
  end

  # Raw usage maps nest: OpenAI-shaped providers report cached and reasoning
  # tokens under `prompt_tokens_details` / `input_tokens_details` /
  # `completion_tokens_details`. Recursing means those inner counts are summed
  # per-key at every depth instead of the earlier map being dropped for the
  # later one. Only maps are traversed. Gemini reports the same class of detail
  # as a list of per-modality objects, which has no key to accumulate against,
  # so the later value is taken whole.
  defp merge_raw_values(raw1, raw2) do
    Map.merge(raw1, raw2, fn
      _k, v1, v2 when is_number(v1) and is_number(v2) ->
        v1 + v2

      _k, v1, v2 ->
        if plain_map?(v1) and plain_map?(v2), do: merge_raw_values(v1, v2), else: v2
    end)
  end

  # Same traversal as `merge_raw_values/2`, but keeping the larger of two counts
  # rather than their sum, because both sides describe the same running total.
  defp merge_cumulative_raw(raw1, raw2) do
    Map.merge(raw1, raw2, fn
      _k, v1, v2 when is_number(v1) and is_number(v2) ->
        max(v1, v2)

      _k, v1, v2 ->
        if plain_map?(v1) and plain_map?(v2), do: merge_cumulative_raw(v1, v2), else: v2
    end)
  end

  # `max/2` compares across types, and every atom sorts above every number, so a
  # bare `max(nil, 5)` returns `nil`. An unreported count is not a larger count.
  defp running_max(nil, value), do: value
  defp running_max(value, nil), do: value
  defp running_max(v1, v2), do: max(v1, v2)

  # Structs are maps, but merging them key-by-key would produce a malformed
  # struct rather than a sum, so only bare maps recurse.
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
