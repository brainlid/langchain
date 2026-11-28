# Parsing and Pre-Checking Tool Arguments

When an LLM calls one of your tools, the arguments arrive as decoded JSON: a
map with string keys, holding whatever values the model chose to send. A
model can send a number as a string, misspell an enum value, leave out a
field, or refer to a record that doesn't exist or that belongs to someone
else.

The `:parse_args` option on `LangChain.Function` gives every tool one place to
deal with that before the tool runs. A parser can:

- check the shape and types of the arguments, and coerce them into the form
  your code wants (atom keys, integers, structs)
- look up the records the arguments refer to, for the user the call is made
  for, and refuse the call when something is wrong
- hand the parsed result to the tool's `function`, so the body works with
  clean data and never parses it again

This guide starts with the 1-arity form, which only sees the arguments. Then
it covers the 2-arity form, which also gets the tool context, and shows how to
use it to check a call against the requesting user's data. The last part shows
how a call is pre-checked before anyone is asked to approve it, which is where
the 2-arity form helps most.

## What happens without a parser

A function declared with `parameters` or `parameters_schema` gets a single
check at execute time: every top-level required key must be present. Types,
enums, formats, and nested shapes are not enforced. The tool's `function`
receives the arguments exactly as the model sent them.

When you supply `:parse_args`, the parser replaces that check. LangChain does
no argument validation of its own for that tool, so your parser can report
missing keys and type problems together, in its own words, in one round trip.

## The parser contract

A parser is a 1- or 2-arity function. It receives the raw, string-keyed
arguments map from the LLM (an empty map when the model sent no arguments) and
returns one of three shapes:

```elixir
:ok                           # the arguments are fine; pass them to `function` unchanged
{:ok, parsed_arguments}       # pass this map (or struct) to `function` instead
{:error, reason}              # refuse the call; `reason` is sent to the LLM
```

When the parser returns `{:error, reason}`:

- the tool's `function` does **not** run
- `reason` becomes the tool result the model sees, so it can correct the call
  and try again
- the `LangChain.Message.ToolResult` is marked `is_error: true` and
  `is_refusal: true`. A refusal means the call was turned down before it
  changed anything, so resending a corrected call is safe.
- tool callbacks such as `:on_tool_response_created` and the
  `[:langchain, :tool, :call]` telemetry span still fire, so refused calls show
  up in your logs, metrics, and trajectories

Write `reason` for the model. Name the argument, say what was wrong, and when
you can, say what would be accepted.

A parser that raises, or returns some other shape, doesn't crash the chain.
The exception is rescued and the call is answered with an error, the same way
an exception in the tool body is.

## Parsing the arguments: `parse_args/1`

A 1-arity parser only sees the arguments. Use it for shape and type checks.

LangChain doesn't depend on any schema library. Anything that can turn a map
into `{:ok, value}` or an error message works. The examples below use an Ecto
embedded schema and Zoi.

### With an Ecto embedded schema

Ecto is already a dependency of LangChain, and a schemaless or embedded
changeset is a natural fit for casting LLM arguments.

```elixir
defmodule MyApp.Tools.SearchProducts do
  use Ecto.Schema
  import Ecto.Changeset

  alias LangChain.Function

  @primary_key false
  embedded_schema do
    field :query, :string
    field :category, Ecto.Enum, values: [:books, :music, :games]
    field :limit, :integer, default: 10
  end

  def function do
    Function.new!(%{
      name: "search_products",
      description: "Search the product catalog.",
      parameters_schema: %{
        type: "object",
        properties: %{
          query: %{type: "string", description: "Search terms"},
          category: %{type: "string", enum: ["books", "music", "games"]},
          limit: %{type: "integer", description: "Max results, 1 to 50. Defaults to 10."}
        },
        required: ["query"]
      },
      parse_args: &parse_args/1,
      function: &execute/2
    })
  end

  def parse_args(args) do
    %__MODULE__{}
    |> cast(args, [:query, :category, :limit])
    |> validate_required([:query])
    |> validate_number(:limit, greater_than_or_equal_to: 1, less_than_or_equal_to: 50)
    |> apply_action(:parse)
    |> case do
      {:ok, params} -> {:ok, params}
      {:error, changeset} -> {:error, format_errors(changeset)}
    end
  end

  # The body receives the struct `parse_args/1` returned, not the raw map.
  defp execute(%__MODULE__{} = params, _context) do
    results = MyApp.Catalog.search(params.query, category: params.category, limit: params.limit)
    {:ok, Jason.encode!(results)}
  end

  defp format_errors(changeset) do
    changeset
    |> traverse_errors(fn {msg, opts} ->
      Enum.reduce(opts, msg, fn {key, value}, acc ->
        String.replace(acc, "%{#{key}}", to_string(value))
      end)
    end)
    |> Enum.map_join("; ", fn {field, messages} ->
      "#{field}: #{Enum.join(messages, ", ")}"
    end)
  end
end
```

If the model sends `%{"query" => "elixir", "limit" => "200"}`, the call is
refused, and the model is told `limit: must be less than or equal to 50`. If
it sends `%{"query" => "elixir", "category" => "books"}`, `execute/2` receives
`%SearchProducts{query: "elixir", category: :books, limit: 10}`: the category
has been cast to an atom and the default limit filled in.

### With Zoi

[Zoi](https://hex.pm/packages/zoi) returns `{:ok, parsed}` or
`{:error, [%Zoi.Error{}]}`, so the adapter only has to turn the error list
into a string. Add `:zoi` to your own dependencies to use it.

```elixir
defmodule MyApp.Tools.DiagnoseDevice do
  alias LangChain.Function

  @schema Zoi.object(
            %{
              device_id: Zoi.string(min_length: 1),
              verbose: Zoi.default(Zoi.boolean(), false)
            },
            coerce: true
          )

  def function do
    Function.new!(%{
      name: "diagnose_device",
      description: "Run diagnostics on a device.",
      parameters_schema: %{
        type: "object",
        properties: %{
          device_id: %{type: "string"},
          verbose: %{type: "boolean"}
        },
        required: ["device_id"]
      },
      parse_args: &parse_args/1,
      function: &execute/2
    })
  end

  def parse_args(args) do
    case Zoi.parse(@schema, args) do
      {:ok, parsed} ->
        {:ok, parsed}

      {:error, errors} ->
        {:error,
         Enum.map_join(errors, "; ", fn %Zoi.Error{path: path, message: message} ->
           "#{Enum.join(path, ".")}: #{message}"
         end)}
    end
  end

  # Zoi's `coerce: true` turns the string keys into atom keys.
  defp execute(%{device_id: device_id, verbose: verbose}, _context) do
    {:ok, MyApp.Devices.diagnose(device_id, verbose: verbose)}
  end
end
```

A missing `device_id` is refused with `device_id: is required`.

The JSON schema sent to the model (`parameters_schema`) and the parser are
declared separately. Keep them in agreement: the schema tells the model what
to send, and the parser is what is enforced.

## Checking the call against the user's data: `parse_args/2`

A valid shape doesn't mean the call can succeed. The order id may be well
formed but belong to another account. The order may exist but have already
shipped. The body could find that out and return an error, but by then the
call has already been accepted, and it may already have been shown to a
person for approval (see the next section).

A 2-arity parser is called as `parse_args.(arguments, context)`, with the same
context the tool body gets: the chain's `custom_context`, plus `:tool_call_id`.
Put the requesting user's scope in `custom_context`, and the parser can
resolve what the arguments refer to for that user:

```elixir
chain =
  %{llm: model, custom_context: %{scope: current_scope}}
  |> LLMChain.new!()
  |> LLMChain.add_tools([MyApp.Tools.CancelOrder.function()])
```

The parser can then:

- fetch the records the arguments name, through the scope, so a record from
  another account is simply not found
- refuse a call the data rules out, with a message that tells the model why
- return the records it fetched, so the body doesn't fetch them again

```elixir
defmodule MyApp.Tools.CancelOrder do
  alias LangChain.Function
  alias MyApp.Orders

  @cancellable [:pending, :processing]

  def function do
    Function.new!(%{
      name: "cancel_order",
      description: "Cancel one of the customer's orders.",
      parameters_schema: %{
        type: "object",
        properties: %{
          order_id: %{type: "string"},
          reason: %{type: "string", description: "Why the customer is cancelling"}
        },
        required: ["order_id", "reason"]
      },
      parse_args: &parse_args/2,
      function: &execute/2
    })
  end

  def parse_args(%{"order_id" => order_id, "reason" => reason}, %{scope: scope})
      when is_binary(order_id) and is_binary(reason) do
    # Read-only lookups. Nothing here changes any data.
    case Orders.get_order(scope, order_id) do
      nil ->
        {:error, "No order #{order_id} exists on this customer's account."}

      %{status: status} = order when status in @cancellable ->
        {:ok, %{order: order, reason: reason}}

      %{status: status} ->
        {:error,
         "Order #{order_id} is #{status} and can no longer be cancelled. " <>
           "Offer the customer a return instead."}
    end
  end

  def parse_args(_args, _context) do
    {:error, "cancel_order requires string values for both order_id and reason."}
  end

  # The body receives the order the parser loaded. It does the write.
  defp execute(%{order: order, reason: reason}, %{scope: scope}) do
    case Orders.cancel(scope, order, reason) do
      {:ok, _order} -> {:ok, "Order #{order.id} was cancelled."}
      {:error, message} -> {:error, message}
    end
  end
end
```

The model never sees another customer's order, and a request to cancel a
shipped order comes back to the model with a reason and a next step, without
the tool body ever running.

The checks can be combined. Cast the shape first with a changeset or Zoi, then
use the scope for the lookups:

```elixir
def parse_args(args, %{scope: scope}) do
  with {:ok, input} <- cast_input(args),
       {:ok, order} <- fetch_cancellable(scope, input.order_id) do
    {:ok, %{order: order, reason: input.reason}}
  end
end
```

## Pre-checking before asking for approval

Some tools shouldn't run until a person approves them. If the call is shown to
the user and they approve it, but it then fails because the order had already
shipped, the user was asked a question that had no right answer. The parser
already knows the call will fail, so it should be refused before anyone is
asked.

LangChain lets you run a call's parser without running the tool:

- `LangChain.Function.parse_arguments/3` takes the function, the arguments,
  and the context, and returns `{:ok, parsed_arguments}` or
  `{:error, message}`. It runs exactly the argument handling `execute/3` runs
  before the body: the parser when there is one, otherwise the required-key
  check.
- `LangChain.Chains.LLMChain.parse_tool_call_arguments/2` takes the chain and a
  `LangChain.Message.ToolCall`, finds the tool, and builds the context the
  same way execution does.

### With Sagents

The `Sagents.Middleware.HumanInTheLoop` middleware in
[Sagents](https://hex.pm/packages/sagents) (v0.17.1 and later) does this for
you. Before it interrupts for approval, it parses each gated call with the
tool's own parser and the context the call would run with. A call the parser
refuses is not shown to the user. It is answered with the parser's message,
and the model can correct it or tell the user why it can't be done. Only calls
that can actually run are put to the user.

The agent's `:scope` reaches the parser as `context.scope`:

```elixir
{:ok, agent} =
  Sagents.Agent.new(
    %{
      model: model,
      scope: current_scope,
      tools: [MyApp.Tools.CancelOrder.function()]
    },
    interrupt_on: %{"cancel_order" => true}
  )
```

With `MyApp.Tools.CancelOrder` from above, asking this agent to cancel a
shipped order never reaches the approval prompt. The model is told the order
can't be cancelled and is offered a return instead.

### In your own review handler

A plain `LLMChain` can ask for approval through an `:on_tool_call_review`
callback (see `LangChain.Chains.ChainCallbacks`). Call
`LangChain.Function.parse_arguments/3` before interrupting. When the parser refuses the
call, return `:ok` and let it go to execution, where the same parser refuses
it again and the model gets the message.

```elixir
review_handler = %{
  on_tool_call_review: fn _chain, call, func, review ->
    cond do
      # The user already answered for this call. Their answer stands.
      review.human_decision ->
        :ok

      call.name != "cancel_order" ->
        :ok

      true ->
        context = Map.put(review.custom_context, :tool_call_id, call.call_id)

        case Function.parse_arguments(func, call.arguments, context) do
          {:ok, %{order: order}} ->
            {:interrupt, "Cancel order #{order.id}?", %{order_id: order.id}}

          {:error, _message} ->
            # The call can't run. Don't ask about it; execution refuses it.
            :ok
        end
    end
  end
}

chain = LLMChain.add_callback(chain, review_handler)
```

The parsed result is useful here too: the handler builds its question from the
loaded order, not from the raw id the model sent.

## Parsers must be read-only

> #### Warning {: .warning}
>
> A parser may **read** data but must **never write** it, and must have no
> other side effects.

A parser can run more than once for a single call. It runs when the call is
pre-checked for approval, and again when the call executes, which may be much
later, after the user has answered. Each run must be safe to repeat and to
throw away.

- Do look up records, check permissions, check quotas, and compute what a
  call would change.
- Don't insert, update, or delete records, charge a card, send a message,
  enqueue a job, or increment a counter. Do those in the tool's `function`.

The run at execution time is the one that counts. Its result is the one passed
to `function`, so a parser checks the data as it is when the call actually
runs. An order that was cancellable when the user was asked but shipped while
they were deciding is refused at execution. Even so, the tool body should still
handle a failed write. Data can change between the parser's read and the
body's write, and the parser doesn't lock anything.

Because a parser can run more than once, keep it reasonably cheap. A database
lookup is fine. A slow call to an outside service on every parse may not be.

## Summary

- Use `parse_args/1` to check and convert the shape of the arguments.
- Use `parse_args/2` to check a call against the requesting user's data,
  through the scope in `custom_context`.
- Return `{:ok, parsed}` so the tool body gets the parsed data, and
  `{:error, message}` to refuse the call with a message the model can act on.
- A refused call never runs. With Sagents' HumanInTheLoop, or a review handler
  that calls `LangChain.Function.parse_arguments/3`, it is never shown to the user for
  approval either.
- Keep parsers read-only. They can run more than once per call.
