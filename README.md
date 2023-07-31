# Langchain

Elixir version of a Langchain styled framework.

**NOTE**: This is under active development and is subject to significant changes.


## Installation

If [available in Hex](https://hex.pm/docs/publish), the package can be installed
by adding `langchain` to your list of dependencies in `mix.exs`:

```elixir
def deps do
  [
    {:langchain, "~> 0.1.0"}
  ]
end
```

Documentation can be generated with [ExDoc](https://github.com/elixir-lang/ex_doc)
and published on [HexDocs](https://hexdocs.pm). Once published, the docs can
be found at <https://hexdocs.pm/langchain>.

## Configuration

Currently, the library is written to use the `Req` library for making API calls.

Rename the `.envrc_template` to `.envrc` and populate it with your private API values.

Using a tool like [Dotenv](https://github.com/motdotla/dotenv), it can load the API values into the ENV when using the library locally.

## Usage


### Exposing a custom Elixir function to ChatGPT

```elixir
alias Langchain.Function
alias Langchain.Message
alias Langchain.Chains.LLMChain
alias Langchain.ChatModels.ChatOpenAI

# map of data we want to be passed as `context` to the function when
# executed.
custom_context = %{
  "user_id" => 123,
  "hairbrush" => "drawer",
  "dog" => "backyard",
  "sandwich" => "kitchen"
}

# a custom Elixir function made available to the LLM
custom_fn =
  Function.new!(%{
    name: "custom",
    description: "Returns the location of the requested element or item.",
    parameters_schema: %{
      type: "object",
      properties: %{
        thing: %{
          type: "string",
          description: "The thing whose location is being requested."
        }
      },
      required: ["thing"]
    },
    function: fn %{"thing" => thing} = _arguments, context ->
      # our context is a pretend item/location location map
      context[thing]
    end
  })

# create and run the chain
{:ok, updated_chain, %Message{} = message} =
  LLMChain.new!(%{
    llm: ChatOpenAI.new!(),
    custom_context: custom_context,
    verbose: true
  })
  |> LLMChain.add_functions(custom_fn)
  |> LLMChain.add_message(Message.new_user!("Where is the hairbrush located?"))
  |> LLMChain.run(while_needs_response: true)

# print the LLM's answer
IO.put message.content
#=> "The hairbrush is located in the drawer."
```

## Testing

To run all the tests including the ones that perform live calls against the OpenAI API, use the following command:

```
mix test --include live_call
```

NOTE: This will use the configured API credentials which creates billable events.

Otherwise, running the following will only run local tests making no external API calls:

```
mix test
```

Executing a specific test, wether it is a `live_call` or not, will execute it creating a potentially billable event.