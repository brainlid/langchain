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

Currently, the library is coded to use the `Req` library for making API calls.

Rename the `.envrc_template` to `.envrc` and populate it with your private API values.

Using a tool like [Dotenv](https://github.com/motdotla/dotenv), it can load the API values into the ENV when using the library locally.