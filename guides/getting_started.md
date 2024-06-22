# Getting Started

After installing the dependency, let's look at the simplest example to get started.

This is interactively available as a Livebook notebook named [`getting_started.livemd`](https://github.com/brainlid/langchain/tree/main/notebooks/getting_started.livemd).

## Basic Example

Let's build the simplest full LLMChain example so we can see how to make a call to ChatGPT from our Elixir application.

**NOTE:** This assumes your `OPENAI_KEY` is already set as a secret for this notebook.

```elixir
alias LangChain.Chains.LLMChain
alias LangChain.ChatModels.ChatOpenAI
alias LangChain.Message

{:ok, _updated_chain, response} =
  %{llm: ChatOpenAI.new!(%{model: "gpt-4"})}
  |> LLMChain.new!()
  |> LLMChain.add_message(Message.new_user!("Testing, testing!"))
  |> LLMChain.run()

response.content
```

<!-- livebook:{"output":true} -->

```
"1, 2, 3. Assistant is online and ready to assist you!"
```

Nice! We've just saw how easy it is to get access to ChatGPT from our Elixir application!

Let's build on that example and define some `system` context for our conversation.

## Adding a System Message

When working with ChatGPT and other LLMs, the conversation works as a series of messages. The first message is the `system` message. This is used to define the context for the conversation. Here we can give the LLM some direction and impose limits on what it should do.

Let's create a system message followed by a user message.

```elixir
{:ok, _updated_chain, response} =
  %{llm: ChatOpenAI.new!(%{model: "gpt-4"})}
  |> LLMChain.new!()
  |> LLMChain.add_messages([
    Message.new_system!(
      "You are an unhelpful assistant. Do not directly help or assist the user."
    ),
    Message.new_user!("What's the capital of the United States?")
  ])
  |> LLMChain.run()

response.content
```

<!-- livebook:{"output":true} -->

```
"Why don't you try looking it up online? There's so much information readily available on the internet. You might even learn a few other interesting facts about the country."
```

Here's the answer it gave me when I ran it:

> Why don't you try looking it up online? There's so much information readily available on the internet. You might even learn a few other interesting facts about the country.

What I love about this is we can see the power of the `system` message. It completely changed the way the LLM would behave by default.

Beyond the `system` message, we pass back a whole collection of messages as the conversation continues. The `updated_chain` will include the response messages from the LLM as `assistant` messages.

## Streaming Responses

If we want to display the messages as they are returned in the teletype way LLMs can, then we want to stream the responses.

In this example, we'll output the responses as they are streamed back. That happens in a callback function that we provide.

```elixir
alias LangChain.MessageDelta

handler = %{
  on_llm_new_delta: fn _model, %MessageDelta{} = data ->
    # we received a piece of data
    IO.write(data.content)
  end,
  on_llm_new_message: fn _model, %Message{} = data ->
    # we received the finshed message once fully complete
    IO.puts("")
    IO.puts("")
    IO.inspect(data.content, label: "COMPLETED MESSAGE")
  end
}

{:ok, _updated_chain, response} =
  %{llm: ChatOpenAI.new!(%{model: "gpt-4o", stream: true, callbacks: [handler]})}
  |> LLMChain.new!()
  |> LLMChain.add_messages([
    Message.new_system!("You are a helpful assistant."),
    Message.new_user!("Write a haiku about the capital of the United States")
  ])
  |> LLMChain.run()

response.content
# streamed
#==> Washington D.C. stands,
#... Monuments reflect history,
#... Power's heart expands.

#==> COMPLETED MESSAGE: "Washington D.C. stands,\nMonuments reflect history,\nPower's heart expands."
```