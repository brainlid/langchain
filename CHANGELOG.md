# Changelog

## v0.3.0-rc.0 (2024-06-05)

**Added:**

* `LangChain.ChatModels.ChatGoogleAI` which differed too significantly from `LangChain.ChatModels.ChatGoogleAI`.  What's up with that? I'm looking at you Google! 👀
  * Thanks for the [contribution](https://github.com/brainlid/langchain/pull/124) Raul Chedrese!
* New callback mechanism was introduced to ChatModels and LLMChain. It was inspired by the approach used in the TS/JS LangChain library.
* Ability to provide plug-like middleware functions for pre-processing an assistant response message. Most helpful when coupled with a new run mode called `:until_success`. The first built-in one is `LangChain.MessageProcessors.JsonProcessor`.
* LLMChain has an internally managed `current_failure_count` and a publicly managed `max_retry_count`.
* New run mode `:until_success` uses failure and retry counts to repeatedly run the chain when the LLMs responses fail a MessageProcessor.
* `LangChain.MessageProcessors.JsonProcessor` is capable of extracting JSON contents and converting it to an Elixir map using `Jason`. Parsing errors are returned to the LLM for it to try again.
* The attribute `processed_content` was added to a `LangChain.Message`. When a MessageProcessor is run on a received assistant message, the results of the processing are accumulated there. The original `content` remains unchanged for when it is sent back to the LLM and used when fixing or correcting it's generated content.
* Callback support for LLM ratelimit information returned in API response headers. These are currently implemented for Anthropic and OpenAI.
* Callback support for LLM token usage information returned when available.

**Changed:**

* `LLMChain.run/2` error result now includes the failed chain up to the point of failure. This is helpful for debugging.
* `ChatOpenAI` and `ChatAnthropic` both support the new callbacks.
* Many smaller changes and contributions were made. This includes updates to the README for clarity,
* `LangChain.Utils.fire_callback/3` was refactored into `LangChain.Utils.fire_streamed_callback/2` where it is only used for processing deltas and uses the new callback mechanism.
* Notebooks were moved to the separate demo project

### Migrations Steps

The `LLMChain.run/2` function changed. Migrating should be easy.

**From:**

```elixir
chain
|> LLMChain.run(while_needs_response: true)
```

**Is changed to:**

```elixir
chain
|> LLMChain.run(mode: :while_needs_response)
```

This change enabled adding the new mode `:until_success`, which is mutually exclusive with `:while_needs_response`.

Additionally, the error return value was changed to include the chain itself.

**From:**

```elixir
{:error, reason} = LLMChain.run(chain)
```

**Is changed to:**

```elixir
{:error, _updated_chain, reason} = LLMChain.run(chain)
```

You can disregard the updated chain if you don't need it.

Callback events work differently now. Previously, a single `callback_fn` was executed and the developer needed to pattern match on a `%Message{}` or `%MessageDelta{}`. Callbacks work differently now.

When creating an LLM chat model, we can optionally pass in a map of callbacks where the event name is linked to the function to execute.

**From:**

```elixir
live_view_pid = self()

callback_fn = fn
  %MessageDelta{} = delta ->
    send(live_view_pid, {:received_delta, delta})

  %Message{} = message ->
    send(live_view_pid, {:received_message, message})
end

{:ok, _result_chain, last_message} =
  LLMChain.new!(%{llm: %ChatAnthropic{stream: false}})
  |> LLMChain.add_message(Message.new_user!("Say, 'Hi!'!"))
  |> LLMChain.run(callback_fn: callback_fn)
```

The equivalent code would look like this:

**Is changed to:**

```elixir
live_view_pid = self()

handler = %{
  on_llm_new_delta: fn _model, delta ->
    send(live_view_pid, {:received_delta, delta})
  end,
  on_llm_new_message: fn _model, message ->
    send(live_view_pid, {:received_message, message})
  end
}

{:ok, _result_chain, last_message} =
  LLMChain.new!(%{llm: %ChatAnthropic{stream: false, callbacks: [handler]}})
  |> LLMChain.add_message(Message.new_user!("Say, 'Hi!'!"))
  |> LLMChain.run()
```

The `Message` and `MessageDelta` callbacks are now set on the model. The callbacks are more granular and new callbacks are supported on the `LLMChain` as well. This more flexible configuration allows for more callbacks to be added as we move forward.

Also of note, is that the callbacks are set as a list of handler maps. This means we can assign multiple sets of callbacks for different purposes and they all get executed.

## v0.2.0 (2024-04-30)

For LLMs that support it (verified with ChatGPT and Anthropic), a user message can now contain multiple `ContentPart`s, making it "multi-modal". This means images and text can be combined into a single message allowing for interactions about the images to now be possible.

**Added:**

* `LangChain.Message.ContentPart` - used for User messages and multi-modal support. Google's AI assistant can return multiple parts as well.
* `LangChain.Message.ToolCall` - an assistant can request multiple tool calls in the same message.
* `LangChain.Message.ToolResult` - the system's answer to a `ToolCall`. It adds an is_error boolean flag. This an be helpful in the UI, but Anthropic specifically wants it.
* Add llama-3 chat template by @bowyern in https://github.com/brainlid/langchain/pull/102

**Changed:**

* The roles of `:function` and `:function_call` are removed. The equivalent of a `function_call` is expressed by an `:assistant` role making one or more `ToolCall` requests. The `:function` was the system's answer to a function call. This is now in the `:tool` role.
* Role `:tool` was added. A tool message contains one or more `ToolResult` messages.

## v0.1.10 (2024-03-07)

**Changes**

- Fix invalid default url for google ai by @pkrawat1 in https://github.com/brainlid/langchain/pull/82

## v0.1.9 (2024-02-29) - The Leap Release!

This adds support for Bumblebee as a Chat model, making it easy to have conversations with Llama 2, Mistral, and Zephyr LLMs.

See the documentation in `LangChain.ChatModels.ChatBumblebee` for getting started.

NOTE: That at this time, none of the models support the `function` ability, so that is not supported yet.

This release includes an experimental change for better support of streamed responses that are broken up over multiple messages from services like ChatGPT and others.

Other library dependencies requirements were relaxed, making it easier to support different versions of libraries like `req` and `nx`.

* Add mistral chat by @michalwarda in https://github.com/brainlid/langchain/pull/76
* handle receiving JSON data broken up over multiple messages by @brainlid in https://github.com/brainlid/langchain/pull/80
* Add initial support for Zephyr 7b Beta by @brainlid in https://github.com/brainlid/langchain/pull/41

## v0.1.8 (2024-02-16)

**Breaking change**: `RoutingChain`'s required values changed. Previously, `default_chain` was assigned an `%LLMChain{}` to return when no more specific routes matched.

This was changed to be `default_route`. It now expects a `%PromptRoute{}` to be provided.

Here's how to make the change:

      selected_route =
        RoutingChain.new(%{
          llm: ChatOpenAI.new(%{model: "gpt-3.5-turbo", stream: false}),
          input_text: user_input_text,
          routes: routes,
          default_route: PromptRoute.new!(%{name: "DEFAULT", chain: fallback_chain})
        })
        |> RoutingChain.evaluate()

The `default_chain` was changed to `default_route` and now expects a `PromptRoute` to be provided. The above example includes a sample default route that includes an optional `fallback_chain`.

Previously, the returned value from `RoutingChain.evaluate/1` was a `selected_chain`; it now returns the `selected_route`.

**Why was this changed?**

This was changed to make it easier to use a `PromptChain` when there isn't an associated `%LLMChain{}` for it. The application must just need the answer of which route was selected.

This includes the change of not requiring a `%PromptChain{}`'s `description` or `chain` field.

**Other Changes**
* Add support for Ollama open source models by @medoror in https://github.com/brainlid/langchain/pull/70
* Add clause to match call_response spec by @medoror in https://github.com/brainlid/langchain/pull/72
* Add max_tokens option for OpenAI calls by @matthusby in https://github.com/brainlid/langchain/pull/73

## v0.1.7 (2024-01-18)

- Improvements for more intelligent agents - https://github.com/brainlid/langchain/pull/61
  - adds `LangChain.Chains.RoutingChain` - first-pass LLM chain to select the best route to take given the user's initial prompt
  - adds `LangChain.Chains.TextToTitleChain` - turn the user's prompt into a title for the conversation
- Removed the need for a function to send a message to the process for how to display the function being executed
- Updated dependencies
- Add support for Google AI / Gemini Pro model by @jadengis in https://github.com/brainlid/langchain/pull/59
- Built-in automatic retries when underlying Mint connection is closed in https://github.com/brainlid/langchain/pull/68

## v0.1.6 (2023-12-12)

- Fix for correct usage of new Req retry setting. PR #57

## v0.1.5 (2023-12-11)

- Upgraded Req to v0.4.8. It contains a needed retry fix for certain situations.
- Fix OpenAI returns "Unrecognized request argument supplied: api_key" PR #54

## v0.1.4 (2023-12-11)

- Merged PR #45 - https://github.com/brainlid/langchain/pull/45
  - Added `LangChain.Utils.ChainResult` for helper functions when working with LLMChain result values.
- Merged PR #46 - https://github.com/brainlid/langchain/pull/46
  - Add possibility to use api_key per chat invocation.
- Merged PR #51 - https://github.com/brainlid/langchain/pull/51
  - Update req 0.4.7
  - Hopefully resolves issue where Finch connections would be closed and a now does a built-in retry.
- Merged PR #52 - https://github.com/brainlid/langchain/pull/52
  - Allow overriding OpenAI compatible API endpoint. Caller can pass an alternate `endpoint`.

## v0.1.3 (2023-12-01)

- Merged PR #43 - https://github.com/brainlid/langchain/pull/43
  - Add Finch retry strategy to OpenAI Chat API requests
- Merged PR #39 - https://github.com/brainlid/langchain/pull/39
  - Changed ENV key from `OPENAI_KEY` to `OPENAI_API_KEY` to be consistent with the OpenAI docs.
- Merged PR #36 - https://github.com/brainlid/langchain/pull/36
  - Support specifying the `seed` with OpenAI calls. Used in testing for more deterministic behavior.
- Merged PR #34 - https://github.com/brainlid/langchain/pull/34
  - Enable sending the `json_response` flag with OpenAI model requests.
- Created `LangChain.FunctionParam` to express JSONSchema-friendly data structures. Supports basic types, arrays, enums, objects, arrays of objects and nested objects.
  - Still allows for full control over JSONSchema by providing an override `parameters_schema` object to full self-describe it.

## v0.1.2 (2023-10-26)

- refactor(chat_open_ai): Harden `do_process_response` by @Cardosaum in https://github.com/brainlid/langchain/pull/21
  - Improve JSON error handling result from ChatGPT
- Update req to 0.4.4 by @medoror in https://github.com/brainlid/langchain/pull/25
  - Updated to Req 0.4.4


## v0.1.1 (2023-10-10)

Minor update release.

- added "update_custom_context" to LLMChain
- added support for setting the OpenAI-Organization header in requests
- fixed data extraction chain and improved the prompt
- make chatgpt response tests more robust


## v0.1.0 (2023-09-18)

Initial release when published to [hex.pm](https://hex.pm).
