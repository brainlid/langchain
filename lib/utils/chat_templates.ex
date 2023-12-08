defmodule LangChain.Utils.ChatTemplates do
  @moduledoc """
  Functions for converting messages into the various commonly used chat template
  formats.
  """
  import Integer, only: [is_even: 1, is_odd: 1]
  alias LangChain.LangChainError
  alias LangChain.Message

  @type chat_format :: :inst | :im_start | :llama_2 | :zephyr

  # Option:
  # - `add_generation_prompt`: boolean. Defaults to False.

  # Mistral guardrailing
  # https://docs.mistral.ai/usage/guardrailing

  # https://hexdocs.pm/bumblebee/Bumblebee.Tokenizer.html#summary
  # - get tokenizer settings for `bos`, `eos`, etc

  @doc """
  Validate that the message are in a supported format. Returns the system
  message (or uses a default one).

  Returns: `{System Message, First User Message, Other Messages}`

  If there is an issue, an exception is raised. Reasons for an exception:

  - Only 1 system message is allowed and, if included, it is the first message.
  - Non-system messages must begin with a user message
  - Alternates message roles between: user, assistant, user, assistant, etc.
  """
  @spec prep_and_validate_messages([Message.t()]) ::
          {Message.t(), Message.t(), [Message.t()]} | no_return()
  def prep_and_validate_messages(messages) do
    {system, first_user, rest} =
      case messages do
        [%Message{role: :user} = first_user | rest] ->
          {nil, first_user, rest}

        [%Message{role: :system} = system, %Message{role: :user} = first_user | rest] ->
          {system, first_user, rest}

        _other ->
          raise LangChainError, "Messages must start with either a system or user message."
      end

    # no additional system messages
    extra_systems_count = Enum.count(rest, &(&1.role == :system))

    if extra_systems_count > 0 do
      raise LangChainError, "Only a single system message is expected and it must be first"
    end

    # must alternate user, assistant, user, etc. Put the first user message back
    # on the list for checking it.
    [first_user | rest]
    |> Enum.with_index()
    |> Enum.each(fn
      {%Message{role: :user}, index} when is_even(index) ->
        :ok

      {%Message{role: :assistant}, index} when is_odd(index) ->
        :ok

      _other ->
        raise LangChainError,
              "Conversation roles must alternate user/assistant/user/assistant/..."
    end)

    # TODO: Check/replace/validate messages don't include tokens for the format.
    # - pass in format tokens list? Exclude tokens?
    # - need the model's tokenizer config passed in.

    # return 3 element tuple of critical message pieces
    {system, first_user, rest}
  end

  @doc """
  Transform a list of messages into a text prompt in the desired format for the
  LLM.

  ## Options

  - `:add_generation_prompt` - Boolean. Defaults to `true` when the last message
    is a user prompt. Depending on the format, when a user message is the last
    message, then the text prompt should begin the portion for the assistant to
    trigger the assistant's text generation.
  - `:tokenizer` -
  """
  @spec apply_chat_template!([Message.t()], chat_format, opts :: Keyword.t()) ::
          String.t() | no_return()
  def apply_chat_template!(messages, chat_format, opts \\ [])

  def apply_chat_template!(messages, :inst, _opts) do
    # https://huggingface.co/docs/transformers/main/en/chat_templating the
    # `:inst` format does not "add_generation_prompt"
    #
    # {{ bos_token }}
    # {% for message in messages %}
    #   {% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}
    #     {{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}
    #   {% endif %}
    #   {% if message['role'] == 'user' %}
    #     {{ '[INST] ' + message['content'] + ' [/INST]' }}
    #   {% elif message['role'] == 'assistant' %}
    #     {{ message['content'] + eos_token + ' ' }}
    #   {% else %}
    #     {{ raise_exception('Only user and assistant roles are supported!') }}
    #   {% endif %}
    # {% endfor %}
    #
    # https://docs.mistral.ai/usage/guardrailing - an example of embedding the
    # system prompt at the start. <s>[INST] System Prompt + Instruction [/INST]
    # Model answer</s>[INST] Follow-up instruction [/INST]
    {system, first_user, rest} = prep_and_validate_messages(messages)

    # intentionally as a single line for explicit control of newlines and spaces.
    text =
      "<s>[INST] <%= if @system do %><%= @system.content %> <% end %><%= @first_user.content %> [/INST]<%= for m <- @rest do %><%= if m.role == :user do %>[INST] <%= m.content %> [/INST]<% else %><%= m.content %></s> <% end %><% end %>"

    EEx.eval_string(text, assigns: [system: system, first_user: first_user, rest: rest])
  end

  # Does Zephyr formatted text
  def apply_chat_template!(messages, :zephyr, opts) do
    # https://huggingface.co/HuggingFaceH4/zephyr-7b-alpha/blob/main/tokenizer_config.json#L34
    # {% for message in messages %}\n
    #   {% if message['role'] == 'user' %}\n
    #     {{ '<|user|>\n' + message['content'] + eos_token }}\n
    #   {% elif message['role'] == 'system' %}\n
    #     {{ '<|system|>\n' + message['content'] + eos_token }}\n
    #   {% elif message['role'] == 'assistant' %}\n
    #     {{ '<|assistant|>\n'  + message['content'] + eos_token }}\n
    #   {% endif %}\n
    #   {% if loop.last and add_generation_prompt %}\n
    #     {{ '<|assistant|>' }}\n
    #   {% endif %}\n
    # {% endfor %}

    add_generation_prompt =
      Keyword.get(opts, :add_generation_prompt, default_add_generation_prompt_value(messages))

    {system, first_user, rest} = prep_and_validate_messages(messages)

    # intentionally as a single line for explicit control of newlines and spaces.
    text =
      "<%= for message <- @messages do %><%= if message.role == :user do %><|user|>\n<%= message.content %></s>\n<% end %><%= if message.role == :system do %><|system|>\n<%= message.content %></s>\n<% end %><%= if message.role == :assistant do %><|assistant|>\n<%= message.content %></s>\n<% end %><% end %><%= if @add_generation_prompt do %><|assistant|>\n<% end %>"

    EEx.eval_string(text,
      assigns: [
        messages: [system, first_user | rest] |> Enum.drop_while(&(&1 == nil)),
        add_generation_prompt: add_generation_prompt
      ]
    )
  end

  # Does ChatML formatted text
  def apply_chat_template!(messages, :im_start, opts) do
    # <|im_start|>user
    # Hi there!<|im_end|>
    # <|im_start|>assistant
    # Nice to meet you!<|im_end|>
    # <|im_start|>user
    # Can I ask a question?<|im_end|>
    # <|im_start|>assistant
    add_generation_prompt =
      Keyword.get(opts, :add_generation_prompt, default_add_generation_prompt_value(messages))

    {system, first_user, rest} = prep_and_validate_messages(messages)

    # intentionally as a single line for explicit control of newlines and spaces.
    text =
      "<%= for message <- @messages do %><|im_start|><%= message.role %>\n<%= message.content %><|im_end|>\n<% end %><%= if @add_generation_prompt do %><|im_start|>assistant\n<% end %>"

    EEx.eval_string(text,
      assigns: [
        messages: [system, first_user | rest] |> Enum.drop_while(&(&1 == nil)),
        add_generation_prompt: add_generation_prompt
      ]
    )

  end

  # Does LLaMa 2 formatted text
  def apply_chat_template!(_messages, :llama_2, _opts) do
    # https://huggingface.co/blog/llama2#how-to-prompt-llama-2
    raise LangChainError, "Not yet implemented!"
  end

  # return the desired true/false value. Only set to true when the last message
  # is a user prompt.
  defp default_add_generation_prompt_value(messages) do
    case List.last(messages) do
      %Message{role: :user} -> true
      _other -> false
    end
  end
end
