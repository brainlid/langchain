defmodule LangChain.Utils.ChatTemplates do
  @moduledoc """
  Functions for converting messages into the various commonly used chat template
  formats.

  ## Format examples

  There are currently no industry standards around model's chat formats. For any
  given model, it's documentation and config may need to be inspected for it's
  format, and it may be something not supported at this time.

  ### `:inst`

  ```
  <s>[INST] System message. User message. [/INST]
  Assistant response
  [INST] User message. [/INST]</s>
  Assistant response
  ```

  Note: The `:inst` format has no concept of a system message. It will be
  combined with the first user message

  ### `:im_start`

  ```
  <|im_start|>user
  User message.<|im_end|>
  <|im_start|>assistant
  Assistant response.<|im_end|>
  <|im_start|>user
  User message.<|im_end|>
  <|im_start|>assistant
  ```

  Note: The `:im_start` format has no concept of a system message. It will be
  combined with the first user message.

  ### `:llama_2`

  ```
  <s>[INST] <<SYS>>
  System message.
  <</SYS>>

  User message [/INST] Assistant response </s><s>[INST] User message. [/INST]
  ```

  Note: The `:llama_2` format supports specific system messages. It is a
  variation of the `:inst` format.

  ### `:llama_3`

  ```
  <|begin_of_text|>
  <|start_header_id|>system<|end_header_id|>

  System message.<|eot_id|>
  <|start_header_id|>user<|end_header_id|>

  User message.<|eot_id|>
  <|start_header_id|>assistant<|end_header_id|>

  Assistant message.<|eot_id|>
  ```

  ### `:zephyr`

  ```
  <|system|>
  System message.</s>
  <|user|>
  User message.</s>
  <|assistant|>
  Assistant message.
  <|user|>
  User message.</s>
  ```

  Note: The `:zephyr` format supports specific system messages.


  ### `:phi_4`

  The `:phi_4` template format is also supported.

  ## Template callback

  It's possible to pass a callback as a template.
  The function receives the list of messages as first argument and `opts` as second and must return a string.
  """
  alias LangChain.LangChainError
  alias LangChain.Message
  alias LangChain.Message.ContentPart

  @type template_callback :: ([Message.t()], Keyword.t() -> String.t())
  @type chat_format ::
          :inst | :im_start | :llama_2 | :llama_3 | :phi_4 | :zephyr | template_callback()

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
  - Non-system messages must begin with a user message or assistant message.

  Recent change:
  - Alternating messages between user / assistant / user / assistant are no longer enforced as not every model has issues.
  - It is up to the programmer to enforce this if this is something they need.
  """
  @spec prep_and_validate_messages([Message.t()]) ::
          {Message.t() | nil, Message.t(), [Message.t()]} | no_return()
  def prep_and_validate_messages(messages) do
    {system, first_user, rest} =
      case messages do
        [%Message{role: :user} = first_user | rest] ->
          {nil, first_user, rest}

        [%Message{role: :system} = system, %Message{role: :user} = first_user | rest] ->
          {system, first_user, rest}

        [%Message{role: :system} = _system | _rest] ->
          raise LangChainError, "Messages must include a user prompt after a system message."

        [] ->
          raise LangChainError, "Messages are required."

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
    |> Enum.each(fn
      %Message{role: :user} ->
        :ok

      %Message{role: :tool} ->
        :ok

      %Message{role: :assistant} ->
        :ok

      _other ->
        raise LangChainError,
              "Conversation roles must be either user or assistant."
    end)

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
      "<% import LangChain.Message.ContentPart, only: [parts_to_string: 1] %><s>[INST] <%= if @system do %><%= parts_to_string(@system.content) %> <% end %><%= parts_to_string(@first_user.content) %> [/INST]<%= for m <- @rest do %><%= if m.role == :user do %>[INST] <%= parts_to_string(m.content) %> [/INST]<% else %><%= parts_to_string(m.content) %></s> <% end %><% end %>"

    EEx.eval_string(text,
      assigns: [system: system, first_user: first_user, rest: rest]
    )
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
      "<% import LangChain.Message.ContentPart, only: [parts_to_string: 1] %><%= for message <- @messages do %><%= if message.role == :user do %><|user|>\n<%= parts_to_string(message.content) %></s>\n<% end %><%= if message.role == :system do %><|system|>\n<%= parts_to_string(message.content) %></s>\n<% end %><%= if message.role == :assistant do %><|assistant|>\n<%= parts_to_string(message.content) %></s>\n<% end %><% end %><%= if @add_generation_prompt do %><|assistant|>\n<% end %>"

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
      "<% import LangChain.Message.ContentPart, only: [parts_to_string: 1] %><%= for message <- @messages do %><|im_start|><%= message.role %>\n<%= parts_to_string(message.content) %><|im_end|>\n<% end %><%= if @add_generation_prompt do %><|im_start|>assistant\n<% end %>"

    EEx.eval_string(text,
      assigns: [
        messages: [system, first_user | rest] |> Enum.drop_while(&(&1 == nil)),
        add_generation_prompt: add_generation_prompt
      ]
    )
  end

  def apply_chat_template!(messages, :phi_4, _opts) do
    # translation form https://huggingface.co/microsoft/phi-4/blob/main/tokenizer_config.json#L774 to Elixir via Claude 3.5 Sonnet Copilot
    # {% for message in messages %}{% if (message['role'] == 'system') %}{{'<|im_start|>system<|im_sep|>' + message['content'] + '<|im_end|>'}}{% elif (message['role'] == 'user') %}{{'<|im_start|>user<|im_sep|>' + message['content'] + '<|im_end|><|im_start|>assistant<|im_sep|>'}}{% elif (message['role'] == 'assistant') %}{{message['content'] + '<|im_end|>'}}{% endif %}{% endfor %}
    {system, first_user, rest} = prep_and_validate_messages(messages)

    text = """
    <% import LangChain.Message.ContentPart, only: [parts_to_string: 1] %><%= if @system != nil do %><|im_start|>system<|im_sep|><%= parts_to_string(@system.content) %><|im_end|><% end %>\
    <%= if @first_user != nil do %><|im_start|>user<|im_sep|><%= parts_to_string(@first_user.content) %><|im_end|><|im_start|>assistant<|im_sep|><% end %>\
    <%= for m <- @rest do %>\
    <%= if m.role == :user do %><|im_start|>user<|im_sep|><%= parts_to_string(m.content) %><|im_end|><|im_start|>assistant<|im_sep|>\
    <% else %><%= parts_to_string(m.content) %><|im_end|><% end %>\
    <% end %>
    """

    EEx.eval_string(text,
      assigns: [
        system: system,
        first_user: first_user,
        rest: rest
      ]
    )
  end

  # Does LLaMa 2 formatted text
  def apply_chat_template!(messages, :llama_2, _opts) do
    # https://huggingface.co/blog/llama2#how-to-prompt-llama-2

    # <s>[INST] <<SYS>>
    # {{ system_prompt }}
    # <</SYS>>

    # {{ user_msg_1 }} [/INST] {{ model_answer_1 }} </s><s>[INST] {{ user_msg_2 }} [/INST]

    {system, first_user, rest} = prep_and_validate_messages(messages)

    system_text =
      if system do
        "<<SYS>>\n#{ContentPart.parts_to_string(system.content)}\n<</SYS>>\n\n"
      else
        ""
      end

    # intentionally as a single line for explicit control of newlines and spaces.
    text =
      "<% import LangChain.Message.ContentPart, only: [parts_to_string: 1] %><s>[INST] <%= @system_text %><%= parts_to_string(@first_user.content) %> [/INST] <%= for m <- @rest do %><%= if m.role == :user do %><s>[INST] <%= parts_to_string(m.content) %> [/INST] <% else %><%= parts_to_string(m.content) %> </s><% end %><% end %>"

    EEx.eval_string(text, assigns: [system_text: system_text, first_user: first_user, rest: rest])
  end

  # Does LLaMa 3 formatted text
  def apply_chat_template!(messages, :llama_3, opts) do
    # <|begin_of_text|>
    # <|start_header_id|>system<|end_header_id|>
    #
    # You are a helpful assistant.<|eot_id|>
    # <|start_header_id|>user<|end_header_id|>
    #
    # What do you know about elixir?<|eot_id|>
    # <|start_header_id|>assistant<|end_header_id|>

    add_generation_prompt =
      Keyword.get(opts, :add_generation_prompt, default_add_generation_prompt_value(messages))

    {system, first_user, rest} = prep_and_validate_messages(messages)

    # intentionally as a single line for explicit control of newlines and spaces.
    text =
      "<% import LangChain.Message.ContentPart, only: [parts_to_string: 1] %><|begin_of_text|>\n<%= for message <- @messages do %><|start_header_id|><%= message.role %><|end_header_id|>\n\n<%= parts_to_string(message.content) %><|eot_id|>\n<% end %><%= if @add_generation_prompt do %><|start_header_id|>assistant<|end_header_id|>\n\n<% end %>"

    EEx.eval_string(text,
      assigns: [
        messages: [system, first_user | rest] |> Enum.drop_while(&(&1 == nil)),
        add_generation_prompt: add_generation_prompt
      ]
    )
  end

  def apply_chat_template!(messages, template_callback, opts)
      when is_function(template_callback, 2),
      do: template_callback.(messages, opts)

  @doc """
  Transform a list of messages into a text prompt in the desired format for the
  LLM.
  And adds tool configuration.

  ## Options

  - `:add_generation_prompt` - Boolean. Defaults to `true` when the last message
    is a user prompt. Depending on the format, when a user message is the last
    message, then the text prompt should begin the portion for the assistant to
    trigger the assistant's text generation.

  """
  def apply_chat_template_with_tools!(messages, chat_format, tools \\ [], opts \\ [])
  # Does LLaMa 3.1 json tool calling formatted text
  def apply_chat_template_with_tools!(messages, :llama_3_1_json_tool_calling, tools, opts) do
    # <|begin_of_text|>
    # <|start_header_id|>system<|end_header_id|>
    #
    # You are a helpful assistant.<|eot_id|>
    # <|start_header_id|>user<|end_header_id|>
    #
    # What do you know about elixir?<|eot_id|>
    # <|start_header_id|>assistant<|end_header_id|>

    add_generation_prompt =
      Keyword.get(opts, :add_generation_prompt, default_add_generation_prompt_value(messages))

    {system, first_user, rest} = prep_and_validate_messages(messages)

    rest =
      rest
      |> Enum.map(fn message ->
        case message.role do
          :tool ->
            tool_result = Enum.at(message.tool_results, 0)
            content = Jason.encode!(%{output: tool_result.content})
            %{message | content: content, role: "ipython"}

          _ ->
            message
        end
      end)

    tools_json_schema_string =
      tools
      |> Enum.map(fn %LangChain.Function{
                       name: name,
                       description: description,
                       parameters_schema: schema
                     } ->
        %{
          type: "function",
          function: %{
            name: name,
            description: description,
            parameters: schema
          }
        }
      end)
      |> Jason.encode!()
      |> Jason.Formatter.pretty_print()

    # intentionally indentation and newlines!!! for explicit control of newlines and spaces.
    text =
      """
      <% import LangChain.Message.ContentPart, only: [parts_to_string: 1] %><|begin_of_text|>
      <|start_header_id|>system<|end_header_id|>

      <%= @system_content %>

      Cutting Knowledge Date: December 2023
      Today Date: <%= @date %>

      When you receive a tool call response, use the output to format an answer to the original user question.

      You are a helpful assistant with tool calling capabilities.<|eot_id|>
      <|start_header_id|>user<|end_header_id|>

      Given the following functions, please respond with a JSON for a function call with its proper arguments that best answers the given prompt.

      Respond in the format {"name": function name, "parameters": dictionary of argument name and its value}. Do not use variables.
      <%= @tools_json_schema_string %>

      <%= parts_to_string(@first_user.content) %><|eot_id|>
      <%= for message <- @rest do %><|start_header_id|><%= message.role %><|end_header_id|>

      <%= parts_to_string(message.content) %><|eot_id|>
      <% end %><%= if @add_generation_prompt do %><|start_header_id|>assistant<|end_header_id|>

      <% end %>
      """
      |> String.slice(0..-2//1)

    EEx.eval_string(text,
      assigns: [
        tools_json_schema_string: tools_json_schema_string,
        date: Calendar.strftime(DateTime.utc_now(), "%d %B %Y"),
        system_content:
          case system do
            nil -> ""
            system -> ContentPart.parts_to_string(system.content)
          end,
        first_user: first_user,
        rest: rest |> Enum.drop_while(&(&1 == nil)),
        add_generation_prompt: add_generation_prompt
      ]
    )
  end

  def apply_chat_template_with_tools!(messages, :llama_3_1_custom_tool_calling, tools, opts) do
    add_generation_prompt =
      Keyword.get(opts, :add_generation_prompt, default_add_generation_prompt_value(messages))

    {system, first_user, rest} = prep_and_validate_messages(messages)

    # Convert tool results
    rest =
      rest
      |> Enum.map(fn message ->
        case message.role do
          :tool ->
            tool_result = Enum.at(message.tool_results, 0)
            content = Jason.encode!(%{output: tool_result.content})
            %{message | content: content, role: "ipython"}

          _ ->
            message
        end
      end)

    # Format tools as simple JSON schema
    tools_json_schema_string =
      llama_3_1_custom_tool_calling_parameter_conversion(tools)

    text =
      """
      <% import LangChain.Message.ContentPart, only: [parts_to_string: 1] %><|begin_of_text|>
      <|start_header_id|>system<|end_header_id|>

      Environment: ipython
      Tools:
      Cutting Knowledge Date: December 2023
      Today Date: <%= @date %>

      # Tool Instructions
      - Always execute python code in messages that you share.
      - When looking for real time information use relevant functions if available

      You have access to the following functions:

      <%= for tool <- @tools_json_schema_string do %>
      Use the function '<%= tool["name"] %>' to: <%= tool["description"] %>
      <%= tool["parameters"] |> Jason.encode!() |> Jason.Formatter.pretty_print() %>


      <% end %>

      If a you choose to call a function ONLY reply in the following format:
      <{start_tag}={function_name}>{parameters}{end_tag}
      where

      start_tag => `<function`
      parameters => a JSON dict with the function argument name as key and function argument value as value.
      end_tag => `</function>`

      Here is an example,
      <function=example_function_name>{"example_name": "example_value"}</function>

      Reminder:
      - Function calls MUST follow the specified format
      - Required parameters MUST be specified
      - Only call one function at a time
      - Put the entire function call reply on one line
      - Always add your sources when using search results to answer the user query

      You are a helpful assistant.<%= @system_content %><|eot_id|>
      <|start_header_id|>user<|end_header_id|>

      <%= parts_to_string(@first_user.content) %><|eot_id|>
      <%= for message <- @rest do %>
      <|start_header_id|><%= message.role %><|end_header_id|>

      <%= parts_to_string(message.content) %><|eot_id|>
      <% end %>
      <%= if @add_generation_prompt do %>
      <|start_header_id|>assistant<|end_header_id|>

      <% end %>
      """
      |> String.slice(0..-2//1)

    EEx.eval_string(text,
      assigns: [
        tools: tools,
        tools_json_schema_string: tools_json_schema_string,
        date: Calendar.strftime(DateTime.utc_now(), "%d %B %Y"),
        first_user: first_user,
        system_content:
          case system do
            nil -> ""
            system -> ContentPart.parts_to_string(system.content)
          end,
        rest: rest |> Enum.drop_while(&(&1 == nil)),
        add_generation_prompt: add_generation_prompt
      ]
    )
  end

  def apply_chat_template_with_tools!(messages, :llama_3_2_custom_tool_calling, tools, opts) do
    add_generation_prompt =
      Keyword.get(opts, :add_generation_prompt, default_add_generation_prompt_value(messages))

    {system, first_user, rest} = prep_and_validate_messages(messages)

    # Convert tool results
    rest =
      rest
      |> Enum.map(fn message ->
        case message.role do
          :tool ->
            tool_result = Enum.at(message.tool_results, 0)
            content = Jason.encode!([%{output: tool_result.content}])
            %{message | content: [ContentPart.text!(content)], role: "ipython"}

          _ ->
            message
        end
      end)

    # Format tools as simple JSON schema
    tools_json_schema_string =
      llama_3_1_custom_tool_calling_parameter_conversion(tools)

    text =
      """
      <% import LangChain.Message.ContentPart, only: [parts_to_string: 1] %><|start_header_id|>system<|end_header_id|>
      You are an expert in composing functions. You are given a question and a set of possible functions.
      Based on the question, you will need to make one or more function/tool calls to achieve the purpose.
      If none of the functions can be used, point it out. If the given question lacks the parameters required by the function,also point it out. You should only return the function call in tools call sections.
      If you decide to invoke any of the function(s), you MUST put it in the format of [func_name1(params_name1=params_value1, params_name2=params_value2...), func_name2(params)]
      You SHOULD NOT include any other text in the response.
      Here is a list of functions in JSON format that you can invoke.<%= @tools_json_schema_string |> Jason.encode!() |> Jason.Formatter.pretty_print() %><%= @system_content %>
      <|eot_id|><|start_header_id|>user<|end_header_id|>

      <%= parts_to_string(@first_user.content) %><|eot_id|><%= for message <- @rest do %><|start_header_id|><%= message.role %><|end_header_id|>

      <%= parts_to_string(message.content) %><|eot_id|><% end %><%= if @add_generation_prompt do %>
      <|start_header_id|>assistant<|end_header_id|>

      <% end %>
      """
      |> String.slice(0..-2//1)

    EEx.eval_string(text,
      assigns: [
        tools: tools,
        tools_json_schema_string: tools_json_schema_string,
        date: Calendar.strftime(DateTime.utc_now(), "%d %B %Y"),
        first_user: first_user,
        system_content:
          case system do
            nil -> ""
            system -> ContentPart.parts_to_string(system.content)
          end,
        rest: rest |> Enum.drop_while(&(&1 == nil)),
        add_generation_prompt: add_generation_prompt
      ]
    )
  end

  def llama_3_1_custom_tool_calling_parameter_conversion(tools) do
    tools
    |> Enum.map(fn %LangChain.Function{
                     name: name,
                     description: description,
                     parameters_schema: schema
                   } ->
      props = schema[:properties] || schema["properties"] || []

      parameters =
        props
        |> Enum.map(fn {param_name, param_config} ->
          {
            param_name,
            %{
              "param_type" =>
                get_param_type(param_config[:type] || param_config["type"] || "string"),
              "description" => param_config[:description] || param_config["description"] || "",
              "required" => param_name in (schema[:required] || schema["required"] || [])
            }
          }
        end)
        |> Enum.into(%{})

      %{
        "name" => name,
        "description" => description,
        "parameters" => parameters
      }
    end)
  end

  defp get_param_type(type) do
    case type do
      "integer" -> "int"
      "number" -> "float"
      "boolean" -> "bool"
      _ -> "string"
    end
  end

  # return the desired true/false value. Only set to true when the last message
  # is a user prompt.
  defp default_add_generation_prompt_value(messages) do
    case List.last(messages) do
      %Message{role: :user} -> true
      %Message{role: :tool} -> true
      _other -> false
    end
  end
end
