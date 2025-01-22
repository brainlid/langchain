defmodule LangChain.Utils.ChatTemplatesTest do
  use ExUnit.Case

  doctest LangChain.Utils.ChatTemplates

  alias LangChain.Utils.ChatTemplates
  alias LangChain.Message
  alias LangChain.LangChainError

  describe "prep_and_validate_messages/1" do
    test "returns 3 item tuple with expected parts" do
      system = Message.new_system!("system_message")
      first = Message.new_user!("user_1st")

      rest = [
        Message.new_assistant!("assistant_response"),
        Message.new_user!("user_2nd")
      ]

      {s, u, r} = ChatTemplates.prep_and_validate_messages([system, first | rest])
      assert s == system
      assert u == first
      assert r == rest
    end

    test "returns nil for system when absent" do
      first = Message.new_user!("user_1st")

      rest = [
        Message.new_assistant!("assistant_response"),
        Message.new_user!("user_2nd")
      ]

      {s, u, r} = ChatTemplates.prep_and_validate_messages([first | rest])
      assert s == nil
      assert u == first
      assert r == rest
    end

    test "raises exception when no messages given" do
      assert_raise LangChainError, "Messages are required.", fn ->
        ChatTemplates.prep_and_validate_messages([])
      end
    end

    test "raises exception when doesn't start with a user message" do
      assert_raise LangChainError,
                   "Messages must include a user prompt after a system message.",
                   fn ->
                     ChatTemplates.prep_and_validate_messages([
                       Message.new_system!("system_message")
                     ])
                   end

      assert_raise LangChainError,
                   "Messages must start with either a system or user message.",
                   fn ->
                     ChatTemplates.prep_and_validate_messages([
                       Message.new_assistant!("assistant_message")
                     ])
                   end
    end

    test "raises exception when multiple system messages given" do
      assert_raise LangChainError,
                   "Messages must include a user prompt after a system message.",
                   fn ->
                     ChatTemplates.prep_and_validate_messages([
                       Message.new_system!("system_message"),
                       Message.new_system!("system_message")
                     ])
                   end
    end

    test "raises no exception when alternating user/assistant not one for one" do
      system = Message.new_system!("system_message")

      first = Message.new_user!("user_1")

      rest = [
        Message.new_user!("user_2"),
        Message.new_assistant!("assistant_response"),
        Message.new_user!("user_3")
      ]

      {s, u, r} =
        ChatTemplates.prep_and_validate_messages([
          Message.new_system!("system_message"),
          Message.new_user!("user_1"),
          Message.new_user!("user_2"),
          Message.new_assistant!("assistant_response"),
          Message.new_user!("user_3")
        ])

      assert s == system
      assert u == first
      assert r == rest
    end

    # test "removes special tokens from the message content"
  end

  describe "apply_chat_template!/3 - :inst format" do
    test "includes provided system message" do
      messages = [
        Message.new_system!("system_message"),
        Message.new_user!("user_prompt")
      ]

      result = ChatTemplates.apply_chat_template!(messages, :inst)
      assert result == "<s>[INST] system_message user_prompt [/INST]"
    end

    test "no system message when not provided" do
      messages = [Message.new_user!("user_prompt")]

      result = ChatTemplates.apply_chat_template!(messages, :inst)
      assert result == "<s>[INST] user_prompt [/INST]"
    end

    test "formats 1st question correctly" do
      messages = [
        Message.new_system!("Only tell the truth."),
        Message.new_user!("How far away is the Sun?")
      ]

      result = ChatTemplates.apply_chat_template!(messages, :inst)
      assert result == "<s>[INST] Only tell the truth. How far away is the Sun? [/INST]"
    end

    test "formats answered question correctly" do
      messages = [
        Message.new_system!("Only tell the truth."),
        Message.new_user!("How far away is the Sun?"),
        Message.new_assistant!("149.6 million kilometers")
      ]

      result = ChatTemplates.apply_chat_template!(messages, :inst)

      assert result ==
               "<s>[INST] Only tell the truth. How far away is the Sun? [/INST]149.6 million kilometers</s> "
    end

    test "formats 2nd question correctly" do
      messages = [
        Message.new_system!("Only tell the truth."),
        Message.new_user!("How far away is the Sun?"),
        Message.new_assistant!("149.6 million kilometers"),
        Message.new_user!("How far is that in miles?")
      ]

      result = ChatTemplates.apply_chat_template!(messages, :inst)

      assert result ==
               "<s>[INST] Only tell the truth. How far away is the Sun? [/INST]149.6 million kilometers</s> [INST] How far is that in miles? [/INST]"
    end

    test "formatting matches example" do
      messages = [
        Message.new_user!("Hello, how are you?"),
        Message.new_assistant!("I'm doing great. How can I help you today?"),
        Message.new_user!("I'd like to show off how chat templating works!")
      ]

      result = ChatTemplates.apply_chat_template!(messages, :inst)

      assert result ==
               "<s>[INST] Hello, how are you? [/INST]I'm doing great. How can I help you today?</s> [INST] I'd like to show off how chat templating works! [/INST]"
    end
  end

  describe "apply_chat_template!/3 - :zephyr format" do
    test "includes provided system message" do
      messages = [
        Message.new_system!("system_message"),
        Message.new_user!("user_prompt")
      ]

      expected = "<|system|>\nsystem_message</s>\n<|user|>\nuser_prompt</s>\n<|assistant|>\n"

      result = ChatTemplates.apply_chat_template!(messages, :zephyr)
      assert result == expected
    end

    test "no system message when not provided" do
      messages = [Message.new_user!("user_prompt")]

      expected = "<|user|>\nuser_prompt</s>\n<|assistant|>\n"

      result = ChatTemplates.apply_chat_template!(messages, :zephyr)
      assert result == expected
    end

    test "does not add generation prompt when set to false" do
      messages = [
        Message.new_system!("system_message"),
        Message.new_user!("user_prompt")
      ]

      expected = "<|system|>\nsystem_message</s>\n<|user|>\nuser_prompt</s>\n"

      result = ChatTemplates.apply_chat_template!(messages, :zephyr, add_generation_prompt: false)
      assert result == expected
    end

    test "formats 1st question correctly" do
      messages = [
        Message.new_system!("Only tell the truth."),
        Message.new_user!("How far away is the Sun?")
      ]

      expected =
        "<|system|>\nOnly tell the truth.</s>\n<|user|>\nHow far away is the Sun?</s>\n<|assistant|>\n"

      result = ChatTemplates.apply_chat_template!(messages, :zephyr)
      assert result == expected
    end

    test "formats answered question correctly" do
      messages = [
        Message.new_system!("Only tell the truth."),
        Message.new_user!("How far away is the Sun?"),
        Message.new_assistant!("149.6 million kilometers")
      ]

      expected =
        "<|system|>\nOnly tell the truth.</s>\n<|user|>\nHow far away is the Sun?</s>\n<|assistant|>\n149.6 million kilometers</s>\n"

      result = ChatTemplates.apply_chat_template!(messages, :zephyr)
      assert result == expected
    end

    test "formats 2nd question correctly" do
      messages = [
        Message.new_system!("Only tell the truth."),
        Message.new_user!("How far away is the Sun?"),
        Message.new_assistant!("149.6 million kilometers"),
        Message.new_user!("How far is that in miles?")
      ]

      expected =
        "<|system|>\nOnly tell the truth.</s>\n<|user|>\nHow far away is the Sun?</s>\n<|assistant|>\n149.6 million kilometers</s>\n<|user|>\nHow far is that in miles?</s>\n<|assistant|>\n"

      result = ChatTemplates.apply_chat_template!(messages, :zephyr)
      assert result == expected
    end

    test "formatting matches example" do
      messages = [
        Message.new_system!(
          "You are a friendly chatbot who always responds in the style of a pirate"
        ),
        Message.new_user!("How many helicopters can a human eat in one sitting?"),
        Message.new_assistant!(
          "Matey, I'm afraid I must inform ye that humans cannot eat helicopters. Helicopters are not food, they are flying machines. Food is meant to be eaten, like a hearty plate o' grog, a savory bowl o' stew, or a delicious loaf o' bread. But helicopters, they be for transportin' and movin' around, not for eatin'. So, I'd say none, me hearties. None at all."
        )
      ]

      expected =
        "<|system|>\nYou are a friendly chatbot who always responds in the style of a pirate</s>\n<|user|>\nHow many helicopters can a human eat in one sitting?</s>\n<|assistant|>\nMatey, I'm afraid I must inform ye that humans cannot eat helicopters. Helicopters are not food, they are flying machines. Food is meant to be eaten, like a hearty plate o' grog, a savory bowl o' stew, or a delicious loaf o' bread. But helicopters, they be for transportin' and movin' around, not for eatin'. So, I'd say none, me hearties. None at all.</s>\n"

      result = ChatTemplates.apply_chat_template!(messages, :zephyr)
      assert result == expected
    end
  end

  describe "apply_chat_template!/3 - :im_start format" do
    test "includes provided system message" do
      messages = [
        Message.new_system!("system_message"),
        Message.new_user!("user_prompt")
      ]

      expected =
        "<|im_start|>system\nsystem_message<|im_end|>\n<|im_start|>user\nuser_prompt<|im_end|>\n<|im_start|>assistant\n"

      result = ChatTemplates.apply_chat_template!(messages, :im_start)
      assert result == expected
    end

    test "no system message when not provided" do
      messages = [Message.new_user!("user_prompt")]

      expected = "<|im_start|>user\nuser_prompt<|im_end|>\n<|im_start|>assistant\n"

      result = ChatTemplates.apply_chat_template!(messages, :im_start)
      assert result == expected
    end

    test "does not add generation prompt when set to false" do
      messages = [
        Message.new_system!("system_message"),
        Message.new_user!("user_prompt")
      ]

      expected =
        "<|im_start|>system\nsystem_message<|im_end|>\n<|im_start|>user\nuser_prompt<|im_end|>\n"

      result =
        ChatTemplates.apply_chat_template!(messages, :im_start, add_generation_prompt: false)

      assert result == expected
    end

    test "formats 1st question correctly" do
      messages = [
        Message.new_system!("Only tell the truth."),
        Message.new_user!("How far away is the Sun?")
      ]

      expected =
        "<|im_start|>system\nOnly tell the truth.<|im_end|>\n<|im_start|>user\nHow far away is the Sun?<|im_end|>\n<|im_start|>assistant\n"

      result = ChatTemplates.apply_chat_template!(messages, :im_start)
      assert result == expected
    end

    test "formats answered question correctly" do
      messages = [
        Message.new_system!("Only tell the truth."),
        Message.new_user!("How far away is the Sun?"),
        Message.new_assistant!("149.6 million kilometers")
      ]

      expected =
        "<|im_start|>system\nOnly tell the truth.<|im_end|>\n<|im_start|>user\nHow far away is the Sun?<|im_end|>\n<|im_start|>assistant\n149.6 million kilometers<|im_end|>\n"

      result = ChatTemplates.apply_chat_template!(messages, :im_start)
      assert result == expected
    end

    test "formats 2nd question correctly" do
      messages = [
        Message.new_system!("Only tell the truth."),
        Message.new_user!("How far away is the Sun?"),
        Message.new_assistant!("149.6 million kilometers"),
        Message.new_user!("How far is that in miles?")
      ]

      expected =
        "<|im_start|>system\nOnly tell the truth.<|im_end|>\n<|im_start|>user\nHow far away is the Sun?<|im_end|>\n<|im_start|>assistant\n149.6 million kilometers<|im_end|>\n<|im_start|>user\nHow far is that in miles?<|im_end|>\n<|im_start|>assistant\n"

      result = ChatTemplates.apply_chat_template!(messages, :im_start)
      assert result == expected
    end

    test "formatting matches example" do
      messages = [
        Message.new_user!("Hi there!"),
        Message.new_assistant!("Nice to meet you!"),
        Message.new_user!("Can I ask a question?")
      ]

      expected =
        "<|im_start|>user\nHi there!<|im_end|>\n<|im_start|>assistant\nNice to meet you!<|im_end|>\n<|im_start|>user\nCan I ask a question?<|im_end|>\n<|im_start|>assistant\n"

      result = ChatTemplates.apply_chat_template!(messages, :im_start)
      assert result == expected
    end
  end

  describe "apply_chat_template!/3 - :lama_2 format" do
    test "includes provided system message" do
      messages = [
        Message.new_system!("system_message"),
        Message.new_user!("user_prompt")
      ]

      expected = "<s>[INST] <<SYS>>\nsystem_message\n<</SYS>>\n\nuser_prompt [/INST] "

      result = ChatTemplates.apply_chat_template!(messages, :llama_2)
      assert result == expected
    end

    test "no system message when not provided" do
      messages = [Message.new_user!("user_prompt")]

      expected = "<s>[INST] user_prompt [/INST] "

      result = ChatTemplates.apply_chat_template!(messages, :llama_2)
      assert result == expected
    end

    test "formats answered question correctly" do
      messages = [
        Message.new_system!("system_message"),
        Message.new_user!("user_prompt"),
        Message.new_assistant!("assistant_response")
      ]

      expected =
        "<s>[INST] <<SYS>>\nsystem_message\n<</SYS>>\n\nuser_prompt [/INST] assistant_response </s>"

      result = ChatTemplates.apply_chat_template!(messages, :llama_2)
      assert result == expected
    end

    test "formats 2nd question correctly" do
      messages = [
        Message.new_system!("system_message"),
        Message.new_user!("user_prompt"),
        Message.new_assistant!("assistant_response"),
        Message.new_user!("user_2nd")
      ]

      expected =
        "<s>[INST] <<SYS>>\nsystem_message\n<</SYS>>\n\nuser_prompt [/INST] assistant_response </s><s>[INST] user_2nd [/INST] "

      result = ChatTemplates.apply_chat_template!(messages, :llama_2)
      assert result == expected
    end
  end

  describe "apply_chat_template!/3 - :llama_3 format" do
    test "includes provided system message" do
      messages = [
        Message.new_system!("system_message"),
        Message.new_user!("user_prompt")
      ]

      expected =
        "<|begin_of_text|>\n<|start_header_id|>system<|end_header_id|>\n\nsystem_message<|eot_id|>\n<|start_header_id|>user<|end_header_id|>\n\nuser_prompt<|eot_id|>\n<|start_header_id|>assistant<|end_header_id|>\n\n"

      result = ChatTemplates.apply_chat_template!(messages, :llama_3)
      assert result == expected
    end

    test "does not add generation prompt when set to false" do
      messages = [
        Message.new_system!("system_message"),
        Message.new_user!("user_prompt")
      ]

      expected =
        "<|begin_of_text|>\n<|start_header_id|>system<|end_header_id|>\n\nsystem_message<|eot_id|>\n<|start_header_id|>user<|end_header_id|>\n\nuser_prompt<|eot_id|>\n"

      result =
        ChatTemplates.apply_chat_template!(messages, :llama_3, add_generation_prompt: false)

      assert result == expected
    end

    test "no system message when not provided" do
      messages = [Message.new_user!("user_prompt")]

      expected =
        "<|begin_of_text|>\n<|start_header_id|>user<|end_header_id|>\n\nuser_prompt<|eot_id|>\n<|start_header_id|>assistant<|end_header_id|>\n\n"

      result = ChatTemplates.apply_chat_template!(messages, :llama_3)
      assert result == expected
    end

    test "formats answered question correctly" do
      messages = [
        Message.new_system!("system_message"),
        Message.new_user!("user_prompt"),
        Message.new_assistant!("assistant_response")
      ]

      expected =
        "<|begin_of_text|>\n<|start_header_id|>system<|end_header_id|>\n\nsystem_message<|eot_id|>\n<|start_header_id|>user<|end_header_id|>\n\nuser_prompt<|eot_id|>\n<|start_header_id|>assistant<|end_header_id|>\n\nassistant_response<|eot_id|>\n"

      result = ChatTemplates.apply_chat_template!(messages, :llama_3)
      assert result == expected
    end

    test "formats 2nd question correctly" do
      messages = [
        Message.new_system!("system_message"),
        Message.new_user!("user_prompt"),
        Message.new_assistant!("assistant_response"),
        Message.new_user!("user_2nd")
      ]

      expected =
        "<|begin_of_text|>\n<|start_header_id|>system<|end_header_id|>\n\nsystem_message<|eot_id|>\n<|start_header_id|>user<|end_header_id|>\n\nuser_prompt<|eot_id|>\n<|start_header_id|>assistant<|end_header_id|>\n\nassistant_response<|eot_id|>\n<|start_header_id|>user<|end_header_id|>\n\nuser_2nd<|eot_id|>\n<|start_header_id|>assistant<|end_header_id|>\n\n"

      result = ChatTemplates.apply_chat_template!(messages, :llama_3)
      assert result == expected
    end
  end

  describe "apply_chat_template!/3 - with template callback" do
    test "formats according to template callback" do
      messages = [
        Message.new_system!("system_message"),
        Message.new_user!("user_prompt"),
        Message.new_assistant!("assistant_response"),
        Message.new_user!("user_2nd")
      ]

      format =
        "<|start_of_template|><%= for message <- @messages do %><%= message.role %>\n<%= message.content %>\n\n<% end %><|end_of_template|>"

      template_callback = fn messages, _opts ->
        EEx.eval_string(format,
          assigns: [messages: messages]
        )
      end

      expected =
        "<|start_of_template|>system\nsystem_message\n\nuser\nuser_prompt\n\nassistant\nassistant_response\n\nuser\nuser_2nd\n\n<|end_of_template|>"

      result = ChatTemplates.apply_chat_template!(messages, template_callback)
      assert result == expected
    end
  end

  defp hello_world(_args, _context) do
    "Hello world!"
  end

  describe "apply_chat_template!/3 - :llama_3_1_json_tool_calling format" do
    test "includes provided system message" do
      messages = [
        Message.new_system!("system_message"),
        Message.new_user!("user_prompt")
      ]

      schema_def = %{
        type: "object",
        properties: %{
          info: %{
            type: "object",
            properties: %{
              name: %{type: "string"}
            },
            required: ["name"]
          }
        },
        required: ["info"]
      }

      {:ok, fun} =
        LangChain.Function.new(%{
          "name" => "say_hi",
          "description" => "Provide a friendly greeting.",
          "parameters_schema" => schema_def,
          "function" => &hello_world/2
        })

      tools = [fun]

      date = Calendar.strftime(DateTime.utc_now(), "%d %B %Y")

      expected =
        "<|begin_of_text|>\n<|start_header_id|>system<|end_header_id|>\n\nsystem_message\n\nCutting Knowledge Date: December 2023\nToday Date: #{date}\n\nWhen you receive a tool call response, use the output to format an answer to the orginal user question.\n\nYou are a helpful assistant with tool calling capabilities.<|eot_id|>\n<|start_header_id|>user<|end_header_id|>\n\nGiven the following functions, please respond with a JSON for a function call with its proper arguments that best answers the given prompt.\n\nRespond in the format {\"name\": function name, \"parameters\": dictionary of argument name and its value}. Do not use variables.\n[\n  {\n    \"function\": {\n      \"name\": \"say_hi\",\n      \"description\": \"Provide a friendly greeting.\",\n      \"parameters\": {\n        \"type\": \"object\",\n        \"required\": [\n          \"info\"\n        ],\n        \"properties\": {\n          \"info\": {\n            \"type\": \"object\",\n            \"required\": [\n              \"name\"\n            ],\n            \"properties\": {\n              \"name\": {\n                \"type\": \"string\"\n              }\n            }\n          }\n        }\n      }\n    },\n    \"type\": \"function\"\n  }\n]\n\nuser_prompt<|eot_id|>\n<|start_header_id|>assistant<|end_header_id|>\n\n"

      result =
        ChatTemplates.apply_chat_template_with_tools!(
          messages,
          :llama_3_1_json_tool_calling,
          tools
        )

      assert result == expected
    end

    test "does not add generation prompt when set to false" do
      messages = [
        Message.new_system!("system_message"),
        Message.new_user!("user_prompt")
      ]

      schema_def = %{
        type: "object",
        properties: %{
          info: %{
            type: "object",
            properties: %{
              name: %{type: "string"}
            },
            required: ["name"]
          }
        },
        required: ["info"]
      }

      {:ok, fun} =
        LangChain.Function.new(%{
          "name" => "say_hi",
          "description" => "Provide a friendly greeting.",
          "parameters_schema" => schema_def,
          "function" => &hello_world/2
        })

      tools = [fun]

      date = Calendar.strftime(DateTime.utc_now(), "%d %B %Y")

      expected =
        "<|begin_of_text|>\n<|start_header_id|>system<|end_header_id|>\n\nsystem_message\n\nCutting Knowledge Date: December 2023\nToday Date: #{date}\n\nWhen you receive a tool call response, use the output to format an answer to the orginal user question.\n\nYou are a helpful assistant with tool calling capabilities.<|eot_id|>\n<|start_header_id|>user<|end_header_id|>\n\nGiven the following functions, please respond with a JSON for a function call with its proper arguments that best answers the given prompt.\n\nRespond in the format {\"name\": function name, \"parameters\": dictionary of argument name and its value}. Do not use variables.\n[\n  {\n    \"function\": {\n      \"name\": \"say_hi\",\n      \"description\": \"Provide a friendly greeting.\",\n      \"parameters\": {\n        \"type\": \"object\",\n        \"required\": [\n          \"info\"\n        ],\n        \"properties\": {\n          \"info\": {\n            \"type\": \"object\",\n            \"required\": [\n              \"name\"\n            ],\n            \"properties\": {\n              \"name\": {\n                \"type\": \"string\"\n              }\n            }\n          }\n        }\n      }\n    },\n    \"type\": \"function\"\n  }\n]\n\nuser_prompt<|eot_id|>\n"

      result =
        ChatTemplates.apply_chat_template_with_tools!(
          messages,
          :llama_3_1_json_tool_calling,
          tools,
          add_generation_prompt: false
        )

      assert result == expected
    end

    test "no system message when not provided" do
      messages = [Message.new_user!("user_prompt")]

      schema_def = %{
        type: "object",
        properties: %{
          info: %{
            type: "object",
            properties: %{
              name: %{type: "string"}
            },
            required: ["name"]
          }
        },
        required: ["info"]
      }

      {:ok, fun} =
        LangChain.Function.new(%{
          "name" => "say_hi",
          "description" => "Provide a friendly greeting.",
          "parameters_schema" => schema_def,
          "function" => &hello_world/2
        })

      tools = [fun]

      date = Calendar.strftime(DateTime.utc_now(), "%d %B %Y")

      expected =
        "<|begin_of_text|>\n<|start_header_id|>system<|end_header_id|>\n\n\n\nCutting Knowledge Date: December 2023\nToday Date: #{date}\n\nWhen you receive a tool call response, use the output to format an answer to the orginal user question.\n\nYou are a helpful assistant with tool calling capabilities.<|eot_id|>\n<|start_header_id|>user<|end_header_id|>\n\nGiven the following functions, please respond with a JSON for a function call with its proper arguments that best answers the given prompt.\n\nRespond in the format {\"name\": function name, \"parameters\": dictionary of argument name and its value}. Do not use variables.\n[\n  {\n    \"function\": {\n      \"name\": \"say_hi\",\n      \"description\": \"Provide a friendly greeting.\",\n      \"parameters\": {\n        \"type\": \"object\",\n        \"required\": [\n          \"info\"\n        ],\n        \"properties\": {\n          \"info\": {\n            \"type\": \"object\",\n            \"required\": [\n              \"name\"\n            ],\n            \"properties\": {\n              \"name\": {\n                \"type\": \"string\"\n              }\n            }\n          }\n        }\n      }\n    },\n    \"type\": \"function\"\n  }\n]\n\nuser_prompt<|eot_id|>\n<|start_header_id|>assistant<|end_header_id|>\n\n"

      result =
        ChatTemplates.apply_chat_template_with_tools!(
          messages,
          :llama_3_1_json_tool_calling,
          tools
        )

      assert result == expected
    end

    test "formats answered question correctly" do
      messages = [
        Message.new_system!("system_message"),
        Message.new_user!("user_prompt"),
        Message.new_assistant!("assistant_response")
      ]

      schema_def = %{
        type: "object",
        properties: %{
          info: %{
            type: "object",
            properties: %{
              name: %{type: "string"}
            },
            required: ["name"]
          }
        },
        required: ["info"]
      }

      {:ok, fun} =
        LangChain.Function.new(%{
          "name" => "say_hi",
          "description" => "Provide a friendly greeting.",
          "parameters_schema" => schema_def,
          "function" => &hello_world/2
        })

      tools = [fun]

      date = Calendar.strftime(DateTime.utc_now(), "%d %B %Y")

      expected =
        "<|begin_of_text|>\n<|start_header_id|>system<|end_header_id|>\n\nsystem_message\n\nCutting Knowledge Date: December 2023\nToday Date: #{date}\n\nWhen you receive a tool call response, use the output to format an answer to the orginal user question.\n\nYou are a helpful assistant with tool calling capabilities.<|eot_id|>\n<|start_header_id|>user<|end_header_id|>\n\nGiven the following functions, please respond with a JSON for a function call with its proper arguments that best answers the given prompt.\n\nRespond in the format {\"name\": function name, \"parameters\": dictionary of argument name and its value}. Do not use variables.\n[\n  {\n    \"function\": {\n      \"name\": \"say_hi\",\n      \"description\": \"Provide a friendly greeting.\",\n      \"parameters\": {\n        \"type\": \"object\",\n        \"required\": [\n          \"info\"\n        ],\n        \"properties\": {\n          \"info\": {\n            \"type\": \"object\",\n            \"required\": [\n              \"name\"\n            ],\n            \"properties\": {\n              \"name\": {\n                \"type\": \"string\"\n              }\n            }\n          }\n        }\n      }\n    },\n    \"type\": \"function\"\n  }\n]\n\nuser_prompt<|eot_id|>\n<|start_header_id|>assistant<|end_header_id|>\n\nassistant_response<|eot_id|>\n"

      result =
        ChatTemplates.apply_chat_template_with_tools!(
          messages,
          :llama_3_1_json_tool_calling,
          tools
        )

      assert result == expected
    end

    test "formats 2nd question correctly" do
      messages = [
        Message.new_system!("system_message"),
        Message.new_user!("user_prompt"),
        Message.new_assistant!("assistant_response"),
        Message.new_user!("user_2nd")
      ]

      schema_def = %{
        type: "object",
        properties: %{
          info: %{
            type: "object",
            properties: %{
              name: %{type: "string"}
            },
            required: ["name"]
          }
        },
        required: ["info"]
      }

      {:ok, fun} =
        LangChain.Function.new(%{
          "name" => "say_hi",
          "description" => "Provide a friendly greeting.",
          "parameters_schema" => schema_def,
          "function" => &hello_world/2
        })

      tools = [fun]
      date = Calendar.strftime(DateTime.utc_now(), "%d %B %Y")

      expected =
        "<|begin_of_text|>\n<|start_header_id|>system<|end_header_id|>\n\nsystem_message\n\nCutting Knowledge Date: December 2023\nToday Date: #{date}\n\nWhen you receive a tool call response, use the output to format an answer to the orginal user question.\n\nYou are a helpful assistant with tool calling capabilities.<|eot_id|>\n<|start_header_id|>user<|end_header_id|>\n\nGiven the following functions, please respond with a JSON for a function call with its proper arguments that best answers the given prompt.\n\nRespond in the format {\"name\": function name, \"parameters\": dictionary of argument name and its value}. Do not use variables.\n[\n  {\n    \"function\": {\n      \"name\": \"say_hi\",\n      \"description\": \"Provide a friendly greeting.\",\n      \"parameters\": {\n        \"type\": \"object\",\n        \"required\": [\n          \"info\"\n        ],\n        \"properties\": {\n          \"info\": {\n            \"type\": \"object\",\n            \"required\": [\n              \"name\"\n            ],\n            \"properties\": {\n              \"name\": {\n                \"type\": \"string\"\n              }\n            }\n          }\n        }\n      }\n    },\n    \"type\": \"function\"\n  }\n]\n\nuser_prompt<|eot_id|>\n<|start_header_id|>assistant<|end_header_id|>\n\nassistant_response<|eot_id|>\n<|start_header_id|>user<|end_header_id|>\n\nuser_2nd<|eot_id|>\n<|start_header_id|>assistant<|end_header_id|>\n\n"

      result =
        ChatTemplates.apply_chat_template_with_tools!(
          messages,
          :llama_3_1_json_tool_calling,
          tools
        )

      assert result == expected
    end
  end

  describe "apply_chat_template!/3 - :llama_3_1_custom_tool_calling format" do
    test "tool use system message" do
      messages = [
        Message.new_system!("system_message"),
        Message.new_user!("user_prompt")
      ]

      schema_def = %{
        type: "object",
        properties: %{
          n: %{
            type: "integer",
            description: ""
          }
        },
        required: ["n"]
      }

      {:ok, fun} =
        LangChain.Function.new(%{
          "name" => "say_hi",
          "description" => "Provide a friendly greeting.",
          "parameters_schema" => schema_def,
          "function" => &hello_world/2
        })

      tools = [fun]

      date = Calendar.strftime(DateTime.utc_now(), "%d %B %Y")

      expected =
        "<|begin_of_text|>\n<|start_header_id|>system<|end_header_id|>\n\nEnvironment: ipython\nTools:\nCutting Knowledge Date: December 2023\nToday Date: #{date}\n\n# Tool Instructions\n- Always execute python code in messages that you share.\n- When looking for real time information use relevant functions if available\n\nYou have access to the following functions:\n\n\nUse the function 'say_hi' to: Provide a friendly greeting.\n{\n  \"n\": {\n    \"description\": \"\",\n    \"param_type\": \"int\",\n    \"required\": false\n  }\n}\n\n\n\n\nIf a you choose to call a function ONLY reply in the following format:\n<{start_tag}={function_name}>{parameters}{end_tag}\nwhere\n\nstart_tag => `<function`\nparameters => a JSON dict with the function argument name as key and function argument value as value.\nend_tag => `</function>`\n\nHere is an example,\n<function=example_function_name>{\"example_name\": \"example_value\"}</function>\n\nReminder:\n- Function calls MUST follow the specified format\n- Required parameters MUST be specified\n- Only call one function at a time\n- Put the entire function call reply on one line\n- Always add your sources when using search results to answer the user query\n\nYou are a helpful assistant.system_message<|eot_id|>\n<|start_header_id|>user<|end_header_id|>\n\nuser_prompt<|eot_id|>\n\n\n<|start_header_id|>assistant<|end_header_id|>\n\n"

      result =
        ChatTemplates.apply_chat_template_with_tools!(
          messages,
          :llama_3_1_custom_tool_calling,
          tools
        )

      assert result == expected
    end
  end

  describe "llama_3_1_custom_tool_calling_parameter_conversion/1" do
    test "converts single tool with basic parameters" do
      tools = [
        %LangChain.Function{
          name: "spotify_trending_songs",
          description: "Get top trending songs on Spotify",
          parameters_schema: %{
            type: "object",
            required: ["n"],
            properties: %{
              "n" => %{
                type: "integer"
              }
            }
          }
        }
      ]

      result =
        LangChain.Utils.ChatTemplates.llama_3_1_custom_tool_calling_parameter_conversion(tools)

      assert [converted_tool] = result
      assert converted_tool["name"] == "spotify_trending_songs"
      assert converted_tool["description"] == "Get top trending songs on Spotify"
      assert converted_tool["parameters"]["n"]["param_type"] == "int"
      assert converted_tool["parameters"]["n"]["required"] == true
    end

    test "handles multiple parameter types" do
      tool = %LangChain.Function{
        name: "test_tool",
        description: "Test tool",
        parameters_schema: %{
          "type" => "object",
          "required" => ["int_param"],
          "properties" => %{
            "int_param" => %{"type" => "integer", "description" => "Integer param"},
            "float_param" => %{"type" => "number", "description" => "Float param"},
            "bool_param" => %{"type" => "boolean", "description" => "Boolean param"},
            "string_param" => %{"type" => "string", "description" => "String param"}
          }
        }
      }

      [result] =
        LangChain.Utils.ChatTemplates.llama_3_1_custom_tool_calling_parameter_conversion([tool])

      params = result["parameters"]
      assert params["int_param"]["param_type"] == "int"
      assert params["float_param"]["param_type"] == "float"
      assert params["bool_param"]["param_type"] == "bool"
      assert params["string_param"]["param_type"] == "string"
    end

    test "handles empty schema" do
      tool = %LangChain.Function{
        name: "empty_tool",
        description: "Empty tool",
        parameters_schema: %{}
      }

      [result] =
        LangChain.Utils.ChatTemplates.llama_3_1_custom_tool_calling_parameter_conversion([tool])

      assert result["parameters"] == %{}
    end

    test "handles optional parameters" do
      tool = %LangChain.Function{
        name: "optional_tool",
        description: "Tool with optional params",
        parameters_schema: %{
          "type" => "object",
          "required" => ["required_param"],
          "properties" => %{
            "required_param" => %{"type" => "string", "description" => "Required"},
            "optional_param" => %{"type" => "string", "description" => "Optional"}
          }
        }
      }

      [result] =
        LangChain.Utils.ChatTemplates.llama_3_1_custom_tool_calling_parameter_conversion([tool])

      assert result["parameters"]["required_param"]["required"] == true
      assert result["parameters"]["optional_param"]["required"] == false
    end
  end

  describe "apply_chat_template!/3 - :llama_3_2_custom_tool_calling format" do
    test "llama3.2 tool use system message" do
      messages = [
        Message.new_system!("system_message"),
        Message.new_user!("user_prompt")
      ]

      schema_def = %{
        type: "object",
        properties: %{
          n: %{
            type: "integer",
            description: ""
          }
        },
        required: ["n"]
      }

      {:ok, fun} =
        LangChain.Function.new(%{
          "name" => "say_hi",
          "description" => "Provide a friendly greeting.",
          "parameters_schema" => schema_def,
          "function" => &hello_world/2
        })

      tools = [fun]

      expected =
        "<|start_header_id|>system<|end_header_id|>\nYou are an expert in composing functions. You are given a question and a set of possible functions.\nBased on the question, you will need to make one or more function/tool calls to achieve the purpose.\nIf none of the functions can be used, point it out. If the given question lacks the parameters required by the function,also point it out. You should only return the function call in tools call sections.\nIf you decide to invoke any of the function(s), you MUST put it in the format of [func_name1(params_name1=params_value1, params_name2=params_value2...), func_name2(params)]\nYou SHOULD NOT include any other text in the response.\nHere is a list of functions in JSON format that you can invoke.[\n  {\n    \"description\": \"Provide a friendly greeting.\",\n    \"name\": \"say_hi\",\n    \"parameters\": {\n      \"n\": {\n        \"description\": \"\",\n        \"param_type\": \"int\",\n        \"required\": false\n      }\n    }\n  }\n]system_message\n<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nuser_prompt<|eot_id|>\n<|start_header_id|>assistant<|end_header_id|>\n\n"

      result =
        ChatTemplates.apply_chat_template_with_tools!(
          messages,
          :llama_3_2_custom_tool_calling,
          tools
        )

      assert result == expected
    end

    test "llama3.2 tool response" do
      schema_def = %{
        type: "object",
        properties: %{
          n: %{
            type: "integer",
            description: ""
          }
        },
        required: ["n"]
      }

      {:ok, fun} =
        LangChain.Function.new(%{
          "name" => "say_hi",
          "description" => "Provide a friendly greeting.",
          "parameters_schema" => schema_def,
          "function" => &hello_world/2
        })

      tools = [fun]

      messages = [
        %LangChain.Message{
          content: "Where is the hairbrush located?",
          processed_content: nil,
          index: nil,
          status: :complete,
          role: :user,
          name: nil,
          tool_calls: [],
          tool_results: nil
        },
        %LangChain.Message{
          content: "[get_location(thing=\"hairbrush\")]",
          processed_content: nil,
          index: nil,
          status: :complete,
          role: :assistant,
          name: nil,
          tool_calls: [
            %LangChain.Message.ToolCall{
              status: :complete,
              type: :function,
              call_id: "test",
              name: "get_location",
              arguments: %{"thing" => "hairbrush"},
              index: nil
            }
          ],
          tool_results: nil
        },
        %LangChain.Message{
          content: nil,
          processed_content: nil,
          index: nil,
          status: :complete,
          role: :tool,
          name: nil,
          tool_calls: [],
          tool_results: [
            %LangChain.Message.ToolResult{
              type: :function,
              tool_call_id: "test",
              name: "get_location",
              content: "drawer",
              display_text: nil,
              is_error: false
            }
          ]
        }
      ]

      result =
        ChatTemplates.apply_chat_template_with_tools!(
          messages,
          :llama_3_2_custom_tool_calling,
          tools
        )

      assert result ==
               "<|start_header_id|>system<|end_header_id|>\nYou are an expert in composing functions. You are given a question and a set of possible functions.\nBased on the question, you will need to make one or more function/tool calls to achieve the purpose.\nIf none of the functions can be used, point it out. If the given question lacks the parameters required by the function,also point it out. You should only return the function call in tools call sections.\nIf you decide to invoke any of the function(s), you MUST put it in the format of [func_name1(params_name1=params_value1, params_name2=params_value2...), func_name2(params)]\nYou SHOULD NOT include any other text in the response.\nHere is a list of functions in JSON format that you can invoke.[\n  {\n    \"description\": \"Provide a friendly greeting.\",\n    \"name\": \"say_hi\",\n    \"parameters\": {\n      \"n\": {\n        \"description\": \"\",\n        \"param_type\": \"int\",\n        \"required\": false\n      }\n    }\n  }\n]\n<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nWhere is the hairbrush located?<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n[get_location(thing=\"hairbrush\")]<|eot_id|><|start_header_id|>ipython<|end_header_id|>\n\n[{\"output\":\"drawer\"}]<|eot_id|>\n<|start_header_id|>assistant<|end_header_id|>\n\n"
    end
  end
end
