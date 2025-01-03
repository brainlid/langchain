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
end
