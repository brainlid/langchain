defmodule Langchain.PromptTemplateTest do
  use ExUnit.Case
  doctest Langchain.PromptTemplate
  alias Langchain.PromptTemplate
  alias Langchain.LangchainError
  alias Langchain.Message

  describe "new/1" do
    test "create with text and no inputs" do
      {:ok, %PromptTemplate{} = p} = PromptTemplate.new(%{text: "text"})
      assert p.text == "text"
    end

    test "create with text and inputs" do
      {:ok, %PromptTemplate{} = p} =
        PromptTemplate.new(%{text: "Tell me about <%= @thing %>.", inputs: %{thing: "cats"}})

      assert p.text == "Tell me about <%= @thing %>."
      assert p.inputs == %{thing: "cats"}
    end

    test "return error when invalid" do
      {:error, changeset} = PromptTemplate.new(%{text: ""})
      assert {"can't be blank", _} = changeset.errors[:text]
    end
  end

  describe "new!/1" do
    test "return the prompt when valid" do
      %PromptTemplate{} = p = PromptTemplate.new!(%{text: "text"})
      assert p.text == "text"
    end

    test "raise exception with text reason when invalid" do
      assert_raise LangchainError, "text: can't be blank", fn ->
        PromptTemplate.new!(%{text: ""})
      end
    end
  end

  describe "from_template/1" do
    test "creates a PromptTemplate with the text set" do
      template = "What is a good name for a company that makes <%= @product %>?"
      assert {:ok, %PromptTemplate{} = p} = PromptTemplate.from_template(template)
      assert p.text == template

      assert result = PromptTemplate.format(p, %{product: "colorful socks"})
      assert result == "What is a good name for a company that makes colorful socks?"
    end

    test "returns an error when invalid" do
      assert {:error, changeset} = PromptTemplate.from_template("")
      assert {"can't be blank", _} = changeset.errors[:text]
    end
  end

  describe "from_template!/1" do
    test "creates a PromptTemplate with the text set" do
      template = "What is a good name for a company that makes <%= @product %>?"
      assert %PromptTemplate{} = p = PromptTemplate.from_template!(template)
      assert p.text == template
    end

    test "raises an exception when invalid" do
      assert_raise LangchainError, "text: can't be blank", fn ->
        PromptTemplate.from_template!("")
      end
    end
  end

  describe "format/2" do
    test "formats a defined prompt using no inputs" do
      {:ok, p} = PromptTemplate.new(%{text: "Nothing to replace."})
      result = PromptTemplate.format(p)
      assert result == "Nothing to replace."
    end

    test "formats a defined prompt using initial inputs" do
      {:ok, p} =
        PromptTemplate.new(%{text: "Replaced <%= @this %>.", inputs: %{this: "with that"}})

      result = PromptTemplate.format(p)
      assert result == "Replaced with that."
    end

    test "formats a defined prompt using initial merged with format inputs" do
      {:ok, p} =
        PromptTemplate.new(%{
          text: "Replaced <%= @this %> and later <%= @that %>.",
          inputs: %{this: "with that"}
        })

      result = PromptTemplate.format(p, %{that: "cows with horses"})
      assert result == "Replaced with that and later cows with horses."
    end
  end

  describe "format_text/2" do
    test "replaces template key with input value" do
      result =
        PromptTemplate.format_text(
          "What is a good name for a company that makes <%= @product %>?",
          %{product: "colorful socks"}
        )

      assert result == "What is a good name for a company that makes colorful socks?"
    end

    test "works when nothing to replace" do
      result = PromptTemplate.format_text("simple", %{})
      assert result == "simple"
    end

    test "returns substitutions removed when replacement missing" do
      result = PromptTemplate.format_text("This is <%= @missing %>", %{other: "something"})
      assert result == "This is "
    end
  end

  # TODO: better multiline string characters to use?
  describe "format_composed/3" do
    # https://js.langchain.com/docs/modules/prompts/prompt_templates/prompt_composition
    test "support composing prompts with input replacement at two levels" do
      full_prompt = PromptTemplate.from_template!(~s(<%= @introduction %>

<%= @example %>

<%= @start %>))

      introduction_prompt = PromptTemplate.from_template!("You are impersonating <%= @person %>.")

      example_prompt = PromptTemplate.from_template!(~s(Here's an example of an interaction:
Q: <%= @example_q %>
A: <%= @example_a %>))

      start_prompt = PromptTemplate.from_template!(~s(Now, do this for real!
Q: <%= @input %>
A:))

      formatted_prompt =
        PromptTemplate.format_composed(
          full_prompt,
          %{
            introduction: introduction_prompt,
            example: example_prompt,
            start: start_prompt
          },
          %{
            person: "Elon Musk",
            example_q: "What's your favorite car?",
            example_a: "Telsa",
            input: "What's your favorite social media site?"
          }
        )

      assert formatted_prompt == ~s(You are impersonating Elon Musk.

Here's an example of an interaction:
Q: What's your favorite car?
A: Telsa

Now, do this for real!
Q: What's your favorite social media site?
A:)
    end

    test "support inputs using EEx syntax" do
      full_prompt = PromptTemplate.from_template!(~s(<%= @setup %>

<%= @code_sample %>))

      setup_prompt =
        PromptTemplate.from_template!(
          "You are a helpful explainer of code for the <%= @language_name %> programming language and the <%= @framework_name %> framework."
        )

      code_sample_prompt = PromptTemplate.from_template!(~s(Explain the following code:

```
<%= @sample_code %>
```))

      formatted_prompt =
        PromptTemplate.format_composed(
          full_prompt,
          %{
            setup: setup_prompt,
            code_sample: code_sample_prompt
          },
          %{
            language_name: "Elixir",
            framework_name: "Phoenix",
            sample_code: ~S[defmodule MyAppWeb.HeaderComponent do
  use Phoenix.Component

  attr :rest, :global
  slot :inner_block, required: true

  def header(assigns) do
    ~H"""
    <h1 class="text-medium text-2xl" {@rest}>
      <%= render_slot(@inner_block) %>
    </h1>
    """
  end
end]
          }
        )

      assert formatted_prompt ==
               ~S[You are a helpful explainer of code for the Elixir programming language and the Phoenix framework.

Explain the following code:

```
defmodule MyAppWeb.HeaderComponent do
  use Phoenix.Component

  attr :rest, :global
  slot :inner_block, required: true

  def header(assigns) do
    ~H"""
    <h1 class="text-medium text-2xl" {@rest}>
      <%= render_slot(@inner_block) %>
    </h1>
    """
  end
end
```]
    end

    test "support plain strings in the composition" do
      full_prompt = PromptTemplate.from_template!(~s(<%= @introduction %>

<%= @ask %>))

      introduction_prompt = "You are a calculator that always gives an incorrect answer."

      ask_prompt =
        PromptTemplate.from_template!(~s(What is the answer to <%= @math_expression %>?))

      formatted_prompt =
        PromptTemplate.format_composed(
          full_prompt,
          %{
            introduction: introduction_prompt,
            ask: ask_prompt
          },
          %{
            math_expression: "1 + 1"
          }
        )

      assert formatted_prompt == ~s(You are a calculator that always gives an incorrect answer.

What is the answer to 1 + 1?)
    end
  end

  describe "to_message/2" do
    test "transforms a PromptTemplate to a Message" do
      {:ok, prompt} =
        PromptTemplate.new(%{
          role: :system,
          text:
            "You are a helpful assistant that translates <%= @input_language %> to <%= @output_language %>."
        })

      {:ok, expected} =
        Message.new_system("You are a helpful assistant that translates English to Norwegian.")

      {:ok, %Message{} = result} =
        PromptTemplate.to_message(prompt, %{
          input_language: "English",
          output_language: "Norwegian"
        })

      assert result == expected
    end
  end

  describe "to_messages!/2" do
    setup do
      expected = [
        Message.new_system!("You are an unhelpful assistant and actively avoid helping."),
        Message.new_user!(
          "I'm planning a trip to Peru, what are some sights I should consider visiting?"
        ),
        Message.new_assistant!("Eh. Does it really matter?"),
        Message.new_user!("Yes, it matters to me!")
      ]

      %{expected: expected}
    end

    test "transforms a list of PromptMessages", %{expected: expected} do
      prompts = [
        PromptTemplate.new!(%{
          role: :system,
          text: "You are an unhelpful assistant and actively avoid helping."
        }),
        PromptTemplate.new!(%{
          role: :user,
          text:
            "I'm planning a trip to <%= @destination %>, what are some sights I should consider visiting?"
        }),
        PromptTemplate.new!(%{role: :assistant, text: "Eh. Does it really matter?"}),
        PromptTemplate.new!(%{role: :user, text: "Yes, it matters to me!"})
      ]

      result = PromptTemplate.to_messages!(prompts, %{destination: "Peru"})
      assert result == expected
    end

    test "returns an existing Message in the list as-is", %{expected: expected} do
      prompts = [
        Message.new_system!("You are an unhelpful assistant and actively avoid helping."),
        PromptTemplate.new!(%{
          role: :user,
          text:
            "I'm planning a trip to <%= @destination %>, what are some sights I should consider visiting?"
        }),
        Message.new_assistant!("Eh. Does it really matter?"),
        Message.new_user!("Yes, it matters to me!")
      ]

      result = PromptTemplate.to_messages!(prompts, %{destination: "Peru"})
      assert result == expected
    end

    test "treats a prompt text string as a user message", %{expected: expected} do
      destination = "Peru"

      prompts = [
        Message.new_system!("You are an unhelpful assistant and actively avoid helping."),
        "I'm planning a trip to #{destination}, what are some sights I should consider visiting?",
        Message.new_assistant!("Eh. Does it really matter?"),
        "Yes, it matters to me!"
      ]

      result = PromptTemplate.to_messages!(prompts)
      assert result == expected
    end
  end
end
