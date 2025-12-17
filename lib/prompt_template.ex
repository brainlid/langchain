defmodule LangChain.PromptTemplate do
  @moduledoc """
  Enables defining a prompt, optionally as a template, but delaying the final
  building of it until a later time when input values are substituted in.

  This also supports the ability to create a Message from a PromptTemplate.

  An LLM conversation is made up of a set of messages. PromptTemplates are a
  tool to help build messages.

      # Create a template and convert it to a message
      prompt = PromptTemplate.new!(%{text: "My template", role: :user})
      %LangChain.Message{} = message = PromptTemplate.to_message!(prompt)

  PromptTemplates are powerful because they support Elixir's EEx templates
  allowing for parameter substitution. This is helpful when we want to prepare a
  template message and plan to later substitute in information from the user.

  Here's an example of setting up a template using a parameter then later
  providing the input value.

      prompt = PromptTemplate.from_template!("What's a name for a company that makes <%= @product %>?")

      # later, format the final text after after applying the values.
      PromptTemplate.format(prompt, %{product: "colorful socks"})
      #=> "What's a name for a company that makes colorful socks?"


  """
  use Ecto.Schema
  import Ecto.Changeset
  require Logger
  alias __MODULE__
  alias LangChain.LangChainError
  alias LangChain.Message
  alias LangChain.Message.ContentPart

  @primary_key false
  embedded_schema do
    field :text, :string
    field :inputs, :map, virtual: true, default: %{}
    field :role, Ecto.Enum, values: [:system, :user, :assistant, :function], default: :user
  end

  @type t :: %PromptTemplate{}

  @create_fields [:role, :text, :inputs]
  @required_fields [:text]

  @spec new(attrs :: map()) :: {:ok, t()} | {:error, Ecto.Changeset.t()}
  def new(attrs) do
    %PromptTemplate{}
    |> cast(attrs, @create_fields)
    |> common_validations()
    |> apply_action(:insert)
  end

  @doc """
  Create a new PromptTemplate struct using the attributes. If invalid, an
  exception is raised with the reason.

  ## Example

  A template is created using a simple map with `text` and `role` keys. The
  created template can be be converted to a `LangChain.Message`.

      PromptTemplate.new!(%{text: "My template", role: :user})

  Typically a `PromptTemplate` is used with parameter substitution as that's
  it's primary purpose. EEx is used to render the final text.

      PromptTemplate.new!(%{text: "My name is <%= @user_name %>. Warmly welcome me.", role: :user})
  """
  @spec new!(attrs :: map()) :: t() | no_return()
  def new!(attrs) do
    case new(attrs) do
      {:ok, prompt} ->
        prompt

      {:error, changeset} ->
        raise LangChainError, changeset
    end
  end

  defp common_validations(changeset) do
    changeset
    |> validate_required(@required_fields)
  end

  @doc """
  Build a PromptTemplate struct from a template string.

  Shortcut function for building a user prompt.

      {:ok, prompt} = PromptTemplate.from_template("Suggest a good name for a company that makes <%= @product %>?")

  """
  @spec from_template(text :: String.t()) :: {:ok, t()} | {:error, Ecto.Changeset.t()}
  def from_template(text) do
    PromptTemplate.new(%{text: text})
  end

  @doc """
  Build a PromptTemplate struct from a template string and return the struct or error if invalid.

  Shortcut function for building a user prompt.

      prompt = PromptTemplate.from_template!("Suggest a good name for a company that makes <%= @product %>?")

  """
  @spec from_template!(text :: String.t()) :: t() | no_return()
  def from_template!(text) do
    PromptTemplate.new!(%{text: text})
  end

  @doc """
  Format the prompt template with inputs to replace with assigns. It returns the
  formatted text.

      prompt = PromptTemplate.from_template!("Suggest a good name for a company that makes <%= @product %>?")
      PromptTemplate.format(prompt, %{product: "colorful socks"})
      #=> "Suggest a good name for a company that makes colorful socks?"

  A PromptTemplate supports storing input values on the struct. These could be
  set when the template is defined. If an input value is not provided when the
  `format` function is called, any inputs on the struct will be used.

  """
  @spec format(t(), inputs :: %{atom() => any()}) :: String.t()
  def format(%PromptTemplate{text: text} = template, %{} = inputs \\ %{}) do
    format_text(text, Map.merge(template.inputs, inputs))
  end

  @doc """
  Format the prompt template with inputs to replace embeds. The final replaced
  text is returned.

  Operates directly on text to apply the inputs. This does not take the
  PromptTemplate struct.

      PromptTemplate.format_text("Hi! My name is <%= @name %>.", %{name: "Jose"})
      #=> "Hi! My name is Jose."

  """
  @spec format_text(text :: String.t(), inputs :: %{atom() => any()}) :: String.t()
  def format_text(text, %{} = inputs) do
    # https://hexdocs.pm/eex/1.18.3/EEx.html
    EEx.eval_string(text, assigns: Map.to_list(inputs), emit_warnings: true)
  end

  @doc """
  Formats a PromptTemplate at two levels. Supports providing a list of
  `composed_of` templates that are all combined into a `full_template`.

  For this example, we'll use an overall template layout like this:

      full_prompt =
        PromptTemplate.from_template!(~s(<%= @introduction %>

        <%= @example %>

        <%= @start %>))

  This template is made up of 3 more specific templates. Let's start with the
  introduction sub-template.

      introduction_prompt =
        PromptTemplate.from_template!("You are impersonating <%= @person %>.")

  The `introduction` takes a parameter for which person it should impersonate.
  The desired person is not provided here and will come in later.

  Let's next look at the `example` prompt:

      example_prompt =
        PromptTemplate.from_template!(~s(Here's an example of an interaction:
          Q: <%= @example_q %>
          A: <%= @example_a %>))

  This defines a sample interaction for the LLM as a model of what we're looking
  for. Primarily, this template is used to define the pattern we want to use for
  the interaction. Again, this template takes parameters for the sample question
  and answer.

  Finally, there is the `start` section of the overall prompt. In this example,
  that might be a question presented by a user asking a question of our
  impersonating AI.

      start_prompt =
        PromptTemplate.from_template!(~s(Now, do this for real!
        Q: <%= @input %>
        A:))

  We have the overall template defined and templates that define each of the
  smaller portions. The `format_composed` function let's us combine those all
  together and build the complete text to pass to an LLM.

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
              example_a: "Tesla",
              input: "What's your favorite social media site?"
            }
          )

  We provide the PromptTemplate for the overall prompt, then provide the inputs,
  which are themselves prompts.

  Finally, we provide a map of values for all the parameters that still need
  values. For this example, this is what the final prompt looks like that is
  presented to the LLM.

      ~s(You are impersonating Elon Musk.

      Here's an example of an interaction:
      Q: What's your favorite car?
      A: Tesla

      Now, do this for real!
      Q: What's your favorite social media site?
      A:)

  Using a setup like this, we can easily swap out who we are impersonating and
  allow the user to interact with that persona.

  With everything defined, this is all it takes to now talk with an Abraham
  Lincoln impersonation:

      formatted_prompt =
        PromptTemplate.format_composed(
          full_prompt,
          %{
            introduction: introduction_prompt,
            example: example_prompt,
            start: start_prompt
          },
          %{
            person: "Abraham Lincoln",
            example_q: "What is your nickname?",
            example_a: "Honest Abe",
            input: "What is one of your favorite pastimes?"
          }
        )

  """
  @spec format_composed(t(), composed_of :: %{atom() => any()}, inputs :: %{atom() => any()}) ::
          String.t()
  def format_composed(%PromptTemplate{} = full_prompt, %{} = composed_of, %{} = inputs) do
    # to avoid potentially trying to replace templates on user provided content,
    # the replacements are done on the composed parts first and then replacing
    # those into the full template.

    composed_inputs =
      Enum.reduce(composed_of, %{}, fn
        {key, %PromptTemplate{} = template}, acc ->
          Map.put(acc, key, PromptTemplate.format(template, inputs))

        # Support a plain text string as a prompt that can be composed
        {key, text_prompt}, acc when is_binary(text_prompt) ->
          Map.put(acc, key, text_prompt)

        {key, other}, _acc ->
          msg = "Unsupported `composed_of` entry for #{inspect(key)}: #{inspect(other)}"
          raise LangChainError, msg
      end)

    PromptTemplate.format(full_prompt, composed_inputs)
  end

  @doc """
  Transform a `PromptTemplate` to a `LangChain.Message`. Provide the inputs at the time of
  transformation to render the final content.
  """
  @spec to_message(t(), input :: %{atom() => any()}) ::
          {:ok, Message.t()} | {:error, Ecto.Changeset.t()}
  def to_message(%PromptTemplate{} = template, %{} = inputs \\ %{}) do
    content = PromptTemplate.format(template, inputs)
    Message.new(%{role: template.role, content: content})
  end

  @doc """
  Transform a PromptTemplate to a `LangChain.Message`. Provide the inputs at the time of
  transformation to render the final content. Raises an exception if invalid.

  ## Example

  It renders a `PromptTemplate`'s `text` by applying the inputs.

      template = PromptTemplate.new!(%{text: "My name is <%= @user_name %>.", role: :user})
      messages = PromptTemplate.to_messages!([template], %{user_name: "Tim"})

  Where the final message has the contents `"My name is Tim."`
  """
  @spec to_message!(t(), input :: %{atom() => any()}) :: Message.t() | no_return()
  def to_message!(%PromptTemplate{} = template, %{} = inputs \\ %{}) do
    content = PromptTemplate.format(template, inputs)
    Message.new!(%{role: template.role, content: content})
  end

  @doc """
  Transform a PromptTemplate to a `LangChain.Message.ContentPart` of type
  `text`. Provide the inputs at the time of transformation to render the final
  content.
  """
  @spec to_content_part(t(), input :: %{atom() => any()}) ::
          {:ok, Message.t()} | {:error, Ecto.Changeset.t()}
  def to_content_part(%PromptTemplate{} = template, %{} = inputs \\ %{}) do
    content = PromptTemplate.format(template, inputs)
    ContentPart.new(%{type: :text, content: content})
  end

  @doc """
  Transform a PromptTemplate to a `LangChain.Message.ContentPart` of type
  `text`. Provide the inputs at the time of transformation to render the final
  content. Raises an exception if invalid.
  """
  @spec to_content_part!(t(), input :: %{atom() => any()}) ::
          ContentPart.t() | no_return()
  def to_content_part!(%PromptTemplate{} = template, %{} = inputs \\ %{}) do
    content = PromptTemplate.format(template, inputs)
    ContentPart.new!(%{type: :text, content: content})
  end

  @doc """
  Transform a list of PromptTemplates into a list of `LangChain.Message`s.
  Applies the inputs to the list of prompt templates. If any of the prompt
  entries are invalid or fail, an exception is raised.
  """
  @spec to_messages!([t() | Message.t() | String.t()], inputs :: %{atom() => any()}) ::
          [Message.t()] | no_return()
  def to_messages!(prompts, %{} = inputs \\ %{}) when is_list(prompts) do
    Enum.map(prompts, fn
      %PromptTemplate{} = template ->
        to_message!(template, inputs)

      # When a message has a list of content, process for PromptTemplates that
      # should become text content parts.
      %Message{content: content} = message when is_list(content) ->
        converted_content =
          Enum.map(content, fn
            %PromptTemplate{} = template ->
              to_content_part!(template, inputs)

            %ContentPart{} = part ->
              part
          end)

        %Message{message | content: converted_content}

      %Message{} = message ->
        message

      text when is_binary(text) ->
        Message.new_user!(text)
    end)
  end
end
