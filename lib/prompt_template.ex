defmodule Langchain.PromptTemplate do
  @moduledoc """
  Enables defining a prompt, optionally as a template, but delaying the final
  building of it until a later time when input values can be substituted in.

  This also supports the ability to create a Message from a PromptTemplate.

      prompt = PromptTemplate.new!(%{text: "My template", role: :user})
      %Langchain.Message{} = message = PromptTemplate.to_message(prompt)
  """
  use Ecto.Schema
  import Ecto.Changeset
  require Logger
  alias __MODULE__
  alias Langchain.LangchainError
  alias Langchain.Message

  # TODO: Langchain has ChatPromptTemplate that can be created `from_prompt_messages`
  # TODO: Then it passes a SystemMessagePromptTemplate.from_template()
  # TODO: - the idea I guess is if you know the message is a template, you can search for replacements. However, if the contents are NOT a template and actually deal with Phoenix template code, it would include things that could be replaced... messing things up.
  # TODO: - could create a `Message.from_template()` which would just be marked as a template internally in the struct.
  #        Then, when formatted, the `template?` flag would be flipped to `false` so it wouldn't be replaced again. That means a user's content could be Elixir template code and it wouldn't get messed up.
  # TODO: I also have the PromptTemplate support for roles. So it could be created that way. Then a chat_prompt list of prompts could be formatted with inputs to create all Messages where they are not templated at that point.
  #      - allow mixing a list of Message and PromptTemplate (with role)
  #      - works for conversation because they are just messages at that point.
  # TODO: a list of PromptTemplates can .format into a list of strings.
  # TODO: a list of PromptTemplates can .to_message into a list of formatted Messages.
  # TODO: a plain String is a prompt just by it's self. Allow Strings and PromptTepmlates to intermix in a list. Wouldn't expect to find those with a Message because a string can't indicate what chat message type it should be.

  @primary_key false
  embedded_schema do
    field(:text, :string)
    field(:inputs, :map, virtual: true, default: %{})
    field(:role, Ecto.Enum, values: [:system, :user, :assistant, :function], default: :user)
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
  """
  @spec new!(attrs :: map()) :: t() | no_return()
  def new!(attrs) do
    case new(attrs) do
      {:ok, prompt} ->
        prompt

      {:error, changeset} ->
        raise LangchainError, changeset
    end
  end

  defp common_validations(changeset) do
    changeset
    |> validate_required(@required_fields)
  end

  # def build_prompt(text, %{} = inputs) do
  #   EEx.eval_string(text, Map.to_list(inputs))
  # end

  @doc """
  Build a PromptTemplate struct from a template string.
  """
  @spec from_template(text :: String.t()) :: {:ok, t()} | {:error, Ecto.Changeset.t()}
  def from_template(text) do
    PromptTemplate.new(%{text: text})
  end

  @doc """
  Build a PromptTemplate struct from a template string.
  """
  @spec from_template!(text :: String.t()) :: t() | no_return()
  def from_template!(text) do
    PromptTemplate.new!(%{text: text})
  end

  @doc """
  Format the prompt template with inputs to replace with assigns. It returns the
  formatted text.
  """
  @spec format(t(), inputs :: %{atom() => any()}) :: String.t()
  def format(%PromptTemplate{text: text} = template, %{} = inputs \\ %{}) do
    format_text(text, Map.merge(template.inputs, inputs))
  end

  @doc """
  Format the prompt template with inputs to replace embeds. The final replaced
  text is returned.
  """
  @spec format_text(text :: String.t(), inputs :: %{atom() => any()}) :: String.t()
  def format_text(text, %{} = inputs) do
    EEx.eval_string(text, assigns: Map.to_list(inputs))
  end

  @doc """
  Format a PromptTemplate at two levels. Supports providing a list of
  `composed_of` templates that can all be combined into a `full_template`.
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
          Logger.error(msg)
          raise LangchainError, msg
      end)

    PromptTemplate.format(full_prompt, composed_inputs)
  end

  @doc """
  Transform a PromptTemplate to a `Message`. Provide the inputs at the time of
  transformation to render the final content.
  """
  @spec to_message(t(), input :: %{atom() => any()}) ::
          {:ok, Message.t()} | {:error, Ecto.Changeset.t()}
  def to_message(%PromptTemplate{} = template, %{} = inputs \\ %{}) do
    content = PromptTemplate.format(template, inputs)
    Message.new(%{role: template.role, content: content})
  end

  @doc """
  Transform a PromptTemplate to a `Message`. Provide the inputs at the time of
  transformation to render the final content. Raises an exception if invalid.
  """
  @spec to_message!(t(), input :: %{atom() => any()}) :: Message.t() | no_return()
  def to_message!(%PromptTemplate{} = template, %{} = inputs \\ %{}) do
    content = PromptTemplate.format(template, inputs)
    Message.new!(%{role: template.role, content: content})
  end

  @doc """
  Transform a list of PromptTemplates into a list of messages. Applies the
  inputs to the list of prompt templates. If any of the prompt entries are
  invalid or fail, an exception is raised.
  """
  @spec to_messages!([t() | Message.t() | String.t()], inputs :: %{atom() => any()}) :: [Message.t()] | no_return()
  def to_messages!(prompts, %{} = inputs \\ %{}) when is_list(prompts) do
    Enum.map(prompts, fn
      %PromptTemplate{} = template ->
        to_message!(template, inputs)

      %Message{} = message ->
        message

      text when is_binary(text) ->
        Message.new_user!(text)
    end)
  end
end
