defmodule LangChain.Message do
  @moduledoc """
  Models a complete `Message` for a chat LLM.

  ## Roles

  - `:system` - a system message. Typically just one and it occurs first as a
    primer for how the LLM should behave.

  - `:user` - The user or application responses. Typically represents the
    "human" element of the exchange.

  - `:assistant` - Responses coming back from the LLM. This includes one or more
    "tool calls", requesting that the system execute a tool on behalf of the LLM
    and return the response.

  - `:tool` - A message for returning the result of executing a `tool` request.

  ## Tools

  A `tool_call` comes from the `:assistant` role. The `tool_id` identifies which
  of the available tool's to execute.

  Create a message of role `:tool` to provide the system responses for one or
  more tool requests. A `ToolResult` handles the response back to the LLM.

  ## User Content Parts

  Some LLMs support multi-modal messages. This means the user's message content
  can be text and/or image data. Within the LLMs, these are often referred to as
  "Vision", meaning you can provide text like "Please identify the what this is
  an image of" and provide an image.

  User Content Parts are implemented through `LangChain.Message.ContentPart`. A
  list of them can be supplied as the "content" for a message. Only a few LLMs
  support it, and they may require using specific models trained for it. See the
  documentation for the LLM or service for details on their level of support.

  ## Assistant Content Parts

  Assistant Content Parts are implemented through
  `LangChain.Message.ContentPart`. A list of them can be supplied as the
  "content" for a message. Only a few LLMs support it, and they may require
  using specific models trained for it. See the documentation for the LLM or
  service for details on their level of support.

  An example of this is Anthropic's Claude 3.7 Sonnet and it's "thinking"
  content parts. The service may also return redacted_thinking content parts
  that can be sent back to maintain continuity of the conversation.

  This also supports the idea of an assistant returning an image along with
  text.

  ## Processed Content

  The `processed_content` field is a handy place to store the results of
  processing a message and needing to hold on to the processed value and store
  it with the message.

  This is particularly helpful for a `LangChain.MessageProcessors.JsonProcessor`
  that can process an assistant message and store the processed value on the
  message itself.

  It is intended for assistant messages when a message processor is applied.
  This contains the results of the processing. This allows the `content` to
  reflect what was actually returned from the LLM so it can easily be sent back
  to the LLM as a part of the entire conversation.

  ## Examples

  A basic system message example:

      alias LangChain.Message

      Message.new_system!("You are a helpful assistant.")

  A basic user message:

      Message.new_user!("Who is Prime Minister of the moon?")

  A multi-part user message:

      alias LangChain.Message.ContentPart

      Message.new_user!([
        ContentPart.text!("What is in this picture?"),
        ContentPart.image_url!("https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg")
      ]

  """
  use Ecto.Schema
  import Ecto.Changeset
  require Logger
  alias __MODULE__
  alias LangChain.Message.ContentPart
  alias LangChain.Message.ToolCall
  alias LangChain.Message.ToolResult
  alias LangChain.PromptTemplate
  alias LangChain.LangChainError
  alias LangChain.Utils

  @primary_key false
  embedded_schema do
    # Message content that the LLM sees.
    field :content, :any, virtual: true
    field :processed_content, :any, virtual: true
    field :index, :integer
    field :status, Ecto.Enum, values: [:complete, :cancelled, :length], default: :complete

    field :role, Ecto.Enum,
      values: [:system, :user, :assistant, :tool],
      default: :user

    # Optional name of the participant. Helps separate input from different
    # individuals of the same role. Like multiple people are all acting as "user".
    field :name, :string

    # An `:assistant` role can request one or more `tool_calls` to be performed.
    field :tool_calls, :any, virtual: true

    # A `:tool` role contains one or more `tool_results` from the system having
    # used tools.
    field :tool_results, :any, virtual: true

    # Additional metadata about the message.
    field :metadata, :map
  end

  @type t :: %Message{}
  @type status :: :complete | :cancelled | :length

  @update_fields [
    :role,
    :content,
    :processed_content,
    :status,
    :tool_calls,
    :tool_results,
    :index,
    :name,
    :metadata
  ]
  @create_fields @update_fields
  @required_fields [:role]

  @doc """
  Build a new message and return an `:ok`/`:error` tuple with the result.
  """
  @spec new(attrs :: map()) :: {:ok, t()} | {:error, Ecto.Changeset.t()}
  def new(attrs \\ %{}) do
    %Message{}
    |> cast(attrs, @create_fields)
    |> Utils.migrate_to_content_parts()
    |> common_validations()
    |> apply_action(:insert)
  end

  @doc """
  Build a new message and return it or raise an error if invalid.
  """
  @spec new!(attrs :: map()) :: t() | no_return()
  def new!(attrs \\ %{}) do
    case new(attrs) do
      {:ok, message} ->
        message

      {:error, changeset} ->
        raise LangChainError, changeset
    end
  end

  @doc false
  def changeset(message, attrs) do
    message
    |> cast(attrs, @update_fields)
    |> common_validations()
  end

  defp common_validations(changeset) do
    changeset
    |> validate_required(@required_fields)
    |> validate_content_required()
    |> validate_content_type()
    |> validate_and_parse_tool_calls()
    |> validate_tool_results_required()
    |> validate_tool_results_list_type()
  end

  # validate that a "user" and "system" message has content. Allow an
  # "assistant" message to be created where we don't have content yet because it
  # can be streamed in through deltas from an LLM and not yet receive the
  # content.
  #
  # A tool result message must have content if it is returned.
  defp validate_content_required(changeset) do
    role = fetch_field!(changeset, :role)

    case role do
      role when role in [:system, :user] ->
        validate_required(changeset, [:content])

      _other ->
        changeset
    end
  end

  defp validate_content_type(changeset) do
    role = fetch_field!(changeset, :role)

    case fetch_change(changeset, :content) do
      # string message content is valid for any role
      {:ok, text} when is_binary(text) ->
        changeset

      {:ok, content} when is_list(content) ->
        if role in [:user, :assistant, :system] do
          # if a list, verify all elements are a ContentPart or PromptTemplate
          if Enum.all?(content, &(match?(%ContentPart{}, &1) or match?(%PromptTemplate{}, &1))) do
            changeset
          else
            add_error(changeset, :content, "must be text or a list of ContentParts")
          end
        else
          # only a user message can have ContentParts (except for ChatAnthropic system messages)
          Logger.error(
            "Invalid message content #{inspect(get_field(changeset, :content))} for role #{role}"
          )

          add_error(changeset, :content, "is invalid for role #{role}")
        end

      {:ok, []} ->
        put_change(changeset, :content, nil)

      # any other value is not valid
      {:ok, _value} ->
        add_error(changeset, :content, "must be text or a list of ContentParts")

      # unchanged
      :error ->
        changeset
    end
  end

  # When the message is "complete", fully validate the tool calls by parsing the
  # JSON arguments to Elixir maps. If something is invalid, errors are added to
  # the changeset.
  defp validate_and_parse_tool_calls(changeset) do
    status = get_field(changeset, :status) || :incomplete

    case status do
      :complete ->
        # fully process the tool calls
        tool_calls = get_field(changeset, :tool_calls) || []

        # Go through each tool call and "complete" it.
        # Collect any errors and report them on the changeset
        completed_calls =
          tool_calls
          |> Enum.map(fn c ->
            with %ToolCall{} = call <- c,
                 {:ok, %ToolCall{} = call} <- ToolCall.complete(call) do
              call
            else
              {:error, %Ecto.Changeset{} = changeset} ->
                # convert the error to text and return error tuple
                {:error, Utils.changeset_error_to_string(changeset)}

              {:error, %LangChainError{message: message}} ->
                {:error, message}

              {:error, reason} when is_binary(reason) ->
                {:error, reason}
            end
          end)

        # If ANY of the completed_calls is an error, add the error to the message
        # changeset
        completed = Enum.filter(completed_calls, &match?(%ToolCall{}, &1))
        errors = Enum.filter(completed_calls, &match?({:error, _reason}, &1))

        # add all valid returned tool_calls
        changeset = put_change(changeset, :tool_calls, completed)

        # add errors to the changeset for invalid entries
        Enum.reduce(errors, changeset, fn {:error, reason}, acc ->
          add_error(acc, :tool_calls, reason)
        end)

      _other ->
        changeset
    end
  end

  def validate_tool_results_required(changeset) do
    # validate that tool_results are only set when role is :tool

    # The `tool` role is required for those message types.
    case fetch_field!(changeset, :role) do
      role when role in [:tool] ->
        validate_required(changeset, [:tool_results])

      role when role in [:system, :user] ->
        if get_field(changeset, :tool_results) == nil do
          changeset
        else
          add_error(changeset, :tool_results, "can't be set with role #{inspect(role)}")
        end

      _other ->
        changeset
    end
  end

  def validate_tool_results_list_type(changeset) do
    # ensure it is a list of ToolResult structs. Testing only the first one. Dev
    # check
    role = get_field(changeset, :role)

    if role == :tool do
      case get_field(changeset, :tool_results) do
        [first | _rest] ->
          if match?(%ToolResult{}, first) do
            # valid
            changeset
          else
            add_error(changeset, :tool_results, "must be a list of ToolResult")
          end

        _other ->
          changeset
      end
    else
      changeset
    end
  end

  @doc """
  Create a new system message which can prime the AI/Assistant for how to
  respond.
  """
  @spec new_system(content :: String.t() | ContentPart.t() | [ContentPart.t()]) ::
          {:ok, t()} | {:error, Ecto.Changeset.t()}
  def new_system(content \\ "You are a helpful assistant.") do
    new(%{role: :system, content: content, status: :complete})
  end

  @doc """
  Create a new system message which can prime the AI/Assistant for how to
  respond.
  """
  @spec new_system!(content :: String.t() | ContentPart.t() | [ContentPart.t()]) ::
          t() | no_return()
  def new_system!(content \\ "You are a helpful assistant.") do
    case new_system(content) do
      {:ok, msg} ->
        msg

      {:error, %Ecto.Changeset{} = changeset} ->
        raise LangChainError, changeset
    end
  end

  @doc """
  Create a new user message which represents a human message or a message from
  the application.
  """
  @spec new_user(content :: String.t() | [ContentPart.t() | PromptTemplate.t()]) ::
          {:ok, t()} | {:error, Ecto.Changeset.t()}
  def new_user(content) do
    new(%{role: :user, content: content, status: :complete})
  end

  @doc """
  Create a new user message which represents a human message or a message from
  the application.
  """
  @spec new_user!(content :: String.t() | [ContentPart.t() | PromptTemplate.t()]) ::
          t() | no_return()
  def new_user!(content) do
    case new_user(content) do
      {:ok, msg} ->
        msg

      {:error, %Ecto.Changeset{} = changeset} ->
        raise LangChainError, changeset
    end
  end

  @doc """
  Create a new assistant message which represents a response from the AI or LLM.
  """
  @spec new_assistant(attrs :: map()) :: {:ok, t()} | {:error, Ecto.Changeset.t()}
  def new_assistant(attrs) when is_list(attrs) do
    new(%{role: :assistant, content: attrs})
  end

  def new_assistant(attrs) when is_map(attrs) do
    attrs
    |> Map.put(:role, :assistant)
    |> new()
  end

  @doc """
  Create a new assistant message which represents a response from the AI or LLM.

  ## Examples

      # Create a simple text message
      iex> new_assistant!("I'll help you with that.")
      %LangChain.Message{role: :assistant, content: [LangChain.Message.ContentPart.text!("I'll help you with that.")], tool_calls: []}

      # Create with content parts
      iex> new_assistant!([
      ...>   LangChain.Message.ContentPart.text!("Here's my response"),
      ...>   LangChain.Message.ContentPart.text!("And some additional thoughts")
      ...> ])
      %LangChain.Message{role: :assistant, content: [%LangChain.Message.ContentPart{content: "Here's my response"}, %LangChain.Message.ContentPart{content: "And some additional thoughts"}], tool_calls: []}

      # Create with tool calls
      iex> new_assistant!(%{
      ...>   tool_calls: [
      ...>     LangChain.Message.ToolCall.new!(%{call_id: "1", name: "calculator", arguments: %{x: 2, y: 3}})
      ...>   ]
      ...> })
      %LangChain.Message{role: :assistant, tool_calls: [%LangChain.Message.ToolCall{call_id: "1", name: "calculator", arguments: %{x: 2, y: 3}, status: :complete}]}
  """
  @spec new_assistant!(String.t() | map() | [ContentPart.t()]) :: t() | no_return()
  def new_assistant!(content) when is_binary(content) do
    new_assistant!(%{content: [ContentPart.text!(content)]})
  end

  def new_assistant!(attrs) when is_map(attrs) do
    case new_assistant(attrs) do
      {:ok, msg} ->
        msg

      {:error, %Ecto.Changeset{} = changeset} ->
        raise LangChainError, changeset
    end
  end

  def new_assistant!(content) when is_list(content) do
    case new_assistant(%{content: content}) do
      {:ok, msg} ->
        msg

      {:error, %Ecto.Changeset{} = changeset} ->
        raise LangChainError, changeset
    end
  end

  @doc """
  Create a new `tool` message to represent the result of a tool's execution.

  ## Attributes

  - `:tool_results` - a list of tool `ToolResult` structs.
  - `:content` - Text content returned from the LLM.
  """
  @spec new_tool_result(attrs :: map()) :: {:ok, t()} | {:error, Ecto.Changeset.t()}
  def new_tool_result(attrs \\ %{}) when is_map(attrs) and not is_struct(attrs) do
    new(%{
      role: :tool,
      tool_results: List.wrap(Map.get(attrs, :tool_results, [])),
      content: Map.get(attrs, :content, nil)
    })
  end

  @doc """
  Create a new tool response message to return the result of an executed
  tool.

  ## Attributes

  - `:tool_results` - a list of tool `ToolResult` structs.
  - `:content` - Text content returned from the LLM.
  """
  @spec new_tool_result!(attrs :: map()) :: t() | no_return()
  def new_tool_result!(attrs \\ %{}) do
    case new_tool_result(attrs) do
      {:ok, msg} ->
        msg

      {:error, %Ecto.Changeset{} = changeset} ->
        raise LangChainError, changeset
    end
  end

  @doc """
  Append a `ToolResult` to a message. A result can only be added to a `:tool`
  role message.
  """
  @spec append_tool_result(t(), ToolResult.t()) :: t() | no_return()
  def append_tool_result(%Message{role: :tool} = message, %ToolResult{} = result) do
    existing_results = message.tool_results || []
    %Message{message | tool_results: existing_results ++ [result]}
  end

  def append_tool_result(%Message{} = _message, %ToolResult{} = _result) do
    raise LangChainError, "Can only append tool results to a tool role message."
  end

  @doc """
  Return if a Message is a tool_call.
  """
  def is_tool_call?(%Message{role: :assistant, status: :complete, tool_calls: tool_calls})
      when is_list(tool_calls) and tool_calls != [],
      do: true

  def is_tool_call?(%Message{}), do: false

  @doc """
  Return if a Message is tool related. It may be a tool call or a tool result.
  """
  def is_tool_related?(%Message{role: :tool}), do: true
  def is_tool_related?(%Message{} = message), do: is_tool_call?(message)

  @doc """
  Return `true` if the message is a `tool` response and any of the `ToolResult`s
  ended in an error. Returns `false` if not a `tool` response or all
  `ToolResult`s succeeded.
  """
  @spec tool_had_errors?(t()) :: boolean()
  def tool_had_errors?(%Message{role: :tool} = message) do
    Enum.any?(message.tool_results, & &1.is_error)
  end

  def tool_had_errors?(%Message{} = _message), do: false

  @doc """
  Determines if a message is considered "empty" and likely indicates a failure.

  This is particularly useful for detecting failure patterns with Anthropic models
  where the LLM may return empty responses when it encounters issues.

  ## Examples

      iex> alias LangChain.Message
      iex> Message.is_empty?(%Message{role: :assistant, content: [], tool_calls: [], status: :complete})
      true

      iex> alias LangChain.Message
      iex> Message.is_empty?(%Message{role: :assistant, content: "Hello", tool_calls: [], status: :complete})
      false

      iex> alias LangChain.Message
      iex> Message.is_empty?(%Message{role: :user, content: [], status: :complete})
      false
  """
  @spec is_empty?(t()) :: boolean()
  def is_empty?(%Message{
        role: :assistant,
        status: :complete,
        content: content,
        tool_calls: tool_calls
      }) do
    empty_content = content == nil or content == []
    empty_tool_calls = tool_calls == nil or tool_calls == []

    empty_content and empty_tool_calls
  end

  def is_empty?(_message), do: false
end
