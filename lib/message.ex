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

  Create a message of role `:function` to provide the function response.

  - `:is_error` - Boolean value used to track a tool response message as being
    an error or not. The error state may be returned to an LLM in different
    ways, whichever is most appropriate for the LLM. When a response is an
    error, the `content` explains the error to the LLM and depending on the
    situation, the LLM may choose try again.

  ## User Content Parts

  Some LLMs support multi-modal messages. This means the user's message content
  can be text and/or image data. Within the LLMs, these are often referred to as
  "Vision", meaning you can provide text like "Please identify the what this is
  an image of" and provide an image.

  User Content Parts are implemented through
  `LangChain.Message.ContentPart`. A list of them can be supplied as the
  "content" for a message. Only a few LLMs support it, and they may require
  using specific models trained for it. See the documentation for the LLM or
  service for details on their level of support.

  ## Examples

  A basic system message example:

      alias LangChain.Message

      Message.new_system!("You are a helpful assistant.")

  A basic user message:

      Message.new_user!("Who is Prime Minister of the moon?")

  A multi-part user message: alias LangChain.Message.ContentPart

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
  alias LangChain.LangChainError
  alias LangChain.Utils

  @primary_key false
  embedded_schema do
    field :content, :any, virtual: true
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

    # # When responding to a tool call, the `tool_call_id` specifies which tool
    # # call this is giving a response to.
    # field :tool_call_id, :string
    # # A tool response state that flags that an error occurred with the tool call
    # field :is_error, :boolean, default: false
    # TODO: Remove "function_name"
    field :function_name, :string
    # TODO: Remove "arguments"
    field :arguments, :any, virtual: true
  end

  @type t :: %Message{}
  @type status :: :complete | :cancelled | :length

  @update_fields [:role, :content, :status, :tool_calls, :tool_results, :index, :name]
  @create_fields @update_fields
  @required_fields [:role]

  @doc """
  Build a new message and return an `:ok`/`:error` tuple with the result.
  """
  @spec new(attrs :: map()) :: {:ok, t()} | {:error, Ecto.Changeset.t()}
  def new(attrs \\ %{}) do
    %Message{}
    |> cast(attrs, @create_fields)
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

      {:ok, [%ContentPart{} | _] = value} ->
        if role in [:user, :assistant] do
          # if a list, verify all elements are a ContentPart
          if Enum.all?(value, &match?(%ContentPart{}, &1)) do
            changeset
          else
            add_error(changeset, :content, "must be text or a list of ContentParts")
          end
        else
          # only a user message can have ContentParts
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

              {:error, reason} ->
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
  @spec new_system(content :: String.t()) :: {:ok, t()} | {:error, Ecto.Changeset.t()}
  def new_system(content \\ "You are a helpful assistant.") do
    new(%{role: :system, content: content, status: :complete})
  end

  @doc """
  Create a new system message which can prime the AI/Assistant for how to
  respond.
  """
  @spec new_system!(content :: String.t()) :: t() | no_return()
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
  @spec new_user(content :: String.t() | [ContentPart.t()]) ::
          {:ok, t()} | {:error, Ecto.Changeset.t()}
  def new_user(content) do
    new(%{role: :user, content: content, status: :complete})
  end

  @doc """
  Create a new user message which represents a human message or a message from
  the application.
  """
  @spec new_user!(content :: String.t() | [ContentPart.t()]) :: t() | no_return()
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
  def new_assistant(attrs \\ %{}) do
    attrs
    |> Map.put(:role, :assistant)
    |> new()
  end

  @doc """
  Create a new assistant message which represents a response from the AI or LLM.
  """
  @spec new_assistant!(String.t() | map()) :: t() | no_return()
  def new_assistant!(content) when is_binary(content) do
    new_assistant!(%{content: content})
  end

  def new_assistant!(attrs) when is_map(attrs) do
    case new_assistant(attrs) do
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
  Return if a Message is a function_call.
  """
  def is_tool_call?(%Message{role: :assistant, status: :complete, tool_calls: tool_calls})
      when is_list(tool_calls) and tool_calls != [],
      do: true

  def is_tool_call?(%Message{}), do: false
end
