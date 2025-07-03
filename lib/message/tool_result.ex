defmodule LangChain.Message.ToolResult do
  @moduledoc """
  Represents a the result of running a requested tool. The LLM's requests a tool
  use through a `ToolCall`. A `ToolResult` returns the answer or result from the
  application back to the AI.

  ## Content
  The `content` can be a string or a list of ContentParts for multi-modal
  responses (text, images, etc.) that gets returned to the LLM as the result.

  ## Processed Content
  The `processed_content` field is optional. When you want to keep the results
  of the Elixir function call as a native Elixir data structure,
  `processed_content` can hold it.

  Advanced use: You can use the `options` field for LLM-specific features like
  cache control.

  To do this, the Elixir function's result should be a `{:ok, "String response
  for LLM", native_elixir_data}`. See `LangChain.Function` for details and
  examples.
  """
  use Ecto.Schema
  import Ecto.Changeset
  require Logger
  alias __MODULE__
  alias LangChain.LangChainError
  alias LangChain.Utils

  @primary_key false
  embedded_schema do
    field :type, Ecto.Enum, values: [:function], default: :function
    # When a unique ID is given for a ToolCall, the same ID is used in the
    # return to the a response to a specific request.
    field :tool_call_id, :string
    # the name of the tool that was run.
    field :name, :string
    # the content returned to the LLM/AI.
    field :content, :any, virtual: true
    # optional stored results of tool result
    field :processed_content, :any, virtual: true
    # Text to display in a UI for the result. Optional.
    field :display_text, :string
    # flag if the result is an error
    field :is_error, :boolean, default: false
    # options potentially LLM specific (i.e. cache control for Anthropic)
    field :options, :any, virtual: true
  end

  @type t :: %ToolResult{}

  @update_fields [
    :type,
    :tool_call_id,
    :name,
    :content,
    :processed_content,
    :display_text,
    :is_error,
    :options
  ]
  @create_fields @update_fields
  @required_fields [:type, :tool_call_id, :content]

  @doc """
  Build a new ToolResult and return an `:ok`/`:error` tuple with the result.
  """
  @spec new(attrs :: map()) :: {:ok, t()} | {:error, Ecto.Changeset.t()}
  def new(attrs \\ %{}) do
    %ToolResult{}
    |> cast(attrs, @create_fields)
    |> Utils.migrate_to_content_parts()
    |> common_validations()
    |> apply_action(:insert)
  end

  @doc """
  Build a new ToolResult and return it or raise an error if invalid.
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

  defp common_validations(changeset) do
    validate_required(changeset, @required_fields)
  end
end
