defmodule LangChain.ChatModels.ChatModel do
  alias LangChain.Message
  alias LangChain.MessageDelta
  alias LangChain.Function

  @type call_response ::
          {:ok, Message.t() | [Message.t()] | [MessageDelta.t()]} | {:error, String.t()}

  @type tool :: Function.t()
  @type tools :: [tool()]

  @type t :: Ecto.Schema.t()

  @callback call(
              t(),
              String.t() | [Message.t()],
              [LangChain.Function.t()]
            ) :: call_response()

  @doc """
  Add a `LangChain.ChatModels.LLMCallbacks` callback map to the ChatModel if
  it includes the `:callback` key.
  """
  @spec add_callback(%{optional(:callbacks) => nil | map()}, map()) :: map() | struct()
  def add_callback(%_{callbacks: callbacks} = model, callback_map) do
    existing_callbacks = callbacks || []
    %{model | callbacks: existing_callbacks ++ [callback_map]}
  end

  def add_callback(model, _callback_map), do: model
end
