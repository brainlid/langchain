defprotocol Langchain.ForOpenAIApi do
  @moduledoc """
  A protocol that defines a way for converting the Langchain Elixir data structs
  to an OpenAI supported data structure and format for making an API call.
  """

  @doc """
  Protocol callback function for converting different structs into a form that
  can be passed to the OpenAI API.
  """
  @spec for_api(struct()) :: nil | %{String.t() => any()}
  def for_api(struct)
end

defimpl Langchain.ForOpenAIApi, for: Any do
  def for_api(_struct), do: nil
end
