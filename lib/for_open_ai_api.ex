# Define a protocol for converting different data types to an OpenAI
# supported data structure for an API call.
defprotocol Langchain.ForOpenAIApi do
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
