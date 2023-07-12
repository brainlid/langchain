# Define a protocol for converting different data types to an OpenAI
# supported data structure for an API call.
defprotocol Langchain.ForOpenAIApi do
  @spec for_api(struct()) :: nil | %{String.t() => any()}
  def for_api(struct)
end

defimpl Langchain.ForOpenAIApi, for: Any do
  def for_api(_struct), do: nil
end
