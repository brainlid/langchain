defmodule LangChain.Utils.ApiOverride do
  @moduledoc """
  Tools for overriding API results. Used for testing.

  Works by setting and checking for special use of the Process dictionary.

  ## Test Example

      import LangChain.Utils.ApiOverride

      model = ChatOpenAI.new!(%{temperature: 1, stream: true})

      # Define the fake response to return
      fake_messages = [
        [MessageDelta.new!(%{role: :assistant, content: nil, status: :incomplete})],
        [MessageDelta.new!(%{content: "Sock", status: :incomplete})]
      ]

      # Made NOT LIVE here. Will not make the external call to the LLM
      set_api_override({:ok, fake_messages})

      # We can construct an LLMChain from a PromptTemplate and an LLM.
      {:ok, updated_chain, _response} =
        %{llm: model, verbose: false}
        |> LLMChain.new!()
        |> LLMChain.add_message(
          Message.new_user!("What is a good name for a company that makes colorful socks?")
        )
        |> LLMChain.run()

      assert %Message{role: :assistant, content: "Sock"} = updated_chain.last_message
  """

  @key :fake_api_response

  @doc """
  Return if an override for the API response is set. Used for testing.
  """
  @spec override_api_return? :: boolean()
  def override_api_return?() do
    @key in Process.get_keys()
  end

  @doc """
  Set the data and callback to use as a fake API response. An `:ok` tuple
  indicates a successful API call. The `fake_response_data` is the data to treat
  as returned. The `callback_name` is the callback handler name to execute.

      set_api_override({:ok, fake_response_data, callback_name_to_fire})

  ## Examples

      set_api_override({:ok, Message.new_assistant!(%{content: "154 bottles"}, :on_llm_new_message})

      set_api_override({:ok, MessageDelta.new!(%{content: "Hi"}), :on_llm_new_delta})
  """
  @spec set_api_override(term()) :: :ok
  def set_api_override(config_tuple) do
    Process.put(@key, config_tuple)
    :ok
  end

  @doc """
  Get the API override to return. Returned as `{:ok, config_tuple}`. If not set,
  it returns `:not_set`.
  """
  @spec get_api_override() :: {:ok, term()} | :not_set
  def get_api_override() do
    case Process.get(@key, :not_set) do
      :not_set ->
        :not_set

      value ->
        {:ok, value}
    end
  end
end
