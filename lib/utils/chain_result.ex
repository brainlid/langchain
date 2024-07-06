defmodule LangChain.Utils.ChainResult do
  @moduledoc """
  Module to help when working with the results of a chain.
  """
  alias LangChain.LangChainError
  alias LangChain.Chains.LLMChain
  alias LangChain.Message
  alias LangChain.Message.ContentPart
  alias __MODULE__

  @doc """
  Return the result of the chain as a string. Returned in an `:ok` tuple format.
  An `{:error, reason}` is returned for various reasons. These include:
  - The last message of the chain is not an `:assistant` message.
  - The last message of the chain is incomplete.
  - There is no last message.

  This supports passing in the final, updated LLMChain, or the result of the
  `LLMChain.run/2` function.
  """
  @spec to_string(
          LLMChain.t()
          | {:ok, LLMChain.t(), Message.t()}
          | {:error, LLMChain.t(), String.t()}
        ) ::
          {:ok, String.t()} | {:error, LLMChain.t(), String.t()}
  def to_string({:error, chain, reason}) when is_binary(reason) do
    # if an error was passed in, forward it through.
    {:error, chain, reason}
  end

  def to_string({:ok, %LLMChain{} = chain, _message}) do
    ChainResult.to_string(chain)
  end

  # when received a single ContentPart
  def to_string(%LLMChain{last_message: %Message{role: :assistant, status: :complete, content: [%ContentPart{type: :text} = part]}} = chain) do
    {:ok, part.content}
  end

  def to_string(%LLMChain{last_message: %Message{role: :assistant, status: :complete}} = chain) do
    {:ok, chain.last_message.content}
  end

  def to_string(%LLMChain{last_message: %Message{role: :assistant, status: _incomplete}} = chain) do
    {:error, chain, "Message is incomplete"}
  end

  def to_string(%LLMChain{last_message: %Message{}} = chain) do
    {:error, chain, "Message is not from assistant"}
  end

  def to_string(%LLMChain{last_message: nil} = chain) do
    {:error, chain, "No last message"}
  end

  @doc """
  Return the last message's content when it is valid to use it. Otherwise it
  raises and exception with the reason why it cannot be used. See the docs for
  `to_string/2` for details.
  """
  @spec to_string!(LLMChain.t() | {:ok, LLMChain.t(), Message.t()} | {:error, String.t()}) ::
          String.t() | no_return()
  def to_string!(%LLMChain{} = chain) do
    case ChainResult.to_string(chain) do
      {:ok, result} -> result
      {:error, _chain, reason} -> raise LangChainError, reason
    end
  end

  @doc """
  Write the result to the given map as the value of the given key.
  """
  @spec to_map(LLMChain.t(), map(), any()) :: {:ok, map()} | {:error, String.t()}
  def to_map(%LLMChain{} = chain, map, key) do
    case ChainResult.to_string(chain) do
      {:ok, value} ->
        {:ok, Map.put(map, key, value)}

      {:error, _chain, _reason} = error ->
        error
    end
  end

  @doc """
  Write the result to the given map as the value of the given key. If invalid,
  an exception is raised.
  """
  @spec to_map!(LLMChain.t(), map(), any()) :: map() | no_return()
  def to_map!(%LLMChain{} = chain, map, key) do
    case ChainResult.to_map(chain, map, key) do
      {:ok, updated} ->
        updated

      {:error, _chain, reason} ->
        raise LangChainError, reason
    end
  end
end
