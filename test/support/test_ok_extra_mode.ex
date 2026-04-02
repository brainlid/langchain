defmodule LangChain.Test.OkExtraMode do
  @moduledoc false
  @behaviour LangChain.Chains.LLMChain.Mode

  @impl true
  def run(chain, _opts) do
    {:ok, chain, %{tool_result: "found_it"}}
  end
end
