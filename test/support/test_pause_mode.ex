defmodule LangChain.Test.PauseMode do
  @moduledoc false
  @behaviour LangChain.Chains.LLMChain.Mode

  @impl true
  def run(chain, _opts) do
    {:pause, chain}
  end
end
