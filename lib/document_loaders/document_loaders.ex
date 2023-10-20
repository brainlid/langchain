defmodule LangChain.DocumentLoaders do
  alias LangChain.Document

  @callback load(options :: keyword()) :: %Document{}
end
