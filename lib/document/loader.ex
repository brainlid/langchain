defmodule LangChain.Document.Loader do
  alias LangChain.Document

  @callback load(options :: map()) :: %Document{}
end
