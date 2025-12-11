defmodule Mix.Sagents.Gen.Persistence.Context do
  @moduledoc false

  alias Mix.Langchain.Gen.Persistence.Template

  def generate(config) do
    path = context_path(config)
    content = Template.render("context.ex", config)
    File.write!(path, content)
    path
  end

  defp context_path(config) do
    config.context_module
    |> String.split(".")
    |> Enum.map(&Macro.underscore/1)
    |> Path.join()
    |> then(&"lib/#{&1}.ex")
  end
end
