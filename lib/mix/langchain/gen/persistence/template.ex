defmodule Mix.Langchain.Gen.Persistence.Template do
  @moduledoc false

  @doc """
  Renders an EEx template from the sagents.gen.persistence templates directory.
  """
  def render(name, config) do
    template_path = Application.app_dir(:langchain, ["priv", "templates", "sagents.gen.persistence", name <> ".eex"])
    EEx.eval_file(template_path, assigns: config)
  end
end
