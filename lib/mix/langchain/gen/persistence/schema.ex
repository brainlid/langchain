defmodule Mix.Sagents.Gen.Persistence.Schema do
  @moduledoc false

  alias Mix.Langchain.Gen.Persistence.Template

  def generate_conversation(config) do
    path = schema_path(config, "conversation")
    content = Template.render("conversation.ex", config)
    File.write!(path, content)
    path
  end

  def generate_agent_state(config) do
    path = schema_path(config, "agent_state")
    content = Template.render("agent_state.ex", config)
    File.write!(path, content)
    path
  end

  def generate_display_message(config) do
    path = schema_path(config, "display_message")
    content = Template.render("display_message.ex", config)
    File.write!(path, content)
    path
  end

  defp schema_path(config, filename) do
    context_dir = config.context_module
    |> String.split(".")
    |> Enum.map(&Macro.underscore/1)
    |> Path.join()

    Path.join(["lib", context_dir, "#{filename}.ex"])
  end
end
