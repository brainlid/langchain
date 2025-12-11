defmodule Mix.Sagents.Gen.Persistence.Migration do
  @moduledoc false

  alias Mix.Langchain.Gen.Persistence.Template

  def generate(config) do
    timestamp = timestamp()
    filename = "#{timestamp}_create_#{config.table_prefix}persistence.exs"
    path = Path.join(["priv", "repo", "migrations", filename])

    content = Template.render("migration.exs", config)
    File.write!(path, content)

    path
  end

  defp timestamp do
    {{y, m, d}, {hh, mm, ss}} = :calendar.universal_time()
    "#{y}#{pad(m)}#{pad(d)}#{pad(hh)}#{pad(mm)}#{pad(ss)}"
  end

  defp pad(i) when i < 10, do: "0#{i}"
  defp pad(i), do: to_string(i)
end
