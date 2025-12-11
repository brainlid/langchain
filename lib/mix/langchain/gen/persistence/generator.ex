defmodule Mix.Sagents.Gen.Persistence.Generator do
  @moduledoc false

  alias Mix.Sagents.Gen.Persistence.{Schema, Context, Migration}

  def generate(config) do
    # Ensure directories exist
    ensure_directories(config)

    # Generate files
    files = []

    # 1. Generate context module
    context_file = Context.generate(config)
    files = [context_file | files]

    # 2. Generate schema files
    conversation_file = Schema.generate_conversation(config)
    agent_state_file = Schema.generate_agent_state(config)
    display_message_file = Schema.generate_display_message(config)
    files = [conversation_file, agent_state_file, display_message_file | files]

    # 3. Generate migration
    migration_file = Migration.generate(config)
    files = [migration_file | files]

    # Print generated files
    print_files(files)

    :ok
  end

  defp ensure_directories(config) do
    # Context directory
    context_dir = context_directory(config)
    File.mkdir_p!(context_dir)

    # Migration directory
    File.mkdir_p!("priv/repo/migrations")
  end

  defp context_directory(config) do
    config.context_module
    |> String.split(".")
    |> Enum.map(&Macro.underscore/1)
    |> Path.join()
    |> then(&"lib/#{&1}")
  end

  defp print_files(files) do
    Mix.shell().info("\nGenerated files:")

    Enum.each(files, fn file ->
      Mix.shell().info("  * #{IO.ANSI.green()}#{file}#{IO.ANSI.reset()}")
    end)
  end
end
