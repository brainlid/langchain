defmodule Mix.Tasks.Sagents.Gen.Persistence do
  @shortdoc "Generates conversation persistence for Sagents agents"

  @moduledoc """
  Generates all files needed for conversation persistence.

  ## Examples

      # Basic generation
      mix sagents.gen.persistence MyApp.Conversations

      # With all options
      mix sagents.gen.persistence MyApp.Conversations \\
        --scope MyApp.Accounts.Scope \\
        --owner-type user \\
        --owner-field user_id \\
        --owner-module MyApp.Accounts.User \\
        --table-prefix sagents_

  ## Generated files

    * Context module for conversation persistence
    * Three schema modules (Conversation, AgentState, DisplayMessage)
    * Database migration

  ## Options

    * `--scope` - Application scope module (required)
    * `--owner-type` - Owner association type (default: user)
    * `--owner-field` - Owner FK field (default: user_id)
    * `--owner-module` - Owner schema module (inferred from type)
    * `--table-prefix` - Table name prefix (default: sagents_)
    * `--repo` - Repo module (inferred from app)
    * `--web` - Web module namespace (inferred from app)
  """

  use Mix.Task
  alias Mix.Sagents.Gen.Persistence.Generator

  @switches [
    scope: :string,
    owner_type: :string,
    owner_field: :string,
    owner_module: :string,
    table_prefix: :string,
    repo: :string,
    web: :string
  ]

  @aliases [
    s: :scope,
    t: :owner_type,
    f: :owner_field
  ]

  @impl Mix.Task
  def run(args) do
    # Parse arguments
    {opts, parsed, _} = OptionParser.parse(args, switches: @switches, aliases: @aliases)

    # Validate context argument
    context_module = parse_context!(parsed)

    # Build configuration
    config = build_config(context_module, opts)

    # Generate files
    Generator.generate(config)

    # Print instructions
    print_instructions(config)
  end

  defp parse_context!([context | _]) do
    # Validate module name format
    unless context =~ ~r/^[A-Z][A-Za-z0-9.]*$/ do
      Mix.raise("Context module must be a valid Elixir module name")
    end

    context
  end

  defp parse_context!([]) do
    Mix.raise("Context module is required. Example: mix sagents.gen.persistence MyApp.Conversations")
  end

  defp build_config(context_module, opts) do
    # Infer application from context
    app = infer_app(context_module)

    # Get or prompt for scope module
    scope_module = opts[:scope] || prompt_scope() || Mix.raise("Scope module is required")

    %{
      context_module: context_module,
      context_name: context_name(context_module),
      app: app,
      app_module: app_module(app),
      scope_module: scope_module,
      owner_type: opts[:owner_type] || "user",
      owner_field: opts[:owner_field] || "user_id",
      owner_module: opts[:owner_module] || infer_owner_module(app, opts[:owner_type] || "user"),
      table_prefix: opts[:table_prefix] || "sagents_",
      repo: opts[:repo] || "#{app_module(app)}.Repo",
      web: opts[:web] || "#{app_module(app)}Web"
    }
  end

  defp prompt_scope do
    Mix.shell().prompt("Application scope module (e.g., MyApp.Accounts.Scope):")
    |> String.trim()
    |> case do
      "" -> nil
      scope -> scope
    end
  end

  defp context_name(context_module) do
    context_module
    |> String.split(".")
    |> List.last()
  end

  defp infer_app(context_module) do
    context_module
    |> String.split(".")
    |> hd()
    |> Macro.underscore()
    |> String.to_atom()
  end

  defp app_module(app) do
    app
    |> Atom.to_string()
    |> Macro.camelize()
  end

  defp infer_owner_module(app, "user"), do: "#{app_module(app)}.Accounts.User"
  defp infer_owner_module(app, "account"), do: "#{app_module(app)}.Accounts.Account"
  defp infer_owner_module(app, "organization"), do: "#{app_module(app)}.Organizations.Organization"
  defp infer_owner_module(app, "org"), do: "#{app_module(app)}.Organizations.Organization"
  defp infer_owner_module(app, "team"), do: "#{app_module(app)}.Teams.Team"
  defp infer_owner_module(_app, "none"), do: nil
  defp infer_owner_module(app, type), do: "#{app_module(app)}.#{Macro.camelize(type)}"

  defp print_instructions(config) do
    Mix.shell().info([
      :green,
      """

      Generated conversation persistence files!

      Next steps:

        1. Run migrations:

           mix ecto.migrate

        2. Customize the generated files:
           * Update #{context_module_path(config)} to use your scope fields
           * Add custom queries or business logic as needed
           * Modify schemas to add fields or validations

        3. Start using persistence in your agents:

           # In your LiveView or controller
           scope = MyApp.Accounts.Scope.for_user(current_user)

           # Create conversation
           {:ok, convo} = #{config.context_module}.create_conversation(scope, %{
             title: "My Chat"
           })

           # Save agent state during execution
           state = AgentServer.export_state(agent_id)
           #{config.context_module}.save_agent_state(convo.id, state)

           # Resume conversation
           {:ok, state} = #{config.context_module}.load_agent_state(convo.id)
           AgentServer.start_link_from_state(state, name: AgentServer.get_name(agent_id))

      """
    ])
  end

  defp context_module_path(config) do
    config.context_module
    |> String.split(".")
    |> Enum.map(&Macro.underscore/1)
    |> Path.join()
    |> then(&"lib/#{&1}.ex")
  end
end
