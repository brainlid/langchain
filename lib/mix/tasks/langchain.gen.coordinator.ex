defmodule Mix.Tasks.Langchain.Gen.Coordinator do
  use Mix.Task

  @shortdoc "Generates a Coordinator module for conversation-centric agents"

  @moduledoc """
  Generates a Coordinator module for managing conversation-centric agents.

      $ mix langchain.gen.coordinator
      $ mix langchain.gen.coordinator --module MyApp.Agents.Coordinator
      $ mix langchain.gen.coordinator --factory MyApp.Agents.Factory

  ## Options

    * `--module` - The Coordinator module name (default: MyApp.Agents.Coordinator)
    * `--factory` - The Factory module name (default: MyApp.Agents.Factory)
    * `--conversations` - The Conversations context (default: MyApp.Conversations)
    * `--pubsub` - The PubSub module (default: MyApp.PubSub)

  ## Generated Files

    * `lib/my_app/agents/coordinator.ex` - Coordinator module

  The generated Coordinator will:
    - Map conversation_id to agent_id
    - Start/stop conversation agents
    - Load state from your Conversations context
    - Create agents using your Factory
    - Handle race conditions and idempotent start

  After generation, customize the following functions:
    - `conversation_agent_id/1` - Change mapping strategy if needed
    - `create_conversation_agent/2` - Integrate with your Factory
    - `create_conversation_state/1` - Integrate with your Conversations context

  """

  @switches [
    module: :string,
    factory: :string,
    conversations: :string,
    pubsub: :string
  ]

  @impl Mix.Task
  def run(args) do
    {opts, _} = OptionParser.parse!(args, switches: @switches)

    # Infer application name
    app_name = Mix.Project.config()[:app]
    app_module = app_name |> to_string() |> Macro.camelize()

    # Parse options
    module = Keyword.get(opts, :module, "#{app_module}.Agents.Coordinator")
    factory = Keyword.get(opts, :factory, "#{app_module}.Agents.Factory")
    conversations = Keyword.get(opts, :conversations, "#{app_module}.Conversations")
    pubsub = Keyword.get(opts, :pubsub, "#{app_module}.PubSub")

    # Generate file
    binding = [
      module: module,
      factory_module: factory,
      conversations_module: conversations,
      pubsub_module: String.to_atom(pubsub)
    ]

    template_path = Application.app_dir(:langchain, "priv/templates/coordinator.ex.eex")
    content = EEx.eval_file(template_path, binding)

    # Write file
    module_path = module_to_path(module)
    File.mkdir_p!(Path.dirname(module_path))
    File.write!(module_path, content)

    Mix.shell().info("""

    Generated Coordinator module:
      #{module_path}

    Next steps:
      1. Integrate with your Factory module in `create_conversation_agent/2`
      2. Integrate with your Conversations context in `create_conversation_state/1`
      3. Customize agent_id mapping in `conversation_agent_id/1` if needed
      4. Add telemetry/logging hooks if desired

    See the module documentation for usage examples.
    """)
  end

  defp module_to_path(module) do
    path =
      module
      |> Module.split()
      |> Enum.map(&Macro.underscore/1)
      |> Path.join()

    "lib/#{path}.ex"
  end
end
