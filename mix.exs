defmodule LangChain.MixProject do
  use Mix.Project

  @source_url "https://github.com/brainlid/langchain"
  @version "0.4.0-rc.1"

  def project do
    [
      app: :langchain,
      version: @version,
      elixir: "~> 1.16",
      elixirc_paths: elixirc_paths(Mix.env()),
      test_options: [docs: true],
      start_permanent: Mix.env() == :prod,
      deps: deps(),
      package: package(),
      docs: &docs/0,
      name: "LangChain",
      homepage_url: @source_url,
      description: """
      Elixir implementation of a LangChain style framework that lets Elixir projects integrate with and leverage LLMs.
      """
    ]
  end

  # Run "mix help compile.app" to learn about applications.
  def application do
    [
      extra_applications: [:logger]
    ]
  end

  defp elixirc_paths(:test), do: ["lib", "test/support"]
  defp elixirc_paths(_), do: ["lib"]

  # Run "mix help deps" to learn about dependencies.
  defp deps do
    [
      {:ecto, "~> 3.10 or ~> 3.11"},
      {:gettext, "~> 0.26"},
      {:req, ">= 0.5.2"},
      {:nimble_parsec, "~> 1.4", optional: true},
      {:abacus, "~> 2.1.0", optional: true},
      {:nx, ">= 0.7.0", optional: true},
      {:ex_doc, "~> 0.34", only: :dev, runtime: false},
      {:mimic, "~> 1.8", only: :test}
    ]
  end

  defp docs do
    [
      main: "readme",
      source_ref: "v#{@version}",
      source_url: @source_url,
      assets: %{"notebooks/files" => "files"},
      skip_undefined_reference_warnings_on: ["CHANGELOG.md"],
      logo: "images/elixir-langchain-link-logo_32px.png",
      extra_section: "Guides",
      extras: extras(),
      groups_for_extras: [
        Notebooks: Path.wildcard("notebooks/*.livemd")
      ],
      groups_for_modules: [
        "Chat Models": [
          LangChain.ChatModels.ChatOpenAI,
          LangChain.ChatModels.ChatAnthropic,
          LangChain.ChatModels.ChatBumblebee,
          LangChain.ChatModels.ChatGoogleAI,
          LangChain.ChatModels.ChatVertexAI,
          LangChain.ChatModels.ChatMistralAI,
          LangChain.ChatModels.ChatOllamaAI,
          LangChain.ChatModels.ChatPerplexity,
          LangChain.ChatModels.ChatModel
        ],
        Chains: [
          LangChain.Chains.LLMChain,
          LangChain.Chains.TextToTitleChain,
          LangChain.Chains.SummarizeConversationChain,
          LangChain.Chains.DataExtractionChain
        ],
        Messages: [
          LangChain.Message,
          LangChain.MessageDelta,
          LangChain.Message.ContentPart,
          LangChain.Message.ToolCall,
          LangChain.Message.ToolResult,
          LangChain.PromptTemplate,
          LangChain.MessageProcessors,
          LangChain.MessageProcessors.JsonProcessor,
          LangChain.TokenUsage
        ],
        Functions: [
          LangChain.Function,
          LangChain.FunctionParam
        ],
        Callbacks: [
          LangChain.Callbacks,
          LangChain.Chains.ChainCallbacks
        ],
        Routing: [
          LangChain.Chains.RoutingChain,
          LangChain.Routing.PromptRoute
        ],
        Images: [
          LangChain.Images,
          LangChain.Images.OpenAIImage,
          LangChain.Images.GeneratedImage
        ],
        "Text Splitter": [
          LangChain.TextSplitter.CharacterTextSplitter,
          LangChain.TextSplitter.RecursiveCharacterTextSplitter,
          LangChain.TextSplitter.LanguageSeparators
        ],
        Tools: [
          LangChain.Tools.Calculator
        ],
        Utils: [
          LangChain.Utils,
          LangChain.Utils.BedrockConfig,
          LangChain.Utils.ChatTemplates,
          LangChain.Utils.ChainResult,
          LangChain.Config,
          LangChain.Gettext
        ]
      ]
    ]
  end

  defp extras do
    [
      "README.md",
      "CHANGELOG.md",
      "notebooks/getting_started.livemd",
      "notebooks/custom_functions.livemd",
      "notebooks/context-specific-image-descriptions.livemd"
    ]
  end

  defp package do
    # Note: the Livebook notebooks and related files are not included in the
    # package.
    [
      files: ["lib", "mix.exs", "README*", "LICENSE*"],
      maintainers: ["Mark Ericksen"],
      licenses: ["Apache-2.0"],
      links: %{"GitHub" => @source_url}
    ]
  end
end
