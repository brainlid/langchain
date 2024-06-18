defmodule LangChain.MixProject do
  use Mix.Project

  @version "0.3.0-rc.0"

  def project do
    [
      app: :langchain,
      version: @version,
      elixir: "~> 1.14",
      elixirc_paths: elixirc_paths(Mix.env()),
      test_options: [docs: true],
      start_permanent: Mix.env() == :prod,
      deps: deps(),
      package: package(),
      docs: &docs/0,
      name: "LangChain",
      homepage_url: "https://github.com/brainlid/langchain",
      description: """
      Elixir implementation of a LangChain style framework.
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
      {:gettext, "~> 0.20"},
      {:req, ">= 0.5.0"},
      {:abacus, "~> 2.1.0"},
      {:nx, ">= 0.7.0", optional: true},
      {:ex_doc, "~> 0.27", only: :dev, runtime: false}
    ]
  end

  defp docs do
    [
      main: "getting_started",
      source_ref: "v#{@version}",
      source_url: "https://github.com/brainlid/langchain",
      extra_section: "GUIDES",
      extras: extras(),
      skip_undefined_reference_warnings_on: ["CHANGELOG.md"]
    ]
  end

  defp extras do
    ["CHANGELOG.md"] ++ Path.wildcard("guides/*.md")
  end

  defp package do
    # Note: the Livebook notebooks and related files are not included in the
    # package.
    [
      files: ["lib", "mix.exs", "README*", "LICENSE*"],
      maintainers: ["Mark Ericksen"],
      licenses: ["Apache-2.0"],
      links: %{"GitHub" => "https://github.com/brainlid/langchain"}
    ]
  end
end
