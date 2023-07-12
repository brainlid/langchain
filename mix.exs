defmodule Langchain.MixProject do
  use Mix.Project

  def project do
    [
      app: :langchain,
      version: "0.1.0",
      elixir: "~> 1.14",
      elixirc_paths: elixirc_paths(Mix.env()),
      start_permanent: Mix.env() == :prod,
      deps: deps()
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
      {:ecto, "~> 3.10.3"},
      {:gettext, "~> 0.20"},
      {:req, "~> 0.3"},
      {:abacus, "~> 2.0.0"}
    ]
  end
end
