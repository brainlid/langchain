defmodule LangChain.Routing.PromptRouteTest do
  use ExUnit.Case

  doctest LangChain.Routing.PromptRoute

  alias LangChain.Routing.PromptRoute
  alias LangChain.Chains.LLMChain
  alias LangChain.LangChainError

  describe "new/1" do
    test "defines a route" do
      assert {:ok, route} =
               PromptRoute.new(%{
                 name: "testing",
                 description: "For testing things.",
                 chain: %LLMChain{}
               })

      assert %PromptRoute{} = route
      assert route.name == "testing"
      assert route.description == "For testing things."
      assert route.chain == %LLMChain{}
    end

    test "requires name" do
      assert {:error, changeset} = PromptRoute.new(%{})
      refute changeset.valid?
      assert {"can't be blank", _} = changeset.errors[:name]
    end

    test "does not require a chain" do
      assert {:ok, %PromptRoute{}} =
               PromptRoute.new(%{name: "thing", description: "stuff", chain: nil})
    end

    test "does not require a description" do
      assert {:ok, %PromptRoute{}} =
               PromptRoute.new(%{name: "thing", description: nil, chain: nil})
    end
  end

  describe "new!/1" do
    test "returns the configured route" do
      assert %PromptRoute{} =
               route = PromptRoute.new!(%{name: "name", description: "desc", chain: %LLMChain{}})

      assert %PromptRoute{} = route
      assert route.name == "name"
      assert route.description == "desc"
      assert route.chain == %LLMChain{}
    end

    test "raises exception when invalid" do
      assert_raise LangChainError, "name: can't be blank", fn ->
        PromptRoute.new!(%{})
      end
    end
  end

  describe "get_selected/2" do
    setup do
      routes = [
        PromptRoute.new!(%{
          name: "blog",
          description: "Drafting a blog post",
          chain: %LLMChain{}
        }),
        PromptRoute.new!(%{name: "memo", description: "Drafting a memo", chain: %LLMChain{}})
      ]

      %{routes: routes}
    end

    test "returns the matching route from the selected name", %{routes: routes} do
      assert %PromptRoute{name: "blog"} = PromptRoute.get_selected("blog", routes)
      assert %PromptRoute{name: "memo"} = PromptRoute.get_selected("memo", routes)
    end

    test "returns nil when not found", %{routes: routes} do
      assert nil == PromptRoute.get_selected("missing", routes)
    end
  end
end
