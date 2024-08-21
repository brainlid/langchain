defmodule LangChain.Chains.RoutingChainTest do
  use LangChain.BaseCase
  use Mimic

  doctest LangChain.Chains.RoutingChain

  alias LangChain.Chains.RoutingChain
  alias LangChain.Routing.PromptRoute
  alias LangChain.Chains.LLMChain
  alias LangChain.Message
  alias LangChain.ChatModels.ChatOpenAI
  alias LangChain.LangChainError

  setup do
    llm = ChatOpenAI.new!(%{model: "gpt-3.5-turbo", stream: false, seed: 0})
    input_text = "Let's start a new blog post about the magical properties of pineapple cookies."

    default_route =
      %{llm: llm}
      |> LLMChain.new!()
      |> LLMChain.add_message(Message.new_system!("You are a helpful assistant."))

    blog_chain =
      %{llm: llm}
      |> LLMChain.new!()
      |> LLMChain.add_message(
        Message.new_system!("You are a helpful assistant for writing blog posts.")
      )

    memo_chain =
      %{llm: llm}
      |> LLMChain.new!()
      |> LLMChain.add_message(
        Message.new_system!("You are a helpful assistant for writing internal company memos.")
      )

    support_chain =
      %{llm: llm}
      |> LLMChain.new!()
      |> LLMChain.add_message(
        Message.new_system!(
          "You are a helpful assistant for writing support request response messages."
        )
      )

    default_route =
      PromptRoute.new!(%{
        name: "default",
        description: "When no other route is a good match",
        chain: default_route
      })

    routes = [
      PromptRoute.new!(%{
        name: "blog",
        description: "Drafting a new blog post",
        chain: blog_chain
      }),
      PromptRoute.new!(%{
        name: "memo",
        description: "Drafting a new internal company memo",
        chain: memo_chain
      }),
      PromptRoute.new!(%{
        name: "support",
        description: "Drafting a support request response",
        chain: support_chain
      })
    ]

    data = %{
      llm: llm,
      input_text: input_text,
      routes: routes,
      default_route: default_route,
      blog_chain: blog_chain,
      memo_chain: memo_chain,
      support_chain: support_chain
    }

    routing_chain = RoutingChain.new!(data)
    Map.put(data, :routing_chain, routing_chain)
  end

  describe "new/1" do
    test "defines a route", data do
      assert {:ok, router} = RoutingChain.new(data)

      assert %RoutingChain{} = router
      assert router.input_text == data[:input_text]
      assert router == data[:routing_chain]
    end

    test "requires llm, input_text, routes, and default_route" do
      assert {:error, changeset} = RoutingChain.new(%{})
      refute changeset.valid?
      assert {"can't be blank", _} = changeset.errors[:llm]
      assert {"can't be blank", _} = changeset.errors[:input_text]
      assert {"can't be blank", _} = changeset.errors[:routes]
      assert {"can't be blank", _} = changeset.errors[:default_route]
    end

    test "requires a PromptRoute assigned to default_route" do
      assert {:error, changeset} = RoutingChain.new(%{default_route: "invalid"})
      refute changeset.valid?
      assert {"must be a PromptRoute", _} = changeset.errors[:default_route]
    end
  end

  describe "new!/1" do
    test "returns the configured router", data do
      assert %RoutingChain{} = router = RoutingChain.new!(data)

      assert %RoutingChain{} = router
      assert router == data[:routing_chain]
    end

    test "raises exception when invalid", data do
      use_data = Map.delete(data, :llm)

      assert_raise LangChainError, "llm: can't be blank", fn ->
        RoutingChain.new!(use_data)
      end
    end
  end

  describe "run/2" do
    test "runs and returns updated chain and last message", %{routing_chain: routing_chain} do
      # Made NOT LIVE here
      fake_message = Message.new_assistant!("blog")

      expect(ChatOpenAI, :call, fn _model, _messages, _tools ->
        {:ok, [fake_message]}
      end)

      assert {:ok, updated_chain, [last_msg]} = RoutingChain.run(routing_chain)
      assert %LLMChain{} = updated_chain
      assert last_msg == fake_message
    end
  end

  describe "evaluate/2" do
    test "returns the selected chain to use", %{
      routing_chain: routing_chain,
      default_route: default_route
    } do
      expect(ChatOpenAI, :call, fn _model, _messages, _tools ->
        {:ok, [Message.new_assistant!("blog")]}
      end)

      assert %PromptRoute{name: "blog"} = RoutingChain.evaluate(routing_chain)

      expect(ChatOpenAI, :call, fn _model, _messages, _tools ->
        {:ok, [Message.new_assistant!("memo")]}
      end)

      assert %PromptRoute{name: "memo"} = RoutingChain.evaluate(routing_chain)

      expect(ChatOpenAI, :call, fn _model, _messages, _tools ->
        {:ok, [Message.new_assistant!("DEFAULT")]}
      end)

      assert default_route == RoutingChain.evaluate(routing_chain)
    end

    test "returns default_route when an invalid selection is made", %{
      routing_chain: routing_chain,
      default_route: default_route
    } do
      expect(ChatOpenAI, :call, fn _model, _messages, _tools ->
        {:ok, [Message.new_assistant!("invalid")]}
      end)

      assert default_route == RoutingChain.evaluate(routing_chain)
    end

    test "returns default_route when something goes wrong", %{
      routing_chain: routing_chain,
      default_route: default_route
    } do
      expect(ChatOpenAI, :call, fn _model, _messages, _tools ->
        {:error, "FAKE API call failure"}
      end)

      assert default_route == RoutingChain.evaluate(routing_chain)
    end
  end
end
