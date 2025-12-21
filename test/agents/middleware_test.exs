defmodule LangChain.Agents.MiddlewareTest do
  use ExUnit.Case, async: true

  alias LangChain.Agents.Middleware
  alias LangChain.Agents.MiddlewareEntry

  # Test middleware implementations

  defmodule MinimalMiddleware do
    @behaviour Middleware

    # All callbacks are optional, so this is valid
  end

  defmodule FullMiddleware do
    @behaviour Middleware

    @impl true
    def init(opts) do
      config = %{
        enabled: Keyword.get(opts, :enabled, true),
        name: Keyword.get(opts, :name, "full")
      }

      {:ok, config}
    end

    @impl true
    def system_prompt(config) do
      "System prompt from #{config.name}"
    end

    @impl true
    def tools(_config) do
      [
        LangChain.Function.new!(%{
          name: "test_tool",
          description: "A test tool",
          function: fn _args, _params -> {:ok, "result"} end
        })
      ]
    end

    @impl true
    def before_model(state, config) do
      updated_state = Map.put(state, :before_hook_called, config.name)
      {:ok, updated_state}
    end

    @impl true
    def after_model(state, config) do
      updated_state = Map.put(state, :after_hook_called, config.name)
      {:ok, updated_state}
    end

    @impl true
    def state_schema do
      __MODULE__.State
    end
  end

  defmodule FailingInitMiddleware do
    @behaviour Middleware

    @impl true
    def init(_opts) do
      {:error, "initialization failed"}
    end
  end

  defmodule MultiPromptMiddleware do
    @behaviour Middleware

    @impl true
    def system_prompt(_config) do
      ["First prompt", "Second prompt"]
    end
  end

  describe "normalize/1" do
    test "normalizes bare module to tuple" do
      assert {MinimalMiddleware, []} = Middleware.normalize(MinimalMiddleware)
    end

    test "passes through {module, opts} tuple" do
      assert {MinimalMiddleware, [key: "value"]} =
               Middleware.normalize({MinimalMiddleware, [key: "value"]})
    end

    test "raises on invalid specification" do
      assert_raise ArgumentError, fn ->
        Middleware.normalize("invalid")
      end
    end
  end

  describe "init_middleware/1" do
    test "initializes middleware with bare module" do
      %MiddlewareEntry{module: module, config: config, id: id} =
        Middleware.init_middleware(MinimalMiddleware)

      assert module == MinimalMiddleware
      assert config == %{}
      assert id == MinimalMiddleware
    end

    test "initializes middleware with options" do
      %MiddlewareEntry{module: module, config: config, id: id} =
        Middleware.init_middleware({FullMiddleware, [name: "custom"]})

      assert module == FullMiddleware
      assert config.name == "custom"
      assert config.enabled == true
      assert id == FullMiddleware
    end

    test "raises on initialization failure" do
      assert_raise RuntimeError, ~r/Failed to initialize/, fn ->
        Middleware.init_middleware(FailingInitMiddleware)
      end
    end
  end

  describe "get_system_prompt/1" do
    test "returns empty string for minimal middleware" do
      mw = Middleware.init_middleware(MinimalMiddleware)
      assert Middleware.get_system_prompt(mw) == ""
    end

    test "returns prompt string from middleware" do
      mw = Middleware.init_middleware({FullMiddleware, [name: "test"]})
      prompt = Middleware.get_system_prompt(mw)
      assert prompt == "System prompt from test"
    end

    test "joins list of prompts" do
      mw = Middleware.init_middleware(MultiPromptMiddleware)
      prompt = Middleware.get_system_prompt(mw)
      assert prompt == "First prompt\n\nSecond prompt"
    end
  end

  describe "get_tools/1" do
    test "returns empty list for minimal middleware" do
      mw = Middleware.init_middleware(MinimalMiddleware)
      assert Middleware.get_tools(mw) == []
    end

    test "returns tools from middleware" do
      mw = Middleware.init_middleware(FullMiddleware)
      tools = Middleware.get_tools(mw)
      assert length(tools) == 1
      assert hd(tools).name == "test_tool"
    end
  end

  describe "apply_before_model/2" do
    test "returns unchanged state for minimal middleware" do
      mw = Middleware.init_middleware(MinimalMiddleware)
      state = %{key: "value"}

      assert {:ok, ^state} = Middleware.apply_before_model(state, mw)
    end

    test "applies before_model hook" do
      mw = Middleware.init_middleware({FullMiddleware, [name: "test"]})
      state = %{key: "value"}

      assert {:ok, updated_state} = Middleware.apply_before_model(state, mw)
      assert updated_state.before_hook_called == "test"
      assert updated_state.key == "value"
    end
  end

  describe "apply_after_model/2" do
    test "returns unchanged state for minimal middleware" do
      mw = Middleware.init_middleware(MinimalMiddleware)
      state = %{key: "value"}

      assert {:ok, ^state} = Middleware.apply_after_model(state, mw)
    end

    test "applies after_model hook" do
      mw = Middleware.init_middleware({FullMiddleware, [name: "test"]})
      state = %{key: "value"}

      assert {:ok, updated_state} = Middleware.apply_after_model(state, mw)
      assert updated_state.after_hook_called == "test"
      assert updated_state.key == "value"
    end
  end

  describe "middleware composition" do
    test "multiple middleware can be initialized" do
      middleware_list = [
        MinimalMiddleware,
        {FullMiddleware, [name: "first"]},
        {FullMiddleware, [name: "second"]}
      ]

      initialized = Enum.map(middleware_list, &Middleware.init_middleware/1)

      assert length(initialized) == 3
    end

    test "system prompts can be collected from multiple middleware" do
      middleware_list = [
        {FullMiddleware, [name: "first"]},
        {FullMiddleware, [name: "second"]}
      ]

      prompts =
        middleware_list
        |> Enum.map(&Middleware.init_middleware/1)
        |> Enum.map(&Middleware.get_system_prompt/1)

      assert length(prompts) == 2
      assert Enum.at(prompts, 0) == "System prompt from first"
      assert Enum.at(prompts, 1) == "System prompt from second"
    end

    test "tools can be collected from multiple middleware" do
      middleware_list = [
        FullMiddleware,
        FullMiddleware
      ]

      tools =
        middleware_list
        |> Enum.map(&Middleware.init_middleware/1)
        |> Enum.flat_map(&Middleware.get_tools/1)

      assert length(tools) == 2
    end

    test "before_model hooks are applied in order" do
      defmodule BeforeMiddleware1 do
        @behaviour Middleware

        @impl true
        def before_model(state, _config) do
          calls = Map.get(state, :calls, [])
          {:ok, Map.put(state, :calls, calls ++ [:before1])}
        end
      end

      defmodule BeforeMiddleware2 do
        @behaviour Middleware

        @impl true
        def before_model(state, _config) do
          calls = Map.get(state, :calls, [])
          {:ok, Map.put(state, :calls, calls ++ [:before2])}
        end
      end

      middleware = [
        Middleware.init_middleware(BeforeMiddleware1),
        Middleware.init_middleware(BeforeMiddleware2)
      ]

      state = %{calls: []}

      final_state =
        Enum.reduce(middleware, {:ok, state}, fn mw, {:ok, current_state} ->
          Middleware.apply_before_model(current_state, mw)
        end)

      assert {:ok, %{calls: [:before1, :before2]}} = final_state
    end

    test "after_model hooks should be applied in reverse order" do
      defmodule AfterMiddleware1 do
        @behaviour Middleware

        @impl true
        def after_model(state, _config) do
          calls = Map.get(state, :calls, [])
          {:ok, Map.put(state, :calls, calls ++ [:after1])}
        end
      end

      defmodule AfterMiddleware2 do
        @behaviour Middleware

        @impl true
        def after_model(state, _config) do
          calls = Map.get(state, :calls, [])
          {:ok, Map.put(state, :calls, calls ++ [:after2])}
        end
      end

      middleware = [
        Middleware.init_middleware(AfterMiddleware1),
        Middleware.init_middleware(AfterMiddleware2)
      ]

      state = %{calls: []}

      # Apply in reverse order (as Agent.execute would do)
      final_state =
        middleware
        |> Enum.reverse()
        |> Enum.reduce({:ok, state}, fn mw, {:ok, current_state} ->
          Middleware.apply_after_model(current_state, mw)
        end)

      assert {:ok, %{calls: [:after2, :after1]}} = final_state
    end
  end
end
