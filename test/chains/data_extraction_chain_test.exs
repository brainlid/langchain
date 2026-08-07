defmodule LangChain.Chains.DataExtractionChainTest do
  use LangChain.BaseCase
  use Mimic

  doctest LangChain.Chains.DataExtractionChain

  alias LangChain.Chains.LLMChain
  alias LangChain.Function
  alias LangChain.FunctionParam
  alias LangChain.Chains.DataExtractionChain
  alias LangChain.ChatModels.ChatOpenAI
  alias LangChain.ChatModels.ChatOpenAIResponses
  alias LangChain.ChatModels.ChatGrok
  alias LangChain.LangChainError
  alias LangChain.Message
  alias LangChain.Message.ToolCall
  alias LangChain.TokenUsage

  describe "build_extract_function/1" do
    test "parameters_schema is set correctly" do
      property_config =
        [
          FunctionParam.new!(%{name: "person_name", type: :string}),
          FunctionParam.new!(%{name: "person_age", type: :number}),
          FunctionParam.new!(%{name: "person_hair_color", type: :string}),
          FunctionParam.new!(%{name: "pet_dog_name", type: :string}),
          FunctionParam.new!(%{name: "pet_dog_breed", type: :string})
        ]
        |> FunctionParam.to_parameters_schema()

      %Function{} = function = DataExtractionChain.build_extract_function(property_config)

      # the full combined JSONSchema structure for function arguments
      assert function.parameters_schema == %{
               type: "object",
               properties: %{
                 info: %{
                   type: "array",
                   items: %{
                     "type" => "object",
                     "properties" => %{
                       "pet_dog_breed" => %{"type" => "string"},
                       "pet_dog_name" => %{"type" => "string"},
                       "person_age" => %{"type" => "number"},
                       "person_hair_color" => %{"type" => "string"},
                       "person_name" => %{"type" => "string"}
                     },
                     "required" => [],
                     "additionalProperties" => false
                   }
                 }
               },
               required: ["info"]
             }
    end
  end

  describe "normalize_extraction_info/1" do
    test "wraps a single map as one row (models sometimes omit the JSON array)" do
      single_map = %{
        "person_name" => "Alex",
        "person_age" => nil,
        "person_hair_color" => "blonde",
        "pet_dog_name" => "Frosty",
        "pet_dog_breed" => "labrador"
      }

      assert {:ok, [^single_map]} = DataExtractionChain.normalize_extraction_info(single_map)
    end

    test "leaves a list of maps unchanged" do
      rows = [
        %{"person_name" => "Alex"},
        %{"person_name" => "Claudia"}
      ]

      assert {:ok, ^rows} = DataExtractionChain.normalize_extraction_info(rows)
    end

    test "returns error for non-list non-map info" do
      assert {:error, %LangChain.LangChainError{}} =
               DataExtractionChain.normalize_extraction_info("not a row")
    end
  end

  describe "extract_result/1" do
    test "returns the extracted data from the chain's tool call" do
      chain = chain_with_last_message(extraction_message())

      assert {:ok, [%{"person_name" => "Alex"}]} = DataExtractionChain.extract_result(chain)
    end

    test "returns an error when the LLM did not make the extraction tool call" do
      chain = chain_with_last_message(Message.new_assistant!(%{content: "I'm not sure."}))

      assert {:error, %LangChainError{message: message}} =
               DataExtractionChain.extract_result(chain)

      assert message =~ "Unexpected response."
    end
  end

  describe "run_chain/4 with strategy: :tool_strategy" do
    setup do
      schema_parameters =
        [FunctionParam.new!(%{name: "person_name", type: :string})]
        |> FunctionParam.to_parameters_schema()

      {:ok, chat} = ChatOpenAI.new(%{model: "gpt-4o-mini-2024-07-18", stream: false})

      %{schema_parameters: schema_parameters, chat: chat}
    end

    test "returns the executed chain, exposing token usage", %{
      schema_parameters: schema_parameters,
      chat: chat
    } do
      message = extraction_message(%{usage: %TokenUsage{input: 42, output: 7}})

      expect(ChatOpenAI, :call, fn _model, _messages, tools ->
        assert tools != []
        {:ok, [message]}
      end)

      assert {:ok, %LLMChain{last_message: %Message{role: :assistant} = last_message} = chain} =
               DataExtractionChain.run_chain(chat, schema_parameters, "Alex is here.",
                 strategy: :tool_strategy
               )

      assert TokenUsage.get(last_message) == %TokenUsage{input: 42, output: 7}
      assert {:ok, [%{"person_name" => "Alex"}]} = DataExtractionChain.extract_result(chain)
    end

    test "returns the chain even when the LLM did not make the extraction tool call", %{
      schema_parameters: schema_parameters,
      chat: chat
    } do
      message =
        Message.new_assistant!(%{
          content: "I'm not sure.",
          metadata: %{usage: %TokenUsage{input: 12, output: 3}}
        })

      expect(ChatOpenAI, :call, fn _model, _messages, _tools -> {:ok, [message]} end)

      assert {:ok, %LLMChain{last_message: last_message} = chain} =
               DataExtractionChain.run_chain(chat, schema_parameters, "Alex is here.",
                 strategy: :tool_strategy
               )

      # the usage is still reportable even though the extraction failed
      assert TokenUsage.get(last_message) == %TokenUsage{input: 12, output: 3}
      assert {:error, %LangChainError{}} = DataExtractionChain.extract_result(chain)
    end
  end

  describe "run/4 with strategy: :tool_strategy" do
    setup do
      schema_parameters =
        [FunctionParam.new!(%{name: "person_name", type: :string})]
        |> FunctionParam.to_parameters_schema()

      {:ok, chat} = ChatOpenAI.new(%{model: "gpt-4o-mini-2024-07-18", stream: false})

      %{schema_parameters: schema_parameters, chat: chat}
    end

    test "returns the extracted result", %{schema_parameters: schema_parameters, chat: chat} do
      expect(ChatOpenAI, :call, fn _model, _messages, tools ->
        assert tools != []
        {:ok, [extraction_message()]}
      end)

      assert {:ok, [%{"person_name" => "Alex"}]} =
               DataExtractionChain.run(chat, schema_parameters, "Alex is here.",
                 strategy: :tool_strategy
               )
    end

    test "returns an error when the LLM did not make the extraction tool call", %{
      schema_parameters: schema_parameters,
      chat: chat
    } do
      expect(ChatOpenAI, :call, fn _model, _messages, _tools ->
        {:ok, [Message.new_assistant!(%{content: "I'm not sure."})]}
      end)

      assert {:error, %LangChainError{message: message}} =
               DataExtractionChain.run(chat, schema_parameters, "Alex is here.",
                 strategy: :tool_strategy
               )

      assert message =~ "Unexpected response."
    end
  end

  describe "supports_provider_strategy?/1" do
    test "returns true for a struct/module defining both :json_schema and :json_response" do
      {:ok, chat} = ChatOpenAI.new(%{model: "gpt-4o-mini-2024-07-18", stream: false})

      assert DataExtractionChain.supports_provider_strategy?(chat)
      assert DataExtractionChain.supports_provider_strategy?(ChatOpenAI)
    end

    test "returns false for a struct/module missing either field" do
      chat = ChatGrok.new!(%{})

      refute DataExtractionChain.supports_provider_strategy?(chat)
      refute DataExtractionChain.supports_provider_strategy?(ChatGrok)
    end
  end

  describe "run_chain/4 default strategy (:provider_strategy)" do
    setup do
      schema_parameters =
        [FunctionParam.new!(%{name: "person_name", type: :string})]
        |> FunctionParam.to_parameters_schema()

      # Nothing but the model itself is configured up front; the chain patches
      # in json_response/json_schema from schema_parameters at call time.
      {:ok, chat} = ChatOpenAI.new(%{model: "gpt-4o-mini-2024-07-18", stream: false})

      %{schema_parameters: schema_parameters, chat: chat}
    end

    test "patches json_response/json_schema (unwrapped, as given) onto the llm and skips adding a tool, with no :strategy option given",
         %{
           schema_parameters: schema_parameters,
           chat: chat
         } do
      message =
        Message.new_assistant!(%{
          content: Jason.encode!(%{"person_name" => "Alex"})
        })

      expect(ChatOpenAI, :call, fn model, _messages, tools ->
        assert tools == []
        assert model.json_response == true
        assert model.json_schema == schema_parameters
        {:ok, [message]}
      end)

      assert {:ok, [%{"person_name" => "Alex"}]} =
               DataExtractionChain.run(chat, schema_parameters, "Alex is here.")
    end

    test "extracts a list directly when json_schema is array-typed", %{chat: chat} do
      item_schema =
        [FunctionParam.new!(%{name: "person_name", type: :string})]
        |> FunctionParam.to_parameters_schema()

      array_schema = %{type: "array", items: item_schema}

      message =
        Message.new_assistant!(%{
          content: Jason.encode!([%{"person_name" => "Alex"}, %{"person_name" => "Claudia"}])
        })

      expect(ChatOpenAI, :call, fn model, _messages, tools ->
        assert tools == []
        assert model.json_schema == array_schema
        {:ok, [message]}
      end)

      assert {:ok, [%{"person_name" => "Alex"}, %{"person_name" => "Claudia"}]} =
               DataExtractionChain.run(chat, array_schema, "Alex and Claudia are here.")
    end

    test "returns an error when the JSON response is not valid JSON", %{
      schema_parameters: schema_parameters,
      chat: chat
    } do
      message = Message.new_assistant!(%{content: "not json"})

      expect(ChatOpenAI, :call, fn _model, _messages, _tools -> {:ok, [message]} end)

      assert {:error, %LangChainError{}} =
               DataExtractionChain.run(chat, schema_parameters, "Alex is here.")
    end

    test "fills in extra fields (e.g. json_schema_name) with sensible defaults when the struct defines them",
         %{schema_parameters: schema_parameters} do
      {:ok, chat} = ChatOpenAIResponses.new(%{stream: false})

      message =
        Message.new_assistant!(%{
          content: Jason.encode!(%{"person_name" => "Alex"})
        })

      expect(ChatOpenAIResponses, :call, fn model, _messages, tools ->
        assert tools == []
        assert model.json_schema == schema_parameters
        assert model.json_schema_name == "information_extraction"
        {:ok, [message]}
      end)

      assert {:ok, [%{"person_name" => "Alex"}]} =
               DataExtractionChain.run(chat, schema_parameters, "Alex is here.")
    end
  end

  describe "run_chain/4 default strategy (:provider_strategy) with an unsupported llm" do
    test "fails gracefully by falling back to :tool_strategy, logging a warning" do
      schema_parameters =
        [FunctionParam.new!(%{name: "person_name", type: :string})]
        |> FunctionParam.to_parameters_schema()

      chat = ChatGrok.new!(%{})

      expect(ChatGrok, :call, fn _model, _messages, tools ->
        assert tools != []
        {:ok, [extraction_message()]}
      end)

      {result, log} =
        ExUnit.CaptureLog.with_log(fn ->
          DataExtractionChain.run(chat, schema_parameters, "Alex is here.")
        end)

      assert {:ok, [%{"person_name" => "Alex"}]} = result
      assert log =~ "does not support :provider_strategy"
      assert log =~ "Falling back to :tool_strategy"
    end
  end

  describe "run_chain/4 :strategy option" do
    test "raises for an invalid :strategy value" do
      schema_parameters =
        [FunctionParam.new!(%{name: "person_name", type: :string})]
        |> FunctionParam.to_parameters_schema()

      {:ok, chat} = ChatOpenAI.new(%{model: "gpt-4o-mini-2024-07-18", stream: false})

      assert_raise LangChainError, ~r/Invalid :strategy/, fn ->
        DataExtractionChain.run_chain(chat, schema_parameters, "Alex is here.", strategy: :bogus)
      end
    end

    test "raises (no fallback) when strategy: :provider_strategy is given explicitly for an unsupported llm" do
      schema_parameters =
        [FunctionParam.new!(%{name: "person_name", type: :string})]
        |> FunctionParam.to_parameters_schema()

      chat = ChatGrok.new!(%{})

      assert_raise LangChainError, ~r/does not support :provider_strategy/, fn ->
        DataExtractionChain.run_chain(chat, schema_parameters, "Alex is here.",
          strategy: :provider_strategy
        )
      end
    end
  end

  describe "run_chain/4 :callbacks option" do
    setup do
      schema_parameters =
        [FunctionParam.new!(%{name: "person_name", type: :string})]
        |> FunctionParam.to_parameters_schema()

      {:ok, chat} = ChatOpenAI.new(%{model: "gpt-4o-mini-2024-07-18", stream: false})

      %{schema_parameters: schema_parameters, chat: chat}
    end

    test "registers callbacks on the internally run LLMChain", %{
      schema_parameters: schema_parameters,
      chat: chat
    } do
      test_pid = self()

      handler = %{
        on_message_processed: fn _chain, message -> send(test_pid, {:processed, message}) end,
        on_llm_token_usage: fn _chain, usage -> send(test_pid, {:usage, usage}) end
      }

      message = extraction_message(%{usage: TokenUsage.new!(%{input: 42, output: 7})})

      expect(ChatOpenAI, :call, fn _model, _messages, _tools -> {:ok, [message]} end)

      assert {:ok, %LLMChain{}} =
               DataExtractionChain.run_chain(chat, schema_parameters, "Alex is here.",
                 callbacks: [handler],
                 strategy: :tool_strategy
               )

      assert_received {:processed, %Message{}}
      assert_received {:usage, %TokenUsage{input: 42, output: 7}}
    end

    test "are also accepted by run/4", %{schema_parameters: schema_parameters, chat: chat} do
      test_pid = self()

      handler = %{
        on_llm_token_usage: fn _chain, usage -> send(test_pid, {:usage, usage}) end
      }

      message = extraction_message(%{usage: TokenUsage.new!(%{input: 42, output: 7})})

      expect(ChatOpenAI, :call, fn _model, _messages, _tools -> {:ok, [message]} end)

      assert {:ok, [%{"person_name" => "Alex"}]} =
               DataExtractionChain.run(chat, schema_parameters, "Alex is here.",
                 callbacks: [handler],
                 strategy: :tool_strategy
               )

      assert_received {:usage, %TokenUsage{input: 42, output: 7}}
    end

    test "fire even when the extraction tool call is missing", %{
      schema_parameters: schema_parameters,
      chat: chat
    } do
      test_pid = self()

      handler = %{
        on_llm_token_usage: fn _chain, usage -> send(test_pid, {:usage, usage}) end
      }

      message =
        Message.new_assistant!(%{
          content: "I'm not sure.",
          metadata: %{usage: TokenUsage.new!(%{input: 12, output: 3})}
        })

      expect(ChatOpenAI, :call, fn _model, _messages, _tools -> {:ok, [message]} end)

      assert {:error, %LangChainError{}} =
               DataExtractionChain.run(chat, schema_parameters, "Alex is here.",
                 callbacks: [handler],
                 strategy: :tool_strategy
               )

      assert_received {:usage, %TokenUsage{input: 12, output: 3}}
    end
  end

  defp extraction_message(metadata \\ nil) do
    Message.new_assistant!(%{
      tool_calls: [
        ToolCall.new!(%{
          call_id: "call_123",
          name: "information_extraction",
          arguments: %{"info" => [%{"person_name" => "Alex"}]}
        })
      ],
      metadata: metadata
    })
  end

  defp chain_with_last_message(%Message{} = message) do
    %{llm: ChatOpenAI.new!(%{model: "gpt-4o-mini-2024-07-18", stream: false})}
    |> LLMChain.new!()
    |> LLMChain.add_message(message)
  end

  # Extraction - https://js.langchain.com/docs/modules/chains/openai_functions/extraction
  @tag live_call: true, live_open_ai: true
  test "data extraction chain" do
    # JSONSchema definition
    schema_parameters =
      [
        FunctionParam.new!(%{name: "person_name", type: :string}),
        FunctionParam.new!(%{name: "person_age", type: :number}),
        FunctionParam.new!(%{name: "person_hair_color", type: :string}),
        FunctionParam.new!(%{name: "pet_dog_name", type: :string}),
        FunctionParam.new!(%{name: "pet_dog_breed", type: :string})
      ]
      |> FunctionParam.to_parameters_schema()

    # Model setup - specify the model and seed
    {:ok, chat} =
      ChatOpenAI.new(%{model: "gpt-4o-mini-2024-07-18", temperature: 0, seed: 0, stream: false})

    # run the chain, chain.run(prompt to extract data from)
    data_prompt = """
      Alex is 5 feet tall. Claudia is 4 feet taller than Alex and jumps higher than him.
      Claudia is a brunette and Alex is blonde.
      Alex's dog Frosty is a labrador and likes to play hide and seek. Identify each person and their relevant information.
    """

    {:ok, result} =
      DataExtractionChain.run(chat, schema_parameters, data_prompt,
        verbose: true,
        strategy: :tool_strategy
      )

    assert result == [
             %{
               "pet_dog_breed" => "labrador",
               "pet_dog_name" => "Frosty",
               "person_age" => nil,
               "person_hair_color" => "blonde",
               "person_name" => "Alex"
             },
             %{
               "pet_dog_breed" => nil,
               "pet_dog_name" => nil,
               "person_age" => nil,
               "person_hair_color" => "brunette",
               "person_name" => "Claudia"
             }
           ]
  end
end
