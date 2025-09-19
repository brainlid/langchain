defmodule LangChain.ChatModels.ChatOpenAIResponsesTest do
  use LangChain.BaseCase

  doctest LangChain.ChatModels.ChatOpenAIResponses
  alias LangChain.ChatModels.ChatOpenAIResponses
  alias LangChain.Function
  alias LangChain.FunctionParam

  @test_model "gpt-4o-mini-2024-07-18"

  setup do
    {:ok, hello_world} =
      Function.new(%{
        name: "hello_world",
        description: "Give a hello world greeting",
        function: fn _args, _context -> {:ok, "Hello world!"} end
      })

    {:ok, weather} =
      Function.new(%{
        name: "get_weather",
        description: "Get the current weather in a given US location",
        parameters: [
          FunctionParam.new!(%{
            name: "city",
            type: "string",
            description: "The city name, e.g. San Francisco",
            required: true
          }),
          FunctionParam.new!(%{
            name: "state",
            type: "string",
            description: "The 2 letter US state abbreviation, e.g. CA, NY, UT",
            required: true
          })
        ],
        function: fn _args, _context -> {:ok, "75 degrees"} end
      })

    %{hello_world: hello_world, weather: weather}
  end

  describe "new/1" do
    test "works with minimal attr" do
      assert {:ok, %ChatOpenAIResponses{} = openai} =
               ChatOpenAIResponses.new(%{"model" => @test_model})

      assert openai.model == @test_model
    end

    test "returns error when invalid" do
      assert {:error, changeset} = ChatOpenAIResponses.new(%{"model" => nil})
      refute changeset.valid?
      assert {"can't be blank", _} = changeset.errors[:model]
    end

    test "supports overriding the API endpoint" do
      override_url = "http://localhost:1234/v1/chat/completions"

      model =
        ChatOpenAIResponses.new!(%{
          endpoint: override_url
        })

      assert model.endpoint == override_url
    end

    test "supports setting json_response and json_schema" do
      json_schema = %{
        "type" => "object",
        "properties" => %{
          "name" => %{"type" => "string"},
          "age" => %{"type" => "integer"}
        }
      }

      {:ok, openai} =
        ChatOpenAIResponses.new(%{
          "model" => @test_model,
          "json_response" => true,
          "json_schema" => json_schema
        })

      assert openai.json_response == true
      assert openai.json_schema == json_schema
    end

    test "supports overriding temperature" do
      {:ok, openai} = ChatOpenAIResponses.new(%{"model" => @test_model, "temperature" => 0.7})
      assert openai.temperature == 0.7
    end

    test "returns error for out-of-bounds temperature" do
      assert {:error, changeset} =
               ChatOpenAIResponses.new(%{"model" => @test_model, "temperature" => 2.5})

      refute changeset.valid?
      assert {"must be less than or equal to %{number}", _} = changeset.errors[:temperature]

      assert {:error, changeset} =
               ChatOpenAIResponses.new(%{"model" => @test_model, "temperature" => -0.1})

      refute changeset.valid?
      assert {"must be greater than or equal to %{number}", _} = changeset.errors[:temperature]
    end

    test "supports setting reasoning options" do
      {:ok, openai} =
        ChatOpenAIResponses.new(%{
          "model" => @test_model,
          "reasoning" => %{
            "effort" => "high"
          }
        })

      assert openai.reasoning.effort == :high
    end

    test "validates reasoning_effort values" do
      assert {:error, changeset} =
               ChatOpenAIResponses.new(%{
                 "model" => @test_model,
                 "reasoning" => %{"effort" => "invalid"}
               })

      refute changeset.valid?
      assert changeset.errors == []
      assert changeset.changes.reasoning.errors[:effort] != nil
    end

    test "supports setting reasoning_summary" do
      {:ok, openai} =
        ChatOpenAIResponses.new(%{
          "model" => @test_model,
          "reasoning" => %{
            "summary" => "detailed"
          }
        })

      assert openai.reasoning.summary == :detailed
    end

    test "validates reasoning_summary values" do
      assert {:error, changeset} =
               ChatOpenAIResponses.new(%{
                 "model" => @test_model,
                 "reasoning" => %{"summary" => "invalid"}
               })

      refute changeset.valid?
      assert changeset.errors == []
      assert changeset.changes.reasoning.errors[:summary] != nil
    end

    test "supports setting reasoning_generate_summary (deprecated)" do
      {:ok, openai} =
        ChatOpenAIResponses.new(%{
          "model" => @test_model,
          "reasoning" => %{
            "generate_summary" => "concise"
          }
        })

      assert openai.reasoning.generate_summary == :concise
    end

    test "validates reasoning_generate_summary values" do
      assert {:error, changeset} =
               ChatOpenAIResponses.new(%{
                 "model" => @test_model,
                 "reasoning" => %{"generate_summary" => "invalid"}
               })

      refute changeset.valid?
      assert changeset.errors == []
      assert changeset.changes.reasoning.errors[:generate_summary] != nil
    end

    test "accepts all valid reasoning_effort values" do
      valid_efforts = ["minimal", "low", "medium", "high"]

      for effort <- valid_efforts do
        assert {:ok, %ChatOpenAIResponses{reasoning: reasoning}} =
                 ChatOpenAIResponses.new(%{
                   "model" => @test_model,
                   "reasoning" => %{"effort" => effort}
                 })

        assert reasoning.effort == String.to_atom(effort)
      end
    end

    test "accepts all valid reasoning summary values" do
      valid_summaries = ["auto", "concise", "detailed"]

      for summary <- valid_summaries do
        assert {:ok, %ChatOpenAIResponses{reasoning: reasoning}} =
                 ChatOpenAIResponses.new(%{
                   "model" => @test_model,
                   "reasoning" => %{"summary" => summary}
                 })

        assert reasoning.summary == String.to_atom(summary)

        assert {:ok, %ChatOpenAIResponses{reasoning: reasoning}} =
                 ChatOpenAIResponses.new(%{
                   "model" => @test_model,
                   "reasoning" => %{"generate_summary" => summary}
                 })

        assert reasoning.generate_summary == String.to_atom(summary)
      end
    end

    # Support
  end

  describe "for_api/3 reasoning options" do
    test "includes reasoning options when set" do
      openai =
        ChatOpenAIResponses.new!(%{
          "model" => @test_model,
          "reasoning" => %{
            "effort" => "high",
            "summary" => "detailed"
          }
        })

      result = ChatOpenAIResponses.for_api(openai, [], [])

      assert result.reasoning == %{
               "effort" => "high",
               "summary" => "detailed"
             }
    end

    test "excludes reasoning when no options are set" do
      openai = ChatOpenAIResponses.new!(%{"model" => @test_model})

      result = ChatOpenAIResponses.for_api(openai, [], [])

      refute Map.has_key?(result, :reasoning)
    end

    test "includes only set reasoning options" do
      openai =
        ChatOpenAIResponses.new!(%{
          "model" => @test_model,
          "reasoning" => %{
            "effort" => "medium"
          }
        })

      result = ChatOpenAIResponses.for_api(openai, [], [])

      assert result.reasoning == %{"effort" => "medium"}
    end

    test "includes deprecated reasoning_generate_summary" do
      openai =
        ChatOpenAIResponses.new!(%{
          "model" => @test_model,
          "reasoning" => %{
            "generate_summary" => "auto"
          }
        })

      result = ChatOpenAIResponses.for_api(openai, [], [])

      assert result.reasoning == %{"generate_summary" => "auto"}
    end

    test "includes all reasoning options when all are set" do
      openai =
        ChatOpenAIResponses.new!(%{
          "model" => @test_model,
          "reasoning" => %{
            "effort" => "low",
            "summary" => "concise",
            "generate_summary" => "auto"
          }
        })

      result = ChatOpenAIResponses.for_api(openai, [], [])

      assert result.reasoning == %{
               "effort" => "low",
               "summary" => "concise",
               "generate_summary" => "auto"
             }
    end
  end
end
