defmodule LangChain.ChatModels.ChatGroqModelsTest do
  use ExUnit.Case
  import Mimic

  alias LangChain.ChatModels.ChatGroq
  alias LangChain.Message

  # Tag all tests for isolated running
  @moduletag :groq
  @moduletag :models

  setup :verify_on_exit!

  # A simple consistent prompt for all model tests
  @test_prompt "Please respond with a short sentence."

  # Special prompt for safety models
  @safety_prompt "Is this harmful content: 'Hello, how are you?'"

  # Production models
  @production_models [
    "gemma2-9b-it",
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant",
    "llama-guard-3-8b",
    "llama3-70b-8192",
    "llama3-8b-8192"
  ]

  # Preview models (evaluation only)
  @preview_models [
    "meta-llama/llama-4-maverick-17b-128e-instruct",
    "meta-llama/llama-4-scout-17b-16e-instruct",
    "mistral-saba-24b",
    "deepseek-r1-distill-llama-70b",
    "qwen-qwq-32b"
  ]

  # Generate a test for each production model
  for model <- @production_models do
    model_name = model

    @tag :live_call
    @tag :live_groq
    test "can make basic call to #{model_name} model" do
      model = unquote(model_name)

      # Create model instance with minimal tokens to keep test fast
      groq = ChatGroq.new!(%{
        model: model,
        max_tokens: 50
      })

      # Use different prompt for safety models vs regular models
      prompt = if model == "llama-guard-3-8b", do: @safety_prompt, else: @test_prompt

      # Make a call to the model
      result = ChatGroq.call(
        groq,
        [Message.new_user!(prompt)],
        []
      )

      # Verify basic response structure
      assert {:ok, response} = result
      assert is_list(response)
      [message] = response
      assert message.role == :assistant
      assert is_binary(message.content)
      # Just verify we got some content back, without checking specific words
      assert String.length(message.content) > 0
    end
  end

  # Generate a test for each preview model
  for model <- @preview_models do
    model_name = model

    @tag :live_call
    @tag :live_groq
    @tag :preview_models
    test "can make basic call to preview model #{model_name}" do
      model = unquote(model_name)

      # Create model instance with minimal tokens to keep test fast
      groq = ChatGroq.new!(%{
        model: model,
        max_tokens: 50
      })

      # Make a call to the model
      result = ChatGroq.call(
        groq,
        [Message.new_user!(@test_prompt)],
        []
      )

      # Handle both successful calls and terms acceptance errors
      case result do
        {:ok, response} ->
          # Verify basic response structure for successful calls
          assert is_list(response)
          [message] = response
          assert message.role == :assistant
          assert is_binary(message.content)
          assert String.length(message.content) > 0

        {:error, %LangChain.LangChainError{message: error_message}} ->
          # Check if this is a terms acceptance error (common for preview models)
          if String.contains?(error_message, "requires terms acceptance") do
            # This is expected for preview models - test passes
            IO.puts("Note: Model #{model} requires terms acceptance in the Groq console")
          else
            # For other errors, fail the test
            flunk("Error calling model #{model}: #{error_message}")
          end
      end
    end
  end

  # Test model availability (can be run without :live_call tag)
  describe "model list validation" do
    test "production models are correctly documented" do
      # Just ensure we have some production models defined
      assert length(@production_models) > 0
    end

    @tag :live_call
    @tag :live_groq
    test "can retrieve available models from Groq API" do
      # This checks if we can access the Groq models API endpoint
      # It's a simpler way to verify API connectivity
      {:ok, client} = Req.new(
        base_url: "https://api.groq.com/openai/v1",
        auth: {:bearer, System.get_env("GROQ_API_KEY")},
        headers: [{"Content-Type", "application/json"}]
      )
      |> then(&{:ok, &1})

      # Fetch models list
      {:ok, response} = Req.get(client, url: "/models")

      assert response.status == 200

      # The response body is already parsed if using recent versions of Req
      body = if is_binary(response.body) do
        Jason.decode!(response.body)
      else
        # It's already decoded
        response.body
      end

      assert %{"data" => models} = body
      assert is_list(models)
      assert length(models) > 0

      # Verify structure of first model
      first_model = List.first(models)
      assert Map.has_key?(first_model, "id")
      assert Map.has_key?(first_model, "object")
    end
  end

end
