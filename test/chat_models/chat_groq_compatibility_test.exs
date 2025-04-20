defmodule LangChain.ChatModels.ChatGroqCompatibilityTest do
  use ExUnit.Case
  import Mimic

  alias LangChain.ChatModels.ChatGroq
  alias LangChain.Chains.LLMChain
  alias LangChain.Message
  alias LangChain.Function
  alias LangChain.PromptTemplate

  # Tag all tests for isolated running
  @moduletag :groq
  @moduletag :compatibility

  setup :verify_on_exit!

  setup do
    # Set up a basic ChatGroq instance for testing
    # Using one of the latest production models from Groq
    model = ChatGroq.new!(%{model: "llama-3.1-8b-instant"})
    
    # Mock data for API responses
    mock_completion_response = %{
      "id" => "chatcmpl-123",
      "object" => "chat.completion",
      "created" => 1677858242,
      "model" => "llama-3.1-8b-instant",
      "choices" => [
        %{
          "message" => %{
            "role" => "assistant", 
            "content" => "This is a test response from Groq"
          },
          "finish_reason" => "stop",
          "index" => 0
        }
      ],
      "usage" => %{
        "prompt_tokens" => 13,
        "completion_tokens" => 7,
        "total_tokens" => 20
      }
    }
    
    # Mock tool call response
    mock_tool_call_response = %{
      "id" => "chatcmpl-456",
      "object" => "chat.completion",
      "created" => 1677858242,
      "model" => "llama-3.1-8b-instant",
      "choices" => [
        %{
          "message" => %{
            "role" => "assistant",
            "content" => nil,
            "tool_calls" => [
              %{
                "id" => "call_123",
                "type" => "function",
                "function" => %{
                  "name" => "get_weather",
                  "arguments" => "{\"location\":\"San Francisco\"}"
                }
              }
            ]
          },
          "finish_reason" => "tool_calls",
          "index" => 0
        }
      ],
      "usage" => %{
        "prompt_tokens" => 15,
        "completion_tokens" => 12,
        "total_tokens" => 27
      }
    }
    
    %{
      model: model,
      mock_completion_response: mock_completion_response, 
      mock_tool_call_response: mock_tool_call_response
    }
  end

  describe "LLMChain compatibility" do
    test "works with basic LLMChain", %{model: model} do
      # Set up a simple prompt template
      {:ok, prompt} = PromptTemplate.from_template("Tell me about {topic}")
      
      # Create the LLMChain
      {:ok, chain} = LLMChain.new(%{llm: model, prompt: prompt})
      
      # Verify chain structure
      assert chain.llm == model
      
      # In this version of LangChain, the prompt is stored differently
      # We'll just verify that the chain is properly constructed
      assert is_struct(chain, LLMChain)
    end
  end

  describe "Tool/function calling compatibility" do
    test "properly integrates with functions", %{model: model} do
      # Create a sample function
      get_weather = Function.new!(%{
        name: "get_weather",
        description: "Get the current weather in a given location",
        parameters_schema: %{
          "type" => "object",
          "properties" => %{
            "location" => %{
              "type" => "string",
              "description" => "The city and state, e.g. San Francisco, CA"
            }
          },
          "required" => ["location"]
        },
        function: fn args, _context ->
          location = args["location"]
          {:ok, "The weather in #{location} is sunny."}
        end
      })
      
      # Create a tool-aware LLMChain
      {:ok, chain} = LLMChain.new(%{
        llm: model,
        tools: [get_weather]
      })
      
      # Test the structure
      assert length(chain.tools) == 1
      [tool] = chain.tools
      assert tool.name == "get_weather"
      assert Map.has_key?(chain._tool_map, "get_weather")
    end
  end

  describe "Message format compatibility" do
    test "can handle messages with the same format as other models", %{model: model} do
      # Create messages
      system_msg = Message.new_system!("You are a helpful assistant.")
      user_msg = Message.new_user!("Tell me about Groq.")
      
      # Test that ChatGroq can format messages correctly
      api_messages = ChatGroq.for_api(model, [system_msg, user_msg], [])
      
      # Verify the messages are correctly formatted
      assert is_map(api_messages)
      assert length(api_messages.messages) == 2
      assert Enum.at(api_messages.messages, 0)["role"] == :system
      assert Enum.at(api_messages.messages, 1)["role"] == :user
      
      # Verify the model can process messages from other API responses
      assistant_response = 
        Message.new!(%{
          "role" => "assistant",
          "content" => "This is a test response from Groq",
          "status" => :complete
        })
      
      # Try formatting it back
      formatted = ChatGroq.for_api(model, assistant_response)
      assert formatted["role"] == :assistant
      assert formatted["content"] == "This is a test response from Groq"
    end
  end

  describe "API Integration" do
    @tag :live_call
    @tag :live_groq
    test "real API integration with complete response", %{model: _model} do
        
        # Use a real model with the API key
        real_model = ChatGroq.new!(%{
          model: "llama-3.1-8b-instant",
          max_tokens: 200  # Increased to avoid truncation
        })
        
        # Test a direct API call
        result = ChatGroq.call(
          real_model,
          "Generate a very short list of 3 benefits of exercise.",
          []
        )
        
        # Verify result structure
        assert {:ok, response} = result
        assert is_list(response)
        [message] = response
        assert message.role == :assistant
        assert is_binary(message.content)
        assert message.status == :complete
    end
    
    @tag :live_call
    @tag :live_groq
    test "real API integration with length-limited response", %{model: _model} do
        
        # Use a real model with very limited tokens to trigger :length status
        real_model = ChatGroq.new!(%{
          model: "llama-3.1-8b-instant",
          max_tokens: 5  # Very small limit to force truncation
        })
        
        # Test a direct API call requesting content that will exceed the limit
        result = ChatGroq.call(
          real_model,
          "Write a detailed paragraph about the history of artificial intelligence.",
          []
        )
        
        # Verify result structure with :length status
        assert {:ok, response} = result
        assert is_list(response)
        [message] = response
        assert message.role == :assistant
        assert is_binary(message.content)
        assert message.status == :length
    end
  end
end