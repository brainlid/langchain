defmodule LangChain.ChatModels.ChatGrokTest do
  use LangChain.BaseCase

  doctest LangChain.ChatModels.ChatGrok
  alias LangChain.ChatModels.ChatGrok
  alias LangChain.Function
  alias LangChain.FunctionParam
  alias LangChain.Message
  alias LangChain.Message.ContentPart
  alias LangChain.Message.ToolCall
  alias LangChain.Message.ToolResult

  @test_model "grok-4"

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
        function: &get_weather/2
      })

    %{hello_world: hello_world, weather: weather}
  end

  defp get_weather(%{"city" => city, "state" => state}, _context) do
    {:ok, "It's 70F and sunny in #{city}, #{state}."}
  end

  describe "new/1" do
    test "works with minimal configuration" do
      assert {:ok, %ChatGrok{} = grok} = ChatGrok.new()
      assert grok.model == "grok-4"
      assert grok.endpoint == "https://api.x.ai/v1/chat/completions"
      assert grok.temperature == 0.7
      assert grok.max_tokens == 4096
      assert grok.stream == false
    end

    test "supports custom configuration" do
      assert {:ok, %ChatGrok{} = grok} =
               ChatGrok.new(%{
                 model: "grok-4",
                 temperature: 0.3,
                 max_tokens: 8192,
                 stream: true,
                 top_p: 0.9,
                 frequency_penalty: 0.1,
                 presence_penalty: 0.1
               })

      assert grok.model == "grok-4"
      assert grok.temperature == 0.3
      assert grok.max_tokens == 8192
      assert grok.stream == true
      assert grok.top_p == 0.9
      assert grok.frequency_penalty == 0.1
      assert grok.presence_penalty == 0.1
    end

    test "validates temperature range" do
      assert {:error, changeset} = ChatGrok.new(%{temperature: -1})
      assert {"must be greater than or equal to %{number}", _} = changeset.errors[:temperature]

      assert {:error, changeset} = ChatGrok.new(%{temperature: 3})
      assert {"must be less than or equal to %{number}", _} = changeset.errors[:temperature]
    end

    test "validates penalty ranges" do
      assert {:error, changeset} = ChatGrok.new(%{frequency_penalty: -3})

      assert {"must be greater than or equal to %{number}", _} =
               changeset.errors[:frequency_penalty]

      assert {:error, changeset} = ChatGrok.new(%{presence_penalty: 3})
      assert {"must be less than or equal to %{number}", _} = changeset.errors[:presence_penalty]
    end
  end

  describe "new!/1" do
    test "succeeds with valid configuration" do
      assert %ChatGrok{} = ChatGrok.new!(%{model: "grok-4"})
    end
  end

  describe "for_api/3" do
    setup do
      {:ok, grok} = ChatGrok.new(%{model: "grok-4", temperature: 0.5})
      messages = [Message.new_user!("Hello")]
      %{grok: grok, messages: messages}
    end

    test "formats basic request correctly", %{grok: grok, messages: messages} do
      result = ChatGrok.for_api(grok, messages)

      assert result.model == "grok-4"
      assert result.stream == false
      assert result.temperature == 0.5
      assert length(result.messages) == 1
      assert hd(result.messages).role == "user"
    end

    test "includes tools when provided", %{grok: grok, messages: messages, hello_world: tool} do
      result = ChatGrok.for_api(grok, messages, [tool])

      assert is_list(result.tools)
      assert length(result.tools) == 1

      tool_data = hd(result.tools)
      assert tool_data.type == "function"
      assert tool_data.function.name == "hello_world"
    end

    test "excludes nil optional fields", %{grok: grok, messages: messages} do
      result = ChatGrok.for_api(grok, messages)

      refute Map.has_key?(result, :top_p)
      refute Map.has_key?(result, :frequency_penalty)
      refute Map.has_key?(result, :tools)
    end
  end

  describe "for_api_message/1" do
    test "formats system message" do
      message = Message.new_system!("You are a helpful assistant")
      result = ChatGrok.for_api_message(message)

      assert result.role == "system"
      assert result.content == "You are a helpful assistant"
    end

    test "formats user message" do
      message = Message.new_user!("Hello")
      result = ChatGrok.for_api_message(message)

      assert result.role == "user"
      assert result.content == "Hello"
    end

    test "formats assistant message with content" do
      message = Message.new_assistant!("Hello there!")
      result = ChatGrok.for_api_message(message)

      assert result.role == "assistant"
      assert Map.get(result, :content) == "Hello there!"
    end

    test "formats assistant message with tool calls" do
      tool_call =
        ToolCall.new!(%{call_id: "123", name: "get_weather", arguments: "{\"city\": \"SF\"}"})

      message = %Message{role: :assistant, content: "", tool_calls: [tool_call]}
      result = ChatGrok.for_api_message(message)

      assert result.role == "assistant"
      assert is_list(result.tool_calls)
      assert length(result.tool_calls) == 1

      call = hd(result.tool_calls)
      assert call.id == "123"
      assert call.type == "function"
      assert call.function.name == "get_weather"
    end

    test "formats tool result message" do
      tool_result = ToolResult.new!(%{tool_call_id: "123", content: "Weather is sunny"})
      message = Message.new_tool_result!(%{tool_results: [tool_result]})
      result = ChatGrok.for_api_message(message)

      assert is_list(result)
      assert length(result) == 1

      tool_msg = hd(result)
      assert tool_msg.role == "tool"
      assert tool_msg.content == "Weather is sunny"
      assert tool_msg.tool_call_id == "123"
    end

    test "formats multimodal user message" do
      content = [
        ContentPart.text!("What's in this image?"),
        ContentPart.image_url!("https://example.com/image.jpg")
      ]

      message = Message.new_user!(content)
      result = ChatGrok.for_api_message(message)

      assert result.role == "user"
      assert is_list(result.content)
      assert length(result.content) == 2

      [text_part, image_part] = result.content
      assert text_part.type == "text"
      assert image_part.type == "image_url"
    end
  end

  describe "serialize_config/1 and restore_from_map/1" do
    test "can serialize and restore configuration" do
      original =
        ChatGrok.new!(%{
          model: "grok-4",
          temperature: 0.3,
          max_tokens: 8192
        })

      serialized = ChatGrok.serialize_config(original)
      assert is_map(serialized)
      assert serialized.model == "grok-4"
      assert serialized.module == ChatGrok

      {:ok, restored} = ChatGrok.restore_from_map(serialized)
      assert restored.model == original.model
      assert restored.temperature == original.temperature
      assert restored.max_tokens == original.max_tokens
    end

    test "handles invalid restore data" do
      assert {:error, _reason} = ChatGrok.restore_from_map(%{invalid: "data"})
    end
  end

  describe "live API integration" do
    @tag live_call: true, live_grok: true
    test "basic chat completion" do
      {:ok, grok} = ChatGrok.new(%{model: @test_model})
      {:ok, [message]} = ChatGrok.call(grok, "Say hello in one word")

      assert %Message{role: :assistant} = message
      assert is_binary(message.content)
      assert String.length(message.content) > 0
    end

    @tag live_call: true, live_grok: true
    test "streaming chat completion" do
      {:ok, grok} =
        ChatGrok.new(%{
          model: @test_model,
          stream: true,
          stream_options: %{include_usage: true}
        })

      {:ok, deltas} = ChatGrok.call(grok, "Count from 1 to 3")

      assert is_list(deltas)
      assert length(deltas) > 0
      assert Enum.all?(deltas, fn delta -> match?(%LangChain.MessageDelta{}, delta) end)
    end

    @tag live_call: true, live_grok: true
    test "function calling" do
      {:ok, grok} = ChatGrok.new(%{model: @test_model})

      {:ok, weather} =
        Function.new(%{
          name: "get_weather",
          description: "Get the current weather",
          parameters_schema: %{
            type: "object",
            properties: %{
              city: %{type: "string", description: "The city name"}
            },
            required: ["city"]
          },
          function: fn %{"city" => city}, _context ->
            {:ok, "It's sunny in #{city}"}
          end
        })

      messages = [Message.new_user!("What's the weather in Paris?")]
      {:ok, [response]} = ChatGrok.call(grok, messages, [weather])

      assert %Message{role: :assistant} = response
      assert is_list(response.tool_calls) and length(response.tool_calls) > 0
    end

    @tag live_call: true, live_grok: true
    test "multimodal support with image URL" do
      {:ok, grok} = ChatGrok.new(%{model: @test_model})

      content = [
        ContentPart.text!("What's in this image?"),
        ContentPart.image_url!(
          "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg"
        )
      ]

      messages = [Message.new_user!(content)]
      {:ok, [response]} = ChatGrok.call(grok, messages)

      assert %Message{role: :assistant} = response
      assert is_binary(response.content)
      assert String.length(response.content) > 0
    end

    @tag live_call: true, live_grok: true
    test "grok-4 model advanced features" do
      {:ok, grok} = ChatGrok.new(%{model: "grok-4", temperature: 0.3})
      {:ok, [message]} = ChatGrok.call(grok, "Explain quantum computing briefly")

      assert %Message{role: :assistant} = message
      assert is_binary(message.content)
      assert String.length(message.content) > 50
    end

    @tag live_call: true, live_grok: true
    test "grok-3-mini model works" do
      {:ok, grok} = ChatGrok.new(%{model: "grok-3-mini", temperature: 0.7})
      {:ok, [message]} = ChatGrok.call(grok, "Say hello in a creative way!")

      assert %Message{role: :assistant} = message
      assert is_binary(message.content)
      assert String.length(message.content) > 0
    end

    @tag live_call: true, live_grok: true
    test "large context window handling" do
      {:ok, grok} = ChatGrok.new(%{model: @test_model, max_tokens: 2048})

      # Create a longer conversation to test context handling
      long_prompt = String.duplicate("This is a test sentence. ", 100)
      {:ok, [message]} = ChatGrok.call(grok, "Summarize this: #{long_prompt}")

      assert %Message{role: :assistant} = message
      assert is_binary(message.content)
    end

    @tag live_call: true, live_grok: true
    test "grok-4-0709 model works" do
      {:ok, grok} = ChatGrok.new(%{model: "grok-4-0709", temperature: 0.3})
      {:ok, [message]} = ChatGrok.call(grok, "Say hello briefly")
      assert %Message{role: :assistant} = message
      assert is_binary(message.content)
      assert String.length(message.content) > 0
    end

    @tag live_call: true, live_grok: true
    test "grok-3 model works" do
      {:ok, grok} = ChatGrok.new(%{model: "grok-3", temperature: 0.5})
      {:ok, [message]} = ChatGrok.call(grok, "Count to 3")

      assert %Message{role: :assistant} = message
      assert is_binary(message.content)
      assert String.length(message.content) > 0
    end

    @tag live_call: true, live_grok: true
    test "grok-3-fast model works" do
      {:ok, grok} = ChatGrok.new(%{model: "grok-3-fast", temperature: 0.7})
      {:ok, [message]} = ChatGrok.call(grok, "What is 2+2?")

      assert %Message{role: :assistant} = message
      assert is_binary(message.content)
      assert String.length(message.content) > 0
    end

    @tag live_call: true, live_grok: true
    test "grok-3-mini-fast model works" do
      {:ok, grok} = ChatGrok.new(%{model: "grok-3-mini-fast", temperature: 0.8})
      {:ok, [message]} = ChatGrok.call(grok, "Say yes or no")

      assert %Message{role: :assistant} = message
      assert is_binary(message.content)
      assert String.length(message.content) > 0
    end

    @tag live_call: true, live_grok: true
    test "grok-2-vision-1212us-east-1 model with image" do
      {:ok, grok} = ChatGrok.new(%{model: "grok-2-vision-1212us-east-1"})

      content = [
        ContentPart.text!("Describe this image briefly"),
        ContentPart.image_url!(
          "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg"
        )
      ]

      messages = [Message.new_user!(content)]
      {:ok, [response]} = ChatGrok.call(grok, messages)

      assert %Message{role: :assistant} = response
      assert is_binary(response.content)
      assert String.length(response.content) > 0
    end

    @tag live_call: true, live_grok: true
    test "grok-2-vision-1212eu-west-1 model with image" do
      {:ok, grok} = ChatGrok.new(%{model: "grok-2-vision-1212eu-west-1"})

      content = [
        ContentPart.text!("What do you see?"),
        ContentPart.image_url!(
          "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg"
        )
      ]

      messages = [Message.new_user!(content)]
      {:ok, [response]} = ChatGrok.call(grok, messages)

      assert %Message{role: :assistant} = response
      assert is_binary(response.content)
      assert String.length(response.content) > 0
    end

    @tag live_call: true, live_grok: true
    test "grok-2-image-1212 model works" do
      {:ok, grok} = ChatGrok.new(%{model: "grok-2-image-1212", temperature: 0.5})
      {:ok, [message]} = ChatGrok.call(grok, "Generate a brief description of a cat")

      assert %Message{role: :assistant} = message
      assert is_binary(message.content)
      assert String.length(message.content) > 0
    end

    @tag live_call: true, live_grok: true
    test "grok-4-heavy model with multi-agent features" do
      {:ok, grok} =
        ChatGrok.new(%{
          model: "grok-4-heavy",
          multi_agent: true,
          temperature: 0.3
        })

      {:ok, [message]} = ChatGrok.call(grok, "Solve: What is the capital of France?")

      assert %Message{role: :assistant} = message
      assert is_binary(message.content)
      assert String.length(message.content) > 0
    end
  end
end
