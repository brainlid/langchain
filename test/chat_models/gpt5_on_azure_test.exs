defmodule LangChain.ChatModels.GPT5OnAzureTest do
  use LangChain.BaseCase
  alias LangChain.ChatModels.ChatOpenAI
  alias LangChain.Message
  alias LangChain.Message.ContentPart
  alias LangChain.MessageDelta
  alias LangChain.Function
  alias LangChain.FunctionParam

  @moduletag :live_call
  @moduletag :live_azure

  defp azure_endpoint! do
    System.fetch_env!("AZURE_OPENAI_ENDPOINT")
  end

  defp azure_key! do
    System.fetch_env!("AZURE_OPENAI_KEY")
  end

  describe "Azure GPT-5 - simple chat" do
    @tag live_call: true, live_azure: true
    test "returns a basic response" do
      {:ok, chat} =
        ChatOpenAI.new(%{
          endpoint: azure_endpoint!(),
          api_key: azure_key!(),
          # model is ignored by Azure endpoint path, but used for GPT-5 request shaping
          model: "gpt-5",
          seed: 0,
          temperature: 0,
          stream: false
        })

      {:ok, [message]} =
        ChatOpenAI.call(
          chat,
          [Message.new_user!("Return exactly: Hi")],
          []
        )

      assert message.role == :assistant
      assert is_list(message.content)
      assert ContentPart.parts_to_string(message.content) =~ "Hi"
    end
  end

  describe "Azure GPT-5 - streaming" do
    @tag live_call: true, live_azure: true
    test "streams message deltas" do
      {:ok, chat} =
        ChatOpenAI.new(%{
          endpoint: azure_endpoint!(),
          api_key: azure_key!(),
          model: "gpt-5",
          temperature: 0,
          stream: true,
          stream_options: %{include_usage: true}
        })

      {:ok, data} =
        ChatOpenAI.call(
          chat,
          [Message.new_user!("Say 'streaming works'")],
          []
        )

      # Expect a list of MessageDelta lists
      assert is_list(data)
      flattened = data |> List.flatten()
      assert Enum.any?(flattened, fn d -> match?(%MessageDelta{}, d) end)
    end
  end

  describe "Azure GPT-5 - multimodal image" do
    @tag live_call: true, live_azure: true
    test "accepts an image url with prompt" do
      {:ok, chat} =
        ChatOpenAI.new(%{
          endpoint: azure_endpoint!(),
          api_key: azure_key!(),
          model: "gpt-5",
          temperature: 0,
          stream: false
        })

      img_url =
        "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg"

      message =
        Message.new_user!([
          ContentPart.text!("Identify what this is a picture of:"),
          ContentPart.image_url!(img_url)
        ])

      {:ok, [response]} = ChatOpenAI.call(chat, [message], [])

      assert response.role == :assistant
      assert is_list(response.content)
      # don't assert on specific words to avoid flakiness
      assert ContentPart.parts_to_string(response.content) != nil
    end
  end

  describe "Azure GPT-5 - tools formatting (sanity)" do
    test "flattens function tools in request" do
      {:ok, chat} =
        ChatOpenAI.new(%{
          endpoint:
            "https://example.invalid/openai/deployments/gpt-5/responses?api-version=2025-01-01-preview",
          api_key: "test",
          model: "gpt-5",
          temperature: 0,
          stream: false
        })

      {:ok, test_function} =
        Function.new(%{
          name: "get_weather",
          description: "Get weather",
          parameters: [
            FunctionParam.new!(%{name: "city", type: "string", required: true}),
            FunctionParam.new!(%{name: "state", type: "string", required: true})
          ],
          function: fn _args, _ctx -> {:ok, "Sunny"} end
        })

      req = ChatOpenAI.for_api(chat, [Message.new_user!("Hello")], [test_function])

      assert Map.has_key?(req, :input)
      assert is_list(req.tools)
      tool = hd(req.tools)
      assert tool["type"] == "function"
      assert tool["name"] == "get_weather"
      assert Map.has_key?(tool, "parameters")
      refute Map.has_key?(tool, "function")
    end
  end
end
