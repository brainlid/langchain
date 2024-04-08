defmodule LangChain.Message.MessagePartTest do
  use ExUnit.Case
  doctest LangChain.Message.MessagePart
  alias LangChain.Message.MessagePart
  alias LangChain.LangChainError

  describe "new/1" do
    test "accepts valid settings" do
      {:ok, %MessagePart{} = part} = MessagePart.new(%{type: :text, content: "Hello!"})
      assert part.type == :text
      assert part.content == "Hello!"
    end

    test "returns error when invalid" do
      assert {:error, changeset} = MessagePart.new(%{"type" => nil, "content" => nil})
      refute changeset.valid?
      assert {"can't be blank", _} = changeset.errors[:type]
      assert {"can't be blank", _} = changeset.errors[:content]
    end
  end

  describe "new!/1" do
    test "accepts valid settings" do
      %MessagePart{} = part = MessagePart.new!(%{type: :text, content: "Hello!"})
      assert part.type == :text
      assert part.content == "Hello!"
    end

    test "returns error when invalid" do
      assert_raise LangChain.LangChainError, "content: can't be blank", fn ->
        MessagePart.text!(nil)
      end
    end
  end

  describe("text!/1") do
    test "returns a text configured MessagePart" do
      %MessagePart{} = part = MessagePart.text!("Hello!")
      assert part.type == :text
      assert part.content == "Hello!"
    end
  end

  describe("image!/1") do
    test "returns an image data configured MessagePart" do
      %MessagePart{} = part = MessagePart.image!(:base64.encode("fake_image_data"))
      assert part.type == :image
      assert part.content == "ZmFrZV9pbWFnZV9kYXRh"
    end
  end

  describe("image_url!/1") do
    test "returns a image_url configured MessagePart" do
      url =
        "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg"

      %MessagePart{} = part = MessagePart.image_url!(url)
      assert part.type == :image_url
      assert part.content == url
    end
  end

  describe "tool_call/1" do
    test "returns MessagePart configured for tool_call" do
      assert {:ok, %MessagePart{} = part} =
               MessagePart.tool_call(%{
                 tool_name: "greeting",
                 tool_type: :function,
                 tool_arguments: Jason.encode!(%{name: "Tom"})
               })

      assert part.type == :tool_call
      assert part.tool_type == :function
      assert part.tool_name == "greeting"
      assert part.tool_arguments == %{"name" => "Tom"}
      assert part.options == nil
    end

    test "adds error when JSON is invalid" do
      {:error, changeset} =
        MessagePart.tool_call(%{
          tool_name: "greeting",
          tool_type: :function,
          tool_arguments: "{\"invalid\"}"
        })

      assert {"invalid json", _} = changeset.errors[:tool_arguments]
    end

    test "returns error when required values missing" do
      {:error, changeset} = MessagePart.tool_call(%{tool_name: nil})
      assert {"can't be blank", _} = changeset.errors[:tool_name]
    end
  end

  describe "tool_call!/1" do
    test "returns valid MessagePart" do
      %MessagePart{} =
        part =
        MessagePart.tool_call!(%{
          tool_name: "greeting",
          tool_type: :function,
          tool_arguments: Jason.encode!(%{name: "Tom"})
        })

      assert part.type == :tool_call
      assert part.tool_type == :function
      assert part.tool_name == "greeting"
      assert part.tool_arguments == %{"name" => "Tom"}
      assert part.options == nil
    end

    test "raises error when invalid" do
      assert_raise LangChainError, "tool_name: can't be blank", fn ->
        MessagePart.tool_call!(%{tool_name: nil})
      end
    end
  end
end
