defmodule LangChain.ForOpenAIApiTest do
  use ExUnit.Case
  doctest LangChain.ForOpenAIApi
  alias LangChain.ForOpenAIApi
  alias LangChain.Message

  describe "for_api/1" do
    test "turns a function_call into expected JSON format" do
      msg = Message.new_function_call!("hello_world", "{}")

      json = ForOpenAIApi.for_api(msg)

      assert json == %{
               "content" => nil,
               "function_call" => %{"arguments" => "{}", "name" => "hello_world"},
               "role" => :assistant
             }
    end

    test "turns a function_call into expected JSON format with arguments" do
      args = %{"expression" => "11 + 10"}
      msg = Message.new_function_call!("hello_world", Jason.encode!(args))

      json = ForOpenAIApi.for_api(msg)

      assert json == %{
               "content" => nil,
               "function_call" => %{
                 "arguments" => "{\"expression\":\"11 + 10\"}",
                 "name" => "hello_world"
               },
               "role" => :assistant
             }
    end

    test "turns a function response into expected JSON format" do
      msg = Message.new_function!("hello_world", "Hello World!")

      json = ForOpenAIApi.for_api(msg)

      assert json == %{"content" => "Hello World!", "name" => "hello_world", "role" => :function}
    end
  end
end
