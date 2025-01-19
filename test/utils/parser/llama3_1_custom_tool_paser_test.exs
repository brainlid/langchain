defmodule Langchain.Utils.Parser.LLAMA_3_1_CustomToolParserTest do
  use ExUnit.Case
  alias Langchain.Utils.Parser.LLAMA_3_1_CustomToolParser

  describe "parse/1" do
    test "successfully parses valid function call" do
      input = ~s(<function=spotify_trending_songs>{"n": "5"}</function>)

      expected = {
        :ok,
        %{
          function_name: "spotify_trending_songs",
          parameters: %{"n" => "5"}
        }
      }

      assert LLAMA_3_1_CustomToolParser.parse(input) == expected
    end


    test "handles invalid JSON in parameters" do
      input = ~s(<function=test>{invalid_json}</function>)
      assert {:error, _} = LLAMA_3_1_CustomToolParser.parse(input)
    end

    test "handles invalid function name" do
      input = ~s(<function=></function>)
      assert {:error, _} = LLAMA_3_1_CustomToolParser.parse(input)
    end

    test "handles missing end tag" do
      input = ~s(<function=test>{"param": "value"})
      assert {:error, _} = LLAMA_3_1_CustomToolParser.parse(input)
    end

    test "handles missing start tag" do
      input = ~s(test>{"param": "value"}</function>)
      assert {:error, _} = LLAMA_3_1_CustomToolParser.parse(input)
    end
  end
end
