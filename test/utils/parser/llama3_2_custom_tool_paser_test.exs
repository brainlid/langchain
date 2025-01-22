defmodule Langchain.Utils.Parser.LLAMA_3_2_CustomToolParserTest do
  use ExUnit.Case
  alias Langchain.Utils.Parser.LLAMA_3_2_CustomToolParser

  describe "parse/1" do
    test "successfully parses valid function call" do
      input = "[get_user_info(user_id=7890, special='black')]"

      expected = {
        :ok,
        [
          %{
            function_name: "get_user_info",
            parameters: %{
              "user_id" => 7890,
              "special" => "black"
            }
          }
        ]
      }

      assert LLAMA_3_2_CustomToolParser.parse(input) == expected
    end

    test "successfully parses valid twp function calls" do
      input =
        "[get_user_info(user_id=7890, special='black'),get_user_info(user_id=7891, special='blue')]"

      expected = {
        :ok,
        [
          %{
            parameters: %{
              "special" => "black",
              "user_id" => 7890
            },
            function_name: "get_user_info"
          },
          %{
            parameters: %{
              "special" => "blue",
              "user_id" => 7891
            },
            function_name: "get_user_info"
          }
        ]
      }

      assert LLAMA_3_2_CustomToolParser.parse(input) == expected
    end

    test "successfully parses valid function call hairbrush" do
      input = "[get_location(thing=\"hairbrush\")]"

      expected = {
        :ok,
        [
          %{
            function_name: "get_location",
            parameters: %{
              "thing" => "hairbrush"
            }
          }
        ]
      }

      assert LLAMA_3_2_CustomToolParser.parse(input) == expected
    end

    test "handles missing parameters" do
      input = "[get_users()]"

      assert {:ok, [%{function_name: "get_users", parameters: %{}}]} =
               LLAMA_3_2_CustomToolParser.parse(input)
    end

    #
    # test "handles invalid input" do
    #  input = "[invalid_function]<|eot_id|>"
    #  assert {:error, _} = LLAMA_3_2_CustomToolParser.parse(input)
    # end
  end
end
