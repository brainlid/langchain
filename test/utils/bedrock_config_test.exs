defmodule LangChain.Utils.BedrockConfigTest do
  alias LangChain.Utils.BedrockConfig
  use ExUnit.Case, async: true

  test "supports aws credentials without session token" do
    Application
    |> stub(:fetch_env, fn :langchain, :aws_session_token -> :error end)
    |> stub(:fetch_env!, fn :langchain, :aws_session_token -> :error end)
    |> stub(:get_env, fn :langchain, :aws_region -> "us-east-1" end)
    |> stub(:get_env, fn :langchain, :aws_session_token -> :error end)

    bedrock_config = BedrockConfig.from_application_env!()

    assert BedrockConfig.aws_sigv4_opts(bedrock_config) == [
             access_key_id: "KEY",
             secret_access_key: "SECRET",
             region: "us-east-1",
             service: :bedrock
           ]
  end

  test "supports aws credentials with session token" do
    Application
    |> stub(:fetch_env, fn :langchain, :aws_session_token -> "TOKEN" end)
    |> stub(:fetch_env!, fn :langchain, :aws_session_token -> "TOKEN" end)
    |> stub(:get_env, fn :langchain, :aws_region -> "ap-southeast-2" end)
    |> stub(:get_env, fn :langchain, :aws_session_token -> "TOKEN" end)

    bedrock_config = BedrockConfig.from_application_env!()

    assert BedrockConfig.aws_sigv4_opts(bedrock_config) == [
             access_key_id: "KEY",
             secret_access_key: "SECRET",
             token: "TOKEN",
             region: "ap-southeast-2",
             service: :bedrock
           ]
  end
end
