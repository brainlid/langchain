defmodule LangChain.Utils.BedrockConfigTest do
  alias LangChain.Utils.BedrockConfig
  use ExUnit.Case, async: true

  test "supports aws credentials without session token" do
    bedrock_config = %BedrockConfig{credentials: fn -> {"KEY", "SECRET"} end, region: "us-east-1"}

    assert BedrockConfig.aws_sigv4_opts(bedrock_config) == [
             access_key_id: "KEY",
             secret_access_key: "SECRET",
             region: "us-east-1",
             service: :bedrock
           ]
  end

  test "supports aws credentials with session token" do
    bedrock_config = %BedrockConfig{
      credentials: fn -> {"KEY", "SECRET", "TOKEN"} end,
      region: "ap-southeast-2"
    }

    assert BedrockConfig.aws_sigv4_opts(bedrock_config) == [
             access_key_id: "KEY",
             secret_access_key: "SECRET",
             token: "TOKEN",
             region: "ap-southeast-2",
             service: :bedrock
           ]
  end
end
