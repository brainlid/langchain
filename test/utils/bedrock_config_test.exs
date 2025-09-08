defmodule LangChain.Utils.BedrockConfigTest do
  alias Ecto.Changeset
  alias LangChain.Utils.BedrockConfig
  use ExUnit.Case, async: true
  use Mimic

  test "supports aws credentials without session token" do
    Application
    |> stub(:fetch_env!, fn app, key ->
      case {app, key} do
        {:langchain, :aws_access_key_id} -> "KEY"
        {:langchain, :aws_secret_access_key} -> "SECRET"
        {:langchain, :aws_region} -> "us-east-1"
      end
    end)
    |> stub(:fetch_env, fn :langchain, :aws_session_token -> :error end)

    bedrock_config = bedrock_config()

    assert BedrockConfig.aws_sigv4_opts(bedrock_config) == [
             access_key_id: "KEY",
             secret_access_key: "SECRET",
             region: "us-east-1",
             service: :bedrock
           ]
  end

  test "supports aws credentials with session token" do
    Application
    |> stub(:fetch_env!, fn app, key ->
      case {app, key} do
        {:langchain, :aws_access_key_id} -> "KEY"
        {:langchain, :aws_secret_access_key} -> "SECRET"
        {:langchain, :aws_region} -> "ap-southeast-2"
      end
    end)
    |> stub(:fetch_env, fn :langchain, :aws_session_token -> {:ok, "TOKEN"} end)

    bedrock_config = bedrock_config()

    assert BedrockConfig.aws_sigv4_opts(bedrock_config) == [
             token: "TOKEN",
             access_key_id: "KEY",
             secret_access_key: "SECRET",
             region: "ap-southeast-2",
             service: :bedrock
           ]
  end

  defp bedrock_config() do
    %BedrockConfig{}
    |> BedrockConfig.changeset(BedrockConfig.from_application_env!())
    |> Changeset.apply_changes()
  end
end
