defmodule LangChain.BedrockHelpers do
  def bedrock_config() do
    %{
      credentials: fn ->
        {Application.fetch_env!(:langchain, :aws_access_key_id),
         Application.fetch_env!(:langchain, :aws_secret_access_key)}
      end,
      region: "us-east-1"
    }
  end

  def prefix_for(api) do
    "#{api} API:"
  end
end
