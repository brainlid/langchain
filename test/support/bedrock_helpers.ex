defmodule LangChain.BedrockHelpers do
  def bedrock_config() do
    %{
      credentials: fn ->
        [
          access_key_id: Application.fetch_env!(:langchain, :aws_access_key_id),
          secret_access_key: Application.fetch_env!(:langchain, :aws_secret_access_key)
        ]
        |> maybe_add_token()
      end,
      region: Application.fetch_env!(:langchain, :aws_region)
    }
  end

  def prefix_for(api) do
    "#{api} API:"
  end

  defp maybe_add_token(credentials) do
    case Application.fetch_env(:langchain, :aws_session_token) do
      {:ok, token} ->
        Keyword.put(credentials, :token, token)

      :error ->
        credentials
    end
  end
end
