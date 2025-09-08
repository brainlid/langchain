defmodule LangChain.Utils.BedrockConfig do
  @moduledoc """
  Configuration for AWS Bedrock.

  ## Examples

  For applications hosted in AWS, [ExAws](https://hex.pm/packages/ex_aws) caches
  temporary credentials (when running on AWS), so in the credentials function
  you can pull the current cached credentials from `ExAws`.

      ChatAnthropic.new!(%{
        model: "anthropic.claude-3-5-sonnet-20241022-v2:0",
        bedrock: %{
          credentials: fn ->
            ExAws.Config.new(:s3)
            |> Map.take([:access_key_id, :secret_access_key])
            |> Map.to_list()
          end,
          region: "us-west-2"
        }
      })

  For applications hosted anywhere, you can configure the Bedrock settings into
  the LangChain config like this (recommended for `config/runtime.exs`):

      config :langchain,
        aws_access_key_id: System.fetch_env!("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key: System.fetch_env!("AWS_SECRET_ACCESS_KEY"),
        aws_region: System.get_env("AWS_REGION", "us-west-1"),
        aws_session_token: System.get_env("AWS_SESSION_TOKEN") # optional

  Then, when you want to later use a Bedrock Anthropic model, this is will load
  it:

        ChatAnthropic.new!(%{
          model: "anthropic.claude-3-5-sonnet-20241022-v2:0",
          bedrock: BedrockConfig.from_application_env!()
        })

  """
  use Ecto.Schema
  import Ecto.Changeset
  alias __MODULE__

  @primary_key false
  embedded_schema do
    # A function that returns a keyword list including access_key_id, secret_access_key, and optionally token.
    # Used to configure Req's aws_sigv4 option.
    field :credentials, :any, virtual: true
    field :region, :string
    field :anthropic_version, :string, default: "bedrock-2023-05-31"
  end

  @type t :: %BedrockConfig{}

  @create_fields [:credentials, :region, :anthropic_version]
  @required_fields @create_fields

  def changeset(bedrock, attrs) do
    bedrock
    |> cast(attrs, @create_fields)
    |> validate_required(@required_fields)
  end

  def aws_sigv4_opts(%BedrockConfig{} = bedrock) do
    Keyword.merge(bedrock.credentials.(),
      region: bedrock.region,
      service: :bedrock
    )
  end

  def url(%BedrockConfig{region: region}, model: model, stream: stream) do
    "https://bedrock-runtime.#{region}.amazonaws.com/model/#{model}/#{action(stream: stream)}"
  end

  defp action(stream: true), do: "invoke-with-response-stream"
  defp action(stream: false), do: "invoke"

  @doc """
  Loads the Bedrock config settings from the previously configured Application settings.

  `config/runtime.exs`:

      config :langchain,
        aws_access_key_id: System.fetch_env!("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key: System.fetch_env!("AWS_SECRET_ACCESS_KEY"),
        aws_region: System.get_env("AWS_REGION", "us-west-1"),
        aws_session_token: System.get_env("AWS_SESSION_TOKEN") # optional

  """
  def from_application_env!() do
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

  defp maybe_add_token(credentials) do
    case Application.fetch_env(:langchain, :aws_session_token) do
      {:ok, token} ->
        Keyword.put(credentials, :token, token)

      :error ->
        credentials
    end
  end
end
