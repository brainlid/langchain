defmodule LangChain.Utils.BedrockConfig do
  @moduledoc """
  Configuration for AWS Bedrock.
  """
  use Ecto.Schema
  import Ecto.Changeset

  @primary_key false
  embedded_schema do
    # A function that returns a keyword list including access_key_id, secret_access_key, and optionally token.
    # Used to configure Req's aws_sigv4 option.
    field :credentials, :any, virtual: true
    field :region, :string
    field :anthropic_version, :string, default: "bedrock-2023-05-31"
  end

  def changeset(bedrock, attrs) do
    bedrock
    |> cast(attrs, [:credentials, :region, :anthropic_version])
    |> validate_required([:credentials, :region, :anthropic_version])
  end

  def aws_sigv4_opts(%__MODULE__{} = bedrock) do
    Keyword.merge(bedrock.credentials.(),
      region: bedrock.region,
      service: :bedrock
    )
  end

  def url(%__MODULE__{region: region}, model: model, stream: stream) do
    "https://bedrock-runtime.#{region}.amazonaws.com/model/#{model}/#{action(stream: stream)}"
  end

  defp action(stream: true), do: "invoke-with-response-stream"
  defp action(stream: false), do: "invoke"
end
