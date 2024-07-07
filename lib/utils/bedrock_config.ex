defmodule LangChain.Utils.BedrockConfig do
  @moduledoc """
  Configuration for AWS Bedrock.
  """
  use Ecto.Schema
  import Ecto.Changeset

  @primary_key false
  embedded_schema do
    # A function that returns a tuple of access_key_id & secret_access_key.
    # TODO: Add session token (STS) support at elem(3) when next req version is released
    field :credentials, :any, virtual: true
    field :region, :string
    field :anthropic_version, :string, default: "bedrock-2023-05-31"
  end

  def changeset(bedrock, attrs) do
    bedrock
    |> cast(attrs, [:credentials, :region, :anthropic_version])
    |> validate_required([:credentials, :region, :anthropic_version])
  end
end