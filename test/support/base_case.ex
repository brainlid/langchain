defmodule LangChain.BaseCase do
  @moduledoc """
  This module defines the test case to be used by
  tests that use LangChain features like Chat or LLMs.
  """

  use ExUnit.CaseTemplate

  # Default test registry used across all tests
  @test_registry LangChain.Test.Registry

  using do
    quote do
      alias LangChain.Message
      alias LangChain.MessageDelta

      # Import conveniences for testing with AI models
      import LangChain.BaseCase

      # Make test registry available to all tests
      @test_registry LangChain.Test.Registry

      @doc """
      Helper function for loading an image as base64 encoded text.
      """
      def load_image_base64(filename) do
        Path.join("./test/support/images", filename)
        |> File.read!()
        |> :base64.encode()
      end
    end
  end

  setup do
    # Ensure the test registry is started
    case Registry.start_link(keys: :unique, name: @test_registry) do
      {:ok, _pid} -> :ok
      {:error, {:already_started, _pid}} -> :ok
    end

    :ok
  end
end
